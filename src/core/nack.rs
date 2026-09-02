//! Receive-side NACK list (**sans-I/O**).
//!
//! Tracks missing [`Header::media_seq`] gaps and periodically emits
//! [`Packet::Nack`] bodies (RFC 4585 Generic NACK role). Aligns with WebRTC
//! `NackRequester` (`modules/video_coding/nack_requester.*`): ~20ms process
//! cadence, retransmit spacing ≥ RTT, list cap → keyframe, skip sequences that
//! arrived (including FEC-recovered media).
//!
//! # Pipeline
//!
//! 1. Demux Media (or FEC-recovered Media) → [`NackRequester::on_received`].
//! 2. Every ~20ms (host timer) → [`NackRequester::process`] → enqueue NACK /
//!    optional keyframe request as feedback.
//! 3. Sender [`crate::core::history::PacketHistory`] answers with retransmits.
//!
//! # Examples
//!
//! ```
//! use std::time::{Duration, Instant};
//!
//! use qrt::core::{
//!     nack::{NackConfig, NackRequester},
//!     packet::Packet,
//! };
//!
//! let t0 = Instant::now();
//! let mut nack = NackRequester::new(1, NackConfig::default());
//! nack.set_rtt(Duration::from_millis(40));
//!
//! // First packet establishes the frontier.
//! assert!(nack.on_received(10, t0).nacks_cleared == 0);
//! // Gap 11..12 when 13 arrives.
//! let _ = nack.on_received(13, t0);
//! assert!(nack.pending_count() >= 2);
//!
//! let batch = nack.process(t0);
//! assert!(!batch.entries.is_empty());
//! let seqs: Vec<u16> = batch
//!     .entries
//!     .iter()
//!     .flat_map(|(b, blp)| Packet::nack_missing_seqs(*b, *blp))
//!     .collect();
//! assert!(seqs.contains(&11) && seqs.contains(&12));
//!
//! // Recovered / late media clears the hole.
//! let cleared = nack.on_received(11, t0 + Duration::from_millis(5));
//! assert!(cleared.nacks_cleared >= 1);
//! ```
//!
//! # Notes
//!
//! - Call [`Self::on_received`] for **FEC-recovered** packets too — that is how
//!   recovered seqs skip NACK (and clear an existing entry).
//! - Never feed FEC / feedback datagrams here; only Media `media_seq`.
//! - Abandon a pending NACK when `now + 2×RTT >= created + packet_lifetime`
//!   (qrt deadline filter from `plan.md`).

use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

use crate::core::packet::{Flags, Header, Packet, PacketType};

/// Default periodic process interval (WebRTC `NackPeriodicProcessor` ~20ms).
pub const DEFAULT_PROCESS_INTERVAL: Duration = Duration::from_millis(20);

/// Default initial RTT when none has been measured yet.
pub const DEFAULT_RTT: Duration = Duration::from_millis(100);

/// Tunables for [`NackRequester`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NackConfig {
    /// Assumed useful lifetime of a media packet (matches typical send `ttl_ms`).
    ///
    /// Pending NACKs are dropped when `now + 2×RTT >= created_at + packet_lifetime`.
    pub packet_lifetime: Duration,
    /// Delay before the first NACK send for a gap (FEC head-start). WebRTC often
    /// uses `0`; a few ms can let FEC recover first.
    pub nack_delay: Duration,
    /// Maximum entries in the NACK table; overflow clears and asks for a keyframe.
    pub max_list_size: usize,
    /// Give up on a sequence after this many NACK transmissions.
    pub max_retries: u32,
    /// Clear pending entries older than this many sequence numbers behind newest.
    pub max_age_seqs: u16,
}

impl Default for NackConfig {
    fn default() -> Self {
        Self {
            packet_lifetime: Duration::from_millis(200),
            nack_delay: Duration::ZERO,
            max_list_size: 1000,
            max_retries: 100,
            max_age_seqs: 10_000,
        }
    }
}

#[derive(Debug, Clone)]
struct NackInfo {
    created_at: Instant,
    send_at: Instant,
    retries: u32,
}

/// Outcome of [`NackRequester::on_received`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OnReceivedResult {
    /// How many pending NACK entries were removed because the seq arrived.
    pub nacks_cleared: u32,
    /// `true` when the table overflowed and was cleared (caller should send PLI).
    pub ask_keyframe: bool,
}

/// Batch produced by [`NackRequester::process`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct NackBatch {
    /// Packed `(base_seq, blp)` ready for [`Packet::Nack`].
    pub entries: Vec<(u16, u16)>,
    /// Send a [`Packet::KeyframeReq`] (list overflow or exhausted retries storm).
    pub ask_keyframe: bool,
}

impl NackBatch {
    /// Builds owned [`Packet::Nack`] views (header `transport_seq` left at 0).
    ///
    /// # Examples
    ///
    /// ```
    /// use qrt::core::{nack::NackBatch, packet::Packet};
    ///
    /// let batch = NackBatch {
    ///     entries: vec![(10, 0b1)],
    ///     ask_keyframe: false,
    /// };
    /// let pkts = batch.to_packets(1, 50);
    /// assert_eq!(pkts.len(), 1);
    /// match &pkts[0] {
    ///     Packet::Nack { base_seq, blp, .. } => {
    ///         assert_eq!(*base_seq, 10);
    ///         assert_eq!(*blp, 0b1);
    ///     }
    ///     _ => panic!("expected Nack"),
    /// }
    /// ```
    pub fn to_packets(&self, stream_id: u8, ttl_ms: u16) -> Vec<Packet<'static>> {
        self.entries
            .iter()
            .map(|&(base_seq, blp)| Packet::Nack {
                header: Header {
                    packet_type: PacketType::Nack,
                    flags: Flags::default(),
                    stream_id,
                    media_seq: 0,
                    transport_seq: 0,
                    frame_id: 0,
                    frag_index: 0,
                    frag_count: 1,
                    timestamp: 0,
                    ttl_ms,
                },
                base_seq,
                blp,
                frame_id: None,
            })
            .collect()
    }
}

/// Per-stream missing-sequence tracker for selective retransmission requests.
#[derive(Debug, Clone)]
pub struct NackRequester {
    stream_id: u8,
    config: NackConfig,
    rtt: Duration,
    newest: Option<u16>,
    pending: BTreeMap<u16, NackInfo>,
}

impl NackRequester {
    /// Creates a requester for one [`Header::stream_id`].
    ///
    /// # Examples
    ///
    /// ```
    /// use qrt::core::nack::{NackConfig, NackRequester};
    /// let n = NackRequester::new(0, NackConfig::default());
    /// assert_eq!(n.stream_id(), 0);
    /// assert_eq!(n.pending_count(), 0);
    /// ```
    pub fn new(stream_id: u8, config: NackConfig) -> Self {
        Self {
            stream_id,
            config,
            rtt: DEFAULT_RTT,
            newest: None,
            pending: BTreeMap::new(),
        }
    }

    /// Stream this requester tracks.
    pub fn stream_id(&self) -> u8 {
        self.stream_id
    }

    /// Number of sequences currently waiting for a NACK / recovery.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Updates the RTT used for send spacing and deadline abandonment.
    pub fn set_rtt(&mut self, rtt: Duration) {
        self.rtt = rtt.max(Duration::from_millis(1));
    }

    /// Current RTT estimate.
    pub fn rtt(&self) -> Duration {
        self.rtt
    }

    /// Replaces config (applies to new gaps and subsequent `process` calls).
    pub fn set_config(&mut self, config: NackConfig) {
        self.config = config;
    }

    /// Notifies that Media with `media_seq` is available (received **or**
    /// FEC-recovered).
    ///
    /// Inserts NACK entries for the open gap `(newest+1 .. media_seq)`, then
    /// advances `newest`. Late / reordered packets only clear an existing entry.
    ///
    /// # Examples
    ///
    /// List overflow clears pending entries and asks for a keyframe:
    ///
    /// ```
    /// use std::time::Instant;
    ///
    /// use qrt::core::nack::{NackConfig, NackRequester};
    ///
    /// let t0 = Instant::now();
    /// let mut nack = NackRequester::new(
    ///     0,
    ///     NackConfig {
    ///         max_list_size: 4,
    ///         ..NackConfig::default()
    ///     },
    /// );
    /// nack.on_received(0, t0);
    /// let r = nack.on_received(10, t0); // gaps 1..9 exceed max_list_size
    /// assert!(r.ask_keyframe);
    /// assert_eq!(nack.pending_count(), 0);
    /// ```
    pub fn on_received(&mut self, media_seq: u16, now: Instant) -> OnReceivedResult {
        let mut result = OnReceivedResult::default();

        if self.pending.remove(&media_seq).is_some() {
            result.nacks_cleared = 1;
        }

        match self.newest {
            None => {
                self.newest = Some(media_seq);
            }
            Some(newest) if seq_ahead(media_seq, newest) => {
                let mut s = newest.wrapping_add(1);
                while s != media_seq {
                    self.insert_gap(s, now);
                    s = s.wrapping_add(1);
                }
                self.newest = Some(media_seq);
            }
            Some(_) => {
                // Reordered / duplicate behind newest — already cleared pending.
            }
        }

        if self.pending.len() > self.config.max_list_size {
            self.pending.clear();
            result.ask_keyframe = true;
        }

        result
    }

    /// Drops all pending NACK entries with `seq` not ahead of `media_seq`
    /// (decoder / reassembly advanced past them).
    pub fn clear_up_to(&mut self, media_seq: u16) {
        self.pending.retain(|&seq, _| seq_ahead(seq, media_seq));
    }

    /// Clears every pending entry (e.g. after a keyframe reset).
    pub fn clear(&mut self) {
        self.pending.clear();
    }

    /// Periodic scan: returns NACK packs that should be sent now.
    ///
    /// Host should call about every [`DEFAULT_PROCESS_INTERVAL`]. Sequences
    /// past lifetime / max retries are pruned; surviving due entries are spaced
    /// at least one RTT apart between sends.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::time::{Duration, Instant};
    ///
    /// use qrt::core::{
    ///     nack::{NackConfig, NackRequester},
    ///     packet::Packet,
    /// };
    ///
    /// let t0 = Instant::now();
    /// let mut nack = NackRequester::new(0, NackConfig::default());
    /// nack.set_rtt(Duration::from_millis(30));
    /// nack.on_received(1, t0);
    /// nack.on_received(3, t0); // missing 2
    ///
    /// let first = nack.process(t0);
    /// assert_eq!(
    ///     Packet::nack_missing_seqs(first.entries[0].0, first.entries[0].1),
    ///     vec![2]
    /// );
    /// assert!(
    ///     nack.process(t0 + Duration::from_millis(10))
    ///         .entries
    ///         .is_empty()
    /// );
    /// assert_eq!(
    ///     nack.process(t0 + Duration::from_millis(30)).entries.len(),
    ///     1
    /// );
    /// ```
    pub fn process(&mut self, now: Instant) -> NackBatch {
        let mut batch = NackBatch::default();
        let lifetime = self.config.packet_lifetime;
        let rtt = self.rtt;
        let max_retries = self.config.max_retries;
        let mut to_send: Vec<u16> = Vec::new();
        let mut remove: Vec<u16> = Vec::new();

        // Drop seqs too far behind newest.
        if let Some(newest) = self.newest {
            for &seq in self.pending.keys() {
                let age = newest.wrapping_sub(seq);
                if age > self.config.max_age_seqs {
                    remove.push(seq);
                }
            }
        }

        for (seq, info) in self.pending.iter_mut() {
            if remove.contains(seq) {
                continue;
            }
            // Abandon: retransmission would arrive after useful lifetime.
            if now.saturating_duration_since(info.created_at) + rtt.saturating_mul(2) >= lifetime {
                remove.push(*seq);
                continue;
            }
            if info.retries >= max_retries {
                remove.push(*seq);
                batch.ask_keyframe = true;
                continue;
            }
            if now >= info.send_at {
                to_send.push(*seq);
                info.retries = info.retries.saturating_add(1);
                info.send_at = now + rtt;
            }
        }

        for seq in remove {
            self.pending.remove(&seq);
        }

        batch.entries = Packet::nack_pack_seqs(to_send);
        batch
    }

    fn insert_gap(&mut self, seq: u16, now: Instant) {
        self.pending.entry(seq).or_insert_with(|| NackInfo {
            created_at: now,
            send_at: now + self.config.nack_delay,
            retries: 0,
        });
    }
}

/// Same wrapping rule as [`Header::seq_ahead`]: `a` is strictly ahead of `b`.
fn seq_ahead(a: u16, b: u16) -> bool {
    let diff = a.wrapping_sub(b);
    diff != 0 && diff < 0x8000
}
