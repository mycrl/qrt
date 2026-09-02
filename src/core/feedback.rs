//! Transport-wide arrival feedback (**sans-I/O**), TWCC / transport-cc role.
//!
//! Receiver records every datagram by [`Header::transport_seq`] and periodically
//! emits [`Packet::ArrivalFeedback`]. Sender matches feedback against a local
//! send history into [`TransportPacketsFeedback`] for BWE (Phase 6).
//!
//! Aligns with WebRTC
//! `TransportSequenceNumberFeedbackGenerator` + `TransportFeedbackAdapter`
//! (`modules/remote_bitrate_estimator/*`, `modules/congestion_controller/rtp/*`).
//!
//! # Pipeline
//!
//! **Receive**
//!
//! 1. On every UDP datagram (media / FEC / retrans / padding) →
//!    [`ArrivalRecorder::on_packet`] with **transport_seq**.
//! 2. Every ~50–100ms → [`ArrivalRecorder::poll`] → enqueue feedback.
//!
//! **Send**
//!
//! 1. At true send time → [`FeedbackAdapter::on_sent`] (after assigning
//!    `transport_seq`).
//! 2. On inbound ArrivalFeedback → [`FeedbackAdapter::on_feedback`] →
//!    [`TransportPacketsFeedback`] (acked / lost / in-flight).
//!
//! # Examples
//!
//! ```
//! use std::time::{Duration, Instant};
//!
//! use qrt::core::feedback::{ArrivalRecorder, FeedbackAdapter, FeedbackConfig};
//!
//! let t0 = Instant::now();
//! let mut rx = ArrivalRecorder::new(FeedbackConfig::default());
//! rx.on_packet(10, t0, 100);
//! rx.on_packet(12, t0 + Duration::from_millis(2), 100);
//!
//! let fb = rx
//!     .poll(t0 + Duration::from_millis(100))
//!     .expect("feedback due");
//! assert_eq!(fb.first_seq, 10);
//! assert!(fb.received_mask & 0b101 == 0b101); // 10 and 12
//!
//! let mut tx = FeedbackAdapter::new(FeedbackConfig::default());
//! tx.on_sent(10, t0, 100, false);
//! tx.on_sent(11, t0 + Duration::from_millis(1), 100, false);
//! tx.on_sent(12, t0 + Duration::from_millis(2), 100, false);
//!
//! let report = tx.on_feedback(
//!     fb.first_seq,
//!     fb.received_mask,
//!     &fb.recv_deltas_250us,
//!     t0 + Duration::from_millis(120),
//! );
//! assert_eq!(report.packets.len(), 3);
//! assert!(report.packets[0].received());
//! assert!(!report.packets[1].received()); // 11 lost in mask
//! assert!(report.packets[2].received());
//! ```
//!
//! # Notes
//!
//! - Always key on **transport_seq**, never `media_seq` (RTX/FEC must be visible).
//! - In-flight includes retransmits and FEC once `on_sent` recorded them.
//! - Networking stays outside this module.

use std::{
    collections::{BTreeMap, VecDeque},
    time::{Duration, Instant},
};

use bytes::Bytes;

use crate::core::packet::{
    ARRIVAL_RECV_DELTA_TICK,
    Flags,
    HEADER_SIZE,
    Header,
    Packet,
    PacketType,
};

/// Bits covered by one [`Packet::ArrivalFeedback`] mask.
pub const ARRIVAL_MASK_BITS: u16 = 64;

/// Default feedback interval (WebRTC often ~100ms, clamped 50–250ms).
pub const DEFAULT_FEEDBACK_INTERVAL: Duration = Duration::from_millis(100);

/// How long the sender retains send records for matching.
pub const DEFAULT_HISTORY_RETENTION: Duration = Duration::from_millis(500);

/// Tunables shared by recorder and adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeedbackConfig {
    /// Minimum time between automatic feedback emits.
    pub interval: Duration,
    /// Sender history retention / receiver prune age.
    pub retention: Duration,
    /// TTL written into emitted feedback headers.
    pub feedback_ttl_ms: u16,
}

impl Default for FeedbackConfig {
    fn default() -> Self {
        Self {
            interval: DEFAULT_FEEDBACK_INTERVAL,
            retention: DEFAULT_HISTORY_RETENTION,
            feedback_ttl_ms: 100,
        }
    }
}

/// Owned ArrivalFeedback ready to encode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrivalFeedbackOwned {
    /// Common header (`packet_type == ArrivalFeedback`).
    pub header: Header,
    /// First transport sequence in the mask window.
    pub first_seq: u16,
    /// Bit `i` set ⇒ `first_seq.wrapping_add(i)` received.
    pub received_mask: u64,
    /// Recv deltas in 250µs ticks (one per set bit, low→high).
    pub recv_deltas_250us: Vec<u16>,
    /// Big-endian encoding of [`Self::recv_deltas_250us`].
    pub recv_delta_bytes: Bytes,
}

impl ArrivalFeedbackOwned {
    /// Borrow as [`Packet::ArrivalFeedback`].
    pub fn as_packet(&self) -> Packet<'_> {
        Packet::ArrivalFeedback {
            header: self.header.clone(),
            first_seq: self.first_seq,
            received_mask: self.received_mask,
            recv_delta_bytes: &self.recv_delta_bytes,
        }
    }

    /// Encode to a full UDP datagram.
    pub fn to_wire(&self) -> Bytes {
        let pkt = self.as_packet();
        let mut buf = vec![0u8; pkt.encoded_len()];
        pkt.encode(&mut buf);
        Bytes::from(buf)
    }
}

/// Receiver: records arrivals and builds periodic feedback.
#[derive(Debug, Clone)]
pub struct ArrivalRecorder {
    config: FeedbackConfig,
    /// transport_seq → receive Instant.
    received: BTreeMap<u16, Instant>,
    /// Next window base to report (wrapping).
    next_base: Option<u16>,
    last_emit: Option<Instant>,
    /// Newest transport_seq seen (for wrapping-aware prune).
    newest: Option<u16>,
}

impl ArrivalRecorder {
    /// Creates a recorder with the given config.
    ///
    /// # Examples
    ///
    /// ```
    /// use qrt::core::feedback::{ArrivalRecorder, FeedbackConfig};
    /// let r = ArrivalRecorder::new(FeedbackConfig::default());
    /// assert_eq!(r.pending_count(), 0);
    /// ```
    pub fn new(config: FeedbackConfig) -> Self {
        Self {
            config,
            received: BTreeMap::new(),
            next_base: None,
            last_emit: None,
            newest: None,
        }
    }

    /// Number of arrivals not yet covered by an emitted window.
    pub fn pending_count(&self) -> usize {
        self.received.len()
    }

    /// Updates the emit interval / retention.
    pub fn set_config(&mut self, config: FeedbackConfig) {
        self.config = config;
    }

    /// Records that `transport_seq` arrived at `now` (size unused for emit).
    pub fn on_packet(&mut self, transport_seq: u16, now: Instant, _size_bytes: usize) {
        self.received.entry(transport_seq).or_insert(now);
        self.newest = Some(match self.newest {
            Some(n) if seq_ahead(transport_seq, n) => transport_seq,
            Some(n) => n,
            None => transport_seq,
        });
        if self.next_base.is_none() {
            self.next_base = Some(transport_seq);
        }
        self.prune(now);
    }

    /// Emits feedback when the interval elapsed and there is something to report.
    ///
    /// Returns `None` if not due or the next 64-seq window has no activity yet.
    pub fn poll(&mut self, now: Instant) -> Option<ArrivalFeedbackOwned> {
        if let Some(last) = self.last_emit {
            if now.saturating_duration_since(last) < self.config.interval {
                return None;
            }
        }
        self.build_window(now)
    }

    /// Forces a feedback build ignoring the interval (e.g. shutdown flush).
    pub fn force_build(&mut self, now: Instant) -> Option<ArrivalFeedbackOwned> {
        self.build_window(now)
    }

    fn build_window(&mut self, now: Instant) -> Option<ArrivalFeedbackOwned> {
        self.prune(now);
        let mut base = self.next_base?;

        // If the current window is empty but newer arrivals exist, jump to the
        // oldest pending receive (numeric min is fine within a 500ms window).
        let window_has_recv =
            (0..ARRIVAL_MASK_BITS).any(|i| self.received.contains_key(&base.wrapping_add(i)));
        if !window_has_recv {
            let Some((&oldest, _)) = self.received.iter().next() else {
                return None;
            };
            base = oldest;
        }

        let mut mask = 0u64;
        let mut times: Vec<Instant> = Vec::new();
        for i in 0..ARRIVAL_MASK_BITS {
            let seq = base.wrapping_add(i);
            if let Some(&t) = self.received.get(&seq) {
                mask |= 1u64 << i;
                times.push(t);
                self.received.remove(&seq);
            }
        }

        if mask == 0 {
            return None;
        }

        let mut deltas = Vec::with_capacity(times.len());
        let mut prev = times[0];
        for (idx, &t) in times.iter().enumerate() {
            if idx == 0 {
                deltas.push(0u16);
            } else {
                let us = t.saturating_duration_since(prev).as_micros();
                let ticks = (us / 250).min(u128::from(u16::MAX)) as u16;
                deltas.push(ticks);
            }
            prev = t;
        }

        let delta_bytes = Packet::encode_arrival_recv_deltas(&deltas);

        let owned = ArrivalFeedbackOwned {
            header: Header {
                packet_type: PacketType::ArrivalFeedback,
                flags: Flags::default(),
                stream_id: 0,
                media_seq: 0,
                transport_seq: 0,
                frame_id: 0,
                frag_index: 0,
                frag_count: 1,
                // Absolute base time unused; relative deltas feed delay BWE.
                timestamp: 0,
                ttl_ms: self.config.feedback_ttl_ms,
            },
            first_seq: base,
            received_mask: mask,
            recv_deltas_250us: deltas,
            recv_delta_bytes: Bytes::from(delta_bytes),
        };

        self.next_base = Some(base.wrapping_add(ARRIVAL_MASK_BITS));
        self.last_emit = Some(now);
        Some(owned)
    }

    fn prune(&mut self, now: Instant) {
        let retention = self.config.retention;
        self.received
            .retain(|_, t| now.saturating_duration_since(*t) <= retention);
    }
}

/// One sent datagram tracked for feedback matching.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SentPacket {
    /// Connection-wide transport sequence.
    pub transport_seq: u16,
    /// Local send Instant.
    pub send_time: Instant,
    /// On-wire size in bytes.
    pub size_bytes: usize,
    /// `true` if audio (optional BWE hint).
    pub audio: bool,
}

/// Per-packet outcome after matching send history with arrival feedback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PacketResult {
    /// Transport sequence.
    pub transport_seq: u16,
    /// Local send time.
    pub send_time: Instant,
    /// On-wire size.
    pub size_bytes: usize,
    /// Receive time on a timeline anchored to feedback arrival; `None` = lost.
    pub receive_time: Option<Instant>,
}

impl PacketResult {
    /// Returns `true` when the packet was reported received.
    pub fn received(&self) -> bool {
        self.receive_time.is_some()
    }
}

/// BWE input: matched send × arrival feedback (WebRTC `TransportPacketsFeedback`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransportPacketsFeedback {
    /// When the feedback datagram was processed locally.
    pub feedback_time: Instant,
    /// Results for transport seqs covered by the feedback window that were
    /// present in send history.
    pub packets: Vec<PacketResult>,
    /// Bytes still in flight after applying this feedback.
    pub data_in_flight: usize,
}

/// Sender: send history + ArrivalFeedback → [`TransportPacketsFeedback`].
#[derive(Debug, Clone)]
pub struct FeedbackAdapter {
    config: FeedbackConfig,
    history: VecDeque<SentPacket>,
    /// Highest transport_seq whose feedback window has been fully applied.
    /// In-flight = sent with seq ahead of this that are not yet acked.
    last_ack_advance: Option<u16>,
    acked: BTreeMap<u16, Instant>,
}

impl FeedbackAdapter {
    /// Creates an adapter with empty history.
    ///
    /// # Examples
    ///
    /// ```
    /// use qrt::core::feedback::{FeedbackAdapter, FeedbackConfig};
    /// let a = FeedbackAdapter::new(FeedbackConfig::default());
    /// assert_eq!(a.in_flight_bytes(), 0);
    /// ```
    pub fn new(config: FeedbackConfig) -> Self {
        Self {
            config,
            history: VecDeque::new(),
            last_ack_advance: None,
            acked: BTreeMap::new(),
        }
    }

    /// Records a datagram at true send time (after `transport_seq` assignment).
    pub fn on_sent(
        &mut self,
        transport_seq: u16,
        send_time: Instant,
        size_bytes: usize,
        audio: bool,
    ) {
        self.history.push_back(SentPacket {
            transport_seq,
            send_time,
            size_bytes,
            audio,
        });
        self.cull(send_time);
    }

    /// Bytes of sent packets not yet covered by feedback (acked or lost).
    pub fn in_flight_bytes(&self) -> usize {
        self.history
            .iter()
            .filter(|p| !self.acked.contains_key(&p.transport_seq))
            .map(|p| p.size_bytes)
            .sum()
    }

    /// Matches one ArrivalFeedback body against send history.
    ///
    /// `recv_deltas_250us` may be empty; received packets then share
    /// `feedback_time` as a coarse receive Instant (loss still correct).
    ///
    /// # Examples
    ///
    /// See the [module-level example](crate::core::feedback).
    pub fn on_feedback(
        &mut self,
        first_seq: u16,
        received_mask: u64,
        recv_deltas_250us: &[u16],
        feedback_time: Instant,
    ) -> TransportPacketsFeedback {
        self.cull(feedback_time);

        // Build receive times for set bits (low→high).
        let mut recv_times: BTreeMap<u16, Instant> = BTreeMap::new();
        let set_bits: Vec<u16> = (0..ARRIVAL_MASK_BITS)
            .filter(|&i| received_mask & (1u64 << i) != 0)
            .collect();

        if !set_bits.is_empty() {
            if recv_deltas_250us.len() == set_bits.len() {
                // Anchor the last received packet at feedback_time and walk back.
                let mut spans = Vec::with_capacity(set_bits.len());
                let mut acc = Duration::ZERO;
                for (idx, &ticks) in recv_deltas_250us.iter().enumerate() {
                    if idx > 0 {
                        acc += ARRIVAL_RECV_DELTA_TICK * u32::from(ticks);
                    }
                    spans.push(acc);
                }
                let total = *spans.last().unwrap_or(&Duration::ZERO);
                for (i, bit) in set_bits.iter().enumerate() {
                    let seq = first_seq.wrapping_add(*bit);
                    let recv_at = feedback_time
                        .checked_sub(total.saturating_sub(spans[i]))
                        .unwrap_or(feedback_time);
                    recv_times.insert(seq, recv_at);
                }
            } else {
                for bit in set_bits {
                    recv_times.insert(first_seq.wrapping_add(bit), feedback_time);
                }
            }
        }

        let mut packets = Vec::new();
        for i in 0..ARRIVAL_MASK_BITS {
            let seq = first_seq.wrapping_add(i);
            let Some(sent) = self.history.iter().find(|p| p.transport_seq == seq) else {
                continue;
            };
            let receive_time = recv_times.get(&seq).copied();
            // Covered by this window (acked or lost) → leave in-flight.
            self.acked.insert(seq, feedback_time);
            packets.push(PacketResult {
                transport_seq: seq,
                send_time: sent.send_time,
                size_bytes: sent.size_bytes,
                receive_time,
            });
        }

        // Window applied; keep recent unacked history for in-flight accounting.
        self.last_ack_advance = Some(first_seq.wrapping_add(ARRIVAL_MASK_BITS.wrapping_sub(1)));
        self.cull(feedback_time);

        let data_in_flight = self.in_flight_bytes();
        TransportPacketsFeedback {
            feedback_time,
            packets,
            data_in_flight,
        }
    }

    /// Convenience: decode an ArrivalFeedback packet and match it.
    pub fn on_feedback_packet(
        &mut self,
        packet: &Packet<'_>,
        feedback_time: Instant,
    ) -> Option<TransportPacketsFeedback> {
        match packet {
            Packet::ArrivalFeedback {
                first_seq,
                received_mask,
                recv_delta_bytes,
                ..
            } => {
                let deltas = Packet::parse_arrival_recv_deltas(recv_delta_bytes);
                Some(self.on_feedback(*first_seq, *received_mask, &deltas, feedback_time))
            }
            _ => None,
        }
    }

    fn cull(&mut self, now: Instant) {
        let retention = self.config.retention;
        while let Some(front) = self.history.front() {
            if now.saturating_duration_since(front.send_time) > retention {
                let seq = front.transport_seq;
                self.history.pop_front();
                self.acked.remove(&seq);
            } else {
                break;
            }
        }
        // Also drop acked packets older than retention from the deque front.
        while let Some(front) = self.history.front() {
            if self.acked.contains_key(&front.transport_seq)
                && now.saturating_duration_since(front.send_time) > retention / 2
            {
                let seq = front.transport_seq;
                self.history.pop_front();
                self.acked.remove(&seq);
            } else {
                break;
            }
        }
    }
}

/// Assigns monotonically increasing [`Header::transport_seq`] values.
#[derive(Debug, Clone, Default)]
pub struct TransportSeqAssigner {
    next: u16,
}

impl TransportSeqAssigner {
    /// Creates an assigner starting at `0`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the next transport sequence and advances the counter.
    ///
    /// # Examples
    ///
    /// ```
    /// use qrt::core::feedback::TransportSeqAssigner;
    /// let mut a = TransportSeqAssigner::new();
    /// assert_eq!(a.next(), 0);
    /// assert_eq!(a.next(), 1);
    /// ```
    pub fn next(&mut self) -> u16 {
        let s = self.next;
        self.next = self.next.wrapping_add(1);
        s
    }

    /// Writes `transport_seq` into the first [`HEADER_SIZE`] bytes of `wire`.
    pub fn stamp(&mut self, wire: &mut [u8]) -> Option<u16> {
        if wire.len() < HEADER_SIZE {
            return None;
        }
        let mut header = Header::decode(wire).ok()?;
        let seq = self.next();
        header.transport_seq = seq;
        header.encode(wire);
        Some(seq)
    }
}

fn seq_ahead(a: u16, b: u16) -> bool {
    let diff = a.wrapping_sub(b);
    diff != 0 && diff < 0x8000
}
