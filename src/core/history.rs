//! Send-side retransmission history.
//!
//! Stores Media datagrams **after** they have been paced onto the wire so a
//! later [`crate::core::packet::NackPacket`] can clone them. A retransmission
//! keeps the media sequence and clears [`crate::core::packet::Packet::sequence`]
//! so the pacer assigns a new one. Aligns with WebRTC `RtpPacketHistory` /
//! `RTPSender::ReSendPacket` (`modules/rtp_rtcp/source/rtp_packet_history.*`),
//! adapted to qrt's TTL deadline and same-`media_seq` retransmit (no RTX SSRC).
//!
//! # Pipeline
//!
//! 1. Pacer sends a Media datagram → [`PacketHistory::put`].
//! 2. Incoming NACK → [`PacketHistory::get_retransmission`] → enqueue at
//!    retransmission priority (host / [`crate::core::send_queue`]).
//! 3. When that retransmit actually leaves the pacer →
//!    [`PacketHistory::mark_sent`] (clears `pending`, updates last-send time).
//!
//! # Examples
//!
//! ```
//! use std::time::{Duration, Instant};
//!
//! use bytes::Bytes;
//! use qrt::core::{
//!     history::{PacketHistory, RetransmitOutcome},
//!     packet::{MediaFragmentPacket, MediaType, Packet, Payload, Stream, StreamPacket},
//! };
//!
//! fn media_wire(seq: u32) -> Bytes {
//!     Packet {
//!         sequence: seq,
//!         timestamp: 0,
//!         payload: Payload::Stream(Stream {
//!             id: 1,
//!             idx: 1,
//!             packet: StreamPacket::Media(MediaFragmentPacket {
//!                 sequence: seq,
//!                 id: 1,
//!                 fragment_idx: 0,
//!                 fragment_count: 1,
//!                 media_type: MediaType::Video,
//!                 is_key_frame: false,
//!                 payload: Bytes::from_static(b"x"),
//!             }),
//!         }),
//!     }
//!     .into_bytes()
//! }
//!
//! let t0 = Instant::now();
//! let mut hist = PacketHistory::new(64);
//! hist.set_rtt(Duration::from_millis(40));
//! hist.put(1, 10, media_wire(10), t0, t0 + Duration::from_millis(200));
//!
//! match hist.get_retransmission(1, 10, t0 + Duration::from_millis(50)) {
//!     RetransmitOutcome::Ready(out) => {
//!         let decoded = Packet::from_bytes(out.wire).unwrap();
//!         assert_eq!(decoded.sequence, 0);
//!         match decoded.payload {
//!             Payload::Stream(stream) => match stream.packet {
//!                 StreamPacket::Media(media) => assert_eq!(media.sequence, 10),
//!                 _ => panic!("expected media"),
//!             },
//!             _ => panic!("expected media"),
//!         }
//!         hist.mark_sent(1, 10, t0 + Duration::from_millis(50));
//!     }
//!     other => panic!("expected Ready, got {other:?}"),
//! }
//!
//! // Within RTT of last send → reject duplicate NACK.
//! assert!(matches!(
//!     hist.get_retransmission(1, 10, t0 + Duration::from_millis(60)),
//!     RetransmitOutcome::TooSoon
//! ));
//! ```
//!
//! # Notes
//!
//! - Key is `(stream_id, media_seq)`. [`Packet::sequence`] on retransmit
//!   is cleared to `0` so the pacer can assign a fresh transport sequence.
//! - Expired entries (`now >= deadline`) are never retransmitted.
//! - Optional [`RetransRateLimiter`] caps RTX bytes over a sliding window
//!   (WebRTC ~500ms, budget ≈ BWE target).

use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

use ahash::{HashMap, HashMapExt};
use bytes::Bytes;

use crate::core::{
    packet::{MediaType, Packet, Payload, StreamPacket},
    send_queue::{OutgoingPacket, Priority},
};

/// Default history capacity (WebRTC video history commonly ~600).
pub const DEFAULT_HISTORY_CAPACITY: usize = 600;

/// Result of looking up a packet for NACK-driven retransmission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RetransmitOutcome {
    /// Ready to enqueue; history marks the entry `pending` until [`PacketHistory::mark_sent`].
    Ready(OutgoingPacket),
    /// No stored Media for that `(stream_id, media_seq)`.
    NotFound,
    /// Past the stored absolute deadline — do not retransmit.
    Expired,
    /// Already waiting in the send queue / pacer.
    AlreadyPending,
    /// Last send was less than one RTT ago (avoid RTX storms on duplicate NACK).
    TooSoon,
    /// [`RetransRateLimiter`] refused the byte cost.
    RateLimited,
}

/// One stored Media datagram eligible for retransmission.
#[derive(Debug, Clone)]
struct HistoryEntry {
    wire: Bytes,
    deadline: Instant,
    last_sent_at: Instant,
    pending: bool,
    retransmit_count: u32,
}

/// Send-side store of recently sent Media packets.
#[derive(Debug, Clone)]
pub struct PacketHistory {
    capacity: usize,
    entries: HashMap<(u8, u32), HistoryEntry>,
    order: VecDeque<(u8, u32)>,
    rtt: Duration,
    limiter: Option<RetransRateLimiter>,
}

impl PacketHistory {
    /// Creates a history with [`DEFAULT_HISTORY_CAPACITY`] and no rate limiter.
    ///
    /// # Examples
    ///
    /// ```
    /// use qrt::core::history::{DEFAULT_HISTORY_CAPACITY, PacketHistory};
    /// let h = PacketHistory::new(DEFAULT_HISTORY_CAPACITY);
    /// assert_eq!(h.len(), 0);
    /// ```
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            entries: HashMap::new(),
            order: VecDeque::new(),
            rtt: Duration::from_millis(100),
            limiter: None,
        }
    }

    /// Attaches a retransmission byte-rate limiter (replaces any previous one).
    pub fn set_rate_limiter(&mut self, limiter: RetransRateLimiter) {
        self.limiter = Some(limiter);
    }

    /// Removes the rate limiter (unlimited RTX aside from RTT / pending gates).
    pub fn clear_rate_limiter(&mut self) {
        self.limiter = None;
    }

    /// Updates the estimated RTT used by the “too soon” gate.
    pub fn set_rtt(&mut self, rtt: Duration) {
        self.rtt = rtt.max(Duration::from_millis(1));
    }

    /// Current RTT estimate.
    pub fn rtt(&self) -> Duration {
        self.rtt
    }

    /// Number of stored packets.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when nothing is stored.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Stores (or replaces) a Media datagram that was just sent.
    ///
    /// `wire` should be the exact UDP payload that left the socket (first send).
    /// `deadline` is the absolute instant after which retransmission is useless
    /// (typically `sent_at + remaining ttl`).
    ///
    /// Non-Media wires are ignored.
    pub fn put(
        &mut self,
        stream_id: u8,
        media_seq: u32,
        wire: Bytes,
        now: Instant,
        deadline: Instant,
    ) {
        if Packet::from_bytes(wire.clone())
            .ok()
            .is_none_or(|packet| !is_media(&packet))
        {
            return;
        }

        let key = (stream_id, media_seq);
        if let std::collections::hash_map::Entry::Occupied(mut slot) = self.entries.entry(key) {
            // Refresh in place; keep order position.
            slot.insert(HistoryEntry {
                wire,
                deadline,
                last_sent_at: now,
                pending: false,
                retransmit_count: 0,
            });

            return;
        }

        while self.entries.len() >= self.capacity {
            if let Some(old) = self.order.pop_front() {
                self.entries.remove(&old);
            } else {
                break;
            }
        }

        self.order.push_back(key);
        self.entries.insert(
            key,
            HistoryEntry {
                wire,
                deadline,
                last_sent_at: now,
                pending: false,
                retransmit_count: 0,
            },
        );
    }

    /// Convenience: [`Self::put`] from an [`OutgoingPacket`] that was just sent.
    ///
    /// Uses [`OutgoingPacket::deadline`] and decodes `media_seq` from the wire.
    /// Ignores non-Media and already-retransmitted wires (those only
    /// [`Self::mark_sent`]).
    pub fn put_outgoing(&mut self, packet: &OutgoingPacket, now: Instant) {
        if packet.retransmit {
            return;
        }

        let Ok(decoded) = Packet::from_bytes(packet.wire.clone()) else {
            return;
        };

        let Some((stream_id, media_seq)) = media_key(&decoded) else {
            return;
        };

        self.put(
            stream_id,
            media_seq,
            packet.wire.clone(),
            now,
            packet.deadline,
        );
    }

    /// Looks up `media_seq` and, if eligible, builds a retransmission
    /// [`OutgoingPacket`] with [`OutgoingPacket::retransmit`] set.
    ///
    /// On [`RetransmitOutcome::Ready`], the entry is marked `pending` until
    /// [`Self::mark_sent`] (or a later failed path clears it — host should
    /// always call `mark_sent` after the retransmit is paced out, or
    /// [`Self::clear_pending`] if the enqueue is abandoned).
    pub fn get_retransmission(
        &mut self,
        stream_id: u8,
        media_seq: u32,
        now: Instant,
    ) -> RetransmitOutcome {
        let key = (stream_id, media_seq);
        let Some(entry) = self.entries.get(&key) else {
            return RetransmitOutcome::NotFound;
        };

        if now >= entry.deadline {
            return RetransmitOutcome::Expired;
        }

        if entry.pending {
            return RetransmitOutcome::AlreadyPending;
        }

        if now.saturating_duration_since(entry.last_sent_at) < self.rtt {
            return RetransmitOutcome::TooSoon;
        }

        let remaining = entry.deadline.saturating_duration_since(now);
        let ttl_ms = u16::try_from(remaining.as_millis().min(u128::from(u16::MAX))).unwrap_or(0);
        if ttl_ms == 0 {
            return RetransmitOutcome::Expired;
        }

        let wire = match prepare_retransmit(&entry.wire) {
            Some(w) => w,
            None => return RetransmitOutcome::NotFound,
        };

        let priority = match Packet::from_bytes(wire.clone()).ok().as_ref() {
            Some(packet) if is_audio(packet) => Priority::Audio,
            _ => Priority::Retransmission,
        };

        if let Some(limiter) = self.limiter.as_mut()
            && !limiter.try_consume(wire.len() as u64, now)
        {
            return RetransmitOutcome::RateLimited;
        }

        let out = OutgoingPacket {
            wire,
            priority,
            stream_id,
            retransmit: true,
            enqueued_at: now,
            deadline: entry.deadline,
        };

        let entry = self.entries.get_mut(&key).expect("key checked");
        entry.pending = true;

        RetransmitOutcome::Ready(out)
    }

    /// Clears `pending` and records that a retransmission (or first send refresh)
    /// left the pacer at `now`.
    pub fn mark_sent(&mut self, stream_id: u8, media_seq: u32, now: Instant) {
        if let Some(entry) = self.entries.get_mut(&(stream_id, media_seq)) {
            entry.pending = false;
            entry.last_sent_at = now;
            entry.retransmit_count = entry.retransmit_count.saturating_add(1);
        }
    }

    /// Clears `pending` without updating last-send time (enqueue abandoned).
    pub fn clear_pending(&mut self, stream_id: u8, media_seq: u32) {
        if let Some(entry) = self.entries.get_mut(&(stream_id, media_seq)) {
            entry.pending = false;
        }
    }

    /// Drops entries at/after their deadline and trims beyond capacity.
    pub fn cull(&mut self, now: Instant) {
        self.order.retain(|key| {
            let keep = self.entries.get(key).is_some_and(|e| now < e.deadline);
            if !keep {
                self.entries.remove(key);
            }

            keep
        });

        while self.entries.len() > self.capacity {
            if let Some(old) = self.order.pop_front() {
                self.entries.remove(&old);
            } else {
                break;
            }
        }
    }

    /// Mutable access to the rate limiter, if any.
    pub fn rate_limiter_mut(&mut self) -> Option<&mut RetransRateLimiter> {
        self.limiter.as_mut()
    }
}

/// Sliding-window byte budget for retransmissions (WebRTC `RateLimiter` role).
///
/// Typical setup: `window = 500ms`, `bytes_per_window ≈ target_bitrate_bps / 8 *
/// 0.5`. Host should call [`Self::set_budget`] when BWE target changes.
#[derive(Debug, Clone)]
pub struct RetransRateLimiter {
    window: Duration,
    budget_bytes: u64,
    used_bytes: u64,
    window_start: Option<Instant>,
}

impl RetransRateLimiter {
    /// Creates a limiter with the given window and byte budget.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::time::{Duration, Instant};
    ///
    /// use qrt::core::history::RetransRateLimiter;
    ///
    /// // 800 kbps target → 50 KB per 500ms window.
    /// let mut lim = RetransRateLimiter::new(Duration::from_millis(500), 50_000);
    /// let t0 = Instant::now();
    /// assert!(lim.try_consume(40_000, t0));
    /// assert!(!lim.try_consume(20_000, t0));
    /// assert!(lim.try_consume(10_000, t0 + Duration::from_millis(500)));
    /// ```
    pub fn new(window: Duration, budget_bytes: u64) -> Self {
        Self {
            window: window.max(Duration::from_millis(1)),
            budget_bytes,
            used_bytes: 0,
            window_start: None,
        }
    }

    /// Builds a limiter from a target bitrate (bits/s) and window length.
    ///
    /// Budget = `target_bps / 8 * window_secs` (integer arithmetic).
    pub fn from_target_bps(target_bps: u64, window: Duration) -> Self {
        let ms = window.as_millis() as u64;
        let budget = target_bps.saturating_mul(ms) / 8 / 1000;

        Self::new(window, budget.max(1))
    }

    /// Updates the per-window byte budget (e.g. after BWE changes).
    pub fn set_budget(&mut self, budget_bytes: u64) {
        self.budget_bytes = budget_bytes;
    }

    /// Current budget in bytes for one window.
    pub fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }

    /// Tries to account for `nbytes`. Returns `false` if the window is exhausted.
    pub fn try_consume(&mut self, nbytes: u64, now: Instant) -> bool {
        self.roll(now);
        if self.used_bytes.saturating_add(nbytes) > self.budget_bytes {
            return false;
        }

        self.used_bytes = self.used_bytes.saturating_add(nbytes);

        true
    }

    fn roll(&mut self, now: Instant) {
        match self.window_start {
            None => {
                self.window_start = Some(now);
                self.used_bytes = 0;
            }
            Some(start) if now.saturating_duration_since(start) >= self.window => {
                self.window_start = Some(now);
                self.used_bytes = 0;
            }
            _ => {}
        }
    }
}

/// Copy `wire` and clear the transport sequence so the pacer stamps a new one.
fn prepare_retransmit(wire: &[u8]) -> Option<Bytes> {
    let mut packet = Packet::from_bytes(Bytes::copy_from_slice(wire)).ok()?;
    if !is_media(&packet) {
        return None;
    }

    packet.sequence = 0;

    Some(packet.into_bytes())
}

fn is_media(packet: &Packet) -> bool {
    matches!(
        &packet.payload,
        Payload::Stream(stream) if matches!(stream.packet, StreamPacket::Media(_))
    )
}

fn media_key(packet: &Packet) -> Option<(u8, u32)> {
    let Payload::Stream(stream) = &packet.payload else {
        return None;
    };

    let StreamPacket::Media(media) = &stream.packet else {
        return None;
    };

    Some((stream.id, media.sequence))
}

fn is_audio(packet: &Packet) -> bool {
    let Payload::Stream(stream) = &packet.payload else {
        return false;
    };

    let StreamPacket::Media(media) = &stream.packet else {
        return false;
    };

    media.media_type == MediaType::Audio
}
