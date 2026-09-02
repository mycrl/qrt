//! Prioritized send queue with TTL drop (**sans-I/O**).
//!
//! Holds already-encoded UDP payloads waiting for the [`crate::core::pacer`]. There is
//! **no** socket or timer thread here: the host application owns I/O and only
//! calls [`SendQueue::enqueue`] / [`SendQueue::pop`] with an [`Instant`] it
//! controls.
//!
//! # How it works
//!
//! ```text
//!  Packet / OutgoingPacket
//!           │
//!           ▼
//!   ttl_ms == 0? ──yes──► drop (never queued)
//!           │ no
//!           ▼
//!   bucket[Priority::of(header)]  (FIFO per level)
//!           │
//!           ▼
//!   pop(now): scan Audio → … → Padding
//!           │
//!           ├─ expired (now >= deadline) → drop, count, try next
//!           └─ else → return OutgoingPacket (wire bytes ready to send)
//! ```
//!
//! Priority order mirrors WebRTC's `PrioritizedPacketQueue`, with an extra
//! feedback rung for qrt (see `docs/webrtc-reference.md` §5):
//!
//! | Level | [`Priority`] | Typical packets |
//! |------:|--------------|-----------------|
//! | 0 | [`Priority::Audio`] | media with `flags.audio` |
//! | 1 | [`Priority::Retransmission`] | media with `flags.retrans` |
//! | 2 | [`Priority::Video`] | video media, [`crate::core::packet::PacketType::Fec`] |
//! | 3 | [`Priority::Feedback`] | NACK / ArrivalFeedback / KeyframeReq |
//! | 4 | [`Priority::Padding`] | probe / padding (lowest) |
//!
//! Classification is [`Priority::of`]. Same level is strict FIFO (no
//! multi-stream round-robin yet).
//!
//! TTL: at enqueue, `deadline = now + ttl_ms`. On pop, late packets are dropped
//! so a large video backlog cannot ship frames that are already useless.
//! Remaining-lifetime shrink while queued is represented by that absolute
//! deadline (same idea as shrinking [`crate::core::packet::Header::ttl_ms`] in place).
//!
//! # Pipeline with the pacer
//!
//! 1. Encode a [`crate::core::packet::Packet`] → [`OutgoingPacket::from_packet`] (or
//!    [`SendQueue::enqueue_packet`]).
//! 2. [`crate::core::pacer::Pacer::poll`] drains this queue under the leaky-bucket budget.
//! 3. The application `send`s [`OutgoingPacket::wire`] on its UDP socket.
//!
//! # Examples
//!
//! Audio jumps ahead of video; zero-TTL never enters the queue; overdue video
//! is dropped on pop:
//!
//! ```
//! use std::time::{Duration, Instant};
//!
//! use qrt::core::{
//!     packet::{Flags, Header, Packet, PacketType},
//!     send_queue::{Priority, SendQueue},
//! };
//!
//! fn media(audio: bool, ttl_ms: u16, payload: &'static [u8]) -> Packet<'static> {
//!     Packet::Media {
//!         header: Header {
//!             packet_type: PacketType::Media,
//!             flags: Flags {
//!                 audio,
//!                 ..Flags::default()
//!             },
//!             stream_id: 0,
//!             media_seq: 0,
//!             transport_seq: 0,
//!             frame_id: 0,
//!             frag_index: 0,
//!             frag_count: 1,
//!             timestamp: 0,
//!             ttl_ms,
//!         },
//!         payload,
//!     }
//! }
//!
//! let t0 = Instant::now();
//! let mut q = SendQueue::new();
//!
//! assert!(q.enqueue_packet(&media(false, 30, b"video"), t0));
//! assert!(q.enqueue_packet(&media(true, 30, b"audio"), t0));
//! assert!(!q.enqueue_packet(&media(false, 0, b"dead"), t0));
//!
//! // Audio leaves first despite being enqueued second.
//! let first = q.pop(t0).unwrap();
//! assert_eq!(first.priority, Priority::Audio);
//! assert_eq!(&first.wire[first.wire.len() - 5..], b"audio");
//!
//! // After the video deadline, pop drops it instead of sending.
//! let late = t0 + Duration::from_millis(40);
//! assert!(q.pop(late).is_none());
//! assert_eq!(q.stats().dropped_ttl_zero, 1);
//! assert_eq!(q.stats().dropped_expired, 1);
//! ```
//!
//! # Notes
//!
//! - [`OutgoingPacket::wire`] is a full datagram (header + body), not a bare
//!   codec payload.
//! - Counters live in [`SendQueueStats`] (`enqueued`, `dropped_ttl_zero`,
//!   `dropped_expired`, `dequeued`).

use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

use bytes::Bytes;

use crate::core::packet::{Header, Packet, PacketType};

/// Send priority (lower discriminant = higher priority).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Priority {
    /// Real-time audio media.
    Audio = 0,
    /// NACK-driven retransmission (`flags.retrans`).
    Retransmission = 1,
    /// Video media (key/delta) and [`PacketType::Fec`].
    Video = 2,
    /// [`PacketType::Nack`], [`PacketType::ArrivalFeedback`], [`PacketType::KeyframeReq`].
    Feedback = 3,
    /// Probe / padding filler (lowest).
    Padding = 4,
}

impl Priority {
    /// Number of distinct priority levels.
    pub const LEVELS: usize = 5;

    /// Classify a header the way the send path should enqueue it.
    ///
    /// # Examples
    ///
    /// ```
    /// use qrt::core::{
    ///     packet::{Flags, Header, PacketType},
    ///     send_queue::Priority,
    /// };
    ///
    /// let mut h = Header {
    ///     packet_type: PacketType::Media,
    ///     flags: Flags {
    ///         audio: true,
    ///         ..Flags::default()
    ///     },
    ///     stream_id: 0,
    ///     media_seq: 0,
    ///     transport_seq: 0,
    ///     frame_id: 0,
    ///     frag_index: 0,
    ///     frag_count: 1,
    ///     timestamp: 0,
    ///     ttl_ms: 40,
    /// };
    /// assert_eq!(Priority::of(&h), Priority::Audio);
    ///
    /// h.flags.audio = false;
    /// h.flags.retrans = true;
    /// assert_eq!(Priority::of(&h), Priority::Retransmission);
    ///
    /// h.packet_type = PacketType::Fec;
    /// h.flags = Flags::default();
    /// assert_eq!(Priority::of(&h), Priority::Video);
    /// ```
    pub fn of(header: &Header) -> Self {
        match header.packet_type {
            PacketType::Media if header.flags.audio => Self::Audio,
            PacketType::Media if header.flags.retrans => Self::Retransmission,
            PacketType::Media | PacketType::Fec => Self::Video,
            PacketType::Nack | PacketType::ArrivalFeedback | PacketType::KeyframeReq => {
                Self::Feedback
            }
        }
    }

    fn index(self) -> usize {
        self as u8 as usize
    }
}

/// One encoded datagram waiting to leave the host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutgoingPacket {
    /// Complete UDP payload (header + body), ready to `send`.
    pub wire: Bytes,
    /// Scheduling priority.
    pub priority: Priority,
    /// [`Header::stream_id`] snapshot.
    pub stream_id: u8,
    /// When the packet entered the queue.
    pub enqueued_at: Instant,
    /// Absolute deadline; at/after this instant the packet must be dropped.
    pub deadline: Instant,
}

impl OutgoingPacket {
    /// Encode `packet` into an owned outgoing datagram.
    ///
    /// Returns `None` when `ttl_ms == 0` (already stale — never enqueue).
    ///
    /// # Examples
    ///
    /// ```
    /// use std::time::Instant;
    ///
    /// use qrt::core::{
    ///     packet::{Flags, Header, Packet, PacketType},
    ///     send_queue::{OutgoingPacket, Priority},
    /// };
    ///
    /// let pkt = Packet::Media {
    ///     header: Header {
    ///         packet_type: PacketType::Media,
    ///         flags: Flags::default(),
    ///         stream_id: 1,
    ///         media_seq: 0,
    ///         transport_seq: 0,
    ///         frame_id: 0,
    ///         frag_index: 0,
    ///         frag_count: 1,
    ///         timestamp: 0,
    ///         ttl_ms: 100,
    ///     },
    ///     payload: b"x",
    /// };
    /// let out = OutgoingPacket::from_packet(&pkt, Instant::now()).unwrap();
    /// assert_eq!(out.priority, Priority::Video);
    /// assert_eq!(out.stream_id, 1);
    /// ```
    pub fn from_packet(packet: &Packet<'_>, now: Instant) -> Option<Self> {
        let header = packet.header();
        if header.ttl_ms == 0 {
            return None;
        }
        let mut wire = vec![0u8; packet.encoded_len()];
        packet.encode(&mut wire);
        Some(Self {
            wire: Bytes::from(wire),
            priority: Priority::of(header),
            stream_id: header.stream_id,
            enqueued_at: now,
            deadline: now + Duration::from_millis(u64::from(header.ttl_ms)),
        })
    }

    /// Wire length in bytes.
    pub fn len(&self) -> usize {
        self.wire.len()
    }

    /// Returns `true` if the deadline has been reached.
    pub fn is_expired(&self, now: Instant) -> bool {
        now >= self.deadline
    }
}

/// Statistics counters for the send queue.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SendQueueStats {
    /// Packets accepted into a priority bucket.
    pub enqueued: u64,
    /// Rejected because `ttl_ms == 0` at enqueue time.
    pub dropped_ttl_zero: u64,
    /// Removed after sitting past [`OutgoingPacket::deadline`].
    pub dropped_expired: u64,
    /// Successfully popped for the pacer/socket.
    pub dequeued: u64,
}

/// Priority FIFO queue with lazy TTL expiry.
///
/// # Examples
///
/// ```
/// use std::time::{Duration, Instant};
///
/// use qrt::core::{
///     packet::{Flags, Header, Packet, PacketType},
///     send_queue::SendQueue,
/// };
///
/// let now = Instant::now();
/// let fresh = Packet::Media {
///     header: Header {
///         packet_type: PacketType::Media,
///         flags: Flags {
///             audio: true,
///             ..Flags::default()
///         },
///         stream_id: 0,
///         media_seq: 0,
///         transport_seq: 0,
///         frame_id: 0,
///         frag_index: 0,
///         frag_count: 1,
///         timestamp: 0,
///         ttl_ms: 50,
///     },
///     payload: b"a",
/// };
/// let stale = Packet::Media {
///     header: Header {
///         ttl_ms: 0,
///         ..fresh.header().clone()
///     },
///     payload: b"b",
/// };
///
/// let mut q = SendQueue::new();
/// assert!(q.enqueue_packet(&fresh, now));
/// assert!(!q.enqueue_packet(&stale, now));
/// assert_eq!(q.len(), 1);
/// assert!(q.pop(now).is_some());
/// assert_eq!(q.stats().dropped_ttl_zero, 1);
/// ```
#[derive(Debug, Default)]
pub struct SendQueue {
    buckets: [VecDeque<OutgoingPacket>; Priority::LEVELS],
    stats: SendQueueStats,
    queued_packets: usize,
    queued_bytes: usize,
}

impl SendQueue {
    /// Create an empty queue.
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshot of drop / enqueue counters.
    pub fn stats(&self) -> SendQueueStats {
        self.stats
    }

    /// Number of packets currently queued (not counting expired until pop).
    pub fn len(&self) -> usize {
        self.queued_packets
    }

    /// Returns `true` if no packets are queued.
    pub fn is_empty(&self) -> bool {
        self.queued_packets == 0
    }

    /// Total payload bytes currently queued.
    pub fn queued_bytes(&self) -> usize {
        self.queued_bytes
    }

    /// Encode and enqueue `packet`. Returns `false` if dropped (`ttl_ms == 0`).
    pub fn enqueue_packet(&mut self, packet: &Packet<'_>, now: Instant) -> bool {
        match OutgoingPacket::from_packet(packet, now) {
            Some(out) => {
                self.enqueue(out);
                true
            }
            None => {
                self.stats.dropped_ttl_zero += 1;
                false
            }
        }
    }

    /// Enqueue an already-built outgoing datagram.
    pub fn enqueue(&mut self, packet: OutgoingPacket) {
        let idx = packet.priority.index();
        self.queued_bytes += packet.len();
        self.queued_packets += 1;
        self.stats.enqueued += 1;
        self.buckets[idx].push_back(packet);
    }

    /// Pop the highest-priority non-expired packet, or `None` if empty/all stale.
    ///
    /// Expired packets are discarded and counted in [`SendQueueStats::dropped_expired`].
    pub fn pop(&mut self, now: Instant) -> Option<OutgoingPacket> {
        loop {
            let idx = self.highest_nonempty()?;
            let packet = self.buckets[idx].pop_front()?;
            self.queued_packets -= 1;
            self.queued_bytes = self.queued_bytes.saturating_sub(packet.len());
            if packet.is_expired(now) {
                self.stats.dropped_expired += 1;
                continue;
            }
            self.stats.dequeued += 1;
            return Some(packet);
        }
    }

    /// Peek priority of the next packet that would be popped (ignores expiry until pop).
    pub fn peek_priority(&self) -> Option<Priority> {
        self.highest_nonempty().map(|i| match i {
            0 => Priority::Audio,
            1 => Priority::Retransmission,
            2 => Priority::Video,
            3 => Priority::Feedback,
            _ => Priority::Padding,
        })
    }

    /// Drop every queued packet (stats unchanged except lengths).
    pub fn clear(&mut self) {
        for bucket in &mut self.buckets {
            bucket.clear();
        }

        self.queued_packets = 0;
        self.queued_bytes = 0;
    }

    /// Rough time to drain `queued_bytes` at `rate_bps` (0 → `None`).
    pub fn expected_queue_time(&self, rate_bps: u64) -> Option<Duration> {
        if rate_bps == 0 || self.queued_bytes == 0 {
            return None;
        }

        let bits = (self.queued_bytes as u128) * 8;
        let us = bits.saturating_mul(1_000_000) / u128::from(rate_bps);
        Some(Duration::from_micros(us as u64))
    }

    fn highest_nonempty(&self) -> Option<usize> {
        self.buckets.iter().position(|b| !b.is_empty())
    }
}
