//! Prioritized send queue with TTL drop.
//!
//! Holds already-encoded UDP payloads waiting for the [`crate::core::pacer`].
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
//! | 0 | [`Priority::Audio`] | audio media |
//! | 1 | [`Priority::Retransmission`] | NACK-driven video resend |
//! | 2 | [`Priority::Feedback`] | NACK / ArrivalFeedback / keyframe request |
//! | 3 | [`Priority::Video`] | video media |
//! | 4 | [`Priority::Padding`] | probe / padding (lowest) |
//!
//! Classification is [`Priority::of`]. Same level is strict FIFO (no
//! multi-stream round-robin yet).
//!
//! TTL: at enqueue, `deadline = now + ttl_ms`. On pop, late packets are dropped
//! so a large video backlog cannot ship frames that are already useless.
//! Remaining-lifetime shrink while queued is represented by that absolute
//! deadline. The packet itself does not carry a TTL.
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
//! use bytes::Bytes;
//! use qrt::core::{
//!     packet::{MediaFragmentPacket, MediaType, Packet, Payload, Stream, StreamPacket},
//!     send_queue::{Priority, SendQueue},
//! };
//!
//! fn media(audio: bool, payload: &'static [u8]) -> Packet {
//!     Packet {
//!         sequence: 0,
//!         timestamp: 0,
//!         payload: Payload::Stream(Stream {
//!             id: 0,
//!             idx: 0,
//!             packet: StreamPacket::Media(MediaFragmentPacket {
//!                 sequence: 0,
//!                 id: 0,
//!                 fragment_idx: 0,
//!                 fragment_count: 1,
//!                 media_type: if audio {
//!                     MediaType::Audio
//!                 } else {
//!                     MediaType::Video
//!                 },
//!                 is_key_frame: false,
//!                 payload: Bytes::from_static(payload),
//!             }),
//!         }),
//!     }
//! }
//!
//! let t0 = Instant::now();
//! let mut q = SendQueue::new();
//!
//! assert!(q.enqueue_packet(&media(false, b"video"), 30, t0));
//! assert!(q.enqueue_packet(&media(true, b"audio"), 30, t0));
//! assert!(!q.enqueue_packet(&media(false, b"dead"), 0, t0));
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

use crate::core::packet::{MediaType, Packet, Payload, StreamPacket};

/// Send priority (lower discriminant = higher priority).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Priority {
    /// Real-time audio media.
    Audio = 0,
    /// NACK-driven video retransmission.
    Retransmission = 1,
    /// NACK, arrival feedback, and keyframe request. Ahead of video so a
    /// video backlog cannot hold recovery and bandwidth reports past their TTL.
    Feedback = 2,
    /// Video media (key or delta).
    Video = 3,
    /// Probe / padding filler (lowest).
    Padding = 4,
}

impl Priority {
    /// Number of distinct priority levels.
    pub const LEVELS: usize = 5;

    /// Classify a packet the way the send path should enqueue it.
    ///
    /// Retransmissions are not a wire flag. The history path sets
    /// [`Self::Retransmission`] on [`OutgoingPacket`] directly.
    ///
    /// # Examples
    ///
    /// ```
    /// use bytes::Bytes;
    /// use qrt::core::{
    ///     packet::{MediaFragmentPacket, MediaType, Packet, Payload, Stream, StreamPacket},
    ///     send_queue::Priority,
    /// };
    ///
    /// let audio = Packet {
    ///     sequence: 0,
    ///     timestamp: 0,
    ///     payload: Payload::Stream(Stream {
    ///         id: 0,
    ///         idx: 0,
    ///         packet: StreamPacket::Media(MediaFragmentPacket {
    ///             sequence: 0,
    ///             id: 0,
    ///             fragment_idx: 0,
    ///             fragment_count: 1,
    ///             media_type: MediaType::Audio,
    ///             is_key_frame: false,
    ///             payload: Bytes::from_static(b"a"),
    ///         }),
    ///     }),
    /// };
    /// assert_eq!(Priority::of(&audio), Priority::Audio);
    ///
    /// let nack = Packet {
    ///     sequence: 0,
    ///     timestamp: 0,
    ///     payload: Payload::Stream(Stream {
    ///         id: 0,
    ///         idx: 0,
    ///         packet: StreamPacket::KeyFrameRequest,
    ///     }),
    /// };
    /// assert_eq!(Priority::of(&nack), Priority::Feedback);
    /// ```
    pub fn of(packet: &Packet) -> Self {
        match &packet.payload {
            Payload::ArrivalFeedback(_) => Self::Feedback,
            Payload::Stream(stream) => match &stream.packet {
                StreamPacket::Media(media) if media.media_type == MediaType::Audio => Self::Audio,
                StreamPacket::Media(_) => Self::Video,
                StreamPacket::Nack(_) | StreamPacket::KeyFrameRequest => Self::Feedback,
            },
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
    /// Stream id, or `0` for connection-wide arrival feedback.
    pub stream_id: u8,
    /// `true` when this datagram is a NACK retransmission of an earlier send.
    pub retransmit: bool,
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
    /// use bytes::Bytes;
    /// use qrt::core::{
    ///     packet::{MediaFragmentPacket, MediaType, Packet, Payload, Stream, StreamPacket},
    ///     send_queue::{OutgoingPacket, Priority},
    /// };
    ///
    /// let pkt = Packet {
    ///     sequence: 0,
    ///     timestamp: 0,
    ///     payload: Payload::Stream(Stream {
    ///         id: 1,
    ///         idx: 0,
    ///         packet: StreamPacket::Media(MediaFragmentPacket {
    ///             sequence: 0,
    ///             id: 0,
    ///             fragment_idx: 0,
    ///             fragment_count: 1,
    ///             media_type: MediaType::Video,
    ///             is_key_frame: false,
    ///             payload: Bytes::from_static(b"x"),
    ///         }),
    ///     }),
    /// };
    /// let out = OutgoingPacket::from_packet(&pkt, 100, Instant::now()).unwrap();
    /// assert_eq!(out.priority, Priority::Video);
    /// assert_eq!(out.stream_id, 1);
    /// ```
    pub fn from_packet(packet: &Packet, ttl_ms: u16, now: Instant) -> Option<Self> {
        if ttl_ms == 0 {
            return None;
        }

        let stream_id = match &packet.payload {
            Payload::Stream(stream) => stream.id,
            Payload::ArrivalFeedback(_) => 0,
        };

        Some(Self {
            wire: packet.into_bytes(),
            priority: Priority::of(packet),
            stream_id,
            retransmit: false,
            enqueued_at: now,
            deadline: now + Duration::from_millis(u64::from(ttl_ms)),
        })
    }

    /// Wire length in bytes.
    pub fn len(&self) -> usize {
        self.wire.len()
    }

    /// Returns `true` when [`Self::len`] is zero.
    pub fn is_empty(&self) -> bool {
        self.wire.is_empty()
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
/// use bytes::Bytes;
/// use qrt::core::{
///     packet::{MediaFragmentPacket, MediaType, Packet, Payload, Stream, StreamPacket},
///     send_queue::SendQueue,
/// };
///
/// let now = Instant::now();
/// let fresh = Packet {
///     sequence: 0,
///     timestamp: 0,
///     payload: Payload::Stream(Stream {
///         id: 0,
///         idx: 0,
///         packet: StreamPacket::Media(MediaFragmentPacket {
///             sequence: 0,
///             id: 0,
///             fragment_idx: 0,
///             fragment_count: 1,
///             media_type: MediaType::Audio,
///             is_key_frame: false,
///             payload: Bytes::from_static(b"a"),
///         }),
///     }),
/// };
///
/// let mut q = SendQueue::new();
/// assert!(q.enqueue_packet(&fresh, 50, now));
/// assert!(!q.enqueue_packet(&fresh, 0, now));
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
    pub fn enqueue_packet(&mut self, packet: &Packet, ttl_ms: u16, now: Instant) -> bool {
        match OutgoingPacket::from_packet(packet, ttl_ms, now) {
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
    ///
    /// # Examples
    ///
    /// Feedback leaves before a video backlog, so the report is not stuck
    /// behind frames until its TTL expires.
    ///
    /// ```
    /// use std::time::{Duration, Instant};
    ///
    /// use bytes::Bytes;
    /// use qrt::core::{
    ///     packet::{
    ///         ArrivalFeedbackPacket,
    ///         MediaFragmentPacket,
    ///         MediaType,
    ///         Packet,
    ///         Payload,
    ///         Stream,
    ///         StreamPacket,
    ///     },
    ///     send_queue::SendQueue,
    /// };
    ///
    /// let now = Instant::now();
    /// let video = Packet {
    ///     sequence: 0,
    ///     timestamp: 0,
    ///     payload: Payload::Stream(Stream {
    ///         id: 0,
    ///         idx: 0,
    ///         packet: StreamPacket::Media(MediaFragmentPacket {
    ///             sequence: 1,
    ///             id: 1,
    ///             fragment_idx: 0,
    ///             fragment_count: 1,
    ///             media_type: MediaType::Video,
    ///             is_key_frame: false,
    ///             payload: Bytes::from_static(b"v"),
    ///         }),
    ///     }),
    /// };
    /// let feedback = Packet {
    ///     sequence: 0,
    ///     timestamp: 0,
    ///     payload: Payload::ArrivalFeedback(ArrivalFeedbackPacket {
    ///         range: 0..1,
    ///         received_mask: 1,
    ///         received: vec![0],
    ///     }),
    /// };
    ///
    /// let mut queue = SendQueue::new();
    /// assert!(queue.enqueue_packet(&video, 1_000, now));
    /// assert!(queue.enqueue_packet(&feedback, 30, now));
    /// let popped = queue.pop(now + Duration::from_millis(10)).unwrap();
    /// assert!(matches!(
    ///     Packet::from_bytes(popped.wire).unwrap().payload,
    ///     Payload::ArrivalFeedback(_)
    /// ));
    /// ```
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
            2 => Priority::Feedback,
            3 => Priority::Video,
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
