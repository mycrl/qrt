//! Leaky-bucket media pacer.
//!
//! Schedules [`OutgoingPacket`]s from an internal [`SendQueue`] using the same
//! debt model as WebRTC's `PacingController`
//! (`modules/pacing/pacing_controller.*`).
//!
//! # How it works
//!
//! ```text
//!  enqueue_packet / enqueue
//!           │
//!           ▼
//!      SendQueue (priority + TTL)
//!           │
//!  poll(now)│
//!           ├─ advance_time: media_debt -= rate × Δt  (capped)
//!           ├─ if next is Audio and !account_for_audio → pop immediately
//!           ├─ else if media_debt <= burst_budget(40ms×rate) → pop, debt += size
//!           └─ else → None  (caller sleeps until next_send_time)
//! ```
//!
//! - **Burst budget**: debt may sit up to ~40ms of paced bytes so small packets
//!   can leave back-to-back without waiting a full inter-packet gap.
//! - **Max debt**: ~500ms of bytes at the configured rate (WebRTC-style cap).
//! - **Audio**: by default unpaced — does not grow debt — so a large I-frame
//!   cannot stall voice.
//! - **Queue drain boost**: if expected queue time exceeds
//!   [`PacerConfig::queue_time_limit`] (~2s), the effective drain rate rises so
//!   the backlog can clear (WebRTC `drain_large_queues`).
//!
//! Typical host loop:
//!
//! 1. On encoded media / feedback: [`Pacer::enqueue_packet`].
//! 2. On timer / writable socket: `while let Some(p) = pacer.poll(now) { send(p.wire); }`.
//! 3. Schedule the next wake at [`Pacer::next_send_time`].
//!
//! # Examples
//!
//! ```
//! use std::time::{Duration, Instant};
//!
//! use qrt::core::{
//!     pacer::{Pacer, PacerConfig},
//!     packet::{Flags, Header, Packet, PacketType},
//! };
//!
//! let mut pacer = Pacer::new(PacerConfig {
//!     pacing_rate_bps: 80_000, // 10 KB/s → ~400 B burst / 40ms
//!     ..PacerConfig::default()
//! });
//!
//! let now = Instant::now();
//! let pkt = Packet::Media {
//!     header: Header {
//!         packet_type: PacketType::Media,
//!         flags: Flags::default(),
//!         stream_id: 0,
//!         media_seq: 0,
//!         transport_seq: 0,
//!         frame_id: 0,
//!         frag_index: 0,
//!         frag_count: 1,
//!         timestamp: 0,
//!         ttl_ms: 200,
//!     },
//!     payload: &[0u8; 500],
//! };
//! assert!(pacer.enqueue_packet(&pkt, now));
//! assert!(pacer.poll(now).is_some());
//!
//! // Debt now exceeds the 40ms burst budget; wait before the next send.
//! assert!(pacer.enqueue_packet(&pkt, now));
//! assert!(pacer.poll(now).is_none());
//! let wake = pacer.next_send_time(now).unwrap();
//! assert!(wake > now);
//! assert!(pacer.poll(wake).is_some());
//! ```
//!
//! # Notes
//!
//! - Padding / probe cluster *generation* is a BWE concern; this type only
//!   rate-limits whatever you enqueue (including future padding packets).
//! - Assign [`crate::core::packet::Header::transport_seq`] at true send time (pacer egress or
//!   socket write), not only at fragment time — see `docs/webrtc-reference.md` §15.

use std::time::{Duration, Instant};

use crate::core::{
    packet::Packet,
    send_queue::{OutgoingPacket, Priority, SendQueue, SendQueueStats},
};

/// Default pacing burst window (WebRTC `PacerConfig::kDefaultTimeInterval`).
pub const DEFAULT_BURST_INTERVAL: Duration = Duration::from_millis(40);

/// Default maximum media debt age.
pub const DEFAULT_MAX_DEBT: Duration = Duration::from_millis(500);

/// Default queue-time limit before temporary rate boost.
pub const DEFAULT_QUEUE_TIME_LIMIT: Duration = Duration::from_secs(2);

/// Configuration for [`Pacer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PacerConfig {
    /// Target send rate in **bits per second** (media path).
    pub pacing_rate_bps: u64,
    /// How much paced debt may accumulate before the next send must wait.
    pub burst_interval: Duration,
    /// Cap on outstanding media debt duration at the pacing rate.
    pub max_debt: Duration,
    /// When expected queue time exceeds this, temporarily boost drain rate.
    pub queue_time_limit: Duration,
    /// When `false` (default), audio packets ignore the leaky bucket.
    pub account_for_audio: bool,
}

impl Default for PacerConfig {
    fn default() -> Self {
        Self {
            pacing_rate_bps: 1_500_000,
            burst_interval: DEFAULT_BURST_INTERVAL,
            max_debt: DEFAULT_MAX_DEBT,
            queue_time_limit: DEFAULT_QUEUE_TIME_LIMIT,
            account_for_audio: false,
        }
    }
}

/// Pacer wrapping a [`SendQueue`].
#[derive(Debug)]
pub struct Pacer {
    queue: SendQueue,
    config: PacerConfig,
    media_debt_bytes: u64,
    last_update: Option<Instant>,
}

impl Pacer {
    /// Create a pacer with the given config and an empty queue.
    pub fn new(config: PacerConfig) -> Self {
        Self {
            queue: SendQueue::new(),
            config,
            media_debt_bytes: 0,
            last_update: None,
        }
    }

    /// Borrow the underlying send queue.
    pub fn queue(&self) -> &SendQueue {
        &self.queue
    }

    /// Queue statistics (includes TTL drops).
    pub fn queue_stats(&self) -> SendQueueStats {
        self.queue.stats()
    }

    /// Replace pacing parameters (e.g. after a BWE update).
    pub fn set_config(&mut self, config: PacerConfig) {
        self.config = config;
        self.clamp_debt();
    }

    /// Current configured pacing rate (bps), before queue-drain boost.
    pub fn pacing_rate_bps(&self) -> u64 {
        self.config.pacing_rate_bps
    }

    /// Enqueue an encoded packet; returns `false` if dropped for zero TTL.
    pub fn enqueue_packet(&mut self, packet: &Packet<'_>, now: Instant) -> bool {
        self.queue.enqueue_packet(packet, now)
    }

    /// Enqueue a pre-built outgoing datagram.
    pub fn enqueue(&mut self, packet: OutgoingPacket) {
        self.queue.enqueue(packet)
    }

    /// Try to take one packet that may leave the host at `now`.
    ///
    /// Returns `None` when the queue is empty or the leaky bucket must wait
    /// (see [`Self::next_send_time`]).
    pub fn poll(&mut self, now: Instant) -> Option<OutgoingPacket> {
        self.advance_time(now);

        if !self.config.account_for_audio {
            if let Some(Priority::Audio) = self.queue.peek_priority() {
                return self.queue.pop(now);
            }
        }

        if !self.can_send_paced() {
            return None;
        }

        let packet = self.queue.pop(now)?;
        let counts_as_debt = self.config.account_for_audio || packet.priority != Priority::Audio;
        if counts_as_debt {
            self.media_debt_bytes = self.media_debt_bytes.saturating_add(packet.len() as u64);
            self.clamp_debt();
        }

        Some(packet)
    }

    /// Earliest instant when [`Self::poll`] may succeed again.
    ///
    /// `None` means the queue is empty (idle until the next enqueue).
    pub fn next_send_time(&self, now: Instant) -> Option<Instant> {
        if self.queue.is_empty() {
            return None;
        }

        if !self.config.account_for_audio {
            if let Some(Priority::Audio) = self.queue.peek_priority() {
                return Some(now);
            }
        }

        if self.can_send_paced() {
            return Some(now);
        }

        let rate = self.effective_rate_bps();
        if rate == 0 {
            return Some(now + self.config.burst_interval);
        }

        let budget = self.burst_budget_bytes();
        let need = self.media_debt_bytes.saturating_sub(budget);
        if need == 0 {
            return Some(now);
        }

        let us = (u128::from(need) * 8 * 1_000_000) / u128::from(rate);
        Some(now + Duration::from_micros(us as u64))
    }

    fn advance_time(&mut self, now: Instant) {
        let Some(last) = self.last_update else {
            self.last_update = Some(now);
            return;
        };

        if now <= last {
            return;
        }

        let elapsed = now - last;
        self.last_update = Some(now);

        let rate = self.effective_rate_bps();
        if rate > 0 {
            let drained = (u128::from(rate) * elapsed.as_micros()) / (8 * 1_000_000);
            self.media_debt_bytes = self.media_debt_bytes.saturating_sub(drained as u64);
        }
    }

    fn can_send_paced(&self) -> bool {
        if self.effective_rate_bps() == 0 {
            return false;
        }

        self.media_debt_bytes <= self.burst_budget_bytes()
    }

    fn burst_budget_bytes(&self) -> u64 {
        bytes_for_rate(self.effective_rate_bps(), self.config.burst_interval)
    }

    fn max_debt_bytes(&self) -> u64 {
        bytes_for_rate(self.config.pacing_rate_bps.max(1), self.config.max_debt)
    }

    fn clamp_debt(&mut self) {
        let max = self.max_debt_bytes();
        if self.media_debt_bytes > max {
            self.media_debt_bytes = max;
        }
    }

    fn effective_rate_bps(&self) -> u64 {
        let base = self.config.pacing_rate_bps;
        if base == 0 {
            return 0;
        }

        match self.queue.expected_queue_time(base) {
            Some(q) if q > self.config.queue_time_limit => {
                let bytes = self.queue.queued_bytes() as u128;
                let limit_us = self.config.queue_time_limit.as_micros().max(1);
                let boosted = (bytes * 8 * 1_000_000) / limit_us;
                (boosted as u64).max(base)
            }
            _ => base,
        }
    }
}

fn bytes_for_rate(rate_bps: u64, window: Duration) -> u64 {
    if rate_bps == 0 {
        return 0;
    }

    let us = window.as_micros();
    ((u128::from(rate_bps) * us) / (8 * 1_000_000)) as u64
}
