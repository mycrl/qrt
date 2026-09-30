//! Receive-side jitter / playout buffering.
//!
//! Two complementary controllers live here:
//!
//! | Path | Type | WebRTC counterpart |
//! |------|------|--------------------|
//! | Video | [`VideoFrameBuffer`] | `FrameBuffer` + `VCMTiming` / deadline drop |
//! | Audio | [`AudioNetEq`] | NetEQ **decision** skeleton (not full WSOLA) |
//!
//! Both sit **after** [`crate::core::fragment::Reassembly`] (video) or after
//! demux of complete audio packets. They decide *when* to release media to the
//! decoder / renderer and when to ask for a keyframe — they never open sockets.
//!
//! Aligns with `api/video/frame_buffer.*`,
//! `modules/video_coding/timing/*`, and
//! `modules/audio_coding/neteq/decision_logic.*` (see
//! `docs/webrtc-reference.md` §7–§9).
//!
//! # Video pipeline
//!
//! 1. [`crate::core::fragment::Reassembly`] emits [`crate::core::fragment::MediaPacket`].
//! 2. [`VideoFrameBuffer::push`] stores it with an arrival Instant and a
//!    playout deadline from the jitter estimate.
//! 3. Host timer / decoder-ready → [`VideoFrameBuffer::poll`]: release, drop
//!    late frames, or request a keyframe (throttled).
//!
//! # Audio pipeline
//!
//! 1. [`AudioNetEq::push`] stores codec frames with arrival time.
//! 2. Every ~10 ms → [`AudioNetEq::get_decision`] relative to target delay.
//! 3. Host decodes / stretches / runs PLC according to [`AudioDecision`]
//!    (codec PLC or simple repeat — WSOLA is out of scope).
//!
//! # Examples
//!
//! Video: a late frame is dropped when a newer one is already decodable:
//!
//! ```
//! use std::time::{Duration, Instant};
//! use bytes::Bytes;
//! use qrt::core::fragment::MediaPacket;
//! use qrt::core::jitter::{VideoFrameBuffer, VideoJitterConfig, VideoPoll};
//! use qrt::core::packet::MediaType;
//!
//! fn frame(id: u32, key: bool) -> MediaPacket {
//!     MediaPacket {
//!         id,
//!         media_type: MediaType::Video,
//!         is_key_frame: key,
//!         payload: Bytes::from_static(b"v"),
//!     }
//! }
//!
//! let t0 = Instant::now();
//! let mut buf = VideoFrameBuffer::new(0, VideoJitterConfig {
//!     max_delay: Duration::from_millis(80),
//!     ..VideoJitterConfig::default()
//! });
//! buf.push(frame(0, true), 0, t0);
//! // First keyframe is released immediately when decoder is ready.
//! assert!(matches!(
//!     buf.poll(t0, true),
//!     VideoPoll::Decode { frame, .. } if frame.id == 0
//! ));
//!
//! buf.push(frame(1, false), 3000, t0 + Duration::from_millis(10));
//! buf.push(frame(2, false), 6000, t0 + Duration::from_millis(20));
//! // Far past deadline with a newer frame waiting → drop 1 (and require key).
//! let late = t0 + Duration::from_millis(500);
//! assert!(matches!(
//!     buf.poll(late, true),
//!     VideoPoll::DroppedLate { frame_id: 1, .. }
//! ));
//! assert!(buf.keyframe_required());
//! ```
//!
//! # Notes
//!
//! - Video must **not** use [`AudioNetEq`]; audio must **not** use
//!   [`VideoFrameBuffer`].
//! - Keyframe requests are throttled (≥ [`DEFAULT_KEYFRAME_INTERVAL`]).
//! - Decode dependency is simplified (next `frame_id` or keyframe); full
//!   temporal-layer graphs are deferred.

use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

use bytes::Bytes;

use crate::core::{
    fragment::MediaPacket,
    packet::{Packet, Payload, Stream, StreamPacket},
};

/// Default minimum gap between keyframe-request emissions.
pub const DEFAULT_KEYFRAME_INTERVAL: Duration = Duration::from_secs(1);

/// How long after a video deadline we still wait before dropping (WebRTC ~5 ms).
pub const DEFAULT_LATE_GRACE: Duration = Duration::from_millis(5);

/// Default audio NetEQ decision period (WebRTC ~10 ms).
pub const DEFAULT_AUDIO_TICK: Duration = Duration::from_millis(10);

/// Default starting audio target delay (WebRTC ~80 ms).
pub const DEFAULT_AUDIO_TARGET_MS: Duration = Duration::from_millis(80);

// ---------------------------------------------------------------------------
// Video
// ---------------------------------------------------------------------------

/// Tunables for [`VideoFrameBuffer`].
///
/// # Examples
///
/// ```
/// use std::time::Duration;
///
/// use qrt::core::jitter::VideoJitterConfig;
///
/// let cfg = VideoJitterConfig::default();
/// assert!(cfg.min_delay <= cfg.max_delay);
/// assert!(cfg.render_delay >= Duration::from_millis(1));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoJitterConfig {
    /// Floor for target playout delay (low-latency mode may use `0`).
    pub min_delay: Duration,
    /// Cap on target playout delay (keeps end-to-end latency bounded).
    pub max_delay: Duration,
    /// Assumed decode + render budget subtracted when forming deadlines
    /// (WebRTC `render_delay` ~10 ms plus a small decode allowance).
    pub render_delay: Duration,
    /// Extra wait after deadline before dropping a frame.
    pub late_grace: Duration,
    /// Maximum buffered assembled frames; overflow drops oldest non-key first.
    pub max_frames: usize,
    /// Minimum interval between keyframe requests.
    pub keyframe_interval: Duration,
    /// If no decodable frame for this long while the stream is active, ask for
    /// a keyframe.
    pub stall_timeout: Duration,
    /// Release the first keyframe as soon as it is decodable.
    ///
    /// Later frames still wait until `now + late_grace >= deadline`. Turn this
    /// off when the first frame should honor [`Self::min_delay`] too.
    pub fast_start: bool,
}

impl Default for VideoJitterConfig {
    fn default() -> Self {
        Self {
            min_delay: Duration::from_millis(20),
            max_delay: Duration::from_millis(200),
            render_delay: Duration::from_millis(15),
            late_grace: DEFAULT_LATE_GRACE,
            max_frames: 32,
            keyframe_interval: DEFAULT_KEYFRAME_INTERVAL,
            stall_timeout: Duration::from_millis(500),
            fast_start: true,
        }
    }
}

/// Result of [`VideoFrameBuffer::poll`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VideoPoll {
    /// Release this frame to the decoder now.
    Decode {
        /// Reassembled frame.
        frame: MediaPacket,
        /// Capture timestamp from the packet that completed the frame.
        timestamp: u32,
    },
    /// Frame was too late and a newer decodable frame exists; do not decode it.
    DroppedLate {
        /// Dropped [`MediaPacket::id`].
        frame_id: u32,
        /// Dropped frame payload size (metrics).
        payload_len: usize,
    },
    /// Nothing ready yet (waiting for deadline, gaps, or `decoder_ready`).
    Wait,
    /// Emit a throttled keyframe request for this stream.
    KeyframeReq {
        /// Target stream id.
        stream_id: u8,
    },
}

/// One assembled frame held until its playout deadline.
#[derive(Debug, Clone)]
struct ScheduledFrame {
    frame: MediaPacket,
    timestamp: u32,
    deadline: Instant,
}

/// Video playout buffer with jitter-based deadlines and late-frame drop.
///
/// Owns assembled frames for one stream. Continuous
/// decoding requires the next [`MediaPacket::id`] after the last decoded frame, or a
/// keyframe (which resets the continuity cursor). Until the first keyframe,
/// only keyframes are released (`keyframe_required`).
///
/// # Examples
///
/// See the [module-level example](crate::core::jitter).
///
/// # Notes
///
/// - Pass `decoder_ready = false` to avoid unbounded growth when the decoder is
///   backed up; the buffer still drops under [`VideoJitterConfig::max_frames`].
/// - Jitter is an EWMA of inter-completion delays (IFDV stand-in for Kalman).
#[derive(Debug, Clone)]
pub struct VideoFrameBuffer {
    stream_id: u8,
    config: VideoJitterConfig,
    frames: VecDeque<ScheduledFrame>,
    last_decoded: Option<u32>,
    keyframe_required: bool,
    jitter_estimate: Duration,
    last_arrival: Option<Instant>,
    last_keyframe_req: Option<Instant>,
    last_decoded_at: Option<Instant>,
    last_push_at: Option<Instant>,
}

impl VideoFrameBuffer {
    /// Creates an empty buffer for `stream_id`.
    ///
    /// # Examples
    ///
    /// ```
    /// use qrt::core::jitter::{VideoFrameBuffer, VideoJitterConfig};
    ///
    /// let buf = VideoFrameBuffer::new(1, VideoJitterConfig::default());
    /// assert_eq!(buf.stream_id(), 1);
    /// assert_eq!(buf.len(), 0);
    /// assert!(buf.keyframe_required());
    /// ```
    pub fn new(stream_id: u8, config: VideoJitterConfig) -> Self {
        Self {
            stream_id,
            config,
            frames: VecDeque::new(),
            last_decoded: None,
            keyframe_required: true,
            jitter_estimate: Duration::from_millis(30),
            last_arrival: None,
            last_keyframe_req: None,
            last_decoded_at: None,
            last_push_at: None,
        }
    }

    /// Stream this buffer serves.
    pub fn stream_id(&self) -> u8 {
        self.stream_id
    }

    /// Number of assembled frames waiting for decode / drop.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Returns `true` when no frames are buffered.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Returns `true` until a keyframe has been successfully released.
    pub fn keyframe_required(&self) -> bool {
        self.keyframe_required
    }

    /// Current target playout delay (`jitter + render`, clamped to min/max).
    pub fn target_delay(&self) -> Duration {
        let raw = self
            .jitter_estimate
            .saturating_add(self.config.render_delay);

        raw.max(self.config.min_delay).min(self.config.max_delay)
    }

    /// Replaces config (applies to newly pushed frames' deadlines).
    pub fn set_config(&mut self, config: VideoJitterConfig) {
        self.config = config;
    }

    /// Inserts a reassembled frame and updates the jitter estimate.
    ///
    /// Duplicates of an already buffered [`MediaPacket::id`] are ignored.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::time::Instant;
    ///
    /// use bytes::Bytes;
    /// use qrt::core::{
    ///     fragment::MediaPacket,
    ///     jitter::{VideoFrameBuffer, VideoJitterConfig},
    ///     packet::MediaType,
    /// };
    ///
    /// let mut buf = VideoFrameBuffer::new(0, VideoJitterConfig::default());
    /// buf.push(
    ///     MediaPacket {
    ///         id: 1,
    ///         media_type: MediaType::Video,
    ///         is_key_frame: true,
    ///         payload: Bytes::from_static(b"i"),
    ///     },
    ///     0,
    ///     Instant::now(),
    /// );
    /// assert_eq!(buf.len(), 1);
    /// ```
    pub fn push(&mut self, frame: MediaPacket, timestamp: u32, now: Instant) {
        if self.frames.iter().any(|s| s.frame.id == frame.id) {
            return;
        }

        if let Some(prev) = self.last_arrival {
            let gap = now.saturating_duration_since(prev);
            // EWMA of inter-arrival (completion) delay as a simple jitter proxy.
            let j = self.jitter_estimate.as_secs_f64() * 1000.0;
            let g = gap.as_secs_f64() * 1000.0;
            let next = 0.85 * j + 0.15 * g;
            self.jitter_estimate = Duration::from_secs_f64((next / 1000.0).clamp(0.0, 1.0));
        }

        self.last_arrival = Some(now);
        self.last_push_at = Some(now);

        let deadline = now + self.target_delay();
        self.frames.push_back(ScheduledFrame {
            frame,
            timestamp,
            deadline,
        });

        self.frames.make_contiguous().sort_by_key(|s| s.frame.id);

        self.trim_overflow();
    }

    /// Advances playout: decode, drop late, wait, or request a keyframe.
    ///
    /// `decoder_ready` should be `false` when the decoder cannot accept more
    /// input (back-pressure). In that case this returns [`VideoPoll::Wait`]
    /// unless a stall forces a keyframe request.
    ///
    /// # Notes
    ///
    /// - Late drop: `now > deadline + late_grace` **and** a newer decodable
    ///   frame exists (WebRTC `DropNextDecodableTemporalUnit` idea).
    /// - Stall: no decode for [`VideoJitterConfig::stall_timeout`] while frames
    ///   are still arriving → throttled [`VideoPoll::KeyframeReq`].
    /// - A non-zero [`VideoJitterConfig::min_delay`] holds the frame until its
    ///   deadline. [`VideoJitterConfig::fast_start`] is the only immediate release,
    ///   and it applies to the first keyframe.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::time::{Duration, Instant};
    ///
    /// use bytes::Bytes;
    /// use qrt::core::{
    ///     fragment::MediaPacket,
    ///     jitter::{VideoFrameBuffer, VideoJitterConfig, VideoPoll},
    ///     packet::MediaType,
    /// };
    ///
    /// let t0 = Instant::now();
    /// let mut buf = VideoFrameBuffer::new(
    ///     0,
    ///     VideoJitterConfig {
    ///         fast_start: false,
    ///         min_delay: Duration::from_millis(40),
    ///         ..VideoJitterConfig::default()
    ///     },
    /// );
    /// buf.push(
    ///     MediaPacket {
    ///         id: 1,
    ///         media_type: MediaType::Video,
    ///         is_key_frame: true,
    ///         payload: Bytes::from_static(b"i"),
    ///     },
    ///     0,
    ///     t0,
    /// );
    /// assert!(matches!(buf.poll(t0, true), VideoPoll::Wait));
    /// assert!(matches!(
    ///     buf.poll(t0 + buf.target_delay(), true),
    ///     VideoPoll::Decode { .. }
    /// ));
    /// ```
    pub fn poll(&mut self, now: Instant, decoder_ready: bool) -> VideoPoll {
        if !decoder_ready {
            // Still allow stall keyframe so the sender can recover while decoder is busy.
            if let Some(req) = self.maybe_keyframe_on_stall(now) {
                return req;
            }

            return VideoPoll::Wait;
        }

        // Drop one late frame when a newer frame is already buffered (skip ahead).
        // One poll returns one drop so the host can count it.
        if let Some(front) = self.frames.front() {
            let late = now > front.deadline + self.config.late_grace;
            if late && self.frames.len() >= 2 {
                let dropped = self.frames.pop_front().expect("front checked");
                // Skipping any frame breaks opaque delta chains → need a keyframe.
                // Do not advance `last_decoded` for dropped keys: the decoder never
                // saw them, so the next delta must not be treated as contiguous.
                self.keyframe_required = true;

                return VideoPoll::DroppedLate {
                    frame_id: dropped.frame.id,
                    payload_len: dropped.frame.payload.len(),
                };
            }
        }

        // Deadline, plus the first keyframe when fast start is on.
        let idx = self.frames.iter().position(|scheduled| {
            let due = now + self.config.late_grace >= scheduled.deadline;
            let startup =
                self.config.fast_start && self.keyframe_required && scheduled.frame.is_key_frame;

            self.is_decodable(&scheduled.frame) && (due || startup)
        });

        if let Some(i) = idx {
            let scheduled = self.frames.remove(i).expect("index from position");
            self.on_decoded(&scheduled.frame, now);
            return VideoPoll::Decode {
                frame: scheduled.frame,
                timestamp: scheduled.timestamp,
            };
        }

        if !self.frames.iter().any(|s| self.is_decodable(&s.frame))
            && let Some(req) = self.try_keyframe_req(now)
        {
            return req;
        }

        if let Some(req) = self.maybe_keyframe_on_stall(now) {
            return req;
        }

        VideoPoll::Wait
    }

    /// Earliest playout wake time while frames are buffered.
    ///
    /// Used by the session I/O loop to `sleep_until` instead of blind polling.
    /// `None` when the buffer is empty (stall PLI is checked on UDP / NACK ticks
    /// via [`Self::poll`]).
    pub fn next_wake_at(&self, now: Instant) -> Option<Instant> {
        if self.config.fast_start
            && self.keyframe_required
            && self
                .frames
                .iter()
                .any(|scheduled| scheduled.frame.is_key_frame)
        {
            return Some(now);
        }

        let wake = self
            .frames
            .iter()
            .map(|scheduled| {
                // [`Self::poll`] releases when `now + late_grace >= deadline`.
                scheduled
                    .deadline
                    .checked_sub(self.config.late_grace)
                    .unwrap_or(scheduled.deadline)
            })
            .min()?;

        Some(if wake < now { now } else { wake })
    }

    /// Builds a keyframe-request [`Packet`] for this stream.
    ///
    /// # Examples
    ///
    /// ```
    /// use qrt::core::{
    ///     jitter::{VideoFrameBuffer, VideoJitterConfig},
    ///     packet::{Packet, Payload, StreamPacket},
    /// };
    ///
    /// let buf = VideoFrameBuffer::new(3, VideoJitterConfig::default());
    /// match buf.keyframe_packet().payload {
    ///     Payload::Stream(stream) => {
    ///         assert_eq!(stream.id, 3);
    ///         assert!(matches!(stream.packet, StreamPacket::KeyFrameRequest));
    ///     }
    ///     _ => panic!("expected keyframe request"),
    /// }
    /// ```
    pub fn keyframe_packet(&self) -> Packet {
        Packet {
            sequence: 0,
            timestamp: 0,
            payload: Payload::Stream(Stream {
                id: self.stream_id,
                idx: 0,
                packet: StreamPacket::KeyFrameRequest,
            }),
        }
    }

    /// Clears buffered frames and requires a keyframe (e.g. after decode error).
    pub fn reset(&mut self, now: Instant) {
        self.frames.clear();
        self.last_decoded = None;
        self.keyframe_required = true;
        self.last_decoded_at = Some(now);
    }

    fn is_decodable(&self, frame: &MediaPacket) -> bool {
        if self.keyframe_required {
            return frame.is_key_frame;
        }

        if frame.is_key_frame {
            return true;
        }

        match self.last_decoded {
            None => frame.is_key_frame,
            Some(id) => frame.id == id.wrapping_add(1) || frame.is_key_frame,
        }
    }

    fn on_decoded(&mut self, frame: &MediaPacket, now: Instant) {
        self.last_decoded = Some(frame.id);
        self.last_decoded_at = Some(now);
        if frame.is_key_frame {
            self.keyframe_required = false;
        }
    }

    fn trim_overflow(&mut self) {
        while self.frames.len() > self.config.max_frames {
            // Prefer dropping the oldest non-key frame.
            let drop_at = self
                .frames
                .iter()
                .position(|s| !s.frame.is_key_frame)
                .unwrap_or(0);
            self.frames.remove(drop_at);
        }
    }

    fn maybe_keyframe_on_stall(&mut self, now: Instant) -> Option<VideoPoll> {
        let last_activity = self.last_decoded_at.or(self.last_push_at)?;
        // Only stall-signal if we have recently been receiving or expecting media.
        let receiving = self
            .last_push_at
            .is_some_and(|t| now.saturating_duration_since(t) < self.config.stall_timeout * 2);
        if !receiving && self.frames.is_empty() {
            return None;
        }

        if now.saturating_duration_since(last_activity) < self.config.stall_timeout {
            return None;
        }

        // No progress decoding.
        if self.last_decoded_at.is_some()
            && now.saturating_duration_since(self.last_decoded_at.unwrap())
                < self.config.stall_timeout
        {
            return None;
        }

        self.try_keyframe_req(now)
    }

    fn try_keyframe_req(&mut self, now: Instant) -> Option<VideoPoll> {
        if let Some(last) = self.last_keyframe_req
            && now.saturating_duration_since(last) < self.config.keyframe_interval
        {
            return None;
        }

        self.last_keyframe_req = Some(now);
        self.keyframe_required = true;

        Some(VideoPoll::KeyframeReq {
            stream_id: self.stream_id,
        })
    }
}

// ---------------------------------------------------------------------------
// Audio (NetEQ decision skeleton)
// ---------------------------------------------------------------------------

/// Tunables for [`AudioNetEq`].
///
/// # Examples
///
/// ```
/// use std::time::Duration;
///
/// use qrt::core::jitter::AudioJitterConfig;
///
/// let cfg = AudioJitterConfig::default();
/// assert!(cfg.min_delay <= cfg.target_delay);
/// assert!(cfg.target_delay <= cfg.max_delay);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioJitterConfig {
    /// Nominal target buffer delay (WebRTC starts near 80 ms).
    pub target_delay: Duration,
    /// Floor after underrun optimization / clamping.
    pub min_delay: Duration,
    /// Ceiling for target delay.
    pub max_delay: Duration,
    /// Decision / output tick (typically 10 ms of audio).
    pub tick: Duration,
    /// Maximum queued packets.
    pub max_packets: usize,
}

impl Default for AudioJitterConfig {
    fn default() -> Self {
        Self {
            target_delay: DEFAULT_AUDIO_TARGET_MS,
            min_delay: Duration::from_millis(20),
            max_delay: Duration::from_millis(200),
            tick: DEFAULT_AUDIO_TICK,
            max_packets: 100,
        }
    }
}

/// One encoded audio frame in the NetEQ packet buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioPacket {
    /// Stream id (demux).
    pub stream_id: u8,
    /// 90 kHz (or codec) timestamp for ordering.
    pub timestamp: u32,
    /// Opaque encoded payload.
    pub payload: Bytes,
    /// When the packet was received locally.
    pub arrived_at: Instant,
}

/// NetEQ-style operations for one audio tick (host implements the DSP).
///
/// Maps to WebRTC `Operation` in `decision_logic.*`. qrt does **not** ship
/// WSOLA Accelerate / Expand; the host should call codec PLC or a simple
/// fade/repeat for [`AudioDecision::Expand`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioDecision {
    /// Decode and play the next packet at normal rate.
    Normal,
    /// Buffer above target — time-compress (Accelerate) if the host can;
    /// otherwise decode Normal and drop a packet to catch up.
    Accelerate,
    /// Buffer below target with packets available — prefer stretch
    /// (PreemptiveExpand) if available; otherwise Normal.
    PreemptiveExpand,
    /// No packet for this tick — run PLC / Expand.
    Expand,
}

/// Outcome of [`AudioNetEq::get_decision`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioTick {
    /// What to do this 10 ms tick.
    pub decision: AudioDecision,
    /// Packet to decode when decision is Normal / Accelerate / PreemptiveExpand.
    pub packet: Option<AudioPacket>,
    /// Current target delay after adapting to arrivals.
    pub target_delay: Duration,
    /// Estimated buffered audio time (packets × tick).
    pub buffer_delay: Duration,
}

/// Audio jitter buffer + NetEQ **decision** skeleton.
///
/// Maintains a packet queue, adapts a target delay from arrival spacing
/// (simplified histogram / EWMA), and emits [`AudioDecision`] each tick.
/// Actual sample-domain Accelerate / PreemptiveExpand / Expand belong in the
/// host or codec.
///
/// # Examples
///
/// ```
/// use std::time::{Duration, Instant};
///
/// use bytes::Bytes;
/// use qrt::core::jitter::{AudioDecision, AudioJitterConfig, AudioNetEq, AudioPacket};
///
/// let t0 = Instant::now();
/// let mut neteq = AudioNetEq::new(0, AudioJitterConfig::default());
/// neteq.push(AudioPacket {
///     stream_id: 0,
///     timestamp: 0,
///     payload: Bytes::from_static(b"a"),
///     arrived_at: t0,
/// });
///
/// // Target delay is 80 ms, so a tick at 10 ms is still buffering.
/// assert!(neteq.get_decision(t0 + Duration::from_millis(10)).is_none());
///
/// let tick = neteq
///     .get_decision(t0 + Duration::from_millis(80))
///     .expect("playout started");
/// assert!(tick.packet.is_some());
/// assert_ne!(tick.decision, AudioDecision::Expand);
///
/// // The same instant is not a second tick.
/// assert!(neteq.get_decision(t0 + Duration::from_millis(80)).is_none());
///
/// let plc = neteq
///     .get_decision(t0 + Duration::from_millis(90))
///     .expect("conceal tick");
/// assert_eq!(plc.decision, AudioDecision::Expand);
/// assert!(plc.packet.is_none());
/// ```
#[derive(Debug, Clone)]
pub struct AudioNetEq {
    stream_id: u8,
    config: AudioJitterConfig,
    packets: VecDeque<AudioPacket>,
    target: Duration,
    arrival_ewma: Duration,
    last_arrival: Option<Instant>,
    /// When the next decision may be produced. `None` until playout starts.
    next_playout: Option<Instant>,
}

impl AudioNetEq {
    /// Creates an empty audio buffer for `stream_id`.
    ///
    /// # Examples
    ///
    /// ```
    /// use qrt::core::jitter::{AudioJitterConfig, AudioNetEq};
    ///
    /// let n = AudioNetEq::new(2, AudioJitterConfig::default());
    /// assert_eq!(n.stream_id(), 2);
    /// assert_eq!(n.len(), 0);
    /// ```
    pub fn new(stream_id: u8, config: AudioJitterConfig) -> Self {
        let target = config.target_delay;
        Self {
            stream_id,
            arrival_ewma: config.target_delay,
            config,
            packets: VecDeque::new(),
            target,
            last_arrival: None,
            next_playout: None,
        }
    }

    /// Stream this instance serves.
    pub fn stream_id(&self) -> u8 {
        self.stream_id
    }

    /// Packets waiting in the buffer.
    pub fn len(&self) -> usize {
        self.packets.len()
    }

    /// Returns `true` when the packet buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.packets.is_empty()
    }

    /// Current adapted target delay.
    pub fn target_delay(&self) -> Duration {
        self.target
    }

    /// Replaces config and reclamps the live target.
    pub fn set_config(&mut self, config: AudioJitterConfig) {
        self.target = self.target.max(config.min_delay).min(config.max_delay);
        self.config = config;
    }

    /// Inserts one encoded audio packet (ordered by `timestamp`).
    ///
    /// Ignores other `stream_id`s. Drops oldest packets when over capacity.
    pub fn push(&mut self, packet: AudioPacket) {
        if packet.stream_id != self.stream_id {
            return;
        }

        if let Some(prev) = self.last_arrival {
            let gap = packet.arrived_at.saturating_duration_since(prev);
            let a = self.arrival_ewma.as_secs_f64() * 1000.0;
            let g = gap.as_secs_f64() * 1000.0;
            // Bias toward larger recent gaps (underrun / high-percentile stand-in).
            let next = 0.9 * a + 0.1 * g.max(a);
            self.arrival_ewma = Duration::from_secs_f64((next / 1000.0).clamp(0.0, 0.5));
            self.target = self
                .arrival_ewma
                .max(self.config.min_delay)
                .min(self.config.max_delay);
        }

        self.last_arrival = Some(packet.arrived_at);

        if let Some(i) = self
            .packets
            .iter()
            .position(|p| p.timestamp == packet.timestamp)
        {
            self.packets[i] = packet;
        } else {
            let ts = packet.timestamp;
            let insert_at = self
                .packets
                .iter()
                .position(|p| {
                    let d = ts.wrapping_sub(p.timestamp);
                    d > 0 && d < 0x8000_0000
                })
                .unwrap_or(self.packets.len());
            self.packets.insert(insert_at, packet);
        }

        while self.packets.len() > self.config.max_packets {
            self.packets.pop_front();
        }
    }

    /// When [`Self::get_decision`] will next return a tick.
    ///
    /// Before playout starts, this is the oldest packet's arrival plus the
    /// target delay. After that, it is one [`AudioJitterConfig::tick`] past
    /// the previous decision. `None` when the buffer is empty and playout has
    /// not started.
    pub fn next_playout_at(&self) -> Option<Instant> {
        if let Some(at) = self.next_playout {
            return Some(at);
        }

        let oldest = self.packets.front()?.arrived_at;

        Some(oldest + self.clamped_target())
    }

    /// Produces at most one decision when `now` has reached the next tick.
    ///
    /// `None` means wait. Playout starts once buffered audio reaches the target
    /// delay, or the oldest packet has waited that long. A later call at the
    /// same `now` returns `None`.
    ///
    /// # Notes
    ///
    /// - Empty buffer on a due tick → [`AudioDecision::Expand`]. The engine
    ///   surfaces that as [`crate::engine::EngineEvent::Conceal`], not as a
    ///   frame. The host runs PLC.
    /// - `buffer_delay > 1.2 × target` → [`AudioDecision::Accelerate`].
    /// - `buffer_delay < 0.8 × target` with packets →
    ///   [`AudioDecision::PreemptiveExpand`].
    /// - Otherwise → [`AudioDecision::Normal`].
    pub fn get_decision(&mut self, now: Instant) -> Option<AudioTick> {
        if let Some(at) = self.next_playout {
            if now < at {
                return None;
            }
        } else if self.packets.is_empty() {
            return None;
        } else {
            let oldest = self.packets.front().expect("buffer checked").arrived_at;
            let waited = now.saturating_duration_since(oldest);
            if self.buffer_delay() < self.clamped_target() && waited < self.clamped_target() {
                return None;
            }
        }

        let target = self.clamped_target();
        let buffer_delay = self.buffer_delay();
        let step = self.config.tick.max(Duration::from_nanos(1));
        self.next_playout = Some(now + step);

        if self.packets.is_empty() {
            return Some(AudioTick {
                decision: AudioDecision::Expand,
                packet: None,
                target_delay: target,
                buffer_delay,
            });
        }

        let high = Duration::from_secs_f64(target.as_secs_f64() * 1.2);
        let low = Duration::from_secs_f64(target.as_secs_f64() * 0.8);
        let decision = if buffer_delay > high {
            AudioDecision::Accelerate
        } else if buffer_delay < low {
            AudioDecision::PreemptiveExpand
        } else {
            AudioDecision::Normal
        };
        let packet = self.packets.pop_front();

        Some(AudioTick {
            decision,
            packet,
            target_delay: target,
            buffer_delay: self.buffer_delay(),
        })
    }

    fn clamped_target(&self) -> Duration {
        self.target
            .max(self.config.min_delay)
            .min(self.config.max_delay)
    }

    fn buffer_delay(&self) -> Duration {
        self.config.tick.saturating_mul(self.packets.len() as u32)
    }
}
