#![allow(dead_code)]
#![allow(clippy::enum_variant_names, clippy::wrong_self_convention)]

//! Synchronous, socket-free media engine.
//!
//! [`Engine`] has no UDP socket and no background thread. The host feeds it
//! three inputs and performs the [`TaskResult`] it returns:
//!
//! | You call | Meaning |
//! |----------|---------|
//! | [`Engine::push_frame`] | one encoded audio or video frame to send |
//! | [`Engine::push_packet`] | one UDP payload from the peer |
//! | [`Engine::tick`] | the wake timer fired |
//!
//! Each call ends in the same tail: due NACK and arrival reports join
//! the pacer, the jitter buffer releases frames, the pacer emits datagrams,
//! then `next_wake` says when to call `tick` again.
//!
//! ```text
//! loop {
//!     select {
//!         encoded frame => engine.push_frame(frame, now),
//!         UDP datagram  => engine.push_packet(datagram, now),
//!         next_wake     => engine.tick(now),
//!     }
//!
//!     for each EngineEvent in result.events:
//!         Packet           -> send on the peer UDP socket
//!         Frame            -> give to the decoder
//!         RateChange       -> retarget the encoder
//!         KeyframeRequest  -> force a video IDR
//!     schedule the next timer at result.next_wake
//! }
//! ```
//!
//! All `now` arguments come from one monotonic clock.
//!
//! ```text
//! EncodedFrame → LocalTrack (media sequence) → Path queue
//!     → stamp transport sequence at drain → EngineEvent::Packet
//!
//! UDP → Packet → Path records the arrival
//!     → Media  → RemoteTrack (NACK, reassembly, jitter) → EngineEvent::Frame
//!     → Nack   → Path history → retransmission onto the same queue
//!     → ArrivalFeedback → Path bandwidth estimate → EngineEvent::RateChange
//!     → KeyFrameRequest → EngineEvent::KeyframeRequest
//! ```
//!
//! [`TrackConfig::stream_id`] is one direction. The local camera and the
//! peer's camera use different ids so their media sequences do not collide.
//!
//! Two sequence spaces stay apart. The media sequence identifies a fragment
//! for NACK and reassembly. The transport sequence is stamped when the datagram
//! leaves, and is the only sequence bandwidth estimation uses.
//!
//! Applications do not need to call [`crate::core`] directly.

mod path;
mod track;

use std::time::{Duration, Instant};

use ahash::{HashMap, HashMapExt};
use bytes::Bytes;
use path::Path;
use track::Track;
pub use track::TrackConfig;

use crate::core::{
    bwe::{BweConfig, NetworkState},
    feedback::FeedbackConfig,
    fragment::DEFAULT_MAX_PACKET_SIZE,
    history::RetransmitOutcome,
    pacer::PacerConfig,
    packet::{Packet, Payload, StreamPacket},
};

/// Default media TTL when [`EncodedFrame::ttl_ms`] is `None`.
pub const DEFAULT_FRAME_TTL_MS: u16 = 200;

/// Whether an [`EncodedFrame`] carries audio or video.
///
/// Set this on [`TrackConfig::kind`] and on every pushed frame. A local track
/// rejects a frame of the other kind.
///
/// # Examples
///
/// ```
/// use qrt::engine::MediaKind;
///
/// assert_ne!(MediaKind::Audio, MediaKind::Video);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MediaKind {
    /// Voice or music. No keyframes.
    Audio,
    /// Camera or screen. May be a keyframe or a delta, and may produce a
    /// [`EngineEvent::KeyframeRequest`].
    Video,
}

/// One encoded media frame, either into [`Engine::push_frame`] or out to a decoder.
///
/// The engine does not interpret `payload`. Fragmentation, NACK, and jitter
/// only look at the fields below.
///
/// # Notes
///
/// `ttl_ms` is remaining lifetime, not a wall clock. `None` uses
/// [`EngineConfig::default_ttl_ms`]. Zero drops the frame without sending it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedFrame {
    /// Stream this frame belongs to ([`crate::core::packet::Stream::id`]).
    pub stream_id: u8,
    /// Capture timestamp in the media clock. Copied onto every fragment.
    pub timestamp: u32,
    /// Audio or video. Must match the registered track.
    pub kind: MediaKind,
    /// `true` for a video keyframe. Ignored for audio.
    pub keyframe: bool,
    /// Opaque codec bitstream.
    pub payload: Bytes,
    /// Remaining lifetime in milliseconds. `None` uses the engine default.
    pub ttl_ms: Option<u16>,
}

impl EncodedFrame {
    /// Builds a frame that uses [`EngineConfig::default_ttl_ms`].
    ///
    /// # Examples
    ///
    /// ```
    /// use bytes::Bytes;
    /// use qrt::engine::{EncodedFrame, MediaKind};
    ///
    /// let frame = EncodedFrame::new(
    ///     0,
    ///     90_000,
    ///     MediaKind::Audio,
    ///     false,
    ///     Bytes::from_static(b"opus"),
    /// );
    /// assert!(frame.ttl_ms.is_none());
    /// assert_eq!(frame.kind, MediaKind::Audio);
    /// ```
    pub fn new(
        stream_id: u8,
        timestamp: u32,
        kind: MediaKind,
        keyframe: bool,
        payload: Bytes,
    ) -> Self {
        Self {
            stream_id,
            timestamp,
            kind,
            keyframe,
            payload,
            ttl_ms: None,
        }
    }

    /// `true` when this video frame is marked as a keyframe.
    pub fn is_video_keyframe(&self) -> bool {
        self.kind == MediaKind::Video && self.keyframe
    }
}

/// Encoder settings from [`EngineEvent::RateChange`].
///
/// Apply `target_bitrate_bps` to the encoder. Probe bursts change how fast
/// the pacer drains; they are not a second target for the encoder.
///
/// # Examples
///
/// ```
/// use std::time::Duration;
///
/// use qrt::engine::RateParams;
///
/// let params = RateParams {
///     target_bitrate_bps: 800_000,
///     rtt: Duration::from_millis(50),
///     loss_ratio: 0.03,
/// };
/// assert!(params.loss_ratio < 0.1);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RateParams {
    /// Target encode bitrate in bits per second, after in-flight pushback.
    pub target_bitrate_bps: u64,
    /// RTT sample that produced this update.
    pub rtt: Duration,
    /// Smoothed loss ratio in `0.0..=1.0`.
    pub loss_ratio: f64,
}

impl RateParams {
    /// Keeps the encoder-facing fields of a transport rate update.
    fn from_rate_update(update: &crate::core::bwe::RateUpdate) -> Self {
        Self {
            target_bitrate_bps: update.target_bitrate_bps,
            rtt: update.rtt,
            loss_ratio: update.loss_ratio,
        }
    }
}

/// One thing the host must do after [`Engine::push_frame`], [`Engine::push_packet`],
/// or [`Engine::tick`].
///
/// # Examples
///
/// A peer keyframe request becomes an encoder event:
///
/// ```
/// use std::time::Instant;
///
/// use qrt::{
///     core::packet::{Packet, Payload, Stream, StreamPacket},
///     engine::{Engine, EngineConfig, EngineEvent, TrackConfig},
/// };
///
/// let mut engine = Engine::new(EngineConfig::default());
/// engine.add_local_track(TrackConfig::video(3)).unwrap();
/// let packet = Packet {
///     sequence: 1,
///     timestamp: 0,
///     payload: Payload::Stream(Stream {
///         id: 3,
///         idx: 0,
///         packet: StreamPacket::KeyFrameRequest,
///     }),
/// };
/// let wire = packet.into_bytes();
/// let out = engine.push_packet(&wire, Instant::now());
/// assert!(
///     out.events
///         .iter()
///         .any(|event| { matches!(event, EngineEvent::KeyframeRequest { stream_id: 3 }) })
/// );
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum EngineEvent {
    /// UDP payload to send now, in event order.
    ///
    /// The transport sequence is already stamped.
    Packet(Bytes),
    /// A complete frame for the decoder. NACK, reassembly, and jitter delay
    /// have already happened.
    Frame(EncodedFrame),
    /// Congestion control wants this local track at a new bitrate.
    ///
    /// Apply [`RateParams`] before the next [`Engine::push_frame`] for `stream_id`.
    RateChange {
        /// Local stream identifier.
        stream_id: u8,
        /// Encoder-facing rate, RTT, and loss.
        params: RateParams,
    },
    /// The peer, or this engine's jitter buffer, needs a video IDR.
    ///
    /// The next encoded video frame for `stream_id` should be a keyframe.
    KeyframeRequest {
        /// Local stream identifier.
        stream_id: u8,
    },
    /// An audio playout tick came due with nothing to decode.
    ///
    /// Run packet-loss concealment for `stream_id`. This is not a frame: the
    /// payload would be silence the engine does not synthesize. The session
    /// API logs it and leaves [`crate::RecvTrack::recv`] waiting for the next
    /// real frame.
    Conceal {
        /// Remote audio stream.
        stream_id: u8,
    },
}

/// Congestion and queue snapshot from [`Engine::info`].
///
/// This is not an [`EngineEvent`]. Poll it for a dashboard.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineInfo {
    /// Connection-wide encoder target, before the audio/video split, in bits/s.
    pub target_bitrate_bps: u64,
    /// Leaky-bucket send rate. Usually a bit above the encoder target.
    pub pacing_rate_bps: u64,
    /// Smoothed transport loss ratio in `0.0..=1.0`.
    pub loss_ratio: f64,
    /// Latest RTT. Arrival feedback, or the configured guess.
    pub rtt: Duration,
    /// Delay-based hypothesis (underuse / delay / overuse).
    pub network: NetworkState,
    /// Bytes sent and not yet covered by an arrival report.
    pub in_flight_bytes: usize,
    /// Packets still waiting in the pacer.
    pub queued_packets: usize,
    /// Registered local and remote tracks.
    pub track_count: usize,
}

impl EngineInfo {
    /// Congestion hints for an encoder, taken from this snapshot.
    ///
    /// This is the connection target, not a per-track split. Per-track targets
    /// arrive as [`EngineEvent::RateChange`].
    pub fn rate_params(&self) -> RateParams {
        RateParams {
            target_bitrate_bps: self.target_bitrate_bps,
            rtt: self.rtt,
            loss_ratio: self.loss_ratio,
        }
    }
}

/// Session-wide pacing, feedback, and history configuration.
///
/// Per-stream NACK and jitter live on [`TrackConfig`]. Build this before
/// [`Engine::new`]. Later arrival reports overwrite the pacer rate.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// TTL used when a pushed frame does not set [`EncodedFrame::ttl_ms`].
    ///
    /// After this many milliseconds a queued packet is not sent, and a stored
    /// copy is not retransmitted.
    pub default_ttl_ms: u16,
    /// RTT used until an arrival report yields a sample.
    ///
    /// Seeds how long NACK waits before asking for the same sequence again.
    pub initial_rtt: Duration,
    /// Maximum media body per UDP datagram, in bytes.
    pub max_packet_size: usize,
    /// Estimator start rate, pacing factor, and probe knobs.
    pub bwe: BweConfig,
    /// Initial leaky-bucket config. Arrival reports overwrite the rate.
    pub pacer: PacerConfig,
    /// How often arrival reports are emitted, and how long arrivals are kept.
    pub feedback: FeedbackConfig,
    /// How many first-send media packets to remember for NACK.
    pub history_capacity: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        let bwe = BweConfig::default();
        let pacer = PacerConfig {
            pacing_rate_bps: ((bwe.start_bitrate_bps as f64) * bwe.pacing_factor) as u64,
            ..PacerConfig::default()
        };

        Self {
            default_ttl_ms: DEFAULT_FRAME_TTL_MS,
            initial_rtt: Duration::from_millis(100),
            max_packet_size: DEFAULT_MAX_PACKET_SIZE,
            bwe,
            pacer,
            feedback: FeedbackConfig::default(),
            history_capacity: 600,
        }
    }
}

/// Failure from [`Engine::add_local_track`], [`Engine::add_remote_track`], or
/// [`Engine::push_frame`].
///
/// # Examples
///
/// ```
/// use qrt::engine::EngineError;
///
/// let err = EngineError::UnknownTrack { stream_id: 1 };
/// assert_eq!(err, EngineError::UnknownTrack { stream_id: 1 });
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineError {
    /// `stream_id` is already registered, in either direction.
    TrackExists {
        /// Conflicting stream identifier.
        stream_id: u8,
    },
    /// No local track is registered for this id.
    UnknownTrack {
        /// Missing stream identifier.
        stream_id: u8,
    },
    /// [`EncodedFrame::payload`] was empty.
    EmptyFrame,
    /// [`Engine::set_send_ready`] is still false for this local track.
    NotReady {
        /// Stream that rejected the push.
        stream_id: u8,
    },
    /// Fragmentation produced no packets. The MTU is zero, or the frame needs
    /// more than [`crate::core::fragment::MAX_FRAGMENTS_PER_FRAME`] pieces.
    Fragment,
    /// [`EncodedFrame::kind`] does not match the track.
    KindMismatch {
        /// Stream that rejected the frame.
        stream_id: u8,
        /// Kind registered on the track.
        track: MediaKind,
        /// Kind on the frame.
        frame: MediaKind,
    },
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TrackExists { stream_id } => {
                write!(f, "track already exists stream_id={stream_id}")
            }
            Self::UnknownTrack { stream_id } => write!(f, "unknown track stream_id={stream_id}"),
            Self::EmptyFrame => write!(f, "empty encoded frame"),
            Self::NotReady { stream_id } => {
                write!(f, "outbound not ready stream_id={stream_id}")
            }
            Self::Fragment => write!(f, "fragmentation failed"),
            Self::KindMismatch {
                stream_id,
                track,
                frame,
            } => write!(
                f,
                "kind mismatch on stream_id={stream_id}: track={track:?} frame={frame:?}"
            ),
        }
    }
}

impl std::error::Error for EngineError {}

/// What one [`Engine::push_frame`], [`Engine::push_packet`], or [`Engine::tick`]
/// produced.
///
/// Dispatch [`Self::events`], then call [`Engine::tick`] at [`Self::next_wake`].
///
/// # Notes
///
/// `next_wake` is absolute. A value at or before the call's `now` means work
/// is already runnable.
#[derive(Debug, Clone, Default)]
pub struct TaskResult {
    /// Events in the order the engine produced them.
    pub events: Vec<EngineEvent>,
    /// When [`Engine::tick`] should run next.
    ///
    /// `None` means idle: nothing queued, nothing to report, no remote track waiting.
    pub next_wake: Option<Instant>,
}

impl TaskResult {
    /// UDP payloads in this batch, in send order.
    pub fn packets(&self) -> impl Iterator<Item = &Bytes> {
        self.events.iter().filter_map(|event| match event {
            EngineEvent::Packet(wire) => Some(wire),
            _ => None,
        })
    }

    /// Complete frames in this batch, in playout order.
    pub fn frames(&self) -> impl Iterator<Item = &EncodedFrame> {
        self.events.iter().filter_map(|event| match event {
            EngineEvent::Frame(frame) => Some(frame),
            _ => None,
        })
    }
}

/// Media engine for one UDP conversation.
///
/// Register streams with [`Self::add_local_track`] and [`Self::add_remote_track`],
/// then pump [`Self::push_frame`], [`Self::push_packet`], and [`Self::tick`].
///
/// # Examples
///
/// A pushed frame becomes one or more paced UDP payloads:
///
/// ```
/// use std::time::Instant;
///
/// use bytes::Bytes;
/// use qrt::engine::{EncodedFrame, Engine, EngineConfig, MediaKind, TrackConfig};
///
/// let now = Instant::now();
/// let mut engine = Engine::new(EngineConfig::default());
/// engine.add_local_track(TrackConfig::audio(1)).unwrap();
/// engine.set_send_ready(1, true);
/// let out = engine
///     .push_frame(
///         EncodedFrame::new(1, 0, MediaKind::Audio, false, Bytes::from_static(b"opus")),
///         now,
///     )
///     .unwrap();
/// assert!(out.packets().next().is_some());
/// ```
///
/// Two engines round-trip one frame:
///
/// ```
/// use std::time::Instant;
///
/// use bytes::Bytes;
/// use qrt::engine::{EncodedFrame, Engine, EngineConfig, MediaKind, TrackConfig};
///
/// let now = Instant::now();
/// let mut tx = Engine::new(EngineConfig::default());
/// let mut rx = Engine::new(EngineConfig::default());
/// tx.add_local_track(TrackConfig::video(7)).unwrap();
/// rx.add_remote_track(TrackConfig::video(7)).unwrap();
/// tx.set_send_ready(7, true);
/// let sent = tx
///     .push_frame(
///         EncodedFrame::new(
///             7,
///             90_000,
///             MediaKind::Video,
///             true,
///             Bytes::from_static(b"frame"),
///         ),
///         now,
///     )
///     .unwrap();
/// let mut frames = Vec::new();
/// for wire in sent.packets() {
///     frames.extend(rx.push_packet(wire, now).frames().cloned());
/// }
/// assert_eq!(frames.len(), 1);
/// assert_eq!(frames[0].payload, Bytes::from_static(b"frame"));
/// ```
///
/// `next_wake` is when the pacer can release what did not fit in this call:
///
/// ```
/// use std::time::Instant;
///
/// use bytes::Bytes;
/// use qrt::engine::{EncodedFrame, Engine, EngineConfig, MediaKind, TrackConfig};
///
/// let now = Instant::now();
/// let mut engine = Engine::new(EngineConfig::default());
/// engine.add_local_track(TrackConfig::video(0)).unwrap();
/// engine.set_send_ready(0, true);
/// let out = engine
///     .push_frame(
///         EncodedFrame::new(0, 0, MediaKind::Video, true, Bytes::from(vec![1; 20_000])),
///         now,
///     )
///     .unwrap();
/// if let Some(wake) = out.next_wake {
///     let later = engine.tick(wake);
///     assert!(wake >= now);
///     assert!(later.packets().next().is_some() || later.next_wake.is_some());
/// }
/// ```
///
/// # Notes
///
/// Every method runs to completion on the caller. Share one engine across
/// tasks only by moving it to a single host thread.
pub struct Engine {
    config: EngineConfig,
    /// Local and remote tracks. The key is [`TrackConfig::stream_id`].
    tracks: HashMap<u8, Track>,
    /// The UDP pipe these tracks share.
    path: Path,
}

impl Engine {
    /// An idle engine. No tracks yet.
    pub fn new(config: EngineConfig) -> Self {
        Self {
            tracks: HashMap::new(),
            path: Path::new(&config),
            config,
        }
    }

    /// Registers a local track. Sending stays closed until [`Self::set_send_ready`].
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::TrackExists`] when `stream_id` is already used.
    pub fn add_local_track(&mut self, config: TrackConfig) -> Result<(), EngineError> {
        let stream_id = config.stream_id;
        if self.tracks.contains_key(&stream_id) {
            return Err(EngineError::TrackExists { stream_id });
        }

        self.tracks.insert(stream_id, Track::local(config));

        Ok(())
    }

    /// Registers a remote track and seeds its NACK timer with the current RTT.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::TrackExists`] when `stream_id` is already used.
    pub fn add_remote_track(&mut self, config: TrackConfig) -> Result<(), EngineError> {
        let stream_id = config.stream_id;
        if self.tracks.contains_key(&stream_id) {
            return Err(EngineError::TrackExists { stream_id });
        }

        let mut track = Track::remote(config);
        if let Some(remote) = track.as_remote_mut() {
            remote.set_rtt(self.path.rtt());
        }

        self.tracks.insert(stream_id, track);

        Ok(())
    }

    /// Removes a track. Returns whether it was registered.
    ///
    /// Dropping a remote track drops the fragments it was still reassembling,
    /// so a later track that reuses the id cannot mix in the old pieces.
    pub fn remove_track(&mut self, stream_id: u8) -> bool {
        self.tracks.remove(&stream_id).is_some()
    }

    /// Opens or closes sending on a local track.
    ///
    /// Returns `false` when `stream_id` is missing or is a remote track.
    pub fn set_send_ready(&mut self, stream_id: u8, ready: bool) -> bool {
        self.tracks
            .get_mut(&stream_id)
            .and_then(Track::as_local_mut)
            .map(|track| track.send_ready = ready)
            .is_some()
    }

    /// Whether `stream_id` is registered, in either direction.
    pub fn has_track(&self, stream_id: u8) -> bool {
        self.tracks.contains_key(&stream_id)
    }

    /// Current congestion and queue snapshot. Not produced by [`Self::tick`].
    pub fn info(&self) -> EngineInfo {
        let status = self.path.status();

        EngineInfo {
            target_bitrate_bps: status.target_bitrate_bps,
            pacing_rate_bps: status.pacing_rate_bps,
            loss_ratio: status.loss_ratio,
            rtt: status.rtt,
            network: status.network,
            in_flight_bytes: status.in_flight_bytes,
            queued_packets: status.queued_packets,
            track_count: self.tracks.len(),
        }
    }

    /// Overrides RTT for NACK spacing and for the retransmission gap.
    ///
    /// Values below one millisecond are clamped to one millisecond. The next
    /// arrival report can replace this sample.
    pub fn set_rtt(&mut self, rtt: Duration) {
        self.path.set_rtt(rtt);
        let rtt = self.path.rtt();
        for track in self.tracks.values_mut() {
            if let Some(remote) = track.as_remote_mut() {
                remote.set_rtt(rtt);
            }
        }
    }

    /// Fragments one encoded frame onto the pacer and runs every effect due at `now`.
    ///
    /// Packets that do not fit the leaky bucket stay queued. A later
    /// [`Self::tick`] emits them.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::EmptyFrame`], [`EngineError::UnknownTrack`],
    /// [`EngineError::NotReady`], [`EngineError::KindMismatch`], or
    /// [`EngineError::Fragment`].
    ///
    /// # Notes
    ///
    /// A zero TTL drops the frame and still runs [`Self::finish`], so timers
    /// do not stall because one frame was already expired.
    pub fn push_frame(
        &mut self,
        frame: EncodedFrame,
        now: Instant,
    ) -> Result<TaskResult, EngineError> {
        if frame.payload.is_empty() {
            return Err(EngineError::EmptyFrame);
        }

        let ttl = frame.ttl_ms.unwrap_or(self.config.default_ttl_ms);
        if ttl == 0 {
            tracing::trace!(stream_id = frame.stream_id, "dropped expired frame");

            return Ok(self.finish(now, TaskResult::default()));
        }

        let packets = {
            let track = self
                .tracks
                .get_mut(&frame.stream_id)
                .and_then(Track::as_local_mut)
                .ok_or(EngineError::UnknownTrack {
                    stream_id: frame.stream_id,
                })?;

            if !track.send_ready {
                return Err(EngineError::NotReady {
                    stream_id: frame.stream_id,
                });
            }

            if track.kind != frame.kind {
                return Err(EngineError::KindMismatch {
                    stream_id: frame.stream_id,
                    track: track.kind,
                    frame: frame.kind,
                });
            }

            track.fragment(&frame, self.config.max_packet_size)?
        };

        tracing::debug!(
            stream_id = frame.stream_id,
            timestamp = frame.timestamp,
            keyframe = frame.keyframe,
            bytes = frame.payload.len(),
            fragments = packets.len(),
            "enqueue frame"
        );

        for packet in &packets {
            self.path.enqueue_packet(packet, ttl, now);
        }

        Ok(self.finish(now, TaskResult::default()))
    }

    /// Demultiplexes one UDP payload and runs every effect due at `now`.
    ///
    /// Malformed datagrams are ignored. Timers, jitter, and the pacer still run.
    ///
    /// # Notes
    ///
    /// The arrival is recorded before the payload is demultiplexed. Bandwidth
    /// estimation counts NACK and feedback bytes, not only media.
    pub fn push_packet(&mut self, payload: &[u8], now: Instant) -> TaskResult {
        let mut result = TaskResult::default();
        let Ok(packet) = Packet::from_bytes(Bytes::copy_from_slice(payload)) else {
            tracing::trace!(bytes = payload.len(), "dropped malformed datagram");

            return self.finish(now, result);
        };

        self.path
            .record_arrival(packet.sequence, now, payload.len());

        match &packet.payload {
            Payload::Stream(stream) => match &stream.packet {
                StreamPacket::Media(media) => {
                    let stream_id = stream.id;
                    let timestamp = packet.timestamp;
                    let fragment = media.clone();
                    let ttl = self.config.default_ttl_ms;

                    tracing::trace!(
                        stream_id,
                        transport_seq = packet.sequence,
                        media_seq = media.sequence,
                        "media"
                    );

                    // An unregistered id is not buffered. There is no remote
                    // track to play the frame, and holding fragments would
                    // mix them into a track registered later under the same id.
                    let keyframe = self
                        .tracks
                        .get_mut(&stream_id)
                        .and_then(Track::as_remote_mut)
                        .and_then(|remote| remote.on_fragment(fragment, timestamp, now));

                    if let Some(keyframe) = keyframe {
                        self.path.enqueue_packet(&keyframe, ttl, now);
                    }
                }
                StreamPacket::Nack(nack) => {
                    let stream_id = stream.id;

                    // History clones the first send. The clone keeps that
                    // media sequence; drain assigns a new transport sequence.
                    // The wire format has no retransmission flag.
                    for seq in nack.sequences() {
                        match self.path.retransmission(stream_id, seq, now) {
                            RetransmitOutcome::Ready(outgoing) => {
                                tracing::debug!(stream_id, media_seq = seq, "retransmit");

                                self.path.enqueue_retransmit(outgoing);
                            }
                            skipped => {
                                tracing::trace!(
                                    stream_id,
                                    media_seq = seq,
                                    ?skipped,
                                    "retransmit skipped"
                                );
                            }
                        }
                    }
                }
                StreamPacket::KeyFrameRequest => {
                    let stream_id = stream.id;

                    tracing::trace!(stream_id, "inbound keyframe request");

                    if self.tracks.get(&stream_id).is_some_and(Track::is_local) {
                        result
                            .events
                            .push(EngineEvent::KeyframeRequest { stream_id });
                    }
                }
            },
            Payload::ArrivalFeedback(_) => {
                tracing::trace!(transport_seq = packet.sequence, "arrival feedback");

                let Some(effect) = self.path.on_arrival_feedback(&packet, now) else {
                    return self.finish(now, result);
                };

                if let Some(rtt) = effect.rtt {
                    self.set_rtt(rtt);
                }

                // An unchanged target must not emit RateChange. Encoders
                // would otherwise reset for a rate they already have.
                let Some(update) = effect.update else {
                    return self.finish(now, result);
                };

                let pacing_factor = self.config.bwe.pacing_factor;
                let locals: Vec<(u8, MediaKind)> = self
                    .tracks
                    .iter()
                    .filter_map(|(stream_id, track)| match track {
                        Track::Local(local) => Some((*stream_id, local.kind)),
                        Track::Remote(_) => None,
                    })
                    .collect();

                // `video_count` is at least 1 so a send path with only audio
                // still divides. That video share is not assigned to anyone.
                let video_count = locals
                    .iter()
                    .filter(|(_, kind)| *kind == MediaKind::Video)
                    .count()
                    .max(1);
                let audio_count = locals
                    .iter()
                    .filter(|(_, kind)| *kind == MediaKind::Audio)
                    .count();

                // About 10% of the target, clamped per audio track, so a video
                // climb cannot take the whole rate from an audio encoder.
                let audio_budget = if audio_count == 0 {
                    0
                } else {
                    (update.target_bitrate_bps / 10).clamp(16_000, 64_000 * audio_count as u64)
                };
                let video_rate =
                    update.target_bitrate_bps.saturating_sub(audio_budget) / video_count as u64;
                let audio_rate = if audio_count == 0 {
                    0
                } else {
                    audio_budget / audio_count as u64
                };

                for (stream_id, kind) in locals {
                    let mut track_update = update.clone();
                    track_update.target_bitrate_bps = match kind {
                        MediaKind::Video => video_rate,
                        MediaKind::Audio => audio_rate,
                    };
                    track_update.pacing_rate_bps =
                        ((track_update.target_bitrate_bps as f64) * pacing_factor).round() as u64;
                    result.events.push(EngineEvent::RateChange {
                        stream_id,
                        params: RateParams::from_rate_update(&track_update),
                    });
                }
            }
        }

        self.finish(now, result)
    }

    /// Runs everything that is due at `now` and was not attached to a frame or a datagram.
    ///
    /// # Notes
    ///
    /// Call again at [`TaskResult::next_wake`]. An earlier call is safe and
    /// usually has nothing new for the pacer to release.
    pub fn tick(&mut self, now: Instant) -> TaskResult {
        self.finish(now, TaskResult::default())
    }

    /// Shared tail of [`Self::push_frame`], [`Self::push_packet`], and [`Self::tick`].
    ///
    /// The three entries have to run the same maintenance or the host would
    /// need a second call to flush NACK, feedback, and the pacer. Order:
    ///
    /// 1. Queue arrival reports, NACK, and a probe rate change.
    /// 2. Release jitter frames. A stalled video buffer may queue a keyframe request.
    /// 3. Drain the pacer into [`EngineEvent::Packet`].
    /// 4. Set [`TaskResult::next_wake`] to the earliest of those timers.
    ///
    /// Control packets are queued before the drain, so a report that is due
    /// at `now` leaves in this same call instead of waiting for the next tick.
    fn finish(&mut self, now: Instant, mut result: TaskResult) -> TaskResult {
        let ttl = self.config.default_ttl_ms;
        let has_local = self.tracks.values().any(Track::is_local);

        {
            if let Some(feedback) = self.path.poll_feedback(now) {
                self.path.enqueue_packet(&feedback, ttl, now);
            }

            let mut nacks = Vec::new();
            for track in self.tracks.values_mut() {
                if let Some(remote) = track.as_remote_mut() {
                    nacks.extend(remote.poll_nacks(now));
                }
            }

            for packet in nacks {
                if let Payload::Stream(stream) = &packet.payload
                    && let StreamPacket::Nack(nack) = &stream.packet
                {
                    tracing::debug!(
                        stream_id = stream.id,
                        sequences = ?nack.sequences(),
                        "nack"
                    );
                }

                self.path.enqueue_packet(&packet, ttl, now);
            }

            self.path.consider_probe(now, has_local);

            // This call's NACK clones are already queued. Drop expired
            // first-sends so a later NACK cannot retransmit a stale frame.
            self.path.cull(now);
        }

        {
            let mut controls = Vec::new();
            for (stream_id, track) in &mut self.tracks {
                let Some(remote) = track.as_remote_mut() else {
                    continue;
                };

                let (frames, packets, conceal) = remote.drain_playout(now);
                if conceal {
                    result.events.push(EngineEvent::Conceal {
                        stream_id: *stream_id,
                    });
                }
                result
                    .events
                    .extend(frames.into_iter().map(EngineEvent::Frame));
                controls.extend(packets);
            }

            // Keyframe requests from the jitter buffer are packets on the
            // path, not Frame events. They leave in the drain below.
            for packet in controls {
                if let Payload::Stream(stream) = &packet.payload {
                    tracing::debug!(stream_id = stream.id, "keyframe request");
                }

                self.path.enqueue_packet(&packet, ttl, now);
            }
        }

        {
            let datagrams = self.path.drain(now);
            result
                .events
                .extend(datagrams.into_iter().map(EngineEvent::Packet));
        }

        let paced = self.path.next_send_time(now);
        let feedback_at = self.path.next_feedback_at(now);
        let probe = self.path.next_probe_at(now, has_local);
        let recv = self
            .tracks
            .values()
            .filter_map(|track| track.as_remote().and_then(|remote| remote.next_wake(now)))
            .min();
        result.next_wake = [paced, feedback_at, probe, recv]
            .into_iter()
            .flatten()
            .min();

        result
    }
}
