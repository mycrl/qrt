//! Synchronous, socket-free media engine.
//!
//! Think of [`Engine`] as a **pure state machine**. It has no UDP socket and no
//! background thread. You feed it three kinds of input; it returns a
//! [`TaskResult`] of things the host must do:
//!
//! | You call | Meaning |
//! |----------|---------|
//! | [`Engine::push_frame`] | "here is one encoded audio/video frame to send" |
//! | [`Engine::push_packet`] | "here is one UDP payload from the peer" |
//! | [`Engine::tick`] | "the wake timer fired; run due maintenance" |
//!
//! Every call ends in the same `finish` path: emit due NACK/feedback/probes,
//! release jitter-buffer frames, drain the pacer onto the wire, then tell you
//! the next [`TaskResult::next_wake`].
//!
//! # What the host does with [`TaskResult`]
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
//!         Packet         -> send on the peer UDP socket
//!         Frame            -> give to your decoder
//!         RateChange       -> retarget your encoder
//!         KeyframeRequest  -> force a video IDR
//!     schedule the next timer at result.next_wake
//! }
//! ```
//!
//! All `now` arguments must come from **one monotonic clock**.
//!
//! # Mental model of the internals
//!
//! ```text
//!                    LOCAL TRACK                         REMOTE TRACK
//!               (we capture / encode)                 (peer sent this)
//!
//!  EncodedFrame ──fragment──┐                    UDP ──decode──┐
//!                           │                                  │
//!                      optional XOR FEC                   arrival log (TWCC)
//!                           │                                  │
//!                           ▼                             Media / FEC / NACK /
//!                      Egress pacer                       Feedback / PLI
//!                    stamp transport_seq                       │
//!                    remember for NACK                    FEC may recover holes
//!                           │                                  │
//!                           ▼                                  ▼
//!                    EngineEvent::Packet              reassembly + jitter
//!                                                    EngineEvent::Frame
//!
//!  Shared across all tracks on this UDP flow:
//!    Egress      = one send queue + pacer + packet history
//!    Ingress     = one arrival recorder + FEC receiver + reassembler
//!    Congestion  = one BWE + RTT (drives pacer rate and encoder RateChange)
//! ```
//!
//! A [`TrackConfig::stream_id`] is **one direction**. Your camera and the
//! peer's camera use different ids so `media_seq` spaces never collide.
//!
//! Applications do not need to call [`crate::core`] algorithms directly.

mod congestion;
mod events;
mod media;
mod tracks;

use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

use ahash::{HashMap, HashMapExt};
use bytes::Bytes;
use congestion::Congestion;
pub use events::{EngineEvent, EngineInfo};
pub use media::{EncodedFrame, MediaKind, RateParams};
pub use tracks::TrackConfig;
use tracks::{Egress, Ingress, Track};

use crate::core::{
    bwe::BweConfig,
    feedback::FeedbackConfig,
    fragment::{FragmentError, PayloadSizeLimits},
    history::RetransmitOutcome,
    pacer::PacerConfig,
    packet::Packet,
};

/// Default media TTL when [`EncodedFrame::ttl_ms`] is `None`.
pub const DEFAULT_FRAME_TTL_MS: u16 = 200;

/// How often we ask BWE whether a probe burst should run.
///
/// Probes are extra paced packets so the delay-based estimator can climb when
/// the queue is empty (application-limited). 50 ms matches the usual GoogCC
/// poll cadence.
const PROBE_INTERVAL: Duration = Duration::from_millis(50);

/// Session-wide congestion, pacing, MTU, and history configuration.
///
/// Per-stream reliability and jitter settings live in [`TrackConfig`].
/// Change these before [`Engine::new`]; live BWE later overwrites pacer rate.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// TTL used when a pushed frame does not specify one.
    ///
    /// After this many milliseconds a queued packet is stale and will not be
    /// sent (or retransmitted).
    pub default_ttl_ms: u16,
    /// RTT used before arrival feedback yields a sample.
    ///
    /// Seeds NACK spacing and RTX "do not ask again this soon".
    pub initial_rtt: Duration,
    /// Max media body per UDP datagram (fragmentation / MTU).
    pub payload_limits: PayloadSizeLimits,
    /// Bandwidth-estimator start rate, pacing factor, and probe knobs.
    pub bwe: BweConfig,
    /// Initial leaky-bucket config. BWE overwrites `pacing_rate_bps` later.
    pub pacer: PacerConfig,
    /// How often we emit ArrivalFeedback and how long we keep arrival times.
    pub feedback: FeedbackConfig,
    /// How many first-send Media packets to keep so a NACK can clone them.
    pub history_capacity: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        let bwe = BweConfig::default();
        let pacer = PacerConfig {
            pacing_rate_bps: ((bwe.start_bitrate_bps as f64) * 1.1) as u64,
            ..PacerConfig::default()
        };

        Self {
            default_ttl_ms: DEFAULT_FRAME_TTL_MS,
            initial_rtt: Duration::from_millis(100),
            payload_limits: PayloadSizeLimits::default(),
            bwe,
            pacer,
            feedback: FeedbackConfig::default(),
            history_capacity: 600,
        }
    }
}

/// Error from a fallible [`Engine`] call ([`Engine::add_local_track`],
/// [`Engine::add_remote_track`], or [`Engine::push_frame`]).
///
/// # Examples
///
/// ```
/// use bytes::Bytes;
/// use qrt::{EncodedFrame, Engine, EngineConfig, EngineError, MediaKind};
///
/// let mut engine = Engine::new(EngineConfig::default());
/// let err = engine
///     .push_frame(
///         EncodedFrame::new(1, 0, MediaKind::Audio, false, Bytes::from_static(b"opus")),
///         std::time::Instant::now(),
///     )
///     .unwrap_err();
/// assert_eq!(err, EngineError::UnknownTrack { stream_id: 1 });
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineError {
    /// The stream identifier is already registered in either direction.
    TrackExists {
        /// Conflicting stream identifier.
        stream_id: u8,
    },
    /// No local track is registered for the stream identifier.
    UnknownTrack {
        /// Missing stream identifier.
        stream_id: u8,
    },
    /// Frame payload was empty.
    EmptyFrame,
    /// UDP egress is not enabled yet ([`Engine::set_send_ready`] is still false).
    NotReady {
        /// Stream that rejected the push.
        stream_id: u8,
    },
    /// Fragmentation failed under the configured MTU limits.
    Fragment(FragmentError),
    /// [`EncodedFrame::kind`] does not match the track kind.
    KindMismatch {
        /// Frame / track stream id.
        stream_id: u8,
        /// Kind on the track.
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
            Self::Fragment(error) => write!(f, "fragment: {error}"),
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

impl std::error::Error for EngineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Fragment(error) => Some(error),
            _ => None,
        }
    }
}

/// Result of one engine step ([`Engine::tick`], [`Engine::push_frame`], or
/// [`Engine::push_packet`]).
///
/// Iterate [`Self::events`] and dispatch on [`EngineEvent`]. Call
/// [`Engine::tick`] at [`Self::next_wake`].
///
/// # Notes
///
/// `next_wake` is an absolute [`Instant`]. A value at or before the call's
/// `now` means the engine has immediately runnable work.
#[derive(Debug, Clone, Default)]
pub struct TaskResult {
    /// Host-facing events in the order the engine produced them.
    ///
    /// Dispatch each variant (send UDP, decode, retarget encoder, force IDR).
    pub events: Vec<EngineEvent>,
    /// Earliest absolute time at which [`Engine::tick`] should run.
    ///
    /// `None` means nothing is scheduled (idle, empty queues, no remote tracks).
    pub next_wake: Option<Instant>,
}

impl TaskResult {
    /// UDP payloads in this batch, in production order.
    pub fn packets(&self) -> impl Iterator<Item = &Bytes> {
        self.events.iter().filter_map(|event| match event {
            EngineEvent::Packet(wire) => Some(wire),
            _ => None,
        })
    }

    /// Complete frames in this batch, in production order.
    pub fn frames(&self) -> impl Iterator<Item = &EncodedFrame> {
        self.events.iter().filter_map(|event| match event {
            EngineEvent::Frame(frame) => Some(frame),
            _ => None,
        })
    }
}

/// Single-threaded media core with no sockets, locks, runtime, or tasks.
///
/// Own one [`Engine`] per UDP conversation. Register local streams with
/// [`Self::add_local_track`] and remote streams with [`Self::add_remote_track`].
/// Then pump [`Self::push_frame`] / [`Self::push_packet`] / [`Self::tick`].
///
/// # Examples
///
/// A pushed frame produces one or more paced UDP payloads:
///
/// ```
/// use std::time::Instant;
///
/// use bytes::Bytes;
/// use qrt::{
///     EncodedFrame,
///     Engine,
///     EngineConfig,
///     MediaKind,
///     TrackConfig,
/// };
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
/// Two engines round-trip a complete frame:
///
/// ```
/// use std::time::Instant;
///
/// use bytes::Bytes;
/// use qrt::{
///     EncodedFrame,
///     Engine,
///     EngineConfig,
///     MediaKind,
///     TrackConfig,
/// };
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
/// `next_wake` schedules remaining paced work:
///
/// ```
/// use std::time::Instant;
///
/// use bytes::Bytes;
/// use qrt::{
///     EncodedFrame,
///     Engine,
///     EngineConfig,
///     MediaKind,
///     TrackConfig,
/// };
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
/// Methods mutate one owner synchronously. Move the engine to a dedicated host
/// thread if serialized access from an asynchronous application is desired.
pub struct Engine {
    /// Session tunables (default TTL, MTU, BWE, pacer, NACK history size).
    config: EngineConfig,
    /// One entry per `stream_id`. Local = we send, Remote = we receive.
    tracks: HashMap<u8, Track>,
    /// Shared send path: priority queue, leaky-bucket pacer, NACK history.
    egress: Egress,
    /// Shared receive path: arrival log, XOR FEC repair, frame reassembly.
    ingress: Ingress,
    /// One bandwidth estimator and RTT for the whole UDP flow.
    congestion: Congestion,
}

impl Engine {
    /// Creates an idle engine using `config`.
    ///
    /// No tracks yet. Call [`Self::add_local_track`] / [`Self::add_remote_track`]
    /// before pushing frames or expecting decode output.
    pub fn new(config: EngineConfig) -> Self {
        Self {
            tracks: HashMap::new(),
            egress: Egress::new(&config),
            ingress: Ingress::new(&config),
            congestion: Congestion::new(&config),
            config,
        }
    }

    /// Registers a local (we encode) track with egress initially gated.
    ///
    /// Media stays queued until [`Self::set_send_ready`] is `true` (typically
    /// after the TCP hello / track-open handshake).
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::TrackExists`] if the identifier is in use.
    pub fn add_local_track(&mut self, config: TrackConfig) -> Result<(), EngineError> {
        let stream_id = config.stream_id;
        if self.tracks.contains_key(&stream_id) {
            return Err(EngineError::TrackExists { stream_id });
        }

        self.tracks.insert(stream_id, Track::local(config));

        Ok(())
    }

    /// Registers a remote (peer encodes) track.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::TrackExists`] if the identifier is in use.
    pub fn add_remote_track(&mut self, config: TrackConfig) -> Result<(), EngineError> {
        let stream_id = config.stream_id;
        if self.tracks.contains_key(&stream_id) {
            return Err(EngineError::TrackExists { stream_id });
        }

        let mut track = Track::remote(config);

        // Seed NACK retransmit spacing with the session RTT guess until
        // arrival feedback produces a real sample.
        if let Some(remote) = track.as_remote_mut() {
            remote.set_rtt(self.congestion.rtt());
        }

        self.tracks.insert(stream_id, track);

        Ok(())
    }

    /// Removes a track and returns whether it existed.
    pub fn remove_track(&mut self, stream_id: u8) -> bool {
        let removed = self.tracks.remove(&stream_id).is_some();
        if removed {
            // Drop incomplete fragments so a reused stream_id cannot mix frames.
            self.ingress.clear_stream(stream_id);
        }

        removed
    }

    /// Enables or disables media egress for a local track.
    ///
    /// Returns `false` for a missing or remote track.
    pub fn set_send_ready(&mut self, stream_id: u8, ready: bool) -> bool {
        self.tracks
            .get_mut(&stream_id)
            .and_then(Track::as_local_mut)
            .map(|track| track.send_ready = ready)
            .is_some()
    }

    /// Returns whether a track uses `stream_id`.
    pub fn has_track(&self, stream_id: u8) -> bool {
        self.tracks.contains_key(&stream_id)
    }

    /// Returns a current congestion and queue snapshot.
    pub fn info(&self) -> EngineInfo {
        EngineInfo {
            target_bitrate_bps: self.congestion.target_bitrate_bps(),
            pacing_rate_bps: self.egress.pacing_rate_bps(),
            loss_ratio: self.congestion.loss_ratio(),
            rtt: self.congestion.rtt(),
            network: self.congestion.network_state(),
            in_flight_bytes: self.egress.in_flight_bytes(),
            queued_packets: self.egress.queued_len(),
            track_count: self.tracks.len(),
        }
    }

    /// Overrides RTT for NACK spacing, history, and future BWE updates.
    ///
    /// # Notes
    ///
    /// Values below one millisecond are clamped to one millisecond.
    pub fn set_rtt(&mut self, rtt: Duration) {
        self.congestion.set_rtt(rtt);
        let rtt = self.congestion.rtt();

        // History uses RTT as "do not RTX the same packet again this soon".
        self.egress.set_rtt(rtt);

        // NACK lists use RTT as the minimum time between asking for a seq.
        for track in self.tracks.values_mut() {
            if let Some(remote) = track.as_remote_mut() {
                remote.set_rtt(rtt);
            }
        }
    }

    /// Queues one encoded frame and returns every effect runnable at `now`.
    ///
    /// The frame is fragmented, optionally protected by XOR FEC, then passed
    /// through the pacer. Packets that cannot leave immediately remain
    /// queued and are exposed by a later [`Self::tick`].
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::EmptyFrame`], [`EngineError::UnknownTrack`],
    /// [`EngineError::NotReady`], [`EngineError::KindMismatch`], or
    /// [`EngineError::Fragment`].
    ///
    /// # Notes
    ///
    /// `now` establishes packet TTL deadlines. A zero effective TTL silently
    /// drops the frame and still advances normal engine maintenance.
    pub fn push_frame(
        &mut self,
        frame: EncodedFrame,
        now: Instant,
    ) -> Result<TaskResult, EngineError> {
        if frame.payload.is_empty() {
            return Err(EngineError::EmptyFrame);
        }

        let ttl = frame.ttl_ms.unwrap_or(self.config.default_ttl_ms);

        // TTL 0 = "already expired". Still run finish() so timers keep moving.
        if ttl == 0 {
            return Ok(self.finish(now, TaskResult::default()));
        }

        let (packets, fec_packets) = {
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

            // Step 1–2: Media fragments, then optional XOR rows for this frame.
            let packets = track.fragment(&frame, ttl, &self.config.payload_limits)?;
            let fec_packets = track.generate_fec(&packets);
            (packets, fec_packets)
        };

        {
            // Step 3: onto the shared pacer. They may leave in this finish() or later.
            for packet in &packets {
                self.egress.enqueue_packet(packet, now);
            }

            for fec in fec_packets {
                self.egress.enqueue_packet(&fec.as_packet(), now);
            }
        }

        Ok(self.finish(now, TaskResult::default()))
    }

    /// Processes one UDP payload and returns every effect runnable at `now`.
    ///
    /// This is the inbound counterpart of [`Self::push_frame`]. Malformed or
    /// irrelevant datagrams are ignored, but normal timer, jitter, and pacer
    /// maintenance still runs.
    ///
    /// # Notes
    ///
    /// Every decodable packet type is recorded for transport-wide arrival
    /// feedback before it is demultiplexed. Recovered FEC media is not recorded
    /// as a separate network arrival.
    pub fn push_packet(&mut self, payload: &[u8], now: Instant) -> TaskResult {
        let mut result = TaskResult::default();
        let Ok(packet) = Packet::decode(payload) else {
            // Garbage bytes: ignore, but still run NACK/pacer/jitter timers.
            return self.finish(now, result);
        };

        let header = packet.header().clone();

        // Every real UDP arrival feeds BWE, including control packets.
        self.ingress
            .record_arrival(header.transport_seq, now, payload.len());

        match packet {
            Packet::Media { .. } => {
                // `false` = this came off the wire, so also feed it to FEC.
                self.process_media_wires(
                    vec![(
                        header.stream_id,
                        header.media_seq,
                        Bytes::copy_from_slice(payload),
                        false,
                    )],
                    now,
                );
            }
            Packet::Fec { .. } => {
                // FEC itself is not a frame. Recovered Media is (`true` = skip
                // feeding FEC again; we already used this repair packet).
                let recovered = self
                    .ingress
                    .recover_fec(&packet)
                    .into_iter()
                    .map(|(stream_id, media_seq, wire)| (stream_id, media_seq, wire, true))
                    .collect();

                self.process_media_wires(recovered, now);
            }
            Packet::Nack { base_seq, blp, .. } => {
                // Peer is missing these media_seq. Clone from history if we can.
                for seq in Packet::nack_missing_seqs(base_seq, blp) {
                    if let RetransmitOutcome::Ready(outgoing) =
                        self.egress.retransmission(header.stream_id, seq, now)
                    {
                        self.egress.enqueue_retransmit(outgoing);
                    }
                }
            }
            Packet::ArrivalFeedback { .. } => {
                // Peer telling us which transport_seq arrived → run BWE.
                self.apply_arrival_feedback(&packet, now, &mut result);
            }
            Packet::KeyframeReq { stream_id, .. } => {
                // Only our local tracks can produce a new IDR.
                if self.tracks.get(&stream_id).is_some_and(Track::is_local) {
                    result
                        .events
                        .push(EngineEvent::KeyframeRequest { stream_id });
                }
            }
        }

        self.finish(now, result)
    }

    /// Advances timers and returns all effects runnable at `now`.
    ///
    /// # Notes
    ///
    /// Call again at [`TaskResult::next_wake`]. Calling earlier is safe but
    /// normally produces no additional paced work.
    pub fn tick(&mut self, now: Instant) -> TaskResult {
        self.finish(now, TaskResult::default())
    }
}

// Private helpers shared by the three public entry points.
// Kept in a second impl so the public API above stays scannable. Hosts never
// call these; every push_frame / push_packet / tick still ends in finish.
impl Engine {
    /// Shared tail of [`Self::push_frame`], [`Self::push_packet`], and
    /// [`Self::tick`].
    ///
    /// Hosts never call this. Every public entry ends here so they do not have
    /// to remember a second “run maintenance” API. Order:
    ///
    /// 1. [`Self::advance_control`] — due NACK / arrival reports / probes
    /// 2. [`Self::collect_inbound`] — jitter buffer → [`EngineEvent::Frame`]
    /// 3. [`Self::drain_pacer`] — leaky bucket → [`EngineEvent::Packet`]
    /// 4. [`Self::next_deadline`] — fill [`TaskResult::next_wake`]
    fn finish(&mut self, now: Instant, mut result: TaskResult) -> TaskResult {
        // Shared tail of push_frame / push_packet / tick. The host should
        // never have to remember a second API for "maintenance".
        {
            // Due NACK, arrival reports, probes.
            self.advance_control(now);

            // Jitter buffer -> Frame events.
            self.collect_inbound(now, &mut result);

            // Leaky bucket -> Packet events.
            self.drain_pacer(now, &mut result);
        }

        result.next_wake = self.next_deadline(now);

        result
    }

    /// Feeds one Media datagram (real or FEC-recovered) through NACK + reassembly.
    ///
    /// The queue exists because inserting one Media packet into FEC can recover
    /// *other* Media packets; those recovered wires go on the back and get the
    /// same NACK / reassembly treatment.
    ///
    /// # Notes
    ///
    /// The last tuple field is `recovered`. `false` = this came off the UDP
    /// socket, so also feed FEC. `true` = we invented this packet here; do not
    /// feed FEC again and do not log it as a new network arrival.
    fn process_media_wires(&mut self, initial: Vec<(u8, u16, Bytes, bool)>, now: Instant) {
        let mut pending = VecDeque::from(initial);
        let ttl = self.config.default_ttl_ms;

        while let Some((stream_id, media_seq, wire, recovered)) = pending.pop_front() {
            if !recovered {
                pending.extend(
                    self.ingress
                        .note_media_for_fec(stream_id, media_seq, &wire)
                        .into_iter()
                        .map(|(stream_id, media_seq, wire)| (stream_id, media_seq, wire, true)),
                );
            }

            {
                let Some(track) = self
                    .tracks
                    .get_mut(&stream_id)
                    .and_then(Track::as_remote_mut)
                else {
                    // Unknown / local-only id: drop. We already logged arrival.
                    continue;
                };

                if let Some(keyframe) = track.on_media_seq(media_seq, now, ttl) {
                    self.egress.enqueue_packet(&keyframe, now);
                }
            }

            {
                let Ok(packet) = Packet::decode(&wire) else {
                    continue;
                };

                // Incomplete frames stay in the reassembler until the last fragment.
                let Some(assembled) = self.ingress.reassemble(&packet) else {
                    continue;
                };

                if let Some(track) = self
                    .tracks
                    .get_mut(&stream_id)
                    .and_then(Track::as_remote_mut)
                {
                    track.push_assembled(assembled, now);
                }
            }
        }
    }

    /// Turns paced packets that are due at `now` into [`EngineEvent::Packet`].
    ///
    /// This is where `transport_seq` is stamped and NACK history is updated
    /// (see [`Egress::drain`]). Leftover queue items wait for a later tick.
    fn drain_pacer(&mut self, now: Instant, result: &mut TaskResult) {
        let mut datagrams = Vec::new();
        self.egress.drain(now, &mut datagrams);
        result
            .events
            .extend(datagrams.into_iter().map(EngineEvent::Packet));
    }

    /// Releases jitter-buffer frames that are due, and queues any PLI they emit.
    ///
    /// [`EngineEvent::Frame`] goes to the host decoder. Keyframe-request packets
    /// go onto [`Egress`] and leave in the same [`Self::drain_pacer`] pass.
    fn collect_inbound(&mut self, now: Instant, result: &mut TaskResult) {
        let ttl = self.config.default_ttl_ms;
        let mut controls = Vec::new();
        for track in self.tracks.values_mut() {
            let Some(remote) = track.as_remote_mut() else {
                continue;
            };

            let (frames, packets) = remote.drain_playout(now, ttl);
            result
                .events
                .extend(frames.into_iter().map(EngineEvent::Frame));
            controls.extend(packets);
        }

        for packet in controls {
            self.egress.enqueue_packet(&packet, now);
        }
    }

    /// Emits due *outgoing* control: our ArrivalFeedback, NACK lists, ALR probe.
    ///
    /// Also drops expired NACK history so we never RTX a stale frame. The
    /// packets sit on the pacer; [`Self::drain_pacer`] turns them into datagrams
    /// in this same [`Self::finish`] call.
    fn advance_control(&mut self, now: Instant) {
        // Tell the peer which transport_seq we have seen (TWCC-style report).
        if let Some(feedback) = self.ingress.poll_feedback(now) {
            self.egress.enqueue_packet(&feedback.as_packet(), now);
        }

        {
            // Ask each remote track for missing media_seq (RFC 4585-style NACK).
            let ttl = self.config.default_ttl_ms;
            let mut nacks = Vec::new();
            for track in self.tracks.values_mut() {
                if let Some(remote) = track.as_remote_mut() {
                    nacks.extend(remote.poll_nacks(now, ttl));
                }
            }

            for packet in nacks {
                self.egress.enqueue_packet(&packet, now);
            }
        }

        {
            let has_local = self.has_local();
            self.congestion
                .maybe_probe(&mut self.egress, has_local, now, self.config.pacer);

            // Drop history entries whose TTL expired — we never RTX a stale frame.
            self.egress.cull(now);
        }
    }

    /// Handles a peer ArrivalFeedback packet (send-side BWE input).
    ///
    /// Matches the report against packets we stamped with `transport_seq`,
    /// refreshes RTT, runs the estimator, and pushes one
    /// [`EngineEvent::RateChange`] per local track when the target moved.
    fn apply_arrival_feedback(
        &mut self,
        packet: &Packet<'_>,
        now: Instant,
        result: &mut TaskResult,
    ) {
        let Some(report) = self.egress.on_feedback_packet(packet, now) else {
            return;
        };

        if let Some(sample) = Congestion::rtt_from_report(&report, now) {
            self.set_rtt(sample);
        }

        // No RateChange event if BWE kept the same target.
        let Some(update) = self.congestion.apply_report(
            &report,
            &mut self.egress,
            now,
            self.config.bwe.pacing_factor,
            self.config.pacer,
        ) else {
            return;
        };

        for (stream_id, track_update) in
            Congestion::split_rates(&update, &self.tracks, self.config.bwe.pacing_factor)
        {
            result.events.push(EngineEvent::RateChange {
                stream_id,
                params: RateParams::from_rate_update(&track_update),
            });
        }
    }

    /// Earliest time any subsystem needs another [`Self::tick`].
    ///
    /// Minimum of leftover paced packets, the next arrival-feedback report,
    /// the next probe slot, and remote jitter / NACK wakes. `None` means idle.
    fn next_deadline(&self, now: Instant) -> Option<Instant> {
        // Wake as soon as *any* subsystem has work: leftover paced packets,
        // the next arrival-feedback report, a probe slot, or jitter/NACK.
        let paced = self.egress.next_send_time(now);
        let feedback = self.ingress.next_feedback_at(now);
        let probe = self.congestion.next_probe_at(now, self.has_local());
        let recv = self
            .tracks
            .values()
            .filter_map(|track| track.as_remote().and_then(|remote| remote.next_wake(now)))
            .min();

        [paced, feedback, probe, recv].into_iter().flatten().min()
    }

    /// True when this engine has at least one local (we encode) track.
    ///
    /// Probes and send-side BWE are pointless on a receive-only engine.
    fn has_local(&self) -> bool {
        self.tracks.values().any(Track::is_local)
    }
}
