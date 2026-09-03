//! Sync media session over [`crate::core`].
//!
//! Models WebRTC's **Call + streams** split:
//!
//! - [`Session`] owns shared BWE, pacer, `transport_seq`, arrival feedback,
//!   packet history, reassembly, and FEC recovery cache.
//! - [`Session::alloc_outbound`] + [`Session::register_track`] create an
//!   independent media **track** (one
//!   [`Header::stream_id`](crate::core::packet::Header::stream_id)): its own
//!   `media_seq` / `frame_id`, FEC generator, NACK requester, and
//!   video/audio jitter buffer, plus a per-track [`Encoder`].
//!
//! Prefer [`crate::Qrt`] for application I/O (loop starts in [`crate::Qrt::new`]).
//! Drive this type directly only when you own a custom socket loop:
//!
//! ```text
//!   Session::alloc_outbound(stream_id, kind) -> EncodedFrameSender
//!   construct Encoder with sender          -> Box<dyn Encoder>
//!   Session::register_track(...)             -> RemoteTrack
//!   your encode loop -> sender.push_frame
//!              |
//!          poll_datagram  ->  UDP send
//!   UDP recv -> handle_datagram
//!              |
//!          pump_inbound -> EncodedFrameReceiver
//! ```

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use ahash::{HashMap, HashMapExt};
use bytes::Bytes;
use tokio::sync::Notify;

use crate::{
    codec::{
        CodecRateParams, EncodedFrame, EncodedFrameReceiver, EncodedFrameSender, Encoder,
        InboundFrameQueue, MediaKind, PendingFrameQueue, PushError, WakeNotify, push_inbound,
    },
    core::{
        bwe::{BandwidthEstimator, BweConfig, NetworkState, RateUpdate, send_side_pushback},
        fec::{FecGenerator, FecProtectionParams, FecReceiver},
        feedback::{ArrivalRecorder, FeedbackAdapter, FeedbackConfig, TransportSeqAssigner},
        fragment::{FragmentParams, PayloadSizeLimits, fragment},
        history::{PacketHistory, RetransRateLimiter, RetransmitOutcome},
        jitter::{
            AudioJitterConfig, AudioNetEq, AudioPacket, VideoFrameBuffer, VideoJitterConfig,
            VideoPoll,
        },
        nack::{NackConfig, NackRequester},
        pacer::{Pacer, PacerConfig},
        packet::{Flags, Header, Packet, PacketType},
        reassembly::FrameReassembler,
    },
};

/// Default media TTL when [`EncodedFrame::ttl_ms`] is `None`.
pub const DEFAULT_FRAME_TTL_MS: u16 = 200;

/// Session-wide knobs shared by every track (congestion + MTU + history).
///
/// Per-stream FEC / NACK / jitter live on [`TrackConfig`].
///
/// # Examples
///
/// ```
/// use qrt::session::QrtConfig;
/// let cfg = QrtConfig::default();
/// assert!(cfg.default_ttl_ms > 0);
/// ```
#[derive(Debug, Clone)]
pub struct QrtConfig {
    /// TTL used when a pushed frame omits [`EncodedFrame::ttl_ms`].
    pub default_ttl_ms: u16,
    /// Media body MTU knobs for fragmentation.
    pub payload_limits: PayloadSizeLimits,
    /// Congestion control bounds / start rate.
    pub bwe: BweConfig,
    /// Initial pacer rate (updated from BWE).
    pub pacer: PacerConfig,
    /// Arrival-feedback interval / retention.
    pub feedback: FeedbackConfig,
    /// Retransmission history capacity.
    pub history_capacity: usize,
}

impl Default for QrtConfig {
    fn default() -> Self {
        let bwe = BweConfig::default();
        let mut pacer = PacerConfig::default();
        pacer.pacing_rate_bps = ((bwe.start_bitrate_bps as f64) * 1.1) as u64;
        Self {
            default_ttl_ms: DEFAULT_FRAME_TTL_MS,
            payload_limits: PayloadSizeLimits::default(),
            bwe,
            pacer,
            feedback: FeedbackConfig::default(),
            history_capacity: 600,
        }
    }
}

/// Configuration for one media track ([`crate::Qrt::add_track`] /
/// [`Session::register_track`]).
///
/// # Examples
///
/// ```
/// use qrt::{codec::MediaKind, session::TrackConfig};
///
/// let video = TrackConfig::video(0);
/// assert_eq!(video.kind, MediaKind::Video);
/// assert!(video.enable_fec);
///
/// let audio = TrackConfig::audio(1);
/// assert_eq!(audio.kind, MediaKind::Audio);
/// assert!(!audio.enable_fec);
/// ```
#[derive(Debug, Clone)]
pub struct TrackConfig {
    /// Wire [`crate::core::packet::Header::stream_id`].
    pub stream_id: u8,
    /// Audio vs video (selects jitter path and FEC default).
    pub kind: MediaKind,
    /// When `true` and [`MediaKind::Video`], protect media with XOR FEC.
    pub enable_fec: bool,
    /// Video FEC rate (ignored when FEC is off).
    pub fec: FecProtectionParams,
    /// Receive NACK requester for this stream.
    pub nack: NackConfig,
    /// Receive video jitter / deadline (video tracks).
    pub video_jitter: VideoJitterConfig,
    /// Receive audio NetEQ skeleton (audio tracks).
    pub audio_jitter: AudioJitterConfig,
}

impl TrackConfig {
    /// Video track defaults (`enable_fec = true`).
    pub fn video(stream_id: u8) -> Self {
        Self {
            stream_id,
            kind: MediaKind::Video,
            enable_fec: true,
            fec: FecProtectionParams::default(),
            nack: NackConfig::default(),
            video_jitter: VideoJitterConfig::default(),
            audio_jitter: AudioJitterConfig::default(),
        }
    }

    /// Audio track defaults (`enable_fec = false`).
    pub fn audio(stream_id: u8) -> Self {
        Self {
            stream_id,
            kind: MediaKind::Audio,
            enable_fec: false,
            fec: FecProtectionParams::default(),
            nack: NackConfig::default(),
            video_jitter: VideoJitterConfig::default(),
            audio_jitter: AudioJitterConfig::default(),
        }
    }
}

/// Remote / receive track delivered via [`crate::QrtObserver::on_track`].
pub struct RemoteTrack {
    /// Wire [`crate::core::packet::Header::stream_id`].
    pub stream_id: u8,
    /// Audio vs video for this receive stream.
    pub kind: MediaKind,
    /// Pull reassembled frames after jitter / NetEQ ([`Session::pump_inbound`]).
    pub receiver: EncodedFrameReceiver,
}

/// Snapshot of congestion / queue state for the application UI or logging.
#[derive(Debug, Clone, PartialEq)]
pub struct QrtInfo {
    /// Encoder target bitrate (bps) after pushback (session total).
    pub target_bitrate_bps: u64,
    /// Pacer rate (bps).
    pub pacing_rate_bps: u64,
    /// Smoothed loss ratio `0.0..=1.0`.
    pub loss_ratio: f64,
    /// Latest RTT estimate.
    pub rtt: Duration,
    /// Delay-based network hypothesis.
    pub network: NetworkState,
    /// Bytes still in flight (send history not yet covered by feedback).
    pub in_flight_bytes: usize,
    /// Packets waiting in the send queue (all priorities).
    pub queued_packets: usize,
    /// Number of registered tracks.
    pub track_count: usize,
}

/// Error from track registration on [`Session`].
///
/// Frame push failures use [`crate::codec::PushError`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QrtError {
    /// A track with this `stream_id` already exists.
    TrackExists {
        /// Conflicting stream id.
        stream_id: u8,
    },
    /// No track registered for this `stream_id`.
    UnknownTrack {
        /// Missing stream id.
        stream_id: u8,
    },
}

impl std::fmt::Display for QrtError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TrackExists { stream_id } => {
                write!(f, "track already exists stream_id={stream_id}")
            }
            Self::UnknownTrack { stream_id } => write!(f, "unknown track stream_id={stream_id}"),
        }
    }
}

impl std::error::Error for QrtError {}

/// Per-track state (WebRTC send+receive stream for one `stream_id`).
struct Track {
    stream_id: u8,
    kind: MediaKind,
    encoder: Box<dyn Encoder>,
    /// Frames pushed by the encoder's [`EncodedFrameSender`].
    pending: PendingFrameQueue,
    /// Frames ready for the app after jitter / NetEQ.
    inbound: InboundFrameQueue,
    /// Wakes [`EncodedFrameReceiver::recv`] waiters.
    inbound_notify: Arc<Notify>,
    next_frame_id: u32,
    next_media_seq: u16,
    fec_gen: Option<FecGenerator>,
    nack: NackRequester,
    video_jitter: Option<VideoFrameBuffer>,
    audio_jitter: Option<AudioNetEq>,
    last_nack_at: Option<Instant>,
}

impl Track {
    fn new(
        config: TrackConfig,
        encoder: Box<dyn Encoder>,
        pending: PendingFrameQueue,
        inbound: InboundFrameQueue,
        inbound_notify: Arc<Notify>,
    ) -> Self {
        let stream = config.stream_id;
        let fec_gen = if config.kind == MediaKind::Video && config.enable_fec {
            Some(FecGenerator::new(stream, config.fec))
        } else {
            None
        };

        let (video_jitter, audio_jitter) = match config.kind {
            MediaKind::Video => (
                Some(VideoFrameBuffer::new(stream, config.video_jitter)),
                None,
            ),
            MediaKind::Audio => (None, Some(AudioNetEq::new(stream, config.audio_jitter))),
        };

        Self {
            stream_id: stream,
            kind: config.kind,
            encoder,
            pending,
            inbound,
            inbound_notify,
            next_frame_id: 1,
            next_media_seq: 1,
            fec_gen,
            nack: NackRequester::new(stream, config.nack),
            video_jitter,
            audio_jitter,
            last_nack_at: None,
        }
    }

    fn set_rtt(&mut self, rtt: Duration) {
        self.nack.set_rtt(rtt);
    }
}

/// End-to-end transport session: register tracks, feed UDP.
///
/// Call [`Self::poll_datagram`] / [`Self::handle_datagram`] from your
/// socket loop with a host [`Instant`]. Allocate an [`EncodedFrameSender`] with
/// [`Self::alloc_outbound`], create the encoder, then
/// [`Self::register_track`]. Decoders pull via [`RemoteTrack::receiver`] after
/// [`Self::pump_inbound`].
///
/// # Examples
///
/// Local loopback:
///
/// ```
/// use std::time::{Duration, Instant};
///
/// use bytes::Bytes;
/// use qrt::{
///     codec::{CodecRateParams, EncodedFrame, Encoder, MediaKind},
///     session::{QrtConfig, Session, TrackConfig},
/// };
///
/// struct NopEnc;
/// impl Encoder for NopEnc {
///     fn on_rate_params(&mut self, _: &CodecRateParams) {}
///     fn on_keyframe_request(&mut self) {}
/// }
///
/// let t0 = Instant::now();
/// let mut sender = Session::new(QrtConfig::default());
/// let mut receiver = Session::new(QrtConfig::default());
///
/// let (mut frame_tx, pending) = sender.alloc_outbound(0, MediaKind::Video);
/// sender
///     .register_track(TrackConfig::video(0), Box::new(NopEnc), pending)
///     .unwrap();
///
/// let (_rx_tx, pending_rx) = receiver.alloc_outbound(0, MediaKind::Video);
/// let mut remote = receiver
///     .register_track(TrackConfig::video(0), Box::new(NopEnc), pending_rx)
///     .unwrap();
///
/// frame_tx
///     .push_frame(
///         EncodedFrame::new(
///             0,
///             90_000,
///             MediaKind::Video,
///             true,
///             Bytes::from(vec![7u8; 100]),
///         ),
///         t0,
///     )
///     .unwrap();
///
/// while let Some(wire) = sender.poll_datagram(t0) {
///     receiver.handle_datagram(&wire, t0 + Duration::from_millis(5));
/// }
/// let mut got = None;
/// for ms in [0u64, 30, 60, 100, 150] {
///     let now = t0 + Duration::from_millis(5 + ms);
///     receiver.pump_inbound(now);
///     if let Some(f) = remote.receiver.try_recv() {
///         got = Some(f);
///         break;
///     }
///     while let Some(wire) = receiver.poll_datagram(now) {
///         sender.handle_datagram(&wire, now);
///     }
/// }
/// let frame = got.expect("receiver should emit the keyframe");
/// assert_eq!(frame.payload.as_ref(), &[7u8; 100]);
/// assert!(frame.keyframe);
/// ```
pub struct Session {
    config: QrtConfig,
    tracks: HashMap<u8, Track>,
    transport_seqs: TransportSeqAssigner,
    fec_rx: FecReceiver,
    pacer: Pacer,
    history: PacketHistory,
    feedback_tx: FeedbackAdapter,
    arrival_rx: ArrivalRecorder,
    bwe: BandwidthEstimator,
    reasm: FrameReassembler,
    rtt: Duration,
    last_probe_at: Option<Instant>,
    last_rate_notify: Option<RateUpdate>,
    /// Wakes the [`crate::Qrt`] I/O task when an encoder pushes.
    wake: WakeNotify,
}

impl Session {
    /// Creates an empty session (no tracks yet).
    ///
    /// # Examples
    ///
    /// ```
    /// use qrt::session::{QrtConfig, Session};
    /// let session = Session::new(QrtConfig::default());
    /// assert_eq!(session.info().track_count, 0);
    /// assert_eq!(session.info().loss_ratio, 0.0);
    /// ```
    pub fn new(config: QrtConfig) -> Self {
        let mut history = PacketHistory::new(config.history_capacity);
        history.set_rate_limiter(RetransRateLimiter::from_target_bps(
            config.bwe.start_bitrate_bps,
            Duration::from_millis(500),
        ));

        let bwe = BandwidthEstimator::new(config.bwe.clone());
        let pacer = Pacer::new(config.pacer.clone());

        Self {
            fec_rx: FecReceiver::new(256),
            history,
            feedback_tx: FeedbackAdapter::new(config.feedback.clone()),
            arrival_rx: ArrivalRecorder::new(config.feedback.clone()),
            bwe,
            pacer,
            reasm: FrameReassembler::new(),
            transport_seqs: TransportSeqAssigner::new(),
            tracks: HashMap::new(),
            rtt: Duration::from_millis(100),
            last_probe_at: None,
            last_rate_notify: None,
            wake: Arc::new(Notify::new()),
            config,
        }
    }

    /// Notifier used by encoded-frame senders (for the [`crate::Qrt`] I/O task).
    pub(crate) fn wake_notify(&self) -> WakeNotify {
        Arc::clone(&self.wake)
    }

    /// Allocates an outbound sender for `stream_id` (before constructing the encoder).
    pub fn alloc_outbound(
        &self,
        stream_id: u8,
        kind: MediaKind,
    ) -> (EncodedFrameSender, PendingFrameQueue) {
        EncodedFrameSender::new(stream_id, kind, Arc::clone(&self.wake))
    }

    /// Registers a track after you constructed the [`Encoder`] with the sender
    /// from [`Self::alloc_outbound`].
    ///
    /// Returns the receive-side [`RemoteTrack`] (caller / [`crate::Qrt`] fires
    /// `on_track`).
    ///
    /// # Errors
    ///
    /// Returns [`QrtError::TrackExists`] if `config.stream_id` is already used.
    ///
    /// # Examples
    ///
    /// ```
    /// use qrt::{
    ///     codec::{CodecRateParams, Encoder, MediaKind},
    ///     session::{QrtConfig, Session, TrackConfig},
    /// };
    ///
    /// struct NopEnc;
    /// impl Encoder for NopEnc {
    ///     fn on_rate_params(&mut self, _: &CodecRateParams) {}
    ///     fn on_keyframe_request(&mut self) {}
    /// }
    ///
    /// let mut session = Session::new(QrtConfig::default());
    /// let (_tx, pending) = session.alloc_outbound(0, MediaKind::Video);
    /// session
    ///     .register_track(TrackConfig::video(0), Box::new(NopEnc), pending)
    ///     .unwrap();
    /// let (_tx, pending) = session.alloc_outbound(1, MediaKind::Audio);
    /// session
    ///     .register_track(TrackConfig::audio(1), Box::new(NopEnc), pending)
    ///     .unwrap();
    /// assert_eq!(session.info().track_count, 2);
    /// let (_tx, pending) = session.alloc_outbound(0, MediaKind::Video);
    /// assert!(
    ///     session
    ///         .register_track(TrackConfig::video(0), Box::new(NopEnc), pending,)
    ///         .is_err()
    /// );
    /// ```
    pub fn register_track(
        &mut self,
        config: TrackConfig,
        encoder: Box<dyn Encoder>,
        pending: PendingFrameQueue,
    ) -> Result<RemoteTrack, QrtError> {
        let stream_id = config.stream_id;
        let kind = config.kind;
        if self.tracks.contains_key(&stream_id) {
            return Err(QrtError::TrackExists { stream_id });
        }

        let (receiver, inbound, inbound_notify) = EncodedFrameReceiver::pair();
        let mut track = Track::new(
            config,
            encoder,
            pending,
            inbound,
            Arc::clone(&inbound_notify),
        );

        track.set_rtt(self.rtt);
        self.tracks.insert(stream_id, track);

        Ok(RemoteTrack {
            stream_id,
            kind,
            receiver,
        })
    }

    /// Moves jitter/NetEQ-ready frames into each track's [`EncodedFrameReceiver`] queue.
    pub fn pump_inbound(&mut self, now: Instant) {
        let ids: Vec<u8> = self.tracks.keys().copied().collect();
        for id in ids {
            while let Some(frame) = self.poll_frame(id, now) {
                let Some(track) = self.tracks.get(&id) else {
                    break;
                };

                push_inbound(&track.inbound, &track.inbound_notify, frame);
            }
        }
    }

    /// Removes a track and returns whether it existed.
    pub fn remove_track(&mut self, stream_id: u8) -> bool {
        self.reasm.clear_stream(stream_id);
        self.tracks.remove(&stream_id).is_some()
    }

    /// Returns `true` if a track with `stream_id` is registered.
    pub fn has_track(&self, stream_id: u8) -> bool {
        self.tracks.contains_key(&stream_id)
    }

    /// Stream ids of all registered tracks (unordered).
    pub fn track_ids(&self) -> impl Iterator<Item = u8> + '_ {
        self.tracks.keys().copied()
    }

    /// Congestion / queue snapshot.
    pub fn info(&self) -> QrtInfo {
        QrtInfo {
            target_bitrate_bps: self.bwe.target_bitrate_bps(),
            pacing_rate_bps: self.pacer.pacing_rate_bps(),
            loss_ratio: self.bwe.loss_ratio(),
            rtt: self.rtt,
            network: self.bwe.network_state(),
            in_flight_bytes: self.feedback_tx.in_flight_bytes(),
            queued_packets: self.queued_packet_count(),
            track_count: self.tracks.len(),
        }
    }

    /// Overrides the RTT estimate (also updates every track's NACK + history).
    pub fn set_rtt(&mut self, rtt: Duration) {
        self.rtt = rtt.max(Duration::from_millis(1));
        self.history.set_rtt(self.rtt);
        for track in self.tracks.values_mut() {
            track.set_rtt(self.rtt);
        }
    }

    /// Enqueues one encoded frame into fragmentation / FEC / pacer.
    fn enqueue_frame(&mut self, frame: EncodedFrame, now: Instant) -> Result<(), PushError> {
        if frame.payload.is_empty() {
            return Err(PushError::EmptyFrame);
        }

        let ttl = frame.ttl_ms.unwrap_or(self.config.default_ttl_ms);
        if ttl == 0 {
            return Ok(());
        }

        let (packets, fec_owned) = {
            let track = self
                .tracks
                .get_mut(&frame.stream_id)
                .ok_or(PushError::UnknownTrack {
                    stream_id: frame.stream_id,
                })?;
            if track.kind != frame.kind {
                return Err(PushError::KindMismatch {
                    stream_id: frame.stream_id,
                    track: track.kind,
                    frame: frame.kind,
                });
            }

            let frame_id = track.next_frame_id;
            track.next_frame_id = track.next_frame_id.wrapping_add(1);
            let first_seq = track.next_media_seq;

            let flags = Flags {
                retrans: false,
                audio: frame.kind == MediaKind::Audio,
                key: frame.keyframe && frame.kind == MediaKind::Video,
            };

            let packets = fragment(
                &frame.payload,
                &FragmentParams {
                    stream_id: frame.stream_id,
                    frame_id,
                    timestamp: frame.timestamp,
                    ttl_ms: ttl,
                    flags,
                    first_media_seq: first_seq,
                    first_transport_seq: 0,
                },
                &self.config.payload_limits,
            )
            .map_err(PushError::Fragment)?;

            let n = packets.len() as u16;
            track.next_media_seq = first_seq.wrapping_add(n);

            let mut fec_owned = Vec::new();
            for pkt in &packets {
                let mut wire = vec![0u8; pkt.encoded_len()];
                pkt.encode(&mut wire);
                let wire = Bytes::from(wire);
                if let Some(fec) = track.fec_gen.as_mut() {
                    let seq = pkt.header().media_seq;
                    let _ = fec.push(seq, wire);
                }
            }

            if let Some(fec) = track.fec_gen.as_mut() {
                fec_owned.extend(fec.flush());
            }

            (packets, fec_owned)
        };

        for pkt in &packets {
            self.pacer.enqueue_packet(pkt, now);
        }

        for owned in fec_owned {
            let pkt = owned.as_packet();
            self.pacer.enqueue_packet(&pkt, now);
        }

        Ok(())
    }

    /// Takes the next UDP datagram that should leave the host socket.
    ///
    /// Also drains encoder sinks and runs NACK / arrival-feedback / probe maintenance.
    pub fn poll_datagram(&mut self, now: Instant) -> Option<Bytes> {
        self.drain_pending_frames();
        self.maintain(now);
        let mut outgoing = self.pacer.poll(now)?;
        let mut wire = outgoing.wire.to_vec();
        let Some(tseq) = self.transport_seqs.stamp(&mut wire) else {
            return Some(Bytes::from(wire));
        };

        outgoing.wire = Bytes::from(wire);

        let header = Header::decode(&outgoing.wire).ok();
        let audio = header.as_ref().is_some_and(|h| h.flags.audio);
        self.feedback_tx.on_sent(tseq, now, outgoing.len(), audio);

        if let Some(h) = header {
            match h.packet_type {
                PacketType::Media if !h.flags.retrans => {
                    self.history.put_outgoing(&outgoing, now);
                }
                PacketType::Media if h.flags.retrans => {
                    self.history.mark_sent(h.stream_id, h.media_seq, now);
                }
                _ => {}
            }
        }

        Some(outgoing.wire)
    }

    /// Earliest time the pacer may emit again (`None` if idle or ready now).
    pub fn next_send_time(&self, now: Instant) -> Option<Instant> {
        self.pacer.next_send_time(now)
    }

    /// Feeds one received UDP datagram into the transport.
    pub fn handle_datagram(&mut self, datagram: &[u8], now: Instant) {
        let Ok(packet) = Packet::decode(datagram) else {
            return;
        };

        let header = packet.header().clone();
        self.arrival_rx
            .on_packet(header.transport_seq, now, datagram.len());

        match packet {
            Packet::Media { .. } => {
                self.on_media_wire(header.stream_id, header.media_seq, datagram, now, false);
            }
            Packet::Fec { .. } => {
                if let Ok(recovered) = self.fec_rx.insert_fec_packet(&packet) {
                    for r in recovered {
                        self.on_media_wire(r.stream_id, r.media_seq, &r.wire, now, true);
                    }
                }
            }
            Packet::Nack { base_seq, blp, .. } => {
                for seq in Packet::nack_missing_seqs(base_seq, blp) {
                    match self.history.get_retransmission(header.stream_id, seq, now) {
                        RetransmitOutcome::Ready(out) => self.pacer.enqueue(out),
                        RetransmitOutcome::AlreadyPending
                        | RetransmitOutcome::TooSoon
                        | RetransmitOutcome::Expired
                        | RetransmitOutcome::NotFound
                        | RetransmitOutcome::RateLimited => {}
                    }
                }
            }
            Packet::ArrivalFeedback { .. } => {
                if let Some(report) = self.feedback_tx.on_feedback_packet(&packet, now) {
                    if let Some(p) = report.packets.iter().rev().find(|p| p.received()) {
                        let sample = now.saturating_duration_since(p.send_time);
                        if sample > Duration::from_millis(1) && sample < Duration::from_secs(2) {
                            self.set_rtt(sample);
                        }
                    }

                    if let Some(update) = self.bwe.on_feedback(&report, self.rtt, now) {
                        self.apply_rate_update(update, now);
                    }
                }
            }
            Packet::KeyframeReq { stream_id, .. } => {
                if let Some(track) = self.tracks.get_mut(&stream_id) {
                    track.encoder.on_keyframe_request();
                }
            }
        }
    }

    /// Pulls the next frame ready for decode on `stream_id`.
    ///
    /// Used by [`Self::pump_inbound`]. Video uses the jitter buffer; audio uses
    /// the NetEQ decision skeleton. Receive-side stalls enqueue a
    /// [`Packet::KeyframeReq`] for the remote sender.
    pub(crate) fn poll_frame(&mut self, stream_id: u8, now: Instant) -> Option<EncodedFrame> {
        let kind = self.tracks.get(&stream_id)?.kind;
        match kind {
            MediaKind::Video => self.poll_video_frame(stream_id, now),
            MediaKind::Audio => self.poll_audio_frame(stream_id, now),
        }
    }

    /// Polls every track once and returns the first ready frame (if any).
    #[allow(dead_code)]
    pub(crate) fn poll_any_frame(&mut self, now: Instant) -> Option<EncodedFrame> {
        let ids: Vec<u8> = self.tracks.keys().copied().collect();
        for id in ids {
            if let Some(frame) = self.poll_frame(id, now) {
                return Some(frame);
            }
        }

        None
    }

    fn poll_video_frame(&mut self, stream_id: u8, now: Instant) -> Option<EncodedFrame> {
        let default_ttl_ms = self.config.default_ttl_ms;
        loop {
            let keyframe_pkt = {
                let track = self.tracks.get_mut(&stream_id)?;
                let jitter = track.video_jitter.as_mut()?;
                match jitter.poll(now, true) {
                    VideoPoll::Decode(assembled) => {
                        return Some(EncodedFrame {
                            stream_id: assembled.stream_id,
                            timestamp: assembled.timestamp,
                            kind: MediaKind::Video,
                            keyframe: assembled.flags.key,
                            payload: assembled.payload,
                            ttl_ms: None,
                        });
                    }
                    VideoPoll::DroppedLate { .. } => None,
                    VideoPoll::KeyframeReq { .. } => {
                        // Ask the *remote* sender - do not notify the local encoder.
                        Some(jitter.keyframe_packet(default_ttl_ms))
                    }
                    VideoPoll::Wait => return None,
                }
            };
            if let Some(pkt) = keyframe_pkt {
                self.pacer.enqueue_packet(&pkt, now);
            }
        }
    }

    fn poll_audio_frame(&mut self, stream_id: u8, now: Instant) -> Option<EncodedFrame> {
        let track = self.tracks.get_mut(&stream_id)?;
        let neteq = track.audio_jitter.as_mut()?;
        let tick = neteq.get_decision(now);
        let packet = tick.packet?;

        Some(EncodedFrame {
            stream_id: packet.stream_id,
            timestamp: packet.timestamp,
            kind: MediaKind::Audio,
            keyframe: false,
            payload: packet.payload,
            ttl_ms: None,
        })
    }

    fn maintain(&mut self, now: Instant) {
        if let Some(fb) = self.arrival_rx.poll(now) {
            let pkt = fb.as_packet();
            self.pacer.enqueue_packet(&pkt, now);
        }

        let default_ttl = self.config.default_ttl_ms;
        let mut keyframe_reqs = Vec::new();
        let mut nack_packets = Vec::new();

        for track in self.tracks.values_mut() {
            let nack_due = track
                .last_nack_at
                .map(|t| now.saturating_duration_since(t) >= Duration::from_millis(20))
                .unwrap_or(true);
            if !nack_due {
                continue;
            }

            track.last_nack_at = Some(now);
            let batch = track.nack.process(now);
            for pkt in batch.to_packets(track.stream_id, default_ttl) {
                nack_packets.push(pkt);
            }

            if batch.ask_keyframe {
                if let Some(jitter) = track.video_jitter.as_ref() {
                    keyframe_reqs.push(jitter.keyframe_packet(default_ttl));
                }
            }
        }

        for pkt in nack_packets {
            self.pacer.enqueue_packet(&pkt, now);
        }

        for pkt in keyframe_reqs {
            self.pacer.enqueue_packet(&pkt, now);
        }

        let probe_due = self
            .last_probe_at
            .map(|t| now.saturating_duration_since(t) >= Duration::from_millis(50))
            .unwrap_or(true);
        if probe_due {
            self.last_probe_at = Some(now);
            let in_alr = self.queued_packet_count() == 0;
            let _clusters = self.bwe.poll_probes(now, in_alr);
            if in_alr {
                let target = self.bwe.target_bitrate_bps().saturating_mul(2);
                let mut cfg = self.config.pacer.clone();
                cfg.pacing_rate_bps = target.max(self.pacer.pacing_rate_bps());
                self.pacer.set_config(cfg);
            }
        }

        self.history.cull(now);
    }

    fn on_media_wire(
        &mut self,
        stream_id: u8,
        media_seq: u16,
        wire: &[u8],
        now: Instant,
        recovered: bool,
    ) {
        let bytes = Bytes::copy_from_slice(wire);
        if !recovered {
            let more = self
                .fec_rx
                .insert_media(stream_id, media_seq, bytes.clone());
            for r in more {
                self.on_media_wire(r.stream_id, r.media_seq, &r.wire, now, true);
            }
        }

        if let Some(track) = self.tracks.get_mut(&stream_id) {
            let _ = track.nack.on_received(media_seq, now);
        }

        let Ok(packet) = Packet::decode(wire) else {
            return;
        };

        if let Ok(outcome) = self.reasm.push(&packet) {
            if let Some(assembled) = outcome.into_assembled() {
                if let Some(track) = self.tracks.get_mut(&assembled.stream_id) {
                    match track.kind {
                        MediaKind::Video => {
                            if let Some(jitter) = track.video_jitter.as_mut() {
                                jitter.push(assembled, now);
                            }
                        }
                        MediaKind::Audio => {
                            if let Some(neteq) = track.audio_jitter.as_mut() {
                                neteq.push(AudioPacket {
                                    stream_id: assembled.stream_id,
                                    timestamp: assembled.timestamp,
                                    payload: assembled.payload,
                                    arrived_at: now,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    fn apply_rate_update(&mut self, update: RateUpdate, now: Instant) {
        let queue_ms = Duration::from_millis(0);
        let pushed = send_side_pushback(
            update.target_bitrate_bps,
            queue_ms,
            self.feedback_tx.in_flight_bytes(),
            None,
        );

        let mut update = update;
        update.target_bitrate_bps = pushed;
        update.pacing_rate_bps = ((pushed as f64) * self.config.bwe.pacing_factor)
            .round()
            .max(pushed as f64) as u64;

        let mut pacer_cfg = self.config.pacer.clone();
        pacer_cfg.pacing_rate_bps = update.pacing_rate_bps;
        self.pacer.set_config(pacer_cfg);

        if let Some(lim) = self.history.rate_limiter_mut() {
            *lim = RetransRateLimiter::from_target_bps(
                update.target_bitrate_bps,
                Duration::from_millis(500),
            );
        } else {
            self.history
                .set_rate_limiter(RetransRateLimiter::from_target_bps(
                    update.target_bitrate_bps,
                    Duration::from_millis(500),
                ));
        }

        // Rough BitrateAllocator stand-in: split session target across video
        // tracks; audio tracks get a small dedicated share when present.
        let video_n = self
            .tracks
            .values()
            .filter(|t| t.kind == MediaKind::Video)
            .count()
            .max(1);
        let audio_n = self
            .tracks
            .values()
            .filter(|t| t.kind == MediaKind::Audio)
            .count();
        let audio_budget = if audio_n > 0 {
            (update.target_bitrate_bps / 10).clamp(16_000, 64_000 * audio_n as u64)
        } else {
            0
        };

        let video_budget = update.target_bitrate_bps.saturating_sub(audio_budget);
        let per_video = video_budget / video_n as u64;
        let per_audio = if audio_n > 0 {
            audio_budget / audio_n as u64
        } else {
            0
        };

        for track in self.tracks.values_mut() {
            let mut track_update = update.clone();
            track_update.target_bitrate_bps = match track.kind {
                MediaKind::Video => per_video,
                MediaKind::Audio => per_audio,
            };
            track_update.pacing_rate_bps = ((track_update.target_bitrate_bps as f64)
                * self.config.bwe.pacing_factor)
                .round() as u64;
            track
                .encoder
                .on_rate_params(&CodecRateParams::from_rate_update(&track_update));
        }

        self.last_rate_notify = Some(update);
        let _ = now;
    }

    fn drain_pending_frames(&mut self) {
        let ids: Vec<u8> = self.tracks.keys().copied().collect();
        for id in ids {
            let batch: Vec<(EncodedFrame, Instant)> = {
                let Some(track) = self.tracks.get(&id) else {
                    continue;
                };

                track.pending.lock().drain(..).collect()
            };

            for (frame, at) in batch {
                let _ = self.enqueue_frame(frame, at);
            }
        }
    }

    fn queued_packet_count(&self) -> usize {
        self.pacer.queue().len()
    }
}

impl QrtInfo {
    /// Latest encoder-facing rate params (session total target).
    pub fn codec_rate_params(&self) -> CodecRateParams {
        CodecRateParams {
            target_bitrate_bps: self.target_bitrate_bps,
            rtt: self.rtt,
            loss_ratio: self.loss_ratio,
        }
    }
}
