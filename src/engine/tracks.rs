//! Per-stream tracks and the connection send/receive paths they share.
//!
//! # Two kinds of track
//!
//! [`Track::Local`] is media **we encode**. It owns `frame_id` / `media_seq`
//! counters and optional XOR FEC. [`Track::Remote`] is media **the peer
//! encodes**. It owns the NACK list and the jitter / NetEQ buffer.
//!
//! `stream_id` is the HashMap key and also the id on the wire. One id is one
//! direction: do not put a Local and a Remote on the same id.
//!
//! # Two shared pipes
//!
//! All local tracks dump packets into one [`Egress`] (priority queue + pacer).
//! All remote tracks share one [`Ingress`] (arrival log + FEC + reassembly).
//! That is why NACK, BWE, and pacing are connection-wide, not per-camera.
//!
//! # Send path ([`super::Engine::push_frame`])
//!
//! ```text
//! EncodedFrame
//!   -> split into MTU-sized Media packets (media_seq assigned here)
//!   -> maybe XOR FEC rows (video)
//!   -> enqueue on Egress (TTL starts now)
//!   -> finish(): pacer may emit Packet events immediately
//! ```
//!
//! # Receive path ([`super::Engine::push_packet`])
//!
//! ```text
//! UDP bytes
//!   -> decode Packet
//!   -> record transport_seq arrival (every type, for BWE)
//!   -> Media: FEC + NACK + reassemble + jitter
//!   -> Fec:  repair missing Media, then same as Media
//!   -> Nack: clone from history, enqueue retransmit
//!   -> ArrivalFeedback: run BWE, maybe RateChange
//!   -> KeyframeReq: EngineEvent if we have that local track
//!   -> finish()
//! ```

use std::time::{Duration, Instant};

use bytes::Bytes;

use super::{EncodedFrame, EngineConfig, EngineError, MediaKind};
use crate::core::{
    fec::{FecGenerator, FecPacketOwned, FecProtectionParams, FecReceiver, RecoveredPacket},
    feedback::{ArrivalRecorder, FeedbackAdapter, TransportPacketsFeedback, TransportSeqAssigner},
    fragment::{FragmentParams, PayloadSizeLimits, fragment},
    history::{PacketHistory, RetransRateLimiter, RetransmitOutcome},
    jitter::{
        AudioJitterConfig, AudioNetEq, AudioPacket, VideoFrameBuffer, VideoJitterConfig, VideoPoll,
    },
    nack::{DEFAULT_PROCESS_INTERVAL, NackConfig, NackRequester},
    pacer::{Pacer, PacerConfig},
    packet::{Flags, Header, Packet, PacketType},
    reassembly::{AssembledFrame, FrameReassembler},
    send_queue::OutgoingPacket,
};

/// Configuration for one unidirectional media track.
///
/// This is not a whole connection — one [`crate::Engine`] holds many tracks. The
/// `stream_id` is unique in that engine in **either** direction (you cannot
/// register local and remote on the same id).
///
/// Use [`Self::video`] or [`Self::audio`] as a starting point, then register
/// with [`crate::Engine::add_local_track`] or [`crate::Engine::add_remote_track`].
#[derive(Debug, Clone)]
pub struct TrackConfig {
    /// Wire id on every packet of this stream. Also the HashMap key.
    pub stream_id: u8,
    /// Audio or video. A local track rejects frames of the other kind.
    pub kind: MediaKind,
    /// Send-side XOR FEC. Ignored for audio and for remote tracks.
    pub enable_fec: bool,
    /// How many Media packets one FEC row protects (send video only).
    pub fec: FecProtectionParams,
    /// How aggressively the receiver lists missing `media_seq`.
    pub nack: NackConfig,
    /// Video playout delay / stall behaviour (receive video only).
    pub video_jitter: VideoJitterConfig,
    /// Audio playout buffer (receive audio only).
    pub audio_jitter: AudioJitterConfig,
}

impl TrackConfig {
    /// Returns video defaults with XOR FEC enabled.
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

    /// Returns audio defaults with XOR FEC disabled.
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

/// One unidirectional media stream keyed by [`TrackConfig::stream_id`].
///
/// Local and Remote hold different state because send vs receive algorithms
/// are different. They are not two halves of one bidirectional object.
pub enum Track {
    /// Locally captured stream this engine sends.
    Local(LocalTrack),
    /// Peer's stream this engine receives.
    Remote(RemoteTrack),
}

impl Track {
    /// Builds a local (we encode) track. Egress starts gated (`send_ready = false`).
    pub fn local(config: TrackConfig) -> Self {
        Self::Local(LocalTrack::new(config))
    }

    /// Builds a remote (peer encodes) track with NACK + jitter / NetEQ state.
    pub fn remote(config: TrackConfig) -> Self {
        Self::Remote(RemoteTrack::new(config))
    }

    /// `Some` only for a local track. Used by `push_frame` / `set_send_ready`.
    pub fn as_local_mut(&mut self) -> Option<&mut LocalTrack> {
        match self {
            Self::Local(track) => Some(track),
            Self::Remote(_) => None,
        }
    }

    /// `Some` only for a remote track (wake-time queries do not need `&mut`).
    pub fn as_remote(&self) -> Option<&RemoteTrack> {
        match self {
            Self::Remote(track) => Some(track),
            Self::Local(_) => None,
        }
    }

    /// `Some` only for a remote track. Used by datagram / jitter / NACK paths.
    pub fn as_remote_mut(&mut self) -> Option<&mut RemoteTrack> {
        match self {
            Self::Remote(track) => Some(track),
            Self::Local(_) => None,
        }
    }

    /// True for a local track. Used to ignore peer keyframe requests we cannot satisfy.
    pub fn is_local(&self) -> bool {
        matches!(self, Self::Local(_))
    }
}

/// Locally captured stream this engine sends.
///
/// `send_ready` starts false so we do not leak media before the signaling
/// handshake says the peer is listening.
pub struct LocalTrack {
    pub kind: MediaKind,
    /// Next [`Header::frame_id`] (wraps). Independent of media_seq.
    next_frame_id: u32,
    /// Next [`Header::media_seq`] for the first fragment of a new frame.
    next_media_seq: u16,
    /// `None` for audio, or when FEC is disabled on video.
    fec_gen: Option<FecGenerator>,
    pub send_ready: bool,
}

impl LocalTrack {
    /// Video gets an XOR generator when [`TrackConfig::enable_fec`] is set.
    ///
    /// `frame_id` / `media_seq` start at 1 (0 is unused). `send_ready` starts
    /// false so media cannot leave before the host opens the gate.
    fn new(config: TrackConfig) -> Self {
        Self {
            kind: config.kind,
            next_frame_id: 1,
            next_media_seq: 1,
            fec_gen: (config.kind == MediaKind::Video && config.enable_fec)
                .then(|| FecGenerator::new(config.stream_id, config.fec)),

            // Closed until the host calls Engine::set_send_ready(true) after
            // signaling says the peer is listening on this stream_id.
            send_ready: false,
        }
    }

    /// Splits one encoded frame into MTU-sized [`Packet::Media`] fragments.
    ///
    /// `media_seq` is assigned here (receiver NACK/FEC keys off it).
    /// `transport_seq` stays 0 until [`Egress::drain`] stamps the real send
    /// number — BWE cares about *when it left*, not when we encoded.
    pub fn fragment<'a>(
        &mut self,
        frame: &'a EncodedFrame,
        ttl_ms: u16,
        limits: &PayloadSizeLimits,
    ) -> Result<Vec<Packet<'a>>, EngineError> {
        let first_seq = self.next_media_seq;
        let packets = fragment(
            &frame.payload,
            &FragmentParams {
                stream_id: frame.stream_id,
                frame_id: self.next_frame_id,
                timestamp: frame.timestamp,
                ttl_ms,
                flags: Flags {
                    retrans: false,
                    audio: frame.kind == MediaKind::Audio,
                    key: frame.keyframe && frame.kind == MediaKind::Video,
                },
                first_media_seq: first_seq,

                // Placeholder: Egress overwrites this on the wire at send time.
                first_transport_seq: 0,
            },
            limits,
        )
        .map_err(EngineError::Fragment)?;

        // Advance even if later enqueue drops packets: seq space must stay
        // contiguous or the receiver NACK/FEC windows desync.
        self.next_frame_id = self.next_frame_id.wrapping_add(1);
        self.next_media_seq = first_seq.wrapping_add(packets.len() as u16);

        Ok(packets)
    }

    /// Builds XOR repair packets for the Media fragments we just created.
    ///
    /// The generator XORs **whole UDP datagrams** (header + body). That is why
    /// we encode each Media packet to bytes first. `flush()` emits leftover
    /// rows when the protection window is not a full group.
    pub fn generate_fec(&mut self, packets: &[Packet<'_>]) -> Vec<FecPacketOwned> {
        let Some(fec) = self.fec_gen.as_mut() else {
            return Vec::new();
        };

        let mut fec_packets = Vec::new();

        for packet in packets {
            let mut wire = vec![0; packet.encoded_len()];
            packet.encode(&mut wire);
            if let Ok(mut produced) = fec.push(packet.header().media_seq, Bytes::from(wire)) {
                fec_packets.append(&mut produced);
            }
        }

        fec_packets.extend(fec.flush());

        fec_packets
    }
}

/// Peer's stream this engine receives.
///
/// Video and audio use different playout buffers (frame buffer vs NetEQ
/// skeleton). Only one of `video_jitter` / `audio_jitter` is `Some`.
pub struct RemoteTrack {
    stream_id: u8,
    kind: MediaKind,
    /// Missing `media_seq` holes we will ask the sender to retransmit.
    nack: NackRequester,
    /// Video only: holds frames until they are decodable / due for playout.
    video_jitter: Option<VideoFrameBuffer>,
    /// Audio only: NetEQ-style buffer (stretch / PLC later).
    audio_jitter: Option<AudioNetEq>,
    /// Last time we ran [`NackRequester::process`]. Throttles NACK packets.
    last_nack_at: Option<Instant>,
}

impl RemoteTrack {
    /// Picks video frame buffer or audio NetEQ from [`TrackConfig::kind`].
    ///
    /// NACK is always created. Audio has no video jitter, so a NACK overflow
    /// never turns into a keyframe packet from this constructor's state.
    fn new(config: TrackConfig) -> Self {
        let stream_id = config.stream_id;
        let (video_jitter, audio_jitter) = match config.kind {
            MediaKind::Video => (
                Some(VideoFrameBuffer::new(stream_id, config.video_jitter)),
                None,
            ),
            MediaKind::Audio => (None, Some(AudioNetEq::new(stream_id, config.audio_jitter))),
        };

        Self {
            stream_id,
            kind: config.kind,
            nack: NackRequester::new(stream_id, config.nack),
            video_jitter,
            audio_jitter,
            last_nack_at: None,
        }
    }

    /// Tells NACK how long to wait before asking again for the same hole.
    /// Copied from the session RTT (initial guess, then arrival-feedback samples).
    pub fn set_rtt(&mut self, rtt: Duration) {
        self.nack.set_rtt(rtt);
    }

    /// Records that `media_seq` arrived so NACK can close that hole.
    ///
    /// Returns a [`Packet::KeyframeReq`] when the missing-seq list overflows
    /// (too many holes to list — delta frames are hopeless; ask for an IDR).
    /// Audio tracks have no video jitter, so overflow yields `None`.
    pub(super) fn on_media_seq(
        &mut self,
        media_seq: u16,
        now: Instant,
        default_ttl_ms: u16,
    ) -> Option<Packet<'static>> {
        let nack_result = self.nack.on_received(media_seq, now);

        // Too many holes to list -> give up on delta frames, ask for a keyframe.
        if nack_result.ask_keyframe {
            self.video_jitter
                .as_ref()
                .map(|jitter| jitter.keyframe_packet(default_ttl_ms))
        } else {
            None
        }
    }

    /// Hands a complete frame to the jitter / NetEQ buffer. Decode happens
    /// later in [`Self::drain_playout`] when the buffer says it is time.
    pub(super) fn push_assembled(&mut self, assembled: AssembledFrame, now: Instant) {
        match self.kind {
            MediaKind::Video => {
                if let Some(jitter) = self.video_jitter.as_mut() {
                    jitter.push(assembled, now);
                }
            }
            MediaKind::Audio => {
                if let Some(jitter) = self.audio_jitter.as_mut() {
                    jitter.push(AudioPacket {
                        stream_id: assembled.stream_id,
                        timestamp: assembled.timestamp,
                        payload: assembled.payload,
                        arrived_at: now,
                    });
                }
            }
        }
    }

    /// Pulls frames the jitter buffer is willing to give the decoder *now*.
    ///
    /// Video may also emit a [`Packet::KeyframeReq`] when it is stuck waiting
    /// on a missing reference. Those packets go onto [`Egress`], not out as
    /// [`crate::EngineEvent::Frame`].
    pub fn drain_playout(
        &mut self,
        now: Instant,
        default_ttl_ms: u16,
    ) -> (Vec<EncodedFrame>, Vec<Packet<'static>>) {
        let mut frames = Vec::new();
        let mut controls = Vec::new();

        match self.kind {
            MediaKind::Video => {
                let Some(jitter) = self.video_jitter.as_mut() else {
                    return (frames, controls);
                };

                loop {
                    // `true` = decoder is free. The engine always is; the host
                    // owns the real decoder and just consumes Frame events.
                    match jitter.poll(now, true) {
                        VideoPoll::Decode(frame) => frames.push(EncodedFrame {
                            stream_id: frame.stream_id,
                            timestamp: frame.timestamp,
                            kind: MediaKind::Video,
                            keyframe: frame.flags.key,
                            payload: frame.payload,
                            ttl_ms: None,
                        }),
                        VideoPoll::DroppedLate { .. } => {}

                        // Buffer stalled (waiting on a missing key/ref). Ask the
                        // sender for an IDR; we emit that as a KeyframeReq packet.
                        VideoPoll::KeyframeReq { .. } => {
                            controls.push(jitter.keyframe_packet(default_ttl_ms))
                        }
                        VideoPoll::Wait => break,
                    }
                }
            }
            MediaKind::Audio => {
                let Some(jitter) = self.audio_jitter.as_mut() else {
                    return (frames, controls);
                };

                // Drain every packet NetEQ is willing to release at `now`.
                while let Some(packet) = jitter.get_decision(now).packet {
                    frames.push(EncodedFrame {
                        stream_id: packet.stream_id,
                        timestamp: packet.timestamp,
                        kind: MediaKind::Audio,
                        keyframe: false,
                        payload: packet.payload,
                        ttl_ms: None,
                    });
                }
            }
        }

        (frames, controls)
    }

    /// Emits due NACK packets (and a keyframe request if the list overflowed).
    pub fn poll_nacks(&mut self, now: Instant, default_ttl_ms: u16) -> Vec<Packet<'static>> {
        let due = self
            .last_nack_at
            .map(|at| now.saturating_duration_since(at) >= DEFAULT_PROCESS_INTERVAL)
            .unwrap_or(true);
        if !due {
            return Vec::new();
        }

        self.last_nack_at = Some(now);

        // process() decides which missing seqs are old enough (≈ RTT) to ask again.
        let batch = self.nack.process(now);
        let mut packets = batch.to_packets(self.stream_id, default_ttl_ms);
        if batch.ask_keyframe
            && let Some(jitter) = self.video_jitter.as_ref()
        {
            packets.push(jitter.keyframe_packet(default_ttl_ms));
        }

        packets
    }

    /// Earliest time this remote track needs another [`crate::Engine::tick`].
    ///
    /// Combines: next NACK process slot, next video playout deadline, and
    /// "audio is buffered so drain it immediately".
    pub fn next_wake(&self, now: Instant) -> Option<Instant> {
        let nack_at = self
            .last_nack_at
            .map(|at| at + DEFAULT_PROCESS_INTERVAL)
            .unwrap_or(now);
        let mut wake = Some(nack_at);

        if let Some(at) = self
            .video_jitter
            .as_ref()
            .and_then(|jitter| jitter.next_wake_at(now))
        {
            wake = Some(wake.map_or(at, |current| current.min(at)));
        }

        if self
            .audio_jitter
            .as_ref()
            .is_some_and(|jitter| !jitter.is_empty())
        {
            wake = Some(wake.map_or(now, |current| current.min(now)));
        }

        wake
    }
}

/// Connection-wide send path shared by every local track.
///
/// There is **one** pacer for the UDP socket. Audio, video, FEC, NACK, and
/// feedback all compete in the same priority queue (audio first, then RTX,
/// then video/FEC, then feedback).
///
/// `transport_seq` is stamped here at true send time — not when the encoder
/// produced the frame — so BWE delay measurements match the wire.
pub struct Egress {
    /// Leaky-bucket that turns queued packets into on-time UDP datagrams.
    pacer: Pacer,
    /// Connection-wide send counter stamped onto every leaving packet.
    transport_seqs: TransportSeqAssigner,
    /// First-send Media copies, so a NACK can clone and retransmit them.
    history: PacketHistory,
    /// Send-side TWCC: remembers send time per `transport_seq` for BWE.
    feedback_tx: FeedbackAdapter,
}

impl Egress {
    /// One pacer, one `transport_seq` counter, NACK history sized from config.
    pub fn new(config: &EngineConfig) -> Self {
        let rtt = config.initial_rtt.max(Duration::from_millis(1));
        let mut history = PacketHistory::new(config.history_capacity);
        history.set_rtt(rtt);

        // RTX must not eat the whole bitrate. 500 ms sliding window of the
        // starting BWE target is the initial cap; BWE replaces it later.
        history.set_rate_limiter(RetransRateLimiter::from_target_bps(
            config.bwe.start_bitrate_bps,
            Duration::from_millis(500),
        ));

        Self {
            pacer: Pacer::new(config.pacer),
            transport_seqs: TransportSeqAssigner::new(),
            history,
            feedback_tx: FeedbackAdapter::new(config.feedback.clone()),
        }
    }

    /// Encodes `packet` and puts it on the pacer. TTL countdown starts at `now`.
    pub fn enqueue_packet(&mut self, packet: &Packet<'_>, now: Instant) {
        self.pacer.enqueue_packet(packet, now);
    }

    /// Queues a packet already in wire form (NACK retransmits from history).
    pub fn enqueue_retransmit(&mut self, packet: OutgoingPacket) {
        self.pacer.enqueue(packet);
    }

    /// Looks up a first-send Media packet the peer asked for via NACK.
    ///
    /// May refuse (rate-limited, expired TTL, never sent). The host only
    /// retransmits when the outcome is [`RetransmitOutcome::Ready`].
    pub fn retransmission(
        &mut self,
        stream_id: u8,
        media_seq: u16,
        now: Instant,
    ) -> RetransmitOutcome {
        self.history.get_retransmission(stream_id, media_seq, now)
    }

    /// Pulls every packet the pacer is allowed to emit at `now`.
    ///
    /// For each one we:
    /// 1. Stamp a fresh `transport_seq` (BWE / arrival feedback key).
    /// 2. Tell send-side TWCC "this seq left at `now`" (`on_sent`).
    /// 3. Keep a first-send Media copy in history (so NACK can clone it).
    /// 4. Mark RTX Media as sent (so we do not RTX the same seq too soon).
    /// 5. Hand the finished bytes to the host as [`crate::EngineEvent::Packet`].
    pub fn drain(&mut self, now: Instant, datagrams: &mut Vec<Bytes>) {
        while let Some(mut outgoing) = self.pacer.poll(now) {
            let mut wire = outgoing.wire.to_vec();

            // Unstamped packets (malformed header) still leave — better a
            // datagram than silently dropping paced work.
            let Some(transport_seq) = self.transport_seqs.stamp(&mut wire) else {
                datagrams.push(Bytes::from(wire));
                continue;
            };

            outgoing.wire = Bytes::from(wire);

            {
                let header = Header::decode(&outgoing.wire).ok();
                let audio = header.as_ref().is_some_and(|header| header.flags.audio);
                self.feedback_tx
                    .on_sent(transport_seq, now, outgoing.len(), audio);

                if let Some(header) = header {
                    match header.packet_type {
                        PacketType::Media if !header.flags.retrans => {
                            self.history.put_outgoing(&outgoing, now);
                        }
                        PacketType::Media if header.flags.retrans => {
                            self.history
                                .mark_sent(header.stream_id, header.media_seq, now);
                        }
                        _ => {}
                    }
                }
            }

            datagrams.push(outgoing.wire);
        }
    }

    /// Next time the leaky bucket has a token (or `now` if already due).
    pub fn next_send_time(&self, now: Instant) -> Option<Instant> {
        self.pacer.next_send_time(now)
    }

    /// Installs a new leaky-bucket rate (BWE / probe). Other pacer fields stay.
    pub fn set_pacing_rate(&mut self, mut config: PacerConfig, pacing_rate_bps: u64) {
        config.pacing_rate_bps = pacing_rate_bps;
        self.pacer.set_config(config);
    }

    /// Minimum gap between two RTXs of the same `media_seq`.
    pub fn set_rtt(&mut self, rtt: Duration) {
        self.history.set_rtt(rtt);
    }

    /// Caps how many RTX bytes may leave in a sliding window (from BWE target).
    pub fn set_retrans_limiter(&mut self, limiter: RetransRateLimiter) {
        self.history.set_rate_limiter(limiter);
    }

    /// Drops history entries whose media TTL expired — never RTX a stale frame.
    pub fn cull(&mut self, now: Instant) {
        self.history.cull(now);
    }

    /// Matches a peer ArrivalFeedback against our `on_sent` log → BWE input.
    pub fn on_feedback_packet(
        &mut self,
        packet: &Packet<'_>,
        now: Instant,
    ) -> Option<TransportPacketsFeedback> {
        self.feedback_tx.on_feedback_packet(packet, now)
    }

    /// Bytes stamped `on_sent` but not yet covered by arrival feedback.
    pub fn in_flight_bytes(&self) -> usize {
        self.feedback_tx.in_flight_bytes()
    }

    /// Current leaky-bucket send rate.
    pub fn pacing_rate_bps(&self) -> u64 {
        self.pacer.pacing_rate_bps()
    }

    /// Packets waiting in the pacer (not yet datagrams).
    pub fn queued_len(&self) -> usize {
        self.pacer.queue().len()
    }

    /// Empty queue means we are application-limited (ALR); probing may help.
    pub fn queue_is_empty(&self) -> bool {
        self.pacer.queue().is_empty()
    }
}

/// Connection-wide receive path shared by every remote track.
///
/// Arrival logging is **per UDP datagram** (BWE). FEC and reassembly are
/// **per `media_seq` / frame**. A packet recovered by XOR is *not* a new
/// network arrival — do not log it again.
pub struct Ingress {
    /// Times we saw each `transport_seq`. Periodically becomes ArrivalFeedback.
    arrival_rx: ArrivalRecorder,
    /// XOR decoder: Media + Fec packets in, recovered Media wires out.
    fec_rx: FecReceiver,
    /// Stitches Media fragments that share `frame_id` back into one payload.
    reasm: FrameReassembler,
}

impl Ingress {
    /// Arrival recorder uses [`EngineConfig::feedback`]. FEC window is 256 seqs.
    pub fn new(config: &EngineConfig) -> Self {
        Self {
            arrival_rx: ArrivalRecorder::new(config.feedback.clone()),

            // 256 media_seq window is enough for typical video GOP + loss.
            fec_rx: FecReceiver::new(256),
            reasm: FrameReassembler::new(),
        }
    }

    /// Notes that *this UDP datagram* arrived. Every packet type counts:
    /// Media, FEC, NACK, and feedback all occupy the path.
    pub fn record_arrival(&mut self, transport_seq: u16, now: Instant, size_bytes: usize) {
        self.arrival_rx.on_packet(transport_seq, now, size_bytes);
    }

    /// If enough arrivals have been logged, build an ArrivalFeedback packet
    /// for the peer's BWE. `None` means "not due yet".
    pub fn poll_feedback(
        &mut self,
        now: Instant,
    ) -> Option<crate::core::feedback::ArrivalFeedbackOwned> {
        self.arrival_rx.poll(now)
    }

    /// Next time [`Self::poll_feedback`] may emit a report, or `None` if idle.
    pub fn next_feedback_at(&self, now: Instant) -> Option<Instant> {
        self.arrival_rx.next_poll_at(now)
    }

    /// Feeds a [`Packet::Fec`] into the XOR decoder. Each recovered item is a
    /// reconstructed Media datagram the receiver never saw on the wire.
    pub fn recover_fec(&mut self, packet: &Packet<'_>) -> Vec<(u8, u16, Bytes)> {
        self.fec_rx
            .insert_fec_packet(packet)
            .unwrap_or_default()
            .into_iter()
            .map(
                |RecoveredPacket {
                     stream_id,
                     media_seq,
                     wire,
                 }| (stream_id, media_seq, wire),
            )
            .collect()
    }

    /// Shows the XOR decoder a Media packet we actually received.
    ///
    /// Sender FEC was computed with `transport_seq = 0` (see [`LocalTrack::fragment`]).
    /// We zero that field here so the XOR matches. Newly recovered packets are
    /// returned and later processed as if they had arrived (minus arrival log).
    pub(super) fn note_media_for_fec(
        &mut self,
        stream_id: u8,
        media_seq: u16,
        wire: &[u8],
    ) -> Vec<(u8, u16, Bytes)> {
        let mut fec_wire = wire.to_vec();
        if let Ok(mut header) = Header::decode(&fec_wire) {
            header.transport_seq = 0;
            header.encode(&mut fec_wire);
        }

        self.fec_rx
            .insert_media(stream_id, media_seq, Bytes::from(fec_wire))
            .into_iter()
            .map(|item| (item.stream_id, item.media_seq, item.wire))
            .collect()
    }

    /// Pushes one Media fragment into the frame reassembler.
    ///
    /// `Some` when this fragment completed a frame (all `frag_count` pieces
    /// present). `None` means we are still waiting, or the packet was not Media.
    pub(super) fn reassemble(&mut self, packet: &Packet<'_>) -> Option<AssembledFrame> {
        self.reasm
            .push(packet)
            .ok()
            .and_then(|result| result.into_assembled())
    }

    /// Forgets in-flight fragments when a remote track is removed, so a
    /// reused `stream_id` cannot mix old pieces into a new stream.
    pub fn clear_stream(&mut self, stream_id: u8) {
        self.reasm.clear_stream(stream_id);
    }
}
