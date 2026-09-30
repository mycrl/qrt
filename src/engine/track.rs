//! One unidirectional media stream.
//!
//! [`Track::Local`] is media this engine encodes: it assigns frame ids and
//! media sequences, then hands packets to the shared path. [`Track::Remote`]
//! is media the peer encodes: it tracks missing sequences, reassembles
//! fragments, and holds them until playout.
//!
//! `stream_id` is one direction. A local track and a remote track must not
//! share an id, or their media sequence spaces collide.
//!
//! Reliability here is NACK plus the jitter buffer. XOR FEC stays in
//! [`crate::core::fec`] and is not applied by the engine, because [`Packet`]
//! has no FEC payload.

use std::time::{Duration, Instant};

use super::{EncodedFrame, EngineError, MediaKind};
use crate::core::{
    fragment::{Fragment, MediaPacket, Reassembly},
    jitter::{
        AudioDecision,
        AudioJitterConfig,
        AudioNetEq,
        AudioPacket,
        VideoFrameBuffer,
        VideoJitterConfig,
        VideoPoll,
    },
    nack::{DEFAULT_PROCESS_INTERVAL, NackConfig, NackRequester},
    packet::{MediaFragmentPacket, MediaType, Packet, Payload, Stream, StreamPacket},
};

/// Configuration for one unidirectional media track.
///
/// One [`super::Engine`] holds many of these. `stream_id` is unique in that
/// engine in either direction.
///
/// Start from [`Self::video`] or [`Self::audio`], then register with
/// [`super::Engine::add_local_track`] or [`super::Engine::add_remote_track`].
#[derive(Debug, Clone)]
pub struct TrackConfig {
    /// Wire id on every packet of this stream. Also the engine's map key.
    pub stream_id: u8,
    /// Audio or video. A local track rejects frames of the other kind.
    pub kind: MediaKind,
    /// How aggressively a remote track lists missing media sequences.
    pub nack: NackConfig,
    /// Video playout delay and stall behaviour. Used by remote video tracks.
    pub video_jitter: VideoJitterConfig,
    /// Audio playout buffer. Used by remote audio tracks.
    pub audio_jitter: AudioJitterConfig,
}

impl TrackConfig {
    /// Video defaults: NACK on, video jitter buffer active once the track is remote.
    ///
    /// # Examples
    ///
    /// ```
    /// use qrt::engine::{MediaKind, TrackConfig};
    ///
    /// let config = TrackConfig::video(7);
    /// assert_eq!(config.stream_id, 7);
    /// assert_eq!(config.kind, MediaKind::Video);
    /// ```
    pub fn video(stream_id: u8) -> Self {
        Self {
            stream_id,
            kind: MediaKind::Video,
            nack: NackConfig::default(),
            video_jitter: VideoJitterConfig::default(),
            audio_jitter: AudioJitterConfig::default(),
        }
    }

    /// Audio defaults: NACK on, NetEQ buffer active once the track is remote.
    ///
    /// # Examples
    ///
    /// ```
    /// use qrt::engine::{MediaKind, TrackConfig};
    ///
    /// let config = TrackConfig::audio(1);
    /// assert_eq!(config.kind, MediaKind::Audio);
    /// ```
    pub fn audio(stream_id: u8) -> Self {
        Self {
            stream_id,
            kind: MediaKind::Audio,
            nack: NackConfig::default(),
            video_jitter: VideoJitterConfig::default(),
            audio_jitter: AudioJitterConfig::default(),
        }
    }
}

/// One unidirectional media stream keyed by [`TrackConfig::stream_id`].
///
/// Local and remote tracks are not two halves of one bidirectional object.
/// Send and receive keep different state, so they are different variants.
pub enum Track {
    /// Media this engine sends.
    Local(LocalTrack),
    /// Media this engine receives.
    ///
    /// Boxed so the send variant does not pay for the jitter buffers.
    Remote(Box<RemoteTrack>),
}

impl Track {
    /// Builds a local track. Egress starts closed (`send_ready` is false).
    pub fn local(config: TrackConfig) -> Self {
        Self::Local(LocalTrack::new(config))
    }

    /// Builds a remote track with NACK, reassembly, and a jitter buffer.
    pub fn remote(config: TrackConfig) -> Self {
        Self::Remote(Box::new(RemoteTrack::new(config)))
    }

    /// `Some` when this track sends. `push_frame` and `set_send_ready` use it.
    pub fn as_local_mut(&mut self) -> Option<&mut LocalTrack> {
        match self {
            Self::Local(track) => Some(track),
            Self::Remote(_) => None,
        }
    }

    /// `Some` when this track receives. Wake-time queries only need a shared borrow.
    pub fn as_remote(&self) -> Option<&RemoteTrack> {
        match self {
            Self::Remote(track) => Some(track.as_ref()),
            Self::Local(_) => None,
        }
    }

    /// `Some` when this track receives. Datagram, jitter, and NACK paths use it.
    pub fn as_remote_mut(&mut self) -> Option<&mut RemoteTrack> {
        match self {
            Self::Remote(track) => Some(track.as_mut()),
            Self::Local(_) => None,
        }
    }

    /// Whether this track encodes. Receive-only engines never probe.
    pub fn is_local(&self) -> bool {
        matches!(self, Self::Local(_))
    }
}

/// Media this engine sends.
///
/// `send_ready` starts false so frames cannot leave before signaling says the
/// peer is listening on this `stream_id`.
pub struct LocalTrack {
    /// Audio or video. [`super::Engine::push_frame`] rejects the other kind.
    pub kind: MediaKind,
    /// Next frame id. Wraps. Independent of the media sequence.
    next_frame_id: u32,
    /// Media sequence of the next fragment.
    next_media_seq: u32,
    /// When false, [`super::Engine::push_frame`] returns [`EngineError::NotReady`].
    pub send_ready: bool,
}

impl LocalTrack {
    /// Frame id and media sequence start at 1. Zero is left unused so a fresh
    /// track is distinguishable from an unset counter.
    fn new(config: TrackConfig) -> Self {
        Self {
            kind: config.kind,
            next_frame_id: 1,
            next_media_seq: 1,
            send_ready: false,
        }
    }

    /// Splits one encoded frame into MTU-sized media fragments.
    ///
    /// The media sequence is assigned here. Receivers NACK that number.
    /// [`Packet::sequence`] stays 0: the shared path stamps the transport
    /// sequence when the datagram actually leaves, which is the instant BWE
    /// measures.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::Fragment`] when the MTU is zero or the frame
    /// would need more than [`crate::core::fragment::MAX_FRAGMENTS_PER_FRAME`] pieces.
    pub fn fragment(
        &mut self,
        frame: &EncodedFrame,
        max_packet_size: usize,
    ) -> Result<Vec<Packet>, EngineError> {
        let frame_id = self.next_frame_id;
        let parts = Fragment { max_packet_size }.split(MediaPacket {
            id: frame_id,
            media_type: match frame.kind {
                MediaKind::Audio => MediaType::Audio,
                MediaKind::Video => MediaType::Video,
            },
            is_key_frame: frame.keyframe && frame.kind == MediaKind::Video,
            payload: frame.payload.clone(),
        });

        if parts.is_empty() {
            return Err(EngineError::Fragment);
        }

        let mut packets = Vec::with_capacity(parts.len());
        for mut part in parts {
            part.sequence = self.next_media_seq;
            self.next_media_seq = self.next_media_seq.wrapping_add(1);
            packets.push(Packet {
                // Transport sequence is stamped at pacer egress, not here.
                sequence: 0,
                timestamp: frame.timestamp,
                payload: Payload::Stream(Stream {
                    id: frame.stream_id,
                    idx: frame_id,
                    packet: StreamPacket::Media(part),
                }),
            });
        }

        // Advance even if a later TTL drop discards the packets. A gap in
        // frame ids makes the receiver treat the next frame as a new one
        // while still waiting on this id.
        self.next_frame_id = self.next_frame_id.wrapping_add(1);

        Ok(packets)
    }
}

/// Media this engine receives.
///
/// Video and audio do not share a playout buffer. Exactly one of
/// `video_jitter` and `audio_jitter` is `Some`, chosen from [`TrackConfig::kind`].
pub struct RemoteTrack {
    stream_id: u8,
    kind: MediaKind,
    /// Holes in the media sequence this track will ask the sender to fill.
    nack: NackRequester,
    /// Fragments of this stream only. The fragment header has no stream id,
    /// so each remote track owns its own reassembler.
    reassembly: Reassembly,
    video_jitter: Option<VideoFrameBuffer>,
    audio_jitter: Option<AudioNetEq>,
    /// Last [`NackRequester::process`]. NACK packets are throttled to
    /// [`DEFAULT_PROCESS_INTERVAL`].
    last_nack_at: Option<Instant>,
}

impl RemoteTrack {
    /// NACK is always on. Audio does not get a video jitter buffer, so a NACK
    /// overflow on audio cannot turn into a keyframe request.
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
            reassembly: Reassembly::new(),
            video_jitter,
            audio_jitter,
            last_nack_at: None,
        }
    }

    /// Copies the session RTT into NACK so the same hole is not requested
    /// again before a retransmission could have arrived.
    pub fn set_rtt(&mut self, rtt: Duration) {
        self.nack.set_rtt(rtt);
    }

    /// Records one media fragment.
    ///
    /// Closes that media sequence in the NACK list, and if the fragment
    /// finishes a frame, pushes the frame into the jitter buffer. Decode
    /// waits for [`Self::drain_playout`].
    ///
    /// Returns a keyframe request when the missing-sequence list overflowed:
    /// there are too many holes to list, so delta frames cannot catch up.
    /// Audio returns `None` in that case.
    pub fn on_fragment(
        &mut self,
        fragment: MediaFragmentPacket,
        timestamp: u32,
        now: Instant,
    ) -> Option<Packet> {
        let nack_result = self.nack.on_received(fragment.sequence, now);
        let keyframe = if nack_result.ask_keyframe {
            self.video_jitter
                .as_ref()
                .map(|jitter| jitter.keyframe_packet())
        } else {
            None
        };

        // `timestamp` is the transport timestamp, not a field of the
        // fragment. Jitter needs it beside the assembled payload.
        if let Some(assembled) = self.reassembly.forward(fragment) {
            match self.kind {
                MediaKind::Video => {
                    if let Some(jitter) = self.video_jitter.as_mut() {
                        jitter.push(assembled, timestamp, now);
                    }
                }
                MediaKind::Audio => {
                    if let Some(jitter) = self.audio_jitter.as_mut() {
                        jitter.push(AudioPacket {
                            stream_id: self.stream_id,
                            timestamp,
                            payload: assembled.payload,
                            arrived_at: now,
                        });
                    }
                }
            }
        }

        keyframe
    }

    /// Pulls frames the jitter buffer will give the decoder at `now`.
    ///
    /// The packet list is keyframe requests, not media. A stalled video
    /// buffer asks for an IDR this way. The `bool` is an audio conceal tick:
    /// the buffer was due and empty, so the host should run PLC. That is not
    /// an [`EncodedFrame`].
    pub fn drain_playout(&mut self, now: Instant) -> (Vec<EncodedFrame>, Vec<Packet>, bool) {
        let mut frames = Vec::new();
        let mut controls = Vec::new();

        match self.kind {
            MediaKind::Video => {
                let Some(jitter) = self.video_jitter.as_mut() else {
                    return (frames, controls, false);
                };

                loop {
                    // The engine has no decoder of its own. The host consumes
                    // Frame events, so the buffer is told the decoder is free.
                    match jitter.poll(now, true) {
                        VideoPoll::Decode { frame, timestamp } => frames.push(EncodedFrame {
                            stream_id: self.stream_id,
                            timestamp,
                            kind: MediaKind::Video,
                            keyframe: frame.is_key_frame,
                            payload: frame.payload,
                            ttl_ms: None,
                        }),
                        VideoPoll::DroppedLate { .. } => {}
                        VideoPoll::KeyframeReq { .. } => controls.push(jitter.keyframe_packet()),
                        VideoPoll::Wait => break,
                    }
                }

                (frames, controls, false)
            }
            MediaKind::Audio => {
                let Some(jitter) = self.audio_jitter.as_mut() else {
                    return (frames, controls, false);
                };

                // One tick per call. A second decision waits for `next_playout`.
                let Some(tick) = jitter.get_decision(now) else {
                    return (frames, controls, false);
                };

                if let Some(packet) = tick.packet {
                    frames.push(EncodedFrame {
                        stream_id: packet.stream_id,
                        timestamp: packet.timestamp,
                        kind: MediaKind::Audio,
                        keyframe: false,
                        payload: packet.payload,
                        ttl_ms: None,
                    });

                    return (frames, controls, false);
                }

                (frames, controls, tick.decision == AudioDecision::Expand)
            }
        }
    }

    /// NACK packets that are due, plus a keyframe request if the list overflowed.
    ///
    /// Empty when the last NACK was sent less than [`DEFAULT_PROCESS_INTERVAL`] ago.
    pub fn poll_nacks(&mut self, now: Instant) -> Vec<Packet> {
        let due = self
            .last_nack_at
            .map(|at| now.saturating_duration_since(at) >= DEFAULT_PROCESS_INTERVAL)
            .unwrap_or(true);
        if !due {
            return Vec::new();
        }

        self.last_nack_at = Some(now);

        // `process` waits about one RTT before asking for the same sequence again.
        let batch = self.nack.process(now);
        let mut packets = batch.to_packets(self.stream_id);
        if batch.ask_keyframe
            && let Some(jitter) = self.video_jitter.as_ref()
        {
            packets.push(jitter.keyframe_packet());
        }

        packets
    }

    /// Earliest time this track needs another [`super::Engine::tick`].
    ///
    /// The NACK slot, the next video playout deadline, and "audio is buffered,
    /// drain it now" are combined. The caller takes the minimum across tracks.
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

        if let Some(at) = self
            .audio_jitter
            .as_ref()
            .and_then(|jitter| jitter.next_playout_at())
        {
            wake = Some(wake.map_or(at, |current| current.min(at)));
        }

        wake
    }
}
