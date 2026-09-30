//! Transport core: packet codec, reliability, pacing, BWE, and jitter.
//!
//! Encoder/decoder wiring lives in [`crate::codec`]. Core never interprets
//! codec payload bytes — only [`packet::Packet`] headers, sequence numbers, and
//! feedback bodies.
//!
//! # One packet family
//!
//! Media and feedback share [`packet::Packet`] on the same UDP path. There is
//! no separate RTCP channel. Roles:
//!
//! | Module | Packet | Job |
//! |--------|--------|-----|
//! | [`packet`] | all types | datagram codec |
//! | [`fragment`] | [`packet::StreamPacket::Media`] | split / join frames |
//! | [`nack`] | [`packet::StreamPacket::Nack`] | ask for missing media sequences |
//! | [`feedback`] | [`packet::Payload::ArrivalFeedback`] | TWCC-style arrivals for BWE |
//! | [`jitter`] | [`packet::StreamPacket::KeyFrameRequest`] | PLI when video is stuck |
//!
//! Two sequence spaces stay orthogonal:
//!
//! - [`packet::MediaFragmentPacket::sequence`] — per-stream media identity (NACK, reassembly).
//! - [`packet::Packet::sequence`] — connection-wide, stamped at **pacer egress**,
//!   used only by arrival feedback / BWE / in-flight accounting.
//!
//! # Send path
//!
//! ```text
//! EncodedFrame
//!   → fragment::Fragment::split
//!   → pacer / send_queue  (priority + TTL drop, leaky-bucket)
//!   → stamp Packet::sequence, remember in history + feedback
//!   → host UDP send
//! ```
//!
//! [`send_queue`] classifies Audio > Retrans > Video > Feedback > Padding.
//! [`pacer`] drains that queue under the BWE pacing rate (audio is unpaced by
//! default). Overdue packets (`now >= deadline`) are dropped rather than sent
//! late. First-send media is stored in [`history`] so a later NACK can clone
//! the datagram; a retransmission keeps the media sequence and takes a new
//! transport sequence.
//!
//! # Receive path
//!
//! ```text
//! host UDP recv → Packet::from_bytes
//!   → feedback::ArrivalRecorder   (every datagram, by transport sequence)
//!   → Media  → nack::on_received → fragment::Reassembly → jitter / NetEQ
//!   → Nack   → history::get_retransmission → pacer
//!   → ArrivalFeedback → FeedbackAdapter → bwe → RateUpdate
//!   → KeyFrameRequest → EngineEvent::KeyframeRequest → host encoder
//! ```
//!
//! Video [`jitter`] can emit a throttled keyframe request when the frame
//! buffer is stalled; audio uses the NetEQ decision skeleton.
//!
//! # Congestion loop
//!
//! ```text
//! send:  FeedbackAdapter::on_sent(transport sequence)
//! recv:  ArrivalRecorder → peer ArrivalFeedback packet
//! send:  FeedbackAdapter::on_feedback → TransportPacketsFeedback
//!        → bwe (delay trend + loss + acked bitrate + probes)
//!        → RateUpdate { target, pacing, rtt, loss, probe_clusters }
//!           ├─ pacer.set pacing_rate
//!           ├─ history RetransRateLimiter (NACK must not starve media)
//!           └─ EngineEvent::RateChange → host encoder
//! ```
//!
//! [`bwe`] is the controller; [`feedback`] is only the sensor. Probes are
//! extra paced bursts so the estimate can climb; the application encoder
//! should follow `target_bitrate_bps`, not consume probe clusters itself.
//!
//! # Host loop
//!
//! Ordinary applications use [`crate::Engine`] rather than driving these
//! components separately. Send its output datagrams over one UDP socket, feed
//! inbound UDP payloads to [`crate::Engine::push_packet`], and call
//! [`crate::Engine::tick`] at the returned absolute wake time. Frames and
//! encoder-control events are returned in [`crate::TaskResult`].
//!
//! # Examples
//!
//! Round-trip one datagram (full pipelines are composed by [`crate::Engine`]):
//!
//! ```
//! use bytes::Bytes;
//! use qrt::core::packet::{
//!     MediaFragmentPacket,
//!     MediaType,
//!     Packet,
//!     Payload,
//!     Stream,
//!     StreamPacket,
//! };
//!
//! let packet = Packet {
//!     sequence: 1,
//!     timestamp: 0,
//!     payload: Payload::Stream(Stream {
//!         id: 0,
//!         idx: 1,
//!         packet: StreamPacket::Media(MediaFragmentPacket {
//!             sequence: 1,
//!             id: 1,
//!             fragment_idx: 0,
//!             fragment_count: 1,
//!             media_type: MediaType::Video,
//!             is_key_frame: false,
//!             payload: Bytes::from_static(b"codec"),
//!         }),
//!     }),
//! };
//! let wire = packet.into_bytes();
//! assert_eq!(Packet::from_bytes(wire).unwrap(), packet);
//! ```
//!
//! # Notes
//!
//! - Sequence comparisons on the wire are wrapping. Media sequences are `u32`.
//! - Security / ICE / SDP / QUIC are out of scope.

pub mod bwe;
pub mod fec;
pub mod feedback;
pub mod fragment;
pub mod history;
pub mod jitter;
pub mod nack;
pub mod pacer;
pub mod packet;
pub mod send_queue;
