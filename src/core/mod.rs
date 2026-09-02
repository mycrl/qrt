//! Transport core: packet codec, reliability, pacing, BWE, and jitter (**sans-I/O**).
//!
//! This is the UDP media pipeline. It never opens a socket or sleeps; the host
//! ([`crate::Session`] / [`crate::Qrt`]) feeds [`std::time::Instant`]s and
//! datagrams. Encoder/decoder wiring lives in [`crate::codec`]. Core never
//! interprets codec payload bytes — only [`packet::Packet`] headers, seqs, and
//! feedback bodies.
//!
//! # One packet family
//!
//! Media, XOR FEC, and RTCP-like feedback share [`packet::Packet`] on the same
//! UDP path. There is no separate RTCP channel, RTCP mux, or QUIC. Roles:
//!
//! | Module | Packet | Job |
//! |--------|--------|-----|
//! | [`packet`] | all types | 20-byte header + type-specific body |
//! | [`fragment`] / [`reassembly`] | [`packet::Packet::Media`] | split / join frames |
//! | [`fec`] | [`packet::Packet::Fec`] | XOR erasure over media datagrams |
//! | [`nack`] | [`packet::Packet::Nack`] | ask for missing `media_seq` |
//! | [`feedback`] | [`packet::Packet::ArrivalFeedback`] | TWCC-style arrivals for BWE |
//! | [`jitter`] | [`packet::Packet::KeyframeReq`] | PLI when video is stuck |
//!
//! Two sequence spaces stay orthogonal:
//!
//! - [`packet::Header::media_seq`] — per-stream media identity (NACK, reassembly, FEC).
//! - [`packet::Header::transport_seq`] — connection-wide, stamped at **pacer egress**,
//!   used only by arrival feedback / BWE / in-flight accounting.
//!
//! # Send path
//!
//! ```text
//! EncodedFrame
//!   → fragment (media_seq, frame_id, frag_index/count)
//!   → fec::FecGenerator   (optional video XOR rows)
//!   → pacer / send_queue  (priority + TTL drop, leaky-bucket)
//!   → stamp transport_seq, remember in history + feedback
//!   → host UDP send
//! ```
//!
//! [`send_queue`] classifies Audio > Retrans > Video/FEC > Feedback > Padding.
//! [`pacer`] drains that queue under the BWE pacing rate (audio is unpaced by
//! default). Overdue packets (`now >= deadline`) are dropped rather than sent
//! late. First-send media is stored in [`history`] so a later NACK can clone
//! the datagram with [`packet::Flags::retrans`]; FEC itself is never NACKed.
//!
//! # Receive path
//!
//! ```text
//! host UDP recv → packet::decode
//!   → feedback::ArrivalRecorder   (every datagram, by transport_seq)
//!   → Media  → nack::on_received → reassembly → jitter / NetEQ → EncodedFrame
//!   → Fec    → fec::FecReceiver (maybe recover Media, then same as Media)
//!   → Nack   → history::get_retransmission → pacer
//!   → ArrivalFeedback → FeedbackAdapter → bwe → RateUpdate
//!   → KeyframeReq → Encoder::on_keyframe_request
//! ```
//!
//! FEC-recovered media is fed to NACK as well, so a repaired `media_seq` is
//! not requested again. Video [`jitter`] can emit a throttled KeyframeReq when
//! the frame buffer is stalled; audio uses the NetEQ decision skeleton.
//!
//! # Congestion loop
//!
//! ```text
//! send:  FeedbackAdapter::on_sent(transport_seq)
//! recv:  ArrivalRecorder → peer ArrivalFeedback packet
//! send:  FeedbackAdapter::on_feedback → TransportPacketsFeedback
//!        → bwe (delay trend + loss + acked bitrate + probes)
//!        → RateUpdate { target, pacing, rtt, loss, probe_clusters }
//!           ├─ pacer.set pacing_rate
//!           ├─ history RetransRateLimiter (NACK must not starve media)
//!           └─ Encoder::on_rate_params (after send_side_pushback)
//! ```
//!
//! [`bwe`] is the controller; [`feedback`] is only the sensor. Probes are
//! extra paced bursts so the estimate can climb; the application encoder
//! should follow `target_bitrate_bps`, not consume probe clusters itself.
//!
//! # Host loop
//!
//! Drive the state machines from a socket loop (this is what [`crate::Qrt`]
//! does internally):
//!
//! ```text
//! loop {
//!   session.pump_inbound(now)          // jitter → EncodedFrameReceiver
//!   while let Some(wire) = session.poll_datagram(now) { udp.send(wire) }
//!   // wait: min(pacer.next_send_time, encoder wake, recv)
//!   session.handle_datagram(recv, now)
//! }
//! ```
//!
//! [`crate::Session::poll_datagram`] also drains pending encoded frames and
//! runs NACK / arrival-feedback / probe maintenance. Prefer that façade
//! unless you are assembling a custom loop from these modules.
//!
//! # Examples
//!
//! Round-trip the shared header (full pipelines live in [`packet`], [`pacer`],
//! and [`crate::session::Session`]):
//!
//! ```
//! use qrt::core::packet::{Flags, HEADER_SIZE, Header, Packet, PacketType};
//!
//! let pkt = Packet::Media {
//!     header: Header {
//!         packet_type: PacketType::Media,
//!         flags: Flags::default(),
//!         stream_id: 0,
//!         media_seq: 1,
//!         transport_seq: 1,
//!         frame_id: 1,
//!         frag_index: 0,
//!         frag_count: 1,
//!         timestamp: 0,
//!         ttl_ms: 100,
//!     },
//!     payload: b"codec",
//! };
//! let mut wire = [0u8; HEADER_SIZE + 5];
//! pkt.encode(&mut wire);
//! assert_eq!(Packet::decode(&wire).unwrap(), pkt);
//! ```
//!
//! # Notes
//!
//! - [`packet::Header::ttl_ms`] is remaining lifetime, not a wall-clock
//!   deadline. Peers do not need synchronized clocks.
//! - Sequence comparisons must use [`packet::Header::seq_ahead`] (wrapping).
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
pub mod reassembly;
pub mod send_queue;
