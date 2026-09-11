//! Host-facing events and status from [`super::Engine`].
//!
//! [`EngineEvent`] is what the host must act on after a call. There are only
//! four kinds:
//!
//! - [`EngineEvent::Packet`] — put these bytes on the UDP socket.
//! - [`EngineEvent::Frame`] — a complete encoded frame, ready to decode.
//! - [`EngineEvent::RateChange`] — BWE wants the encoder at a new bitrate.
//! - [`EngineEvent::KeyframeRequest`] — the peer (or our jitter buffer) needs
//!   a video IDR; tell the encoder.
//!
//! [`EngineInfo`] is a snapshot you poll with [`super::Engine::info`]; it is
//! not produced by `tick`.

use std::time::Duration;

use bytes::Bytes;

use super::{EncodedFrame, RateParams};
use crate::core::bwe::NetworkState;

/// One host-facing event from the media core.
///
/// # Examples
///
/// A peer keyframe request becomes an encoder-facing event:
///
/// ```
/// use std::time::Instant;
///
/// use qrt::{
///     Engine,
///     EngineConfig,
///     EngineEvent,
///     TrackConfig,
///     core::packet::{Flags, Header, Packet, PacketType},
/// };
///
/// let mut engine = Engine::new(EngineConfig::default());
/// engine.add_local_track(TrackConfig::video(3)).unwrap();
/// let packet = Packet::KeyframeReq {
///     header: Header {
///         packet_type: PacketType::KeyframeReq,
///         flags: Flags::default(),
///         stream_id: 3,
///         media_seq: 0,
///         transport_seq: 1,
///         frame_id: 0,
///         frag_index: 0,
///         frag_count: 1,
///         timestamp: 0,
///         ttl_ms: 100,
///     },
///     stream_id: 3,
/// };
/// let mut wire = vec![0; packet.encoded_len()];
/// packet.encode(&mut wire);
///
/// let out = engine.push_packet(&wire, Instant::now());
/// assert!(
///     out.events
///         .iter()
///         .any(|event| { matches!(event, EngineEvent::KeyframeRequest { stream_id: 3 }) })
/// );
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum EngineEvent {
    /// UDP payload to put on the peer socket **now**.
    ///
    /// Send in event order. These bytes already include the stamped
    /// `transport_seq`.
    Packet(Bytes),
    /// Complete encoded frame ready for your decoder / playout.
    ///
    /// Fragments, FEC repair, NACK, and jitter delay have already happened.
    Frame(EncodedFrame),
    /// Congestion control wants this local track at a new bitrate.
    ///
    /// Apply [`RateParams`] to that encoder before the next `push_frame`.
    RateChange {
        /// Local stream identifier.
        stream_id: u8,
        /// Encoder-facing rate, RTT, and loss hints.
        params: RateParams,
    },
    /// The peer (or our jitter buffer) needs a video IDR on this local track.
    ///
    /// The next encoded video frame for `stream_id` should be a keyframe.
    KeyframeRequest {
        /// Local stream identifier.
        stream_id: u8,
    },
}

/// Snapshot of congestion and queue state from [`super::Engine::info`].
///
/// Poll this when you want a dashboard; it is not an [`EngineEvent`].
#[derive(Debug, Clone, PartialEq)]
pub struct EngineInfo {
    /// Connection-wide encoder target (before audio/video split), bits/s.
    pub target_bitrate_bps: u64,
    /// Leaky-bucket send rate. Usually a bit above the encoder target.
    pub pacing_rate_bps: u64,
    /// Smoothed transport loss ratio in `0.0..=1.0`.
    pub loss_ratio: f64,
    /// Latest RTT estimate (arrival feedback, or the configured guess).
    pub rtt: Duration,
    /// Delay-based hypothesis (underuse / delay / overuse).
    pub network: NetworkState,
    /// Bytes stamped `on_sent` but not yet acked by arrival feedback.
    pub in_flight_bytes: usize,
    /// Packets waiting in the pacer (not yet Packet events).
    pub queued_packets: usize,
    /// Number of registered local and remote tracks.
    pub track_count: usize,
}

impl EngineInfo {
    /// Returns the congestion hints intended for an encoder.
    pub fn rate_params(&self) -> RateParams {
        RateParams {
            target_bitrate_bps: self.target_bitrate_bps,
            rtt: self.rtt,
            loss_ratio: self.loss_ratio,
        }
    }
}
