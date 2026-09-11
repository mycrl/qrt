//! Host-facing media types exchanged with [`crate::Engine`].
//!
//! The engine never inspects codec bitstream. [`EncodedFrame::payload`] is
//! opaque. [`MediaKind`] only selects audio vs video *transport* behaviour
//! (FEC, jitter, NACK overflow → keyframe).
//!
//! Push encoded frames in with [`crate::Engine::push_frame`]. Pull decoded-ready
//! frames out as [`crate::EngineEvent::Frame`]. Apply [`RateParams`] from
//! [`crate::EngineEvent::RateChange`] to your encoder.

use std::time::Duration;

use bytes::Bytes;

use crate::core::bwe::RateUpdate;

/// Whether an [`EncodedFrame`] carries audio or video.
///
/// Set this on [`crate::TrackConfig::kind`] and on every pushed frame. A local
/// track rejects [`EncodedFrame`] values of the other kind.
///
/// # Examples
///
/// ```
/// use qrt::MediaKind;
///
/// assert_ne!(MediaKind::Audio, MediaKind::Video);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MediaKind {
    /// Voice / music — typically shorter jitter target, no keyframes.
    Audio,
    /// Camera / screen — may be key or delta; subject to
    /// [`crate::EngineEvent::KeyframeRequest`].
    Video,
}

/// One encoded media frame for [`crate::Engine::push_frame`] or playout decode.
///
/// Codec-opaque: the engine never interprets `payload`. Fragmentation, FEC,
/// NACK, and jitter only look at the fields below.
///
/// # Notes
///
/// `ttl_ms` is remaining lifetime, not a wall clock. `None` uses
/// [`crate::EngineConfig::default_ttl_ms`]. A zero TTL is dropped without
/// sending.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedFrame {
    /// Logical stream multiplex id ([`crate::core::packet::Header::stream_id`]).
    pub stream_id: u8,
    /// Capture / RTP-style timestamp in 90 kHz ticks (shared by all fragments).
    pub timestamp: u32,
    /// Audio vs video. Must match the registered track.
    pub kind: MediaKind,
    /// `true` for a video keyframe (or IDR). Ignored for audio (treat as false).
    pub keyframe: bool,
    /// Opaque codec bitstream for this frame.
    pub payload: Bytes,
    /// Optional remaining lifetime in milliseconds; `None` → engine default.
    pub ttl_ms: Option<u16>,
}

impl EncodedFrame {
    /// Convenience constructor with the engine-default TTL (`ttl_ms = None`).
    ///
    /// # Examples
    ///
    /// ```
    /// use bytes::Bytes;
    /// use qrt::{EncodedFrame, MediaKind};
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

    /// Returns `true` when this is video marked as a keyframe.
    pub fn is_video_keyframe(&self) -> bool {
        self.kind == MediaKind::Video && self.keyframe
    }
}

/// Rate / network hints the engine asks the host encoder to apply.
///
/// Produced by [`crate::EngineEvent::RateChange`] after BWE runs. Apply
/// `target_bitrate_bps` to the encoder; do not follow probe pacing bursts.
///
/// # Examples
///
/// ```
/// use std::time::Duration;
///
/// use qrt::RateParams;
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
    /// Target encode bitrate in bits per second (from BWE, after pushback).
    pub target_bitrate_bps: u64,
    /// Latest RTT sample used by congestion control.
    pub rtt: Duration,
    /// Smoothed loss ratio in `0.0..=1.0` (for encoder FEC / resilience knobs).
    pub loss_ratio: f64,
}

impl RateParams {
    /// Projects a transport [`RateUpdate`] down to encoder-facing fields.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::time::Duration;
    ///
    /// use qrt::{RateParams, core::bwe::RateUpdate};
    ///
    /// let update = RateUpdate {
    ///     target_bitrate_bps: 400_000,
    ///     pacing_rate_bps: 440_000,
    ///     rtt: Duration::from_millis(30),
    ///     loss_ratio: 0.01,
    ///     probe_clusters: vec![],
    /// };
    /// let params = RateParams::from_rate_update(&update);
    /// assert_eq!(params.target_bitrate_bps, 400_000);
    /// assert_eq!(params.rtt, Duration::from_millis(30));
    /// ```
    pub fn from_rate_update(update: &RateUpdate) -> Self {
        Self {
            target_bitrate_bps: update.target_bitrate_bps,
            rtt: update.rtt,
            loss_ratio: update.loss_ratio,
        }
    }
}

impl From<&RateUpdate> for RateParams {
    fn from(update: &RateUpdate) -> Self {
        Self::from_rate_update(update)
    }
}
