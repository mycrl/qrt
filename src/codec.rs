//! Application-facing codec boundary (**sans-I/O**).
//!
//! Transport owns congestion and reliability. Codec wiring:
//!
//! 1. [`crate::Qrt::add_track`] (or [`crate::Session::alloc_outbound`] +
//!    [`crate::Session::register_track`]) gives you an [`EncodedFrameSender`].
//!    Construct your [`Encoder`] with that sender and hand the instance back.
//! 2. Your encoder owns the sender and pushes bitstream via
//!    [`EncodedFrameSender::push_frame`].
//! 3. BWE / PLI call [`Encoder::on_rate_params`] /
//!    [`Encoder::on_keyframe_request`] on the boxed encoder stored in the
//!    session.
//! 4. Receive path: [`crate::QrtObserver::on_track`] delivers a
//!    [`crate::RemoteTrack`] whose [`EncodedFrameReceiver`] yields frames after
//!    jitter / NetEQ.
//!
//! # Typical app wiring
//!
//! ```text
//!   Qrt::new(transport, config, observer)
//!   add_track(TrackConfig, |sender| MyEnc { sender })
//!       ──► observer.on_track(RemoteTrack { receiver, … })
//!   capture → encode → sender.push_frame
//!   receiver.recv / try_recv → decode
//! ```

use std::{
    collections::VecDeque,
    sync::Arc,
    time::{Duration, Instant},
};

use bytes::Bytes;
use parking_lot::Mutex;
use tokio::sync::Notify;

use crate::core::{bwe::RateUpdate, fragment::FragmentError};

/// Shared queue between [`EncodedFrameSender`] and the owning session track.
pub(crate) type PendingFrameQueue = Arc<Mutex<VecDeque<(EncodedFrame, Instant)>>>;

/// Shared queue between the session and [`EncodedFrameReceiver`].
pub(crate) type InboundFrameQueue = Arc<Mutex<VecDeque<EncodedFrame>>>;

/// Wakes the [`crate::Qrt`] I/O task when an encoder pushes a frame.
pub(crate) type WakeNotify = Arc<Notify>;

/// Whether an [`EncodedFrame`] carries audio or video.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MediaKind {
    /// Voice / music — typically shorter jitter target, no keyframes.
    Audio,
    /// Camera / screen — may be key or delta; subject to
    /// [`Encoder::on_keyframe_request`].
    Video,
}

/// One encoded media frame ready for [`EncodedFrameSender::push_frame`] or
/// playout decode.
///
/// Codec-opaque: `qrt` never interprets `payload`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedFrame {
    /// Logical stream multiplex id ([`crate::core::packet::Header::stream_id`]).
    pub stream_id: u8,
    /// Capture / RTP-style timestamp in 90 kHz ticks (shared by all fragments).
    pub timestamp: u32,
    /// Audio vs video.
    pub kind: MediaKind,
    /// `true` for a video keyframe (or IDR). Ignored for audio (treat as false).
    pub keyframe: bool,
    /// Opaque codec bitstream for this frame.
    pub payload: Bytes,
    /// Optional remaining lifetime in milliseconds; `None` → session default.
    pub ttl_ms: Option<u16>,
}

impl EncodedFrame {
    /// Convenience constructor with session-default TTL.
    ///
    /// # Examples
    ///
    /// ```
    /// use bytes::Bytes;
    /// use qrt::codec::{EncodedFrame, MediaKind};
    ///
    /// let f = EncodedFrame::new(
    ///     0,
    ///     90_000,
    ///     MediaKind::Audio,
    ///     false,
    ///     Bytes::from_static(b"opus"),
    /// );
    /// assert!(f.ttl_ms.is_none());
    /// assert_eq!(f.kind, MediaKind::Audio);
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

/// Error from [`EncodedFrameSender::push_frame`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PushError {
    /// Frame payload was empty.
    EmptyFrame,
    /// Fragmentation failed under the configured MTU limits.
    Fragment(FragmentError),
    /// No track registered for this `stream_id`.
    UnknownTrack {
        /// Missing stream id.
        stream_id: u8,
    },
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

impl std::fmt::Display for PushError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyFrame => write!(f, "empty encoded frame"),
            Self::Fragment(e) => write!(f, "fragment: {e}"),
            Self::UnknownTrack { stream_id } => write!(f, "unknown track stream_id={stream_id}"),
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

impl std::error::Error for PushError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Fragment(e) => Some(e),
            _ => None,
        }
    }
}

/// Rate / network hints the transport asks the encoder to apply.
///
/// # Examples
///
/// ```
/// use std::time::Duration;
///
/// use qrt::codec::CodecRateParams;
///
/// let p = CodecRateParams {
///     target_bitrate_bps: 800_000,
///     rtt: Duration::from_millis(50),
///     loss_ratio: 0.03,
/// };
/// assert!(p.loss_ratio < 0.1);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CodecRateParams {
    /// Target encode bitrate in bits per second (from BWE, after pushback).
    pub target_bitrate_bps: u64,
    /// Latest RTT sample used by congestion control.
    pub rtt: Duration,
    /// Smoothed loss ratio in `0.0..=1.0` (for encoder FEC / resilience knobs).
    pub loss_ratio: f64,
}

impl CodecRateParams {
    /// Projects a transport [`RateUpdate`] down to encoder-facing fields.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::time::Duration;
    ///
    /// use qrt::{codec::CodecRateParams, core::bwe::RateUpdate};
    ///
    /// let update = RateUpdate {
    ///     target_bitrate_bps: 400_000,
    ///     pacing_rate_bps: 440_000,
    ///     rtt: Duration::from_millis(30),
    ///     loss_ratio: 0.01,
    ///     probe_clusters: vec![],
    /// };
    /// let params = CodecRateParams::from_rate_update(&update);
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

impl From<&RateUpdate> for CodecRateParams {
    fn from(update: &RateUpdate) -> Self {
        Self::from_rate_update(update)
    }
}

/// Application encoder stored on each track for BWE / PLI callbacks.
///
/// Object-safe: construct the concrete type yourself with the
/// [`EncodedFrameSender`] from [`crate::Qrt::add_track`] or
/// [`crate::Session::alloc_outbound`], then pass the instance into the session
/// (`Box<dyn Encoder>`).
///
/// # Examples
///
/// ```
/// use std::time::Duration;
///
/// use qrt::codec::{CodecRateParams, Encoder};
///
/// struct MockEnc;
///
/// impl Encoder for MockEnc {
///     fn on_rate_params(&mut self, _params: &CodecRateParams) {}
/// }
///
/// let mut enc: Box<dyn Encoder> = Box::new(MockEnc);
/// enc.on_rate_params(&CodecRateParams {
///     target_bitrate_bps: 400_000,
///     rtt: Duration::from_millis(40),
///     loss_ratio: 0.0,
/// });
/// enc.on_keyframe_request();
/// ```
pub trait Encoder: Send + 'static {
    /// Apply the latest bitrate / RTT / loss targets.
    fn on_rate_params(&mut self, params: &CodecRateParams);

    /// Receiver (or remote PLI) asked for a video keyframe.
    ///
    /// Default is a no-op so audio encoders need not override it.
    fn on_keyframe_request(&mut self) {}
}

/// Bridge so transport-level [`crate::core::bwe::RateObserver`] call sites can
/// wrap an [`Encoder`].
#[derive(Debug, Default)]
pub struct EncoderRateObserver<E> {
    /// Inner encoder.
    pub encoder: E,
}

impl<E> EncoderRateObserver<E> {
    /// Wraps `encoder` for use as a [`crate::core::bwe::RateObserver`].
    pub fn new(encoder: E) -> Self {
        Self { encoder }
    }

    /// Returns the inner encoder.
    pub fn into_inner(self) -> E {
        self.encoder
    }
}

impl<E: Encoder> crate::core::bwe::RateObserver for EncoderRateObserver<E> {
    fn on_target_bitrate(&mut self, update: &RateUpdate) {
        self.encoder
            .on_rate_params(&CodecRateParams::from_rate_update(update));
    }
}

/// Send-side sink given to the encoder at construction.
///
/// Own this value inside your [`Encoder`] and call [`Self::push_frame`] whenever
/// you have bitstream to send.
pub struct EncodedFrameSender {
    queue: PendingFrameQueue,
    wake: WakeNotify,
    stream_id: u8,
    kind: MediaKind,
}

impl EncodedFrameSender {
    /// Creates a sender + the shared queue retained by the track.
    pub(crate) fn new(
        stream_id: u8,
        kind: MediaKind,
        wake: WakeNotify,
    ) -> (Self, PendingFrameQueue) {
        let queue = Arc::new(Mutex::new(VecDeque::new()));
        (
            Self {
                queue: Arc::clone(&queue),
                wake,
                stream_id,
                kind,
            },
            queue,
        )
    }

    /// Queues `frame` into the track pipeline (fragment / FEC / pacer).
    ///
    /// # Errors
    ///
    /// Returns [`PushError`] when the frame is rejected (empty, kind mismatch, …).
    pub fn push_frame(&mut self, mut frame: EncodedFrame, now: Instant) -> Result<(), PushError> {
        if frame.payload.is_empty() {
            return Err(PushError::EmptyFrame);
        }

        frame.stream_id = self.stream_id;

        if frame.kind != self.kind {
            return Err(PushError::KindMismatch {
                stream_id: self.stream_id,
                track: self.kind,
                frame: frame.kind,
            });
        }

        self.queue.lock().push_back((frame, now));

        self.wake.notify_one();

        Ok(())
    }

    /// Stream id this sender is bound to.
    pub fn stream_id(&self) -> u8 {
        self.stream_id
    }

    /// Media kind this sender is bound to.
    pub fn kind(&self) -> MediaKind {
        self.kind
    }
}

/// Receive-side handle living on a [`crate::RemoteTrack`].
///
/// The session pushes reassembled frames here after jitter / NetEQ; the app
/// pulls via [`Self::try_recv`] or [`Self::recv`].
pub struct EncodedFrameReceiver {
    queue: InboundFrameQueue,
    notify: Arc<Notify>,
}

impl EncodedFrameReceiver {
    pub(crate) fn new(queue: InboundFrameQueue, notify: Arc<Notify>) -> Self {
        Self { queue, notify }
    }

    pub(crate) fn pair() -> (Self, InboundFrameQueue, Arc<Notify>) {
        let queue = Arc::new(Mutex::new(VecDeque::new()));
        let notify = Arc::new(Notify::new());
        (
            Self::new(Arc::clone(&queue), Arc::clone(&notify)),
            queue,
            notify,
        )
    }

    /// Non-blocking poll for the next ready frame.
    pub fn try_recv(&mut self) -> Option<EncodedFrame> {
        self.queue.lock().pop_front()
    }

    /// Waits until a frame is available.
    pub async fn recv(&mut self) -> EncodedFrame {
        loop {
            if let Some(frame) = self.try_recv() {
                return frame;
            }
            self.notify.notified().await;
        }
    }
}

/// Pushes one inbound frame into a track queue and wakes waiters.
pub(crate) fn push_inbound(queue: &InboundFrameQueue, notify: &Arc<Notify>, frame: EncodedFrame) {
    queue.lock().push_back(frame);
    notify.notify_one();
}
