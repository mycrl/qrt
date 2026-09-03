//! Minimal real-time media transport over bare UDP (**QRT**).
//!
//! - [`session::Session`] — sync state machine (tracks, BWE, pacer, …).
//! - [`transport::Transport`] — host-injected datagram send/recv.
//! - [`Qrt`] — Tokio façade: I/O loop starts in [`Qrt::new`]; callbacks via
//!   [`QrtObserver`] (`on_track`, …).
//!
//! Codec: [`Qrt::add_track`] builds an [`codec::Encoder`] with an
//! [`codec::EncodedFrameSender`]. Receive frames arrive on the
//! [`session::RemoteTrack::receiver`] delivered to [`QrtObserver::on_track`].
//!
//! There is **no** separate RTCP channel — media, FEC, and feedback share
//! [`core::packet::Packet`]. No QUIC.
//!
//! # Examples
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
//!
//! let mut wire = [0u8; HEADER_SIZE + 5];
//! pkt.encode(&mut wire);
//! assert_eq!(Packet::decode(&wire).unwrap(), pkt);
//! ```

pub mod codec;
pub mod core;
pub mod session;
pub mod transport;

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use tokio::{task::JoinHandle, time::timeout};

use crate::{
    codec::{EncodedFrameSender, Encoder},
    session::{QrtConfig, QrtError, QrtInfo, RemoteTrack, Session, TrackConfig},
    transport::Transport,
};

/// Default receive buffer (Ethernet MTU-ish).
const RECV_BUF: usize = 2048;

/// Application callbacks from [`Qrt`] (WebRTC `PeerConnectionObserver` style).
///
/// Hang all host-facing notifications here — start with [`Self::on_track`];
/// more methods can grow with default no-ops later.
///
/// # Examples
///
/// ```
/// use qrt::{NullQrtObserver, QrtObserver, session::RemoteTrack};
///
/// struct App;
/// impl QrtObserver for App {
///     fn on_track(&mut self, track: RemoteTrack) {
///         assert_eq!(track.stream_id, track.stream_id);
///         let _ = track.receiver;
///     }
/// }
///
/// let _ = NullQrtObserver;
/// ```
pub trait QrtObserver: Send {
    /// A receive path is ready for `track.stream_id` (after local
    /// [`Qrt::add_track`] registers the stream).
    ///
    /// Keep [`session::RemoteTrack::receiver`] and pull frames via
    /// [`codec::EncodedFrameReceiver::try_recv`] / [`codec::EncodedFrameReceiver::recv`].
    fn on_track(&mut self, track: RemoteTrack);
}

/// No-op [`QrtObserver`] for tests and unused receive wiring.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullQrtObserver;

impl QrtObserver for NullQrtObserver {
    fn on_track(&mut self, _track: RemoteTrack) {}
}

/// Tokio-driven media session handle.
///
/// The I/O loop runs on a background task spawned by [`Self::new`]. Shared
/// control goes through an internal mutex; drop aborts the task. Host callbacks
/// go through the [`QrtObserver`] passed to [`Self::new`].
///
/// # Examples
///
/// ```
/// use std::convert::Infallible;
///
/// use qrt::{NullQrtObserver, Qrt, session::QrtConfig, transport::Transport};
///
/// struct NullTransport;
///
/// impl Transport for NullTransport {
///     type Error = Infallible;
///
///     async fn send(&mut self, _: &[u8]) -> Result<(), Self::Error> {
///         Ok(())
///     }
///
///     async fn recv(&mut self, _: &mut [u8]) -> Result<usize, Self::Error> {
///         std::future::pending().await
///     }
/// }
///
/// let rt = tokio::runtime::Builder::new_current_thread()
///     .enable_time()
///     .build()
///     .unwrap();
/// let _enter = rt.enter();
/// let _qrt = Qrt::new(NullTransport, QrtConfig::default(), NullQrtObserver);
/// ```
///
/// # Panics
///
/// [`Self::new`] panics if called outside a Tokio runtime
/// ([`tokio::spawn`](tokio::spawn)).
pub struct Qrt {
    session: Arc<Mutex<Session>>,
    observer: Arc<Mutex<Box<dyn QrtObserver>>>,
    io: Option<JoinHandle<()>>,
}

impl Qrt {
    /// Creates a session, installs `observer`, and spawns the I/O loop on the
    /// current Tokio runtime.
    ///
    /// # Panics
    ///
    /// Panics when there is no Tokio runtime context for [`tokio::spawn`].
    pub fn new<T, O>(transport: T, config: QrtConfig, observer: O) -> Self
    where
        T: Transport + 'static,
        O: QrtObserver + 'static,
    {
        let session = Arc::new(Mutex::new(Session::new(config)));
        let observer = Arc::new(Mutex::new(Box::new(observer) as Box<dyn QrtObserver>));
        let join = tokio::spawn({
            let session = Arc::clone(&session);
            async move {
                let mut transport = transport;
                let wake = session.lock().wake_notify();
                let mut recv_buf = vec![0u8; RECV_BUF];

                loop {
                    {
                        let mut session = session.lock();
                        session.pump_inbound(Instant::now());
                    }

                    loop {
                        let wire = {
                            let mut session = session.lock();
                            session.poll_datagram(Instant::now())
                        };
                        let Some(wire) = wire else {
                            break;
                        };
                        if transport.send(wire.as_ref()).await.is_err() {
                            return;
                        }
                    }

                    let wait = {
                        let session = session.lock();
                        let now = Instant::now();
                        session
                            .next_send_time(now)
                            .map(|t| t.saturating_duration_since(now))
                            .filter(|d| !d.is_zero())
                            .unwrap_or(Duration::from_millis(20))
                    };

                    tokio::select! {
                        _ = wake.notified() => {}
                        result = timeout(wait, transport.recv(&mut recv_buf)) => {
                            match result {
                                Ok(Ok(n)) => {
                                    let now = Instant::now();
                                    session
                                        .lock()
                                        .handle_datagram(&recv_buf[..n], now);
                                }
                                Ok(Err(_)) => return,
                                Err(_) => {}
                            }
                        }
                    }
                }
            }
        });
        Self {
            session,
            observer,
            io: Some(join),
        }
    }

    /// Registers a local track: `create` receives the outbound
    /// [`codec::EncodedFrameSender`], returns your [`codec::Encoder`], then
    /// [`QrtObserver::on_track`] fires with the matching [`session::RemoteTrack`].
    ///
    /// # Errors
    ///
    /// Returns [`session::QrtError::TrackExists`] if the stream id is already registered.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::convert::Infallible;
    ///
    /// use qrt::{
    ///     NullQrtObserver,
    ///     Qrt,
    ///     codec::{CodecRateParams, EncodedFrameSender, Encoder},
    ///     session::{QrtConfig, TrackConfig},
    ///     transport::Transport,
    /// };
    ///
    /// struct NullTransport;
    /// impl Transport for NullTransport {
    ///     type Error = Infallible;
    ///     async fn send(&mut self, _: &[u8]) -> Result<(), Self::Error> {
    ///         Ok(())
    ///     }
    ///     async fn recv(&mut self, _: &mut [u8]) -> Result<usize, Self::Error> {
    ///         std::future::pending().await
    ///     }
    /// }
    ///
    /// struct NopEnc {
    ///     _sender: EncodedFrameSender,
    /// }
    /// impl Encoder for NopEnc {
    ///     fn on_rate_params(&mut self, _: &CodecRateParams) {}
    /// }
    ///
    /// let rt = tokio::runtime::Builder::new_current_thread()
    ///     .enable_time()
    ///     .build()
    ///     .unwrap();
    /// let _enter = rt.enter();
    /// let qrt = Qrt::new(NullTransport, QrtConfig::default(), NullQrtObserver);
    /// qrt.add_track(TrackConfig::video(0), |sender| NopEnc { _sender: sender })
    ///     .unwrap();
    /// assert!(qrt.has_track(0));
    /// ```
    pub fn add_track<Enc: Encoder>(
        &self,
        config: TrackConfig,
        create: impl FnOnce(EncodedFrameSender) -> Enc,
    ) -> Result<(), QrtError> {
        let (sender, pending) = {
            let session = self.session.lock();
            session.alloc_outbound(config.stream_id, config.kind)
        };

        let encoder = create(sender);

        let remote = {
            let mut session = self.session.lock();
            session.register_track(config, Box::new(encoder), pending)?
        };

        self.observer.lock().on_track(remote);

        Ok(())
    }

    /// Removes a track.
    pub fn remove_track(&self, stream_id: u8) -> bool {
        self.session.lock().remove_track(stream_id)
    }

    /// Whether `stream_id` is registered.
    pub fn has_track(&self, stream_id: u8) -> bool {
        self.session.lock().has_track(stream_id)
    }

    /// Congestion / queue snapshot.
    pub fn info(&self) -> QrtInfo {
        self.session.lock().info()
    }

    /// Overrides RTT on the inner session.
    pub fn set_rtt(&self, rtt: Duration) {
        self.session.lock().set_rtt(rtt)
    }
}

impl Drop for Qrt {
    fn drop(&mut self) {
        if let Some(handle) = self.io.take() {
            handle.abort();
        }
    }
}
