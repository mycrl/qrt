//! QRT (Quick Real-time Transport): low-latency media over bare UDP.
//!
//! A [`Session`] is one peer, and it is full duplex: it can [`Session::open`]
//! send tracks and [`Session::accept`] receive tracks at the same time. A
//! track is one direction only. Sending and receiving are two tracks, the
//! same way a QUIC connection opens and accepts unidirectional streams.
//!
//! [`Client`] and [`Server`] are the only place the TCP signal connection
//! has a direction. [`Client::connect`] returns one session.
//! [`Server::accept`] returns one session per TCP connection. After that,
//! the session API is the same on both sides.
//!
//! Stream ids are allocated by the session. The dialing side uses even ids
//! and the listening side uses odd ids, so the two sides never pick the same
//! one. That split stays inside the session.
//!
//! TCP carries open, ready, and close. [`Server::bind_tls`] wraps that TCP
//! connection in TLS, and [`Client::connect_tls`] checks the server
//! certificate. The dialer then puts one AES key in Hello. Every encoded
//! frame is sealed with that key before the engine sees it. Packet headers,
//! NACK, arrival feedback, and keyframe requests stay in the clear so the
//! engine can pace and repair them. [`Server::bind`] and [`Client::connect`]
//! leave the session in the clear.
//!
//! ```text
//! Client::connect / Server::accept
//!   TCP hello, then one Session
//!
//! session.open(kind)  → SendTrack   (this side sends)
//! session.accept()    → RecvTrack   (peer sends)
//! ```
//!
//! The session is the public API. Packet framing, pacing, and congestion
//! stay inside the crate.
//!
//! # Examples
//!
//! The server accepts one session and one video track. The client opens
//! that track and sends a keyframe:
//!
//! ```
//! use bytes::Bytes;
//! use qrt::{Client, Server, engine::MediaKind};
//!
//! # #[tokio::main]
//! # async fn main() {
//! let server = Server::bind("127.0.0.1:0".parse().unwrap(), Default::default())
//!     .await
//!     .unwrap();
//! let addr = server.local_addr().unwrap();
//! let accepted = tokio::spawn(async move {
//!     let session = server.accept().await.unwrap();
//!     let track = session.accept().await.unwrap();
//!     let frame = track.recv().await.unwrap();
//!     assert_eq!(frame.payload.as_ref(), b"frame");
//!     assert_eq!(track.kind(), MediaKind::Video);
//! });
//!
//! let session = Client::connect(addr, Default::default()).await.unwrap();
//! let track = session.open(MediaKind::Video).await.unwrap();
//! track
//!     .send(90_000, true, Bytes::from_static(b"frame"))
//!     .await
//!     .unwrap();
//! accepted.await.unwrap();
//! # }
//! ```
//!
//! # Notes
//!
//! There is no NAT traversal. The UDP peer address is the TCP peer's IP
//! plus the port from hello. That is the host that accepted the signal
//! connection, which is right for a known peer on a reachable address and
//! wrong for a machine behind a translator.
//!
//! # Logging
//!
//! Spans and events go through [`tracing`]. This crate does not install a
//! subscriber. One session is a span named `session`, carrying the UDP peer
//! and whether this side dialed. `qrt=info` is the lifecycle: listen,
//! handshake, tracks, bitrate, and keyframe requests. `qrt=debug` adds
//! frames, NACK, retransmission, and bandwidth updates. `qrt=trace` adds
//! every signal frame and every datagram.
//!
//! # Encryption
//!
//! Encryption is off unless TLS is requested. The server presents the
//! certificate in [`crypto::TlsConfig`]. The client trusts the CA passed to
//! [`Client::connect_tls`]. rustls runs the TLS handshake. The UDP key is
//! 16 random bytes in Hello, not a second key exchange. Each encoded frame
//! is then AES-128-GCM. The engine fragments and retransmits the ciphertext.
//! An 8-byte counter and a 16-byte tag travel inside the frame payload.

pub mod core;
pub mod crypto;
pub mod engine;
pub mod signal;

use std::{
    collections::{HashMap, VecDeque},
    future::pending,
    io,
    net::{Ipv4Addr, Ipv6Addr, SocketAddr},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use bytes::Bytes;
use engine::{Engine, EngineEvent, TaskResult, TrackConfig};
use tokio::{
    net::{TcpListener, TcpStream, UdpSocket},
    sync::{
        Mutex, Notify,
        mpsc::{self, UnboundedReceiver, UnboundedSender, WeakSender},
        oneshot,
    },
};
use tracing::Instrument;

use self::{
    crypto::TlsConfig,
    engine::{EncodedFrame, EngineConfig, EngineError, EngineInfo, MediaKind, RateParams},
};

/// How long [`Session::open`] waits for the peer to arm the matching remote track.
const OPEN_TIMEOUT: Duration = Duration::from_secs(5);

/// Receive buffer for one UDP datagram.
///
/// `recv` truncates what does not fit. 64 KiB is the maximum UDP payload, so a
/// legal datagram reaches the engine intact. The engine's own media
/// body is much smaller; a truncated datagram would be a silent loss.
const UDP_RECV_BYTES: usize = 65_536;

/// How many video frames may wait for [`RecvTrack::recv`].
///
/// Past this, the oldest delta is dropped. If every queued frame is a
/// keyframe, the oldest keyframe is dropped and the newest stays.
const VIDEO_RECV_CAP: usize = 8;

/// How many audio frames may wait for [`RecvTrack::recv`].
///
/// About half a second at a 10 ms frame. The oldest frame is dropped past this.
const AUDIO_RECV_CAP: usize = 50;

/// Even ids belong to the TCP dialer, odd ids to the TCP listener.
const DIALER_IDS: u16 = 0;
const LISTENER_IDS: u16 = 1;

/// Failure of a [`Client`], [`Server`], or [`Session`] call.
#[derive(Debug)]
pub enum Error {
    /// The engine refused the track or the frame.
    Engine(EngineError),
    /// The TCP or UDP socket failed.
    Io(io::Error),
    /// The signal connection is gone, or this session has been dropped.
    ///
    /// A malformed signal frame after the handshake also ends here: the task
    /// stops, and the next call observes the closed channel.
    Closed,
    /// The peer refused this stream id.
    Rejected {
        /// Stream the peer refused.
        stream_id: u8,
    },
    /// The peer did not answer [`Session::open`] within [`OPEN_TIMEOUT`].
    TimedOut {
        /// Stream that was rolled back.
        stream_id: u8,
    },
    /// This side has used every id in its half of the space (128 tracks).
    Exhausted,
    /// A signal frame during hello was the wrong size, type, or order.
    Protocol(&'static str),
    /// The ephemeral key exchange or AES-GCM failed.
    Crypto(&'static str),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Engine(err) => write!(f, "{err}"),
            Self::Io(err) => write!(f, "{err}"),
            Self::Closed => write!(f, "session closed"),
            Self::Rejected { stream_id } => {
                write!(f, "peer rejected stream_id={stream_id}")
            }
            Self::TimedOut { stream_id } => {
                write!(f, "timed out waiting for stream_id={stream_id}")
            }
            Self::Exhausted => write!(f, "no stream ids left"),
            Self::Protocol(reason) => write!(f, "signal protocol: {reason}"),
            Self::Crypto(reason) => write!(f, "media encryption: {reason}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Engine(err) => Some(err),
            Self::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Self {
        if err.kind() == io::ErrorKind::UnexpectedEof {
            Self::Closed
        } else {
            Self::Io(err)
        }
    }
}

/// Encoder feedback for one [`SendTrack`].
///
/// The media on that track still only flows out. These values come back
/// because the peer's jitter buffer and this side's bandwidth estimate have
/// something to say about what to encode next.
#[derive(Debug, Clone, PartialEq)]
pub enum SendControl {
    /// New encoder target for this track. Probe pacing is not included.
    Rate(RateParams),
    /// The peer, or this side's receive path, needs a video keyframe.
    Keyframe,
}

/// UDP socket bound to port 0, in the same family as the signal address.
fn bind_unspecified(family_of: SocketAddr) -> SocketAddr {
    match family_of {
        SocketAddr::V4(_) => SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0)),
        SocketAddr::V6(_) => SocketAddr::from((Ipv6Addr::UNSPECIFIED, 0)),
    }
}

/// Next id for one side. `next` starts at [`DIALER_IDS`] or [`LISTENER_IDS`]
/// and then steps by 2, so the other side's ids never overlap.
fn alloc_id(next: &mut u16) -> Result<u8, Error> {
    if *next > u16::from(u8::MAX) {
        return Err(Error::Exhausted);
    }

    let id = *next as u8;
    *next += 2;

    Ok(id)
}

/// Listens for signal connections. Each accepted TCP connection becomes one
/// [`Session`].
///
/// The UDP socket is created per session, not here. Hello tells the peer
/// which port that socket bound.
pub struct Server {
    tcp: TcpListener,
    config: EngineConfig,
    tls: Option<tokio_rustls::TlsAcceptor>,
}

impl Server {
    /// Binds the signal socket. `config` is cloned into every accepted session.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] when the address is unavailable.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[tokio::main]
    /// # async fn main() {
    /// use qrt::Server;
    ///
    /// let server = Server::bind("127.0.0.1:0".parse().unwrap(), Default::default())
    ///     .await
    ///     .unwrap();
    /// assert_ne!(server.local_addr().unwrap().port(), 0);
    /// # }
    /// ```
    pub async fn bind(addr: SocketAddr, config: EngineConfig) -> Result<Self, Error> {
        let tcp = TcpListener::bind(addr).await?;
        let local = tcp.local_addr()?;

        tracing::info!(%local, "listening");

        Ok(Self {
            tcp,
            config,
            tls: None,
        })
    }

    /// Binds the signal socket and encrypts every session accepted on it.
    ///
    /// `tls` is this side's certificate chain and private key. Clients are
    /// not asked for a certificate. The signal connection is TLS 1.3. The
    /// dialer sends an AES key in Hello, and every encoded frame is sealed
    /// before the engine.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] when the address is unavailable, and
    /// [`Error::Crypto`] when a PEM value does not parse.
    pub async fn bind_tls(
        addr: SocketAddr,
        config: EngineConfig,
        tls: TlsConfig,
    ) -> Result<Self, Error> {
        let tcp = TcpListener::bind(addr).await?;
        let local = tcp.local_addr()?;
        let acceptor = crypto::server_acceptor(&tls)?;

        tracing::info!(%local, encrypted = true, "listening");

        Ok(Self {
            tcp,
            config,
            tls: Some(acceptor),
        })
    }

    /// Local signal address, including the port chosen when binding to `0`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] when the socket has been closed.
    pub fn local_addr(&self) -> Result<SocketAddr, Error> {
        Ok(self.tcp.local_addr()?)
    }

    /// Waits for one peer and returns its session.
    ///
    /// Call this again for the next peer. Sessions do not share tracks.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] when the socket or the TLS handshake fails,
    /// [`Error::Closed`] when the peer hangs up, and [`Error::Protocol`] when
    /// hello is missing or the two sides disagree about encryption. A bad PEM
    /// was already refused by [`Self::bind_tls`].
    #[tracing::instrument(skip(self), err)]
    pub async fn accept(&self) -> Result<Session, Error> {
        let (tcp, _) = self.tcp.accept().await?;

        // Control frames are a few dozen bytes. Nagle would hold Ready behind
        // a timer and add a round trip to every open. Set it before the TLS
        // handshake so the handshake itself is not delayed.
        tcp.set_nodelay(true)?;

        let peer_tcp = tcp.peer_addr()?;
        let udp = UdpSocket::bind(bind_unspecified(peer_tcp)).await?;
        let local_port = udp.local_addr()?.port();
        let engine = Engine::new(self.config.clone());

        if let Some(acceptor) = &self.tls {
            let tls = acceptor.accept(tcp).await?;
            let (read, write) = tokio::io::split(tls);

            signal::complete_handshake(
                Box::new(read),
                Box::new(write),
                udp,
                peer_tcp.ip(),
                local_port,
                engine,
                LISTENER_IDS,
                true,
            )
            .await
        } else {
            let (read, write) = tcp.into_split();

            signal::complete_handshake(
                Box::new(read),
                Box::new(write),
                udp,
                peer_tcp.ip(),
                local_port,
                engine,
                LISTENER_IDS,
                false,
            )
            .await
        }
    }
}

/// Dials one peer. The only product is a [`Session`].
pub struct Client;

impl Client {
    /// Dials `addr` and completes the UDP hello.
    ///
    /// A second call opens a second TCP connection and a second session.
    /// One client value does not hold more than the session it returned.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] when the dial or the bind fails, [`Error::Closed`]
    /// when the peer hangs up during hello, and [`Error::Protocol`] when the
    /// peer's first frame is not a hello.
    #[tracing::instrument(skip(config), err)]
    pub async fn connect(addr: SocketAddr, config: EngineConfig) -> Result<Session, Error> {
        dial(addr, config, None).await
    }

    /// Dials `addr` over TLS and encrypts every frame on the session.
    ///
    /// `trust_anchor_pem` is the CA that signed the server certificate, in
    /// PEM. rustls also checks that certificate against `addr`'s IP, so the
    /// certificate must list that IP. This side generates the AES key and
    /// sends it in Hello.
    ///
    /// # Errors
    ///
    /// Same as [`Self::connect`], plus [`Error::Crypto`] when the PEM does
    /// not parse, and [`Error::Protocol`] when the peer's hello carries a key
    /// of its own.
    #[tracing::instrument(skip(config, trust_anchor_pem), err)]
    pub async fn connect_tls(
        addr: SocketAddr,
        config: EngineConfig,
        trust_anchor_pem: Vec<u8>,
    ) -> Result<Session, Error> {
        dial(addr, config, Some(trust_anchor_pem)).await
    }
}

/// Shared dial path. `trust_anchor_pem` turns on TLS and media encryption.
async fn dial(
    addr: SocketAddr,
    config: EngineConfig,
    trust_anchor_pem: Option<Vec<u8>>,
) -> Result<Session, Error> {
    let udp = UdpSocket::bind(bind_unspecified(addr)).await?;
    let local_port = udp.local_addr()?.port();
    let tcp = TcpStream::connect(addr).await?;
    tcp.set_nodelay(true)?;
    let peer_ip = tcp.peer_addr()?.ip();
    let engine = Engine::new(config);

    if let Some(trust_anchor_pem) = trust_anchor_pem {
        let connector = crypto::client_connector(&trust_anchor_pem)?;
        let tls = connector
            .connect(crypto::server_name(addr.ip()), tcp)
            .await?;
        let (read, write) = tokio::io::split(tls);

        signal::complete_handshake(
            Box::new(read),
            Box::new(write),
            udp,
            peer_ip,
            local_port,
            engine,
            DIALER_IDS,
            true,
        )
        .await
    } else {
        let (read, write) = tcp.into_split();

        signal::complete_handshake(
            Box::new(read),
            Box::new(write),
            udp,
            peer_ip,
            local_port,
            engine,
            DIALER_IDS,
            false,
        )
        .await
    }
}

/// One peer, after the TCP signal connection exists.
///
/// Full duplex: [`Self::open`] creates a track this side sends, and
/// [`Self::accept`] waits for a track the peer sends. Neither call changes
/// which side dialed the TCP connection.
///
/// Dropping the session drops its command channel. Tracks still held keep
/// the task alive until they are dropped too. There is no shutdown handshake.
pub struct Session {
    commands: mpsc::Sender<Command>,
    incoming: Mutex<UnboundedReceiver<RecvTrack>>,
}

impl Session {
    /// Allocates a send track and waits until the peer is receiving it.
    ///
    /// The id is chosen here. `kind` is the only property the peer is told;
    /// it builds the remote track from the audio or video defaults.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Exhausted`] when this side's 128 ids are used,
    /// [`Error::Rejected`] when the peer refuses the id, [`Error::TimedOut`]
    /// when no answer arrives, and [`Error::Closed`] when the session ends.
    #[tracing::instrument(skip(self), err)]
    pub async fn open(&self, kind: MediaKind) -> Result<SendTrack, Error> {
        roundtrip(&self.commands, |reply| Command::Open { kind, reply }).await
    }

    /// The next track the peer opened.
    ///
    /// The track already exists when this returns. Frames that arrived
    /// before the call are buffered on it. One accept at a time; a second
    /// waits on the same queue.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Closed`] when the session task has stopped.
    pub async fn accept(&self) -> Result<RecvTrack, Error> {
        self.incoming.lock().await.recv().await.ok_or(Error::Closed)
    }

    /// Congestion and queue snapshot for the whole UDP path, not one track.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Closed`] when the session task has stopped.
    pub async fn info(&self) -> Result<EngineInfo, Error> {
        roundtrip(&self.commands, |reply| Command::Info { reply }).await
    }
}

/// A track this session sends. Media flows out only.
///
/// [`Self::control`] is the encoder's inbox: bitrate and keyframe requests.
/// It does not carry media. Poll it, or the queue grows for as long as the
/// track is open.
///
/// Dropping the track closes it.
pub struct SendTrack {
    id: u8,
    kind: MediaKind,
    commands: mpsc::Sender<Command>,
    control: Mutex<UnboundedReceiver<SendControl>>,
}

impl SendTrack {
    /// Id allocated by [`Session::open`].
    pub fn id(&self) -> u8 {
        self.id
    }

    /// Audio or video, as passed to [`Session::open`].
    pub fn kind(&self) -> MediaKind {
        self.kind
    }

    /// Sends one frame on this track.
    ///
    /// `keyframe` is meaningful for video. Audio ignores it. The TTL is the
    /// session default.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Engine`] for an empty payload or a track that is
    /// already closed, and [`Error::Closed`] when the session ends.
    pub async fn send(&self, timestamp: u32, keyframe: bool, payload: Bytes) -> Result<(), Error> {
        let frame = EncodedFrame {
            stream_id: self.id,
            timestamp,
            kind: self.kind,
            keyframe,
            payload,
            ttl_ms: None,
        };
        roundtrip(&self.commands, |reply| Command::Frame { frame, reply }).await
    }

    /// Next bitrate update or keyframe request for this track.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Closed`] when the track or the session has ended.
    pub async fn control(&self) -> Result<SendControl, Error> {
        self.control.lock().await.recv().await.ok_or(Error::Closed)
    }

    /// Tells the peer this send track is finished.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Closed`] when the session ends first.
    pub async fn close(&self) -> Result<(), Error> {
        roundtrip(&self.commands, |reply| Command::Close {
            stream_id: self.id,
            reply,
        })
        .await
    }
}

impl Drop for SendTrack {
    fn drop(&mut self) {
        let (reply, _) = oneshot::channel();
        let _ = self.commands.try_send(Command::Close {
            stream_id: self.id,
            reply,
        });
    }
}

/// A track the peer sends. Media flows in only.
///
/// Dropping the track tells the peer to stop sending it. [`Self::recv`]
/// returns queued frames first. Audio concealment is not a frame, so
/// [`Self::recv`] waits until a real frame arrives.
pub struct RecvTrack {
    id: u8,
    kind: MediaKind,
    commands: mpsc::Sender<Command>,
    frames: Arc<FrameInbox>,
}

impl RecvTrack {
    /// Id the peer allocated when it opened this track.
    pub fn id(&self) -> u8 {
        self.id
    }

    /// Audio or video, as the peer announced it.
    pub fn kind(&self) -> MediaKind {
        self.kind
    }

    /// Next complete frame.
    ///
    /// Frames already queued are returned first. [`Error::Closed`] after that
    /// means the peer stopped the track. Video keeps at most
    /// [`VIDEO_RECV_CAP`] frames and drops the oldest delta when the peer
    /// outruns this call. Audio keeps at most [`AUDIO_RECV_CAP`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Closed`] when the track or the session has ended.
    pub async fn recv(&self) -> Result<EncodedFrame, Error> {
        loop {
            let notified = self.frames.notify.notified();
            tokio::pin!(notified);

            if let Some(frame) = self.frames.queue.lock().pop_front() {
                return Ok(frame);
            }

            if self.frames.closed.load(Ordering::Acquire) {
                return Err(Error::Closed);
            }

            notified.await;
        }
    }

    /// Frames dropped because this track's queue was full.
    pub fn dropped(&self) -> u64 {
        self.frames.dropped.load(Ordering::Relaxed)
    }

    /// Stops this receive track and tells the peer.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Closed`] when the session ends first.
    pub async fn close(&self) -> Result<(), Error> {
        roundtrip(&self.commands, |reply| Command::Close {
            stream_id: self.id,
            reply,
        })
        .await
    }
}

impl Drop for RecvTrack {
    fn drop(&mut self) {
        let (reply, _) = oneshot::channel();
        let _ = self.commands.try_send(Command::Close {
            stream_id: self.id,
            reply,
        });
    }
}

async fn roundtrip<T>(
    commands: &mpsc::Sender<Command>,
    build: impl FnOnce(oneshot::Sender<Result<T, Error>>) -> Command,
) -> Result<T, Error> {
    let (reply, response) = oneshot::channel();
    commands
        .send(build(reply))
        .await
        .map_err(|_| Error::Closed)?;

    response.await.map_err(|_| Error::Closed)?
}

enum Command {
    Open {
        kind: MediaKind,
        reply: oneshot::Sender<Result<SendTrack, Error>>,
    },
    Close {
        stream_id: u8,
        reply: oneshot::Sender<Result<(), Error>>,
    },
    Frame {
        frame: EncodedFrame,
        reply: oneshot::Sender<Result<(), Error>>,
    },
    Info {
        reply: oneshot::Sender<Result<EngineInfo, Error>>,
    },
}

/// An [`Session::open`] that has been announced and is waiting for Ready.
struct PendingOpen {
    reply: oneshot::Sender<Result<SendTrack, Error>>,
    deadline: Instant,
    kind: MediaKind,
    control_tx: UnboundedSender<SendControl>,
    control_rx: UnboundedReceiver<SendControl>,
}

/// Bounded queue of complete frames for one receive track.
///
/// The session task never waits on [`RecvTrack::recv`]. A full video queue
/// drops the oldest delta and keeps the newest keyframe. A full audio queue
/// drops the oldest frame.
struct FrameInbox {
    queue: parking_lot::Mutex<VecDeque<EncodedFrame>>,
    notify: Notify,
    dropped: AtomicU64,
    closed: AtomicBool,
    cap: usize,
    video: bool,
}

impl FrameInbox {
    fn new(video: bool) -> Self {
        Self {
            queue: parking_lot::Mutex::new(VecDeque::new()),
            notify: Notify::new(),
            dropped: AtomicU64::new(0),
            closed: AtomicBool::new(false),
            cap: if video {
                VIDEO_RECV_CAP
            } else {
                AUDIO_RECV_CAP
            },
            video,
        }
    }

    fn push(&self, frame: EncodedFrame) {
        {
            let mut queue = self.queue.lock();
            if queue.len() >= self.cap {
                if self.video {
                    if let Some(index) = queue.iter().position(|queued| !queued.keyframe) {
                        queue.remove(index);
                    } else {
                        queue.pop_front();
                    }
                } else {
                    queue.pop_front();
                }

                self.dropped.fetch_add(1, Ordering::Relaxed);
            }

            queue.push_back(frame);
        }

        self.notify.notify_one();
    }

    fn close(&self) {
        self.closed.store(true, Ordering::Release);
        self.notify.notify_waiters();
    }
}

/// Receive routes. Dropping the map unblocks every [`RecvTrack::recv`].
struct Inbound(HashMap<u8, Arc<FrameInbox>>);

impl std::ops::Deref for Inbound {
    type Target = HashMap<u8, Arc<FrameInbox>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for Inbound {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Drop for Inbound {
    fn drop(&mut self) {
        for inbox in self.0.values() {
            inbox.close();
        }
    }
}

fn spawn_session(
    engine: Engine,
    read: signal::ReadHalf,
    write: signal::WriteHalf,
    udp: UdpSocket,
    peer: SocketAddr,
    next_id: u16,
    cipher: Option<crypto::Cipher>,
) -> Session {
    let role = if next_id == DIALER_IDS {
        "dialer"
    } else {
        "listener"
    };

    // Root span. accept and connect are still entered here; a child would
    // keep that handshake on every later event.
    let span = tracing::info_span!(parent: None, "session", %peer, role);
    let (commands_tx, commands_rx) = mpsc::channel(32);
    let (incoming_tx, incoming_rx) = mpsc::unbounded_channel();
    let (signal_tx, signal_rx) = mpsc::channel(32);

    // Separate task so a `read_exact` is never cancelled by the engine select.
    // Same span as the drive task: signal traces belong to this session.
    tokio::spawn({
        let span = span.clone();
        async move {
            let mut read = read;
            loop {
                match signal::read(&mut read).await {
                    Ok(signal) => {
                        if signal_tx.send(signal).await.is_err() {
                            break;
                        }
                    }
                    Err(Error::Closed) => {
                        tracing::debug!("signal connection closed");

                        break;
                    }
                    Err(err) => {
                        tracing::warn!(%err, "signal read failed");

                        break;
                    }
                }
            }
        }
        .instrument(span)
    });

    let commands = commands_tx.downgrade();
    tokio::spawn(
        async move {
            if let Err(err) = drive(
                engine,
                write,
                udp,
                commands,
                commands_rx,
                incoming_tx,
                signal_rx,
                next_id,
                cipher,
            )
            .await
            {
                tracing::warn!(%err, "session ended");
            } else {
                tracing::debug!("session ended");
            }
        }
        .instrument(span),
    );

    Session {
        commands: commands_tx,
        incoming: Mutex::new(incoming_rx),
    }
}

/// Sends the datagrams in `result` and routes frames and encoder feedback
/// to the track that owns them.
///
/// The engine stamped transport sequence and send time before this runs.
/// Waiting on the socket here does not move that timestamp. A frame with no
/// route is dropped: the sender does not transmit until Ready, and the route
/// is installed before that Ready is written, so this is a packet that
/// arrived after the track was closed.
async fn emit(
    result: TaskResult,
    udp: &UdpSocket,
    outgoing: &HashMap<u8, UnboundedSender<SendControl>>,
    incoming: &HashMap<u8, Arc<FrameInbox>>,
    cipher: &mut Option<crypto::Cipher>,
) -> Result<Option<Instant>, Error> {
    let wake = result.next_wake;
    for event in result.events {
        match event {
            EngineEvent::Packet(bytes) => {
                udp.send(&bytes).await?;
            }
            EngineEvent::Frame(mut frame) => {
                // The engine reassembled ciphertext. The counter in the
                // payload is the nonce. A replay or a bad tag is dropped
                // here and does not enter the track queue.
                if let Some(cipher) = cipher.as_mut() {
                    match cipher.open(std::mem::take(&mut frame.payload)) {
                        Ok(payload) => frame.payload = payload,
                        Err(err) => {
                            tracing::warn!(stream_id = frame.stream_id, %err, "dropped frame");

                            continue;
                        }
                    }
                }

                tracing::debug!(
                    stream_id = frame.stream_id,
                    timestamp = frame.timestamp,
                    keyframe = frame.keyframe,
                    bytes = frame.payload.len(),
                    "frame"
                );

                if let Some(inbox) = incoming.get(&frame.stream_id) {
                    inbox.push(frame);
                }
            }
            EngineEvent::RateChange { stream_id, params } => {
                tracing::info!(
                    stream_id,
                    bps = params.target_bitrate_bps,
                    rtt = ?params.rtt,
                    loss = tracing::field::display(format_args!("{:.3}", params.loss_ratio)),
                    "rate"
                );

                if let Some(tx) = outgoing.get(&stream_id) {
                    let _ = tx.send(SendControl::Rate(params));
                }
            }
            EngineEvent::KeyframeRequest { stream_id } => {
                tracing::info!(stream_id, "keyframe requested");

                if let Some(tx) = outgoing.get(&stream_id) {
                    let _ = tx.send(SendControl::Keyframe);
                }
            }
            EngineEvent::Conceal { stream_id } => {
                tracing::debug!(stream_id, "conceal");
            }
        }
    }

    Ok(wake)
}

/// Owns the engine, the UDP socket, and the signal write half.
///
/// The read half lives on another task and arrives as `signals`. Writes
/// happen only after a `select` arm has won, so `write_all` is not cancelled
/// halfway through a frame.
#[allow(clippy::too_many_arguments)]
async fn drive(
    mut engine: Engine,
    mut write: signal::WriteHalf,
    udp: UdpSocket,
    commands_tx: WeakSender<Command>,
    mut commands: mpsc::Receiver<Command>,
    incoming_tracks: UnboundedSender<RecvTrack>,
    mut signals: mpsc::Receiver<signal::Signal>,
    mut next_id: u16,
    mut cipher: Option<crypto::Cipher>,
) -> Result<(), Error> {
    tracing::info!("session started");

    let local_parity = (next_id & 1) as u8;
    let mut next_wake: Option<Instant> = None;
    let mut opening: HashMap<u8, PendingOpen> = HashMap::new();
    let mut outgoing: HashMap<u8, UnboundedSender<SendControl>> = HashMap::new();
    let mut incoming = Inbound(HashMap::new());
    let mut datagram = vec![0u8; UDP_RECV_BYTES];

    loop {
        let open_deadline = opening.values().map(|pending| pending.deadline).min();
        let wake = match (next_wake, open_deadline) {
            (Some(engine_at), Some(open_at)) => Some(engine_at.min(open_at)),
            (engine_at, open_at) => engine_at.or(open_at),
        };

        tokio::select! {
            command = commands.recv() => {
                let Some(command) = command else {
                    tracing::debug!("command channel closed");

                    return Ok(());
                };

                match command {
                    Command::Open { kind, reply } => match alloc_id(&mut next_id) {
                        Err(err) => {
                            tracing::warn!(%err, "no stream id left");

                            let _ = reply.send(Err(err));
                        }
                        Ok(stream_id) => {
                            let config = match kind {
                                MediaKind::Video => TrackConfig::video(stream_id),
                                MediaKind::Audio => TrackConfig::audio(stream_id),
                            };
                            if let Err(err) = engine.add_local_track(config) {
                                tracing::warn!(stream_id, %err, "send track rejected locally");

                                let _ = reply.send(Err(Error::Engine(err)));
                            } else {
                                tracing::info!(stream_id, ?kind, "opening send track");

                                signal::write(&mut write, &signal::Signal::Open { stream_id, kind }).await?;
                                let (control_tx, control_rx) = mpsc::unbounded_channel();
                                opening.insert(
                                    stream_id,
                                    PendingOpen {
                                        reply,
                                        deadline: Instant::now() + OPEN_TIMEOUT,
                                        kind,
                                        control_tx,
                                        control_rx,
                                    },
                                );
                            }
                        }
                    },
                    Command::Close { stream_id, reply } => {
                        // An open still waiting for Ready is the caller's answer.
                        if let Some(pending) = opening.remove(&stream_id) {
                            let _ = pending.reply.send(Err(Error::Closed));
                        }

                        outgoing.remove(&stream_id);
                        if let Some(inbox) = incoming.remove(&stream_id) {
                            inbox.close();
                        }
                        if engine.remove_track(stream_id) {
                            tracing::info!(stream_id, "track closed");

                            signal::write(&mut write, &signal::Signal::Close(stream_id)).await?;
                            let result = engine.tick(Instant::now());
                            next_wake = emit(result, &udp, &outgoing, &incoming, &mut cipher).await?;
                            let _ = reply.send(Ok(()));
                        } else {
                            tracing::trace!(stream_id, "close ignored");

                            let _ = reply.send(Err(Error::Engine(EngineError::UnknownTrack {
                                stream_id,
                            })));
                        }
                    }
                    Command::Frame { mut frame, reply } => {
                        // The oneshot moves into the arm that answers the caller.
                        let reply = if let Some(cipher) = cipher.as_mut() {
                            match cipher.seal(std::mem::take(&mut frame.payload)) {
                                Ok(payload) => {
                                    frame.payload = payload;

                                    Some(reply)
                                }
                                Err(err) => {
                                    let _ = reply.send(Err(err));

                                    None
                                }
                            }
                        } else {
                            Some(reply)
                        };

                        if let Some(reply) = reply {
                            match engine.push_frame(frame, Instant::now()) {
                                Ok(result) => {
                                    match emit(result, &udp, &outgoing, &incoming, &mut cipher).await {
                                        Ok(wake) => {
                                            next_wake = wake;
                                            let _ = reply.send(Ok(()));
                                        }

                                        // The caller cannot receive this io error: the
                                        // oneshot carries one value, and the task is about
                                        // to stop. Closed is what the next call sees too.
                                        Err(err) => {
                                            let _ = reply.send(Err(Error::Closed));

                                            return Err(err);
                                        }
                                    }
                                }
                                Err(err) => {
                                    let _ = reply.send(Err(Error::Engine(err)));
                                }
                            }
                        }
                    }
                    Command::Info { reply } => {
                        let _ = reply.send(Ok(engine.info()));
                    }
                }
            }
            signal = signals.recv() => {
                let Some(signal) = signal else {
                    tracing::debug!("signal channel closed");

                    return Ok(());
                };

                match signal {
                    signal::Signal::Hello { .. } => return Err(Error::Protocol("hello after handshake")),
                    signal::Signal::Open { stream_id, kind } => {
                        let config = match kind {
                            MediaKind::Video => TrackConfig::video(stream_id),
                            MediaKind::Audio => TrackConfig::audio(stream_id),
                        };
                        if stream_id & 1 == local_parity || engine.add_remote_track(config).is_err() {
                            tracing::warn!(stream_id, ?kind, "rejected receive track");

                            signal::write(&mut write, &signal::Signal::Reject(stream_id)).await?;
                        } else {
                            tracing::info!(stream_id, ?kind, "receive track");

                            let inbox = Arc::new(FrameInbox::new(kind == MediaKind::Video));
                            // Install the route before Ready. The peer's first
                            // media packet is sent only after it sees Ready.
                            incoming.insert(stream_id, Arc::clone(&inbox));
                            signal::write(&mut write, &signal::Signal::Ready(stream_id)).await?;

                            // add_remote_track does not run the pacer. Tick
                            // once so this track's NACK timer is in next_wake.
                            let result = engine.tick(Instant::now());
                            next_wake = emit(result, &udp, &outgoing, &incoming, &mut cipher).await?;
                            let Some(commands) = commands_tx.upgrade() else {
                                inbox.close();

                                return Ok(());
                            };
                            let _ = incoming_tracks.send(RecvTrack {
                                id: stream_id,
                                kind,
                                commands,
                                frames: inbox,
                            });
                        }
                    }
                    signal::Signal::Ready(stream_id) => {
                        // False when we already timed the open out and removed
                        // the track. The Ready is stale; the Close we sent is
                        // what the peer should honor.
                        if stream_id & 1 != local_parity {
                            tracing::debug!(stream_id, "ignored ready for a remote id");
                        } else if engine.set_send_ready(stream_id, true)
                            && let Some(pending) = opening.remove(&stream_id)
                        {
                            tracing::info!(stream_id, "send track ready");

                            let Some(commands) = commands_tx.upgrade() else {
                                let _ = pending.reply.send(Err(Error::Closed));

                                return Ok(());
                            };
                            outgoing.insert(stream_id, pending.control_tx);
                            let _ = pending.reply.send(Ok(SendTrack {
                                id: stream_id,
                                kind: pending.kind,
                                commands,
                                control: Mutex::new(pending.control_rx),
                            }));
                        } else {
                            tracing::debug!(stream_id, "ignored ready");
                        }
                    }
                    signal::Signal::Reject(stream_id) => {
                        if stream_id & 1 != local_parity {
                            tracing::debug!(stream_id, "ignored reject for a remote id");
                        } else if let Some(pending) = opening.remove(&stream_id) {
                            tracing::warn!(stream_id, "send track rejected");

                            engine.remove_track(stream_id);
                            let _ = pending.reply.send(Err(Error::Rejected { stream_id }));
                        } else {
                            tracing::debug!(stream_id, "ignored reject");
                        }
                    }
                    signal::Signal::Close(stream_id) => {
                        if let Some(pending) = opening.remove(&stream_id) {
                            tracing::info!(stream_id, "peer closed track before ready");

                            engine.remove_track(stream_id);
                            let _ = pending.reply.send(Err(Error::Closed));
                        } else if engine.remove_track(stream_id) {
                            tracing::info!(stream_id, "peer closed track");

                            // Dropping the sender ends RecvTrack::recv and
                            // SendTrack::control after anything already queued.
                            outgoing.remove(&stream_id);
                            if let Some(inbox) = incoming.remove(&stream_id) {
                                inbox.close();
                            }
                        }
                    }
                }
            }
            received = udp.recv(&mut datagram) => {
                // A connected UDP socket on Windows reports ICMP unreachable
                // as ConnectionReset. That is one lost datagram, not the end
                // of the conversation.
                let n = match received {
                    Ok(n) => n,
                    Err(err) if err.kind() == io::ErrorKind::ConnectionReset => {
                        tracing::trace!("udp reset");

                        continue;
                    }
                    Err(err) => return Err(err.into()),
                };
                let result = engine.push_packet(&datagram[..n], Instant::now());
                next_wake = emit(result, &udp, &outgoing, &incoming, &mut cipher).await?;
            }
            () = async {
                match wake {
                    Some(at) => {
                        tokio::time::sleep(at.saturating_duration_since(Instant::now())).await;
                    }
                    None => pending::<()>().await,
                }
            } => {
                let now = Instant::now();
                let expired: Vec<u8> = opening
                    .iter()
                    .filter(|(_, pending)| pending.deadline <= now)
                    .map(|(stream_id, _)| *stream_id)
                    .collect();

                for stream_id in expired {
                    let Some(pending) = opening.remove(&stream_id) else {
                        continue;
                    };

                    tracing::warn!(stream_id, "send track open timed out");

                    engine.remove_track(stream_id);
                    signal::write(&mut write, &signal::Signal::Close(stream_id)).await?;
                    let _ = pending.reply.send(Err(Error::TimedOut { stream_id }));
                }

                if next_wake.is_some_and(|at| at <= now) {
                    let result = engine.tick(now);
                    next_wake = emit(result, &udp, &outgoing, &incoming, &mut cipher).await?;
                }
            }
        }
    }
}
