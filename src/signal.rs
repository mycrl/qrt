//! Length-prefixed messages on the signal TCP connection.
//!
//! Media stays on UDP. This connection carries hello, open, ready, close,
//! and reject. When the session is encrypted, the dialer's hello also
//! carries the AES key. TLS, if any, already wraps the socket before these
//! frames are read.
//!
//! ```text
//!  0      1          3
//! +------+----------+----------+
//! | type | length   | body     |
//! | u8   | u16 BE   | `length` |
//! +------+----------+----------+
//! ```

use std::net::{IpAddr, SocketAddr};

use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt},
    net::UdpSocket,
};

use crate::{
    Error,
    core::packet::MediaType,
    crypto,
    engine::{Engine, MediaKind},
};

/// Read half of the signal connection. TCP and TLS both fit.
pub type ReadHalf = Box<dyn AsyncRead + Unpin + Send>;

/// Write half of the signal connection. TCP and TLS both fit.
pub type WriteHalf = Box<dyn AsyncWrite + Unpin + Send>;

/// Wire type of one signal frame.
#[repr(u8)]
#[derive(Clone, Copy, Debug)]
enum Type {
    Hello = 1,
    Open = 2,
    Close = 3,
    Ready = 4,
    Reject = 5,
}

impl TryFrom<u8> for Type {
    type Error = Error;

    fn try_from(byte: u8) -> Result<Self, Error> {
        [
            Self::Hello,
            Self::Open,
            Self::Close,
            Self::Ready,
            Self::Reject,
        ]
        .into_iter()
        .find(|item| *item as u8 == byte)
        .ok_or(Error::Protocol("unknown signal"))
    }
}

const PORT_LEN: usize = 2;
const HELLO_WITH_KEY: usize = PORT_LEN + crypto::KEY_LEN;

/// Largest body that will be allocated.
///
/// Open, ready, and close are a few bytes. Hello with an AES key is 18
/// bytes. The cap refuses anything that would pin a large buffer from a peer.
const MAX_BODY: usize = 1024;

/// One message on the signal connection.
///
/// `Hello` is exchanged once, before the session task starts. Its body is
/// the local UDP port, and, when the session is encrypted, the dialer's
/// 16-byte AES key after that port. The listener's hello is only the port.
/// The address family and IP come from the TCP peer, not from this body.
#[derive(Debug)]
pub enum Signal {
    Hello {
        port: u16,
        media_key: Option<[u8; crypto::KEY_LEN]>,
    },
    Open {
        stream_id: u8,
        kind: MediaKind,
    },
    Close(u8),
    Ready(u8),
    Reject(u8),
}

impl Signal {
    fn encode(&self) -> Vec<u8> {
        let (message_type, body) = match self {
            Self::Hello { port, media_key } => {
                let mut body = port.to_be_bytes().to_vec();
                if let Some(key) = media_key {
                    body.extend_from_slice(key);
                }

                (Type::Hello, body)
            }
            Self::Open { stream_id, kind } => {
                let kind = match kind {
                    MediaKind::Video => MediaType::Video,
                    MediaKind::Audio => MediaType::Audio,
                };

                (Type::Open, vec![*stream_id, kind as u8])
            }
            Self::Close(stream_id) => (Type::Close, vec![*stream_id]),
            Self::Ready(stream_id) => (Type::Ready, vec![*stream_id]),
            Self::Reject(stream_id) => (Type::Reject, vec![*stream_id]),
        };

        let mut bytes = Vec::with_capacity(3 + body.len());
        bytes.push(message_type as u8);

        let len = u16::try_from(body.len()).expect("signal body fits in u16");
        bytes.extend(len.to_be_bytes());

        bytes.extend(body);

        bytes
    }

    fn decode(byte: u8, body: &[u8]) -> Result<Self, Error> {
        match Type::try_from(byte)? {
            Type::Hello => match body.len() {
                PORT_LEN => Ok(Self::Hello {
                    port: u16::from_be_bytes(body.try_into().expect("hello port is 2 bytes")),
                    media_key: None,
                }),
                HELLO_WITH_KEY => {
                    let port = u16::from_be_bytes(
                        body[..PORT_LEN].try_into().expect("hello port is 2 bytes"),
                    );

                    let mut media_key = [0u8; crypto::KEY_LEN];
                    media_key.copy_from_slice(&body[PORT_LEN..]);

                    Ok(Self::Hello {
                        port,
                        media_key: Some(media_key),
                    })
                }
                _ => Err(Error::Protocol("hello")),
            },
            Type::Open => {
                let &[stream_id, kind] = body else {
                    return Err(Error::Protocol("open"));
                };

                let kind = match kind {
                    v if v == MediaType::Video as u8 => MediaKind::Video,
                    v if v == MediaType::Audio as u8 => MediaKind::Audio,
                    _ => return Err(Error::Protocol("kind")),
                };

                Ok(Self::Open { stream_id, kind })
            }
            Type::Close => {
                let &[stream_id] = body else {
                    return Err(Error::Protocol("close"));
                };

                Ok(Self::Close(stream_id))
            }
            Type::Ready => {
                let &[stream_id] = body else {
                    return Err(Error::Protocol("ready"));
                };

                Ok(Self::Ready(stream_id))
            }
            Type::Reject => {
                let &[stream_id] = body else {
                    return Err(Error::Protocol("reject"));
                };

                Ok(Self::Reject(stream_id))
            }
        }
    }
}

/// Reads one frame.
///
/// Used by the handshake and by the dedicated reader task. It is not polled
/// inside `select!`: cancelling `read_exact` mid-frame would drop bytes
/// already taken off the socket.
pub async fn read<R>(read: &mut R) -> Result<Signal, Error>
where
    R: AsyncRead + Unpin,
{
    let mut header = [0u8; 3];
    read.read_exact(&mut header).await?;
    let kind = header[0];
    let len = usize::from(u16::from_be_bytes([header[1], header[2]]));
    if len > MAX_BODY {
        return Err(Error::Protocol("signal body too large"));
    }

    let mut body = vec![0u8; len];
    if len > 0 {
        read.read_exact(&mut body).await?;
    }

    let signal = Signal::decode(kind, &body)?;

    tracing::trace!(?signal, "signal in");

    Ok(signal)
}

/// Writes one frame.
pub async fn write<W>(write: &mut W, signal: &Signal) -> Result<(), Error>
where
    W: AsyncWrite + Unpin,
{
    tracing::trace!(?signal, "signal out");

    write.write_all(&signal.encode()).await?;

    Ok(())
}

/// Exchanges Hello, connects UDP, and starts the session task.
///
/// The dialer writes first. A listener that also wrote first would deadlock.
/// The AES key, when present, is in that same Hello, so encryption does not
/// add a round trip. The listener does not send a key back. `next_id` is
/// [`super::DIALER_IDS`] on the dialer and [`super::LISTENER_IDS`] on the listener.
#[allow(clippy::too_many_arguments)]
pub async fn complete_handshake(
    mut read_half: ReadHalf,
    mut write_half: WriteHalf,
    udp: UdpSocket,
    peer_ip: IpAddr,
    local_port: u16,
    engine: Engine,
    next_id: u16,
    encrypt: bool,
) -> Result<super::Session, Error> {
    let dialer = next_id == super::DIALER_IDS;

    let local_key = if encrypt && dialer {
        Some(crypto::random_key()?)
    } else {
        None
    };

    let hello = Signal::Hello {
        port: local_port,
        media_key: local_key,
    };

    if dialer {
        write(&mut write_half, &hello).await?;
    }

    let Signal::Hello {
        port: peer_port,
        media_key,
    } = read(&mut read_half).await?
    else {
        return Err(Error::Protocol("expected hello"));
    };

    if !dialer {
        write(&mut write_half, &hello).await?;
    }

    let cipher = match (encrypt, dialer, local_key, media_key) {
        (false, _, None, None) => None,
        (true, true, Some(key), None) => Some(crypto::Cipher::new(&key, true)?),
        (true, false, None, Some(key)) => Some(crypto::Cipher::new(&key, false)?),
        _ => return Err(Error::Protocol("encryption mismatch")),
    };

    let peer = SocketAddr::new(peer_ip, peer_port);
    udp.connect(peer).await?;

    tracing::info!(%peer, local_port, encrypted = cipher.is_some(), "udp connected");

    Ok(super::spawn_session(
        engine, read_half, write_half, udp, peer, next_id, cipher,
    ))
}
