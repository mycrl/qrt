//! Optional session encryption.
//!
//! TLS is rustls wrapping the signal TCP connection. There is no second
//! handshake. The dialer generates one AES-128 key and sends it in Hello,
//! which is already inside TLS. Every encoded frame is then sealed with
//! that key before it reaches the engine.
//!
//! Both sides start their frame counters at zero. The nonce's first byte is
//! the direction, so the two counters do not reuse a nonce under the one key.

use std::{net::IpAddr, sync::Arc};

use bytes::Bytes;
use ring::{
    aead::{self, Aad, LessSafeKey, Nonce, UnboundKey},
    rand::{self, SecureRandom},
};
use rustls::{
    RootCertStore,
    pki_types::{CertificateDer, PrivateKeyDer, ServerName},
    version::TLS13,
};
use tokio_rustls::{TlsAcceptor, TlsConnector};

use crate::Error;

/// AES-128 key length. Also the Hello suffix when encryption is on.
pub const KEY_LEN: usize = 16;

const COUNTER_LEN: usize = 8;
const NONCE_LEN: usize = 12;
const TAG_LEN: usize = 16;

/// First nonce byte for frames the dialer seals.
const DIALER_DIRECTION: u8 = 0;
/// First nonce byte for frames the listener seals.
const LISTENER_DIRECTION: u8 = 1;

/// How many receive counters behind the newest one are still accepted.
///
/// A frame older than this is treated as a replay. UDP reordering inside the
/// window is still delivered.
const REPLAY_WINDOW: u64 = 64;

/// Certificate and private key for [`crate::Server::bind_tls`].
///
/// This is the server identity only. The client does not present a
/// certificate. [`crate::Client::connect_tls`] takes the CA that signed this
/// certificate and lets rustls verify it.
pub struct TlsConfig {
    /// PEM certificate chain, leaf first.
    pub certificate_pem: Vec<u8>,
    /// PEM private key for the leaf. PKCS#8 or PKCS#1.
    pub private_key_pem: Vec<u8>,
}

impl TlsConfig {
    /// Builds a server identity from PEM bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use qrt::crypto::TlsConfig;
    ///
    /// let config = TlsConfig::from_pem(b"cert", b"key");
    /// assert_eq!(config.certificate_pem, b"cert");
    /// ```
    pub fn from_pem(
        certificate_pem: impl Into<Vec<u8>>,
        private_key_pem: impl Into<Vec<u8>>,
    ) -> Self {
        Self {
            certificate_pem: certificate_pem.into(),
            private_key_pem: private_key_pem.into(),
        }
    }
}

/// One random AES-128 key for this session.
pub fn random_key() -> Result<[u8; KEY_LEN], Error> {
    let mut key = [0u8; KEY_LEN];
    rand::SystemRandom::new()
        .fill(&mut key)
        .map_err(|_| Error::Crypto("key"))?;

    Ok(key)
}

/// AES-128-GCM state for one session.
///
/// The engine never sees this. [`Self::seal`] runs on the frame payload before
/// `push_frame`. [`Self::open`] runs on the frame the engine has reassembled.
pub struct Cipher {
    key: LessSafeKey,
    seal_direction: u8,
    open_direction: u8,
    send_counter: u64,
    replay_started: bool,
    recv_max: u64,
    recv_mask: u64,
}

impl Cipher {
    /// `local_is_dialer` picks which nonce direction seals and which opens.
    pub fn new(key: &[u8], local_is_dialer: bool) -> Result<Self, Error> {
        let (seal_direction, open_direction) = if local_is_dialer {
            (DIALER_DIRECTION, LISTENER_DIRECTION)
        } else {
            (LISTENER_DIRECTION, DIALER_DIRECTION)
        };

        Ok(Self {
            key: aead_key(key)?,
            seal_direction,
            open_direction,
            send_counter: 0,
            replay_started: false,
            recv_max: 0,
            recv_mask: 0,
        })
    }

    /// Seals `payload` and prepends the frame counter those bytes were sealed under.
    ///
    /// The counter is also additional data, so it cannot be moved onto another
    /// ciphertext. Layout: `counter u64 BE || ciphertext || tag`.
    pub fn seal(&mut self, payload: Bytes) -> Result<Bytes, Error> {
        let counter = self.send_counter;
        self.send_counter = self.send_counter.wrapping_add(1);
        let mut body = payload.to_vec();
        self.key
            .seal_in_place_append_tag(
                nonce(self.seal_direction, counter),
                Aad::from(&counter.to_be_bytes()),
                &mut body,
            )
            .map_err(|_| Error::Crypto("seal"))?;

        let mut out = Vec::with_capacity(COUNTER_LEN + body.len());
        out.extend_from_slice(&counter.to_be_bytes());
        out.extend_from_slice(&body);

        Ok(Bytes::from(out))
    }

    /// Checks the counter, then opens one sealed frame.
    pub fn open(&mut self, payload: Bytes) -> Result<Bytes, Error> {
        if payload.len() < COUNTER_LEN + TAG_LEN {
            return Err(Error::Crypto("short frame"));
        }

        let counter = u64::from_be_bytes(payload[..COUNTER_LEN].try_into().expect("8 bytes"));
        if !self.accept_counter(counter) {
            return Err(Error::Crypto("replay"));
        }

        let mut body = payload[COUNTER_LEN..].to_vec();
        let plain = self
            .key
            .open_in_place(
                nonce(self.open_direction, counter),
                Aad::from(&payload[..COUNTER_LEN]),
                &mut body,
            )
            .map_err(|_| Error::Crypto("open"))?;

        Ok(Bytes::copy_from_slice(plain))
    }

    /// Sliding window of the last [`REPLAY_WINDOW`] counters.
    fn accept_counter(&mut self, counter: u64) -> bool {
        if !self.replay_started {
            self.replay_started = true;
            self.recv_max = counter;
            self.recv_mask = 1;

            return true;
        }

        if counter > self.recv_max {
            let shift = counter - self.recv_max;
            self.recv_mask = if shift >= REPLAY_WINDOW {
                1
            } else {
                (self.recv_mask << shift) | 1
            };
            self.recv_max = counter;

            return true;
        }

        let behind = self.recv_max - counter;
        if behind >= REPLAY_WINDOW {
            return false;
        }

        let bit = 1u64 << behind;
        if self.recv_mask & bit != 0 {
            return false;
        }

        self.recv_mask |= bit;

        true
    }
}

/// TLS acceptor for [`crate::Server::bind_tls`].
///
/// Same shape as a rustls server with no client certificate: the provider,
/// TLS 1.3, and one certificate chain.
pub fn server_acceptor(config: &TlsConfig) -> Result<TlsAcceptor, Error> {
    let server = rustls::ServerConfig::builder_with_provider(provider())
        .with_protocol_versions(&[&TLS13])
        .map_err(|_| Error::Crypto("tls version"))?
        .with_no_client_auth()
        .with_single_cert(
            certificate_chain(&config.certificate_pem)?,
            private_key(&config.private_key_pem)?,
        )
        .map_err(|_| Error::Crypto("tls identity"))?;

    Ok(TlsAcceptor::from(Arc::new(server)))
}

/// TLS connector for [`crate::Client::connect_tls`].
///
/// `trust_anchor_pem` is the CA rustls uses to check the server certificate.
pub fn client_connector(trust_anchor_pem: &[u8]) -> Result<TlsConnector, Error> {
    let mut roots = RootCertStore::empty();
    for cert in certificate_chain(trust_anchor_pem)? {
        roots.add(cert).map_err(|_| Error::Crypto("trust anchor"))?;
    }

    let client = rustls::ClientConfig::builder_with_provider(provider())
        .with_protocol_versions(&[&TLS13])
        .map_err(|_| Error::Crypto("tls version"))?
        .with_root_certificates(roots)
        .with_no_client_auth();

    Ok(TlsConnector::from(Arc::new(client)))
}

/// Name rustls checks against the certificate. The dial API takes an IP, so
/// the certificate must list that IP.
pub fn server_name(ip: IpAddr) -> ServerName<'static> {
    ServerName::from(ip)
}

/// ring provider passed in explicitly. The process-wide default is never installed.
fn provider() -> Arc<rustls::crypto::CryptoProvider> {
    Arc::new(rustls::crypto::ring::default_provider())
}

/// AES-128-GCM key whose nonce is chosen by the caller.
///
/// ring names this `LessSafeKey` because uniqueness is the caller's job.
/// [`Cipher`] uses each frame counter once, and the direction byte keeps the
/// two sides apart.
fn aead_key(key: &[u8]) -> Result<LessSafeKey, Error> {
    let unbound = UnboundKey::new(&aead::AES_128_GCM, key).map_err(|_| Error::Crypto("aes key"))?;

    Ok(LessSafeKey::new(unbound))
}

/// Nonce is `direction || 3 zero bytes || counter`.
fn nonce(direction: u8, counter: u64) -> Nonce {
    let mut bytes = [0u8; NONCE_LEN];
    bytes[0] = direction;
    bytes[NONCE_LEN - COUNTER_LEN..].copy_from_slice(&counter.to_be_bytes());

    Nonce::assume_unique_for_key(bytes)
}

/// Every certificate in the PEM, in file order.
fn certificate_chain(pem: &[u8]) -> Result<Vec<CertificateDer<'static>>, Error> {
    let mut reader = std::io::Cursor::new(pem);
    let certs = rustls_pemfile::certs(&mut reader)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| Error::Crypto("certificate pem"))?;
    if certs.is_empty() {
        return Err(Error::Crypto("certificate pem"));
    }

    Ok(certs)
}

/// First private key in the PEM. PKCS#8 or PKCS#1.
fn private_key(pem: &[u8]) -> Result<PrivateKeyDer<'static>, Error> {
    let mut reader = std::io::Cursor::new(pem);
    rustls_pemfile::private_key(&mut reader)
        .map_err(|_| Error::Crypto("private key pem"))?
        .ok_or(Error::Crypto("private key pem"))
}
