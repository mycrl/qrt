//! Datagram I/O boundary for [`crate::Qrt`].
//!
//! Timers are Tokio ([`tokio::time`]) inside [`crate::Qrt`]; this trait is only
//! bare UDP send/recv (no QUIC).

use std::future::Future;

/// Async datagram transport used by [`crate::Qrt`].
///
/// # Examples
///
/// ```ignore
/// use bytes::Bytes;
/// use qrt::transport::Transport;
///
/// struct MyUdp { /* tokio::net::UdpSocket + peer */ }
///
/// impl Transport for MyUdp {
///     type Error = std::io::Error;
///
///     async fn send(&mut self, datagram: Bytes) -> Result<(), Self::Error> {
///         // socket.send_to(&datagram, peer).await?;
///         let _ = datagram;
///         Ok(())
///     }
///
///     async fn recv(&mut self, buf: &mut [u8]) -> Result<usize, Self::Error> {
///         let _ = buf;
///         Ok(0)
///     }
/// }
/// ```
pub trait Transport: Send {
    /// I/O failure type.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Sends one full datagram to the peer.
    fn send(&mut self, buf: &[u8]) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// Receives one datagram into `buf`.
    ///
    /// Prefer a buffer of at least ~1500 bytes.
    fn recv(&mut self, buf: &mut [u8]) -> impl Future<Output = Result<usize, Self::Error>> + Send;
}
