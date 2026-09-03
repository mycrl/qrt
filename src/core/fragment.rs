//! Codec-opaque frame fragmentation for [`Packet::Media`].
//!
//! Splits an encoded frame into MTU-sized payloads using the same
//! **about-equally** algorithm as WebRTC's
//! `RtpPacketizer::SplitAboutEqually` (`modules/rtp_rtcp/source/rtp_format.cc`).
//! The transport never inspects codec bytes (no VP8/H264 FU-A headers); first
//! and last fragment are identified by [`Header::frag_index`] /
//! [`Header::frag_count`] instead of an RTP marker bit.
//!
//! # Pipeline
//!
//! 1. [`split_about_equally`] — compute per-packet payload lengths.
//! 2. [`fragment`] — build [`Packet::Media`] values with consecutive
//!    [`Header::media_seq`], shared `frame_id` / `timestamp`, and
//!    `frag_index ∈ [0, frag_count)`.
//!
//! Sequence numbers are assigned here at fragment time (WebRTC assigns them
//! later in the pacer/`PacketSequencer`; either is fine for a single sender).
//!
//! # Examples
//!
//! ```
//! use qrt::core::{
//!     fragment::{FragmentParams, PayloadSizeLimits, fragment},
//!     packet::{Flags, PacketType},
//! };
//!
//! let frame = vec![0u8; 2500];
//! let packets = fragment(
//!     &frame,
//!     &FragmentParams {
//!         stream_id: 1,
//!         frame_id: 7,
//!         timestamp: 90_000,
//!         ttl_ms: 120,
//!         flags: Flags {
//!             key: true,
//!             ..Flags::default()
//!         },
//!         first_media_seq: 10,
//!         first_transport_seq: 10,
//!     },
//!     &PayloadSizeLimits::default(),
//! )
//! .unwrap();
//!
//! assert!(packets.len() >= 2);
//! assert_eq!(packets[0].header().media_seq, 10);
//! assert_eq!(packets[0].header().frag_index, 0);
//! assert_eq!(
//!     packets.last().unwrap().header().frag_index + 1,
//!     packets[0].header().frag_count
//! );
//! assert!(
//!     packets
//!         .iter()
//!         .all(|p| p.header().packet_type == PacketType::Media)
//! );
//! ```
//!
//! # Notes
//!
//! - [`PayloadSizeLimits::max_payload_len`] is the **media body** budget after
//!   [`crate::core::packet::HEADER_SIZE`], matching WebRTC's default of 1200 payload bytes.
//! - Reduction fields reserve room for larger first/last/single packet
//!   overheads (extensions, etc.); leave them `0` unless you need that.

use crate::core::packet::{Flags, Header, Packet, PacketType};

/// Default media-body size budget, same as WebRTC `kVideoMtu` / packetizer default.
pub const DEFAULT_MAX_PAYLOAD_LEN: usize = 1200;

/// Per-packet payload capacity knobs (WebRTC `RtpPacketizer::PayloadSizeLimits`).
///
/// Use when first/last/single datagrams need extra header space beyond the
/// common [`crate::core::packet::HEADER_SIZE`]. For a plain UDP path with a fixed header, defaults
/// (`max_payload_len = 1200`, reductions `0`) are enough.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PayloadSizeLimits {
    /// Maximum media payload bytes in any fragment body.
    pub max_payload_len: usize,
    /// Extra bytes reserved on the first packet of a multi-packet frame.
    pub first_packet_reduction_len: usize,
    /// Extra bytes reserved on the last packet of a multi-packet frame.
    pub last_packet_reduction_len: usize,
    /// Extra bytes reserved when the whole frame fits in one packet.
    pub single_packet_reduction_len: usize,
}

impl Default for PayloadSizeLimits {
    fn default() -> Self {
        Self {
            max_payload_len: DEFAULT_MAX_PAYLOAD_LEN,
            first_packet_reduction_len: 0,
            last_packet_reduction_len: 0,
            single_packet_reduction_len: 0,
        }
    }
}

/// Header fields shared by every fragment of one encoded frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FragmentParams {
    /// [`Header::stream_id`].
    pub stream_id: u8,
    /// [`Header::frame_id`] (reassembly key).
    pub frame_id: u32,
    /// [`Header::timestamp`] (90 kHz).
    pub timestamp: u32,
    /// Initial [`Header::ttl_ms`] for each fragment.
    pub ttl_ms: u16,
    /// Media flags (`key` / `audio`); `retrans` should be `false` on first send.
    pub flags: Flags,
    /// [`Header::media_seq`] of the first fragment; later fragments increment.
    pub first_media_seq: u16,
    /// [`Header::transport_seq`] of the first fragment; later fragments increment.
    ///
    /// Final values are usually overwritten at pacer egress; set here only for
    /// offline / test encoding.
    pub first_transport_seq: u16,
}

/// Error returned when a frame cannot be fragmented under the given limits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FragmentError {
    /// Frame payload is empty; there is nothing to send.
    EmptyPayload,
    /// Limits leave no capacity for a payload byte, or force more packets than bytes.
    ImpossibleLimits,
    /// Fragment count would exceed `u16::MAX` ([`Header::frag_count`]).
    TooManyFragments,
}

impl std::fmt::Display for FragmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyPayload => write!(f, "empty frame payload"),
            Self::ImpossibleLimits => {
                write!(f, "payload size limits cannot carry this frame")
            }
            Self::TooManyFragments => write!(f, "fragment count exceeds u16::MAX"),
        }
    }
}

impl std::error::Error for FragmentError {}

/// Compute about-equal payload lengths for one frame.
///
/// Port of WebRTC `RtpPacketizer::SplitAboutEqually`. Returns an empty vector
/// when the limits cannot carry `payload_len` (callers map that to
/// [`FragmentError::ImpossibleLimits`]).
///
/// # Panics
///
/// Panics if `payload_len == 0` (WebRTC `DCHECK_GT(payload_len, 0)`).
/// Prefer [`fragment`], which rejects empty payloads without panicking.
///
/// # Examples
///
/// ```
/// use qrt::core::fragment::{PayloadSizeLimits, split_about_equally};
///
/// // Fits in one packet after single-packet reduction.
/// let limits = PayloadSizeLimits {
///     max_payload_len: 30,
///     single_packet_reduction_len: 10,
///     first_packet_reduction_len: 29,
///     last_packet_reduction_len: 29,
/// };
/// assert_eq!(split_about_equally(20, &limits), vec![20]);
///
/// // Forced into two packets (WebRTC unit-test vector).
/// let limits = PayloadSizeLimits {
///     max_payload_len: 29,
///     first_packet_reduction_len: 3,
///     last_packet_reduction_len: 1,
///     single_packet_reduction_len: 10,
/// };
/// assert_eq!(split_about_equally(20, &limits), vec![9, 11]);
/// ```
pub fn split_about_equally(payload_len: usize, limits: &PayloadSizeLimits) -> Vec<usize> {
    assert!(payload_len > 0);

    if limits.max_payload_len >= limits.single_packet_reduction_len + payload_len {
        return vec![payload_len];
    }

    if limits
        .max_payload_len
        .saturating_sub(limits.first_packet_reduction_len)
        < 1
        || limits
            .max_payload_len
            .saturating_sub(limits.last_packet_reduction_len)
            < 1
    {
        return Vec::new();
    }

    // Pretend first/last reductions are extra payload so remaining capacity is even.
    let total_bytes =
        payload_len + limits.first_packet_reduction_len + limits.last_packet_reduction_len;
    let mut num_packets_left = (total_bytes + limits.max_payload_len - 1) / limits.max_payload_len;
    if num_packets_left == 1 {
        // Single-packet case already handled; force a split.
        num_packets_left = 2;
    }

    if payload_len < num_packets_left {
        return Vec::new();
    }

    let mut bytes_per_packet = total_bytes / num_packets_left;
    let num_larger_packets = total_bytes % num_packets_left;
    let mut remaining_data = payload_len;
    let mut result = Vec::with_capacity(num_packets_left);
    let mut first_packet = true;

    while remaining_data > 0 {
        // Last `num_larger_packets` packets are one byte wider; bump sticks for the tail.
        if num_packets_left == num_larger_packets {
            bytes_per_packet += 1;
        }

        let mut current_packet_bytes = bytes_per_packet;
        if first_packet {
            if current_packet_bytes > limits.first_packet_reduction_len + 1 {
                current_packet_bytes -= limits.first_packet_reduction_len;
            } else {
                current_packet_bytes = 1;
            }
        }

        if current_packet_bytes > remaining_data {
            current_packet_bytes = remaining_data;
        }

        // Leave at least one byte for the final packet.
        if num_packets_left == 2 && current_packet_bytes == remaining_data {
            current_packet_bytes -= 1;
        }

        result.push(current_packet_bytes);
        remaining_data -= current_packet_bytes;
        num_packets_left -= 1;
        first_packet = false;
    }

    result
}

/// Fragment an opaque encoded frame into [`Packet::Media`] datagrams.
///
/// Each packet borrows a slice of `frame`. All fragments share `frame_id` and
/// `timestamp`; `media_seq` runs from [`FragmentParams::first_media_seq`].
///
/// # Errors
///
/// - [`FragmentError::EmptyPayload`] if `frame` is empty.
/// - [`FragmentError::ImpossibleLimits`] if [`split_about_equally`] yields nothing.
/// - [`FragmentError::TooManyFragments`] if the fragment count does not fit in `u16`.
///
/// # Examples
///
/// ```
/// use qrt::core::{
///     fragment::{DEFAULT_MAX_PAYLOAD_LEN, FragmentParams, PayloadSizeLimits, fragment},
///     packet::{Flags, Packet},
/// };
///
/// let frame = vec![7u8; DEFAULT_MAX_PAYLOAD_LEN + 1];
/// let packets = fragment(
///     &frame,
///     &FragmentParams {
///         stream_id: 0,
///         frame_id: 1,
///         timestamp: 0,
///         ttl_ms: 100,
///         flags: Flags::default(),
///         first_media_seq: 0,
///         first_transport_seq: 0,
///     },
///     &PayloadSizeLimits::default(),
/// )
/// .unwrap();
///
/// assert_eq!(packets.len(), 2);
/// assert!(packets[0].header().is_first_frag());
/// assert!(packets[1].header().is_last_frag());
/// assert_eq!(packets[0].header().frag_count, 2);
///
/// // Payloads cover the whole frame without overlap.
/// let mut rebuilt = Vec::new();
/// for p in &packets {
///     if let Packet::Media { payload, .. } = p {
///         rebuilt.extend_from_slice(payload);
///     }
/// }
/// assert_eq!(rebuilt, frame);
/// ```
pub fn fragment<'a>(
    frame: &'a [u8],
    params: &FragmentParams,
    limits: &PayloadSizeLimits,
) -> Result<Vec<Packet<'a>>, FragmentError> {
    if frame.is_empty() {
        return Err(FragmentError::EmptyPayload);
    }

    let sizes = split_about_equally(frame.len(), limits);
    if sizes.is_empty() {
        return Err(FragmentError::ImpossibleLimits);
    }

    if sizes.len() > u16::MAX as usize {
        return Err(FragmentError::TooManyFragments);
    }

    let frag_count = sizes.len() as u16;
    let mut fragments = Vec::with_capacity(sizes.len());
    let mut offset = 0usize;

    for (i, size) in sizes.into_iter().enumerate() {
        let end = offset + size;
        let payload = &frame[offset..end];
        offset = end;

        fragments.push(Packet::Media {
            header: Header {
                packet_type: PacketType::Media,
                flags: params.flags,
                stream_id: params.stream_id,
                media_seq: params.first_media_seq.wrapping_add(i as u16),
                transport_seq: params.first_transport_seq.wrapping_add(i as u16),
                frame_id: params.frame_id,
                frag_index: i as u16,
                frag_count,
                timestamp: params.timestamp,
                ttl_ms: params.ttl_ms,
            },
            payload,
        });
    }

    debug_assert_eq!(offset, frame.len());

    Ok(fragments)
}
