//! Fixed-layout binary codec for real-time media **and** feedback over **bare UDP**.
//!
//! This module is the on-wire format for `qrt`. One datagram family covers what
//! WebRTC splits across **RTP** (media) and **RTCP** (feedback): there is **no**
//! separate RTCP channel, compound packet, or RTCP mux.
//!
//! | Role (WebRTC / RFC idea) | This crate |
//! |--------------------------|------------|
//! | RTP media | [`Packet::Media`] |
//! | RTCP Generic NACK (RFC 4585) | [`Packet::Nack`] |
//! | Transport-wide arrival / TWCC-style stats | [`Packet::ArrivalFeedback`] |
//! | PLI / FIR (keyframe request) | [`Packet::KeyframeReq`] |
//! | ULPFEC / FlexFEC (XOR erasure) | [`Packet::Fec`] |
//!
//! The layout is intentionally **not** wire-compatible with RTP/RTCP. It borrows
//! the ideas (seq, timestamp, frag, NACK, arrival bits, transport-wide CC) but
//! uses a custom [`HEADER_SIZE`]-byte header for a known-peer, no-ICE, no-QUIC stack.
//!
//! # Why hand-rolled binary?
//!
//! Real-time UDP packets need a tiny, predictable header that fits MTU
//! budgets. Document formats (BSON/JSON) and heavy schema codecs are a poor
//! fit for the per-packet hot path. Media, FEC, and feedback share this header
//! so the pacer can schedule them on the same socket.
//!
//! # Wire layout
//!
//! Every datagram begins with [`HEADER_SIZE`] bytes:
//!
//! ```text
//!  0                   1                   2                   3
//!  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! |V=0|Type |Flags|   Stream ID   |         Media Seq             |
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! |       Transport Seq           |           Frame ID...         |
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! |                        ...Frame ID                            |
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! |          Frag Index           |          Frag Count           |
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! |                     Timestamp (90 kHz)                        |
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! |        TTL (ms)               |           Payload...          |
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! ```
//!
//! Multi-byte integers are encoded **big-endian**. The `Type` field selects
//! media / FEC / feedback body layout (see [`PacketType`]).
//!
//! # Examples
//!
//! Round-trip a media datagram:
//!
//! ```
//! use qrt::core::packet::{Flags, HEADER_SIZE, Header, Packet, PacketType};
//!
//! let pkt = Packet::Media {
//!     header: Header {
//!         packet_type: PacketType::Media,
//!         flags: Flags {
//!             key: true,
//!             ..Flags::default()
//!         },
//!         stream_id: 1,
//!         media_seq: 7,
//!         transport_seq: 42,
//!         frame_id: 3,
//!         frag_index: 0,
//!         frag_count: 1,
//!         timestamp: 90_000,
//!         ttl_ms: 120,
//!     },
//!     payload: b"opaque-codec-bytes",
//! };
//!
//! let mut wire = [0u8; HEADER_SIZE + 18];
//! pkt.encode(&mut wire);
//! assert_eq!(Packet::decode(&wire).unwrap(), pkt);
//! ```
//!
//! # Notes
//!
//! - [`Header::ttl_ms`] is a *remaining lifetime*, not a wall-clock deadline.
//!   Peers do not need synchronized clocks; the sender shrinks TTL as the
//!   packet waits in local queues.
//! - [`Header::media_seq`] is per-stream media identity (NACK / reassembly).
//!   [`Header::transport_seq`] is connection-wide (TWCC / BWE / in-flight) and
//!   must advance on **every** paced send, including retrans and FEC.
//! - Sequence comparisons must use [`Header::seq_ahead`], not raw `<`.
//! - Feedback packets still carry the common header so demux, TTL, and
//!   stream affinity stay uniform; unused frag fields should be
//!   `frag_count = 1`, `frag_index = 0`.

use std::time::Duration;

use bytes::{Buf, BufMut};

/// Size in bytes of the fixed header that prefixes every datagram.
///
/// # Examples
///
/// ```
/// use qrt::core::packet::HEADER_SIZE;
/// assert_eq!(HEADER_SIZE, 20);
/// ```
pub const HEADER_SIZE: usize = 20;

/// Protocol version stored in the top 2 bits of the first header byte.
///
/// Decoders must reject any other value with [`DecodeError::UnsupportedVersion`].
pub const VERSION: u8 = 0;

/// Error returned when a datagram cannot be decoded.
///
/// # Examples
///
/// ```
/// use qrt::core::packet::{DecodeError, HEADER_SIZE, Header};
///
/// let err = Header::decode(&b"hi"[..]).unwrap_err();
/// assert!(matches!(
///     err,
///     DecodeError::TooShort {
///         need: HEADER_SIZE,
///         have: 2
///     }
/// ));
///
/// // Unsupported version bits in the first byte.
/// let mut bad_ver = [0u8; HEADER_SIZE];
/// bad_ver[0] = 0b01_000_000;
/// assert!(matches!(
///     Header::decode(&bad_ver),
///     Err(DecodeError::UnsupportedVersion(1))
/// ));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// The buffer does not contain enough bytes for the expected structure.
    TooShort {
        /// Minimum number of bytes required.
        need: usize,
        /// Number of bytes actually available.
        have: usize,
    },
    /// Header version bits do not match [`VERSION`].
    UnsupportedVersion(u8),
    /// `Type` field is not a known [`PacketType`].
    UnknownType(u8),
    /// Media header claimed `frag_count == 0` (a frame must have >= 1 fragment).
    InvalidFragCount,
    /// Media header claimed `frag_index >= frag_count`.
    InvalidFragIndex,
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooShort { need, have } => {
                write!(f, "buffer too short: need {need}, have {have}")
            }
            Self::UnsupportedVersion(v) => write!(f, "unsupported version {v}"),
            Self::UnknownType(t) => write!(f, "unknown packet type {t}"),
            Self::InvalidFragCount => write!(f, "frag_count is 0"),
            Self::InvalidFragIndex => write!(f, "frag_index >= frag_count"),
        }
    }
}

impl std::error::Error for DecodeError {}

/// Discriminator stored in the 3-bit `Type` field of the first header byte.
///
/// Selects media, FEC, or feedback body. Feedback and FEC share this UDP path
/// with media (no separate RTCP / FEC session).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PacketType {
    /// Encoded media fragment (audio or video); RTP-like role.
    Media = 0,
    /// Selective retransmission request (RTCP Generic NACK role).
    Nack = 1,
    /// Packet-arrival bitmap for loss / delay stats (TWCC / transport-cc role).
    ArrivalFeedback = 2,
    /// Ask the sender for a keyframe (RTCP PLI / FIR role).
    KeyframeReq = 3,
    /// Forward error correction parity (ULPFEC / FlexFEC XOR role).
    Fec = 4,
}

impl PacketType {
    /// Parse a 3-bit type field from the wire.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeError::UnknownType`] if `v` is outside `0..=4`.
    ///
    /// # Examples
    ///
    /// ```
    /// use qrt::core::packet::PacketType;
    /// assert_eq!(PacketType::from_u8(0).unwrap(), PacketType::Media);
    /// assert_eq!(PacketType::from_u8(4).unwrap(), PacketType::Fec);
    /// assert!(PacketType::from_u8(7).is_err());
    /// ```
    pub fn from_u8(v: u8) -> Result<Self, DecodeError> {
        match v {
            0 => Ok(Self::Media),
            1 => Ok(Self::Nack),
            2 => Ok(Self::ArrivalFeedback),
            3 => Ok(Self::KeyframeReq),
            4 => Ok(Self::Fec),
            _ => Err(DecodeError::UnknownType(v)),
        }
    }
}

/// Three flag bits packed into the low 3 bits of the first header byte.
///
/// Bit layout: `retrans | (audio << 1) | (key << 2)`.
///
/// These flags are **not** RTP Marker/Padding bits. They let send/receive
/// logic classify a packet without parsing the codec payload.
///
/// # Examples
///
/// ```
/// use qrt::core::packet::Flags;
///
/// let f = Flags {
///     retrans: true,
///     audio: false,
///     key: true,
/// };
/// assert_eq!(f.pack() & 0b111, 0b101);
/// assert_eq!(Flags::unpack(0b101), f);
/// ```
///
/// # Notes
///
/// [`Flags::unpack`] ignores bits above the low 3; callers typically pass
/// `first_byte & 0b111`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Flags {
    /// Retransmission (NACK-driven resend; same [`Header::media_seq`] as original).
    ///
    /// Send schedulers may raise priority; stats can split first-send vs retrans.
    pub retrans: bool,
    /// `true` = audio, `false` = video.
    ///
    /// Schedulers should prefer audio; receivers may use a shorter jitter buffer.
    pub audio: bool,
    /// Packet belongs to a keyframe (or a fragment of one).
    ///
    /// Heavy loss may trigger [`Packet::KeyframeReq`]; senders may prioritize
    /// key fragments.
    pub key: bool,
}

impl Flags {
    /// Pack into 3 bits: bit0=`retrans`, bit1=`audio`, bit2=`key`.
    ///
    /// Higher bits of the returned `u8` are always zero.
    pub fn pack(self) -> u8 {
        (u8::from(self.retrans)) | (u8::from(self.audio) << 1) | (u8::from(self.key) << 2)
    }

    /// Unpack flags from the low 3 bits of a byte.
    ///
    /// Bits `3..8` are ignored.
    pub fn unpack(v: u8) -> Self {
        Self {
            retrans: v & 0b001 != 0,
            audio: v & 0b010 != 0,
            key: v & 0b100 != 0,
        }
    }
}

/// Common [`HEADER_SIZE`]-byte header shared by all datagram types.
///
/// See the [module-level wire layout](crate::core::packet) for the bit diagram.
///
/// # Examples
///
/// ```
/// use qrt::core::packet::{DecodeError, Flags, HEADER_SIZE, Header, PacketType};
///
/// let header = Header {
///     packet_type: PacketType::Media,
///     flags: Flags::default(),
///     stream_id: 0,
///     media_seq: 1,
///     transport_seq: 1,
///     frame_id: 1,
///     frag_index: 0,
///     frag_count: 1,
///     timestamp: 0,
///     ttl_ms: 100,
/// };
///
/// let mut buf = [0u8; HEADER_SIZE];
/// header.encode(&mut buf);
/// assert_eq!(Header::decode(&buf).unwrap(), header);
///
/// // Media fragmentation fields are validated on decode.
/// let mut bad = header.clone();
/// bad.frag_count = 0;
/// let mut wire = [0u8; HEADER_SIZE];
/// bad.encode(&mut wire);
/// assert!(matches!(
///     Header::decode(&wire),
///     Err(DecodeError::InvalidFragCount)
/// ));
///
/// bad.frag_count = 2;
/// bad.frag_index = 2;
/// bad.encode(&mut wire);
/// assert!(matches!(
///     Header::decode(&wire),
///     Err(DecodeError::InvalidFragIndex)
/// ));
/// ```
///
/// # Notes
///
/// For non-[`PacketType::Media`] packets, `frag_index` / `frag_count` are
/// unused placeholders; encoders should set `frag_count = 1`, `frag_index = 0`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    /// Datagram kind (media, FEC, NACK, arrival feedback, or keyframe request).
    pub packet_type: PacketType,
    /// Retrans / audio / keyframe hints (see [`Flags`]).
    pub flags: Flags,
    /// Logical media stream multiplex id (bare UDP has no transport stream id).
    pub stream_id: u8,
    /// Per-stream media sequence for NACK, reassembly, and media dedup.
    ///
    /// Unchanged across retransmission of the same media packet. Wraps at 16
    /// bits; use [`Header::seq_ahead`] for ordering comparisons.
    pub media_seq: u16,
    /// Connection-wide transport sequence (WebRTC transport-cc role).
    ///
    /// Assigned when the datagram is actually sent (pacer egress). Advances for
    /// media, retrans, FEC, and padding alike so BWE sees real in-flight load.
    /// [`Packet::ArrivalFeedback`] reports this space, **not** [`Self::media_seq`].
    pub transport_seq: u16,
    /// Frame identity used as the reassembly / deadline key.
    ///
    /// Opaque to the codec; the transport never interprets payload bytes.
    pub frame_id: u32,
    /// Zero-based fragment index within the frame (`0` = first fragment).
    pub frag_index: u16,
    /// Total number of fragments that make up the frame.
    ///
    /// The last fragment satisfies `frag_index + 1 == frag_count`.
    pub frag_count: u16,
    /// Capture-clock timestamp in 90 kHz ticks (RTP-video-style clock rate).
    pub timestamp: u32,
    /// Remaining lifetime in milliseconds.
    ///
    /// `0` means the packet is already stale and must not be sent (or should
    /// be dropped on receive). This avoids depending on synchronized clocks.
    pub ttl_ms: u16,
}

impl Header {
    /// Returns `true` if this is the first fragment of a frame (`frag_index == 0`).
    pub fn is_first_frag(&self) -> bool {
        self.frag_index == 0
    }

    /// Returns `true` if this is the last fragment of a frame.
    ///
    /// Requires `frag_count > 0` and `frag_index + 1 == frag_count`.
    pub fn is_last_frag(&self) -> bool {
        self.frag_count > 0 && self.frag_index + 1 == self.frag_count
    }

    /// Write the [`HEADER_SIZE`]-byte header into the front of `bytes` (big-endian).
    ///
    /// Does not validate media fragmentation fields; invalid media headers
    /// are rejected on [`Header::decode`] instead.
    ///
    /// # Panics
    ///
    /// Panics if `bytes.len() < `[`HEADER_SIZE`].
    pub fn encode(&self, mut bytes: &mut [u8]) {
        assert!(bytes.len() >= HEADER_SIZE);

        bytes.put_u8(
            ((VERSION & 0b11) << 6)
                | (((self.packet_type as u8) & 0b111) << 3)
                | (self.flags.pack() & 0b111),
        );

        bytes.put_u8(self.stream_id);
        bytes.put_u16(self.media_seq);
        bytes.put_u16(self.transport_seq);
        bytes.put_u32(self.frame_id);
        bytes.put_u16(self.frag_index);
        bytes.put_u16(self.frag_count);
        bytes.put_u32(self.timestamp);
        bytes.put_u16(self.ttl_ms);
    }

    /// Decode one header from the first [`HEADER_SIZE`] bytes of `bytes`.
    ///
    /// # Errors
    ///
    /// - [`DecodeError::TooShort`] if fewer than [`HEADER_SIZE`] bytes remain.
    /// - [`DecodeError::UnsupportedVersion`] if the version bits are not
    ///   [`VERSION`].
    /// - [`DecodeError::UnknownType`] if the type field is reserved.
    /// - [`DecodeError::InvalidFragCount`] / [`DecodeError::InvalidFragIndex`]
    ///   when `packet_type` is [`PacketType::Media`] and fragmentation fields
    ///   are impossible (`frag_count == 0` or `frag_index >= frag_count`).
    ///
    /// Non-media types skip fragmentation validation because those fields are
    /// unused placeholders.
    pub fn decode(mut bytes: &[u8]) -> Result<Self, DecodeError> {
        if bytes.len() < HEADER_SIZE {
            return Err(DecodeError::TooShort {
                need: HEADER_SIZE,
                have: bytes.len(),
            });
        }

        let first_byte = bytes.get_u8();

        let version = first_byte >> 6;
        if version != VERSION {
            return Err(DecodeError::UnsupportedVersion(version));
        }

        let packet_type = PacketType::from_u8((first_byte >> 3) & 0b111)?;
        let flags = Flags::unpack(first_byte & 0b111);

        let header = Self {
            packet_type,
            flags,
            stream_id: bytes.get_u8(),
            media_seq: bytes.get_u16(),
            transport_seq: bytes.get_u16(),
            frame_id: bytes.get_u32(),
            frag_index: bytes.get_u16(),
            frag_count: bytes.get_u16(),
            timestamp: bytes.get_u32(),
            ttl_ms: bytes.get_u16(),
        };

        // Reject impossible media fragmentation before reassembly so a
        // corrupt/buggy peer cannot stall the frame buffer forever.
        if header.packet_type == PacketType::Media {
            // A frame must consist of at least one fragment.
            if header.frag_count == 0 {
                return Err(DecodeError::InvalidFragCount);
            }

            // frag_index is zero-based and must lie in [0, frag_count).
            if header.frag_index >= header.frag_count {
                return Err(DecodeError::InvalidFragIndex);
            }
        }

        Ok(header)
    }

    /// Wrap-safe ordering for 16-bit [`Header::media_seq`] values.
    ///
    /// Returns `true` if `a` is *ahead of* `b` on the circular 16-bit space
    /// (RFC 1982 / RTP-style comparison). Equal sequences return `false`.
    ///
    /// # Examples
    ///
    /// ```
    /// use qrt::core::packet::Header;
    ///
    /// assert!(Header::seq_ahead(1, 0));
    /// assert!(Header::seq_ahead(0, 0xFFFF)); // wrap
    /// assert!(!Header::seq_ahead(0, 1));
    /// assert!(!Header::seq_ahead(5, 5));
    /// ```
    ///
    /// # Warning
    ///
    /// This is undefined for distances >= `0x8000`; peers must not let the
    /// unacked window grow that large.
    pub fn seq_ahead(a: u16, b: u16) -> bool {
        let diff = a.wrapping_sub(b);
        diff != 0 && diff < 0x8000
    }
}

/// Fixed body prefix size for [`Packet::ArrivalFeedback`] before optional
/// recv-delta bytes (`first_seq` + `received_mask`).
pub const ARRIVAL_FEEDBACK_BODY_PREFIX: usize = 2 + 8;

/// Receive-time delta tick used in [`Packet::ArrivalFeedback`] trailing fields
/// (WebRTC transport-cc uses 250µs).
pub const ARRIVAL_RECV_DELTA_TICK: Duration = Duration::from_micros(250);

/// Fixed body prefix size for [`Packet::Fec`] before the XOR bytes
/// (`seq_base` + `mask` + `length_xor`).
pub const FEC_BODY_HEADER_SIZE: usize = 2 + 8 + 2;

/// One UDP datagram: fixed [`Header`] plus a type-specific body.
///
/// Carries **media**, **FEC**, and the feedback normally sent as RTCP. Peers
/// demux only on [`Header::packet_type`]; there is no parallel RTCP/FEC port.
///
/// # Variants
///
/// - [`Packet::Media`] - opaque codec bytes (RTP role).
/// - [`Packet::Nack`] - `base_seq` + BLP (RTCP NACK role).
/// - [`Packet::ArrivalFeedback`] - received-seq mask for BWE (TWCC-style role).
/// - [`Packet::KeyframeReq`] - request a keyframe (PLI/FIR role).
/// - [`Packet::Fec`] - XOR parity over recent media datagrams (ULPFEC/FlexFEC role).
///
/// # Examples
///
/// See the [module-level example](crate::core::packet).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Packet<'a> {
    /// Media fragment. `payload` is never interpreted by the transport.
    Media {
        /// Packet header (`packet_type` should be [`PacketType::Media`]).
        header: Header,
        /// Opaque encoded media bytes for this fragment.
        payload: &'a [u8],
    },
    /// Selective retransmission request (RTCP Generic NACK role).
    ///
    /// Body layout: `base_seq:u16`, `blp:u16`, optional `frame_id:u32`.
    /// Missing sequences are `base_seq` plus each set bit `i` in `blp`
    /// meaning `base_seq + 1 + i` (RFC 4585 PID+BLP style). Use
    /// [`Packet::nack_missing_seqs`] to expand.
    Nack {
        /// Header with `packet_type == Nack`.
        header: Header,
        /// First missing sequence number (PID).
        base_seq: u16,
        /// Bitmask of further losses after `base_seq`.
        blp: u16,
        /// Optional frame hint for the sender's retransmission history.
        frame_id: Option<u32>,
    },
    /// Arrival feedback for loss / delay observation (TWCC-style RTCP role).
    ///
    /// Feeds sender-side BWE; this packet is not a congestion controller by
    /// itself. Body: `first_seq:u16`, `received_mask:u64`, then optionally one
    /// big-endian `u16` per set bit in `received_mask` (low bit first) — recv
    /// deltas in **250µs** ticks relative to the previous reported packet
    /// (first delta is typically `0`). Bit `i` means [`Header::transport_seq`]
    /// `first_seq.wrapping_add(i)` was received (never use [`Header::media_seq`]
    /// here). [`Header::timestamp`] may carry a base receive time in
    /// milliseconds (wrapping) for the first reported packet.
    ///
    /// Mask-only feedback (no trailing deltas) remains valid; senders then know
    /// loss but not per-packet receive times.
    ArrivalFeedback {
        /// Header with `packet_type == ArrivalFeedback`.
        header: Header,
        /// First **transport** sequence covered by `received_mask`.
        first_seq: u16,
        /// Bit `i` set => `first_seq + i` received.
        received_mask: u64,
        /// Big-endian `u16` recv deltas (250µs ticks), one per set mask bit;
        /// empty when omitted.
        recv_delta_bytes: &'a [u8],
    },
    /// Request that `stream_id` produce a keyframe (RTCP PLI / FIR role).
    KeyframeReq {
        /// Header with `packet_type == KeyframeReq`.
        header: Header,
        /// Target media stream (duplicated in the body for easy parsing).
        stream_id: u8,
    },
    /// XOR forward-error-correction parity over media datagrams.
    ///
    /// Body: `seq_base:u16`, `mask:u64`, `length_xor:u16`, then `payload` =
    /// XOR of the protected **full Media datagrams** ([`HEADER_SIZE`]-byte header + body),
    /// each zero-padded to `payload.len()`. Bit `i` in `mask` protects
    /// `seq_base.wrapping_add(i)` (use at most 48 bits, matching WebRTC's
    /// `kUlpfecMaxMediaPackets`). `length_xor` is the XOR of each protected
    /// datagram's length as `u16`.
    ///
    /// # Notes
    ///
    /// - Recovery works when exactly one protected packet is missing (single
    ///   parity); multiple FEC rows with different masks raise the repair rate.
    /// - FEC packets must **not** be NACK'd; recovered media clears NACK state.
    /// - [`Header::stream_id`] is the protected media stream; FEC is not
    ///   retransmitted (`flags.retrans` stays false).
    Fec {
        /// Header with `packet_type == Fec`.
        header: Header,
        /// First media sequence covered by `mask`.
        seq_base: u16,
        /// Bit `i` set => protects `seq_base + i`.
        mask: u64,
        /// XOR of protected Media datagram lengths (`u16`).
        length_xor: u16,
        /// XOR of zero-padded Media datagrams.
        payload: &'a [u8],
    },
}

impl<'a> Packet<'a> {
    /// Returns the shared header for any variant.
    pub fn header(&self) -> &Header {
        match self {
            Self::Media { header, .. }
            | Self::Nack { header, .. }
            | Self::ArrivalFeedback { header, .. }
            | Self::KeyframeReq { header, .. }
            | Self::Fec { header, .. } => header,
        }
    }

    /// Byte length of the encoded datagram (header + body).
    pub fn encoded_len(&self) -> usize {
        HEADER_SIZE
            + match self {
                Self::Media { payload, .. } => payload.len(),
                Self::Nack { frame_id, .. } => 4 + usize::from(frame_id.is_some()) * 4,
                Self::ArrivalFeedback {
                    recv_delta_bytes, ..
                } => 10 + recv_delta_bytes.len(),
                Self::KeyframeReq { .. } => 1,
                Self::Fec { payload, .. } => FEC_BODY_HEADER_SIZE + payload.len(),
            }
    }

    /// Serialize this packet into `bytes`.
    ///
    /// Writes the header at offset `0` and the type-specific body immediately
    /// after it. `bytes` must be at least [`Self::encoded_len`] long.
    ///
    /// # Panics
    ///
    /// Panics if `bytes.len() < self.encoded_len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use qrt::core::packet::{Flags, Header, Packet, PacketType};
    ///
    /// let pkt = Packet::Nack {
    ///     header: Header {
    ///         packet_type: PacketType::Nack,
    ///         flags: Flags::default(),
    ///         stream_id: 1,
    ///         media_seq: 0,
    ///         transport_seq: 0,
    ///         frame_id: 0,
    ///         frag_index: 0,
    ///         frag_count: 1,
    ///         timestamp: 0,
    ///         ttl_ms: 50,
    ///     },
    ///     base_seq: 100,
    ///     blp: 0b0101,
    ///     frame_id: Some(9),
    /// };
    ///
    /// let mut wire = vec![0u8; pkt.encoded_len()];
    /// pkt.encode(&mut wire);
    /// let decoded = Packet::decode(&wire).unwrap();
    /// match decoded {
    ///     Packet::Nack {
    ///         base_seq,
    ///         blp,
    ///         frame_id,
    ///         ..
    ///     } => {
    ///         assert_eq!(
    ///             Packet::nack_missing_seqs(base_seq, blp),
    ///             vec![100, 101, 103]
    ///         );
    ///         assert_eq!(frame_id, Some(9));
    ///     }
    ///     _ => panic!("expected nack"),
    /// }
    /// ```
    pub fn encode(&self, bytes: &mut [u8]) {
        assert!(bytes.len() >= self.encoded_len());

        self.header().encode(&mut bytes[..HEADER_SIZE]);
        let mut body = &mut bytes[HEADER_SIZE..];

        match self {
            Self::Media { payload, .. } => {
                body.put_slice(payload);
            }
            Self::Nack {
                base_seq,
                blp,
                frame_id,
                ..
            } => {
                body.put_u16(*base_seq);
                body.put_u16(*blp);

                if let Some(fid) = frame_id {
                    body.put_u32(*fid);
                }
            }
            Self::ArrivalFeedback {
                first_seq,
                received_mask,
                recv_delta_bytes,
                ..
            } => {
                body.put_u16(*first_seq);
                body.put_u64(*received_mask);
                body.put_slice(recv_delta_bytes);
            }
            Self::KeyframeReq { stream_id, .. } => {
                body.put_u8(*stream_id);
            }
            Self::Fec {
                seq_base,
                mask,
                length_xor,
                payload,
                ..
            } => {
                body.put_u16(*seq_base);
                body.put_u64(*mask);
                body.put_u16(*length_xor);
                body.put_slice(payload);
            }
        }
    }

    /// Decode one datagram from `bytes`.
    ///
    /// # Errors
    ///
    /// Propagates [`Header::decode`] errors, and returns
    /// [`DecodeError::TooShort`] when the type-specific body is truncated.
    ///
    /// # Notes
    ///
    /// - Media / FEC `payload` borrows `bytes` after the type-specific prefix.
    /// - NACK bodies shorter than 4 bytes are rejected; an optional trailing
    ///   `frame_id` is accepted when at least 8 body bytes are present.
    pub fn decode(bytes: &'a [u8]) -> Result<Self, DecodeError> {
        if bytes.len() < HEADER_SIZE {
            return Err(DecodeError::TooShort {
                need: HEADER_SIZE,
                have: bytes.len(),
            });
        }

        let header = Header::decode(bytes)?;
        let mut payload = &bytes[HEADER_SIZE..];

        match header.packet_type {
            PacketType::Media => Ok(Self::Media { header, payload }),
            PacketType::Nack => {
                if payload.len() < 4 {
                    return Err(DecodeError::TooShort {
                        need: 4,
                        have: payload.len(),
                    });
                }

                let base_seq = payload.get_u16();
                let blp = payload.get_u16();
                let frame_id = if payload.len() >= 4 {
                    Some(payload.get_u32())
                } else {
                    None
                };

                Ok(Self::Nack {
                    header,
                    base_seq,
                    blp,
                    frame_id,
                })
            }
            PacketType::ArrivalFeedback => {
                if payload.len() < 10 {
                    return Err(DecodeError::TooShort {
                        need: 10,
                        have: payload.len(),
                    });
                }
                let first_seq = payload.get_u16();
                let received_mask = payload.get_u64();
                // Remaining bytes are optional BE u16 recv deltas.
                if payload.len() % 2 != 0 {
                    return Err(DecodeError::TooShort {
                        need: payload.len() + 1,
                        have: payload.len(),
                    });
                }
                Ok(Self::ArrivalFeedback {
                    header,
                    first_seq,
                    received_mask,
                    recv_delta_bytes: payload,
                })
            }
            PacketType::KeyframeReq => {
                if payload.is_empty() {
                    return Err(DecodeError::TooShort { need: 1, have: 0 });
                }

                Ok(Self::KeyframeReq {
                    header,
                    stream_id: payload.get_u8(),
                })
            }
            PacketType::Fec => {
                if payload.len() < FEC_BODY_HEADER_SIZE {
                    return Err(DecodeError::TooShort {
                        need: FEC_BODY_HEADER_SIZE,
                        have: payload.len(),
                    });
                }

                let seq_base = payload.get_u16();
                let mask = payload.get_u64();
                let length_xor = payload.get_u16();
                Ok(Self::Fec {
                    header,
                    seq_base,
                    mask,
                    length_xor,
                    payload,
                })
            }
        }
    }

    /// Encode recv-delta ticks as big-endian bytes for [`Packet::ArrivalFeedback`].
    ///
    /// # Examples
    ///
    /// ```
    /// use qrt::core::packet::Packet;
    /// assert_eq!(
    ///     Packet::encode_arrival_recv_deltas(&[0, 4]),
    ///     vec![0, 0, 0, 4]
    /// );
    /// ```
    pub fn encode_arrival_recv_deltas(deltas: &[u16]) -> Vec<u8> {
        let mut out = Vec::with_capacity(deltas.len() * 2);
        for d in deltas {
            out.extend_from_slice(&d.to_be_bytes());
        }
        out
    }

    /// Parse big-endian recv-delta ticks from [`Packet::ArrivalFeedback`] trailer bytes.
    pub fn parse_arrival_recv_deltas(bytes: &[u8]) -> Vec<u16> {
        bytes
            .chunks_exact(2)
            .map(|c| u16::from_be_bytes([c[0], c[1]]))
            .collect()
    }

    /// Expand a NACK `base_seq` + `blp` into the full list of missing sequences.
    ///
    /// Always includes `base_seq`. For each set bit `i` in `blp` (0..=15), also
    /// includes `base_seq.wrapping_add(i + 1)`. Used with [`Packet::Nack`].
    ///
    /// # Examples
    ///
    /// ```
    /// use qrt::core::packet::Packet;
    /// // bits 0 and 2 => sequences 100, 101, 103
    /// assert_eq!(Packet::nack_missing_seqs(100, 0b0101), vec![100, 101, 103]);
    /// ```
    pub fn nack_missing_seqs(base_seq: u16, blp: u16) -> Vec<u16> {
        let mut out = vec![base_seq];
        for i in 0..16u16 {
            if blp & (1 << i) != 0 {
                out.push(base_seq.wrapping_add(i + 1));
            }
        }
        out
    }

    /// Pack a list of missing sequence numbers into `(base_seq, blp)` entries.
    ///
    /// The input is sorted and deduplicated. Contiguous gaps that fit in a 16-bit
    /// BLP window are merged like RFC 4585 Generic NACK, ready to fill
    /// [`Packet::Nack`] bodies.
    ///
    /// # Examples
    ///
    /// ```
    /// use qrt::core::packet::Packet;
    ///
    /// let packed = Packet::nack_pack_seqs(vec![10, 11, 13, 30]);
    /// assert_eq!(packed.len(), 2);
    /// assert_eq!(
    ///     Packet::nack_missing_seqs(packed[0].0, packed[0].1),
    ///     vec![10, 11, 13]
    /// );
    /// assert_eq!(
    ///     Packet::nack_missing_seqs(packed[1].0, packed[1].1),
    ///     vec![30]
    /// );
    /// ```
    ///
    /// # Notes
    ///
    /// Returns an empty vector when `seqs` is empty.
    pub fn nack_pack_seqs(mut seqs: Vec<u16>) -> Vec<(u16, u16)> {
        if seqs.is_empty() {
            return Vec::new();
        }

        seqs.sort_unstable();
        seqs.dedup();

        let mut out = Vec::new();
        let mut i = 0;
        while i < seqs.len() {
            let base = seqs[i];
            let mut blp = 0u16;
            i += 1;

            while i < seqs.len() {
                let d = seqs[i].wrapping_sub(base);
                if d == 0 || d > 16 {
                    break;
                }

                blp |= 1 << (d - 1);
                i += 1;
            }

            out.push((base, blp));
        }

        out
    }
}
