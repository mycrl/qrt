//! In-memory packet types and a UDP datagram codec.
//!
//! One datagram is one [`Packet`]. The header starts `type` (u8), `size` (u16
//! big-endian byte length of this packet), then transport [`Packet::sequence`]
//! and [`Packet::timestamp`]. [`Packet::from_bytes`] uses `size` to ignore
//! trailing padding and to reject truncated buffers. After the media header,
//! the payload is **whatever bytes remain inside `size`**.
//!
//! Integers are big-endian. [`Packet::sequence`] is the connection-wide
//! transport sequence. Stream-scoped bodies ([`Payload::Stream`]) carry
//! [`Stream::id`]; [`Payload::ArrivalFeedback`] does not.
//!
//! # Examples
//!
//! ```
//! use bytes::Bytes;
//! use qrt::core::packet::{
//!     MediaFragmentPacket,
//!     MediaType,
//!     Packet,
//!     Payload,
//!     Stream,
//!     StreamPacket,
//! };
//!
//! let packet = Packet {
//!     sequence: 42,
//!     timestamp: 90_000,
//!     payload: Payload::Stream(Stream {
//!         id: 1,
//!         idx: 3,
//!         packet: StreamPacket::Media(MediaFragmentPacket {
//!             sequence: 7,
//!             id: 3,
//!             fragment_idx: 0,
//!             fragment_count: 1,
//!             media_type: MediaType::Video,
//!             is_key_frame: true,
//!             payload: Bytes::from_static(b"frame"),
//!         }),
//!     }),
//! };
//!
//! let wire = packet.into_bytes();
//! let decoded = Packet::from_bytes(wire).unwrap();
//!
//! assert_eq!(decoded.sequence, 42);
//! ```

use std::ops::Range;

use bytes::{Buf, BufMut, Bytes, BytesMut};

/// Codec / track kind for a media fragment.
///
/// Use with [`MediaFragmentPacket::is_key_frame`]: a keyframe is still
/// [`MediaType::Video`].
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaType {
    /// Video track fragment.
    Video = 1,
    /// Audio track fragment.
    Audio = 2,
}

/// One media fragment (a slice of an encoded frame).
///
/// [`Self::sequence`] is the per-stream media sequence (NACK identity).
/// [`Self::id`] plus [`Self::fragment_idx`] / [`Self::fragment_count`] reassemble
/// the frame. Retransmits keep [`Self::sequence`] and take a new
/// [`Packet::sequence`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaFragmentPacket {
    /// Per-stream media sequence; unchanged across RTX of this fragment.
    pub sequence: u32,
    /// Frame identity shared by every fragment of the same frame.
    pub id: u32,
    /// Zero-based index of this fragment within the frame.
    pub fragment_idx: u16,
    /// Number of fragments that make up the frame; must be `>= 1`.
    pub fragment_count: u16,
    /// Whether this fragment is video or audio.
    pub media_type: MediaType,
    /// `true` when this fragment belongs to a video keyframe.
    pub is_key_frame: bool,
    /// Opaque codec bytes for this fragment.
    pub payload: Bytes,
}

/// Selective retransmission request (RFC 4585 Generic NACK role).
///
/// Reports missing [`MediaFragmentPacket::sequence`] values on [`Stream::id`]: always
/// `base_media_sequence`, plus `base_media_sequence + 1 + i` for each set bit
/// `i` in [`Self::blp`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NackPacket {
    /// First missing media sequence (PID).
    pub base_media_sequence: u32,
    /// Bitmask of further losses after [`Self::base_media_sequence`].
    pub blp: u16,
}

impl NackPacket {
    /// [`Self::base_media_sequence`] plus each `base + 1 + i` whose bit `i` is
    /// set in [`Self::blp`].
    ///
    /// # Examples
    ///
    /// ```
    /// use qrt::core::packet::NackPacket;
    ///
    /// // bits 0 and 2 => sequences 100, 101, 103
    /// assert_eq!(
    ///     NackPacket {
    ///         base_media_sequence: 100,
    ///         blp: 0b0101,
    ///     }
    ///     .sequences(),
    ///     vec![100, 101, 103]
    /// );
    /// ```
    pub fn sequences(&self) -> Vec<u32> {
        let mut out = vec![self.base_media_sequence];
        for i in 0..16u32 {
            if self.blp & (1 << i) != 0 {
                out.push(self.base_media_sequence.wrapping_add(i + 1));
            }
        }

        out
    }

    /// Pack missing [`MediaFragmentPacket::sequence`] values into RFC 4585
    /// Generic NACK bodies.
    ///
    /// Sorted and deduplicated. Contiguous holes that fit in a 16-bit BLP
    /// window share one packet; a jump larger than 16 starts another.
    ///
    /// # Examples
    ///
    /// ```
    /// use qrt::core::packet::NackPacket;
    ///
    /// let packed = NackPacket::pack(vec![10, 11, 13, 30]);
    ///
    /// assert_eq!(packed.len(), 2);
    /// assert_eq!(packed[0].sequences(), vec![10, 11, 13]);
    /// assert_eq!(packed[1].sequences(), vec![30]);
    /// ```
    ///
    /// # Notes
    ///
    /// Returns an empty vector when `seqs` is empty. Wrap-around clusters may
    /// split because the input is sorted as plain `u32`.
    pub fn pack(mut seqs: Vec<u32>) -> Vec<Self> {
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
                let distance = seqs[i].wrapping_sub(base);
                if distance == 0 || distance > 16 {
                    break;
                }

                blp |= 1 << (distance - 1);
                i += 1;
            }

            out.push(Self {
                base_media_sequence: base,
                blp,
            });
        }

        out
    }
}

/// Per-stream body: media, NACK, or a keyframe request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamPacket {
    /// Encoded media fragment.
    Media(MediaFragmentPacket),
    /// Missing-media retransmission request for this stream.
    Nack(NackPacket),
    /// Ask this stream to produce a keyframe (PLI / FIR role).
    KeyFrameRequest,
}

/// Stream-scoped payload multiplexed on the connection.
///
/// [`Self::idx`] is the frame id used with media; NACK and keyframe requests
/// still carry it on the wire (unused by those types).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stream {
    /// Logical track id.
    pub id: u8,
    /// Frame id (meaningful for [`StreamPacket::Media`]).
    pub idx: u32,
    /// Body for this stream.
    pub packet: StreamPacket,
}

/// Transport-wide arrival report for send-side BWE (TWCC role).
///
/// [`Self::range`] is a half-open transport-sequence window `[start, end)` of
/// at most 64 sequence numbers. Bit `i` of [`Self::received_mask`] means
/// `range.start + i` arrived. [`Self::received`] is the matching recv-delta
/// list (250µs ticks, one `u16` per set bit, low bit first). Sequences in the
/// window with a clear bit are loss samples. This packet is not tied to
/// [`Stream::id`].
///
/// # Notes
///
/// Sequence numbers are not repeated next to each delta: the mask maps each
/// `u16` onto a transport sequence. `recv_delta` is an inter-arrival, not a
/// wall-clock timestamp.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrivalFeedbackPacket {
    /// Half-open transport-sequence window this report covers.
    ///
    /// `end.wrapping_sub(start)` is the width and is at most 64. The window may
    /// cross `u32::MAX`; a plain `start > end` comparison is not "inverted".
    pub range: Range<u32>,
    /// Bit `i` set ⇒ [`Packet::sequence`] `range.start + i` was received.
    pub received_mask: u64,
    /// Recv deltas in 250µs ticks, one per set bit in [`Self::received_mask`].
    pub received: Vec<u16>,
}

/// Datagram body: either a stream message or connection-wide feedback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Payload {
    /// Media / NACK / keyframe request for one [`Stream::id`].
    Stream(Stream),
    /// Connection-wide arrival feedback (not a stream).
    ArrivalFeedback(ArrivalFeedbackPacket),
}

/// One UDP datagram in this protocol.
///
/// [`Self::sequence`] is assigned at send time (transport-wide). Encode with
/// [`Packet::into_bytes`]; decode a full datagram with [`Packet::from_bytes`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Packet {
    /// Connection-wide transport sequence (BWE / arrival feedback).
    pub sequence: u32,
    /// Capture-clock timestamp in 90 kHz ticks for media; `0` on control.
    pub timestamp: u32,
    /// Type-specific body.
    pub payload: Payload,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PacketType {
    Media = 1,
    Nack = 2,
    KeyFrameRequest = 3,
    ArrivalFeedback = 4,
}

/// Failure when parsing a datagram with [`Packet::from_bytes`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PacketError {
    /// Truncated datagram, `size` mismatch, or invalid fragment / arrival-window
    /// fields.
    InvalidPacket,
    /// First byte is not a known [`Packet`] type.
    InvalidPacketType(u8),
    /// Media type byte is not [`MediaType::Video`] or [`MediaType::Audio`].
    InvalidMediaPacketType(u8),
}

impl std::error::Error for PacketError {}

impl std::fmt::Display for PacketError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPacket => write!(f, "invalid packet"),
            Self::InvalidPacketType(byte) => write!(f, "invalid packet type {byte}"),
            Self::InvalidMediaPacketType(byte) => write!(f, "invalid media type {byte}"),
        }
    }
}

impl Packet {
    fn packet_type(&self) -> PacketType {
        match &self.payload {
            Payload::Stream(stream) => match stream.packet {
                StreamPacket::Media(_) => PacketType::Media,
                StreamPacket::Nack(_) => PacketType::Nack,
                StreamPacket::KeyFrameRequest => PacketType::KeyFrameRequest,
            },
            Payload::ArrivalFeedback(_) => PacketType::ArrivalFeedback,
        }
    }

    fn ensure_remaining(bytes: &Bytes, need: usize) -> Result<(), PacketError> {
        if bytes.remaining() < need {
            return Err(PacketError::InvalidPacket);
        }

        Ok(())
    }
}

const COMMON_HEADER_LEN: usize = 1 + 2 + 4 + 4;
const STREAM_PREFIX_LEN: usize = 1 + 4;
const MEDIA_FIELDS_LEN: usize = 4 + 4 + 2 + 2 + 1 + 1;
const NACK_FIELDS_LEN: usize = 4 + 2;
const ARRIVAL_WINDOW_LEN: usize = 4 + 4 + 8;
const ARRIVAL_MASK_BITS: u32 = 64;

impl Packet {
    /// Serialize this packet as one UDP payload.
    ///
    /// Writes a big-endian `size` after the type byte (total length of this
    /// packet). Media bytes follow the media header with no extra length prefix.
    /// Arrival feedback writes `received_mask` then each recv-delta `u16`.
    ///
    /// # Panics
    ///
    /// Panics if the encoded datagram is longer than `u16::MAX` bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use qrt::core::packet::{Packet, PacketError, Payload, Stream, StreamPacket};
    ///
    /// let packet = Packet {
    ///     sequence: 1,
    ///     timestamp: 0,
    ///     payload: Payload::Stream(Stream {
    ///         id: 2,
    ///         idx: 0,
    ///         packet: StreamPacket::KeyFrameRequest,
    ///     }),
    /// };
    ///
    /// let wire = packet.into_bytes();
    /// assert_eq!(Packet::from_bytes(wire).unwrap(), packet);
    ///
    /// assert!(matches!(
    ///     Packet::from_bytes(bytes::Bytes::from_static(&[99])),
    ///     Err(PacketError::InvalidPacket)
    /// ));
    /// ```
    pub fn into_bytes(&self) -> Bytes {
        let mut bytes = BytesMut::with_capacity(2048);

        bytes.put_u8(self.packet_type() as u8);
        bytes.put_u16(0); // packet size placeholder
        bytes.put_u32(self.sequence);
        bytes.put_u32(self.timestamp);

        match &self.payload {
            Payload::Stream(stream) => {
                bytes.put_u8(stream.id);
                bytes.put_u32(stream.idx);

                match &stream.packet {
                    StreamPacket::Media(media) => {
                        bytes.put_u32(media.sequence);
                        bytes.put_u32(media.id);
                        bytes.put_u16(media.fragment_idx);
                        bytes.put_u16(media.fragment_count);
                        bytes.put_u8(media.media_type as u8);
                        bytes.put_u8(u8::from(media.is_key_frame));
                        bytes.put_slice(&media.payload);
                    }
                    StreamPacket::Nack(nack) => {
                        bytes.put_u32(nack.base_media_sequence);
                        bytes.put_u16(nack.blp);
                    }
                    StreamPacket::KeyFrameRequest => {}
                }
            }
            Payload::ArrivalFeedback(feedback) => {
                bytes.put_u32(feedback.range.start);
                bytes.put_u32(feedback.range.end);
                bytes.put_u64(feedback.received_mask);

                for delta in &feedback.received {
                    bytes.put_u16(*delta);
                }
            }
        }

        // write packet size
        {
            let packet_size = u16::try_from(bytes.len()).expect("packet fits in u16 size field");
            bytes[1..3].copy_from_slice(&packet_size.to_be_bytes());
        }

        bytes.freeze()
    }

    /// Parse one complete UDP datagram.
    ///
    /// Reads `size` after the type byte. If the buffer is **longer** than
    /// `size`, trailing bytes are treated as transport padding and ignored. If
    /// the buffer is **shorter** than `size`, the datagram is truncated.
    ///
    /// # Errors
    ///
    /// Returns [`PacketError`] when the buffer is too short, `size` disagrees
    /// with the buffer, the type or media kind is unknown, fragment fields are
    /// inconsistent, the arrival window is empty or wider than 64, or the
    /// arrival mask has bits outside the window. Bytes after a parsed body are
    /// ignored.
    ///
    /// # Notes
    ///
    /// Does not panic on truncated input: every read is preceded by a remaining
    /// check. The media payload is the unparsed tail inside `size` (zero-copy).
    ///
    /// # Examples
    ///
    /// ```
    /// use bytes::Bytes;
    /// use qrt::core::packet::{ArrivalFeedbackPacket, Packet, Payload};
    ///
    /// let packet = Packet {
    ///     sequence: 9,
    ///     timestamp: 0,
    ///     payload: Payload::ArrivalFeedback(ArrivalFeedbackPacket {
    ///         range: 10..13,
    ///         received_mask: 0b101,
    ///         received: vec![0, 8],
    ///     }),
    /// };
    ///
    /// let mut padded = packet.into_bytes().to_vec();
    /// padded.extend_from_slice(&[0xAA, 0xBB]);
    ///
    /// assert_eq!(Packet::from_bytes(Bytes::from(padded)).unwrap(), packet);
    ///
    /// let first = u32::MAX - 2;
    /// let wrapped = Packet {
    ///     sequence: 1,
    ///     timestamp: 0,
    ///     payload: Payload::ArrivalFeedback(ArrivalFeedbackPacket {
    ///         range: first..first.wrapping_add(64),
    ///         received_mask: 1,
    ///         received: vec![0],
    ///     }),
    /// };
    /// assert_eq!(
    ///     Packet::from_bytes(wrapped.clone().into_bytes()).unwrap(),
    ///     wrapped
    /// );
    /// ```
    pub fn from_bytes(mut bytes: Bytes) -> Result<Self, PacketError> {
        Self::ensure_remaining(&bytes, COMMON_HEADER_LEN)?;

        let packet_type = match bytes.get_u8() {
            1 => PacketType::Media,
            2 => PacketType::Nack,
            3 => PacketType::KeyFrameRequest,
            4 => PacketType::ArrivalFeedback,
            byte => return Err(PacketError::InvalidPacketType(byte)),
        };

        let packet_size = usize::from(bytes.get_u16());
        let received_len = 3 + bytes.remaining();
        if packet_size < COMMON_HEADER_LEN || packet_size > received_len {
            return Err(PacketError::InvalidPacket);
        }

        bytes = bytes.slice(0..packet_size - 3);

        let sequence = bytes.get_u32();
        let timestamp = bytes.get_u32();

        let payload = match packet_type {
            PacketType::Media => {
                Self::ensure_remaining(&bytes, STREAM_PREFIX_LEN + MEDIA_FIELDS_LEN)?;

                let stream_id = bytes.get_u8();
                let stream_idx = bytes.get_u32();
                let media_sequence = bytes.get_u32();
                let id = bytes.get_u32();
                let fragment_idx = bytes.get_u16();
                let fragment_count = bytes.get_u16();

                if fragment_count == 0 || fragment_idx >= fragment_count {
                    return Err(PacketError::InvalidPacket);
                }

                let media_type = match bytes.get_u8() {
                    1 => MediaType::Video,
                    2 => MediaType::Audio,
                    byte => return Err(PacketError::InvalidMediaPacketType(byte)),
                };

                let is_key_frame = match bytes.get_u8() {
                    0 => false,
                    1 => true,
                    _ => return Err(PacketError::InvalidPacket),
                };

                Payload::Stream(Stream {
                    id: stream_id,
                    idx: stream_idx,
                    packet: StreamPacket::Media(MediaFragmentPacket {
                        sequence: media_sequence,
                        id,
                        fragment_idx,
                        fragment_count,
                        media_type,
                        is_key_frame,
                        payload: bytes,
                    }),
                })
            }
            PacketType::Nack => {
                Self::ensure_remaining(&bytes, STREAM_PREFIX_LEN + NACK_FIELDS_LEN)?;

                let stream_id = bytes.get_u8();
                let stream_idx = bytes.get_u32();
                let base_media_sequence = bytes.get_u32();
                let blp = bytes.get_u16();

                Payload::Stream(Stream {
                    id: stream_id,
                    idx: stream_idx,
                    packet: StreamPacket::Nack(NackPacket {
                        base_media_sequence,
                        blp,
                    }),
                })
            }
            PacketType::KeyFrameRequest => {
                Self::ensure_remaining(&bytes, STREAM_PREFIX_LEN)?;

                let stream_id = bytes.get_u8();
                let stream_idx = bytes.get_u32();

                Payload::Stream(Stream {
                    id: stream_id,
                    idx: stream_idx,
                    packet: StreamPacket::KeyFrameRequest,
                })
            }
            PacketType::ArrivalFeedback => {
                Self::ensure_remaining(&bytes, ARRIVAL_WINDOW_LEN)?;

                let range_start = bytes.get_u32();
                let range_end = bytes.get_u32();
                let width = range_end.wrapping_sub(range_start);
                if width == 0 || width > ARRIVAL_MASK_BITS {
                    return Err(PacketError::InvalidPacket);
                }

                let allowed_mask = if width == ARRIVAL_MASK_BITS {
                    u64::MAX
                } else {
                    (1u64 << width) - 1
                };

                let received_mask = bytes.get_u64();
                if received_mask & !allowed_mask != 0 {
                    return Err(PacketError::InvalidPacket);
                }

                let delta_count = received_mask.count_ones() as usize;
                Self::ensure_remaining(&bytes, delta_count * 2)?;

                let mut received = Vec::with_capacity(delta_count);
                for _ in 0..delta_count {
                    received.push(bytes.get_u16());
                }

                Payload::ArrivalFeedback(ArrivalFeedbackPacket {
                    range: range_start..range_end,
                    received_mask,
                    received,
                })
            }
        };

        Ok(Packet {
            sequence,
            timestamp,
            payload,
        })
    }
}
