//! XOR forward-error correction (**sans-I/O**), WebRTC ULPFEC / FlexFEC style.
//!
//! Generates [`crate::core::packet::Packet::Fec`] parity over **full Media datagrams**
//! (fixed [`HEADER_SIZE`] header + body) and recovers a missing media packet
//! when a FEC row covers **exactly one** loss. Multiple rows with different
//! masks can repair more than one loss across a block.
//!
//! Aligns with WebRTC `ForwardErrorCorrection` / `UlpfecGenerator` /
//! `UlpfecReceiver` (`modules/rtp_rtcp/source/forward_error_correction.*`),
//! but uses qrt's compact [`Packet::Fec`] body instead of RTP RED / FlexFEC
//! headers. Mask tables are a simple round-robin partition (not the full
//! bursty/random private tables); that is enough for single- and multi-row
//! parity at `kUlpfecMaxMediaPackets = 48`.
//!
//! # Wire reminder
//!
//! ```text
//! Packet::Fec body:
//!   seq_base:u16 | mask:u64 | length_xor:u16 | xor_payload...
//! ```
//!
//! Bit `i` in `mask` protects media with
//! [`Header::media_seq`] `seq_base.wrapping_add(i)`. `length_xor` is the XOR of
//! each protected datagram's length as `u16`. `xor_payload` is the XOR of those
//! datagrams, each zero-padded to `xor_payload.len()`.
//!
//! # Pipeline
//!
//! **Send**
//!
//! 1. After encoding each media UDP datagram, [`FecGenerator::push`] the wire
//!    bytes (same bytes that go on the socket / into history).
//! 2. [`FecGenerator::flush`] (end of frame / protection window) →
//!    [`FecPacketOwned`] rows →enqueue at video priority (`retrans = false`).
//! 3. Assign [`Header::transport_seq`] at pacer egress.
//!
//! **Receive**
//!
//! 1. [`FecReceiver::insert_media`] / [`FecReceiver::insert_fec`] with owned
//!    wire bytes.
//! 2. When a row has exactly one hole, recover the full Media datagram and
//!    feed it to reassembly; mark recovered so NACK skips it.
//!
//! # Examples
//!
//! Protect three media packets with one parity row; drop the middle packet and
//! recover it:
//!
//! ```
//! use bytes::Bytes;
//! use qrt::core::{
//!     fec::{FecGenerator, FecProtectionParams, FecReceiver},
//!     packet::{Flags, HEADER_SIZE, Header, Packet, PacketType},
//! };
//!
//! fn media_wire(seq: u16, payload: &[u8]) -> Bytes {
//!     let pkt = Packet::Media {
//!         header: Header {
//!             packet_type: PacketType::Media,
//!             flags: Flags::default(),
//!             stream_id: 1,
//!             media_seq: seq,
//!             transport_seq: seq,
//!             frame_id: 1,
//!             frag_index: 0,
//!             frag_count: 1,
//!             timestamp: 90_000,
//!             ttl_ms: 100,
//!         },
//!         payload,
//!     };
//!     let mut buf = vec![0u8; pkt.encoded_len()];
//!     pkt.encode(&mut buf);
//!     Bytes::from(buf)
//! }
//!
//! let w0 = media_wire(10, b"aaa");
//! let w1 = media_wire(11, b"bbbb");
//! let w2 = media_wire(12, b"cc");
//!
//! let mut fec_gen = FecGenerator::new(1, FecProtectionParams { fec_rate: 64 });
//! fec_gen.push(10, w0.clone()).unwrap();
//! fec_gen.push(11, w1.clone()).unwrap();
//! fec_gen.push(12, w2.clone()).unwrap();
//! let fec_rows = fec_gen.flush();
//! assert_eq!(fec_rows.len(), 1);
//!
//! let mut rx = FecReceiver::new(64);
//! assert!(rx.insert_media(1, 10, w0).is_empty());
//! assert!(rx.insert_media(1, 12, w2).is_empty());
//! let recovered = rx.insert_fec_owned(&fec_rows[0]);
//! assert_eq!(recovered.len(), 1);
//! assert_eq!(recovered[0].media_seq, 11);
//! assert_eq!(recovered[0].wire, w1);
//! let _ = HEADER_SIZE;
//! ```
//!
//! # Notes
//!
//! - Maximum media packets per protection block: [`MAX_MEDIA_PACKETS`] (48).
//! - FEC packets must **not** be NACK'd or retransmitted.
//! - Networking stays outside this module (sans-I/O).

use std::collections::VecDeque;

use ahash::{HashMap, HashMapExt};
use bytes::Bytes;

use crate::core::packet::{
    DecodeError,
    FEC_BODY_HEADER_SIZE,
    Flags,
    HEADER_SIZE,
    Header,
    Packet,
    PacketType,
};

/// Maximum media packets in one FEC protection block (WebRTC `kUlpfecMaxMediaPackets`).
pub const MAX_MEDIA_PACKETS: usize = 48;

/// Sender-side FEC rate and related knobs (WebRTC `FecProtectionParams` subset).
///
/// # Examples
///
/// ```
/// use qrt::core::fec::FecProtectionParams;
///
/// let p = FecProtectionParams::default();
/// assert!(p.fec_rate > 0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FecProtectionParams {
    /// Relative FEC amount in `0..=255`.
    ///
    /// `num_fec ≈round(num_media * fec_rate / 256)`; if `fec_rate > 0` the
    /// result is at least one packet and at most `num_media`. Cap overhead near
    /// 50% by keeping `fec_rate ≈128` unless you intentionally want denser
    /// parity.
    pub fec_rate: u8,
}

impl Default for FecProtectionParams {
    fn default() -> Self {
        // ~25% overhead (64/255), a common WebRTC starting band for video.
        Self { fec_rate: 64 }
    }
}

/// Computes how many FEC rows to emit for a media block.
///
/// Matches WebRTC's rounding: `(num_media * fec_rate + 128) / 256`, then clamp
/// to `[1, num_media]` when `fec_rate > 0`.
///
/// # Examples
///
/// ```
/// use qrt::core::fec::num_fec_packets;
///
/// assert_eq!(num_fec_packets(4, 0), 0);
/// assert_eq!(num_fec_packets(4, 64), 1);
/// assert_eq!(num_fec_packets(4, 255), 4);
/// ```
pub fn num_fec_packets(num_media: usize, fec_rate: u8) -> usize {
    if num_media == 0 || fec_rate == 0 {
        return 0;
    }
    let n = (num_media * usize::from(fec_rate) + 128) / 256;
    n.max(1).min(num_media)
}

/// Error from FEC encode / receive helpers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FecError {
    /// Wire bytes are not a decodable [`Packet::Media`].
    NotMedia,
    /// [`Header::stream_id`] does not match the generator / expected stream.
    StreamMismatch {
        /// Stream id carried by the packet.
        got: u8,
        /// Stream id expected by the FEC state.
        expected: u8,
    },
    /// Protection block already holds [`MAX_MEDIA_PACKETS`] packets.
    BlockFull,
    /// Media sequences in the block cannot fit in a 48-bit mask window.
    SeqOutOfWindow {
        /// Candidate window base that failed (often the lowest tried seq).
        seq_base: u16,
        /// Offending sequence.
        media_seq: u16,
    },
    /// `media_seq` argument does not match the decoded Media header.
    SeqMismatch {
        /// Sequence passed to [`FecGenerator::push`].
        arg: u16,
        /// Sequence in the Media header.
        header: u16,
    },
    /// Underlying packet decode failure.
    Decode(DecodeError),
}

impl std::fmt::Display for FecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotMedia => write!(f, "datagram is not Packet::Media"),
            Self::StreamMismatch { got, expected } => {
                write!(f, "stream_id {got} != expected {expected}")
            }
            Self::BlockFull => write!(f, "FEC block already has {MAX_MEDIA_PACKETS} media packets"),
            Self::SeqOutOfWindow {
                seq_base,
                media_seq,
            } => {
                write!(
                    f,
                    "media_seq {media_seq} outside 48-bit window from seq_base {seq_base}"
                )
            }
            Self::SeqMismatch { arg, header } => {
                write!(f, "media_seq arg {arg} != header {header}")
            }
            Self::Decode(e) => write!(f, "decode error: {e}"),
        }
    }
}

impl std::error::Error for FecError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Decode(e) => Some(e),
            _ => None,
        }
    }
}

impl From<DecodeError> for FecError {
    fn from(value: DecodeError) -> Self {
        Self::Decode(value)
    }
}

/// Owned FEC parity packet ready to encode onto the wire.
///
/// [`Header::transport_seq`] is left at `0` until the pacer / host assigns a
/// connection-wide transport sequence at send time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FecPacketOwned {
    /// Common header (`packet_type == Fec`, `retrans == false`).
    pub header: Header,
    /// First media sequence covered by [`Self::mask`].
    pub seq_base: u16,
    /// Bit `i` set ⇒protects `seq_base.wrapping_add(i)`.
    pub mask: u64,
    /// XOR of protected Media datagram lengths (`u16`).
    pub length_xor: u16,
    /// XOR of zero-padded Media datagrams.
    pub payload: Bytes,
}

impl FecPacketOwned {
    /// Borrow as a [`Packet::Fec`] view.
    ///
    /// # Examples
    ///
    /// ```
    /// use bytes::Bytes;
    /// use qrt::core::{
    ///     fec::{FecGenerator, FecProtectionParams},
    ///     packet::{Flags, Header, Packet, PacketType},
    /// };
    ///
    /// let pkt = Packet::Media {
    ///     header: Header {
    ///         packet_type: PacketType::Media,
    ///         flags: Flags::default(),
    ///         stream_id: 0,
    ///         media_seq: 1,
    ///         transport_seq: 1,
    ///         frame_id: 0,
    ///         frag_index: 0,
    ///         frag_count: 1,
    ///         timestamp: 0,
    ///         ttl_ms: 50,
    ///     },
    ///     payload: b"x",
    /// };
    /// let mut wire = vec![0u8; pkt.encoded_len()];
    /// pkt.encode(&mut wire);
    /// let mut fec_gen = FecGenerator::new(0, FecProtectionParams { fec_rate: 255 });
    /// fec_gen.push(1, Bytes::from(wire)).unwrap();
    /// let fec = &fec_gen.flush()[0];
    /// assert!(matches!(fec.as_packet(), Packet::Fec { .. }));
    /// ```
    pub fn as_packet(&self) -> Packet<'_> {
        Packet::Fec {
            header: self.header.clone(),
            seq_base: self.seq_base,
            mask: self.mask,
            length_xor: self.length_xor,
            payload: &self.payload,
        }
    }

    /// Encode into a newly allocated datagram buffer.
    pub fn to_wire(&self) -> Bytes {
        let pkt = self.as_packet();
        let mut buf = vec![0u8; pkt.encoded_len()];
        pkt.encode(&mut buf);
        Bytes::from(buf)
    }

    /// Encoded length of the full FEC datagram.
    pub fn encoded_len(&self) -> usize {
        HEADER_SIZE + FEC_BODY_HEADER_SIZE + self.payload.len()
    }
}

/// Media datagram recovered by XOR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveredPacket {
    /// Media stream id from the recovered header.
    pub stream_id: u8,
    /// [`Header::media_seq`] of the recovered Media packet.
    pub media_seq: u16,
    /// Full Media datagram (header + body), ready for reassembly / decode.
    pub wire: Bytes,
}

/// Accumulates Media wires and emits XOR FEC rows on [`Self::flush`].
///
/// Call [`Self::push`] for each media datagram that was (or will be) sent, then
/// [`Self::flush`] at the end of a frame or protection window. Pushing the
/// 49th packet auto-flushes the previous full block first.
#[derive(Debug, Clone)]
pub struct FecGenerator {
    stream_id: u8,
    params: FecProtectionParams,
    /// Pending media: `(media_seq, full wire)`.
    pending: Vec<(u16, Bytes)>,
}

impl FecGenerator {
    /// Creates a generator for one media [`Header::stream_id`].
    ///
    /// # Examples
    ///
    /// ```
    /// use qrt::core::fec::{FecGenerator, FecProtectionParams};
    ///
    /// let fec_gen = FecGenerator::new(2, FecProtectionParams::default());
    /// assert_eq!(fec_gen.stream_id(), 2);
    /// assert_eq!(fec_gen.pending_count(), 0);
    /// ```
    pub fn new(stream_id: u8, params: FecProtectionParams) -> Self {
        Self {
            stream_id,
            params,
            pending: Vec::new(),
        }
    }

    /// Stream this generator protects.
    pub fn stream_id(&self) -> u8 {
        self.stream_id
    }

    /// Current protection parameters.
    pub fn params(&self) -> FecProtectionParams {
        self.params
    }

    /// Updates `fec_rate` (and related knobs). Applies to the next flush.
    pub fn set_params(&mut self, params: FecProtectionParams) {
        self.params = params;
    }

    /// Number of media packets waiting for the next flush.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Adds one full Media datagram to the protection window.
    ///
    /// `wire` must be an encoded [`Packet::Media`] for [`Self::stream_id`].
    /// Sequences should stay within a 48-wide window from the first packet in
    /// the block (`seq_base`).
    ///
    /// Returns any FEC rows produced by an automatic flush when the block was
    /// already full (caller should enqueue them before continuing).
    ///
    /// # Errors
    ///
    /// - [`FecError::NotMedia`] / [`FecError::Decode`] if `wire` is invalid.
    /// - [`FecError::StreamMismatch`] if `stream_id` differs.
    /// - [`FecError::SeqOutOfWindow`] if `media_seq` cannot fit in 48 mask bits.
    ///
    /// # Examples
    ///
    /// See the [module-level example](crate::core::fec).
    pub fn push(&mut self, media_seq: u16, wire: Bytes) -> Result<Vec<FecPacketOwned>, FecError> {
        self.validate_media(media_seq, &wire)?;

        let mut produced = Vec::new();
        if self.pending.len() >= MAX_MEDIA_PACKETS {
            produced = self.flush();
        }

        let mut seqs: Vec<u16> = self.pending.iter().map(|(s, _)| *s).collect();
        seqs.push(media_seq);
        if window_base(&seqs).is_none() {
            let seq_base = seqs.iter().copied().min().unwrap_or(media_seq);
            return Err(FecError::SeqOutOfWindow {
                seq_base,
                media_seq,
            });
        }

        self.pending.push((media_seq, wire));
        Ok(produced)
    }

    /// Builds FEC rows for the pending media and clears the window.
    ///
    /// Uses round-robin masks: FEC row `r` protects indices `j` where
    /// `j % num_fec == r`. With `num_fec == 1` this is a single parity over the
    /// whole block.
    ///
    /// Returns an empty vec when there is nothing to protect or `fec_rate == 0`.
    ///
    /// # Examples
    ///
    /// Two FEC rows can recover two losses that fall on different rows:
    ///
    /// ```
    /// use bytes::Bytes;
    /// use qrt::core::{
    ///     fec::{FecGenerator, FecProtectionParams, FecReceiver},
    ///     packet::{Flags, Header, Packet, PacketType},
    /// };
    ///
    /// fn media_wire(seq: u16, payload: &[u8]) -> Bytes {
    ///     let pkt = Packet::Media {
    ///         header: Header {
    ///             packet_type: PacketType::Media,
    ///             flags: Flags::default(),
    ///             stream_id: 0,
    ///             media_seq: seq,
    ///             transport_seq: seq,
    ///             frame_id: 1,
    ///             frag_index: 0,
    ///             frag_count: 1,
    ///             timestamp: 1,
    ///             ttl_ms: 80,
    ///         },
    ///         payload,
    ///     };
    ///     let mut buf = vec![0u8; pkt.encoded_len()];
    ///     pkt.encode(&mut buf);
    ///     Bytes::from(buf)
    /// }
    ///
    /// let wires: Vec<_> = (0..4u16).map(|s| media_wire(s, &[s as u8; 4])).collect();
    /// let mut fec_gen = FecGenerator::new(0, FecProtectionParams { fec_rate: 128 });
    /// for (s, w) in wires.iter().enumerate() {
    ///     assert!(fec_gen.push(s as u16, w.clone()).unwrap().is_empty());
    /// }
    /// // rate 128 → num_fec = (4*128+128)/256 = 2
    /// let rows = fec_gen.flush();
    /// assert_eq!(rows.len(), 2);
    ///
    /// let mut rx = FecReceiver::new(32);
    /// rx.insert_media(0, 2, wires[2].clone());
    /// rx.insert_media(0, 3, wires[3].clone());
    /// let mut recovered = Vec::new();
    /// for row in &rows {
    ///     recovered.extend(rx.insert_fec_owned(row));
    /// }
    /// let mut seqs: Vec<_> = recovered.iter().map(|r| r.media_seq).collect();
    /// seqs.sort_unstable();
    /// assert_eq!(seqs, vec![0, 1]);
    /// ```
    pub fn flush(&mut self) -> Vec<FecPacketOwned> {
        if self.pending.is_empty() {
            return Vec::new();
        }

        let media = std::mem::take(&mut self.pending);
        let num_fec = num_fec_packets(media.len(), self.params.fec_rate);
        if num_fec == 0 {
            return Vec::new();
        }

        let seqs: Vec<u16> = media.iter().map(|(s, _)| *s).collect();
        let seq_base = window_base(&seqs).unwrap_or(0);
        // Map media_seq →index in mask (relative to seq_base).
        let mut by_bit: [Option<&Bytes>; MAX_MEDIA_PACKETS] = [None; MAX_MEDIA_PACKETS];
        let mut meta_ts = 0u32;
        let mut meta_ttl = u16::MAX;
        let mut any_key = false;
        let mut any_audio = false;
        let mut last_frame_id = 0u32;

        for (seq, wire) in &media {
            let bit = usize::from(seq.wrapping_sub(seq_base));
            debug_assert!(bit < MAX_MEDIA_PACKETS);
            by_bit[bit] = Some(wire);
            if let Ok(Packet::Media { header, .. }) = Packet::decode(wire) {
                meta_ts = meta_ts.max(header.timestamp);
                meta_ttl = meta_ttl.min(header.ttl_ms);
                any_key |= header.flags.key;
                any_audio |= header.flags.audio;
                last_frame_id = header.frame_id;
            }
        }

        let present_bits: Vec<usize> = (0..MAX_MEDIA_PACKETS)
            .filter(|&i| by_bit[i].is_some())
            .collect();

        let mut rows = Vec::with_capacity(num_fec);
        for fec_idx in 0..num_fec {
            let mut mask: u64 = 0;
            let mut members: Vec<usize> = Vec::new();
            for (round, &bit) in present_bits.iter().enumerate() {
                if round % num_fec == fec_idx {
                    mask |= 1u64 << bit;
                    members.push(bit);
                }
            }
            if members.is_empty() {
                continue;
            }

            let mut length_xor = 0u16;
            let mut max_len = 0usize;
            for &bit in &members {
                let w = by_bit[bit].expect("bit present");
                length_xor ^= w.len() as u16;
                max_len = max_len.max(w.len());
            }

            let mut payload = vec![0u8; max_len];
            for &bit in &members {
                xor_bytes(&mut payload, by_bit[bit].expect("bit present"));
            }

            rows.push(FecPacketOwned {
                header: Header {
                    packet_type: PacketType::Fec,
                    flags: Flags {
                        retrans: false,
                        audio: any_audio,
                        key: any_key,
                    },
                    stream_id: self.stream_id,
                    media_seq: 0,
                    transport_seq: 0,
                    frame_id: last_frame_id,
                    frag_index: 0,
                    frag_count: 1,
                    timestamp: meta_ts,
                    ttl_ms: if meta_ttl == u16::MAX { 0 } else { meta_ttl },
                },
                seq_base,
                mask,
                length_xor,
                payload: Bytes::from(payload),
            });
        }

        rows
    }

    fn validate_media(&self, media_seq: u16, wire: &[u8]) -> Result<(), FecError> {
        match Packet::decode(wire)? {
            Packet::Media { header, .. } => {
                if header.stream_id != self.stream_id {
                    return Err(FecError::StreamMismatch {
                        got: header.stream_id,
                        expected: self.stream_id,
                    });
                }
                if header.media_seq != media_seq {
                    return Err(FecError::SeqMismatch {
                        arg: media_seq,
                        header: header.media_seq,
                    });
                }
                Ok(())
            }
            _ => Err(FecError::NotMedia),
        }
    }
}

/// Finds a `seq_base` such that every seq is in `[base, base+48)` (wrapping).
fn window_base(seqs: &[u16]) -> Option<u16> {
    if seqs.is_empty() {
        return Some(0);
    }
    for &base in seqs {
        if seqs
            .iter()
            .all(|&s| s.wrapping_sub(base) < MAX_MEDIA_PACKETS as u16)
        {
            return Some(base);
        }
    }
    None
}

/// Receive-side store of recent Media + FEC for single-loss recovery.
///
/// Insert media and FEC as they arrive (any order). Whenever a FEC row has
/// exactly one missing protected packet, that Media datagram is reconstructed
/// and returned. Cascading recoveries across multiple rows are attempted in a
/// loop.
#[derive(Debug, Clone)]
pub struct FecReceiver {
    /// `(stream_id, media_seq)` →full Media wire.
    media: HashMap<(u8, u16), Bytes>,
    /// Recent FEC rows (oldest first); bounded with media capacity.
    fec: VecDeque<StoredFec>,
    capacity: usize,
}

#[derive(Debug, Clone)]
struct StoredFec {
    stream_id: u8,
    seq_base: u16,
    mask: u64,
    length_xor: u16,
    payload: Bytes,
}

impl FecReceiver {
    /// Creates a receiver that retains up to `capacity` media datagrams
    /// (FIFO eviction of oldest insert order is approximate via clear of half
    /// when over capacity).
    ///
    /// # Examples
    ///
    /// ```
    /// use qrt::core::fec::FecReceiver;
    /// let rx = FecReceiver::new(128);
    /// assert_eq!(rx.media_count(), 0);
    /// ```
    pub fn new(capacity: usize) -> Self {
        Self {
            media: HashMap::new(),
            fec: VecDeque::new(),
            capacity: capacity.max(1),
        }
    }

    /// Number of stored Media datagrams.
    pub fn media_count(&self) -> usize {
        self.media.len()
    }

    /// Number of stored FEC rows.
    pub fn fec_count(&self) -> usize {
        self.fec.len()
    }

    /// Returns whether this media sequence is already known (received or recovered).
    pub fn has_media(&self, stream_id: u8, media_seq: u16) -> bool {
        self.media.contains_key(&(stream_id, media_seq))
    }

    /// Inserts a received Media datagram and tries FEC recovery.
    ///
    /// Duplicates are ignored. Returns any newly recovered Media wires
    /// (possibly several if rows cascade).
    ///
    /// # Examples
    ///
    /// See the [module-level example](crate::core::fec).
    pub fn insert_media(
        &mut self,
        stream_id: u8,
        media_seq: u16,
        wire: Bytes,
    ) -> Vec<RecoveredPacket> {
        let key = (stream_id, media_seq);
        if self.media.contains_key(&key) {
            return Vec::new();
        }
        self.media.insert(key, wire);
        self.evict_if_needed();
        self.recover_all()
    }

    /// Inserts a FEC row from an owned packet and tries recovery.
    pub fn insert_fec_owned(&mut self, fec: &FecPacketOwned) -> Vec<RecoveredPacket> {
        self.insert_fec(
            fec.header.stream_id,
            fec.seq_base,
            fec.mask,
            fec.length_xor,
            fec.payload.clone(),
        )
    }

    /// Inserts a decoded [`Packet::Fec`] (copies the XOR payload).
    ///
    /// # Errors
    ///
    /// Returns [`FecError::NotMedia`] is not used here; pass only FEC packets.
    /// Invalid types should be filtered by the demuxer before calling.
    pub fn insert_fec_packet(
        &mut self,
        packet: &Packet<'_>,
    ) -> Result<Vec<RecoveredPacket>, FecError> {
        match packet {
            Packet::Fec {
                header,
                seq_base,
                mask,
                length_xor,
                payload,
            } => Ok(self.insert_fec(
                header.stream_id,
                *seq_base,
                *mask,
                *length_xor,
                Bytes::copy_from_slice(payload),
            )),
            _ => Err(FecError::NotMedia),
        }
    }

    /// Inserts raw FEC fields and tries recovery.
    pub fn insert_fec(
        &mut self,
        stream_id: u8,
        seq_base: u16,
        mask: u64,
        length_xor: u16,
        payload: Bytes,
    ) -> Vec<RecoveredPacket> {
        self.fec.push_back(StoredFec {
            stream_id,
            seq_base,
            mask,
            length_xor,
            payload,
        });
        // Keep FEC list from growing without bound.
        while self.fec.len() > self.capacity {
            self.fec.pop_front();
        }
        self.recover_all()
    }

    fn recover_all(&mut self) -> Vec<RecoveredPacket> {
        let mut out = Vec::new();
        loop {
            let mut progressed = false;
            // Index snapshot so we can mutate media while scanning fec.
            for i in 0..self.fec.len() {
                let row = self.fec[i].clone();
                if let Some(recovered) = try_recover_row(&row, &self.media) {
                    let key = (recovered.stream_id, recovered.media_seq);
                    if !self.media.contains_key(&key) {
                        self.media.insert(key, recovered.wire.clone());
                        out.push(recovered);
                        progressed = true;
                    }
                }
            }
            if !progressed {
                break;
            }
        }
        if !out.is_empty() {
            self.evict_if_needed();
        }
        out
    }

    fn evict_if_needed(&mut self) {
        if self.media.len() <= self.capacity {
            return;
        }
        // Drop arbitrary oldest-ish half by clearing lowest seq keys first.
        let mut keys: Vec<(u8, u16)> = self.media.keys().copied().collect();
        keys.sort_by(|a, b| {
            a.0.cmp(&b.0).then_with(|| {
                // wrapping-aware: keep higher seqs (recent) when same stream.
                let da = b.1.wrapping_sub(a.1);
                let db = a.1.wrapping_sub(b.1);
                da.cmp(&db)
            })
        });
        let remove_n = self.media.len() - self.capacity;
        for key in keys.into_iter().take(remove_n) {
            self.media.remove(&key);
        }
    }
}

fn xor_bytes(dst: &mut [u8], src: &[u8]) {
    let n = dst.len().min(src.len());
    for i in 0..n {
        dst[i] ^= src[i];
    }
    // Bytes past `src.len()` in `dst` stay as-is (zero pad of src).
}

fn try_recover_row(row: &StoredFec, media: &HashMap<(u8, u16), Bytes>) -> Option<RecoveredPacket> {
    let mut missing_bit: Option<u16> = None;
    let mut missing_count = 0u32;
    let mut length_acc = row.length_xor;
    let mut buf = row.payload.to_vec();

    for bit in 0..MAX_MEDIA_PACKETS as u16 {
        if row.mask & (1u64 << bit) == 0 {
            continue;
        }
        let seq = row.seq_base.wrapping_add(bit);
        match media.get(&(row.stream_id, seq)) {
            Some(wire) => {
                length_acc ^= wire.len() as u16;
                if wire.len() > buf.len() {
                    // Inconsistent FEC / media —cannot recover safely.
                    return None;
                }
                xor_bytes(&mut buf, wire);
            }
            None => {
                missing_count += 1;
                if missing_count > 1 {
                    return None;
                }
                missing_bit = Some(bit);
            }
        }
    }

    if missing_count != 1 {
        return None;
    }
    let bit = missing_bit?;
    let miss_len = usize::from(length_acc);
    if miss_len > buf.len() {
        return None;
    }
    // Residual beyond miss_len should be zero if the FEC row is consistent.
    if buf[miss_len..].iter().any(|&b| b != 0) {
        return None;
    }
    buf.truncate(miss_len);

    // Sanity: recovered bytes must decode as Media with matching seq / stream.
    let media_seq = row.seq_base.wrapping_add(bit);
    match Packet::decode(&buf) {
        Ok(Packet::Media { header, .. })
            if header.stream_id == row.stream_id && header.media_seq == media_seq =>
        {
            Some(RecoveredPacket {
                stream_id: row.stream_id,
                media_seq,
                wire: Bytes::from(buf),
            })
        }
        _ => None,
    }
}
