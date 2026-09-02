//! Codec-opaque frame reassembly for [`Packet::Media`].
//!
//! Counterpart to [`crate::core::fragment`]: groups fragments by
//! `(stream_id, frame_id)`, tolerates reorder, and concatenates payloads in
//! `frag_index` order once `frag_count` slots are filled.
//!
//! Inspired by WebRTC's receive-side packet → frame path
//! (`modules/video_coding/packet_buffer.*`, `video/rtp_video_stream_receiver2.*`),
//! but keyed on explicit [`Header::frag_index`] / [`Header::frag_count`] instead
//! of RTP marker bits or codec FU headers.
//!
//! # Pipeline
//!
//! 1. [`FrameReassembler::push`] — copy one media fragment into an incomplete slot.
//! 2. When all indices `[0, frag_count)` are present → [`InsertOutcome::Assembled`].
//! 3. Late duplicates / already-finished frames are reported without panicking.
//!
//! # Examples
//!
//! ```
//! use qrt::core::{
//!     fragment::{DEFAULT_MAX_PAYLOAD_LEN, FragmentParams, PayloadSizeLimits, fragment},
//!     packet::Flags,
//!     reassembly::FrameReassembler,
//! };
//!
//! let frame = vec![9u8; DEFAULT_MAX_PAYLOAD_LEN + 50];
//! let packets = fragment(
//!     &frame,
//!     &FragmentParams {
//!         stream_id: 1,
//!         frame_id: 3,
//!         timestamp: 90_000,
//!         ttl_ms: 100,
//!         flags: Flags {
//!             key: true,
//!             ..Flags::default()
//!         },
//!         first_media_seq: 0,
//!         first_transport_seq: 0,
//!     },
//!     &PayloadSizeLimits::default(),
//! )
//! .unwrap();
//!
//! let mut reasm = FrameReassembler::new();
//! // Arrive out of order: last fragment first.
//! assert!(reasm.push(&packets[1]).unwrap().is_incomplete());
//! let done = reasm.push(&packets[0]).unwrap();
//! let assembled = done.into_assembled().expect("frame complete");
//! assert_eq!(assembled.payload.as_ref(), frame.as_slice());
//! assert_eq!(assembled.frame_id, 3);
//! assert!(assembled.flags.key);
//! ```
//!
//! # Notes
//!
//! - Payload bytes are **copied** on insert so the reassembler does not borrow
//!   the UDP receive buffer.
//! - Incomplete frames are capped by [`FrameReassembler::max_incomplete_frames`];
//!   overflow drops the oldest incomplete frame (insertion order).
//! - Deadline / jitter drop of whole frames belongs in [`crate::core::jitter`]; this
//!   type only assembles.

use ahash::{HashMap, HashMapExt};
use bytes::Bytes;

use crate::core::packet::{Flags, Packet};

/// Default cap on concurrent incomplete frames across all streams.
pub const DEFAULT_MAX_INCOMPLETE_FRAMES: usize = 64;

/// Default how many recently completed `(stream_id, frame_id)` keys to remember
/// so late fragments are ignored as duplicates.
pub const DEFAULT_COMPLETED_HISTORY: usize = 128;

/// A fully reassembled encoded frame (codec-opaque).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssembledFrame {
    /// [`crate::core::packet::Header::stream_id`].
    pub stream_id: u8,
    /// [`crate::core::packet::Header::frame_id`].
    pub frame_id: u32,
    /// [`crate::core::packet::Header::timestamp`] shared by all fragments.
    pub timestamp: u32,
    /// Flags taken from the first accepted fragment (typically frag 0).
    pub flags: Flags,
    /// [`crate::core::packet::Header::media_seq`] of fragment 0 when that slot was received.
    pub first_media_seq: Option<u16>,
    /// Concatenated fragment payloads in index order.
    pub payload: Bytes,
}

/// Result of pushing one media packet into the reassembler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InsertOutcome {
    /// Frame still missing fragments.
    Incomplete {
        /// Distinct fragment indices stored so far.
        have: u16,
        /// Expected [`crate::core::packet::Header::frag_count`].
        need: u16,
    },
    /// All fragments present; frame removed from the incomplete map.
    Assembled(AssembledFrame),
    /// Same `frag_index` already stored for this frame.
    DuplicateFragment,
    /// This `(stream_id, frame_id)` was already assembled recently.
    DuplicateFrame,
}

impl InsertOutcome {
    /// Returns `true` if the frame is still incomplete.
    pub fn is_incomplete(&self) -> bool {
        matches!(self, Self::Incomplete { .. })
    }

    /// Returns the assembled frame when this push completed it.
    pub fn into_assembled(self) -> Option<AssembledFrame> {
        match self {
            Self::Assembled(frame) => Some(frame),
            _ => None,
        }
    }
}

/// Error when a packet cannot be accepted into reassembly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReassemblyError {
    /// Only [`Packet::Media`] can be reassembled.
    NotMedia,
    /// `frag_count` / `timestamp` / conflicting slot metadata disagree with the
    /// incomplete frame already started.
    Conflict(&'static str),
}

impl std::fmt::Display for ReassemblyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotMedia => write!(f, "packet is not Media"),
            Self::Conflict(why) => write!(f, "reassembly conflict: {why}"),
        }
    }
}

impl std::error::Error for ReassemblyError {}

#[derive(Debug)]
struct IncompleteFrame {
    frag_count: u16,
    timestamp: u32,
    flags: Flags,
    slots: Vec<Option<Bytes>>,
    received: u16,
    first_media_seq: Option<u16>,
    insert_order: u64,
}

impl IncompleteFrame {
    fn new(
        frag_count: u16,
        timestamp: u32,
        flags: Flags,
        insert_order: u64,
    ) -> Result<Self, ReassemblyError> {
        if frag_count == 0 {
            return Err(ReassemblyError::Conflict("frag_count is 0"));
        }
        let mut slots = Vec::with_capacity(frag_count as usize);
        slots.resize(frag_count as usize, None);
        Ok(Self {
            frag_count,
            timestamp,
            flags,
            slots,
            received: 0,
            first_media_seq: None,
            insert_order,
        })
    }

    fn try_finish(&mut self, stream_id: u8, frame_id: u32) -> Option<AssembledFrame> {
        if self.received != self.frag_count {
            return None;
        }
        let mut out = Vec::new();
        for slot in &self.slots {
            out.extend_from_slice(slot.as_ref().expect("slot filled when received == count"));
        }
        Some(AssembledFrame {
            stream_id,
            frame_id,
            timestamp: self.timestamp,
            flags: self.flags,
            first_media_seq: self.first_media_seq,
            payload: Bytes::from(out),
        })
    }
}

/// Reassembles [`Packet::Media`] fragments into [`AssembledFrame`] values.
///
/// # Examples
///
/// See the [module-level example](crate::core::reassembly).
///
/// # Notes
///
/// Call [`Self::push`] once per received media datagram (including FEC-recovered
/// copies). Non-media packets must not be passed in.
#[derive(Debug)]
pub struct FrameReassembler {
    frames: HashMap<(u8, u32), IncompleteFrame>,
    completed: HashMap<(u8, u32), ()>,
    completed_order: std::collections::VecDeque<(u8, u32)>,
    max_incomplete_frames: usize,
    completed_history: usize,
    next_insert_order: u64,
}

impl Default for FrameReassembler {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameReassembler {
    /// Create a reassembler with default incomplete / history caps.
    pub fn new() -> Self {
        Self::with_limits(DEFAULT_MAX_INCOMPLETE_FRAMES, DEFAULT_COMPLETED_HISTORY)
    }

    /// Create a reassembler with explicit resource caps.
    ///
    /// # Panics
    ///
    /// Panics if `max_incomplete_frames == 0`.
    pub fn with_limits(max_incomplete_frames: usize, completed_history: usize) -> Self {
        assert!(max_incomplete_frames > 0);
        Self {
            frames: HashMap::new(),
            completed: HashMap::new(),
            completed_order: std::collections::VecDeque::new(),
            max_incomplete_frames,
            completed_history,
            next_insert_order: 0,
        }
    }

    /// Maximum number of concurrent incomplete frames.
    pub fn max_incomplete_frames(&self) -> usize {
        self.max_incomplete_frames
    }

    /// Number of frames waiting for more fragments.
    pub fn incomplete_len(&self) -> usize {
        self.frames.len()
    }

    /// Insert one media fragment.
    ///
    /// # Errors
    ///
    /// - [`ReassemblyError::NotMedia`] if `packet` is not [`Packet::Media`].
    /// - [`ReassemblyError::Conflict`] if metadata disagrees with an existing
    ///   incomplete frame (different `frag_count` / `timestamp`, or
    ///   `frag_index` out of range — should already be rejected by decode).
    ///
    /// # Examples
    ///
    /// ```
    /// use qrt::core::{
    ///     packet::{Flags, Header, Packet, PacketType},
    ///     reassembly::FrameReassembler,
    /// };
    ///
    /// let pkt = Packet::Media {
    ///     header: Header {
    ///         packet_type: PacketType::Media,
    ///         flags: Flags::default(),
    ///         stream_id: 0,
    ///         media_seq: 5,
    ///         transport_seq: 9,
    ///         frame_id: 1,
    ///         frag_index: 0,
    ///         frag_count: 1,
    ///         timestamp: 0,
    ///         ttl_ms: 50,
    ///     },
    ///     payload: b"one",
    /// };
    ///
    /// let mut reasm = FrameReassembler::new();
    /// let frame = reasm.push(&pkt).unwrap().into_assembled().unwrap();
    /// assert_eq!(frame.payload.as_ref(), b"one");
    /// assert_eq!(frame.first_media_seq, Some(5));
    /// ```
    pub fn push(&mut self, packet: &Packet<'_>) -> Result<InsertOutcome, ReassemblyError> {
        let (header, payload) = match packet {
            Packet::Media { header, payload } => (header, *payload),
            _ => return Err(ReassemblyError::NotMedia),
        };

        let key = (header.stream_id, header.frame_id);
        if self.completed.contains_key(&key) {
            return Ok(InsertOutcome::DuplicateFrame);
        }

        if header.frag_index >= header.frag_count {
            return Err(ReassemblyError::Conflict("frag_index >= frag_count"));
        }

        if !self.frames.contains_key(&key) {
            self.evict_if_needed();
            let order = self.next_insert_order;
            self.next_insert_order = self.next_insert_order.wrapping_add(1);
            let incomplete =
                IncompleteFrame::new(header.frag_count, header.timestamp, header.flags, order)?;
            self.frames.insert(key, incomplete);
        }

        let frame = self.frames.get_mut(&key).expect("just inserted or present");

        if frame.frag_count != header.frag_count {
            return Err(ReassemblyError::Conflict("frag_count mismatch"));
        }
        if frame.timestamp != header.timestamp {
            return Err(ReassemblyError::Conflict("timestamp mismatch"));
        }
        if frame.slots.len() != header.frag_count as usize {
            return Err(ReassemblyError::Conflict("internal slot size"));
        }

        let idx = header.frag_index as usize;
        if frame.slots[idx].is_some() {
            return Ok(InsertOutcome::DuplicateFragment);
        }

        frame.slots[idx] = Some(Bytes::copy_from_slice(payload));
        frame.received = frame.received.saturating_add(1);
        if header.frag_index == 0 {
            frame.first_media_seq = Some(header.media_seq);
            // Prefer flags from the first fragment when it arrives (may be late).
            frame.flags = header.flags;
        }

        if frame.received != frame.frag_count {
            return Ok(InsertOutcome::Incomplete {
                have: frame.received,
                need: frame.frag_count,
            });
        }

        let mut finished = self.frames.remove(&key).expect("present");
        let assembled = finished
            .try_finish(header.stream_id, header.frame_id)
            .expect("received == frag_count");

        self.remember_completed(key);

        Ok(InsertOutcome::Assembled(assembled))
    }

    /// Drop all incomplete state (completed history is kept).
    pub fn clear_incomplete(&mut self) {
        self.frames.clear();
    }

    /// Drop incomplete frames for one media stream.
    pub fn clear_stream(&mut self, stream_id: u8) {
        self.frames.retain(|&(sid, _), _| sid != stream_id);
    }

    fn remember_completed(&mut self, key: (u8, u32)) {
        if self.completed_history == 0 {
            return;
        }
        if self.completed.insert(key, ()).is_none() {
            self.completed_order.push_back(key);
        }
        while self.completed_order.len() > self.completed_history {
            if let Some(old) = self.completed_order.pop_front() {
                self.completed.remove(&old);
            }
        }
    }

    fn evict_if_needed(&mut self) {
        while self.frames.len() >= self.max_incomplete_frames {
            let victim = self
                .frames
                .iter()
                .min_by_key(|(_, f)| f.insert_order)
                .map(|(k, _)| *k);
            if let Some(key) = victim {
                self.frames.remove(&key);
            } else {
                break;
            }
        }
    }
}
