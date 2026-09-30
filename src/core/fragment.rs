//! Frame split and reassembly for a **fixed** header.
//!
//! [`Fragment::split`] turns a [`MediaPacket`] (one encoded frame) into
//! [`MediaFragmentPacket`] slices. [`Reassembly::forward`] is the inverse:
//! fragments of one [`MediaPacket::id`] are stored until `fragment_count`
//! slots are filled, then concatenated in [`MediaFragmentPacket::fragment_idx`]
//! order.
//!
//! Unlike WebRTC `PayloadSizeLimits`, there are no first/last/single-packet
//! reductions: this stack's header length does not change across fragments.
//! Use one [`Reassembly`] per stream — fragments have no stream id.
//! Assembled and [`Reassembly::remove`]d ids stay retired for
//! [`RETIRED_HISTORY`] entries so a late fragment cannot reopen the frame.
//!
//! # Examples
//!
//! ```
//! use bytes::Bytes;
//! use qrt::core::{
//!     fragment::{Fragment, MediaPacket, Reassembly},
//!     packet::MediaType,
//! };
//!
//! let parts = Fragment::default().split(MediaPacket {
//!     id: 3,
//!     media_type: MediaType::Video,
//!     is_key_frame: true,
//!     payload: Bytes::from(vec![0u8; 2500]),
//! });
//!
//! assert_eq!(parts.len(), 3);
//! assert_eq!(
//!     parts.iter().map(|p| p.payload.len()).collect::<Vec<_>>(),
//!     vec![833, 833, 834]
//! );
//!
//! let mut reassembly = Reassembly::new();
//! let mut assembled = None;
//!
//! for part in parts.into_iter().rev() {
//!     assembled = reassembly.forward(part);
//! }
//!
//! assert_eq!(assembled.unwrap().payload.len(), 2500);
//! ```

use std::collections::{HashMap, HashSet, VecDeque};

use bytes::{Bytes, BytesMut};

use super::packet::{MediaFragmentPacket, MediaType};

/// Default media-body budget (WebRTC `kVideoMtu`), not the full UDP MTU.
pub const DEFAULT_MAX_PACKET_SIZE: usize = 1200;

/// How many frames may sit incomplete at once. A new `id` past this cap is ignored
/// until the host [`Reassembly::remove`]s a stale frame.
pub const MAX_INCOMPLETE_FRAMES: usize = 64;

/// Reject a fragment whose `fragment_count` is larger than this.
///
/// [`Fragment::split`] uses the same cap. A frame bigger than this times
/// [`Fragment::max_packet_size`] is not sent: split returns an empty list, and
/// the engine returns [`crate::engine::EngineError::Fragment`] before it
/// assigns media sequence numbers or queues the frame.
pub const MAX_FRAGMENTS_PER_FRAME: u16 = 256;

/// Suggested age after which the host should [`Reassembly::remove`] a hole
/// (WebRTC / qrt packet lifetime). This type does not apply the timer itself.
pub const DEFAULT_FRAME_LIFETIME: std::time::Duration = std::time::Duration::from_millis(200);

/// How many recently retired frame ids to remember.
///
/// Assembled frames and [`Reassembly::remove`]d frames share this ring. A
/// fragment whose `id` is still in the ring is dropped instead of starting a
/// new incomplete entry. Older ids fall out so `u32` wrap-around can reuse
/// them.
///
/// # Examples
///
/// ```
/// use bytes::Bytes;
/// use qrt::core::{
///     fragment::{RETIRED_HISTORY, Reassembly},
///     packet::{MediaFragmentPacket, MediaType},
/// };
///
/// fn one_byte(id: u32) -> MediaFragmentPacket {
///     MediaFragmentPacket {
///         sequence: id,
///         id,
///         fragment_idx: 0,
///         fragment_count: 1,
///         media_type: MediaType::Video,
///         is_key_frame: false,
///         payload: Bytes::from_static(b"x"),
///     }
/// }
///
/// let mut reassembly = Reassembly::new();
/// reassembly.remove(1);
///
/// for id in 2..=RETIRED_HISTORY as u32 + 1 {
///     assert!(reassembly.forward(one_byte(id)).is_some());
/// }
///
/// assert!(reassembly.forward(one_byte(1)).is_some());
/// ```
pub const RETIRED_HISTORY: usize = 128;

/// A fully reassembled encoded frame (codec-opaque).
///
/// Input to [`Fragment::split`] and output of [`Reassembly::forward`]. `payload`
/// is the whole frame; fragments share [`Self::id`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaPacket {
    /// Frame identity shared by every fragment of the same frame.
    pub id: u32,
    /// Whether this frame is video or audio.
    pub media_type: MediaType,
    /// `true` when this frame is a video keyframe (taken from fragment 0 when
    /// that slot arrives).
    pub is_key_frame: bool,
    /// Concatenated codec bytes for the whole frame.
    pub payload: Bytes,
}

/// How to cut one encoded frame into datagram payloads.
///
/// [`Self::max_packet_size`] is the **media payload** budget per datagram after
/// the fixed header, not IP+UDP+header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fragment {
    /// Maximum media payload bytes in any fragment.
    pub max_packet_size: usize,
}

impl Default for Fragment {
    fn default() -> Self {
        Self {
            max_packet_size: DEFAULT_MAX_PACKET_SIZE,
        }
    }
}

impl Fragment {
    /// Split one encoded frame into datagram-sized [`MediaFragmentPacket`]s.
    ///
    /// Fits in one fragment when `packet.payload` is at most `max_packet_size`.
    /// Otherwise uses `ceil(len / max)` pieces; remainder bytes go on the
    /// **last** fragments (any two pieces differ by at most one byte). Empty
    /// payload, `max_packet_size == 0`, or more than `u16::MAX` fragments yield
    /// an empty vec.
    ///
    /// Each fragment copies [`MediaPacket::id`] / media type / keyframe flag.
    /// [`MediaFragmentPacket::sequence`] is `0..frag_count` for this split; the
    /// sender should stamp the per-stream media sequence before send.
    ///
    /// # Examples
    ///
    /// ```
    /// use bytes::Bytes;
    /// use qrt::core::{
    ///     fragment::{Fragment, MediaPacket},
    ///     packet::MediaType,
    /// };
    ///
    /// let parts = Fragment {
    ///     max_packet_size: 10,
    /// }
    /// .split(MediaPacket {
    ///     id: 1,
    ///     media_type: MediaType::Video,
    ///     is_key_frame: false,
    ///     payload: Bytes::from_static(b"abcdefghijKLMN"),
    /// });
    ///
    /// assert_eq!(parts.len(), 2);
    /// assert_eq!(parts[0].fragment_idx, 0);
    /// assert_eq!(parts[1].fragment_count, 2);
    /// assert_eq!(parts[0].payload.as_ref(), b"abcdefg");
    /// assert_eq!(parts[1].payload.as_ref(), b"hijKLMN");
    ///
    /// // 257 pieces is past [`MAX_FRAGMENTS_PER_FRAME`]. Nothing is queued.
    /// assert!(
    ///     Fragment { max_packet_size: 1 }
    ///         .split(MediaPacket {
    ///             id: 2,
    ///             media_type: MediaType::Video,
    ///             is_key_frame: false,
    ///             payload: Bytes::from(vec![0u8; 257]),
    ///         })
    ///         .is_empty()
    /// );
    /// ```
    pub fn split(&self, packet: MediaPacket) -> Vec<MediaFragmentPacket> {
        if packet.payload.is_empty() || self.max_packet_size == 0 {
            return Vec::new();
        }

        if packet.payload.len() <= self.max_packet_size {
            return vec![MediaFragmentPacket {
                sequence: 0,
                id: packet.id,
                fragment_idx: 0,
                fragment_count: 1,
                media_type: packet.media_type,
                is_key_frame: packet.is_key_frame,
                payload: packet.payload,
            }];
        }

        let packet_count = packet.payload.len().div_ceil(self.max_packet_size);
        if packet_count > usize::from(MAX_FRAGMENTS_PER_FRAME) {
            return Vec::new();
        }

        let fragment_count = packet_count as u16;
        let mut fragments = Vec::with_capacity(packet_count);
        let mut offset = 0;
        let base = packet.payload.len() / packet_count;
        let extra = packet.payload.len() % packet_count;

        for i in 0..packet_count {
            let size = if i < packet_count - extra {
                base
            } else {
                base + 1
            };

            let end = offset + size;
            fragments.push(MediaFragmentPacket {
                sequence: i as u32,
                id: packet.id,
                fragment_idx: i as u16,
                fragment_count,
                media_type: packet.media_type,
                is_key_frame: packet.is_key_frame,
                payload: packet.payload.slice(offset..end),
            });

            offset = end;
        }

        fragments
    }
}

/// Receive-side counterpart to [`Fragment::split`].
///
/// One [`HashMap`] from frame id to a compact slot `Vec`. Each slot holds only
/// the fragment payload (`Bytes` is refcounted). [`Self::forward`] fills a slot;
/// when every index is present the entry is removed and concatenated.
///
/// Dropping a frame that will never complete (NACK given up after FEC cannot
/// repair) is the host's job: call [`Self::remove`]. This type does not run a
/// timer. Retired ids (assembled or abandoned) reject late fragments so a
/// leftover piece cannot reopen the frame.
///
/// # Notes
///
/// Create one instance per stream. The first fragment of an `id` freezes
/// `fragment_count` and [`MediaType`]. New ids are refused while
/// [`MAX_INCOMPLETE_FRAMES`] incomplete frames are already held.
#[derive(Debug)]
pub struct Reassembly {
    frames: HashMap<u32, IncompleteFrame>,
    retired: HashSet<u32>,
    retired_order: VecDeque<u32>,
}

#[derive(Debug)]
struct IncompleteFrame {
    fragment_count: u16,
    media_type: MediaType,
    is_key_frame: bool,
    slots: Vec<Option<Bytes>>,
    received: u16,
}

impl Default for Reassembly {
    fn default() -> Self {
        Self::new()
    }
}

impl Reassembly {
    /// Create an empty reassembler.
    pub fn new() -> Self {
        Self {
            frames: HashMap::new(),
            retired: HashSet::new(),
            retired_order: VecDeque::new(),
        }
    }

    /// Abandon a frame and refuse later fragments of the same [`MediaPacket::id`].
    ///
    /// Call this when NACK gives up a fragment (which abandons the whole frame)
    /// after FEC can no longer repair it. The incomplete slots are dropped and
    /// `id` is retired: a late original, RTX, or FEC-recovered fragment of this
    /// frame will not start a new entry.
    ///
    /// Unknown ids are still retired, so a frame that never received any
    /// fragment cannot appear later as a zombie. Already-retired ids are
    /// unchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// use bytes::Bytes;
    /// use qrt::core::{
    ///     fragment::{Fragment, MediaPacket, Reassembly},
    ///     packet::MediaType,
    /// };
    ///
    /// let parts = Fragment {
    ///     max_packet_size: 10,
    /// }
    /// .split(MediaPacket {
    ///     id: 7,
    ///     media_type: MediaType::Video,
    ///     is_key_frame: false,
    ///     payload: Bytes::from_static(b"abcdefghijKLMN"),
    /// });
    /// let mut reassembly = Reassembly::new();
    ///
    /// assert!(reassembly.forward(parts[0].clone()).is_none());
    ///
    /// reassembly.remove(7);
    ///
    /// assert!(reassembly.forward(parts[1].clone()).is_none());
    /// assert!(reassembly.forward(parts[0].clone()).is_none());
    /// ```
    ///
    /// An `id` that never received a fragment is still retired:
    ///
    /// ```
    /// use bytes::Bytes;
    /// use qrt::core::{
    ///     fragment::{Fragment, MediaPacket, Reassembly},
    ///     packet::MediaType,
    /// };
    ///
    /// let parts = Fragment {
    ///     max_packet_size: 10,
    /// }
    /// .split(MediaPacket {
    ///     id: 4,
    ///     media_type: MediaType::Video,
    ///     is_key_frame: false,
    ///     payload: Bytes::from_static(b"abcdefghijKLMN"),
    /// });
    /// let mut reassembly = Reassembly::new();
    ///
    /// reassembly.remove(4);
    ///
    /// assert!(reassembly.forward(parts[0].clone()).is_none());
    /// ```
    pub fn remove(&mut self, id: u32) {
        self.frames.remove(&id);
        self.remember_retired(id);
    }

    /// Insert one media fragment.
    ///
    /// Returns [`Some`] when this push completed `packet.id`. Duplicates,
    /// illegal fragment fields, a retired `id` (already assembled or
    /// [`Self::remove`]d), a layout mismatch with the frozen frame, or a new
    /// `id` while at [`MAX_INCOMPLETE_FRAMES`] return [`None`].
    ///
    /// # Examples
    ///
    /// ```
    /// use bytes::Bytes;
    /// use qrt::core::{
    ///     fragment::{Fragment, MediaPacket, Reassembly},
    ///     packet::MediaType,
    /// };
    ///
    /// let parts = Fragment {
    ///     max_packet_size: 10,
    /// }
    /// .split(MediaPacket {
    ///     id: 7,
    ///     media_type: MediaType::Video,
    ///     is_key_frame: true,
    ///     payload: Bytes::from_static(b"abcdefghijKLMN"),
    /// });
    /// let mut reassembly = Reassembly::new();
    ///
    /// assert!(reassembly.forward(parts[1].clone()).is_none());
    ///
    /// let assembled = reassembly.forward(parts[0].clone()).unwrap();
    ///
    /// assert_eq!(assembled.payload.as_ref(), b"abcdefghijKLMN");
    /// assert!(assembled.is_key_frame);
    /// assert!(reassembly.forward(parts[0].clone()).is_none());
    /// ```
    pub fn forward(&mut self, packet: MediaFragmentPacket) -> Option<MediaPacket> {
        if packet.fragment_count == 0
            || packet.fragment_count > MAX_FRAGMENTS_PER_FRAME
            || packet.fragment_idx >= packet.fragment_count
        {
            return None;
        }

        if self.retired.contains(&packet.id) {
            return None;
        }

        if !self.frames.contains_key(&packet.id) {
            if self.frames.len() >= MAX_INCOMPLETE_FRAMES {
                return None;
            }

            self.frames.insert(
                packet.id,
                IncompleteFrame {
                    fragment_count: packet.fragment_count,
                    media_type: packet.media_type,
                    is_key_frame: packet.is_key_frame,
                    slots: vec![None; usize::from(packet.fragment_count)],
                    received: 0,
                },
            );
        }

        let frame = self.frames.get_mut(&packet.id)?;
        if frame.fragment_count != packet.fragment_count || frame.media_type != packet.media_type {
            return None;
        }

        let index = usize::from(packet.fragment_idx);
        if frame.slots[index].is_some() {
            return None;
        }

        frame.slots[index] = Some(packet.payload);
        frame.received += 1;

        if packet.fragment_idx == 0 {
            frame.is_key_frame = packet.is_key_frame;
        }

        if frame.received != frame.fragment_count {
            return None;
        }

        let finished = self.frames.remove(&packet.id)?;
        self.remember_retired(packet.id);
        let mut payload = BytesMut::new();

        for slot in finished.slots {
            payload.extend_from_slice(&slot.expect("slot filled when received == count"));
        }

        Some(MediaPacket {
            id: packet.id,
            media_type: finished.media_type,
            is_key_frame: finished.is_key_frame,
            payload: payload.freeze(),
        })
    }

    fn remember_retired(&mut self, id: u32) {
        if !self.retired.insert(id) {
            return;
        }

        self.retired_order.push_back(id);
        while self.retired_order.len() > RETIRED_HISTORY {
            if let Some(old) = self.retired_order.pop_front() {
                self.retired.remove(&old);
            }
        }
    }
}
