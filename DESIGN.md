# QRT Technical Design

## 1. Purpose and scope

QRT is a low-latency transport for encoded audio and video between controlled
peers. It sends media and transport feedback over one UDP flow and uses a
separate TCP connection for session and track control.

The design favors timeliness over complete delivery:

- frames are codec-opaque units, not a byte stream;
- queued and stored media expire instead of being delivered indefinitely;
- missing fragments may be selectively retransmitted while still useful;
- one connection-wide bandwidth estimator controls all tracks;
- receive-side jitter buffers trade a bounded delay for smoother playout.

QRT borrows algorithmic ideas from WebRTC, including pacing, Generic
NACK-style loss reports, transport-wide arrival feedback, delay/loss-based
bandwidth estimation, and jitter buffering. Its wire format is proprietary.
It is not compatible with RTP, RTCP, WebRTC, or QUIC.

QRT does not provide codecs, capture, rendering, ICE, STUN/TURN, NAT
traversal, multipath, or browser interoperability. The current endpoint model
assumes that each peer can receive UDP at the port announced over TCP.

## 2. Architecture

The crate has two usable layers.

### 2.1 Session layer

`Client`, `Server`, `Session`, `SendTrack`, and `RecvTrack` form the primary
Tokio API. This layer owns:

- the TCP or TLS signal connection;
- one connected UDP socket per session;
- an `Engine` driven by one Tokio task;
- track negotiation and stream-id allocation;
- frame encryption when TLS mode is selected;
- bounded application receive queues and encoder-control routing.

`Client::connect` creates one session. `Server::accept` creates one independent
session for each accepted TCP connection. Client and server roles end after
connection setup: both resulting sessions are full duplex.

### 2.2 Engine and core layer

`engine::Engine` is a synchronous, socket-free state machine. A custom host
can call:

- `push_frame(frame, now)` when an encoder produces a frame;
- `push_packet(datagram, now)` when UDP receives a datagram;
- `tick(now)` when `TaskResult::next_wake` becomes due.

Each call returns a `TaskResult` containing ordered `EngineEvent` values:

- `Packet`: a UDP payload ready to send;
- `Frame`: a complete frame ready for the decoder;
- `RateChange`: a new encoder target;
- `KeyframeRequest`: the next encoded video frame should be an IDR;
- `Conceal`: an audio playout tick had no packet and needs PLC.

All `Instant` values supplied to one engine must come from the same monotonic
clock. The engine is intended to be owned by one execution context rather
than concurrently mutated.

The `core` modules expose the codec, fragmentation, NACK, history, pacing,
arrival-feedback, bandwidth-estimation, jitter, and standalone FEC building
blocks.

## 3. Session and track model

A session is bidirectional; a track is unidirectional.

```text
session.open(Video)  -> SendTrack -> peer RecvTrack
session.accept()     -> RecvTrack <- peer SendTrack
```

The dialing side allocates even stream ids beginning at 0. The listening side
allocates odd ids beginning at 1. Each side advances by two and can allocate
128 ids. Closed ids are not reused during the session.

Only the media kind (`Audio` or `Video`) is signaled. The receiver constructs
the corresponding default `TrackConfig`; custom per-track NACK or jitter
settings are available to direct `Engine` users but are not negotiated by the
session protocol.

Opening a track follows this sequence:

```text
opener                         peer
  |---- Open(id, kind) -------->|
  |                         add remote track
  |<--------- Ready(id) --------|
add local send-ready gate
return SendTrack
```

The receive route is installed before `Ready` is sent, so the opener cannot
legitimately send media before the peer can route it. A conflicting id or
invalid parity receives `Reject`. `Session::open` times out after five seconds
and rolls the local track back.

Dropping either track attempts to send `Close`. Dropping a `Session` does not
force immediate shutdown while track handles still exist; the task stops when
all public command senders are gone. There is no separate session-shutdown
message.

## 4. Network establishment and signaling

The server listens on TCP. For every accepted connection, each endpoint binds
an ephemeral UDP socket and exchanges its UDP port in `Hello`. The peer IP is
taken from the TCP connection; it is never supplied by the signal body. The
UDP socket then connects to:

```text
TCP peer IP + UDP port from Hello
```

This avoids address injection through signaling, but it is not NAT traversal.
The design does not discover a translated UDP address.

### 4.1 Signal framing

Every signal message has a three-byte big-endian header:

```text
offset  size  field
0       1     type
1       2     body length
3       N     body
```

Bodies larger than 1024 bytes are rejected. Defined messages are:

```text
type  name    body
1     Hello   UDP port:u16 [media key:16 bytes]
2     Open    stream id:u8, media kind:u8
3     Close   stream id:u8
4     Ready   stream id:u8
5     Reject  stream id:u8
```

The dialer writes `Hello` first and the listener replies, preventing a
write/write deadlock. TCP `NODELAY` is enabled before this exchange. A
dedicated reader task owns the read half so cancellation cannot interrupt
`read_exact` after it has consumed a partial frame.

## 5. UDP packet format

All integers are big-endian. One `Packet` is one UDP payload. Every packet
starts with this 11-byte common header:

```text
offset  size  field
0       1     packet type
1       2     encoded packet size, including this header
3       4     transport sequence
7       4     media timestamp (zero for control)
```

The size field permits trailing transport padding: bytes after the encoded
size are ignored. A datagram shorter than the declared size is invalid.

Packet types are `Media = 1`, `Nack = 2`, `KeyFrameRequest = 3`, and
`ArrivalFeedback = 4`.

### 5.1 Media

```text
offset  size  field
11      1     stream id
12      4     stream frame index
16      4     per-stream media sequence
20      4     frame id
24      2     fragment index
26      2     fragment count
28      1     media kind: Video = 1, Audio = 2
29      1     keyframe: 0 or 1
30      N     encoded frame fragment
```

The current engine writes the same frame counter to the stream frame index
and frame id. Fragment count must be nonzero, fragment index must be smaller
than it, and the receiver accepts at most 256 fragments per frame.

`EngineConfig::max_packet_size` is the media-body budget, not the IP MTU and
not the complete UDP payload. Its default is 1200 bytes, producing a media
UDP payload of up to 1230 bytes before IP and UDP headers.

The timestamp is transported unchanged. QRT does not convert clock rates; the
application must use a clock understood by both encoder and decoder.

### 5.2 NACK

```text
offset  size  field
11      1     stream id
12      4     stream frame index (unused, written as zero)
16      4     base media sequence
20      2     bit-loss pattern
```

The base sequence is missing. Bit `i` reports
`base + 1 + i`, matching the role of an RFC 4585 Generic NACK PID/BLP without
using RTCP framing.

### 5.3 Keyframe request

A keyframe request consists only of the five-byte stream prefix:

```text
offset  size  field
11      1     stream id
12      4     stream frame index (unused, written as zero)
```

It serves the role of PLI/FIR and is delivered to the matching local video
encoder as `SendControl::Keyframe` or `EngineEvent::KeyframeRequest`.

### 5.4 Arrival feedback

```text
offset  size  field
11      4     range start
15      4     range end, exclusive and wrapping
19      8     received mask
27      2*N   receive deltas for set bits
```

The window contains 1 through 64 transport sequences. Mask bit `i` corresponds
to `range.start + i`. One unsigned 16-bit delta in 250-microsecond units is
present for each set bit, in low-bit order. The window may cross `u32::MAX`;
sequence arithmetic is wrapping.

## 6. Sequence spaces and frame identity

QRT deliberately maintains separate counters:

- the media sequence is per track and identifies a fragment for NACK and
  retransmission;
- the transport sequence is per session and identifies an actual transmission
  for arrival feedback, loss measurement, and in-flight accounting;
- the frame id is per track and groups fragments for reassembly.

A local track assigns frame ids and media sequences while fragmenting.
Transport sequence remains zero until the shared pacer releases the datagram.
A retransmission keeps its original media sequence but receives a new
transport sequence. There is no retransmission flag on the wire.

All counters wrap. Code comparing network sequences must use wrapping-aware
ordering rather than ordinary integer ordering.

## 7. Send path

The send pipeline is:

```text
encoded frame
  -> optional frame encryption
  -> equal-size fragmentation
  -> per-track media sequence
  -> priority queue with local deadline
  -> leaky-bucket pacer
  -> transport-sequence stamp and send history
  -> UDP socket
```

Fragments are split as evenly as possible, with remainder bytes assigned to
the last fragments. Empty frames, a zero media-body budget, and frames needing
more than 256 fragments are rejected.

`EncodedFrame::ttl_ms`, or the session default of 200 ms, becomes a local
absolute deadline when each packet enters the queue. TTL is not serialized.
A zero TTL drops the frame immediately. Expired queued packets are dropped,
and expired history entries cannot be retransmitted.

The shared queue orders packets as follows:

1. audio;
2. retransmissions;
3. NACK, arrival feedback, and keyframe requests;
4. video;
5. padding.

Audio is unpaced by default. Other packets accumulate media debt under a
leaky-bucket pacer. The default burst interval is 40 ms, maximum debt is
500 ms, and excessive estimated queue time may temporarily raise the drain
rate.

First-send media is stored in a bounded history (600 packets by default).
Retransmissions are additionally constrained by a 500 ms rate budget derived
from the current bandwidth target, preventing repeated NACKs from starving
new media.

## 8. Receive path

The receive pipeline is:

```text
UDP datagram
  -> packet validation
  -> transport arrival recorder
  -> stream demultiplex
  -> media sequence / NACK tracking
  -> per-track fragment reassembly
  -> video or audio jitter buffer
  -> complete frame
  -> optional frame decryption
  -> bounded RecvTrack queue
```

Malformed datagrams and media for unregistered tracks are discarded. A
reassembler accepts out-of-order fragments, keeps at most 64 incomplete
frames, and remembers 128 completed or abandoned frame ids to reject late
fragments that would otherwise reopen an old frame.

The high-level API bounds queued application frames:

- video keeps eight frames; overflow drops the oldest delta frame, or the
  oldest keyframe when every queued frame is a keyframe;
- audio keeps fifty frames and drops the oldest on overflow.

`RecvTrack::dropped` reports these application-queue drops. It does not report
network loss or jitter-buffer drops.

## 9. Reliability and playout

### 9.1 NACK and retransmission

Each remote track tracks holes in its media sequence. NACK processing runs
approximately every 20 ms. By default, entries have a 200 ms useful lifetime,
are retried no sooner than the current RTT, and are abandoned when they are
too old, exceed retry limits, or overflow the missing list.

An unrecoverable video gap may trigger a keyframe request. Audio has no
keyframe equivalent.

Retransmission is selective and deadline-aware; delivery is not guaranteed.
This is the primary semantic difference from TCP.

### 9.2 Video jitter

The video buffer requires decode continuity or a new keyframe. Its default
target range is 20–200 ms with a 15 ms decode/render allowance and 5 ms late
grace. The first keyframe may be released immediately (`fast_start`). A
500 ms active-stream stall can trigger a throttled keyframe request.

### 9.3 Audio jitter

Audio uses a NetEQ-style decision skeleton with an 80 ms initial target,
20–200 ms bounds, and 10 ms playout ticks. It chooses normal decode,
acceleration, preemptive expansion, or expansion. QRT does not implement the
sample-domain DSP for these operations.

The low-level engine emits `EngineEvent::Conceal` for an empty due tick. The
high-level session currently logs that event and does not synthesize a frame
or expose PLC through `RecvTrack::recv`.

### 9.4 FEC status

`core::fec` contains standalone XOR FEC generation and recovery primitives.
The current `Packet` codec has no FEC packet variant, and `Engine` does not
send or receive FEC. Applications must not assume FEC protection from the
session API.

## 10. Arrival feedback and congestion control

Every valid inbound datagram, including control traffic, is recorded by its
transport sequence. The receiver emits arrival feedback approximately every
100 ms. The sender retains matching send records for 500 ms.

The send-side estimator combines:

- acknowledged bitrate;
- loss EWMA;
- inter-arrival grouping and delay trend;
- AIMD target adjustment;
- startup and application-limited probing.

Default bandwidth bounds are 30 kbps to 2.5 Mbps with a 300 kbps initial
target. The normal pacing rate is approximately target bitrate multiplied by
1.1. Bytes still in flight can push the encoder target down before it is
published.

When a connection target changes, the engine allocates about ten percent to
audio, clamped to a total of 16–64 kbps per active audio track, and divides
the remainder equally among video tracks. Each local track receives a
`RateChange`; the encoder should apply its `target_bitrate_bps`. Probe pacing
is transport behavior and is not an encoder target.

`Session::info` reports a connection-wide snapshot: target and pacing rates,
loss, RTT, delay hypothesis, bytes in flight, queued packets, and track count.

## 11. Scheduling

Every engine entry point runs the same completion phase:

1. enqueue due arrival feedback and NACKs, consider probing, and cull history;
2. release due jitter-buffer frames and enqueue keyframe requests;
3. drain the pacer into `Packet` events;
4. compute the earliest pacer, feedback, probe, NACK, or playout deadline.

That deadline is returned as `TaskResult::next_wake`. A custom host must call
`tick` at that time even when no frames or datagrams arrive; otherwise
feedback, retransmission requests, pacing, and playout stall.

The Tokio session task performs this scheduling automatically.

## 12. Encryption

Encryption is selected by using `Server::bind_tls` and
`Client::connect_tls`. Plain `bind` and `connect` provide no encryption.
Both peers must select the same mode; a mismatch fails during TLS
establishment or protocol setup.

TLS 1.3 protects the TCP signal connection. The server presents a certificate;
the client presents none. The client trust anchor must sign the server leaf,
and the leaf subject alternative name must contain the dialed IP address. A
self-signed CA plus a leaf signed by that CA is suitable for private
deployments.

Inside TLS, the dialer generates one random 128-bit AES key and appends it to
its `Hello`. This adds no extra handshake round trip.

Encoded frame payloads are sealed with AES-128-GCM before fragmentation:

```text
counter:u64 BE || ciphertext || authentication tag:16 bytes
```

The counter is authenticated as additional data. Nonces contain a direction
byte plus the counter, allowing both directions to share the key without
nonce reuse. The receiver accepts reordering inside a 64-counter replay
window.

Encryption adds 24 bytes to every encoded frame before fragmentation. Because
encryption is frame-level, retransmission reuses the existing ciphertext
rather than sealing the same plaintext under a reused nonce.

Only encoded frame payloads are encrypted. UDP packet headers, stream ids,
timestamps, media metadata, NACKs, arrival feedback, and keyframe requests
remain visible and are not authenticated by the media AEAD. TLS protects all
signal messages.

## 13. Backpressure and lifecycle

Application commands and signal forwarding use bounded channels of 32 items.
Frame delivery never blocks the UDP/engine task: it enters the bounded
per-track queues described above.

Encoder controls use an unbounded per-send-track channel. Applications should
continuously poll `SendTrack::control`; otherwise rate changes and keyframe
requests accumulate for the life of the track.

A TCP read failure, UDP send failure, protocol violation, or loss of all
public handles ends the session task. Queued receive frames are delivered
before `RecvTrack::recv` returns `Closed`.

## 14. Observability

The library emits `tracing` events but does not install a subscriber.

- `qrt=info`: listener, session, track, bitrate, and keyframe lifecycle;
- `qrt=debug`: frames, NACK, retransmission, congestion updates, and concealment;
- `qrt=trace`: signal frames and individual UDP datagrams.

Each session runs inside a root `session` span containing the UDP peer and the
`dialer` or `listener` role.

## 15. Current constraints

- Both endpoints must implement QRT; the protocol is not standardized.
- There is no ICE, NAT traversal, address migration, or path validation.
- One session uses one UDP path and one connection-wide congestion controller.
- The media timestamp clock is application-defined and not negotiated.
- Track settings beyond media kind are not negotiated.
- Stream ids are limited to 128 allocations per side and are not reused.
- The engine integrates NACK but not the available XOR FEC module.
- Audio concealment DSP is left to the host and is not surfaced by the
  high-level receive API.
- Frame encryption protects payload bytes, not UDP metadata or control packets.

## 16. Source map

- `src/lib.rs`: Tokio session API, sockets, tasks, routing, and queueing;
- `src/signal.rs`: signal wire codec and Hello handshake;
- `src/crypto.rs`: TLS construction and frame AES-GCM;
- `src/engine/mod.rs`: socket-free engine contract and orchestration;
- `src/engine/track.rs`: unidirectional send/receive track state;
- `src/engine/path.rs`: shared pacer, history, feedback, RTT, and BWE;
- `src/core/packet.rs`: UDP wire codec;
- `src/core/fragment.rs`: frame fragmentation and reassembly;
- `src/core/nack.rs`, `history.rs`: selective recovery;
- `src/core/feedback.rs`, `bwe.rs`: arrival sensor and congestion controller;
- `src/core/pacer.rs`, `send_queue.rs`: deadline-aware transmission;
- `src/core/jitter.rs`: video and audio playout logic;
- `src/core/fec.rs`: standalone XOR FEC primitives.
