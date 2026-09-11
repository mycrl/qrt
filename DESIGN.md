Chinese original: [docs/info.md](docs/info.md).

**QRT is a “real-time TCP” built on bare UDP.** It does not replace your socket, and it does not retransmit forever the way TCP does.

## How this differs from TCP and UDP

**TCP** is reliable and ordered. It retransmits until the peer acknowledges. Delay can grow without bound. Fine for files and web pages.

**UDP** just throws a datagram onto the network. Loss, reordering, and sending too fast are not its problem. That is why games and live media use it: **a late frame is worse than a dropped one**.

QRT keeps UDP, then adds only what live media actually needs:

| What TCP does for you | What QRT does | The important difference |
| --------------------- | ------------- | ------------------------ |
| Split / reassemble    | One frame becomes several UDP datagrams; the peer stitches them back | Packets have a TTL and are dropped when stale |
| Loss recovery         | NACK: retransmit only packets that can still arrive in time | Delivery is not guaranteed |
| Congestion control    | BWE: estimate capacity and retarget the encoder | Latency over throughput |
| Byte stream           | Complete audio/video frames | The app sees frames, not a stream of bytes |

Signaling (handshake, open/close tracks) uses **TCP**. The media path is **UDP**. `Engine` only owns the UDP half.

## Engine is not a socket

Open `src/engine/mod.rs`. `Engine` has **no UDP socket and no background thread**. It is a pure state machine: you feed it three kinds of input; it returns a `TaskResult` of work the host must do.

```1:11:src/engine/mod.rs
//! Think of [`Engine`] as a **pure state machine**. It has no UDP socket and no
//! background thread. You feed it three kinds of input; it returns a
//! [`TaskResult`] of things the host must do:
//!
//! | You call | Meaning |
//! |----------|---------|
//! | [`Engine::push_frame`] | "here is one encoded audio/video frame to send" |
//! | [`Engine::push_packet`] | "here is one UDP payload from the peer" |
//! | [`Engine::tick`] | "the wake timer fired; run due maintenance" |
```

Your application (the host) owns the socket. A typical loop:

1. Encoder produces a frame → `push_frame`
2. `recvfrom` yields a datagram → `push_packet`
3. The wake timer fires → `tick`
4. Dispatch each `EngineEvent`:
    - `Packet` → `sendto` on the peer socket
    - `Frame` → give it to your decoder
    - `RateChange` → retarget that encoder (`RateParams`)
    - `KeyframeRequest` → the next video frame must be an IDR

Remember: **QRT does the bookkeeping; you put bytes on the wire.**

An engine is four pieces of state:

```385:396:src/engine/mod.rs
pub struct Engine {
    /// Session tunables (default TTL, MTU, BWE, pacer, NACK history size).
    config: EngineConfig,
    /// One entry per `stream_id`. Local = we send, Remote = we receive.
    tracks: HashMap<u8, Track>,
    /// Shared send path: priority queue, leaky-bucket pacer, NACK history.
    egress: Egress,
    /// Shared receive path: arrival log, XOR FEC repair, frame reassembly.
    ingress: Ingress,
    /// One bandwidth estimator and RTT for the whole UDP flow.
    congestion: Congestion,
}
```

- **tracks**: one audio/video stream each. Local = we encode; Remote = the peer encoded it.
- **egress**: one send queue for the whole UDP flow (the toll booth).
- **ingress**: one receive path (arrival log, FEC repair, reassembly).
- **congestion**: how fast this path can still go.

Register streams with `add_local_track` / `add_remote_track`. `EncodedFrame`, `MediaKind`, and `RateParams` live in `src/engine/media.rs` — there is no separate codec session layer.

## A track is one-way, not a TCP connection

A TCP connection is a bidirectional byte stream. Here **one `stream_id` is one direction**.

Your camera is Local `stream_id=1`; the peer’s camera is Remote `stream_id=2`. The two ids must differ or sequence spaces collide.

After adding a local track you still need `set_send_ready(true)`, or media will not leave. That gate exists so you do not leak frames before the signaling handshake says the peer is listening.

## Two sequence numbers — do not mix them

Every UDP datagram starts with a 20-byte header (`HEADER_SIZE` in `src/core/packet.rs`). It carries two counters:

- **`media_seq`**: packet *n* of this media stream. NACK, reassembly, and FEC key off it. Think “page number in this comic”.
- **`transport_seq`**: packet *n* of this UDP conversation. Bandwidth estimation keys off it. Think “post office send stamp”.

`media_seq` is assigned when the frame is fragmented. `transport_seq` is stamped only when the packet actually leaves (`Egress::drain`). Congestion cares about what is *on the path*, not what we encoded.

There are five packet types, all on the same UDP flow:

- `Media` — audio/video fragments
- `Fec` — repair packets (a missing media packet can be reconstructed)
- `Nack` — “I am missing these pages” (the same idea as WebRTC: RTCP Generic NACK + optional RTX)
- `ArrivalFeedback` — “these send numbers arrived”
- `KeyframeReq` — the picture is stuck; send a keyframe

That is why there is no separate RTCP channel: control and media share one send valve.

---

The rest of this file walks one live call through the code.

## Send: `push_frame`

The encoder gives you a frame (maybe 20 KB). A UDP datagram is usually ~1200 bytes, so it must be split. The method returns `Result<TaskResult, EngineError>`.

```549:589:src/engine/mod.rs
        let (packets, fec_packets) = {
            let track = self
                .tracks
                .get_mut(&frame.stream_id)
                .and_then(Track::as_local_mut)
                .ok_or(EngineError::UnknownTrack {
                    stream_id: frame.stream_id,
                })?;
            // send_ready / kind checks …
            let packets = track.fragment(&frame, ttl, &self.config.payload_limits)?;
            let fec_packets = track.generate_fec(&packets);
            (packets, fec_packets)
        };

        {
            for packet in &packets {
                self.egress.enqueue_packet(packet, now);
            }
            for fec in fec_packets {
                self.egress.enqueue_packet(&fec.as_packet(), now);
            }
        }

        Ok(self.finish(now, TaskResult::default()))
```

Three steps:

1. **Fragment** (`LocalTrack::fragment`): one frame → several Media packets, with `frame_id` and `frag_index` / `frag_count`.
2. **Optional FEC** (`generate_fec`): video also gets XOR repair rows. Like handing out two copies of an exam so one lost copy can still be recovered.
3. **Pacer**: not an immediate `sendto`. A leaky bucket emits at the estimated bitrate so a burst does not explode router queues (that would explode delay).

Every public entry then runs `finish`: due control packets, due decode frames, and due UDP, settled in one pass.

**TTL** is remaining lifetime in milliseconds, not a wall clock. If `EncodedFrame::ttl_ms` is `None`, `EngineConfig::default_ttl_ms` is used (200 ms). Expired packets are not sent and not retransmitted. That is the largest value gap versus TCP: **a late frame is garbage**.

## Receive: `push_packet`

Feed the raw UDP payload in as-is. Returns a `TaskResult` (garbage bytes still run `finish`, so timers keep moving).

```603:666:src/engine/mod.rs
    pub fn push_packet(&mut self, payload: &[u8], now: Instant) -> TaskResult {
        let mut result = TaskResult::default();
        let Ok(packet) = Packet::decode(payload) else {
            return self.finish(now, result);
        };
        self.ingress
            .record_arrival(header.transport_seq, now, payload.len());

        match packet {
            Packet::Media { .. } => { /* NACK, reassemble, jitter */ }
            Packet::Fec { .. } => { /* XOR-recover missing Media */ }
            Packet::Nack { .. } => { /* clone from history, retransmit */ }
            Packet::ArrivalFeedback { .. } => { /* peer arrivals → BWE */ }
            Packet::KeyframeReq { .. } => { /* tell your encoder to IDR */ }
        }
        self.finish(now, result)
    }
```

In plain language:

1. **Log the arrival**: this `transport_seq` showed up now. Later we pack that into `ArrivalFeedback` so the peer can run BWE (like a TCP ACK, except it reports *which* send numbers arrived and *when*, not “please retransmit a byte stream”).
2. **Media**: record the seq for NACK, try to finish a frame, push it into the jitter buffer.
3. **Fec**: if a protection row is missing exactly one packet, XOR reconstructs it. Recovered media is **not** logged as a new network arrival (it did not consume path capacity).
4. **Nack**: the peer is missing `media_seq=10,11`. We clone from send history and enqueue on the pacer. TTL expiry or RTX budget → refuse. We do not grind like TCP.
5. **ArrivalFeedback**: run congestion control; may emit `RateChange` (“encoder, please use 800 kbps”).
6. **KeyframeReq**: too many reference frames were lost. The next encoded frame must be a keyframe.

A completed frame is **not** handed to the decoder immediately. It waits in the jitter buffer so reordering does not stutter the picture. That wait is the cost of UDP reordering; TCP does something similar in the kernel, but it is willing to wait much longer.

`TaskResult::packets()` / `TaskResult::frames()` are convenience iterators over `events`. The real list is still `events`.

## Why `tick` exists

UDP has no kernel timer that retransmits for you. NACK, arrival reports, the leaky bucket, and bandwidth probes all fire **when they are due**.

So `tick` is almost one line:

```674:676:src/engine/mod.rs
    pub fn tick(&mut self, now: Instant) -> TaskResult {
        self.finish(now, TaskResult::default())
    }
```

Remember the `finish` order:

```683:692:src/engine/mod.rs
    /// Shared tail of [`Self::push_frame`], [`Self::push_packet`], and
    /// [`Self::tick`].
    ///
    /// 1. [`Self::advance_control`] — due NACK / arrival reports / probes
    /// 2. [`Self::collect_inbound`] — jitter buffer → [`EngineEvent::Frame`]
    /// 3. [`Self::drain_pacer`] — leaky bucket → [`EngineEvent::Packet`]
    /// 4. [`Self::next_deadline`] — fill [`TaskResult::next_wake`]
```

Outgoing control (NACK, feedback) is queued first, then decode frames are released, then UDP actually leaks out. `next_wake` is when you should call `tick` again. If you never call it, control packets go to sleep.

## Glossary (TCP terms)

- **Pacer / leaky bucket**: rate-limit sending. TCP has cwnd; here a token bucket leaks at the estimated bitrate.
- **NACK**: selective retransmit. “I am missing pages 10 and 11”, not TCP’s cumulative ACK. WebRTC video does the same (RTCP NACK + optional RTX).
- **FEC**: send redundancy so one lost packet can be reconstructed without waiting an RTT.
- **BWE / Congestion**: estimate path capacity. When the path is full, slow the encoder instead of stuffing more media.
- **Probe**: an empty pacer queue means we are application-limited; the path might hold more. Briefly raise the send rate to find out.
- **Jitter buffer**: wait tens of milliseconds on purpose so reordering can settle before playout.
- **PLI / KeyframeReq**: video decode broke; ask for a picture that does not depend on previous frames.
