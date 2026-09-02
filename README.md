# QRT

**QRT (Quick Real-time Transport)** is a real-time media transport over **bare UDP**.

It copies WebRTC’s _ideas_ (pacing, GCC/GoogCC, NACK, XOR FEC, TWCC-style arrival feedback, jitter/NetEQ) and **not** WebRTC’s _wire_ (RTP/RTCP, ICE, DTLS, SDP). One 20-byte header carries media, FEC, and feedback on a single datagram path. There is no QUIC.

## Why this instead of WebRTC

libwebrtc is a full calling stack: ICE/STUN/TURN, SDP, DTLS-SRTP, RTP/RTCP (often muxed), RTX/RED line formats, browser interop, codecs and capture. That is the right tool when you must talk to Chrome or sit in a standards mesh.

QRT is for a **known peer** over UDP where that surface is cost, not value:

- No NAT traversal, signaling, or SRTP handshake in the hot path.
- One packet family, one socket, one pacer — feedback is not a second protocol.
- Core is **sans-I/O**: state machines take `Instant` and datagrams; the host owns the socket (Tokio façade optional).
- Codec-opaque: the transport never parses VP8/H.264 FU-A; frames are `frame_id` + `frag_index` / `frag_count`.
- **Timeliness over reliability**: TTL/deadline drop beats delivering a stale frame. Expired packets are not retransmitted.
- Encryption, ICE, and browser compatibility are out of scope unless added later.

Use WebRTC when you need a browser or a standards-compatible SFU. Use QRT when you control both ends and want the congestion/reliability _algorithms_ without the calling stack.

## Same as WebRTC (algorithms)

| Problem               | WebRTC                                                                                               | QRT                                                                                               |
| --------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Split a frame for MTU | `RtpPacketizer::SplitAboutEqually` (~1200 B body, about-equal packets, first/last/single reductions) | Same split; `media_seq` assigned at fragment time                                                 |
| Reassemble frames     | `packet_buffer` / `rtp_video_stream_receiver2`                                                       | `(stream_id, frame_id)` + `frag_count`; reorder OK                                                |
| Send scheduling       | `PacingController` leaky bucket (~40 ms burst, ~500 ms max debt, audio unpaced, drain large queues)  | Same debt model                                                                                   |
| Queue priority        | `PrioritizedPacketQueue` (audio > RTX > video > padding)                                             | Audio > retrans > video/FEC > feedback > padding                                                  |
| Selective retransmit  | RTCP Generic NACK (RFC 4585 PID+BLP) + `RtpPacketHistory`                                            | `Packet::Nack` + history; skip if FEC already recovered; RTX rate-capped from BWE target          |
| Erasure               | ULPFEC / FlexFEC XOR (`ForwardErrorCorrection`, max 48 media)                                        | XOR over **full media datagrams**; recover when a row has exactly one hole                        |
| Arrival / BWE sensor  | transport-cc / TWCC (transport-wide seq, 250 µs recv deltas, ~100 ms reports)                        | `ArrivalFeedback` on `transport_seq`; same semantics, not the RTCP bit layout                     |
| Congestion control    | GoogCC: InterArrival 5 ms → Trendline → AIMD; acked bitrate; probes                                  | Same delay path + **legacy** 2% / 10% loss rules (not LossBasedBweV2); startup/ALR probe clusters |
| Video playout         | FrameBuffer + VCMTiming; PLI/FIR                                                                     | Video jitter buffer; stalled → `KeyframeReq`                                                      |
| Audio playout         | NetEQ                                                                                                | Decision skeleton (`AudioNetEq`), not a full WebRTC NetEQ                                         |
| Encoder coupling      | `OnTargetTransferRate`                                                                               | BWE target / RTT / loss → encoder; probe clusters stay on the pacer                               |

## Different from WebRTC (stack and wire)

| WebRTC                                                  | QRT                                                                                  |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| RTP media + RTCP (often muxed, compound RTCP)           | One `Packet` type on one UDP flow                                                    |
| SSRC, PT, marker, extensions                            | `stream_id`, 3-bit `Type`, flags (`audio` / `key` / `retrans`), explicit frag fields |
| transport-cc as an RTP header extension + RTCP feedback | `transport_seq` in every 20-byte header, stamped at **pacer egress**                 |
| Separate RTX SSRC / RED for retrans and FEC             | Same `media_seq` with `flags.retrans`; FEC is its own packet type, never NACKed      |
| ICE, DTLS-SRTP, SDP                                     | None                                                                                 |
| REMB / SR-RR as primary BWE or RTT                      | RTT from arrival feedback send time vs now; no REMB                                  |
| Browser / RFC wire compatibility                        | Intentionally **not** RTP/RTCP-compatible                                            |

`media_seq` is per-stream identity (NACK, reassembly, FEC). `transport_seq` is connection-wide (BWE, in-flight). They must not be mixed.

## Implementation

### Header (20 bytes, big-endian)

```text
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|V=0|Type |Flags|   Stream ID   |         Media Seq             |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|       Transport Seq           |           Frame ID...         |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                        ...Frame ID                            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|          Frag Index           |          Frag Count           |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                     Timestamp (90 kHz)                        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|        TTL (ms)               |           Payload...          |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

`Type`: Media, Nack, ArrivalFeedback, KeyframeReq, Fec.  
`ttl_ms` is **remaining lifetime**, not a wall clock; peers need no synced clocks. Sequence compare is wrapping (`seq_ahead`), not integer `<`.

NACK body is RFC 4585-style `base_seq` + BLP. Arrival feedback is `first_seq` + 64-bit received mask + optional 250 µs recv deltas. FEC body is `seq_base` + 64-bit mask + `length_xor` + XOR payload (mask bit `i` protects `media_seq = seq_base + i`).

### Send

```text
encoded frame
  → SplitAboutEqually fragment (media_seq, frame_id, frag_index/count)
  → optional XOR FEC rows (video)
  → priority queue + TTL drop
  → leaky-bucket pacer
  → stamp transport_seq, remember first-send media, note send time
  → UDP
```

Overdue packets (`now >= deadline`) are dropped in the queue. Retransmits are clones from history with `flags.retrans`, same `media_seq`, new `transport_seq`. A retransmit rate limiter (budget ≈ BWE target over a sliding window) stops NACK storms from starving new media. If the send queue / in-flight window is overloaded, the encoder target is pushed back before it is applied.

### Receive

```text
UDP → decode
  → record arrival by transport_seq (every datagram)
  → Media  → NACK list → reassembly → video jitter / audio NetEQ
  → Fec    → XOR recover (single hole per row) → same as Media
  → Nack   → history lookup → pacer
  → ArrivalFeedback → match send history → GoogCC
  → KeyframeReq → encoder
```

FEC-recovered media is marked received for NACK so it is not requested again.

### Congestion loop

```text
on_sent(transport_seq)
  → peer ArrivalFeedback
  → acked bitrate EWMA + loss EWMA
  → InterArrival (5 ms) → Trendline → AIMD (± legacy loss)
  → pacing = target × ~1.1
  → probe clusters (startup 3×/6×, ALR 2×) so the estimate can climb
  → pacer rate + encoder target
```

Arrival recording is the **sensor**; GoogCC is the **controller**. Probes are paced bursts; the encoder should follow `target_bitrate_bps`, not the probe schedule.

## License

[GNU General Public License v3.0](LICENSE) only ([SPDX: GPL-3.0-only](https://spdx.org/licenses/GPL-3.0-only.html)).

You may use, modify, and distribute this software, including in commercial products, provided that derivative works you **distribute** are also licensed under GPLv3 and you provide corresponding source.

Linking this crate into a closed-source binary you ship typically makes that binary a GPL derivative. If you cannot accept copyleft, do not use this crate.
