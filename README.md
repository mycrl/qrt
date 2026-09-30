# QRT

QRT (Quick Real-time Transport) is a low-latency, codec-agnostic media
transport for controlled peers. Encoded audio and video frames travel over
UDP; session and track control travel over TCP or TLS.

QRT borrows transport ideas from WebRTC—pacing, NACK, arrival feedback,
bandwidth estimation, and jitter buffering—but it is not RTP/RTCP, QUIC, or a
browser-compatible WebRTC stack. It deliberately does not provide ICE or NAT
traversal.

Start with the [technical design](DESIGN.md). It documents the architecture,
public API, wire formats, reliability and congestion behavior, encryption,
operational limits, and current implementation boundaries.

Runnable examples:

```text
cargo run --example server
cargo run --example client
```

The primary application API is `Server` / `Client` → `Session` →
`SendTrack` / `RecvTrack`. The lower-level socket-free `engine` and protocol
components under `core` are also public for custom hosts.

API documentation can be built with:

```text
cargo doc --open
```

QRT is licensed under [GNU GPL-3.0-only](LICENSE).
