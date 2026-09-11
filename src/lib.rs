//! QRT (Quick Real-time Transport): a low-latency media core for bare UDP.
//!
//! Start with [`Engine`]. It synchronously turns [`EncodedFrame`] values
//! into UDP datagrams ([`Engine::push_frame`]) and peer datagrams back into
//! frames ([`Engine::push_packet`]). The application owns sockets and calls
//! [`Engine::tick`] at the returned wake deadline. Packet, FEC, pacing,
//! feedback, and BWE building blocks remain available in [`core`].

pub mod codec;
pub mod core;
pub mod engine;
