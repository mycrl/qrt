//! The one UDP pipe shared by every track on this engine.
//!
//! Local tracks enqueue here. Remote tracks do not: their NACK and keyframe
//! requests are also packets, so they join the same queue. One leaky bucket
//! paces audio, video, retransmission, and feedback.
//!
//! Two sequence spaces meet in [`Path::drain`]:
//!
//! - media sequence — already written by [`super::track::LocalTrack`], used by NACK
//! - transport sequence — stamped here, at the instant the datagram leaves
//!
//! A retransmission keeps its media sequence and receives a new transport
//! sequence. Nothing on the wire says "this is a retransmission".
//!
//! Arrival times of every inbound datagram are recorded here too. Peer
//! arrival reports are matched against the send log and fed to the bandwidth
//! estimator, which then rewrites the pacer rate and the retransmission budget.

use std::time::{Duration, Instant};

use bytes::Bytes;

use super::EngineConfig;
use crate::core::{
    bwe::{BandwidthEstimator, NetworkState, RateUpdate, send_side_pushback},
    feedback::{ArrivalRecorder, FeedbackAdapter, TransportSeqAssigner},
    history::{PacketHistory, RetransRateLimiter, RetransmitOutcome},
    pacer::{Pacer, PacerConfig},
    packet::{MediaType, Packet, Payload, StreamPacket},
    send_queue::OutgoingPacket,
};

/// How often a sending engine may raise the pacer to probe for more capacity.
const PROBE_INTERVAL: Duration = Duration::from_millis(50);

/// Numbers [`super::Engine::info`] shows. Track count is not included; the
/// engine adds that from its own map.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PathStatus {
    /// Connection-wide encoder target before the engine splits audio and video.
    pub target_bitrate_bps: u64,
    /// Leaky-bucket send rate. Usually a bit above the encoder target.
    pub pacing_rate_bps: u64,
    /// Smoothed transport loss ratio in `0.0..=1.0`.
    pub loss_ratio: f64,
    /// Latest RTT. Arrival feedback, or the configured initial guess.
    pub rtt: Duration,
    /// Delay-based hypothesis (underuse / delay / overuse).
    pub network: NetworkState,
    /// Bytes stamped sent and not yet covered by an arrival report.
    pub in_flight_bytes: usize,
    /// Packets still waiting in the pacer.
    pub queued_packets: usize,
}

/// What one matched arrival report changed.
///
/// `rtt` is set only when the report contained a received packet with a
/// plausible delay. `update` is set only when the estimator moved the
/// connection target; an unchanged target must not retarget encoders.
pub struct ArrivalOutcome {
    /// Clamped RTT sample, when the report yielded one.
    pub rtt: Option<Duration>,
    /// Connection target after in-flight pushback, when it moved.
    pub update: Option<RateUpdate>,
}

/// Send queue, pacer, retransmission memory, arrival log, and bandwidth estimate.
pub struct Path {
    pacer: Pacer,
    /// Burst and debt limits from construction. Only the rate is replaced later.
    pacer_config: PacerConfig,
    /// Connection-wide counter written onto the datagram at real send time.
    transport_seqs: TransportSeqAssigner,
    /// First-send media, so a NACK can clone it.
    history: PacketHistory,
    /// Send times keyed by transport sequence. Matched against peer reports.
    feedback_tx: FeedbackAdapter,
    /// Receive times keyed by transport sequence. Becomes our arrival reports.
    arrival_rx: ArrivalRecorder,
    bwe: BandwidthEstimator,
    /// `pacing_rate ≈ target × pacing_factor`. Copied from [`EngineConfig`] at
    /// construction so a report does not have to carry it back in.
    pacing_factor: f64,
    rtt: Duration,
    last_probe_at: Option<Instant>,
}

impl Path {
    /// One pacer, one transport-sequence counter, history sized from `config`.
    ///
    /// The retransmission budget starts as a 500 ms window of the initial
    /// target. A later arrival report replaces it.
    pub fn new(config: &EngineConfig) -> Self {
        let rtt = config.initial_rtt.max(Duration::from_millis(1));
        let mut history = PacketHistory::new(config.history_capacity);
        history.set_rtt(rtt);
        history.set_rate_limiter(RetransRateLimiter::from_target_bps(
            config.bwe.start_bitrate_bps,
            Duration::from_millis(500),
        ));

        Self {
            pacer: Pacer::new(config.pacer),
            pacer_config: config.pacer,
            transport_seqs: TransportSeqAssigner::new(),
            history,
            feedback_tx: FeedbackAdapter::new(config.feedback.clone()),
            arrival_rx: ArrivalRecorder::new(config.feedback.clone()),
            bwe: BandwidthEstimator::new(config.bwe.clone()),
            pacing_factor: config.bwe.pacing_factor,
            rtt,
            last_probe_at: None,
        }
    }

    /// Latest RTT, already clamped to at least one millisecond.
    pub fn rtt(&self) -> Duration {
        self.rtt
    }

    /// Clamps to at least 1 ms. A zero RTT would make NACK and retransmission
    /// ask again on the next tick.
    pub fn set_rtt(&mut self, rtt: Duration) {
        self.rtt = rtt.max(Duration::from_millis(1));
        self.history.set_rtt(self.rtt);
    }

    /// Dashboard fields. Does not include how many tracks the engine has.
    pub fn status(&self) -> PathStatus {
        PathStatus {
            target_bitrate_bps: self.bwe.target_bitrate_bps(),
            pacing_rate_bps: self.pacer.pacing_rate_bps(),
            loss_ratio: self.bwe.loss_ratio(),
            rtt: self.rtt,
            network: self.bwe.network_state(),
            in_flight_bytes: self.feedback_tx.in_flight_bytes(),
            queued_packets: self.pacer.queue().len(),
        }
    }

    /// Encodes `packet` and puts it on the pacer. The TTL deadline is `now`.
    pub fn enqueue_packet(&mut self, packet: &Packet, ttl_ms: u16, now: Instant) {
        self.pacer.enqueue_packet(packet, ttl_ms, now);
    }

    /// Queues a retransmission already built by history. It keeps the media
    /// sequence inside the payload and is marked so [`Self::drain`] will not
    /// store a second copy.
    pub fn enqueue_retransmit(&mut self, packet: OutgoingPacket) {
        self.pacer.enqueue(packet);
    }

    /// Looks up a first-send packet the peer NACKed.
    ///
    /// [`RetransmitOutcome::Ready`] is the only outcome the engine should
    /// enqueue. The others mean rate-limited, expired, or never sent.
    pub fn retransmission(
        &mut self,
        stream_id: u8,
        media_seq: u32,
        now: Instant,
    ) -> RetransmitOutcome {
        self.history.get_retransmission(stream_id, media_seq, now)
    }

    /// Datagrams the pacer allows out at `now`, each with a fresh transport sequence.
    ///
    /// Leftover queue items wait for a later tick. This is the only place a
    /// transport sequence is assigned.
    pub fn drain(&mut self, now: Instant) -> Vec<Bytes> {
        let mut datagrams = Vec::new();

        while let Some(mut outgoing) = self.pacer.poll(now) {
            let mut wire = outgoing.wire.to_vec();

            // The header is too short to hold a sequence. Still return the
            // bytes: dropping work the pacer already released loses the packet
            // with no NACK and no trace.
            let Some(transport_seq) = self.transport_seqs.stamp(&mut wire) else {
                tracing::trace!(bytes = wire.len(), "send without transport sequence");

                datagrams.push(Bytes::from(wire));
                continue;
            };

            outgoing.wire = Bytes::from(wire);

            {
                let decoded = Packet::from_bytes(outgoing.wire.clone()).ok();
                let audio = decoded.as_ref().is_some_and(|packet| {
                    matches!(
                        &packet.payload,
                        Payload::Stream(stream) if matches!(
                            &stream.packet,
                            StreamPacket::Media(media) if media.media_type == MediaType::Audio
                        )
                    )
                });

                // `now` is send time. Delay-based BWE subtracts this from the
                // peer's receive time, so stamping at encode time would look
                // like extra queuing delay.
                self.feedback_tx
                    .on_sent(transport_seq, now, outgoing.len(), audio);

                if outgoing.retransmit {
                    // Same media sequence, new transport sequence. `mark_sent`
                    // starts the "do not retransmit this sequence again until
                    // one RTT" timer. The payload is not stored again.
                    if let Some(packet) = &decoded
                        && let Payload::Stream(stream) = &packet.payload
                        && let StreamPacket::Media(media) = &stream.packet
                    {
                        self.history.mark_sent(stream.id, media.sequence, now);
                    }
                } else if decoded.as_ref().is_some_and(|packet| {
                    matches!(
                        &packet.payload,
                        Payload::Stream(stream) if matches!(stream.packet, StreamPacket::Media(_))
                    )
                }) {
                    self.history.put_outgoing(&outgoing, now);
                }
            }

            tracing::trace!(
                transport_seq,
                retransmit = outgoing.retransmit,
                bytes = outgoing.wire.len(),
                "send"
            );

            datagrams.push(outgoing.wire);
        }

        datagrams
    }

    /// Next time the leaky bucket can release a packet, or `now` if one is already due.
    pub fn next_send_time(&self, now: Instant) -> Option<Instant> {
        self.pacer.next_send_time(now)
    }

    /// Drops history entries past their media TTL. A stale frame must not be
    /// cloned into a retransmission.
    pub fn cull(&mut self, now: Instant) {
        self.history.cull(now);
    }

    /// Notes that this UDP datagram arrived. Media, NACK, and feedback all
    /// occupy the path, so every type is recorded before it is demultiplexed.
    pub fn record_arrival(&mut self, transport_seq: u32, now: Instant, size_bytes: usize) {
        self.arrival_rx.on_packet(transport_seq, now, size_bytes);
    }

    /// An arrival report for the peer, when one is due. `None` means wait.
    pub fn poll_feedback(&mut self, now: Instant) -> Option<Packet> {
        self.arrival_rx
            .poll(now)
            .map(|feedback| feedback.as_packet())
    }

    /// Next time [`Self::poll_feedback`] may return a report.
    pub fn next_feedback_at(&self, now: Instant) -> Option<Instant> {
        self.arrival_rx.next_poll_at(now)
    }

    /// Matches a peer arrival report, refreshes RTT, and maybe installs a new rate.
    ///
    /// Returns `None` when `packet` does not match anything we sent. A `Some`
    /// with `update: None` still may carry an RTT sample; the target itself
    /// did not move.
    pub fn on_arrival_feedback(&mut self, packet: &Packet, now: Instant) -> Option<ArrivalOutcome> {
        let report = self.feedback_tx.on_feedback_packet(packet, now)?;

        // Newest packet the peer marked received. Sub-millisecond samples are
        // clock noise. Multi-second gaps are a report that sat too long to
        // describe the path we are on now.
        let rtt = report
            .packets
            .iter()
            .rev()
            .find(|packet| packet.received())
            .map(|packet| now.saturating_duration_since(packet.send_time))
            .filter(|sample| {
                *sample > Duration::from_millis(1) && *sample < Duration::from_secs(2)
            });

        if let Some(sample) = rtt {
            self.set_rtt(sample);
        }

        // The estimator sees the RTT we just stored. An unchanged target
        // returns here; pacer and retransmission budget stay as they are.
        let Some(mut update) = self.bwe.on_feedback(&report, self.rtt, now) else {
            return Some(ArrivalOutcome { rtt, update: None });
        };

        {
            // Bytes already on the wire count against the encoder. Otherwise
            // a high target keeps pushing media into a queue the report just
            // said is occupied.
            let pushed = send_side_pushback(
                update.target_bitrate_bps,
                Duration::ZERO,
                self.feedback_tx.in_flight_bytes(),
                None,
            );
            update.target_bitrate_bps = pushed;
            update.pacing_rate_bps = ((pushed as f64) * self.pacing_factor)
                .round()
                .max(pushed as f64) as u64;
        }

        self.apply_pacing_rate(update.pacing_rate_bps);
        self.history
            .set_rate_limiter(RetransRateLimiter::from_target_bps(
                update.target_bitrate_bps,
                Duration::from_millis(500),
            ));

        tracing::debug!(
            target_bps = update.target_bitrate_bps,
            pacing_bps = update.pacing_rate_bps,
            ?rtt,
            loss = tracing::field::display(format_args!("{:.3}", update.loss_ratio)),
            "bandwidth"
        );

        Some(ArrivalOutcome {
            rtt,
            update: Some(update),
        })
    }

    /// Maybe raise the pacer so the estimator can discover capacity above the
    /// current send rate.
    ///
    /// Receive-only engines pass `has_local = false` and return immediately.
    /// There is nothing of ours on the path to probe with.
    pub fn consider_probe(&mut self, now: Instant, has_local: bool) {
        if !has_local {
            return;
        }

        let due = self
            .last_probe_at
            .map(|at| now.saturating_duration_since(at) >= PROBE_INTERVAL)
            .unwrap_or(true);
        if !due {
            return;
        }

        self.last_probe_at = Some(now);

        // An empty queue means the application is not filling the pipe.
        // Sending faster for a moment is what produces the delay samples
        // `poll_probes` is accounting for. The returned clusters are not
        // turned into packets: this path paces by rate, and the encoder is
        // told the target bitrate, not the probe.
        let in_alr = self.pacer.queue().is_empty();
        let pacing_rate = self.pacer.pacing_rate_bps();
        let _clusters = self.bwe.poll_probes(now, in_alr);

        if in_alr {
            let pacing_bps = self
                .bwe
                .target_bitrate_bps()
                .saturating_mul(2)
                .max(pacing_rate);

            tracing::trace!(pacing_bps, "probe");

            self.apply_pacing_rate(pacing_bps);
        }
    }

    /// Next probe attempt. `None` when `has_local` is false.
    ///
    /// The first attempt is `now`, so the first tick of a sending engine can probe.
    pub fn next_probe_at(&self, now: Instant, has_local: bool) -> Option<Instant> {
        has_local.then(|| {
            self.last_probe_at
                .map(|at| at + PROBE_INTERVAL)
                .unwrap_or(now)
        })
    }

    /// Replaces the leaky-bucket rate and keeps the burst and debt limits
    /// captured in [`Self::new`].
    fn apply_pacing_rate(&mut self, pacing_rate_bps: u64) {
        let mut config = self.pacer_config;
        config.pacing_rate_bps = pacing_rate_bps;
        self.pacer.set_config(config);
    }
}
