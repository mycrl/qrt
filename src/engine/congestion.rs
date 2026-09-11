//! Session-wide congestion control (BWE + RTT).
//!
//! [`Congestion`] holds the estimator and RTT. [`super::Engine`] drives it
//! from `finish` → `advance_control` (NACK, arrival reports, ALR probes).
//! Packet generation still goes through the shared send pacer so everything
//! shares one leaky bucket.
//!
//! ```text
//! peer ArrivalFeedback
//!    -> match against packets we stamped with transport_seq
//!    -> BandwidthEstimator (delay trend + loss)
//!    -> new target bitrate
//!         ├─ pacer send rate  (how fast UDP leaves)
//!         ├─ NACK RTX budget  (retransmit must not starve new media)
//!         └─ EngineEvent::RateChange per local track (encoder)
//!
//! On a timer, independently:
//!    - send our own ArrivalFeedback so the peer can run the same loop
//!    - emit NACK for holes in remote media_seq
//!    - maybe probe (send faster for a moment) so BWE can climb
//! ```

use std::time::{Duration, Instant};

use ahash::HashMap;

use super::{
    EngineConfig, MediaKind, PROBE_INTERVAL,
    tracks::{Egress, Track},
};
use crate::core::{
    bwe::{BandwidthEstimator, NetworkState, RateUpdate, send_side_pushback},
    feedback::TransportPacketsFeedback,
    history::RetransRateLimiter,
    pacer::PacerConfig,
};

/// Session-wide delay/loss controller and RTT estimate.
///
/// One instance per engine / UDP flow. Individual tracks do not have their
/// own BWE — they share this target, then [`Self::split_rates`] carves it
/// into audio vs video encoder hints.
pub struct Congestion {
    /// GoogCC-style delay + loss estimator.
    bwe: BandwidthEstimator,
    /// Latest RTT sample (from arrival feedback send time vs now).
    rtt: Duration,
    /// Last time we asked BWE to consider a probe cluster.
    last_probe_at: Option<Instant>,
}

impl Congestion {
    /// Starts from [`EngineConfig::bwe`] and [`EngineConfig::initial_rtt`].
    pub fn new(config: &EngineConfig) -> Self {
        Self {
            bwe: BandwidthEstimator::new(config.bwe.clone()),
            rtt: config.initial_rtt.max(Duration::from_millis(1)),
            last_probe_at: None,
        }
    }

    /// Latest RTT sample (arrival feedback, or the configured guess).
    pub fn rtt(&self) -> Duration {
        self.rtt
    }

    /// Clamps to ≥ 1 ms so NACK / RTX timers never use a zero interval.
    pub fn set_rtt(&mut self, rtt: Duration) {
        self.rtt = rtt.max(Duration::from_millis(1));
    }

    /// Connection-wide encoder target before audio / video split.
    pub fn target_bitrate_bps(&self) -> u64 {
        self.bwe.target_bitrate_bps()
    }

    /// Smoothed transport loss ratio in `0.0..=1.0`.
    pub fn loss_ratio(&self) -> f64 {
        self.bwe.loss_ratio()
    }

    /// Delay-based hypothesis (underuse / delay / overuse).
    pub fn network_state(&self) -> NetworkState {
        self.bwe.network_state()
    }

    /// RTT ≈ now minus the send time of the newest packet the peer marked received.
    ///
    /// Ignore 0/1 ms (clock noise) and multi-second gaps (stale report).
    pub(super) fn rtt_from_report(
        report: &TransportPacketsFeedback,
        now: Instant,
    ) -> Option<Duration> {
        let received = report
            .packets
            .iter()
            .rev()
            .find(|packet| packet.received())?;
        let sample = now.saturating_duration_since(received.send_time);
        (sample > Duration::from_millis(1) && sample < Duration::from_secs(2)).then_some(sample)
    }

    /// Runs BWE on one report and installs the new rates on `egress`.
    ///
    /// Returns `None` when the estimator keeps the same target (no
    /// [`super::EngineEvent::RateChange`] needed). A `Some` update has
    /// send-side pushback applied so in-flight bytes do not pile up, then
    /// the pacer rate and NACK RTX budget are overwritten to match.
    pub(super) fn apply_report(
        &mut self,
        report: &TransportPacketsFeedback,
        egress: &mut Egress,
        now: Instant,
        pacing_factor: f64,
        pacer_config: PacerConfig,
    ) -> Option<RateUpdate> {
        // BWE may return None when the report does not move the target.
        let mut update = self.bwe.on_feedback(report, self.rtt, now)?;

        {
            // If lots of bytes are still in flight, cut the encoder target so we
            // do not pile more media on a congested path.
            let pushed = send_side_pushback(
                update.target_bitrate_bps,
                Duration::ZERO,
                egress.in_flight_bytes(),
                None,
            );
            update.target_bitrate_bps = pushed;

            // Pacer runs a bit above the encoder (~pacing_factor, default 1.1)
            // so the leaky bucket can absorb small bursts.
            update.pacing_rate_bps =
                ((pushed as f64) * pacing_factor).round().max(pushed as f64) as u64;
        }

        {
            egress.set_pacing_rate(pacer_config, update.pacing_rate_bps);

            // Cap NACK retransmits to a sliding window of the target bitrate.
            egress.set_retrans_limiter(RetransRateLimiter::from_target_bps(
                update.target_bitrate_bps,
                Duration::from_millis(500),
            ));
        }

        Some(update)
    }

    /// Split one connection target across local audio and video tracks.
    ///
    /// Audio gets a small reserved bucket (about 10%, clamped 16–64 kbps per
    /// track). The rest is divided equally among local video tracks.
    pub(super) fn split_rates(
        update: &RateUpdate,
        tracks: &HashMap<u8, Track>,
        pacing_factor: f64,
    ) -> Vec<(u8, RateUpdate)> {
        let locals: Vec<(u8, MediaKind)> = tracks
            .iter()
            .filter_map(|(stream_id, track)| match track {
                Track::Local(local) => Some((*stream_id, local.kind)),
                Track::Remote(_) => None,
            })
            .collect();

        let (video_rate, audio_rate) = {
            // Avoid divide-by-zero when only audio is sending.
            let video_count = locals
                .iter()
                .filter(|(_, kind)| *kind == MediaKind::Video)
                .count()
                .max(1);

            let audio_count = locals
                .iter()
                .filter(|(_, kind)| *kind == MediaKind::Audio)
                .count();

            let audio_budget = if audio_count == 0 {
                0
            } else {
                (update.target_bitrate_bps / 10).clamp(16_000, 64_000 * audio_count as u64)
            };

            let video_rate =
                update.target_bitrate_bps.saturating_sub(audio_budget) / video_count as u64;

            let audio_rate = if audio_count == 0 {
                0
            } else {
                audio_budget / audio_count as u64
            };

            (video_rate, audio_rate)
        };

        locals
            .into_iter()
            .map(|(stream_id, kind)| {
                let mut track_update = update.clone();
                track_update.target_bitrate_bps = match kind {
                    MediaKind::Video => video_rate,
                    MediaKind::Audio => audio_rate,
                };
                track_update.pacing_rate_bps =
                    ((track_update.target_bitrate_bps as f64) * pacing_factor).round() as u64;
                (stream_id, track_update)
            })
            .collect()
    }

    /// Maybe start a probe burst so BWE can discover more capacity.
    ///
    /// When the pacer queue is empty we are application-limited (ALR): the
    /// path might hold more than we are sending. Temporarily raising the
    /// pacer rate lets extra packets out; delay/loss on those packets teach
    /// the estimator.
    pub(super) fn maybe_probe(
        &mut self,
        egress: &mut Egress,
        has_local: bool,
        now: Instant,
        pacer_config: PacerConfig,
    ) {
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

        {
            // Empty queue => we are not filling the pipe; probing is useful.
            let in_alr = egress.queue_is_empty();
            let pacing_rate = egress.pacing_rate_bps();
            let _clusters = self.bwe.poll_probes(now, in_alr);

            if in_alr {
                // 2× the target is the usual ALR probe; never slower than now.
                egress.set_pacing_rate(
                    pacer_config,
                    self.bwe
                        .target_bitrate_bps()
                        .saturating_mul(2)
                        .max(pacing_rate),
                );
            }
        }
    }

    /// Next time [`Self::maybe_probe`] should run.
    ///
    /// `None` when this engine has no local track (receive-only: never probe).
    /// If we have never probed, returns `now` so the first tick can try.
    pub(super) fn next_probe_at(&self, now: Instant, has_local: bool) -> Option<Instant> {
        // Remote-only engines never probe.
        has_local.then(|| {
            self.last_probe_at
                .map(|at| at + PROBE_INTERVAL)
                .unwrap_or(now)
        })
    }
}
