//! Send-side bandwidth estimation (**sans-I/O**), simplified GoogCC / GCC.
//!
//! Turns [`crate::core::feedback::TransportPacketsFeedback`] into a
//! [`RateUpdate`] for the encoder and [`crate::core::pacer`]. This is the congestion
//! **controller**, not the TWCC sensor — pair it with [`crate::core::feedback`].
//!
//! Aligns with WebRTC `GoogCcNetworkController` and helpers under
//! `modules/congestion_controller/goog_cc/` and
//! `modules/remote_bitrate_estimator/aimd_rate_control.*`. v1 keeps:
//!
//! - delay-based path: InterArrival (5 ms groups) → Trendline → AIMD
//! - acked-throughput EWMA
//! - legacy 2 % / 10 % loss rules (not LossBasedBweV2)
//! - startup + ALR **probe clusters** (required so the rate can climb)
//!
//! # Host loop
//!
//! 1. On each matched arrival feedback → [`BandwidthEstimator::on_feedback`].
//! 2. On a ~25–100 ms timer → [`BandwidthEstimator::poll_probes`]; run clusters
//!    on the pacer, then [`BandwidthEstimator::on_probe_result`].
//! 3. Apply [`RateUpdate::pacing_rate_bps`] to the pacer and notify the
//!    encoder via [`crate::codec::Encoder::on_rate_params`] (apps should not
//!    consume probe clusters themselves).
//! 4. Optionally pass the encoder target through [`send_side_pushback`] when
//!    the send queue or in-flight window is overloaded.
//! 5. Feed `target_bitrate_bps` into
//!    [`crate::core::history::RetransRateLimiter::from_target_bps`] so NACK storms
//!    cannot starve new media.
//!
//! # Pipeline
//!
//! ```text
//! TransportPacketsFeedback + RTT
//!   → acked bitrate (EWMA) + loss ratio (EWMA)
//!   → InterArrival (5ms) → Trendline → NetworkState
//!   → AIMD (+ legacy loss rules) → target_bitrate
//!   → pacing = target × pacing_factor (default 1.1)
//!   → ProbeController (startup 3×/6×, ALR 2×)
//!   → RateUpdate → pacer + [`crate::codec::Encoder::on_rate_params`]
//! ```
//!
//! # Examples
//!
//! Heavy loss lowers the target; a successful probe raises it again:
//!
//! ```
//! use std::time::{Duration, Instant};
//!
//! use qrt::core::{
//!     bwe::{BandwidthEstimator, BweConfig},
//!     feedback::{PacketResult, TransportPacketsFeedback},
//! };
//!
//! let t0 = Instant::now();
//! let mut bwe = BandwidthEstimator::new(BweConfig {
//!     start_bitrate_bps: 300_000,
//!     min_bitrate_bps: 50_000,
//!     max_bitrate_bps: 2_500_000,
//!     ..BweConfig::default()
//! });
//!
//! // ~50% loss → legacy decrease when smoothed loss > 10%.
//! let lossy = TransportPacketsFeedback {
//!     feedback_time: t0 + Duration::from_millis(100),
//!     data_in_flight: 0,
//!     packets: (0..10u16)
//!         .map(|i| PacketResult {
//!             transport_seq: i,
//!             send_time: t0 + Duration::from_millis(u64::from(i) * 5),
//!             size_bytes: 1000,
//!             receive_time: if i % 2 == 0 {
//!                 Some(t0 + Duration::from_millis(40 + u64::from(i) * 5))
//!             } else {
//!                 None
//!             },
//!         })
//!         .collect(),
//! };
//! let start = bwe.target_bitrate_bps();
//! for k in 0..8u64 {
//!     let mut fb = lossy.clone();
//!     fb.feedback_time = t0 + Duration::from_millis(100 + k * 400);
//!     let _ = bwe.on_feedback(&fb, Duration::from_millis(50), fb.feedback_time);
//! }
//! assert!(bwe.target_bitrate_bps() < start);
//! assert!(bwe.loss_ratio() > 0.2);
//!
//! let before_probe = bwe.target_bitrate_bps();
//! let update = bwe
//!     .on_probe_result(900_000, 850_000, t0 + Duration::from_secs(2))
//!     .expect("probe applied");
//! assert!(update.target_bitrate_bps >= before_probe);
//! assert!(update.pacing_rate_bps >= update.target_bitrate_bps);
//! ```
//!
//! # Notes
//!
//! - Feedback must be keyed on **transport_seq** (see [`crate::core::feedback`]);
//!   using `media_seq` hides RTX/FEC from BWE and overestimates capacity.
//! - This module never opens sockets or sleeps; the host supplies [`Instant`].
//! - LossBasedBweV2 is intentionally deferred; replace the legacy loss block
//!   later without changing the public [`RateUpdate`] shape.
//! - See `docs/webrtc-reference.md` §2–§4 for the WebRTC mapping.

use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

use crate::core::feedback::TransportPacketsFeedback;

/// Pacing multiplier once TWCC-style feedback is available (WebRTC default ~1.1).
///
/// Used as [`BweConfig::pacing_factor`]: `pacing_rate ≈ target × factor`.
/// Without transport-wide feedback WebRTC historically used ~2.5; qrt assumes
/// [`crate::core::feedback`] is always on, so 1.1 is the right default.
///
/// # Examples
///
/// ```
/// use qrt::core::bwe::DEFAULT_PACING_FACTOR;
/// assert!((DEFAULT_PACING_FACTOR - 1.1).abs() < 1e-9);
/// ```
pub const DEFAULT_PACING_FACTOR: f64 = 1.1;

/// Send-time span under which packets form one InterArrival group (WebRTC 5 ms).
///
/// Packets whose send times fall within this window are aggregated before
/// Trendline sees `send_delta` / `recv_delta`. Do not lower this without
/// revisiting delay sensitivity.
pub const INTER_ARRIVAL_GROUP_MS: Duration = Duration::from_millis(5);

/// Minimum number of packets the pacer must send in one [`ProbeCluster`].
///
/// WebRTC probe clusters require at least five packets so send/recv rate
/// estimates are meaningful.
pub const PROBE_MIN_PACKETS: u32 = 5;

/// Minimum wall-clock duration of one [`ProbeCluster`] (WebRTC ≥15 ms).
pub const PROBE_MIN_DURATION: Duration = Duration::from_millis(15);

/// Hard cap on probe cluster target bitrate (~5 Mbps, WebRTC-style).
///
/// Prevents a runaway startup probe from flooding constrained links.
pub const PROBE_MAX_BITRATE_BPS: u64 = 5_000_000;

/// How often an application-limited (ALR) probe may be requested.
///
/// When the pacer queue is often empty, capacity can sit unused; probing at
/// about `2 × estimate` every five seconds lets the rate climb again.
pub const ALR_PROBE_INTERVAL: Duration = Duration::from_secs(5);

/// Host callback for full BWE [`RateUpdate`]s (pacing, probes, target).
///
/// **Application encoders should implement [`crate::codec::Encoder`]
/// instead** and use [`crate::codec::EncoderRateObserver`] to adapt a
/// [`RateUpdate`]. Keep this trait for Peer / pacer
/// wiring that must see pacing rates and probe clusters.
///
/// # Examples
///
/// ```
/// use qrt::core::bwe::{RateObserver, RateUpdate};
///
/// struct LogObserver;
/// impl RateObserver for LogObserver {
///     fn on_target_bitrate(&mut self, update: &RateUpdate) {
///         let _ = update.target_bitrate_bps;
///         let _ = update.pacing_rate_bps;
///     }
/// }
/// ```
pub trait RateObserver {
    /// Called when target / pacing / probe state should be applied.
    ///
    /// Transport reactions: set the pacer to [`RateUpdate::pacing_rate_bps`],
    /// run [`RateUpdate::probe_clusters`], refresh
    /// [`crate::core::history::RetransRateLimiter`], and notify the encoder with
    /// [`crate::codec::Encoder::on_rate_params`].
    fn on_target_bitrate(&mut self, update: &RateUpdate);
}

/// One BWE decision for the encoder, pacer, and optional probe bursts.
///
/// Produced by [`BandwidthEstimator::on_feedback`],
/// [`BandwidthEstimator::on_probe_result`], or
/// [`BandwidthEstimator::current_update`]. Apply `target_bitrate_bps` to the
/// media source and `pacing_rate_bps` to [`crate::core::pacer::Pacer`]; enqueue
/// `probe_clusters` as elevated-rate bursts when non-empty.
///
/// # Notes
///
/// - `pacing_rate_bps` is at least `target_bitrate_bps` (factor ≥ 1.0).
/// - `loss_ratio` is smoothed; do not treat a single feedback's raw loss as
///   this field.
#[derive(Debug, Clone, PartialEq)]
pub struct RateUpdate {
    /// Encoder / media target bitrate in bits per second.
    pub target_bitrate_bps: u64,
    /// Pacer send rate in bits per second (`target × pacing_factor`, floored at
    /// target).
    pub pacing_rate_bps: u64,
    /// Latest RTT sample the controller used (feedback round-trip).
    pub rtt: Duration,
    /// Smoothed fraction of lost transport packets in `0.0..=1.0`.
    pub loss_ratio: f64,
    /// Probe clusters the pacer should run next (often empty).
    pub probe_clusters: Vec<ProbeCluster>,
}

/// One elevated-rate burst the pacer should send to probe path capacity.
///
/// Created by [`BandwidthEstimator::poll_probes`]. The host sends at least
/// `min_packets` packets over at least `min_duration` near `target_bps`, then
/// reports measured send/recv rates via
/// [`BandwidthEstimator::on_probe_result`].
///
/// # Notes
///
/// - `id` is monotonic per estimator instance (for logging / dedup).
/// - Caps at [`PROBE_MAX_BITRATE_BPS`] and [`BweConfig::max_bitrate_bps`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeCluster {
    /// Monotonic cluster id (unique per [`BandwidthEstimator`] instance).
    pub id: u32,
    /// Target bitrate for the burst, in bits per second.
    pub target_bps: u64,
    /// Minimum packets to include in the cluster (≥ [`PROBE_MIN_PACKETS`]).
    pub min_packets: u32,
    /// Minimum wall time the cluster should span (≥ [`PROBE_MIN_DURATION`]).
    pub min_duration: Duration,
}

/// Tunables for [`BandwidthEstimator`].
///
/// Construct with [`BweConfig::default`] then override bounds for your link.
/// Changing config at runtime via [`BandwidthEstimator::set_config`] clamps
/// the live target into the new `[min, max]` range.
///
/// # Examples
///
/// ```
/// use qrt::core::bwe::{BweConfig, DEFAULT_PACING_FACTOR};
///
/// let cfg = BweConfig::default();
/// assert_eq!(cfg.pacing_factor, DEFAULT_PACING_FACTOR);
/// assert!(cfg.min_bitrate_bps < cfg.start_bitrate_bps);
/// assert!(cfg.start_bitrate_bps < cfg.max_bitrate_bps);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct BweConfig {
    /// Initial [`BandwidthEstimator`] target (also seeds startup probes).
    pub start_bitrate_bps: u64,
    /// Floor for target bitrate and probe clusters.
    pub min_bitrate_bps: u64,
    /// Ceiling for target bitrate (probes also respect this and
    /// [`PROBE_MAX_BITRATE_BPS`]).
    pub max_bitrate_bps: u64,
    /// Multiplier for [`RateUpdate::pacing_rate_bps`] (see
    /// [`DEFAULT_PACING_FACTOR`]).
    pub pacing_factor: f64,
    /// Number of `(arrival, smoothed_delay)` points Trendline keeps (~20).
    pub trendline_window: usize,
    /// Initial overuse threshold in Trendline units (WebRTC 12.5; adapts
    /// online into roughly 6–600).
    pub trendline_threshold: f64,
}

impl Default for BweConfig {
    /// Reasonable video defaults: 300 kbps start, 30 kbps–2.5 Mbps, TWCC pacing
    /// factor 1.1, Trendline window 20.
    fn default() -> Self {
        Self {
            start_bitrate_bps: 300_000,
            min_bitrate_bps: 30_000,
            max_bitrate_bps: 2_500_000,
            pacing_factor: DEFAULT_PACING_FACTOR,
            trendline_window: 20,
            trendline_threshold: 12.5,
        }
    }
}

/// Delay-based network hypothesis from the Trendline estimator.
///
/// Drives AIMD: overuse decreases, underuse holds, normal may increase.
/// Updated on each InterArrival group delta inside
/// [`BandwidthEstimator::on_feedback`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkState {
    /// One-way delay trend indicates building queues — AIMD decreases.
    Overusing,
    /// Delay trend indicates draining queues — AIMD holds (no increase).
    Underusing,
    /// No strong delay signal — AIMD may increase toward capacity.
    Normal,
}

/// Send-side GoogCC-style bandwidth estimator (**sans-I/O**).
///
/// Own one instance per sending connection. Feed it
/// [`crate::core::feedback::TransportPacketsFeedback`] from
/// [`crate::core::feedback::FeedbackAdapter`], poll probes on a timer, and push
/// [`RateUpdate`]s to the encoder / pacer.
///
/// # Examples
///
/// See the [module-level example](crate::core::bwe) for loss + probe, and
/// [`Self::poll_probes`] for startup clusters.
///
/// # Notes
///
/// - Not thread-safe; call from a single send-side task.
/// - RTT must come from feedback timing (or an equivalent), not from SR/RR.
#[derive(Debug, Clone)]
pub struct BandwidthEstimator {
    config: BweConfig,
    target_bps: u64,
    acked_bps: f64,
    loss_ratio: f64,
    rtt: Duration,
    network: NetworkState,
    aimd: AimdRateControl,
    trendline: TrendlineEstimator,
    inter_arrival: InterArrival,
    probe: ProbeController,
    last_loss_decrease: Option<Instant>,
    last_increase: Option<Instant>,
    startup_probes_pending: bool,
}

impl BandwidthEstimator {
    /// Creates a controller starting at [`BweConfig::start_bitrate_bps`].
    ///
    /// The start rate is clamped into `[min_bitrate_bps, max_bitrate_bps]`.
    /// Startup probe clusters are armed until the first [`Self::poll_probes`].
    ///
    /// # Examples
    ///
    /// ```
    /// use qrt::core::bwe::{BandwidthEstimator, BweConfig};
    ///
    /// let bwe = BandwidthEstimator::new(BweConfig::default());
    /// assert_eq!(
    ///     bwe.target_bitrate_bps(),
    ///     BweConfig::default().start_bitrate_bps
    /// );
    /// assert_eq!(bwe.network_state(), qrt::core::bwe::NetworkState::Normal);
    /// ```
    pub fn new(config: BweConfig) -> Self {
        let start = config
            .start_bitrate_bps
            .clamp(config.min_bitrate_bps, config.max_bitrate_bps);
        Self {
            aimd: AimdRateControl::new(start, config.min_bitrate_bps, config.max_bitrate_bps),
            trendline: TrendlineEstimator::new(config.trendline_window, config.trendline_threshold),
            inter_arrival: InterArrival::new(),
            probe: ProbeController::new(config.min_bitrate_bps, config.max_bitrate_bps),
            target_bps: start,
            acked_bps: start as f64,
            loss_ratio: 0.0,
            rtt: Duration::from_millis(100),
            network: NetworkState::Normal,
            last_loss_decrease: None,
            last_increase: None,
            startup_probes_pending: true,
            config,
        }
    }

    /// Current media target bitrate in bits per second.
    ///
    /// Same value as [`RateUpdate::target_bitrate_bps`] from the latest update.
    pub fn target_bitrate_bps(&self) -> u64 {
        self.target_bps
    }

    /// Latest delay-based [`NetworkState`] from Trendline.
    ///
    /// Useful for metrics / debugging; the controller already applies AIMD from
    /// this state inside [`Self::on_feedback`].
    pub fn network_state(&self) -> NetworkState {
        self.network
    }

    /// Smoothed loss ratio in `0.0..=1.0` (EWMA over feedback windows).
    ///
    /// Legacy loss rules treat ≤2 % as increase-friendly, 2–10 % as hold, and
    /// >10 % as periodic multiplicative decrease.
    pub fn loss_ratio(&self) -> f64 {
        self.loss_ratio
    }

    /// Replaces tunables and clamps the live target into the new bounds.
    ///
    /// Does not reset Trendline / InterArrival history. Call when the
    /// application changes min/max bitrate (e.g. user preference).
    pub fn set_config(&mut self, config: BweConfig) {
        self.target_bps = self
            .target_bps
            .clamp(config.min_bitrate_bps, config.max_bitrate_bps);
        self.aimd
            .set_bounds(config.min_bitrate_bps, config.max_bitrate_bps);
        self.probe
            .set_bounds(config.min_bitrate_bps, config.max_bitrate_bps);
        self.config = config;
    }

    /// Ingests one transport-wide feedback report and may emit a [`RateUpdate`].
    ///
    /// Updates acked throughput, loss, delay state, then AIMD and legacy loss
    /// rules. Returns `Some` only when [`Self::target_bitrate_bps`] changed so
    /// the host can skip no-op encoder reconfigs.
    ///
    /// # Parameters
    ///
    /// - `fb` — output of [`crate::core::feedback::FeedbackAdapter::on_feedback`]
    /// - `rtt` — feedback round-trip (or best RTT estimate); floored at 1 ms
    /// - `now` — host clock for AIMD / loss timers
    ///
    /// # Notes
    ///
    /// Probe clusters are **not** attached here; merge with
    /// [`Self::poll_probes`] via [`Self::current_update`] if you need a single
    /// [`RateUpdate`] that carries both.
    ///
    /// # Examples
    ///
    /// See the [module-level example](crate::core::bwe).
    pub fn on_feedback(
        &mut self,
        fb: &TransportPacketsFeedback,
        rtt: Duration,
        now: Instant,
    ) -> Option<RateUpdate> {
        self.rtt = rtt.max(Duration::from_millis(1));
        let prev = self.target_bps;

        self.update_acked_and_loss(fb);
        self.update_delay(fb);
        self.apply_aimd(now);
        self.apply_loss_rules(now);

        self.target_bps = self
            .target_bps
            .clamp(self.config.min_bitrate_bps, self.config.max_bitrate_bps);

        if self.target_bps != prev {
            Some(self.make_update(Vec::new()))
        } else {
            None
        }
    }

    /// Applies a measured probe cluster result and returns a fresh [`RateUpdate`].
    ///
    /// `send_bps` / `recv_bps` are throughput estimates over the cluster
    /// duration. Estimation follows WebRTC `ProbeBitrateEstimator`:
    ///
    /// - if `recv < 0.9 × send` → use `0.95 × recv` (receiver-limited)
    /// - else → `min(send, recv)`
    /// - zero on either side falls back to the non-zero side
    ///
    /// The estimate is clamped to config bounds and applied with
    /// `target = max(current, estimate)` so a probe never shrinks the rate
    /// (overuse / loss paths handle decreases).
    ///
    /// Always returns `Some` so the host can refresh pacing after a probe even
    /// when the numeric target is unchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::time::Instant;
    ///
    /// use qrt::core::bwe::{BandwidthEstimator, BweConfig};
    ///
    /// let mut bwe = BandwidthEstimator::new(BweConfig {
    ///     start_bitrate_bps: 300_000,
    ///     ..BweConfig::default()
    /// });
    /// let u = bwe
    ///     .on_probe_result(900_000, 880_000, Instant::now())
    ///     .unwrap();
    /// assert!(u.target_bitrate_bps >= 300_000);
    /// ```
    pub fn on_probe_result(
        &mut self,
        send_bps: u64,
        recv_bps: u64,
        _now: Instant,
    ) -> Option<RateUpdate> {
        let estimate = match (send_bps, recv_bps) {
            (0, r) => r,
            (s, 0) => s,
            (s, r) if (r as f64) < 0.9 * (s as f64) => ((r as f64) * 0.95) as u64,
            (s, r) => s.min(r),
        };
        let estimate = estimate.clamp(self.config.min_bitrate_bps, self.config.max_bitrate_bps);
        self.target_bps = estimate.max(self.target_bps);
        self.aimd.set_estimate(self.target_bps);
        self.acked_bps = self.acked_bps.max(self.target_bps as f64);
        Some(self.make_update(Vec::new()))
    }

    /// Returns probe clusters that should start now (startup and/or ALR).
    ///
    /// Call on a timer (~25–100 ms), same cadence as WebRTC's network
    /// controller process interval. The first call emits startup clusters at
    /// roughly `3×` and `6×` [`BweConfig::start_bitrate_bps`]. Later, when
    /// `in_alr` is true (pacer application-limited / queue usually empty), an
    /// ALR cluster at about `2 ×` current target may appear at most once per
    /// [`ALR_PROBE_INTERVAL`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::time::Instant;
    ///
    /// use qrt::core::bwe::{BandwidthEstimator, BweConfig, PROBE_MIN_PACKETS};
    ///
    /// let mut bwe = BandwidthEstimator::new(BweConfig::default());
    /// let clusters = bwe.poll_probes(Instant::now(), true);
    /// assert!(clusters.len() >= 2); // startup 3× and 6×
    /// assert!(clusters.iter().all(|c| c.min_packets >= PROBE_MIN_PACKETS));
    /// // Second call in the same ALR window does not repeat startup.
    /// assert!(bwe.poll_probes(Instant::now(), false).is_empty());
    /// ```
    pub fn poll_probes(&mut self, now: Instant, in_alr: bool) -> Vec<ProbeCluster> {
        let mut out = Vec::new();
        if self.startup_probes_pending {
            out.extend(self.probe.startup_clusters(self.config.start_bitrate_bps));
            self.startup_probes_pending = false;
        }
        out.extend(self.probe.maybe_alr_probe(now, in_alr, self.target_bps));
        out
    }

    /// Builds a [`RateUpdate`] snapshot without mutating estimator state.
    ///
    /// Use after [`send_side_pushback`] (pass the reduced target back via
    /// config / a future setter if you need the controller to track it), or to
    /// attach `probe_clusters` from [`Self::poll_probes`] onto the current
    /// rates in one struct for [`RateObserver`].
    pub fn current_update(&self, probe_clusters: Vec<ProbeCluster>) -> RateUpdate {
        self.make_update(probe_clusters)
    }

    fn make_update(&self, probe_clusters: Vec<ProbeCluster>) -> RateUpdate {
        let pacing = ((self.target_bps as f64) * self.config.pacing_factor)
            .round()
            .max(self.target_bps as f64) as u64;
        RateUpdate {
            target_bitrate_bps: self.target_bps,
            pacing_rate_bps: pacing.min(self.config.max_bitrate_bps.saturating_mul(2)),
            rtt: self.rtt,
            loss_ratio: self.loss_ratio,
            probe_clusters,
        }
    }

    fn update_acked_and_loss(&mut self, fb: &TransportPacketsFeedback) {
        let mut acked_bytes = 0usize;
        let mut received = 0u32;
        let mut lost = 0u32;
        let mut min_t: Option<Instant> = None;
        let mut max_t: Option<Instant> = None;

        for p in &fb.packets {
            if p.received() {
                received += 1;
                acked_bytes += p.size_bytes;
                min_t = Some(min_t.map_or(p.send_time, |t| t.min(p.send_time)));
                max_t = Some(max_t.map_or(p.send_time, |t| t.max(p.send_time)));
            } else {
                lost += 1;
            }
        }

        let total = received + lost;
        if total > 0 {
            let sample = f64::from(lost) / f64::from(total);
            self.loss_ratio = 0.9 * self.loss_ratio + 0.1 * sample;
        }

        if let (Some(a), Some(b)) = (min_t, max_t) {
            // Floor dt at 100ms so a tiny send span cannot inflate acked_bps.
            let dt = b
                .saturating_duration_since(a)
                .max(Duration::from_millis(100));
            let sample_bps = (acked_bytes as f64) * 8.0 * 1000.0 / (dt.as_millis() as f64);
            if sample_bps.is_finite() && sample_bps > 0.0 {
                self.acked_bps = 0.85 * self.acked_bps + 0.15 * sample_bps;
            }
        }
    }

    fn update_delay(&mut self, fb: &TransportPacketsFeedback) {
        let deltas = self.inter_arrival.compute(fb);
        for (send_delta, recv_delta) in deltas {
            self.network = self.trendline.update(send_delta, recv_delta);
        }
    }

    fn apply_aimd(&mut self, now: Instant) {
        let acked = self.acked_bps.max(1.0) as u64;
        match self.network {
            NetworkState::Overusing => {
                self.target_bps = self.aimd.decrease(acked, self.target_bps);
            }
            NetworkState::Underusing => {
                // Hold while queues drain — matching AimdRateControl.
            }
            NetworkState::Normal => {
                let since = self
                    .last_increase
                    .map(|t| now.saturating_duration_since(t))
                    .unwrap_or(Duration::from_secs(1));
                // Pace increases (~200ms) so a burst of Normal samples cannot
                // multiplicative-ramp in one feedback.
                if since >= Duration::from_millis(200) {
                    self.target_bps = self.aimd.increase(self.target_bps, acked, since);
                    self.last_increase = Some(now);
                }
            }
        }
    }

    fn apply_loss_rules(&mut self, now: Instant) {
        let loss = self.loss_ratio;
        if loss <= 0.02 {
            // Low loss: AIMD increase path already grows the rate.
            return;
        }
        if loss <= 0.10 {
            // Mid loss: hold (WebRTC legacy SendSideBandwidthEstimation).
            return;
        }
        // High loss: decrease every 300ms + RTT.
        let interval = Duration::from_millis(300) + self.rtt;
        let due = self
            .last_loss_decrease
            .map(|t| now.saturating_duration_since(t) >= interval)
            .unwrap_or(true);
        if due {
            let factor = 1.0 - 0.5 * loss;
            self.target_bps = ((self.target_bps as f64) * factor).round().max(1.0) as u64;
            self.aimd.set_estimate(self.target_bps);
            self.last_loss_decrease = Some(now);
        }
    }
}

/// Reduces the encoder target when the send path is overloaded.
///
/// Call after reading [`RateUpdate::target_bitrate_bps`] when
/// [`crate::core::send_queue`] delay or in-flight bytes (from
/// [`crate::core::feedback::TransportPacketsFeedback::data_in_flight`]) show the
/// pacer cannot clear media in time. Returns a lower bitrate; never zero.
///
/// Heuristic (v1, not a full congestion window controller):
///
/// - `queue_time > 500ms` → ×0.75
/// - else `queue_time > 250ms` → ×0.9
/// - if `in_flight_bytes > congestion_window` → ×0.5 (additional)
///
/// # Examples
///
/// ```
/// use std::time::Duration;
///
/// use qrt::core::bwe::send_side_pushback;
///
/// let reduced = send_side_pushback(
///     1_000_000,
///     Duration::from_millis(800),
///     200_000,
///     Some(100_000),
/// );
/// assert!(reduced < 1_000_000);
/// assert!(reduced >= 1);
/// ```
///
/// # Notes
///
/// This does not mutate [`BandwidthEstimator`]; feed the result to the encoder
/// only, or also lower [`BweConfig::max_bitrate_bps`] if you want the
/// controller to stop probing above the pushed-back rate.
pub fn send_side_pushback(
    target_bps: u64,
    queue_time: Duration,
    in_flight_bytes: usize,
    congestion_window_bytes: Option<usize>,
) -> u64 {
    let mut rate = target_bps;
    if queue_time > Duration::from_millis(500) {
        rate = rate * 3 / 4;
    } else if queue_time > Duration::from_millis(250) {
        rate = rate * 9 / 10;
    }
    if let Some(cwnd) = congestion_window_bytes {
        if in_flight_bytes > cwnd {
            rate = rate / 2;
        }
    }
    rate.max(1)
}

// ---------------------------------------------------------------------------
// InterArrival — WebRTC InterArrivalDelta (5ms send groups)
// ---------------------------------------------------------------------------

/// Groups received packets whose send times fall within
/// [`INTER_ARRIVAL_GROUP_MS`], then emits `(send_delta, recv_delta)` between
/// consecutive completed groups for Trendline.
#[derive(Debug, Clone)]
struct InterArrival {
    current: Option<ArrivalGroup>,
    /// Last completed group: `(last_send, last_recv, size)`.
    prev_complete: Option<(Instant, Instant, usize)>,
}

#[derive(Debug, Clone)]
struct ArrivalGroup {
    first_send: Instant,
    last_send: Instant,
    last_recv: Instant,
    size: usize,
}

impl InterArrival {
    fn new() -> Self {
        Self {
            current: None,
            prev_complete: None,
        }
    }

    /// Returns `(send_delta, recv_delta)` pairs between consecutive completed
    /// groups. Only packets with a receive time are considered.
    fn compute(&mut self, fb: &TransportPacketsFeedback) -> Vec<(Duration, Duration)> {
        let mut out = Vec::new();
        let mut pkts: Vec<_> = fb
            .packets
            .iter()
            .filter_map(|p| p.receive_time.map(|r| (p.send_time, r, p.size_bytes)))
            .collect();
        pkts.sort_by_key(|(s, _, _)| *s);

        for (send, recv, size) in pkts {
            match &mut self.current {
                None => {
                    self.current = Some(ArrivalGroup {
                        first_send: send,
                        last_send: send,
                        last_recv: recv,
                        size,
                    });
                }
                Some(g) => {
                    if send.saturating_duration_since(g.first_send) <= INTER_ARRIVAL_GROUP_MS {
                        g.last_send = send;
                        g.last_recv = recv;
                        g.size += size;
                    } else {
                        let completed = (g.last_send, g.last_recv, g.size);
                        if let Some((ps, pr, _)) = self.prev_complete {
                            let sd = completed.0.saturating_duration_since(ps);
                            let rd = completed.1.saturating_duration_since(pr);
                            if !sd.is_zero() {
                                out.push((sd, rd));
                            }
                        }
                        self.prev_complete = Some(completed);
                        self.current = Some(ArrivalGroup {
                            first_send: send,
                            last_send: send,
                            last_recv: recv,
                            size,
                        });
                    }
                }
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Trendline — WebRTC TrendlineEstimator (simplified)
// ---------------------------------------------------------------------------

/// Tracks `recv_delta − send_delta`, smooths it, and compares a linear slope
/// against an adaptive threshold to yield [`NetworkState`].
#[derive(Debug, Clone)]
struct TrendlineEstimator {
    window: usize,
    threshold: f64,
    smoothed: f64,
    accumulated: f64,
    /// `(arrival_ms, smoothed_delay)` samples for slope fitting.
    points: VecDeque<(f64, f64)>,
    num_deltas: usize,
    overuse_time: Duration,
    state: NetworkState,
}

impl TrendlineEstimator {
    fn new(window: usize, threshold: f64) -> Self {
        Self {
            window: window.max(2),
            threshold,
            smoothed: 0.0,
            accumulated: 0.0,
            points: VecDeque::new(),
            num_deltas: 0,
            overuse_time: Duration::ZERO,
            state: NetworkState::Normal,
        }
    }

    fn update(&mut self, send_delta: Duration, recv_delta: Duration) -> NetworkState {
        let delta_ms = recv_delta.as_secs_f64() * 1000.0 - send_delta.as_secs_f64() * 1000.0;
        self.accumulated += delta_ms;
        self.smoothed = 0.9 * self.smoothed + 0.1 * self.accumulated;
        self.num_deltas += 1;

        let arrival = self
            .points
            .back()
            .map(|(a, _)| a + recv_delta.as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        self.points.push_back((arrival, self.smoothed));
        while self.points.len() > self.window {
            self.points.pop_front();
        }

        let trend = linear_slope(&self.points).unwrap_or(0.0);
        // WebRTC: modified_trend = min(num_deltas, 60) * trend * 4.0
        let modified = (self.num_deltas.min(60) as f64) * trend * 4.0;

        // Adaptive threshold (k_up≈0.0087, k_down≈0.039).
        let abs_m = modified.abs();
        if abs_m > self.threshold {
            self.threshold += 0.0087 * (abs_m - self.threshold);
        } else {
            self.threshold += 0.039 * (abs_m - self.threshold);
        }
        self.threshold = self.threshold.clamp(6.0, 600.0);

        if modified > self.threshold {
            self.overuse_time += send_delta.max(Duration::from_millis(1));
            // Require >10ms of sustained overuse before flipping state.
            if self.overuse_time > Duration::from_millis(10) {
                self.state = NetworkState::Overusing;
            }
        } else if modified < -self.threshold {
            self.overuse_time = Duration::ZERO;
            self.state = NetworkState::Underusing;
        } else {
            self.overuse_time = Duration::ZERO;
            self.state = NetworkState::Normal;
        }
        self.state
    }
}

fn linear_slope(points: &VecDeque<(f64, f64)>) -> Option<f64> {
    let n = points.len();
    if n < 2 {
        return None;
    }
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_xy = 0.0;
    for &(x, y) in points {
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }
    let nf = n as f64;
    let denom = nf * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-9 {
        return Some(0.0);
    }
    Some((nf * sum_xy - sum_x * sum_y) / denom)
}

// ---------------------------------------------------------------------------
// AIMD — WebRTC AimdRateControl (send-side subset)
// ---------------------------------------------------------------------------

/// Additive-increase / multiplicative-decrease toward acked throughput.
#[derive(Debug, Clone)]
struct AimdRateControl {
    estimate: u64,
    min_bps: u64,
    max_bps: u64,
}

impl AimdRateControl {
    fn new(start: u64, min_bps: u64, max_bps: u64) -> Self {
        Self {
            estimate: start,
            min_bps,
            max_bps,
        }
    }

    fn set_bounds(&mut self, min_bps: u64, max_bps: u64) {
        self.min_bps = min_bps;
        self.max_bps = max_bps;
        self.estimate = self.estimate.clamp(min_bps, max_bps);
    }

    fn set_estimate(&mut self, bps: u64) {
        self.estimate = bps.clamp(self.min_bps, self.max_bps);
    }

    /// Overuse: `≈ 0.85 × acked − 5kbps`, never above `current`.
    fn decrease(&mut self, acked_bps: u64, current: u64) -> u64 {
        let decreased = ((acked_bps as f64) * 0.85) as u64;
        let decreased = decreased.saturating_sub(5_000);
        // Overuse must not raise the estimate (inflated acked windows).
        self.estimate = decreased.min(current).clamp(self.min_bps, self.max_bps);
        self.estimate
    }

    /// Normal: multiplicative `1.08^Δt` (Δt≤1s), at least +1 kbps, capped near
    /// `1.5 × acked + 10kbps`.
    fn increase(&mut self, current: u64, acked_bps: u64, dt: Duration) -> u64 {
        let secs = dt.as_secs_f64().clamp(0.0, 1.0);
        let mult = 1.08_f64.powf(secs);
        let mut next = ((current as f64) * mult).round() as u64;
        next = next.max(current.saturating_add(1_000));
        let cap = ((acked_bps as f64) * 1.5) as u64 + 10_000;
        next = next.min(cap);
        self.estimate = next.clamp(self.min_bps, self.max_bps);
        self.estimate
    }
}

// ---------------------------------------------------------------------------
// ProbeController — WebRTC ProbeController (startup + ALR)
// ---------------------------------------------------------------------------

/// Emits startup (`3×` / `6×` start) and ALR (`2×` estimate) probe clusters.
#[derive(Debug, Clone)]
struct ProbeController {
    min_bps: u64,
    max_bps: u64,
    next_id: u32,
    last_alr_probe: Option<Instant>,
}

impl ProbeController {
    fn new(min_bps: u64, max_bps: u64) -> Self {
        Self {
            min_bps,
            max_bps,
            next_id: 1,
            last_alr_probe: None,
        }
    }

    fn set_bounds(&mut self, min_bps: u64, max_bps: u64) {
        self.min_bps = min_bps;
        self.max_bps = max_bps;
    }

    fn startup_clusters(&mut self, start_bps: u64) -> Vec<ProbeCluster> {
        let a = self.cluster((start_bps.saturating_mul(3)).min(PROBE_MAX_BITRATE_BPS));
        let b = self.cluster((start_bps.saturating_mul(6)).min(PROBE_MAX_BITRATE_BPS));
        vec![a, b]
    }

    fn maybe_alr_probe(
        &mut self,
        now: Instant,
        in_alr: bool,
        estimate_bps: u64,
    ) -> Vec<ProbeCluster> {
        if !in_alr {
            return Vec::new();
        }
        let due = self
            .last_alr_probe
            .map(|t| now.saturating_duration_since(t) >= ALR_PROBE_INTERVAL)
            .unwrap_or(true);
        if !due {
            return Vec::new();
        }
        self.last_alr_probe = Some(now);
        let target = (estimate_bps.saturating_mul(2))
            .clamp(self.min_bps, self.max_bps.min(PROBE_MAX_BITRATE_BPS));
        vec![self.cluster(target)]
    }

    fn cluster(&mut self, target_bps: u64) -> ProbeCluster {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        ProbeCluster {
            id,
            target_bps: target_bps.clamp(self.min_bps, self.max_bps.min(PROBE_MAX_BITRATE_BPS)),
            min_packets: PROBE_MIN_PACKETS,
            min_duration: PROBE_MIN_DURATION,
        }
    }
}
