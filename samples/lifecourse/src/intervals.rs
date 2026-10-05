// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements calibrated risk prediction with honest
// uncertainty for its clients. If your team needs expertise in reporting a
// model's risk as an interval it can stand behind, you can procure our
// services by sending an email to info@swedishembedded.com.

//! A risk by ten years as an interval: a secondary analysis, not one of the
//! pre-registered criteria.
//!
//! The early-stopping subjects of a run were never trained on. They calibrate
//! the run's predicted all-cause risk by ten years into a Venn-Abers interval
//! per test subject, under the same restriction to cycles that support the
//! horizon as every other ten-year metric. Reported: the calibration and
//! Brier score of the raw and the interval's single (merged) probability, and
//! how wide the intervals are.

use serde::{Deserialize, Serialize};
use splinter_sdk::model::timeline::survival::brier::brier;
use splinter_sdk::model::timeline::survival::calibration::at_horizon;
use splinter_sdk::model::timeline::survival::estimate::censoring;
use splinter_sdk::model::timeline::survival::venn_abers::{merged, VennAbers};
use splinter_sdk::model::timeline::survival::Obs;
use splinter_sdk::model::timeline::{observed, Prediction, Subject};

use crate::build::CODES;
use crate::metrics::{all_cause, subset, Calibration, Horizons};

/// The horizon of the intervals, in years.
pub const INTERVAL_HORIZON: f64 = 10.0;

/// Calibration and Brier score of one set of probabilities.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Scored {
    /// Calibration at the horizon.
    pub calibration: Calibration,
    /// IPCW Brier score at the horizon.
    pub brier: f64,
}

/// Venn-Abers intervals for the risk by the horizon on one test set.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Intervals {
    /// Calibration subjects whose outcome by the horizon is known.
    pub n_calibration: usize,
    /// The model's own probabilities.
    pub raw: Scored,
    /// The intervals' merged probabilities.
    pub merged: Scored,
    /// Interval width `p1 - p0`: 10th, 50th and 90th percentile over the test
    /// subjects.
    pub width: (f64, f64, f64),
}

fn scored(risk: &[f64], obs: &[Obs], t: f64) -> Option<Scored> {
    let g = censoring(obs);
    let cal = at_horizon(risk, obs, 0, t, &g, 10);
    Some(Scored {
        calibration: Calibration {
            slope: cal.slope,
            intercept: cal.intercept,
            oe_ratio: cal.oe_ratio,
            mean_abs_gap: cal.mean_abs_gap,
            n: obs.len(),
        },
        brier: brier(risk, obs, 0, t, &g)?,
    })
}

/// Intervals for `test_risk` (risk of the outcome, cause 0, by `t`) from the
/// calibration subjects; `None` when either set cannot be scored.
pub fn intervals_at(
    cal_risk: &[f64],
    cal_obs: &[Obs],
    test_risk: &[f64],
    test_obs: &[Obs],
    t: f64,
) -> Option<Intervals> {
    let va = VennAbers::at_horizon(cal_risk, cal_obs, 0, t, &censoring(cal_obs));
    if va.is_empty() || test_risk.is_empty() {
        return None;
    }
    // A new subject enters as an average calibration subject: its own survey
    // weight would let one heavily weighted subject swing its interval.
    let weight = va.mean_weight()?;
    let pairs: Vec<(f64, f64)> = test_risk.iter().map(|&r| va.interval(r, weight)).collect();
    let mut widths: Vec<f64> = pairs.iter().map(|(p0, p1)| p1 - p0).collect();
    widths.sort_by(f64::total_cmp);
    let q = |p: f64| widths[((widths.len() - 1) as f64 * p).round() as usize];
    let merged_risk: Vec<f64> = pairs.into_iter().map(merged).collect();
    Some(Intervals {
        n_calibration: cal_obs
            .iter()
            .filter(|o| o.time > t || o.cause.is_some())
            .count(),
        raw: scored(test_risk, test_obs, t)?,
        merged: scored(&merged_risk, test_obs, t)?,
        width: (q(0.1), q(0.5), q(0.9)),
    })
}

fn risk_and_obs(subjects: &[Subject], preds: &[Prediction], h: &Horizons) -> (Vec<f64>, Vec<Obs>) {
    let (subs, ps) = subset(subjects, preds, h, INTERVAL_HORIZON);
    let risk = ps
        .iter()
        .map(|p| 1.0 - p.survival(INTERVAL_HORIZON))
        .collect();
    (risk, all_cause(observed(&subs, &CODES)))
}

/// Intervals for the all-cause risk by ten years of `test`, calibrated on
/// `held` (subjects the model was not trained on).
pub fn ten_year(
    held: &[Subject],
    held_preds: &[Prediction],
    test: &[Subject],
    test_preds: &[Prediction],
    h: &Horizons,
) -> Option<Intervals> {
    let (cal_risk, cal_obs) = risk_and_obs(held, held_preds, h);
    let (test_risk, test_obs) = risk_and_obs(test, test_preds, h);
    intervals_at(&cal_risk, &cal_obs, &test_risk, &test_obs, INTERVAL_HORIZON)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rate `0.05 exp(x)`; the score doubles the true log-odds by `t`.
    fn overconfident(n: usize, t: f64, seed: u64) -> (Vec<f64>, Vec<Obs>) {
        let mut s = seed;
        let mut u = move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        let (mut risk, mut obs) = (vec![], vec![]);
        for _ in 0..n {
            let rate = 0.05 * (3.0 * u() - 1.5).exp();
            let time = -u().ln() / rate;
            let censor = 12.0 + 8.0 * u();
            obs.push(if time <= censor {
                Obs::event(time, 0)
            } else {
                Obs::censored(censor)
            });
            let p: f64 = 1.0 - (-rate * t).exp();
            let doubled = 2.0 * (p / (1.0 - p)).ln();
            risk.push(1.0 / (1.0 + (-doubled).exp()));
        }
        (risk, obs)
    }

    #[test]
    fn intervals_recalibrate_an_overconfident_risk() {
        let (cal_risk, cal_obs) = overconfident(3000, 10.0, 1);
        let (test_risk, test_obs) = overconfident(3000, 10.0, 2);
        let iv = intervals_at(&cal_risk, &cal_obs, &test_risk, &test_obs, 10.0).unwrap();
        assert_eq!(
            iv.n_calibration, 3000,
            "follow-up of 12+ years labels everyone at 10"
        );
        assert!(iv.raw.calibration.slope < 0.7, "{iv:?}");
        assert!((iv.merged.calibration.slope - 1.0).abs() < 0.15, "{iv:?}");
        assert!(iv.merged.brier < iv.raw.brier, "{iv:?}");
        assert!(iv.width.0 > 0.0 && iv.width.0 <= iv.width.1 && iv.width.1 <= iv.width.2);
    }
}
