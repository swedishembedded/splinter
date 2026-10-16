// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements release gates whose error rates are
// computed, not guessed, for its clients. If your team needs expertise in
// statistical acceptance bounds for model updates, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: the anchor check may be decided on the net count of answers lost,
//! and the constant chosen for a 200-question suite is the one whose error
//! rates are stated in the preregistration.

use super::*;

fn suite(n: usize) -> SuiteSummary {
    SuiteSummary {
        name: "anchor".into(),
        tasks: n,
        excluded: BTreeMap::new(),
    }
}

/// `n` paired items: the first `lost` the champion alone answers, the next
/// `gained` the candidate alone answers, the rest both answer.
fn outcomes(n: usize, lost: usize, gained: usize) -> Vec<PairedOutcome> {
    (0..n)
        .map(|i| {
            let (candidate, baseline) = if i < lost {
                (false, true)
            } else if i < lost + gained {
                (true, false)
            } else {
                (true, true)
            };
            PairedOutcome {
                item: format!("q{i}"),
                candidate: Some(candidate),
                baseline: Some(baseline),
                cluster: None,
            }
        })
        .collect()
}

fn decide(lost: usize, gained: usize, bound: AnchorBound) -> Check<Anchor> {
    let digest = Digest::of(b"anchor");
    anchor(1, digest, suite(200), &outcomes(200, lost, gained), bound)
}

#[test]
fn six_lost_and_none_gained_pass_and_seven_refuse() {
    assert!(decide(6, 0, AnchorBound::NetLost(6)).passed);
    let refused = decide(7, 0, AnchorBound::NetLost(6));
    assert!(!refused.passed);
    assert!(refused.reason.unwrap().contains("net lost 7 > 6"));
}

#[test]
fn gains_offset_losses() {
    assert!(decide(9, 3, AnchorBound::NetLost(6)).passed);
    assert!(!decide(10, 3, AnchorBound::NetLost(6)).passed);
}

#[test]
fn the_rate_bound_stays_the_default() {
    assert_eq!(
        GateConfig::default().anchor_bound(),
        AnchorBound::Rate(0.02)
    );
    // 3 of 200 is 0.015 and passes; 5 of 200 is 0.025 and refuses.
    assert!(decide(3, 0, GateConfig::default().anchor_bound()).passed);
    assert!(!decide(5, 0, GateConfig::default().anchor_bound()).passed);
    // A report written before the net bound existed reads as the rate bound.
    let old: GateConfig = {
        let mut value = serde_json::to_value(GateConfig::default()).unwrap();
        value.as_object_mut().unwrap().remove("anchor_net_lost");
        serde_json::from_value(value).unwrap()
    };
    assert_eq!(old.anchor_bound(), AnchorBound::Rate(0.02));
}

#[test]
fn a_configured_net_bound_replaces_the_rate_bound() {
    let config = GateConfig {
        anchor_net_lost: Some(6),
        ..GateConfig::default()
    };
    assert_eq!(config.anchor_bound(), AnchorBound::NetLost(6));
}

#[test]
fn the_report_names_the_bound_that_decided() {
    let measured = decide(7, 1, AnchorBound::NetLost(6)).measured.unwrap();
    assert_eq!(measured.max_net_lost, Some(6));
    assert_eq!(
        decide(1, 0, AnchorBound::Rate(0.02))
            .measured
            .unwrap()
            .max_net_lost,
        None
    );
}

/// P(X - Y >= threshold) for independent X ~ Poisson(`lost_mean`) and
/// Y ~ Poisson(`gained_mean`), summed exactly (to double precision) over the
/// joint support.
fn net_at_least(lost_mean: f64, gained_mean: f64, threshold: i64) -> f64 {
    const SUPPORT: usize = 400;
    let pmf = |mean: f64| -> Vec<f64> {
        (0..SUPPORT)
            .scan(f64::exp(-mean), |p, i| {
                let current = *p;
                *p *= mean / (i + 1) as f64;
                Some(current)
            })
            .collect()
    };
    let (lost, gained) = (pmf(lost_mean), pmf(gained_mean));
    let mut total = 0.0;
    for (x, px) in lost.iter().enumerate() {
        for (y, py) in gained.iter().enumerate() {
            if x as i64 - y as i64 >= threshold {
                total += px * py;
            }
        }
    }
    total
}

/// P(net lost <= k) when the champion loses `true_loss` answers on top of
/// independent flips at rate `flip` each way.
fn released(n: f64, flip: f64, true_loss: f64, k: i64) -> f64 {
    1.0 - net_at_least(n * flip + true_loss, n * flip, k + 1)
}

#[test]
fn the_bound_of_six_has_the_error_rates_the_prereg_states() {
    let within = |got: f64, want: f64| (got - want).abs() < 0.0005;
    // A drift-free candidate is refused by chance at net >= 7.
    assert!(
        within(net_at_least(200.0 * 0.02, 200.0 * 0.02, 7), 0.011),
        "2%"
    );
    assert!(
        within(net_at_least(200.0 * 0.03, 200.0 * 0.03, 7), 0.030),
        "3%"
    );
    // k = 4 would refuse it far more often, which is why k = 6.
    assert!(within(net_at_least(200.0 * 0.02, 200.0 * 0.02, 5), 0.055));
    assert!(within(net_at_least(200.0 * 0.03, 200.0 * 0.03, 5), 0.095));
    // The old exam: 40 items, zero loss allowed (the five refused 0.6B candidates).
    assert!(within(net_at_least(40.0 * 0.02, 40.0 * 0.02, 1), 0.323));
    assert!(within(net_at_least(40.0 * 0.03, 40.0 * 0.03, 1), 0.362));
    // A real net loss of 15 answers (7.5%) or 10 (5%) is released this often, at 3% flips.
    assert!(within(released(200.0, 0.03, 15.0, 6), 0.047));
    assert!(within(released(200.0, 0.03, 10.0, 6), 0.229));
}
