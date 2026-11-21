// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measurements whose arithmetic is checked
// against exact references, for its clients. If your team needs expertise in
// clustered evaluation statistics and the learning signal of agent rollouts,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: competence intervals and the learning signal agree with exact
//! references computed independently of them (`conformance.json`, written by
//! `generate_conformance.py` beside it in exact rational arithmetic).
//!
//! * The quality interval resamples whole families: its bounds are quantiles
//!   of the exactly enumerated family-bootstrap distribution of the pooled
//!   share, within the Monte Carlo error of its 1,000 resamples.
//! * `LS = V / mean cost`, `V` the population variance shrunk towards the
//!   Beta(1, 1) posterior's `p (1 - p)` with the weight of two attempts; a
//!   group is informative only with two or more attempts that differ.
//!
//! Each check is also run against deliberate defects written here (an
//! item-level resample, families weighted equally, a sample variance, no
//! shrinkage, zero-variance groups kept): every one must be caught.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use serde::Deserialize;
use splinter_core::annotation::Strength;
use splinter_core::digest::Digest;
use splinter_core::experience::ExperienceId;
use splinter_core::outcome::TaskOutcome;
use splinter_core::rng::SplitMix64;
use splinter_eval::campaign::learning_signal::{selection_weight, Attempt as Rollout, TaskSignal};
use splinter_eval::competence::{competence, Attempt};

#[derive(Deserialize)]
struct Fixture {
    clustered_bootstrap: Vec<ClusteredCase>,
    learning_signal: Vec<SignalCase>,
}

#[derive(Deserialize)]
struct ClusteredCase {
    name: String,
    families: Vec<Vec<f64>>,
    level: f64,
    mean: f64,
    cdf: Vec<(f64, f64)>,
    item_level_detectable: bool,
}

#[derive(Deserialize)]
struct SignalCase {
    rewards: Vec<f64>,
    costs: Vec<Option<u64>>,
    mean_reward: Option<f64>,
    variance: Option<f64>,
    shrunk_variance: f64,
    mean_cost: Option<f64>,
    learning_signal: Option<f64>,
    informative: bool,
}

fn fixture() -> Fixture {
    serde_json::from_str(include_str!("../fixtures/conformance/conformance.json")).unwrap()
}

/// A seed no test depends on, printed so a failure can be replayed.
fn drawn_seed() -> u64 {
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    println!("seed {seed}");
    seed
}

/// How far, in CDF mass, an empirical bound from 1,000 resamples may sit
/// from the exact quantile: five standard errors of an empirical 2.5%
/// quantile (`sqrt(0.025 * 0.975 / 1000)` is 0.0049).
const CDF_TOLERANCE: f64 = 0.025;

/// Whether `bound` is the `q` quantile of the distribution `cdf` (support
/// points with their cumulative mass) within [`CDF_TOLERANCE`].
fn is_quantile(cdf: &[(f64, f64)], bound: f64, q: f64) -> Result<(), String> {
    let at = cdf
        .iter()
        .position(|(x, _)| (x - bound).abs() < 1e-9)
        .ok_or_else(|| format!("{bound} is no value the resampled statistic can take"))?;
    let below = if at == 0 { 0.0 } else { cdf[at - 1].1 };
    if cdf[at].1 + CDF_TOLERANCE < q || below - CDF_TOLERANCE > q {
        return Err(format!(
            "{bound} is not the {q} quantile: CDF {below} below it, {} at it",
            cdf[at].1
        ));
    }
    Ok(())
}

/// An interval method over families of pass (`1.0`) or fail (`0.0`)
/// attempts: `(value, low, high)` of the pooled pass share.
type ShareInterval<'a> = &'a dyn Fn(&[Vec<f64>], u64) -> (f64, f64, f64);

/// The first pass/fail fixture case, and bound, where `interval` diverges
/// from the exact family bootstrap.
fn check_quality_interval(interval: ShareInterval<'_>) -> Result<(), String> {
    let seed = drawn_seed();
    for case in fixture().clustered_bootstrap {
        let pass_fail = case
            .families
            .iter()
            .flatten()
            .all(|v| *v == 0.0 || *v == 1.0);
        if !pass_fail || !case.item_level_detectable {
            continue;
        }
        let (value, low, high) = interval(&case.families, seed);
        let name = &case.name;
        if (value - case.mean).abs() > 1e-12 {
            return Err(format!("{name}: value {value}, exact {}", case.mean));
        }
        let tail = (1.0 - case.level) / 2.0;
        is_quantile(&case.cdf, low, tail).map_err(|e| format!("{name}: low: {e}"))?;
        is_quantile(&case.cdf, high, 1.0 - tail).map_err(|e| format!("{name}: high: {e}"))?;
    }
    Ok(())
}

fn attempts_of(families: &[Vec<f64>]) -> Vec<Attempt> {
    let mut attempts = Vec::new();
    for (f, family) in families.iter().enumerate() {
        for (t, value) in family.iter().enumerate() {
            let name = format!("family-{f}");
            attempts.push(Attempt {
                family: name.clone(),
                outcome: TaskOutcome {
                    task_id: Digest::of(format!("{name}-{t}").as_bytes()),
                    experience_id: ExperienceId(Digest::of(format!("{name}-{t}-e").as_bytes())),
                    verified_success: *value == 1.0,
                    verification_strength: Some(Strength::Executable),
                    submitted: true,
                    wall_seconds: None,
                    gpu_seconds: None,
                    prompt_tokens: None,
                    completion_tokens: None,
                    tool_calls: None,
                    tool_errors: None,
                    attempted_actions: None,
                    unauthorised_actions: None,
                    failure: None,
                    human_minutes: None,
                },
            });
        }
    }
    attempts
}

fn splinter_quality(families: &[Vec<f64>], seed: u64) -> (f64, f64, f64) {
    let q = competence(&attempts_of(families), seed).quality;
    (q.value.unwrap(), q.low.unwrap(), q.high.unwrap())
}

#[test]
fn the_quality_interval_is_the_exact_family_bootstrap_of_the_pooled_share() {
    check_quality_interval(&splinter_quality).unwrap();
}

/// A percentile interval of `statistic` over 1,000 draws of `draw`.
fn percentile(
    seed: u64,
    mut draw: impl FnMut(&mut SplitMix64) -> f64,
) -> impl FnOnce(f64) -> (f64, f64) {
    let mut rng = SplitMix64::new(seed);
    let mut draws: Vec<f64> = (0..1000).map(|_| draw(&mut rng)).collect();
    draws.sort_by(f64::total_cmp);
    move |tail| {
        let at = |q: f64| draws[(q * 999.0).round() as usize];
        (at(tail), at(1.0 - tail))
    }
}

#[test]
fn the_quality_check_catches_an_item_level_resample_and_equally_weighted_families() {
    let pooled = |families: &[Vec<f64>]| {
        let all: Vec<f64> = families.iter().flatten().copied().collect();
        all.iter().sum::<f64>() / all.len() as f64
    };
    let item_level = |families: &[Vec<f64>], seed: u64| {
        let all: Vec<f64> = families.iter().flatten().copied().collect();
        let n = all.len();
        let (low, high) = percentile(seed, |rng| {
            (0..n).map(|_| all[rng.below(n)]).sum::<f64>() / n as f64
        })(0.025);
        (pooled(families), low, high)
    };
    assert!(
        check_quality_interval(&item_level).is_err(),
        "the quality check did not catch an item-level resample"
    );
    let equal_families = |families: &[Vec<f64>], seed: u64| {
        let k = families.len();
        let means: Vec<f64> = families
            .iter()
            .map(|f| f.iter().sum::<f64>() / f.len() as f64)
            .collect();
        let (low, high) = percentile(seed, |rng| {
            (0..k).map(|_| means[rng.below(k)]).sum::<f64>() / k as f64
        })(0.025);
        (pooled(families), low, high)
    };
    assert!(
        check_quality_interval(&equal_families).is_err(),
        "the quality check did not catch families weighted equally"
    );
}

/// What a learning-signal method reports for one group.
struct Signal {
    mean_reward: Option<f64>,
    variance: Option<f64>,
    shrunk_variance: f64,
    mean_cost: Option<f64>,
    learning_signal: Option<f64>,
    informative: bool,
}

fn rollouts(case: &SignalCase) -> Vec<Rollout> {
    case.rewards
        .iter()
        .zip(&case.costs)
        .map(|(reward, cost)| Rollout {
            reward: *reward,
            cost: *cost,
            completion: cost.map(|c| c / 2),
        })
        .collect()
}

fn splinter_signal(rollouts: &[Rollout]) -> Signal {
    let t = TaskSignal::of("task", "family", rollouts);
    Signal {
        mean_reward: t.mean_reward,
        variance: t.variance,
        shrunk_variance: t.shrunk_variance,
        mean_cost: t.mean_cost,
        learning_signal: t.learning_signal,
        informative: t.informative,
    }
}

/// A learning-signal method over one group's attempts.
type SignalOf<'a> = &'a dyn Fn(&[Rollout]) -> Signal;

/// The first fixture case, and field, where `signal_of` diverges from the
/// exact values, by relative error.
fn check_signal(signal_of: SignalOf<'_>) -> Result<(), String> {
    let close = |a: Option<f64>, b: Option<f64>| match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => (a - b).abs() <= 1e-12 * b.abs().max(1e-12),
        _ => false,
    };
    for case in fixture().learning_signal {
        let got = signal_of(&rollouts(&case));
        let fields = [
            ("mean reward", got.mean_reward, case.mean_reward),
            ("variance", got.variance, case.variance),
            (
                "shrunk variance",
                Some(got.shrunk_variance),
                Some(case.shrunk_variance),
            ),
            ("mean cost", got.mean_cost, case.mean_cost),
            ("learning signal", got.learning_signal, case.learning_signal),
        ];
        for (field, got, want) in fields {
            if !close(got, want) {
                return Err(format!(
                    "rewards {:?}: {field} {got:?}, exact {want:?}",
                    case.rewards
                ));
            }
        }
        if got.informative != case.informative {
            return Err(format!(
                "rewards {:?}: informative {}, exact {}",
                case.rewards, got.informative, case.informative
            ));
        }
    }
    Ok(())
}

#[test]
fn the_learning_signal_is_the_shrunk_reward_variance_per_token() {
    check_signal(&splinter_signal).unwrap();
}

#[test]
fn the_signal_check_catches_a_sample_variance_no_shrinkage_and_kept_flat_groups() {
    // Each mutant is splinter's own result with one defect applied, the
    // Beta prior's contribution read back from it where the defect needs it.
    let prior_part =
        |s: &Signal, n: f64| (n + 2.0) * s.shrunk_variance - n * s.variance.unwrap_or(0.0);
    let reshrunk = |s: Signal, n: f64, variance: f64| {
        let shrunk = (n * variance + prior_part(&s, n)) / (n + 2.0);
        Signal {
            shrunk_variance: shrunk,
            learning_signal: s.mean_cost.map(|c| shrunk / c),
            ..s
        }
    };
    let mutants: [(&str, SignalOf<'_>); 3] = [
        ("sample variance", &|r| {
            let s = splinter_signal(r);
            let n = r.len() as f64;
            match s.variance {
                Some(v) if n > 1.0 => reshrunk(s, n, v * n / (n - 1.0)),
                _ => s,
            }
        }),
        ("no shrinkage", &|r| {
            let s = splinter_signal(r);
            let v = s.variance.unwrap_or(0.0);
            Signal {
                shrunk_variance: v,
                learning_signal: s.mean_cost.map(|c| v / c),
                ..s
            }
        }),
        ("flat groups informative", &|r| Signal {
            informative: r.len() >= 2,
            ..splinter_signal(r)
        }),
    ];
    for (name, mutant) in mutants {
        assert!(
            check_signal(mutant).is_err(),
            "the signal check did not catch the {name} mutant"
        );
    }
}

/// The variance-per-cost draw weighs a task by `sqrt(V) / mean cost`, never
/// draws a group measured flat, and draws an unmeasured one at the fallback
/// cost.
#[test]
fn the_selection_weight_is_root_shrunk_variance_per_cost_and_zero_for_flat_groups() {
    for case in fixture().learning_signal {
        let task = TaskSignal::of("task", "family", &rollouts(&case));
        let fallback = 750.0;
        let measured_flat = case.rewards.len() >= 2 && !case.informative;
        let expected = if measured_flat {
            0.0
        } else {
            case.shrunk_variance.sqrt() / case.mean_cost.unwrap_or(fallback)
        };
        let got = selection_weight(&task, fallback);
        assert!(
            (got - expected).abs() <= 1e-12 * expected.max(1e-12),
            "rewards {:?}: weight {got}, exact {expected}",
            case.rewards
        );
    }
}
