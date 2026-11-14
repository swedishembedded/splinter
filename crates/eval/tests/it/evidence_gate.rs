// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for self-improvement loops that
// keep only the changes the evidence supports, for its clients. If your team
// needs expertise in anytime-valid testing and false-discovery control for
// model and agent gates you can procure our services by sending an email to
// info@swedishembedded.com.

//! The evidence gate's guarantees, by simulation: under a null candidate (a
//! true loss of `eps` exactly, or worse) a round accepts at most `alpha` of
//! the time, with fresh or reused dev items; selecting the best of several
//! null candidates on a reused dev set makes an uncorrected gate accept far
//! more often, and the overlap correction removes that optimism; across
//! rounds the false evolution rate stays within the budget `q`; and a true
//! gain is accepted with the power each test prints.
//!
//! Every simulation draws its seed and prints it; the bounds hold for any
//! seed with the Monte Carlo slack of three standard errors.

use std::collections::BTreeSet;

use splinter_core::rng::SplitMix64;
use splinter_eval::campaign::dev_pool::DevPool;
use splinter_eval::campaign::evidence::{Budget, EProcess, EvidenceGate};
use splinter_eval::campaign::gate::{decide_evidence, Action, DevGate, DevResult};
use splinter_eval::significance::{SignTest, Significance};

/// A seed no test depends on, printed so a failure can be replayed.
fn drawn_seed() -> u64 {
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    println!("seed {seed}");
    seed
}

/// A uniform draw in `[0, 1)`.
fn uniform(rng: &mut SplitMix64) -> f64 {
    (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64
}

/// Paired outcomes with a tie share and a mean difference: a candidate win
/// with probability `(1 - ties + mean) / 2`, a loss with `(1 - ties - mean) /
/// 2`.
#[derive(Clone, Copy)]
struct Pairs {
    ties: f64,
    mean: f64,
}

impl Pairs {
    fn draw(self, rng: &mut SplitMix64) -> f64 {
        let win = (1.0 - self.ties + self.mean) / 2.0;
        let loss = (1.0 - self.ties - self.mean) / 2.0;
        assert!(
            win >= 0.0 && loss >= 0.0,
            "a mean of {} needs fewer ties than {}",
            self.mean,
            self.ties
        );
        let r = uniform(rng);
        if r < win {
            1.0
        } else if r < win + loss {
            -1.0
        } else {
            0.0
        }
    }
}

/// Whether one round over `differences` (with exposure `zeta` each) crosses
/// at level `alpha`, and after how many observations.
fn round(gate: &EvidenceGate, alpha: f64, differences: &[(f64, f64)]) -> Option<usize> {
    let mut process = EProcess::new(gate, alpha);
    for &(d, zeta) in differences {
        match process.observe(d, zeta) {
            Ok(true) => return process.crossed_at(),
            Ok(false) => {}
            Err(e) => panic!("a simulated observation is out of range: {e}"),
        }
    }
    None
}

/// The upper end of the Monte Carlo slack around a rate `p` over `n` runs.
fn slack(p: f64, n: usize) -> f64 {
    p + 3.0 * (p * (1.0 - p) / n as f64).sqrt()
}

#[test]
fn under_the_null_at_the_margin_and_beyond_a_round_accepts_at_most_alpha() {
    let gate = EvidenceGate::new(0.10, 0.30, 0.0).unwrap();
    let (alpha, n, sims) = (0.10, 200, 4000);
    let mut rng = SplitMix64::new(drawn_seed());
    for (label, mean) in [("delta = -eps", -0.10), ("delta = -eps - 0.05", -0.15)] {
        for ties in [0.2, 0.6, 0.85] {
            let pairs = Pairs { ties, mean };
            let accepted = (0..sims)
                .filter(|_| {
                    let d: Vec<(f64, f64)> = (0..n).map(|_| (pairs.draw(&mut rng), 0.0)).collect();
                    round(&gate, alpha, &d).is_some()
                })
                .count();
            let rate = accepted as f64 / sims as f64;
            println!("null {label}, ties {ties}: false acceptance {rate:.4} (alpha {alpha})");
            assert!(rate <= slack(alpha, sims), "{label}, ties {ties}: {rate}");
        }
    }
}

#[test]
fn reused_items_within_the_allowed_optimism_stay_below_alpha() {
    // Assumption 1: on an item an earlier round used, the candidate's mean
    // may be optimistic by up to beta. A pool of 300, 150 used; each round
    // draws 100. Fresh items sit at the null exactly; reused ones beat it by
    // the full allowance.
    let (alpha, sims, optimism) = (0.10, 3000, 0.03);
    let used: BTreeSet<String> = (0..150).map(|i| format!("d{i:03}")).collect();
    let pool = DevPool::new((0..300).map(|i| format!("d{i:03}")), &used);
    let mut rng = SplitMix64::new(drawn_seed());
    let fresh = Pairs {
        ties: 0.6,
        mean: -0.10,
    };
    let reused = Pairs {
        ties: 0.6,
        mean: -0.10 + 2.0 * optimism,
    };
    for beta in [optimism, 0.0] {
        let gate = EvidenceGate::new(0.10, 0.30, beta).unwrap();
        let accepted = (0..sims)
            .filter(|_| {
                let drawn = pool.draw(100, &mut rng).unwrap();
                let d: Vec<(f64, f64)> = drawn
                    .iter()
                    .map(|item| {
                        let pairs = if item.reused { reused } else { fresh };
                        (pairs.draw(&mut rng), item.zeta)
                    })
                    .collect();
                round(&gate, alpha, &d).is_some()
            })
            .count();
        let rate = accepted as f64 / sims as f64;
        println!("reused half, optimism {optimism}, beta {beta}: false acceptance {rate:.4}");
        if beta >= optimism {
            assert!(rate <= slack(alpha, sims), "beta {beta}: {rate}");
        }
    }
}

#[test]
fn selection_on_a_reused_dev_set_inflates_an_uncorrected_gate_and_the_correction_removes_it() {
    // Every round reuses the whole dev set (zeta = 1, the RSI3 regime), and
    // the candidate is the best of eight null candidates by net dev wins on
    // that same set: the selection the overlap correction is for.
    let (alpha, n, candidates, sims) = (0.10, 200, 8, 2000);
    let null = Pairs {
        ties: 0.2,
        mean: -0.10,
    };
    let mut rng = SplitMix64::new(drawn_seed());
    let (mut greedy, mut uncorrected, mut corrected) = (0, 0, 0);
    let plain = EvidenceGate::new(0.10, 0.30, 0.0).unwrap();
    let correcting = EvidenceGate::new(0.10, 0.30, 0.06).unwrap();
    for _ in 0..sims {
        let best = (0..candidates)
            .map(|_| (0..n).map(|_| null.draw(&mut rng)).collect::<Vec<f64>>())
            .max_by(|a, b| a.iter().sum::<f64>().total_cmp(&b.iter().sum::<f64>()))
            .unwrap();
        greedy += usize::from(best.iter().sum::<f64>() > 0.0);
        let mut order = best.clone();
        rng.shuffle(&mut order);
        let d: Vec<(f64, f64)> = order.iter().map(|&x| (x, 1.0)).collect();
        uncorrected += usize::from(round(&plain, alpha, &d).is_some());
        corrected += usize::from(round(&correcting, alpha, &d).is_some());
    }
    let rate = |count: usize| count as f64 / sims as f64;
    println!(
        "best of {candidates} nulls on a reused set: greedy {:.4}, beta 0 {:.4}, beta 0.06 {:.4} (alpha {alpha})",
        rate(greedy),
        rate(uncorrected),
        rate(corrected)
    );
    assert!(
        rate(uncorrected) > 1.3 * alpha,
        "selection on reused items is optimistic: {}",
        rate(uncorrected)
    );
    assert!(rate(corrected) <= slack(alpha, sims), "{}", rate(corrected));
    assert!(rate(greedy) > rate(uncorrected));
}

#[test]
fn a_true_gain_is_accepted_with_the_power_printed_and_usually_before_the_draw_runs_out() {
    let (alpha, n, sims) = (0.10, 200, 2000);
    let gain = Pairs {
        ties: 0.6,
        mean: 0.05,
    };
    let mut rng = SplitMix64::new(drawn_seed());
    let mut powers = Vec::new();
    for (beta, zeta) in [(0.0, 0.0), (0.01, 1.0)] {
        let gate = EvidenceGate::new(0.10, 0.30, beta).unwrap();
        let stops: Vec<usize> = (0..sims)
            .filter_map(|_| {
                let d: Vec<(f64, f64)> = (0..n).map(|_| (gain.draw(&mut rng), zeta)).collect();
                round(&gate, alpha, &d)
            })
            .collect();
        let power = stops.len() as f64 / sims as f64;
        let mean_stop = stops.iter().sum::<usize>() as f64 / stops.len().max(1) as f64;
        println!(
            "true gain +0.05, beta {beta}, zeta {zeta}: power {power:.4}, mean items to accept {mean_stop:.1} of {n}"
        );
        powers.push(power);
        assert!(mean_stop < n as f64);
    }
    assert!(powers[0] >= 0.75, "{powers:?}");
    assert!(powers[1] <= powers[0] + 0.05, "the correction costs power");
}

/// A sign test the property tests do not read.
struct Unread;

impl Significance for Unread {
    fn sign_test(&self, _: &[(bool, bool)]) -> SignTest {
        SignTest {
            discordant: 0,
            candidate_wins: 0,
            p_value: 1.0,
        }
    }
}

/// The pass and fail of a pair with difference `d`.
fn passes(d: f64, rng: &mut SplitMix64) -> (bool, bool) {
    match d {
        x if x > 0.0 => (true, false),
        x if x < 0.0 => (false, true),
        _ => {
            let both = rng.coin();
            (both, both)
        }
    }
}

#[test]
fn across_rounds_the_false_evolution_rate_stays_within_the_budget() {
    // Twelve rounds over a learner pool of 1000, 150 drawn each round and
    // every graded item spent; reused items carry the optimism beta allows.
    // Each round's candidate is a null at the margin or a true gain, by a
    // coin; FER is false acceptances over acceptances, averaged over runs.
    let (rounds, pool_size, draw, sims) = (12, 1000, 150, 400);
    let gate = EvidenceGate::new(0.10, 0.30, 0.02).unwrap();
    let mut rng = SplitMix64::new(drawn_seed());
    for share_null in [1.0, 0.5] {
        let (mut fer, mut true_accepted, mut true_rounds) = (0.0, 0usize, 0usize);
        for _ in 0..sims {
            let mut pool =
                DevPool::new((0..pool_size).map(|i| format!("d{i:04}")), &BTreeSet::new());
            let mut budget = Budget::start(&gate);
            let (mut accepted, mut false_accepted) = (0usize, 0usize);
            for _ in 0..rounds {
                let null = uniform(&mut rng) < share_null;
                let mean = if null { -0.10 } else { 0.05 };
                let allocation = budget.allocate(&gate, rounds);
                let drawn = pool.draw(draw, &mut rng).unwrap();
                let (mut champion, mut candidate) = (DevResult::default(), DevResult::default());
                for item in &drawn {
                    let lift = if item.reused { 2.0 * gate.beta } else { 0.0 };
                    let d = Pairs {
                        ties: 0.6,
                        mean: mean + lift,
                    }
                    .draw(&mut rng);
                    let (c, b) = passes(d, &mut rng);
                    candidate.dev.insert(item.task.clone(), c);
                    champion.dev.insert(item.task.clone(), b);
                }
                let decision = decide_evidence(
                    &DevGate::default(),
                    &gate,
                    &allocation,
                    (&drawn, drawn.len()),
                    &champion,
                    &candidate,
                    &Unread,
                )
                .unwrap();
                let advanced = decision.action == Action::Advance;
                let record = decision.evidence.unwrap();
                pool.spend(&record.spent);
                budget.settle(&allocation, advanced);
                accepted += usize::from(advanced);
                false_accepted += usize::from(advanced && null);
                if !null {
                    true_rounds += 1;
                    true_accepted += usize::from(advanced);
                }
            }
            assert!(budget.spent(&gate) <= gate.q + 1e-12);
            fer += false_accepted as f64 / accepted.max(1) as f64;
        }
        let fer = fer / sims as f64;
        let ter = true_accepted as f64 / true_rounds.max(1) as f64;
        println!(
            "{rounds} rounds, null share {share_null}: FER {fer:.4} (q {}), true gains accepted {ter:.4}",
            gate.q
        );
        assert!(fer <= slack(gate.q, sims), "FER {fer}");
    }
}
