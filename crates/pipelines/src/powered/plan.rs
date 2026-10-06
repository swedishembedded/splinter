// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements power planning for paired model
// examinations, for its clients. If your team needs expertise in sizing an
// exam so that a real gain is seen and luck is not, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Sizing an exam before it is paid for: the planned paired test simulated
//! under what is assumed of the arms, so that a sample size is frozen on
//! evidence and not on a rule of thumb.
//!
//! The model is the plain one. A task's difference between the two arms is
//! +1 (only the first right), -1 (only the second) or 0. Overall, a share
//! `discordance` of tasks are +1 or -1 and the first leads by `effect` in the
//! share right. Families differ: the mean difference of a family is the
//! overall one plus a draw with variance `icc` times the variance of a task's
//! difference, so that a family's tasks are alike in how the arms differ as
//! much as the intraclass correlation says. The test simulated is the one the
//! exam reports as its primary: the first arm is better when the lower end of
//! the family-clustered bootstrap interval of the difference is above zero.

use serde::{Deserialize, Serialize};

use super::analysis::interval_of;

/// What is assumed of the exam and the arms.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Plan {
    /// Families.
    pub families: usize,
    /// Tasks of each.
    pub tasks_per_family: usize,
    /// The first arm's lead in the share of tasks right (0.10 is ten points).
    pub effect: f64,
    /// The share of tasks only one of the arms gets right.
    pub discordance: f64,
    /// The intraclass correlation of the per-task difference in a family.
    pub icc: f64,
    /// Simulated exams.
    pub replicates: usize,
    /// Bootstrap resamples inside each.
    pub resamples: usize,
    /// Seeds the simulation: the same plan gives the same answer.
    pub seed: u64,
}

/// What the plan came to.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Planned {
    /// Tasks in all.
    pub tasks: usize,
    /// How much clustering inflates the variance of the mean difference
    /// over independent tasks: `1 + (tasks per family - 1) * icc`.
    pub design_effect: f64,
    /// The share of simulated exams in which the planned test found the first
    /// arm better: its power at this effect.
    pub power: f64,
    /// The same with no effect: how often the test calls a difference that is
    /// not there, which should be near 0.025 for a two-sided 95% interval
    /// read one way.
    pub false_positive_rate: f64,
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal, by Box-Muller.
    fn normal(&mut self) -> f64 {
        let (a, b) = (self.unit().max(1e-12), self.unit());
        (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
    }
}

/// One simulated exam's families as `(sum of differences, tasks)`.
fn exam(plan: &Plan, effect: f64, rng: &mut Rng) -> Vec<(f64, usize)> {
    let variance = plan.discordance.max(1e-9);
    let between = (plan.icc.clamp(0.0, 0.99) * variance).sqrt();
    (0..plan.families)
        .map(|_| {
            let mean = (effect + between * rng.normal()).clamp(-plan.discordance, plan.discordance);
            // +1 and -1 with the family's mean and the overall discordance.
            let up = (plan.discordance + mean) / 2.0;
            let down = (plan.discordance - mean) / 2.0;
            let sum: f64 = (0..plan.tasks_per_family)
                .map(|_| {
                    let u = rng.unit();
                    if u < up {
                        1.0
                    } else if u < up + down {
                        -1.0
                    } else {
                        0.0
                    }
                })
                .sum();
            (sum, plan.tasks_per_family)
        })
        .collect()
}

/// Simulates `plan`.
#[must_use]
pub fn simulate(plan: &Plan) -> Planned {
    let mut rng = Rng(plan.seed);
    let mut found = |effect: f64| -> f64 {
        let hits = (0..plan.replicates)
            .filter(|_| {
                interval_of(exam(plan, effect, &mut rng), plan.resamples)
                    .is_some_and(|interval| interval.low > 0.0)
            })
            .count();
        hits as f64 / plan.replicates.max(1) as f64
    };
    let power = found(plan.effect);
    let false_positive_rate = found(0.0);
    Planned {
        tasks: plan.families * plan.tasks_per_family,
        design_effect: 1.0 + (plan.tasks_per_family.saturating_sub(1)) as f64 * plan.icc,
        power,
        false_positive_rate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(families: usize, effect: f64) -> Plan {
        Plan {
            families,
            tasks_per_family: 8,
            effect,
            discordance: 0.30,
            icc: 0.10,
            replicates: 300,
            resamples: 300,
            seed: 7,
        }
    }

    #[test]
    fn power_grows_with_the_families_and_a_null_effect_is_rarely_called() {
        let small = simulate(&plan(10, 0.10));
        let large = simulate(&plan(50, 0.10));
        assert!(large.power > small.power, "{small:?} {large:?}");
        assert!(
            large.power > 0.6,
            "fifty families of eight see ten points: {large:?}"
        );
        assert!(large.false_positive_rate < 0.08, "{large:?}");
        assert_eq!(large.tasks, 400);
        assert!((large.design_effect - 1.7).abs() < 1e-12);
    }

    #[test]
    fn the_same_plan_gives_the_same_answer() {
        assert_eq!(simulate(&plan(20, 0.10)), simulate(&plan(20, 0.10)));
    }
}
