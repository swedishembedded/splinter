// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for spending a reinforcement-learning
// budget only on prompts that teach, for its clients. If your team needs
// expertise in dynamic sampling for policy-gradient training you can
// procure our services by sending an email to info@swedishembedded.com.

//! Dynamic sampling: which prompts to roll out, and which groups of
//! rollouts enter a training batch.
//!
//! A group whose attempts all earned the same reward has a zero advantage
//! everywhere: it costs a rollout and teaches nothing. Only groups with a
//! reward spread enter a batch ([`Group::kind`]). Because the share of such
//! groups is unknown in advance and moves with the policy, a batch is built
//! by over-provisioning and refilling ([`BatchBuilder`]): ask for more
//! groups than the batch needs, keep the informative ones, and ask again for
//! what is still missing, up to a bounded number of rounds. The share of
//! informative groups among everything offered is reported
//! ([`Batch::informative_fraction`]); it is the first number a gate reads,
//! because a run whose groups carry no spread cannot move the policy however
//! long it trains.
//!
//! Groups that all failed alike are the cliff: no advantage exists, but the
//! task is not solved. They are routed to a strategy ([`AllFailStrategy`])
//! instead of being silently dropped. Groups that all passed are counted and
//! their tasks reported so the campaign can retire them
//! ([`super::pass_rate`]).

use serde::{Deserialize, Serialize};
use splinter_core::rng::SplitMix64;

use super::pass_rate::{Band, PassRates};

/// Rewards closer than this are the same reward.
pub const SPREAD_EPSILON: f64 = 1e-9;

/// How likely, relative to a frontier task, a task never solved at the
/// calibration depth is to be drawn: it is revisited rarely, since the policy
/// may have learned it since.
pub const CLIFF_WEIGHT: f64 = 0.1;

/// The share of groups expected to be informative before any is seen.
const PRIOR_INFORMATIVE_SHARE: f64 = 0.5;

/// Pseudo-groups the prior counts for.
const PRIOR_STRENGTH: f64 = 2.0;

/// One attempt of a group.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Attempt {
    /// The reward it earned.
    pub reward: f64,
    /// Whether the verifier called it a pass.
    pub passed: bool,
}

/// The attempts of one policy at one task.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Group {
    /// The task.
    pub task: String,
    /// Its graded attempts; ungraded ones are not in a group.
    pub attempts: Vec<Attempt>,
}

/// What a group can teach.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GroupKind {
    /// The attempts earned different rewards: advantages exist.
    Informative,
    /// Every attempt passed and earned the same: nothing to compare.
    AllPass,
    /// No attempt passed and all earned the same: the cliff.
    AllFail,
    /// Fewer than two graded attempts: nothing to compare.
    TooSmall,
}

impl Group {
    /// What the group can teach.
    #[must_use]
    pub fn kind(&self) -> GroupKind {
        if self.attempts.len() < 2 {
            return GroupKind::TooSmall;
        }
        let (low, high) = self
            .attempts
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), a| {
                (low.min(a.reward), high.max(a.reward))
            });
        if high - low > SPREAD_EPSILON {
            GroupKind::Informative
        } else if self.attempts.iter().all(|a| a.passed) {
            GroupKind::AllPass
        } else {
            GroupKind::AllFail
        }
    }
}

/// What to do with a task whose group all failed alike.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AllFailStrategy {
    /// Leave it: it stays in the pool at a low weight.
    Skip,
    /// A teacher's verified answer, imitated.
    Teacher,
    /// The attempts distilled per token from the same policy shown what the
    /// verifier knows (see the pipelines' distillation strategy).
    CliffDistill,
    /// Split into easier sub-tasks that join the pool.
    Decompose,
}

impl std::str::FromStr for AllFailStrategy {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, String> {
        match text {
            "skip" => Ok(Self::Skip),
            "teacher" => Ok(Self::Teacher),
            "cliff-distill" | "cliff_distill" => Ok(Self::CliffDistill),
            "decompose" | "decomposition" => Ok(Self::Decompose),
            other => Err(format!(
                "{other:?} is not an all-fail strategy: skip, teacher, cliff-distill or decompose"
            )),
        }
    }
}

/// How a batch is built.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BatchPolicy {
    /// Informative groups a batch holds.
    pub target_groups: usize,
    /// How many times the expected need each request asks for: at least 1.
    pub overprovision: f64,
    /// Most requests a batch may take; a batch short of its target after them
    /// is reported short, never padded.
    pub max_rounds: usize,
    /// Most groups one request may ask for.
    pub max_request: usize,
}

/// A refused batch policy.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct BatchPolicyError(String);

impl BatchPolicy {
    /// Refuses a policy that could not build a batch.
    ///
    /// # Errors
    /// A target of zero, an over-provision below one or not finite, no
    /// rounds, or a request limit below the target.
    pub fn validate(&self) -> Result<(), BatchPolicyError> {
        let refuse = |why: String| Err(BatchPolicyError(why));
        if self.target_groups == 0 {
            return refuse("a batch needs at least one group".into());
        }
        if !(self.overprovision.is_finite() && self.overprovision >= 1.0) {
            return refuse(format!(
                "overprovision is {}; it is a factor of at least 1",
                self.overprovision
            ));
        }
        if self.max_rounds == 0 {
            return refuse("a batch needs at least one round".into());
        }
        if self.max_request < self.target_groups {
            return refuse(format!(
                "max_request is {}, below the {} groups a batch needs",
                self.max_request, self.target_groups
            ));
        }
        Ok(())
    }
}

/// A batch under construction.
#[derive(Clone, Debug)]
pub struct BatchBuilder {
    policy: BatchPolicy,
    kept: Vec<Group>,
    surplus: usize,
    informative_seen: usize,
    offered: usize,
    all_fail: Vec<Group>,
    all_pass: Vec<String>,
    too_small: usize,
    rounds: usize,
}

impl BatchBuilder {
    /// An empty batch.
    ///
    /// # Errors
    /// The policy is refused.
    pub fn new(policy: BatchPolicy) -> Result<Self, BatchPolicyError> {
        policy.validate()?;
        Ok(Self {
            policy,
            kept: Vec::new(),
            surplus: 0,
            informative_seen: 0,
            offered: 0,
            all_fail: Vec::new(),
            all_pass: Vec::new(),
            too_small: 0,
            rounds: 0,
        })
    }

    /// The informative share expected of the next groups: what was seen so
    /// far, smoothed by a weak prior so the first request is not unbounded.
    fn expected_share(&self) -> f64 {
        (self.informative_seen as f64 + PRIOR_INFORMATIVE_SHARE * PRIOR_STRENGTH)
            / (self.offered as f64 + PRIOR_STRENGTH)
    }

    /// How many groups to collect next; `None` when the batch is full or the
    /// rounds are used up.
    #[must_use]
    pub fn next_request(&self) -> Option<usize> {
        let missing = self.policy.target_groups.saturating_sub(self.kept.len());
        if missing == 0 || self.rounds >= self.policy.max_rounds {
            return None;
        }
        let share = self
            .expected_share()
            .max(1.0 / self.policy.max_request as f64);
        let wanted = (missing as f64 * self.policy.overprovision / share).ceil() as usize;
        Some(wanted.clamp(1, self.policy.max_request))
    }

    /// Takes the groups one request returned.
    pub fn offer(&mut self, groups: Vec<Group>) {
        self.rounds += 1;
        for group in groups {
            self.offered += 1;
            match group.kind() {
                GroupKind::Informative => {
                    self.informative_seen += 1;
                    if self.kept.len() < self.policy.target_groups {
                        self.kept.push(group);
                    } else {
                        self.surplus += 1;
                    }
                }
                GroupKind::AllPass => self.all_pass.push(group.task),
                GroupKind::AllFail => self.all_fail.push(group),
                GroupKind::TooSmall => self.too_small += 1,
            }
        }
    }

    /// The batch as built.
    #[must_use]
    pub fn finish(self) -> Batch {
        Batch {
            shortfall: self.policy.target_groups.saturating_sub(self.kept.len()),
            informative: self.kept,
            surplus: self.surplus,
            informative_seen: self.informative_seen,
            offered: self.offered,
            all_fail: self.all_fail,
            all_pass: self.all_pass,
            too_small: self.too_small,
            rounds: self.rounds,
        }
    }
}

/// A finished batch and what was learned about the pool while building it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Batch {
    /// The informative groups, at most the target.
    pub informative: Vec<Group>,
    /// Informative groups collected beyond the target and left out.
    pub surplus: usize,
    /// Informative groups seen, kept or not.
    pub informative_seen: usize,
    /// Groups offered in all.
    pub offered: usize,
    /// The groups that all failed alike, for [`AllFailStrategy`].
    pub all_fail: Vec<Group>,
    /// The tasks whose groups all passed.
    pub all_pass: Vec<String>,
    /// Groups of fewer than two graded attempts.
    pub too_small: usize,
    /// Requests it took.
    pub rounds: usize,
    /// Informative groups short of the target after the last round; zero when
    /// the batch is full.
    pub shortfall: usize,
}

impl Batch {
    /// The share of offered groups that were informative; `None` when none
    /// was offered (unmeasured, not zero).
    #[must_use]
    pub fn informative_fraction(&self) -> Option<f64> {
        (self.offered > 0).then(|| self.informative_seen as f64 / self.offered as f64)
    }
}

/// `count` tasks of `pool` to roll out next, never a retired one.
///
/// Tasks not yet visited come first (fresh prompts before revisits: a task
/// revisited with the same prompt lets negative advantages memorise it), then
/// those visited least; among equals the draw is weighted, frontier and
/// unmeasured tasks at 1 and cliff tasks at [`CLIFF_WEIGHT`], without
/// replacement. `rng` is the campaign's, seeded from a recorded seed.
#[must_use]
pub fn draw_prompts(
    pool: &[String],
    rates: &PassRates,
    count: usize,
    rng: &mut SplitMix64,
) -> Vec<String> {
    let mut keyed: Vec<(u64, f64, &String)> = pool
        .iter()
        .filter_map(|task| {
            let weight = match rates.band(task) {
                Band::Retired => return None,
                Band::Cliff => CLIFF_WEIGHT,
                Band::Unmeasured | Band::Frontier => 1.0,
            };
            // Efraimidis-Spirakis: the larger u^(1/w), the likelier to be drawn.
            let uniform = ((rng.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
            Some((rates.visits(task), uniform.powf(1.0 / weight), task))
        })
        .collect();
    keyed.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.total_cmp(&a.1)));
    keyed
        .into_iter()
        .take(count)
        .map(|(_, _, task)| task.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A seed no test depends on, printed so a failure can be replayed.
    fn drawn_seed() -> u64 {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64);
        println!("seed {seed}");
        seed
    }

    fn attempt(reward: f64, passed: bool) -> Attempt {
        Attempt { reward, passed }
    }

    fn group(task: &str, rewards: &[f64]) -> Group {
        Group {
            task: task.into(),
            attempts: rewards.iter().map(|&r| attempt(r, r >= 1.0)).collect(),
        }
    }

    fn policy(target: usize) -> BatchPolicy {
        BatchPolicy {
            target_groups: target,
            overprovision: 1.5,
            max_rounds: 3,
            max_request: 64,
        }
    }

    #[test]
    fn only_a_reward_spread_makes_a_group_informative() {
        assert_eq!(group("a", &[1.0, 0.0, 0.0]).kind(), GroupKind::Informative);
        assert_eq!(
            group("b", &[0.2, 0.7]).kind(),
            GroupKind::Informative,
            "partial credit among failures is a spread"
        );
        assert_eq!(group("c", &[1.0, 1.0, 1.0]).kind(), GroupKind::AllPass);
        assert_eq!(group("d", &[0.0, 0.0]).kind(), GroupKind::AllFail);
        assert_eq!(
            group("e", &[0.3, 0.3]).kind(),
            GroupKind::AllFail,
            "equal partial credit and no pass is still the cliff"
        );
        assert_eq!(group("f", &[1.0]).kind(), GroupKind::TooSmall);
        assert_eq!(group("g", &[]).kind(), GroupKind::TooSmall);
    }

    #[test]
    fn a_batch_keeps_informative_groups_and_sorts_the_rest() {
        let mut batch = BatchBuilder::new(policy(2)).unwrap();
        batch.offer(vec![
            group("solved", &[1.0, 1.0]),
            group("cliff", &[0.0, 0.0]),
            group("mixed-1", &[1.0, 0.0]),
            group("mixed-2", &[0.0, 1.0]),
            group("mixed-3", &[1.0, 0.0]),
            group("one", &[1.0]),
        ]);
        assert_eq!(batch.next_request(), None, "the batch is full");
        let batch = batch.finish();
        assert_eq!(
            batch
                .informative
                .iter()
                .map(|g| g.task.as_str())
                .collect::<Vec<_>>(),
            ["mixed-1", "mixed-2"]
        );
        assert_eq!(batch.surplus, 1);
        assert_eq!(batch.all_pass, ["solved"]);
        assert_eq!(batch.all_fail.len(), 1);
        assert_eq!(batch.too_small, 1);
        assert_eq!(batch.shortfall, 0);
        assert_eq!(batch.informative_fraction(), Some(3.0 / 6.0));
    }

    #[test]
    fn a_short_batch_is_refilled_then_reported_short_never_padded() {
        let mut batch = BatchBuilder::new(policy(4)).unwrap();
        let first = batch.next_request().unwrap();
        assert!(
            first >= 6,
            "4 groups at an expected half share, over-provisioned: {first}"
        );
        batch.offer(vec![group("m", &[1.0, 0.0]), group("c", &[0.0, 0.0])]);
        let refill = batch.next_request().unwrap();
        assert!(refill >= 3, "three groups are still missing: {refill}");
        batch.offer(vec![group("c2", &[0.0, 0.0])]);
        batch.offer(vec![group("c3", &[0.0, 0.0])]);
        assert_eq!(batch.next_request(), None, "the rounds are used up");
        let batch = batch.finish();
        assert_eq!(batch.informative.len(), 1);
        assert_eq!(batch.shortfall, 3);
        assert_eq!(batch.rounds, 3);
        assert_eq!(batch.informative_fraction(), Some(0.25));
    }

    #[test]
    fn a_pool_with_no_spread_asks_for_more_but_within_the_limit() {
        let mut batch = BatchBuilder::new(BatchPolicy {
            max_request: 10,
            ..policy(4)
        })
        .unwrap();
        for _ in 0..2 {
            batch.offer(
                (0..8)
                    .map(|i| group(&format!("c{i}"), &[0.0, 0.0]))
                    .collect(),
            );
        }
        assert_eq!(batch.next_request(), Some(10), "capped at max_request");
        assert_eq!(
            BatchBuilder::new(policy(1))
                .unwrap()
                .finish()
                .informative_fraction(),
            None
        );
    }

    #[test]
    fn a_policy_that_cannot_build_a_batch_is_refused() {
        for bad in [
            BatchPolicy {
                target_groups: 0,
                ..policy(1)
            },
            BatchPolicy {
                overprovision: 0.5,
                ..policy(1)
            },
            BatchPolicy {
                overprovision: f64::NAN,
                ..policy(1)
            },
            BatchPolicy {
                max_rounds: 0,
                ..policy(1)
            },
            BatchPolicy {
                max_request: 1,
                ..policy(2)
            },
        ] {
            assert!(BatchBuilder::new(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn prompts_are_drawn_fresh_first_and_never_retired() {
        let mut rates = PassRates::default();
        let pool: Vec<String> = (0..10).map(|i| format!("t{i}")).collect();
        let ok = |p: usize, f: usize| -> Vec<Option<bool>> {
            std::iter::repeat_n(Some(true), p)
                .chain(std::iter::repeat_n(Some(false), f))
                .collect()
        };
        rates.observe("t0", 0, &ok(8, 0));
        for task in ["t1", "t2", "t3"] {
            rates.observe(task, 0, &ok(4, 4));
        }
        let mut rng = SplitMix64::new(7); // seed: the test pins one draw order
        let drawn = draw_prompts(&pool, &rates, 6, &mut rng);
        assert_eq!(drawn.len(), 6);
        assert!(!drawn.contains(&"t0".to_string()), "retired: {drawn:?}");
        for task in ["t1", "t2", "t3"] {
            assert!(
                !drawn.contains(&task.to_string()),
                "visited tasks come after the six unvisited ones: {drawn:?}"
            );
        }
        let everything = draw_prompts(&pool, &rates, 100, &mut SplitMix64::new(drawn_seed()));
        assert_eq!(everything.len(), 9, "the pool minus the retired task");
    }

    #[test]
    fn a_cliff_task_is_drawn_less_often_than_a_frontier_task() {
        let mut rates = PassRates::default();
        let ok = |p: usize, f: usize| -> Vec<Option<bool>> {
            std::iter::repeat_n(Some(true), p)
                .chain(std::iter::repeat_n(Some(false), f))
                .collect()
        };
        rates.observe("cliff", 0, &ok(0, 8));
        rates.observe("mixed", 0, &ok(4, 4));
        let pool = vec!["cliff".to_string(), "mixed".to_string()];
        let mut rng = SplitMix64::new(drawn_seed());
        let first = (0..400)
            .filter(|_| draw_prompts(&pool, &rates, 1, &mut rng)[0] == "cliff")
            .count();
        assert!(first < 100, "cliff drawn first {first} of 400 times");
    }

    #[test]
    fn strategies_are_read_from_their_names() {
        assert_eq!("cliff-distill".parse(), Ok(AllFailStrategy::CliffDistill));
        assert_eq!("teacher".parse(), Ok(AllFailStrategy::Teacher));
        assert_eq!("decompose".parse(), Ok(AllFailStrategy::Decompose));
        assert_eq!("skip".parse(), Ok(AllFailStrategy::Skip));
        assert!("ignore".parse::<AllFailStrategy>().is_err());
    }
}
