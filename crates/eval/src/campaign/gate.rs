// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for keeping a continuously trained
// model from drifting worse while it is being improved, for its clients. If
// your team needs expertise in dev-set gating and rollback for training
// campaigns you can procure our services by sending an email to
// info@swedishembedded.com.

//! The dev-validation gate of a campaign: after every training arm, the
//! candidate and the champion are compared task by task on dev tasks the
//! campaign never trains on, and the candidate either becomes the champion or
//! is rolled back.
//!
//! This is the inner loop's gate. It is not a release gate: only an
//! independent, sealed exam moves a release alias. What it must do is stop a
//! run that is getting worse before it spends more compute, so it advances on
//! a net gain of dev passes (more tasks won than lost) and refuses any arm that
//! loses more retention passes than allowed. Whether the gain is also
//! significant at the level this gate may spend ([`spent_alpha`]) is reported
//! beside the decision and recorded in the ledger, so a run of point-estimate
//! advances is visible as one, and repeated gates spend alpha instead of
//! running ten tests at 5 percent.
//!
//! That is the greedy gate. [`decide_evidence`] is the opt-in alternative: a
//! noninferiority e-process over dev items drawn per round from a pool
//! ([`super::evidence`], [`super::dev_pool`]), which advances only when the
//! evidence crosses the round's level and so controls the false evolution
//! rate across rounds. Both apply the same retention check.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::dev_pool::Drawn;
use super::evidence::{Allocation, EProcess, EvidenceError, EvidenceGate};
use super::plan::{spent_alpha, DEFAULT_TOTAL_ALPHA};
use crate::significance::Significance;

/// Pass or fail of each task, by task id.
pub type Outcomes = BTreeMap<String, bool>;

/// What one policy did on the validation tasks.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DevResult {
    /// The dev tasks, which no arm trains on.
    pub dev: Outcomes,
    /// Tasks of earlier work the campaign must not forget; may be empty.
    pub retention: Outcomes,
}

/// What the gate requires.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DevGate {
    /// The level the whole sequence of gates spends.
    pub total_alpha: f64,
    /// Retention passes the candidate may lose against the champion; zero
    /// means any net loss rolls the arm back.
    pub max_retention_loss: usize,
    /// A non-inferiority margin: retention passes lost, as a share of the
    /// retention set, that are still allowed. The larger of this and
    /// `max_retention_loss` applies.
    pub max_retention_loss_rate: f64,
}

impl Default for DevGate {
    fn default() -> Self {
        Self {
            total_alpha: DEFAULT_TOTAL_ALPHA,
            max_retention_loss: 0,
            max_retention_loss_rate: 0.0,
        }
    }
}

/// What the gate does with the candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Action {
    /// The candidate becomes the champion.
    Advance,
    /// The champion stays.
    Rollback,
}

/// A gate decision with the numbers behind it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Decision {
    /// What to do.
    pub action: Action,
    /// Whether the dev gain is significant at the level this gate spends.
    pub significant: bool,
    /// Dev tasks the candidate passed and the champion failed.
    pub wins: usize,
    /// Dev tasks the champion passed and the candidate failed.
    pub losses: usize,
    /// Dev tasks both were graded on.
    pub compared: usize,
    /// The one-sided sign-test p-value of the dev gain.
    pub p_value: f64,
    /// The level this gate may spend.
    pub alpha: f64,
    /// Retention passes the candidate lost.
    pub retention_losses: usize,
    /// The candidate's dev pass rate; `None` with no dev task.
    pub candidate_rate: Option<f64>,
    /// The champion's dev pass rate; `None` with no dev task.
    pub champion_rate: Option<f64>,
    /// Why, in one sentence.
    pub reason: String,
    /// The evidence gate's numbers, when it decided; absent for the greedy
    /// gate, whose records stay as they were.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence: Option<EvidenceRecord>,
}

/// Where a round's evidence stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceStatus {
    /// The e-value reached the threshold: the round stopped there.
    Crossed,
    /// Below the threshold with drawn items still ungraded: grade more.
    Open,
    /// Below the threshold after every drawn item: the incumbent stays.
    Exhausted,
    /// The round's level is zero: nothing is graded, the incumbent stays.
    Skipped,
}

/// What the evidence gate measured in one round, as its decision record
/// carries it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvidenceRecord {
    /// The gate's parameters.
    pub gate: EvidenceGate,
    /// The round's level and budget.
    pub allocation: Allocation,
    /// Where the evidence stands.
    pub status: EvidenceStatus,
    /// The mixture e-value at the stop (at the crossing when crossed).
    pub e_value: f64,
    /// The threshold, `1 / alpha`.
    pub threshold: f64,
    /// Dev items drawn for the round.
    pub drawn: usize,
    /// Drawn items graded so far.
    pub graded: usize,
    /// Paired observations the e-process took (up to the crossing).
    pub observed: usize,
    /// Graded items without an outcome for both policies; not observed.
    pub unpaired: usize,
    /// Of the observed items, those an earlier round used.
    pub reused: usize,
    /// The mean exposure probability of the observed draws.
    pub zeta_mean: Option<f64>,
    /// The dev items the round spent: every one it graded.
    pub spent: Vec<String>,
}

fn rate(outcomes: &Outcomes) -> Option<f64> {
    (!outcomes.is_empty())
        .then(|| outcomes.values().filter(|passed| **passed).count() as f64 / outcomes.len() as f64)
}

/// Decides whether `candidate` replaces `champion` at gate number
/// `generation` (from 1).
#[must_use]
pub fn decide(
    gate: &DevGate,
    generation: u32,
    champion: &DevResult,
    candidate: &DevResult,
    test: &dyn Significance,
) -> Decision {
    let pairs: Vec<(bool, bool)> = candidate
        .dev
        .iter()
        .filter_map(|(task, &c)| champion.dev.get(task).map(|&b| (c, b)))
        .collect();
    let wins = pairs.iter().filter(|(c, b)| *c && !*b).count();
    let losses = pairs.iter().filter(|(c, b)| !*c && *b).count();
    let (retention_losses, allowed) = retention(gate, champion, candidate);
    let alpha = spent_alpha(gate.total_alpha, generation);
    let sign = test.sign_test(&pairs);
    let (action, reason) = if pairs.is_empty() {
        (
            Action::Rollback,
            "no dev task was graded for both the champion and the candidate".to_string(),
        )
    } else if retention_losses > allowed {
        (
            Action::Rollback,
            format!(
                "the candidate lost {retention_losses} retention passes, more than the {} allowed",
                allowed
            ),
        )
    } else if wins > losses {
        (
            Action::Advance,
            format!("the candidate won {wins} dev tasks and lost {losses}"),
        )
    } else {
        (
            Action::Rollback,
            format!("the candidate won {wins} dev tasks and lost {losses}: no net gain"),
        )
    };
    Decision {
        significant: action == Action::Advance && sign.p_value <= alpha,
        action,
        wins,
        losses,
        compared: pairs.len(),
        p_value: sign.p_value,
        alpha,
        retention_losses,
        candidate_rate: rate(&candidate.dev),
        champion_rate: rate(&champion.dev),
        reason,
        evidence: None,
    }
}

/// Retention passes the candidate lost, and how many the gate allows.
fn retention(gate: &DevGate, champion: &DevResult, candidate: &DevResult) -> (usize, usize) {
    let lost = candidate
        .retention
        .iter()
        .filter(|(task, &c)| !c && champion.retention.get(*task) == Some(&true))
        .count();
    let allowed = gate
        .max_retention_loss
        .max((gate.max_retention_loss_rate * champion.retention.len() as f64).floor() as usize);
    (lost, allowed)
}

/// What the evidence gate decides from the first `graded` items of the
/// round's `drawn` dev items (in draw order), at the level of `allocation`.
///
/// The e-process takes each graded item both policies have an outcome for,
/// in draw order, and stops at the first crossing; the candidate advances
/// only when it crossed and the retention check of `gate` passes. A caller
/// that grades in batches calls this after each batch and grades the next
/// while the status is [`EvidenceStatus::Open`]; the decision is the same
/// as grading every item at once and stopping at the crossing, because the
/// e-process is valid at any stopping time. Wins, losses and the sign test
/// are over the observed items, reported beside the decision.
///
/// # Errors
/// A drawn item's exposure probability is outside `[0, 1]` (the pool never
/// draws one).
pub fn decide_evidence(
    gate: &DevGate,
    evidence: &EvidenceGate,
    allocation: &Allocation,
    (drawn, graded): (&[Drawn], usize),
    champion: &DevResult,
    candidate: &DevResult,
    test: &dyn Significance,
) -> Result<Decision, EvidenceError> {
    let graded = graded.min(drawn.len());
    let mut process = EProcess::new(evidence, allocation.alpha);
    let (mut pairs, mut unpaired, mut reused, mut zeta_sum) = (Vec::new(), 0, 0, 0.0);
    if allocation.alpha > 0.0 {
        for item in &drawn[..graded] {
            let (Some(&c), Some(&b)) =
                (candidate.dev.get(&item.task), champion.dev.get(&item.task))
            else {
                unpaired += 1;
                continue;
            };
            pairs.push((c, b));
            reused += usize::from(item.reused);
            zeta_sum += item.zeta;
            let difference = f64::from(i8::from(c) - i8::from(b));
            if process.observe(difference, item.zeta)? {
                break;
            }
        }
    }
    let status = if allocation.alpha <= 0.0 {
        EvidenceStatus::Skipped
    } else if process.crossed_at().is_some() {
        EvidenceStatus::Crossed
    } else if graded < drawn.len() {
        EvidenceStatus::Open
    } else {
        EvidenceStatus::Exhausted
    };
    let wins = pairs.iter().filter(|(c, b)| *c && !*b).count();
    let losses = pairs.iter().filter(|(c, b)| !*c && *b).count();
    let (retention_losses, allowed) = retention(gate, champion, candidate);
    let e_value = process.value();
    let threshold = process.threshold();
    let (action, reason) = match status {
        EvidenceStatus::Skipped => (
            Action::Rollback,
            format!(
                "round {} has no level left to test at: the incumbent stays",
                allocation.round
            ),
        ),
        EvidenceStatus::Crossed if retention_losses > allowed => (
            Action::Rollback,
            format!(
                "the evidence crossed, but the candidate lost {retention_losses} retention passes, more than the {allowed} allowed"
            ),
        ),
        EvidenceStatus::Crossed => (
            Action::Advance,
            format!(
                "e-value {e_value:.3} reached 1/{:.4} after {} dev items (won {wins}, lost {losses})",
                allocation.alpha,
                process.observed()
            ),
        ),
        EvidenceStatus::Open | EvidenceStatus::Exhausted => (
            Action::Rollback,
            format!(
                "e-value {e_value:.3} below 1/{:.4} after {} dev items (won {wins}, lost {losses}): not enough evidence of noninferiority",
                allocation.alpha,
                process.observed()
            ),
        ),
    };
    let observed = process.observed();
    Ok(Decision {
        action,
        significant: status == EvidenceStatus::Crossed,
        wins,
        losses,
        compared: pairs.len(),
        p_value: test.sign_test(&pairs).p_value,
        alpha: allocation.alpha,
        retention_losses,
        candidate_rate: rate(&candidate.dev),
        champion_rate: rate(&champion.dev),
        reason,
        evidence: Some(EvidenceRecord {
            gate: *evidence,
            allocation: *allocation,
            status,
            e_value,
            threshold,
            drawn: drawn.len(),
            graded,
            observed,
            unpaired,
            reused,
            zeta_mean: (observed > 0).then(|| zeta_sum / observed as f64),
            spent: drawn[..graded].iter().map(|d| d.task.clone()).collect(),
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::significance::SignTest;

    /// A sign test that is significant whenever the candidate wins at least
    /// four discordant pairs without losing any.
    struct Strict;

    impl Significance for Strict {
        fn sign_test(&self, pairs: &[(bool, bool)]) -> SignTest {
            let wins = pairs.iter().filter(|(c, b)| *c && !*b).count();
            let losses = pairs.iter().filter(|(c, b)| !*c && *b).count();
            SignTest {
                discordant: wins + losses,
                candidate_wins: wins,
                p_value: if losses == 0 && wins >= 4 { 0.001 } else { 0.5 },
            }
        }
    }

    fn outcomes(spec: &[(&str, bool)]) -> Outcomes {
        spec.iter().map(|(t, p)| ((*t).to_string(), *p)).collect()
    }

    fn result(dev: &[(&str, bool)], retention: &[(&str, bool)]) -> DevResult {
        DevResult {
            dev: outcomes(dev),
            retention: outcomes(retention),
        }
    }

    #[test]
    fn a_net_gain_advances_and_a_significant_one_says_so() {
        let champion = result(
            &[
                ("a", false),
                ("b", false),
                ("c", false),
                ("d", false),
                ("e", true),
            ],
            &[],
        );
        let candidate = result(
            &[
                ("a", true),
                ("b", true),
                ("c", true),
                ("d", true),
                ("e", true),
            ],
            &[],
        );
        let decision = decide(&DevGate::default(), 1, &champion, &candidate, &Strict);
        assert_eq!(decision.action, Action::Advance);
        assert!(decision.significant);
        assert_eq!(
            (decision.wins, decision.losses, decision.compared),
            (4, 0, 5)
        );
        assert_eq!(decision.candidate_rate, Some(1.0));
        assert_eq!(decision.champion_rate, Some(0.2));
    }

    #[test]
    fn a_small_gain_advances_without_claiming_significance() {
        let champion = result(&[("a", false), ("b", true)], &[]);
        let candidate = result(&[("a", true), ("b", true)], &[]);
        let decision = decide(&DevGate::default(), 1, &champion, &candidate, &Strict);
        assert_eq!(decision.action, Action::Advance);
        assert!(!decision.significant, "one win is not significant");
    }

    #[test]
    fn no_net_gain_rolls_back() {
        let champion = result(&[("a", false), ("b", true)], &[]);
        let candidate = result(&[("a", true), ("b", false)], &[]);
        let decision = decide(&DevGate::default(), 1, &champion, &candidate, &Strict);
        assert_eq!(decision.action, Action::Rollback);
        assert!(decision.reason.contains("no net gain"));
        let unchanged = decide(&DevGate::default(), 1, &champion, &champion, &Strict);
        assert_eq!(unchanged.action, Action::Rollback);
    }

    #[test]
    fn a_retention_loss_rolls_back_a_dev_gain() {
        let champion = result(&[("a", false)], &[("old1", true), ("old2", true)]);
        let candidate = result(&[("a", true)], &[("old1", false), ("old2", true)]);
        let strict = decide(&DevGate::default(), 1, &champion, &candidate, &Strict);
        assert_eq!(strict.action, Action::Rollback);
        assert_eq!(strict.retention_losses, 1);
        let lenient = DevGate {
            max_retention_loss: 1,
            max_retention_loss_rate: 0.0,
            ..DevGate::default()
        };
        assert_eq!(
            decide(&lenient, 1, &champion, &candidate, &Strict).action,
            Action::Advance
        );
    }

    #[test]
    fn a_non_inferiority_margin_as_a_share_of_the_retention_set_is_allowed() {
        let names: Vec<String> = (0..50).map(|i| format!("r{i}")).collect();
        let retention = |lost: usize| -> Vec<(&str, bool)> {
            names
                .iter()
                .enumerate()
                .map(|(i, n)| (n.as_str(), i >= lost))
                .collect()
        };
        let champion = result(&[("a", false)], &retention(0));
        let candidate = result(&[("a", true)], &retention(3));
        let strict = decide(&DevGate::default(), 1, &champion, &candidate, &Strict);
        assert_eq!(strict.action, Action::Rollback);
        let margin = DevGate {
            max_retention_loss_rate: 0.06,
            ..DevGate::default()
        };
        assert_eq!(
            decide(&margin, 1, &champion, &candidate, &Strict).action,
            Action::Advance
        );
    }

    #[test]
    fn only_tasks_graded_for_both_are_compared_and_none_is_a_rollback() {
        let champion = result(&[("a", false)], &[]);
        let candidate = result(&[("z", true)], &[]);
        let decision = decide(&DevGate::default(), 1, &champion, &candidate, &Strict);
        assert_eq!(decision.action, Action::Rollback);
        assert_eq!(decision.compared, 0);
        assert!(decision.reason.contains("no dev task"));
        assert_eq!(
            decide(
                &DevGate::default(),
                1,
                &DevResult::default(),
                &DevResult::default(),
                &Strict
            )
            .champion_rate,
            None
        );
    }

    mod evidence {
        use super::*;
        use crate::campaign::dev_pool::Drawn;
        use crate::campaign::evidence::{Budget, EvidenceGate};

        fn drawn(n: usize, zeta: f64) -> Vec<Drawn> {
            (0..n)
                .map(|i| Drawn {
                    task: format!("t{i:02}"),
                    zeta,
                    reused: zeta > 0.0,
                })
                .collect()
        }

        /// Results where the candidate wins the first `wins` items and ties the
        /// rest (both pass).
        fn results(n: usize, wins: usize, retention_lost: bool) -> (DevResult, DevResult) {
            let dev = |candidate: bool| -> Outcomes {
                (0..n)
                    .map(|i| (format!("t{i:02}"), candidate || i >= wins))
                    .collect()
            };
            (
                DevResult {
                    dev: dev(false),
                    retention: outcomes(&[("old", true)]),
                },
                DevResult {
                    dev: dev(true),
                    retention: outcomes(&[("old", !retention_lost)]),
                },
            )
        }

        fn first_round(gate: &EvidenceGate, horizon: u32) -> Allocation {
            Budget::start(gate).allocate(gate, horizon)
        }

        #[test]
        fn strong_evidence_crosses_early_advances_and_records_what_it_spent() {
            let gate = EvidenceGate::new(0.1, 0.3, 0.01).unwrap();
            let allocation = first_round(&gate, 3);
            let (champion, candidate) = results(40, 40, false);
            let items = drawn(40, 1.0);
            let d = decide_evidence(
                &DevGate::default(),
                &gate,
                &allocation,
                (&items, 40),
                &champion,
                &candidate,
                &Strict,
            )
            .unwrap();
            assert_eq!(d.action, Action::Advance, "{}", d.reason);
            assert!(d.significant);
            let record = d.evidence.unwrap();
            assert_eq!(record.status, EvidenceStatus::Crossed);
            assert!(record.e_value >= record.threshold);
            assert!(record.observed < 40, "it stops at the crossing");
            assert_eq!(d.compared, record.observed);
            assert_eq!(record.spent.len(), 40, "every graded item is spent");
            assert_eq!(record.reused, record.observed);
            assert_eq!(record.zeta_mean, Some(1.0));
            assert!((d.alpha - allocation.alpha).abs() < 1e-15);
        }

        #[test]
        fn grading_in_batches_is_open_until_the_draw_is_exhausted() {
            let gate = EvidenceGate::new(0.0, 0.3, 0.0).unwrap();
            let allocation = first_round(&gate, 3);
            // Ties only: with eps = 0 and no correction each tie leaves the
            // evidence at one, so it never crosses.
            let (champion, candidate) = results(30, 0, false);
            let items = drawn(30, 0.0);
            let decide = |graded| {
                decide_evidence(
                    &DevGate::default(),
                    &gate,
                    &allocation,
                    (&items, graded),
                    &champion,
                    &candidate,
                    &Strict,
                )
                .unwrap()
            };
            let partial = decide(10);
            assert_eq!(
                partial.evidence.as_ref().unwrap().status,
                EvidenceStatus::Open
            );
            assert_eq!(partial.action, Action::Rollback);
            let whole = decide(30);
            let record = whole.evidence.unwrap();
            assert_eq!(record.status, EvidenceStatus::Exhausted);
            assert!((record.e_value - 1.0).abs() < 1e-12);
            assert!(whole.reason.contains("not enough evidence"));
        }

        #[test]
        fn a_retention_loss_vetoes_a_crossing_and_a_zero_level_grades_nothing() {
            let gate = EvidenceGate::new(0.1, 0.3, 0.0).unwrap();
            let allocation = first_round(&gate, 3);
            let (champion, candidate) = results(40, 40, true);
            let items = drawn(40, 0.0);
            let vetoed = decide_evidence(
                &DevGate::default(),
                &gate,
                &allocation,
                (&items, 40),
                &champion,
                &candidate,
                &Strict,
            )
            .unwrap();
            assert_eq!(vetoed.action, Action::Rollback);
            assert!(vetoed.reason.contains("retention"));
            let mut spent = Budget::start(&gate);
            for _ in 0..3 {
                let a = spent.allocate(&gate, 3);
                spent.settle(&a, false);
            }
            let beyond = spent.allocate(&gate, 3);
            let skipped = decide_evidence(
                &DevGate::default(),
                &gate,
                &beyond,
                (&items, 0),
                &champion,
                &candidate,
                &Strict,
            )
            .unwrap();
            assert_eq!(skipped.action, Action::Rollback);
            let record = skipped.evidence.unwrap();
            assert_eq!(record.status, EvidenceStatus::Skipped);
            assert_eq!((record.observed, record.spent.len()), (0, 0));
        }

        #[test]
        fn items_without_both_outcomes_are_counted_and_not_observed() {
            let gate = EvidenceGate::new(0.1, 0.3, 0.0).unwrap();
            let allocation = first_round(&gate, 3);
            let (champion, mut candidate) = results(10, 0, false);
            candidate.dev.remove("t03");
            let items = drawn(10, 0.0);
            let d = decide_evidence(
                &DevGate::default(),
                &gate,
                &allocation,
                (&items, 10),
                &champion,
                &candidate,
                &Strict,
            )
            .unwrap();
            let record = d.evidence.unwrap();
            assert_eq!((record.unpaired, record.observed), (1, 9));
        }

        #[test]
        fn a_greedy_decision_record_is_written_as_before() {
            let champion = result(&[("a", false)], &[]);
            let candidate = result(&[("a", true)], &[]);
            let decision = decide(&DevGate::default(), 1, &champion, &candidate, &Strict);
            let json = serde_json::to_string(&decision).unwrap();
            assert!(!json.contains("evidence"), "{json}");
            let read: Decision = serde_json::from_str(&json).unwrap();
            assert_eq!(read, decision);
        }
    }

    #[test]
    fn later_gates_spend_a_smaller_level() {
        let first = decide(
            &DevGate::default(),
            1,
            &DevResult::default(),
            &DevResult::default(),
            &Strict,
        );
        let tenth = decide(
            &DevGate::default(),
            10,
            &DevResult::default(),
            &DevResult::default(),
            &Strict,
        );
        assert!(tenth.alpha < first.alpha);
    }
}
