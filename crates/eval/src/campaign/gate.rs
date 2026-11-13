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

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

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
    let retention_losses = candidate
        .retention
        .iter()
        .filter(|(task, &c)| !c && champion.retention.get(*task) == Some(&true))
        .count();
    let allowed = gate
        .max_retention_loss
        .max((gate.max_retention_loss_rate * champion.retention.len() as f64).floor() as usize);
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
    }
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
