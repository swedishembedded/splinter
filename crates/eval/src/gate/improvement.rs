// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements release gates that adopt a model update
// only on paired, held-out evidence, for its clients. If your team needs
// expertise in evaluation statistics for model releases, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The improvement check: the paired sign test plus the count over every
//! assigned task, so a candidate cannot remove a loss by having an attempt
//! withheld.

use serde::{Deserialize, Serialize};

use super::{Check, Improvement, SuiteSummary};
use crate::paired::{by_cluster, compare, PairedOutcome};
use crate::significance::Significance;

/// What each arm solved over every task the suite assigned, beside the paired
/// comparison: a task an arm was not graded on (an attempt set aside as
/// exposed, or as an environment failure) is still a task it did not solve.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Verified {
    /// Tasks the candidate solved.
    pub candidate: usize,
    /// Tasks the champion solved.
    pub baseline: usize,
    /// Tasks the suite assigned to both.
    pub tasks: usize,
}

/// The improvement check over `outcomes` on `suite`.
#[must_use]
pub fn improvement(
    suite: SuiteSummary,
    variants: Option<SuiteSummary>,
    outcomes: &[PairedOutcome],
    alpha: f64,
    significance: &dyn Significance,
) -> Check<Improvement> {
    let comparison = compare(outcomes);
    if comparison.paired == 0 {
        return Check::unmeasured(format!(
            "no held-out task of {} was graded for both models ({} task(s), {} excluded before \
             grading)",
            suite.name,
            suite.tasks,
            suite.excluded.values().sum::<usize>()
        ));
    }
    let pairs = by_cluster(outcomes);
    let test = significance.sign_test(&pairs);
    let verified = Verified {
        candidate: outcomes
            .iter()
            .filter(|outcome| outcome.candidate == Some(true))
            .count(),
        baseline: outcomes
            .iter()
            .filter(|outcome| outcome.baseline == Some(true))
            .count(),
        tasks: suite.tasks,
    };
    let failure = if verified.candidate < verified.baseline {
        Some(format!(
            "no improvement: the candidate solved fewer tasks than the champion ({} against {} \
             of {}): a task the candidate was not graded on counts as a task it did not solve, \
             so an attempt set aside as exposed or as an environment failure cannot remove a \
             loss from the paired test",
            verified.candidate, verified.baseline, verified.tasks
        ))
    } else {
        (test.p_value > alpha).then(|| {
            format!(
                "no significant improvement: the candidate won {} of {} discordant task(s) \
                 (tasks about one family of sources count once), p = {:.4} > alpha {alpha}",
                test.candidate_wins, test.discordant, test.p_value
            )
        })
    };
    Check::decided(
        Improvement {
            suite,
            verified,
            variants,
            comparison,
            sign_test: test,
            alpha,
            generalisation: None,
            recall: None,
            task_level: None,
            agentic: None,
            power: None,
            hidden_strength: None,
        },
        failure,
    )
}
