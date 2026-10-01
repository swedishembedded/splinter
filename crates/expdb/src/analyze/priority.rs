// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! What to spend the next unit of compute on: a score over measured
//! properties.

/// Properties of a candidate task, skill or state, each measured or not.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Priority {
    /// How unsure the policy is about the outcome.
    pub uncertainty: Option<f64>,
    /// How unlike earlier experience it is.
    pub novelty: Option<f64>,
    /// How much the policy has recently improved on similar work.
    pub learning_progress: Option<f64>,
    /// How strong the verification of its outcome is.
    pub verification: Option<f64>,
    /// How likely learning it is to carry over to other domains.
    pub transfer: Option<f64>,
    /// What it costs to run, in whatever unit the caller budgets.
    pub cost: Option<f64>,
}

impl Priority {
    /// Uncertainty x novelty x learning progress x verification x transfer,
    /// per unit cost. `None` if any property is unmeasured or the cost is not
    /// positive: an unmeasured property is not a zero.
    pub fn score(&self) -> Option<f64> {
        let cost = self.cost.filter(|c| *c > 0.0)?;
        Some(
            self.uncertainty?
                * self.novelty?
                * self.learning_progress?
                * self.verification?
                * self.transfer?
                / cost,
        )
    }
}

/// Candidates ordered by score, best first; unscored ones last, in the order
/// given.
pub fn rank<T>(candidates: Vec<(T, Priority)>) -> Vec<(T, Option<f64>)> {
    let mut scored: Vec<(T, Option<f64>)> = candidates
        .into_iter()
        .map(|(t, p)| (t, p.score()))
        .collect();
    scored.sort_by(|a, b| match (a.1, b.1) {
        (Some(x), Some(y)) => y.total_cmp(&x),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    });
    scored
}
