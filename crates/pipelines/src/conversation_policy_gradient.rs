// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reinforcement learning for agents that work
// with people, for its clients. If your team needs expertise in training
// agents against simulated clients and students, you can procure our services
// by sending an email to info@swedishembedded.com.

//! A policy-gradient dataset from graded conversations with a counterpart.
//!
//! Each line of the input is one conversation (sven's `ConversationRecord`),
//! the case it was held for, the model that played the learner, and the reward
//! its case's grade earned. Only conversations
//! [`conversation_admission::admit`](crate::conversation_admission::admit)
//! accepts are used, so a seat that leaked its answer key, a record without a
//! seat measurement or a live transcript not scrubbed never reaches the
//! trainer. The admitted ones go through
//! [`splinter_data::conversation_pg::project`].

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use splinter_data::conversation_pg::{project, Attempt};
use splinter_data::{AdvantageKind, Projection};

use splinter_orchestrator::{Context, OrchestratorError};

use crate::conversation_admission::admit;
use crate::datasets::{store_dataset, Built};

/// One line of the input.
#[derive(Clone, Debug, Deserialize)]
pub struct Graded {
    /// The case; the conversations held for one case form a group.
    pub case: String,
    /// The model that played the learner.
    pub policy: String,
    /// This conversation's name.
    pub trial: String,
    /// The grade of the case, in [0, 1].
    pub reward: f64,
    /// The conversation record.
    pub record: Value,
}

/// What the build saw: how much signal the groups carry.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Report {
    /// Conversations read.
    pub read: usize,
    /// Conversations admitted, by pool.
    pub admitted: usize,
    /// Conversations refused, by the reason.
    pub refused: BTreeMap<String, usize>,
    /// Groups (cases).
    pub groups: usize,
    /// Groups whose attempts differ in reward: the ones with advantages.
    pub groups_with_signal: usize,
    /// Records (learner turns) in the dataset.
    pub records: usize,
    /// Mean reward over admitted conversations.
    pub mean_reward: f64,
    /// Mean over groups of the standard deviation of their rewards: how much
    /// the policy's own attempts differ at a case.
    pub mean_group_std: f64,
}

/// The dataset projection of `graded` under the role prompt `system_prompt`,
/// and the report of what it holds.
#[must_use]
pub fn build(
    system_prompt: &str,
    graded: &[Graded],
    advantage: AdvantageKind,
) -> (Projection, Report) {
    let mut report = Report {
        read: graded.len(),
        ..Report::default()
    };
    let mut attempts = Vec::new();
    for g in graded {
        match admit(&g.record) {
            Ok(_) => attempts.push(Attempt {
                case: g.case.clone(),
                policy: g.policy.clone(),
                trial: g.trial.clone(),
                reward: g.reward,
                record: g.record.clone(),
            }),
            Err(refusal) => *report.refused.entry(format!("{refusal:?}")).or_insert(0) += 1,
        }
    }
    report.admitted = attempts.len();
    let projection = project(system_prompt, &attempts, advantage);
    report.records = projection.records.len();
    report.groups = projection.groups.len();
    report.groups_with_signal = projection.groups.iter().filter(|g| g.signal).count();
    if !attempts.is_empty() {
        report.mean_reward = attempts.iter().map(|a| a.reward).sum::<f64>() / attempts.len() as f64;
    }
    if !projection.groups.is_empty() {
        let stds: Vec<f64> = projection
            .groups
            .iter()
            .map(|g| {
                let n = g.trials.len() as f64;
                let mean = g.trials.iter().map(|t| t.reward).sum::<f64>() / n;
                (g.trials
                    .iter()
                    .map(|t| (t.reward - mean).powi(2))
                    .sum::<f64>()
                    / n)
                    .sqrt()
            })
            .collect();
        report.mean_group_std = stds.iter().sum::<f64>() / stds.len() as f64;
    }
    (projection, report)
}

/// A stored conversation dataset and what it holds.
#[derive(Clone, Debug, Serialize)]
pub struct Stored {
    /// The dataset, as `train` and `dataset export` name it.
    pub dataset: Built,
    /// How much signal the groups carry.
    pub report: Report,
}

/// Reads the graded conversations of `lines` (one JSON [`Graded`] per
/// non-empty line), builds the dataset under `system_prompt` and stores it.
///
/// # Errors
///
/// A line that is not a [`Graded`], a projection with no record (no group
/// has a spread of rewards among the admitted conversations), or a refusal
/// of the store.
pub fn store(
    ctx: &Context,
    system_prompt: &str,
    lines: &str,
    advantage: AdvantageKind,
) -> Result<Stored, OrchestratorError> {
    let graded = lines
        .lines()
        .filter(|l| !l.trim().is_empty())
        .enumerate()
        .map(|(n, l)| {
            serde_json::from_str::<Graded>(l)
                .map_err(|e| OrchestratorError::Refused(format!("line {}: {e}", n + 1)))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (projection, report) = build(system_prompt, &graded, advantage);
    let dataset = store_dataset(ctx, &projection, false)?;
    Ok(Stored {
        dataset: dataset.into(),
        report,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn conversation(leaked: bool) -> Value {
        let violations = if leaked {
            json!([[0, {"kind": "answer_key_leak", "detail": "k"}]])
        } else {
            json!([])
        };
        json!({"provenance": {"kind": "simulated"},
               "checks": {"replies": 1, "clean_replies": 1, "violations": violations},
               "exchanges": [{"speaker": "seat", "text": "Hi"}, {"speaker": "learner", "text": "Hello"}]})
    }

    fn graded(case: &str, trial: &str, reward: f64, leaked: bool) -> Graded {
        Graded {
            case: case.into(),
            policy: "m".into(),
            trial: trial.into(),
            reward,
            record: conversation(leaked),
        }
    }

    #[test]
    fn only_admitted_conversations_train_and_the_report_counts_the_signal() {
        let input = [
            graded("a", "1", 1.0, false),
            graded("a", "2", 0.0, false),
            graded("a", "3", 1.0, true),
            graded("b", "1", 0.5, false),
            graded("b", "2", 0.5, false),
        ];
        let (projection, report) = build("role", &input, AdvantageKind::Standardised);
        assert_eq!(report.read, 5);
        assert_eq!(report.admitted, 4);
        assert_eq!(report.refused["AnswerKeyLeak"], 1);
        assert_eq!((report.groups, report.groups_with_signal), (2, 1));
        assert_eq!(report.records, 2);
        assert_eq!(projection.records.len(), 2);
        assert!((report.mean_group_std - 0.25).abs() < 1e-9, "{report:?}");
    }
}
