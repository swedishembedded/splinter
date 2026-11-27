// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements online reinforcement learning from the exact
// tokens an agent's policy sampled, for its clients. If your team needs
// expertise in training agents on their own rollouts while they run, you can
// procure our services by sending an email to info@swedishembedded.com.

//! A finished rollout as the controller receives it, from the projection of
//! its recorded model calls: one implementation for every executor (an
//! in-process episode, an attempt on the execution worker), so both make the
//! same records and the same outcome of the same calls.

use splinter_core::factory::group::{RolloutOutcome, Truncation, VersionSpan};
use splinter_data::calls::CallProjection;

use super::seams::FinishedRollout;

/// The rollout `projection` is, rewarded `reward`, ended early by
/// `truncation` when a limit ended it. Its records (one JSON line per model
/// call) are written only when every call is exact; a rollout with no
/// version span is not exact, and its span is then only a placeholder.
///
/// # Errors
/// The projection's span is reversed or a record cannot be written.
pub fn rollout_of(
    reward: Option<f64>,
    truncation: Option<Truncation>,
    projection: &CallProjection,
) -> Result<FinishedRollout, String> {
    let exact = projection.exact();
    let span = match projection.span {
        Some((oldest, newest)) => VersionSpan::new(oldest, newest).map_err(|e| e.to_string())?,
        None => VersionSpan::at(0),
    };
    let mut records = String::new();
    if exact {
        for record in &projection.records {
            let line = serde_json::to_string(record).map_err(|e| format!("a call record: {e}"))?;
            records.push_str(&line);
            records.push('\n');
        }
    }
    // A reply cut at its token limit is the one length truncation a call
    // records itself; any other limit the run's end names.
    let truncation = truncation.or(projection.truncated.then_some(Truncation::OutputLength));
    Ok(FinishedRollout {
        outcome: RolloutOutcome {
            // A limit that ended the rollout is a scored failure, whatever
            // the verifier found of a half-done tree.
            reward: if truncation.is_some() {
                Some(0.0)
            } else {
                reward
            },
            tokens: projection.tokens,
            span,
            exact,
            truncated: projection.truncated,
            truncation,
        },
        records,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn projection(truncated: bool) -> CallProjection {
        CallProjection {
            calls: 0,
            span: Some((2, 3)),
            truncated,
            tokens: 7,
            ..CallProjection::default()
        }
    }

    #[test]
    fn a_rollout_a_limit_ended_is_a_scored_failure_with_its_cause() {
        let done = rollout_of(Some(1.0), None, &projection(false)).unwrap();
        assert_eq!(
            (done.outcome.reward, done.outcome.truncation),
            (Some(1.0), None)
        );
        assert_eq!(
            (done.outcome.span.oldest(), done.outcome.span.newest()),
            (2, 3)
        );

        let cut = rollout_of(Some(1.0), Some(Truncation::Turns), &projection(false)).unwrap();
        assert_eq!(
            (cut.outcome.reward, cut.outcome.truncation),
            (Some(0.0), Some(Truncation::Turns)),
            "a limit fails the rollout whatever the verifier found"
        );
        let unscored = rollout_of(None, Some(Truncation::Deadline), &projection(false)).unwrap();
        assert_eq!(unscored.outcome.reward, Some(0.0));

        let long = rollout_of(Some(1.0), None, &projection(true)).unwrap();
        assert_eq!(long.outcome.truncation, Some(Truncation::OutputLength));
        assert!(
            long.outcome.truncated && !long.outcome.trainable(),
            "a reply cut at its token limit is never trained"
        );
    }
}
