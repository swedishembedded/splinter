// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements contamination-free evaluation of fine-tuned
// models, for its clients. If your team needs expertise in model
// evaluation, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Held-out tasks the candidate was trained on: left out of every suite
//! the gate grades.
//!
//! A task measures generalisation only if its question was not among the
//! texts the candidate was trained on. The split that holds records out is
//! positional, so a task solved several times, a question generated twice
//! in other words, or an earlier release's held-out question asked again in
//! new data can sit on both sides of it. Before anything is graded, every
//! suite's instructions are checked against the prompts of every record
//! the candidate trained on - its datasets' trained-on records and the
//! records it replayed - by the task generator's near-duplicate rule
//! ([`Seen::leaks`]: the same question normalised, an overlap of word
//! shingles at the generator's threshold, or the question's words inside a
//! longer prompt). A task that matches is left out and counted under
//! [`LEAKED`]; the gate decides on the rest.

use splinter_knowledge::tasks::dedup::Seen;
use splinter_knowledge::tasks::{DEFAULT_MAX_OVERLAP, DEFAULT_SHINGLE_WORDS};

use crate::context::Context;
use crate::error::{io, CampaignError};
use crate::release::probe::{split_records, Suite};
use crate::train::Candidate;

/// Why a task was left out of a suite: the candidate was trained on it.
pub const LEAKED: &str = "leaked";

/// The prompts of every record `candidate` was trained on.
pub(crate) fn trained_prompts(ctx: &Context, candidate: &Candidate) -> Result<Seen, CampaignError> {
    let (mut trained_on, _held_out) = split_records(ctx, &candidate.datasets)?;
    if candidate
        .replay
        .as_ref()
        .is_some_and(|r| r.digest.is_some())
    {
        let digest = candidate.replay.as_ref().and_then(|r| r.digest.as_ref());
        let path = ctx.artifacts().path(digest.ok_or_else(|| {
            CampaignError::Refused("a candidate that replayed records names no replay file".into())
        })?)?;
        let text = std::fs::read_to_string(&path).map_err(io(&path))?;
        trained_on.extend(
            text.lines()
                .filter(|l| !l.trim().is_empty())
                .map(str::to_string),
        );
    }
    let mut seen = Seen::new(DEFAULT_SHINGLE_WORDS, DEFAULT_MAX_OVERLAP);
    for (index, line) in trained_on.iter().enumerate() {
        let record: serde_json::Value =
            serde_json::from_str(line).map_err(|source| CampaignError::Json {
                what: format!(
                    "trained-on record {index} of candidate {}",
                    candidate.candidate
                ),
                source,
            })?;
        for prompt in user_turns(&record) {
            seen.admit(prompt);
        }
    }
    Ok(seen)
}

/// The user turns of a chat record (`messages`) or a preference pair's
/// prompt (`prompt`).
fn user_turns(record: &serde_json::Value) -> impl Iterator<Item = &str> {
    ["messages", "prompt"]
        .into_iter()
        .filter_map(|field| record.get(field)?.as_array())
        .flatten()
        .filter(|message| message.get("role").and_then(|r| r.as_str()) == Some("user"))
        .filter_map(|message| message.get("content")?.as_str())
}

/// `suite` without the tasks whose instruction leaks from `trained`,
/// counted under [`LEAKED`].
pub(crate) fn exclude_leaked(suite: &mut Suite, trained: &Seen) {
    let before = suite.tasks.len();
    suite
        .tasks
        .retain(|task| trained.leaks(&task.instruction).is_none());
    let leaked = before - suite.tasks.len();
    if leaked > 0 {
        *suite.excluded.entry(LEAKED.into()).or_default() += leaked;
    }
}
