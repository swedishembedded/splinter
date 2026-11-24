// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the loop from a failed agent attempt to a
// verified corrected one, for its clients. If your team needs expertise in
// turning agent failures into training data you can procure our services by
// sending an email to info@swedishembedded.com.

//! Fork corrections: an attempt that continues a failed one.
//!
//! An attempt stage given `continue_from` continues, for each of its tasks,
//! the last failed attempt of that stage, cut after the step its brief
//! names (`keep_through`, a step id, which a teacher `agent` stage over
//! `attempts.failures` locates: the last step before the attempt went
//! wrong). The worker runs the kept steps' tool calls again in the fresh
//! sandbox, so the workspace is as they left it, and the model goes on from
//! there, after the brief's `guidance` as a user turn.
//!
//! What the continuation then solves is learned as one conversation, the
//! kept steps and the continuation, without the turn the continuation
//! added ([`uncontinue`]); a continuation that repeats its guidance is not
//! learned, as a hinted trial that quotes its hint is not.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::Value;
use splinter_core::chat::WireMessage;
use splinter_core::digest::Digest;
use splinter_core::factory::execution::{Continuation, FileRef};
use splinter_core::factory::task::TaskRef;
use splinter_data::RecordBody;

use super::rollout::trials_of;
use super::{io, stage_error, RunError};
use crate::hint::brief_item;

/// Where a stage keeps the kept conversations it continues, below its
/// pinned files.
const FORKS: &str = "forks";

/// A brief's fork point: the last step kept, and the guidance turn.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForkPoint {
    /// The step id of the last step kept.
    pub keep_through: u64,
    /// The user turn the continuation adds: never empty, so the record can
    /// tell it from every other turn.
    pub guidance: String,
}

/// The fork point the `briefs` collection at `briefs` gives the task `id`:
/// its item's `output.json`, `{ "keep_through": N, "guidance": "..." }`.
///
/// # Errors
/// No brief for the task, or one without a step to keep through or without
/// guidance.
pub fn fork_point(briefs: &Path, id: &str) -> Result<ForkPoint, String> {
    let path = briefs.join(brief_item(id)).join("output.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("no fork point for task {id} ({}: {e})", path.display()))?;
    let value: Value =
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let keep_through = value
        .get("keep_through")
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{}: no `keep_through` step id", path.display()))?;
    let guidance = value
        .get("guidance")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|g| !g.is_empty())
        .ok_or_else(|| format!("{}: no `guidance` for the continuation", path.display()))?
        .to_owned();
    Ok(ForkPoint {
        keep_through,
        guidance,
    })
}

/// The continuation of every task of `tasks`: the last failed attempt of
/// the stage viewed at `from`, cut at its fork point in `briefs`, pinned
/// below `files` (the stage's pinned files).
///
/// # Errors
/// A task has no failed attempt with a trajectory, no fork point, or a
/// fork point the attempt cannot be cut at.
pub(crate) fn continuations(
    key: &str,
    from: &Path,
    briefs: &Path,
    tasks: &[&TaskRef],
    files: &Path,
) -> Result<BTreeMap<String, Continuation>, RunError> {
    let mut last_failed = BTreeMap::new();
    for trial in trials_of(from)? {
        if !trial.verified_success() && trial.trace.is_some() {
            last_failed.insert(trial.task.clone(), trial);
        }
    }
    let mut out = BTreeMap::new();
    for task in tasks {
        let trial = last_failed.get(&task.id).ok_or_else(|| {
            stage_error(
                key,
                format!("task {} has no failed attempt to continue", task.id),
            )
        })?;
        let trace = trial
            .trace
            .as_ref()
            .map(|t| from.join(&t.path))
            .unwrap_or_default();
        let trajectory: atif::Trajectory =
            serde_json::from_slice(&std::fs::read(&trace).map_err(io(&trace))?)
                .map_err(|e| stage_error(key, format!("{}: {e}", trace.display())))?;
        let point = fork_point(briefs, &task.id).map_err(|e| stage_error(key, e))?;
        // A fork point a model turn cannot follow (a reply that called no tool) falls back to the latest
        // earlier step it can: the continuation then sees a little less of the attempt, never none of it.
        let kept = (0..=point.keep_through)
            .rev()
            .find_map(|step| splinter_agent::fork_at::kept_prefix(&trajectory, step).ok())
            .ok_or_else(|| {
                stage_error(
                    key,
                    format!(
                        "task {}: no step of the attempt at or before step {} can be continued",
                        task.id, point.keep_through
                    ),
                )
            })?;
        let relative = format!("{FORKS}/{}.json", brief_item(&task.id));
        let path = files.join(&relative);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(io(parent))?;
        }
        let bytes = serde_json::to_vec_pretty(&kept).map_err(|e| stage_error(key, e))?;
        std::fs::write(&path, &bytes).map_err(io(&path))?;
        let digest = Digest::sha256_of_reader(bytes.as_slice()).map_err(io(&path))?;
        out.insert(
            task.id.clone(),
            Continuation {
                trace: FileRef {
                    path: relative,
                    digest,
                },
                prompt: point.guidance,
            },
        );
    }
    Ok(out)
}

/// `body`, a continued conversation, without the user turn the
/// continuation added: the first user turn after the first that begins
/// with `guidance` (the agent may add to a prompt, as sven adds its time
/// line).
///
/// # Errors
/// `(reason, detail)`: `continuation_not_found` when no such turn is
/// there; `guidance_repeated` when a reply repeats the guidance.
pub fn uncontinue(body: &mut RecordBody, guidance: &str) -> Result<(), (String, String)> {
    let RecordBody::Chat { messages, .. } = body else {
        return Err((
            "continuation_not_found".into(),
            "the record is not a conversation".into(),
        ));
    };
    let guidance = guidance.trim();
    let added = messages
        .iter()
        .enumerate()
        .skip_while(|(_, m)| m.role != "user")
        .skip(1)
        .find(|(_, m)| m.role == "user" && m.content.trim_start().starts_with(guidance))
        .map(|(at, _)| at)
        .ok_or_else(|| {
            (
                "continuation_not_found".to_owned(),
                "no user turn after the kept steps holds the continuation's guidance".to_owned(),
            )
        })?;
    messages.remove(added);
    if guidance.chars().count() >= 16 {
        let repeats = |m: &WireMessage| {
            std::iter::once(m.content.as_str())
                .chain(m.reasoning_content.as_deref())
                .any(|words| words.contains(guidance))
        };
        if messages.iter().any(|m| m.role == "assistant" && repeats(m)) {
            return Err((
                "guidance_repeated".into(),
                "a reply repeats the continuation's guidance".into(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(role: &str, content: &str) -> WireMessage {
        serde_json::from_value(
            serde_json::json!({ "role": role, "content": content, "train": role == "assistant" }),
        )
        .unwrap()
    }

    fn chat(messages: Vec<WireMessage>) -> RecordBody {
        serde_json::from_value(serde_json::json!({ "shape": "chat", "messages": messages }))
            .unwrap()
    }

    fn messages(body: &RecordBody) -> Vec<(String, String)> {
        let RecordBody::Chat { messages, .. } = body else {
            panic!("a chat")
        };
        messages
            .iter()
            .map(|m| (m.role.clone(), m.content.clone()))
            .collect()
    }

    #[test]
    fn the_turn_a_continuation_added_is_not_learned() {
        let guidance = "Check the file before you finish the job";
        let mut body = chat(vec![
            message("user", "Do the task."),
            message("assistant", "I start."),
            message("user", guidance),
            message("assistant", "Done."),
        ]);
        uncontinue(&mut body, guidance).unwrap();
        assert_eq!(
            messages(&body),
            [
                ("user".into(), "Do the task.".into()),
                ("assistant".into(), "I start.".into()),
                ("assistant".into(), "Done.".into())
            ]
        );
        let mut timed = chat(vec![
            message("user", "Do the task."),
            message(
                "user",
                &format!("{guidance}\n\nTime: 0 of 8 min used, 8 min left."),
            ),
            message("assistant", "Done."),
        ]);
        uncontinue(&mut timed, guidance).unwrap();
        assert_eq!(
            messages(&timed).len(),
            2,
            "the agent's addition goes with the turn"
        );
    }

    #[test]
    fn a_continuation_without_its_turn_or_that_repeats_its_guidance_is_refused() {
        let guidance = "Check the file before you finish the job";
        let mut missing = chat(vec![
            message("user", "Do the task."),
            message("assistant", "Done."),
        ]);
        assert_eq!(
            uncontinue(&mut missing, guidance).unwrap_err().0,
            "continuation_not_found"
        );
        let mut parrot = chat(vec![
            message("user", "Do the task."),
            message("user", guidance),
            message("assistant", &format!("As told: {guidance}.")),
        ]);
        assert_eq!(
            uncontinue(&mut parrot, guidance).unwrap_err().0,
            "guidance_repeated"
        );
    }
}
