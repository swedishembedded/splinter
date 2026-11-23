// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the loop from a failed agent attempt to a
// verified corrected one, for its clients. If your team needs expertise in
// turning agent failures into training data you can procure our services by
// sending an email to info@swedishembedded.com.

//! Hinted passes: rationalisation of tasks a model could not solve unaided.
//!
//! A pass with a hint runs each of its tasks from a copy whose instruction
//! is the task's own followed by what the hint shows, in a delimited
//! section: the task's reference solution (`hint: solution`, from
//! `solution/solve.sh`), or a brief written for the task by an earlier
//! stage (`hint: brief`: knowledge passages, a plan of subgoals, a
//! teacher's notes on the failed attempt). The task's environment, tests and
//! verifier are the task's own, so a trial passes only by doing the work in
//! the task's sandbox. The task repository is never written: the copies are
//! the run's.
//!
//! A brief is read from the stage's `briefs` collection: the item named by
//! the task's id ([`brief_item`]) holds `output.json`, an object whose
//! `brief` is the text (what an `agent` stage asked for `{ brief }` writes).
//!
//! A verified success of a hinted pass is learned as the model could have
//! produced it unaided: every user turn that holds the hinted instruction is
//! given the task's own instruction instead ([`unhint`]). A trial whose own
//! words (a reply or its reasoning) repeat a line of the solution, or its
//! delimiters, learned to quote the hint rather than to do the task, and is
//! excluded (`hint_repeated`); a record in which the hint is still seen
//! after that is excluded too (`hint_not_removed`), as is one whose
//! conversation never held the hinted instruction (`hint_not_found`). The
//! actions themselves (tool calls) may of course do what the solution does.

use std::path::Path;

use serde_json::Value;

use splinter_core::chat::WireMessage;
use splinter_core::factory::task::TaskRef;
use splinter_data::RecordBody;

use crate::ops::Hint;
use crate::tree::link_tree;

/// The reference solution of a Harbor task, relative to the task.
pub const SOLUTION: &str = "solution/solve.sh";
/// The instruction of a Harbor task, relative to the task.
pub const INSTRUCTION: &str = "instruction.md";
/// What a brief's item holds.
const BRIEF_FILE: &str = "output.json";
/// A solution line shorter than this is too common to be a quotation.
const QUOTED_LINE_CHARS: usize = 16;

/// The delimiters a hint of `kind` is shown between.
fn delimiters(kind: Hint) -> (&'static str, &'static str) {
    match kind {
        Hint::Solution => ("<reference-solution>", "</reference-solution>"),
        Hint::Brief => ("<briefing>", "</briefing>"),
    }
}

/// The task's instruction with `shown` appended as its hint of `kind`.
#[must_use]
pub fn hinted_instruction(instruction: &str, kind: Hint, shown: &str) -> String {
    let (open, close) = delimiters(kind);
    let preface = match kind {
        Hint::Solution => {
            "A reference solution of this task follows. Reach the same result by your own work in the \
             task's environment; do not quote it."
        }
        Hint::Brief => {
            "Notes on this task follow. Use them to do the task by your own work in the task's \
             environment; do not quote them."
        }
    };
    format!(
        "{instruction}\n\n## Hint\n\n{preface}\n\n{open}\n{}\n{close}\n",
        shown.trim_end()
    )
}

/// The item of a `briefs` collection that holds the brief of the task `id`:
/// the id with each `/` written `--` (an item id is one path component).
#[must_use]
pub fn brief_item(id: &str) -> String {
    id.replace('/', "--")
}

/// The brief of the task `id` in the `briefs` collection at `briefs`.
///
/// # Errors
/// The collection holds no brief for the task, or not a text one.
pub fn brief_of(briefs: &Path, id: &str) -> Result<String, String> {
    let path = briefs.join(brief_item(id)).join(BRIEF_FILE);
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("no brief for task {id} ({}: {e})", path.display()))?;
    let value: Value =
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    match value.get("brief").and_then(Value::as_str) {
        Some(brief) if !brief.trim().is_empty() => Ok(brief.to_owned()),
        _ => Err(format!("{}: no `brief` text", path.display())),
    }
}

/// A task's instruction and what the hint `hint` shows beside it: the
/// task's own solution, or its brief in `briefs`.
///
/// # Errors
/// The task has no instruction, no reference solution, or no brief.
pub fn material(
    task_dir: &Path,
    id: &str,
    hint: Hint,
    briefs: Option<&Path>,
) -> Result<(String, String), String> {
    let read = |relative: &str| {
        std::fs::read_to_string(task_dir.join(relative))
            .map_err(|e| format!("{}: {e}", task_dir.join(relative).display()))
    };
    let instruction = read(INSTRUCTION)?;
    let solution = match hint {
        Hint::Solution => read(SOLUTION).map_err(|e| {
            format!("a hinted pass shows each task's reference solution, and there is none: {e}")
        })?,
        Hint::Brief => brief_of(
            briefs.ok_or("a briefed pass is given its briefs, and it has none")?,
            id,
        )?,
    };
    Ok((instruction, solution))
}

/// Write the hinted copy of every task of `tasks` (ids relative to `root`)
/// below `out`, replacing an earlier one.
///
/// # Errors
/// A task has no instruction or solution, or a copy cannot be written.
pub fn stage(
    root: &Path,
    tasks: &[&TaskRef],
    hint: Hint,
    briefs: Option<&Path>,
    out: &Path,
) -> Result<(), String> {
    for task in tasks {
        let (src, dst) = (root.join(&task.id), out.join(&task.id));
        let (instruction, solution) = material(&src, &task.id, hint, briefs)?;
        if dst.exists() {
            std::fs::remove_dir_all(&dst).map_err(|e| format!("{}: {e}", dst.display()))?;
        }
        link_tree(&src, &dst).map_err(|e| e.to_string())?;
        // The copy's instruction is its own file, never a link to the task's.
        let path = dst.join(INSTRUCTION);
        std::fs::remove_file(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        std::fs::write(&path, hinted_instruction(&instruction, hint, &solution))
            .map_err(|e| format!("{}: {e}", path.display()))?;
    }
    Ok(())
}

/// The lines of `solution` a reply would quote: long enough not to be
/// common words, without a comment's marker.
fn quotable(solution: &str) -> Vec<&str> {
    solution
        .lines()
        .filter(|l| !l.starts_with("#!"))
        .map(|l| l.trim().trim_start_matches('#').trim())
        .filter(|l| l.chars().count() >= QUOTED_LINE_CHARS)
        .collect()
}

fn own_words(message: &WireMessage) -> impl Iterator<Item = &str> {
    std::iter::once(message.content.as_str()).chain(message.reasoning_content.as_deref())
}

/// The record of a hinted trial with the task's own instruction in place of
/// the hinted one, or why it cannot be learned: `(reason, detail)`.
///
/// # Errors
/// `hint_not_found`, `hint_repeated` or `hint_not_removed` (module docs).
pub fn unhint(
    body: &mut RecordBody,
    instruction: &str,
    hint: Hint,
    solution: &str,
) -> Result<(), (String, String)> {
    let (open, close) = delimiters(hint);
    let RecordBody::Chat { messages, .. } = body else {
        return Err((
            "hint_not_found".into(),
            "the record is not a conversation".into(),
        ));
    };
    // Trimmed: an agent may send its instruction without the file's last
    // newline.
    let hinted = hinted_instruction(instruction, hint, solution);
    let (hinted, instruction) = (hinted.trim(), instruction.trim());
    let mut replaced = 0;
    for message in messages.iter_mut().filter(|m| m.role == "user") {
        if message.content.contains(hinted) {
            message.content = message.content.replace(hinted, instruction);
            replaced += 1;
        }
    }
    if replaced == 0 {
        return Err((
            "hint_not_found".into(),
            "no user turn holds the hinted instruction the trial was given".into(),
        ));
    }
    let lines = quotable(solution);
    for message in messages.iter().filter(|m| m.role == "assistant") {
        for words in own_words(message) {
            let quoted = [open, close]
                .into_iter()
                .chain(lines.iter().copied())
                .find(|line| words.contains(line));
            if let Some(line) = quoted {
                return Err((
                    "hint_repeated".into(),
                    format!("a reply repeats the hint: {line:?}"),
                ));
            }
        }
    }
    if let Some(message) = messages.iter().find(|m| m.content.contains(open)) {
        return Err((
            "hint_not_removed".into(),
            format!("a {} turn still shows the hint", message.role),
        ));
    }
    Ok(())
}
