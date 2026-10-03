// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements frozen, resumable evaluations of language
// models against source material for its clients. If your team needs
// expertise in measuring what a fine-tune changed, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Running the frozen exam against one model and grading every answer by code.
//!
//! Results are appended one line per question and the run resumes where it
//! stopped, so an interrupted run costs nothing already measured.

use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::ask::Answerer;
use crate::grade::{final_answer, recipient_ok, work_ok, year_ok};
use crate::tasks::{Kind, Task, PERSONA};
use crate::works::{rival_names, WORKS};

/// One graded answer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Graded {
    /// The task it answers.
    pub id: String,
    /// What the task asked.
    pub kind: Kind,
    /// `exam` or `seen`.
    pub split: String,
    /// The reference the answer was compared with.
    pub reference: String,
    /// The answer, cut for the record; `None` when the model gave none.
    pub answer: Option<String>,
    /// Whether the answer is right.
    pub correct: bool,
    /// Tokens generated.
    pub tokens: u32,
    /// Whether the budget ran out.
    pub truncated: bool,
    /// Seconds the answer took.
    pub seconds: f64,
}

/// Whether `answer` is a right answer to `task`.
#[must_use]
pub fn is_correct(task: &Task, answer: &str) -> bool {
    match task.kind {
        Kind::Recipient => recipient_ok(answer, &task.reference),
        Kind::Year => task
            .reference
            .parse()
            .is_ok_and(|year| year_ok(answer, year)),
        Kind::Work => task.reference.parse::<usize>().ok().is_some_and(|work| {
            WORKS
                .get(work)
                .is_some_and(|w| work_ok(answer, w.names, &rival_names(work)))
        }),
    }
}

/// Reads every task of a JSON-lines file.
///
/// # Errors
/// The file cannot be read or a line is not a task.
pub fn read_tasks(path: &Path) -> anyhow::Result<Vec<Task>> {
    std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(|e| anyhow::anyhow!("{}: {e}", path.display())))
        .collect()
}

/// Reads every graded answer of a results file; none when it does not exist.
///
/// # Errors
/// The file exists and a line is not a graded answer.
pub fn read_results(path: &Path) -> anyhow::Result<Vec<Graded>> {
    if !path.is_file() {
        return Ok(Vec::new());
    }
    std::fs::read_to_string(path)?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(|e| anyhow::anyhow!("{}: {e}", path.display())))
        .collect()
}

/// What one exam run needs.
pub struct Run {
    /// The frozen exam, JSON lines of tasks.
    pub exam: PathBuf,
    /// Where graded answers are appended.
    pub out: PathBuf,
    /// The base checkpoint.
    pub base: PathBuf,
    /// The adapter, for the trained arm.
    pub adapter: Option<PathBuf>,
    /// Tokens a reply may run.
    pub max_tokens: u32,
    /// Stop after this many new answers (for a measured trial).
    pub limit: Option<usize>,
}

/// Asks every question not yet answered in `run.out` and appends the graded
/// answer. Returns how many it asked.
///
/// # Errors
/// The model cannot be loaded or a generation fails.
pub fn run(run: &Run) -> anyhow::Result<usize> {
    let tasks = read_tasks(&run.exam)?;
    let done: HashSet<String> = read_results(&run.out)?.into_iter().map(|g| g.id).collect();
    let todo: Vec<&Task> = tasks.iter().filter(|t| !done.contains(&t.id)).collect();
    let todo = &todo[..run.limit.map_or(todo.len(), |n| n.min(todo.len()))];
    if todo.is_empty() {
        return Ok(0);
    }
    let answerer = Answerer::load(&run.base, run.adapter.as_deref(), 4096)?;
    let runtime = tokio::runtime::Runtime::new()?;
    let mut out = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&run.out)?;
    for (n, task) in todo.iter().enumerate() {
        let reply = runtime.block_on(answerer.ask(PERSONA, &task.prompt, run.max_tokens))?;
        let answer = final_answer(&reply.thinking, &reply.text, reply.truncated);
        let graded = Graded {
            id: task.id.clone(),
            kind: task.kind,
            split: task.split.clone(),
            reference: task.reference.clone(),
            correct: answer.as_deref().is_some_and(|a| is_correct(task, a)),
            answer: answer.map(|a| a.chars().take(400).collect()),
            tokens: reply.tokens,
            truncated: reply.truncated,
            seconds: reply.seconds,
        };
        writeln!(out, "{}", serde_json::to_string(&graded)?)?;
        out.flush()?;
        eprintln!(
            "[{}/{}] {:?} {} correct={}",
            n + 1,
            todo.len(),
            graded.kind,
            graded.split,
            graded.correct
        );
    }
    Ok(todo.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task(kind: Kind, reference: &str) -> Task {
        Task {
            id: "t".into(),
            kind,
            family: "f".into(),
            split: "exam".into(),
            prompt: "p".into(),
            reference: reference.into(),
            voice: "v".into(),
        }
    }

    #[test]
    fn each_kind_is_graded_against_its_own_reference() {
        assert!(is_correct(
            &task(Kind::Recipient, "Madison"),
            "I wrote this letter to James Madison."
        ));
        assert!(!is_correct(
            &task(Kind::Recipient, "Madison"),
            "I wrote this letter to John Jay."
        ));
        assert!(is_correct(
            &task(Kind::Year, "1789"),
            "I wrote it in 1789, from Paris."
        ));
        assert!(!is_correct(
            &task(Kind::Year, "1789"),
            "I wrote it in 1788."
        ));
        let hobbes = WORKS
            .iter()
            .position(|w| w.title == "Leviathan")
            .unwrap_or(usize::MAX)
            .to_string();
        assert!(is_correct(
            &task(Kind::Work, &hobbes),
            "That passage is from Leviathan, by Thomas Hobbes."
        ));
        assert!(!is_correct(
            &task(Kind::Work, &hobbes),
            "That passage is from Common Sense."
        ));
        assert!(
            !is_correct(&task(Kind::Work, "999"), "Leviathan"),
            "an unknown work is never right"
        );
    }
}
