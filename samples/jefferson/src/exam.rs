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

use std::path::{Path, PathBuf};

use splinter_sdk::measure::verifiers::answer::year_ok;
use splinter_sdk::model::exam::{Model, Question};

use crate::grade::{recipient_ok, work_ok};
use crate::tasks::{Kind, Task, PERSONA};
use crate::works::{rival_names, WORKS};

pub use splinter_sdk::model::exam::{read_results, Graded};

impl Question for Task {
    fn id(&self) -> &str {
        &self.id
    }
    fn kind(&self) -> &str {
        self.kind.name()
    }
    fn split(&self) -> &str {
        &self.split
    }
    fn reference(&self) -> &str {
        &self.reference
    }
    fn prompt(&self) -> &str {
        &self.prompt
    }
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
    /// The system message to ask under, in place of the sample's persona: for
    /// an adapter Splinter trained under its own system prompt.
    pub system: Option<String>,
}

/// Asks every question not yet answered in `run.out` and appends the graded
/// answer. Returns how many it asked.
///
/// # Errors
/// The model cannot be loaded or a generation fails.
pub fn run(run: &Run) -> anyhow::Result<usize> {
    let tasks = read_tasks(&run.exam)?;
    let model = Model {
        base: &run.base,
        adapter: run.adapter.as_deref(),
        system: run.system.as_deref().unwrap_or(PERSONA),
        max_tokens: run.max_tokens,
        label: "jefferson",
    };
    splinter_sdk::model::exam::run(&tasks, &run.out, run.limit, &model, &|task, answer| {
        is_correct(task, answer).into()
    })
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
