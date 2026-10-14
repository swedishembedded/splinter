// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements resumable measurement runs whose judged
// verdicts are kept as they are made, for its clients. If your team needs
// expertise in running long model-graded evaluations that survive
// interruption, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The judge's verdicts, kept in a file as they are made.
//!
//! A verdict is a function of the judge, the question, the statement and the
//! answer, so it is asked once: an interrupted screening, a larger one that
//! includes the first, and the same answer judged as a day-0 answer and again
//! as a control all find it here.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::Context as _;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct Line {
    key: String,
    verdict: Option<bool>,
}

/// The kept verdicts of one judge.
pub struct Verdicts {
    path: PathBuf,
    judge: String,
    kept: RefCell<HashMap<String, Option<bool>>>,
}

impl Verdicts {
    /// The verdicts of `judge` in the file at `path`, empty when it does not exist.
    ///
    /// # Errors
    /// The file exists and a line of it is not a verdict.
    pub fn open(path: &Path, judge: &str) -> anyhow::Result<Self> {
        let mut kept = HashMap::new();
        if path.exists() {
            for (n, line) in std::fs::read_to_string(path)?.lines().enumerate() {
                let line: Line = serde_json::from_str(line)
                    .with_context(|| format!("{} line {}", path.display(), n + 1))?;
                kept.insert(line.key, line.verdict);
            }
        }
        Ok(Self {
            path: path.to_path_buf(),
            judge: judge.to_string(),
            kept: RefCell::new(kept),
        })
    }

    fn key(&self, question: &str, statement: &str, answer: &str) -> String {
        blake3::hash(format!("{}\0{question}\0{statement}\0{answer}", self.judge).as_bytes())
            .to_hex()
            .to_string()
    }

    /// The verdict kept for these, or the one `ask` gives (and keeps).
    ///
    /// # Errors
    /// `ask` fails or the file cannot be written.
    pub fn get_or_ask(
        &self,
        question: &str,
        statement: &str,
        answer: &str,
        ask: impl FnOnce() -> anyhow::Result<Option<bool>>,
    ) -> anyhow::Result<Option<bool>> {
        let key = self.key(question, statement, answer);
        if let Some(verdict) = self.kept.borrow().get(&key) {
            return Ok(*verdict);
        }
        let verdict = ask()?;
        splinter_sdk::store::jsonl::Log::open(&self.path)?.append(&Line {
            key: key.clone(),
            verdict,
        })?;
        self.kept.borrow_mut().insert(key, verdict);
        Ok(verdict)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_verdict_is_asked_once_and_found_again_by_a_later_run_of_the_same_judge_only() {
        let dir = tempfile::tempdir().expect("dir");
        let file = dir.path().join("verdicts.jsonl");
        let asked = std::cell::Cell::new(0);
        let ask = |verdict: Option<bool>| {
            asked.set(asked.get() + 1);
            Ok(verdict)
        };
        let first = Verdicts::open(&file, "judge-a").expect("open");
        assert_eq!(
            first
                .get_or_ask("q", "s", "a", || ask(Some(true)))
                .expect("v"),
            Some(true)
        );
        assert_eq!(
            first
                .get_or_ask("q", "s", "a", || ask(Some(false)))
                .expect("v"),
            Some(true)
        );
        assert_eq!(
            first.get_or_ask("q", "s", "b", || ask(None)).expect("v"),
            None
        );
        assert_eq!(asked.get(), 2);
        let again = Verdicts::open(&file, "judge-a").expect("open");
        assert_eq!(
            again
                .get_or_ask("q", "s", "b", || ask(Some(true)))
                .expect("v"),
            None,
            "an abstention is kept too"
        );
        assert_eq!(asked.get(), 2);
        let other = Verdicts::open(&file, "judge-b").expect("open");
        assert_eq!(
            other
                .get_or_ask("q", "s", "a", || ask(Some(false)))
                .expect("v"),
            Some(false)
        );
        assert_eq!(asked.get(), 3, "another judge's verdict is its own");
    }
}
