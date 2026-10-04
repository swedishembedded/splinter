// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements frozen, resumable evaluations of language
// models against source material for its clients. If your team needs
// expertise in measuring what a fine-tune changed, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Running a frozen exam against one model and grading every answer by code.
//!
//! Results are appended one line per question and the run resumes where it
//! stopped, so an interrupted run costs nothing already measured.

use std::collections::HashSet;
use std::io::Write;
use std::path::Path;

use serde::{Deserialize, Serialize};

use splinter_eval::verifiers::answer::final_answer;

use crate::answer::{Answerer, Reply};

/// One question of an exam: what is asked, what it is compared with, and which
/// kind and side of the split it belongs to.
pub trait Question {
    /// A stable identifier: the same question has the same one in every run.
    fn id(&self) -> &str;
    /// What the question asks, as a short name: `year`, `attribution`.
    fn kind(&self) -> &str;
    /// `exam` (never trained on) or `seen` (trained on).
    fn split(&self) -> &str;
    /// What an answer is compared with.
    fn reference(&self) -> &str;
    /// What the model is asked.
    fn prompt(&self) -> &str;
}

/// One graded answer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Graded {
    /// The question it answers.
    pub id: String,
    /// What the question asked.
    pub kind: String,
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

/// The most of an answer a record keeps.
const RECORDED_ANSWER_CHARS: usize = 400;

/// The graded answers already in `path`; none when there is no file yet.
///
/// # Errors
/// The file cannot be read or a line is not a graded answer.
pub fn read_results(path: &Path) -> anyhow::Result<Vec<Graded>> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Ok(Vec::new());
    };
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(|e| anyhow::anyhow!("{}: {e}", path.display())))
        .collect()
}

/// The questions not yet answered in `out`, at most `limit` of them.
///
/// # Errors
/// `out` exists and cannot be read.
pub fn pending<'a, Q: Question>(
    questions: &'a [Q],
    out: &Path,
    limit: Option<usize>,
) -> anyhow::Result<Vec<&'a Q>> {
    let done: HashSet<String> = read_results(out)?.into_iter().map(|g| g.id).collect();
    let todo: Vec<&Q> = questions
        .iter()
        .filter(|q| !done.contains(q.id()))
        .collect();
    let take = limit.map_or(todo.len(), |n| n.min(todo.len()));
    Ok(todo.into_iter().take(take).collect())
}

/// Ask every pending question and append its graded answer to `out`, one line
/// at a time, flushed, so a stop loses at most the question in flight.
///
/// # Errors
/// `out` cannot be written or `ask` fails.
pub fn run_with<Q: Question>(
    questions: &[Q],
    out: &Path,
    limit: Option<usize>,
    ask: &mut dyn FnMut(&Q) -> anyhow::Result<Reply>,
    is_correct: &dyn Fn(&Q, &str) -> bool,
) -> anyhow::Result<usize> {
    let todo = pending(questions, out, limit)?;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out)?;
    for (n, question) in todo.iter().enumerate() {
        let reply = ask(question)?;
        let answer = final_answer(&reply.thinking, &reply.text, reply.truncated);
        let graded = Graded {
            id: question.id().to_string(),
            kind: question.kind().to_string(),
            split: question.split().to_string(),
            reference: question.reference().to_string(),
            correct: answer.as_deref().is_some_and(|a| is_correct(question, a)),
            answer: answer.map(|a| a.chars().take(RECORDED_ANSWER_CHARS).collect()),
            tokens: reply.tokens,
            truncated: reply.truncated,
            seconds: reply.seconds,
        };
        writeln!(file, "{}", serde_json::to_string(&graded)?)?;
        file.flush()?;
        eprintln!(
            "[{}/{}] {} {} correct={} thinking={} chars, visible={} chars",
            n + 1,
            todo.len(),
            graded.kind,
            graded.split,
            graded.correct,
            reply.thinking.chars().count(),
            reply.text.chars().count()
        );
    }
    Ok(todo.len())
}

/// How a model is asked: which weights, under which system message, and how
/// long it may answer.
pub struct Model<'a> {
    /// The base checkpoint.
    pub base: &'a Path,
    /// The adapter, for a trained arm.
    pub adapter: Option<&'a Path>,
    /// The system message every question is asked under.
    pub system: &'a str,
    /// Tokens a reply may run.
    pub max_tokens: u32,
    /// The name the model's records carry.
    pub label: &'a str,
}

/// The exam against a local model. The model is loaded only when something is
/// pending, so a finished exam costs nothing to resume.
///
/// # Errors
/// The model cannot be loaded, a generation fails or `out` cannot be written.
pub fn run<Q: Question>(
    questions: &[Q],
    out: &Path,
    limit: Option<usize>,
    model: &Model<'_>,
    is_correct: &dyn Fn(&Q, &str) -> bool,
) -> anyhow::Result<usize> {
    if pending(questions, out, limit)?.is_empty() {
        return Ok(0);
    }
    let answerer = Answerer::load(model.base, model.adapter, 4096, model.label)?;
    let runtime = tokio::runtime::Runtime::new()?;
    let mut ask =
        |q: &Q| runtime.block_on(answerer.ask(model.system, q.prompt(), model.max_tokens));
    run_with(questions, out, limit, &mut ask, is_correct)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Q {
        id: String,
        reference: String,
    }

    impl Question for Q {
        fn id(&self) -> &str {
            &self.id
        }
        fn kind(&self) -> &str {
            "year"
        }
        fn split(&self) -> &str {
            "exam"
        }
        fn reference(&self) -> &str {
            &self.reference
        }
        fn prompt(&self) -> &str {
            "when?"
        }
    }

    fn questions(n: usize) -> Vec<Q> {
        (0..n)
            .map(|i| Q {
                id: format!("q{i}"),
                reference: "1770".into(),
            })
            .collect()
    }

    fn replying(text: &'static str) -> impl FnMut(&Q) -> anyhow::Result<Reply> {
        move |_| {
            Ok(Reply {
                text: text.into(),
                tokens: 5,
                seconds: 0.5,
                ..Reply::default()
            })
        }
    }

    fn dir() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    #[test]
    fn every_question_is_graded_against_its_reference_and_recorded() {
        let d = dir();
        let out = d.path().join("r.jsonl");
        let n = run_with(&questions(3), &out, None, &mut replying("1770"), &|q, a| {
            a.contains(&q.reference)
        })
        .unwrap();
        assert_eq!(n, 3);
        let results = read_results(&out).unwrap();
        assert_eq!(results.len(), 3);
        assert!(results
            .iter()
            .all(|g| g.correct && g.kind == "year" && g.split == "exam" && g.tokens == 5));
    }

    #[test]
    fn a_run_resumes_where_it_stopped_and_asks_nothing_twice() {
        let d = dir();
        let out = d.path().join("r.jsonl");
        let qs = questions(5);
        run_with(&qs, &out, Some(2), &mut replying("1770"), &|_, _| true).unwrap();
        assert_eq!(read_results(&out).unwrap().len(), 2);
        let mut asked = Vec::new();
        let mut recording = |q: &Q| {
            asked.push(q.id.clone());
            Ok(Reply {
                text: "x".into(),
                ..Reply::default()
            })
        };
        let n = run_with(&qs, &out, None, &mut recording, &|_, _| true).unwrap();
        assert_eq!(n, 3);
        assert_eq!(asked, ["q2", "q3", "q4"]);
        assert_eq!(read_results(&out).unwrap().len(), 5);
    }

    #[test]
    fn a_question_the_model_did_not_answer_is_recorded_as_unanswered_not_wrong_by_guess() {
        let d = dir();
        let out = d.path().join("r.jsonl");
        let mut silent = |_: &Q| {
            Ok(Reply {
                truncated: true,
                thinking: "still weighing 1770 or 1771".into(),
                ..Reply::default()
            })
        };
        run_with(&questions(1), &out, None, &mut silent, &|_, _| true).unwrap();
        let g = &read_results(&out).unwrap()[0];
        assert!(g.answer.is_none() && !g.correct && g.truncated);
    }

    #[test]
    fn a_long_answer_is_cut_for_the_record() {
        let d = dir();
        let out = d.path().join("r.jsonl");
        let mut long = |_: &Q| {
            Ok(Reply {
                text: "x".repeat(2000),
                ..Reply::default()
            })
        };
        run_with(&questions(1), &out, None, &mut long, &|_, _| false).unwrap();
        assert_eq!(
            read_results(&out).unwrap()[0]
                .answer
                .as_deref()
                .unwrap()
                .chars()
                .count(),
            RECORDED_ANSWER_CHARS
        );
    }

    #[test]
    fn a_failed_generation_stops_the_run_and_keeps_what_was_measured() {
        let d = dir();
        let out = d.path().join("r.jsonl");
        let mut calls = 0;
        let mut flaky = |_: &Q| {
            calls += 1;
            if calls == 3 {
                anyhow::bail!("device lost");
            }
            Ok(Reply {
                text: "1770".into(),
                ..Reply::default()
            })
        };
        assert!(run_with(&questions(5), &out, None, &mut flaky, &|_, _| true).is_err());
        assert_eq!(read_results(&out).unwrap().len(), 2);
    }

    #[test]
    fn a_missing_results_file_is_an_empty_one() {
        assert!(read_results(Path::new("/nonexistent/results.jsonl"))
            .unwrap()
            .is_empty());
    }
}
