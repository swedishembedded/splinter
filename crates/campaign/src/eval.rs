// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements delegated-task coding agents that measure
// what their fine-tuned models actually retained. If your team needs
// expertise in evaluation harnesses for model interfaces, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Fact evaluation: a facts dataset becomes a scored report.
//!
//! `eval-facts` reads the same JSONL schema `learn` and `explore` write
//! (user message = question, assistant message = reference answer), asks
//! the configured model each question exactly like `ask` does, and judges
//! every reply against its reference. A record is correct when TWO ways
//! of comparing agree:
//!
//! 1. normalized text: lowercase, whitespace collapsed, punctuation
//!    stripped and unit-spacing variance erased ("10.5 Mbit/s" equals
//!    "10.5Mbps");
//! 2. numeric-tolerant: every number in the reference reappears in the
//!    answer with the same value (exact for integers, 1% relative
//!    tolerance otherwise) carrying the same unit token.
//!
//! A reference without numbers is judged by the text way only. A reply
//! that is not exactly one `{"answer": string}` object is a parse
//! failure, scored wrong - a salvaged half-answer would inflate a score
//! the promotion gate trusts.

use anyhow::Context;
use splinter_store::write_atomic;

/// Everything one evaluation needs, as configured by the caller.
#[derive(Clone, Debug)]
pub struct EvalOptions {
    /// The JSONL facts dataset to evaluate.
    pub dataset: std::path::PathBuf,
    /// The report file, written atomically at the end.
    pub out: std::path::PathBuf,
    /// The model evaluated.
    pub model: splinter_policy::ModelSelection,
    /// Shuffle the records (with a time-seeded LCG, no dependency) before
    /// evaluating, so `--limit N` samples rather than truncates.
    pub shuffle: bool,
    /// Evaluate at most N records.
    pub limit: Option<usize>,
}

/// One judged question.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Entry {
    pub question: String,
    pub expected: String,
    pub got: String,
    pub correct: bool,
    /// Which comparison decided: the text way for numberless references,
    /// the numeric way when numbers disagreed, "parse_failure" when the
    /// reply was never an answer at all.
    pub basis: String,
}

/// What one evaluation produced, for the CLI's summary line and the report.
#[derive(Debug, serde::Serialize)]
pub struct Report {
    pub total: usize,
    pub correct: usize,
    pub incorrect: usize,
    pub parse_failures: usize,
    pub accuracy: f64,
    pub entries: Vec<Entry>,
    pub model: String,
}

/// One dataset record: the question and its reference answer.
#[derive(Debug, PartialEq, Eq)]
struct Record {
    question: String,
    expected: String,
}

/// Reads the facts dataset: one JSON object per line, the schema `learn`
/// and `explore` write - user message is the question, assistant message
/// the reference answer.
fn read_dataset(path: &std::path::Path) -> anyhow::Result<Vec<Record>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut records = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(line)
            .with_context(|| format!("{} line {}: not a JSON object", path.display(), n + 1))?;
        let messages = value
            .get("messages")
            .and_then(|m| m.as_array())
            .with_context(|| format!("{} line {}: no \"messages\" array", path.display(), n + 1))?;
        let content = |role: &str| -> anyhow::Result<String> {
            messages
                .iter()
                .find(|m| m.get("role").and_then(|r| r.as_str()) == Some(role))
                .and_then(|m| m.get("content").and_then(|c| c.as_str()))
                .map(str::to_string)
                .with_context(|| format!("{} line {}: no {role} message", path.display(), n + 1))
        };
        let expected = content("assistant")?;
        // A training file fed here by mistake carries the reply wrapper in
        // the reference (train wraps, eval doesn't). No correct reply could
        // match such a reference, so the report would read near-zero and lie
        // about the model - refuse the file instead of scoring it.
        if serde_json::from_str::<serde_json::Value>(&expected)
            .ok()
            .and_then(|v| v.get("answer").and_then(|a| a.as_str()).map(str::to_string))
            .is_some()
        {
            anyhow::bail!(
                "{} line {}: the reference is wrapped in an {{\"answer\": ...}} object - \
                 this looks like a TRAINING file; eval-facts needs bare reference answers",
                path.display(),
                n + 1
            );
        }
        records.push(Record {
            question: content("user")?,
            expected,
        });
    }
    Ok(records)
}

/// Time-seeded Fisher-Yates shuffle. No dependency: a fact evaluation
/// needs reproducible-enough sampling, not cryptography.
fn shuffle<T>(items: &mut [T]) {
    if items.len() < 2 {
        return;
    }
    let mut seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x2545F4914F6CDD1D)
        | 1;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for i in (1..items.len()).rev() {
        let j = (next() % (i as u64 + 1)) as usize;
        items.swap(i, j);
    }
}

/// Runs the whole evaluation: read, ask, judge, report.
pub fn run(options: EvalOptions) -> anyhow::Result<Report> {
    let mut records = read_dataset(&options.dataset)?;
    anyhow::ensure!(
        !records.is_empty(),
        "{}: no records found",
        options.dataset.display()
    );
    if options.shuffle {
        shuffle(&mut records);
    }
    if let Some(limit) = options.limit {
        records.truncate(limit);
        anyhow::ensure!(!records.is_empty(), "--limit must be at least 1");
    }

    let model = options.model.identity();
    // One provider for the whole evaluation: the load is the expensive
    // step and every question wants the same model anyway.
    let provider = options.model.provider()?;
    let rt = tokio::runtime::Runtime::new()?;

    let mut entries = Vec::new();
    let mut parse_failures = 0usize;
    for record in &records {
        let result: anyhow::Result<String> = rt.block_on(async {
            splinter_policy::complete_text(
                provider.as_ref(),
                &splinter_lab::answers::question_prompt(&record.question),
            )
            .await
        });
        let (correct, basis, got) = match result
            .ok()
            .and_then(|reply| splinter_lab::answers::parse_answer(&reply).ok())
        {
            Some(answer) => {
                let (correct, basis) = splinter_lab::answers::judge(&record.expected, &answer);
                (correct, basis.to_string(), answer)
            }
            None => {
                parse_failures += 1;
                (false, "parse_failure".to_string(), String::new())
            }
        };
        entries.push(Entry {
            question: record.question.clone(),
            expected: record.expected.clone(),
            got,
            correct,
            basis,
        });
    }

    let correct = entries.iter().filter(|e| e.correct).count();
    let report = Report {
        total: entries.len(),
        correct,
        incorrect: entries.len() - correct,
        parse_failures,
        accuracy: if entries.is_empty() {
            0.0
        } else {
            correct as f64 / entries.len() as f64
        },
        entries,
        model,
    };
    // Atomic write: a crash mid-report leaves no half-file a gate could
    // read as a finished evaluation.
    write_atomic(&options.out, &serde_json::to_string_pretty(&report)?)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dataset in the learn/explore schema reads as question/answer.
    #[test]
    fn dataset_records_read_from_the_learn_schema() {
        let dir = std::env::temp_dir().join(format!("loop-eval-ds-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("facts.jsonl");
        std::fs::write(
            &path,
            concat!(
                "{\"messages\":[{\"role\":\"user\",\"content\":\"Q1\",\"train\":false},",
                "{\"role\":\"assistant\",\"content\":\"A1\",\"train\":true}]}\n",
                "\n",
                "{\"messages\":[{\"role\":\"user\",\"content\":\"Q2\",\"train\":false},",
                "{\"role\":\"assistant\",\"content\":\"A2\",\"train\":true}]}\n",
            ),
        )
        .unwrap();
        let records = read_dataset(&path).unwrap();
        assert_eq!(
            records,
            vec![
                Record {
                    question: "Q1".into(),
                    expected: "A1".into()
                },
                Record {
                    question: "Q2".into(),
                    expected: "A2".into()
                },
            ]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A training file fed to the evaluator by mistake carries the reply
    /// wrapper in the reference - no correct reply could ever match it, so
    /// the whole report would read 0/37 and LIE about the model. The gate
    /// refuses the file instead.
    #[test]
    fn a_wrapped_reference_is_refused_not_scored() {
        let dir = std::env::temp_dir().join(format!("loop-eval-ds-w-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("train.jsonl");
        std::fs::write(
            &path,
            concat!(
                "{\"messages\":[{\"role\":\"user\",\"content\":\"Q1\",\"train\":false},",
                "{\"role\":\"assistant\",\"content\":\"{\\\"answer\\\": \\\"A1\\\"}\",",
                "\"train\":true}]}\n",
            ),
        )
        .unwrap();
        let err = read_dataset(&path).unwrap_err();
        assert!(
            err.to_string().to_lowercase().contains("training file"),
            "error should name the training-file mixup, got: {err:#}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The shuffle must not lose or duplicate anything.
    #[test]
    fn shuffle_is_a_permutation() {
        let mut items: Vec<u32> = (0..64).collect();
        shuffle(&mut items);
        items.sort_unstable();
        assert_eq!(items, (0..64).collect::<Vec<_>>());
    }
}
