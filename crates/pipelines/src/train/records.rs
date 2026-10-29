// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gated self-improvement loops where a model
// update is adopted only on held-out evidence. If your team needs expertise
// in evaluation-gated fine-tuning, you can procure our services by sending
// an email to info@swedishembedded.com.

//! The dataset files a training run reads: the run's datasets concatenated
//! into one record file, and whether its records carry step reasoning (which
//! trains the model for thinking mode).

use std::path::{Path, PathBuf};

use splinter_orchestrator::error::{io, OrchestratorError};

use super::TrainPlan;

/// Whether any message of the JSON Lines chat records at `path` carries the
/// reasoning of its step (`reasoning_content`).
pub(super) fn carries_reasoning(path: &Path) -> Result<bool, OrchestratorError> {
    use std::io::BufRead;
    let file = std::fs::File::open(path).map_err(io(path))?;
    for line in std::io::BufReader::new(file).lines() {
        let line = line.map_err(io(path))?;
        // Cheap test first: most lines of a dataset without reasoning never
        // parse.
        if !line.contains("\"reasoning_content\"") {
            continue;
        }
        let record: serde_json::Value = serde_json::from_str(&line).map_err(|e| {
            OrchestratorError::Refused(format!("{}: not a JSON record: {e}", path.display()))
        })?;
        let reasons = record["messages"].as_array().is_some_and(|messages| {
            messages.iter().any(|m| {
                m["reasoning_content"]
                    .as_str()
                    .is_some_and(|text| !text.trim().is_empty())
            })
        });
        if reasons {
            return Ok(true);
        }
    }
    Ok(false)
}

/// `plan`'s datasets concatenated in order into one file in its directory.
pub(super) fn combine(plan: &TrainPlan) -> Result<PathBuf, OrchestratorError> {
    let paths: Vec<&Path> = plan.datasets.iter().map(|d| d.path.as_path()).collect();
    concatenate(&paths, &plan.dir.join("dataset.jsonl"))
}

/// The records of `files`, in order, written to `into`; a record is a
/// non-blank line.
pub(super) fn concatenate(files: &[&Path], into: &Path) -> Result<PathBuf, OrchestratorError> {
    let mut text = String::new();
    for file in files {
        for line in std::fs::read_to_string(file)
            .map_err(io(file))?
            .lines()
            .filter(|l| !l.trim().is_empty())
        {
            text.push_str(line);
            text.push('\n');
        }
    }
    std::fs::write(into, &text).map_err(io(into))?;
    Ok(into.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dataset(name: &str, lines: &[&str]) -> PathBuf {
        let path = std::env::temp_dir().join(format!("{name}-{}.jsonl", std::process::id()));
        std::fs::write(&path, lines.join("\n")).unwrap();
        path
    }

    /// An agent's records carry the reasoning of their steps and train the
    /// model to reason; bare answers (and an empty reasoning field) do not.
    #[test]
    fn records_with_step_reasoning_train_for_thinking_mode() {
        let bare = dataset(
            "bare",
            &[
                r#"{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a","train":true}]}"#,
                r#"{"messages":[{"role":"assistant","content":"a","reasoning_content":"  "}]}"#,
            ],
        );
        let reasoning = dataset(
            "reasoning",
            &[
                r#"{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}"#,
                r#"{"messages":[{"role":"assistant","content":"","reasoning_content":"look at main.c first"}]}"#,
            ],
        );
        assert!(!carries_reasoning(&bare).unwrap());
        assert!(carries_reasoning(&reasoning).unwrap());
        std::fs::remove_file(bare).unwrap();
        std::fs::remove_file(reasoning).unwrap();
    }
}
