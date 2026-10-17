// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verifiers that grade a model's answers
// against references without trusting the model, for its clients. If your
// team needs expertise in evidence-based evaluation of language models, you
// can procure our services by sending an email to info@swedishembedded.com.

//! Re-grades the answers a stored anchor run kept with the current stated
//! reference verifier, with no model run.
//!
//! `cargo run -p splinter-eval --example regrade_anchor -- <anchor.jsonl>
//! <run.json>...` prints one JSON object per run. A run report keeps every
//! graded item (`items`: the answer, the verdict and the grader's reason), so
//! every item is re-graded. An answer the report cut (`truncated`) cannot be,
//! and is counted under `unregradable`; it stays in the store under its
//! `stored` address.

use std::collections::BTreeMap;
use std::process::ExitCode;

use atif::{AgentProfile, Trajectory};
use serde_json::{json, Value};
use splinter_core::annotation::{Outcome, Producer};
use splinter_core::clock::FixedClock;
use splinter_core::experience::{
    Digest, Environment, Experience, Privileged, PrivilegedKind, Provenance, Span, Task,
};
use splinter_eval::verifiers::formal::StatedReferenceVerifier;
use splinter_eval::verifiers::normalise::Normalisation;
use splinter_eval::verifiers::Verifier;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some((suite, runs)) = args.split_first().filter(|(_, runs)| !runs.is_empty()) else {
        eprintln!("usage: regrade_anchor <anchor.jsonl> <run.json>...");
        return ExitCode::from(2);
    };
    match regrade(suite, runs) {
        Ok(reports) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&reports).unwrap_or_default()
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("regrade_anchor: {e}");
            ExitCode::FAILURE
        }
    }
}

fn regrade(suite: &str, runs: &[String]) -> Result<Vec<Value>, String> {
    let read = |path: &str| std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"));
    let mut references: BTreeMap<String, String> = BTreeMap::new();
    for line in read(suite)?.lines().filter(|l| !l.trim().is_empty()) {
        let task: Value = serde_json::from_str(line).map_err(|e| format!("{suite}: {e}"))?;
        if let (Some(instruction), Some(reference)) =
            (task["instruction"].as_str(), task["reference"].as_str())
        {
            references.insert(instruction.into(), reference.into());
        }
    }
    let verifier = StatedReferenceVerifier::new(
        Producer {
            name: "regrade/stated".into(),
            version: "1".into(),
        },
        Normalisation::LENIENT,
    );
    let mut reports = Vec::new();
    for path in runs {
        let run: Value = serde_json::from_str(&read(path)?).map_err(|e| format!("{path}: {e}"))?;
        let score = &run["scores"][0];
        let tasks = score["suite"]["tasks"].as_u64().unwrap_or(0);
        let items = score["items"].as_array().cloned().unwrap_or_default();
        let count = |verdict: &str| {
            items
                .iter()
                .filter(|item| item["verdict"].as_str() == Some(verdict))
                .count() as u64
        };
        let (passed_before, wrong_before, undecided) =
            (count("passed"), count("failed"), count("undecided"));
        let decided = passed_before + wrong_before;
        let (mut now_pass, mut now_fail, mut now_abstain, mut unregradable) =
            (0u64, 0u64, 0u64, 0u64);
        for entry in &items {
            let instruction = entry["instruction"].as_str().unwrap_or_default();
            let reference = references
                .get(instruction)
                .ok_or_else(|| format!("{path}: no reference for {instruction:?}"))?;
            if entry["answer"]["truncated"].as_bool() == Some(true) {
                unregradable += 1;
                continue;
            }
            match grade(
                &verifier,
                instruction,
                reference,
                entry["answer"]["text"].as_str(),
            )? {
                Outcome::Pass => now_pass += 1,
                Outcome::Fail => now_fail += 1,
                Outcome::Abstain => now_abstain += 1,
            }
        }
        reports.push(json!({
            "run": path,
            "model": run["model"],
            "tasks": tasks,
            "before": { "decided": decided, "passed": passed_before, "wrong": wrong_before, "undecided": undecided },
            "after": {
                "decided": now_pass + now_fail,
                "passed": now_pass,
                "wrong": now_fail,
                "undecided": now_abstain,
            },
            "unregradable": unregradable,
        }));
    }
    Ok(reports)
}

fn grade(
    verifier: &StatedReferenceVerifier,
    instruction: &str,
    reference: &str,
    answer: Option<&str>,
) -> Result<Outcome, String> {
    let source = "anchor";
    let span = Span::new(Digest::of(source.as_bytes()), 0, source.len() as u64)
        .map_err(|e| e.to_string())?;
    let task = Task::new(
        "recall",
        vec![span.clone()],
        Environment {
            kind: "closed-book".into(),
            spec: json!({}),
            snapshot: None,
        },
        instruction,
        vec![Privileged {
            kind: PrivilegedKind::Reference,
            content: reference.into(),
            span: Some(span),
        }],
    )
    .map_err(|e| e.to_string())?;
    let profile = AgentProfile {
        name: "regrade".into(),
        version: "1".into(),
        model_name: None,
        tool_definitions: None,
        extra: None,
    };
    let exp = Experience::new(
        task.clone(),
        Trajectory::new("ATIF-v1.7", profile),
        answer.map(str::to_string),
        Provenance::new("stored", &FixedClock::new("2026-10-16T00:00:00.000Z")),
    )
    .map_err(|e| e.to_string())?;
    let finding = verifier.verify(&task, &exp).map_err(|e| e.to_string())?;
    Ok(finding.outcome)
}
