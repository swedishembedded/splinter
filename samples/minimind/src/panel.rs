// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training runs that show, stage by stage,
// what a model can do, for its clients. If your team needs expertise in
// observable small-model training, you can procure our services by sending
// an email to info@swedishembedded.com.

//! What the terminal shows of a model's capability: after every probe a
//! compact panel (passes by family, and a solved and a failed task with the
//! tool calls the model made, the answer it wrote and the verdict), and at
//! the end one table across the stages with the examination's tiers. Read
//! from ledger summaries alone, so a reused stage shows the same panel.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use serde_json::Value;
use splinter_sdk::store::stage_ledger::Commit;

/// A task's family: the last segment of its id (`dev-00002-count`).
fn family(task: &str) -> &str {
    task.rsplit('-').next().unwrap_or(task)
}

/// The probe stages in the order a model goes through them.
const STAGES: [&str; 4] = ["init", "pretrained", "sft", "improved"];

/// Passes by family, in first-seen order: `combine 1/4, lookup 0/4`.
fn by_family(examples: &[Value]) -> String {
    let mut order = Vec::new();
    let mut counts: BTreeMap<&str, (u32, u32)> = BTreeMap::new();
    for example in examples {
        let f = family(example["task"].as_str().unwrap_or(""));
        if !counts.contains_key(f) {
            order.push(f);
        }
        let entry = counts.entry(f).or_default();
        entry.1 += 1;
        if example["outcome"]["passed"].as_bool().unwrap_or(false) {
            entry.0 += 1;
        }
    }
    order
        .iter()
        .map(|f| format!("{f} {}/{}", counts[f].0, counts[f].1))
        .collect::<Vec<_>>()
        .join(", ")
}

fn clip(text: &str, n: usize) -> String {
    if text.chars().count() > n {
        format!("{}...", text.chars().take(n).collect::<String>())
    } else {
        text.to_string()
    }
}

/// One example task: its verdict, the request, then what the model did.
fn transcript(s: &mut String, example: &Value) {
    let o = &example["outcome"];
    let passed = o["passed"].as_bool().unwrap_or(false);
    let _ = writeln!(
        s,
        "    {} {}  {}",
        if passed { "PASS" } else { "fail" },
        example["task"].as_str().unwrap_or(""),
        clip(example["instruction"].as_str().unwrap_or(""), 90)
    );
    let actions: Vec<&str> = o["actions"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect();
    for action in &actions {
        let _ = writeln!(s, "         {action}");
    }
    if actions.is_empty() {
        let _ = writeln!(s, "         (no model call recorded)");
    }
    let _ = writeln!(
        s,
        "         answer.txt: {}  ({})",
        o["answer"]
            .as_str()
            .map_or("none".to_string(), |a| format!("{a:?}")),
        o["reason"].as_str().unwrap_or("")
    );
}

/// A probe's result as the terminal shows it.
#[must_use]
pub fn probe(name: &str, summary: &Value) -> String {
    let examples: Vec<Value> = summary["examples"].as_array().cloned().unwrap_or_default();
    let mut s = String::new();
    let _ = writeln!(
        s,
        "  probe {name}: {}/{} development tasks solved ({})",
        summary["passed"],
        summary["total"],
        by_family(&examples)
    );
    if let Some(text) = summary["text"]["continuation"].as_str() {
        let _ = writeln!(
            s,
            "    text  {:?} -> {:?}",
            summary["text"]["prompt"].as_str().unwrap_or(""),
            clip(text, 80)
        );
    } else if let Some(why) = summary["text"]["unavailable"].as_str() {
        let _ = writeln!(s, "    text  (not measured: {})", clip(why, 80));
    }
    let passed = |e: &&Value| e["outcome"]["passed"].as_bool().unwrap_or(false);
    let solved = examples.iter().find(passed);
    let failed = examples.iter().find(|e| !passed(e));
    for example in [solved, failed].into_iter().flatten() {
        transcript(&mut s, example);
    }
    s
}

fn find<'a>(commits: &'a [Commit], unit: &str) -> Option<&'a Commit> {
    commits.iter().rev().find(|c| c.unit == unit)
}

/// The exam pass rate of the model at `stage`, from whichever verdict
/// examined it, by tier.
fn exam_rates(commits: &[Commit], label: &str, stage: &str) -> BTreeMap<String, f64> {
    let sides: &[(&str, &str)] = match stage {
        "pretrained" => &[("scratch", "base")],
        "sft" => &[("scratch", "candidate"), ("experience", "base")],
        "improved" => &[("experience", "candidate")],
        _ => &[],
    };
    for (pair, side) in sides {
        if let Some(c) = find(commits, &format!("{label}/{pair}/verdict")) {
            return c.summary["tiers"]
                .as_object()
                .into_iter()
                .flatten()
                .filter_map(|(tier, r)| Some((tier.clone(), r[*side].as_f64()?)))
                .collect();
        }
    }
    BTreeMap::new()
}

/// Training throughput of the stage that made the checkpoint at `stage`.
fn training(commits: &[Commit], label: &str, stage: &str) -> String {
    let unit = match stage {
        "pretrained" => "pretrain",
        "sft" => "sft",
        "improved" => "improve",
        _ => return String::new(),
    };
    let Some(c) = find(commits, &format!("{label}/{unit}")) else {
        return String::new();
    };
    let secs = c.summary["train_seconds"].as_f64().unwrap_or(0.0);
    let loss = c.summary["last_loss"]
        .as_f64()
        .map_or("?".into(), |l| format!("{l:.3}"));
    match c.summary["tokens_per_second"].as_f64() {
        Some(rate) if rate > 0.0 => format!("{secs:.0} s, {rate:.0} tok/s, loss {loss}"),
        _ => format!("{secs:.0} s, loss {loss}"),
    }
}

/// The table across stages of every model in `labels`: probe passes by
/// family, the exam's pass rate per tier, and the training that made it.
#[must_use]
pub fn summary(commits: &[Commit], labels: &[String]) -> String {
    let mut s = String::new();
    for label in labels {
        let _ = writeln!(
            s,
            "\n{label}: what each checkpoint can do\n  {:<11} {:<7} {:<34} {:<26} training",
            "stage", "probe", "by family", "exam a / b / c / anchor"
        );
        for stage in STAGES {
            let Some(p) = find(commits, &format!("probe/{label}/{stage}")) else {
                continue;
            };
            let examples: Vec<Value> = p.summary["examples"]
                .as_array()
                .cloned()
                .unwrap_or_default();
            let rates = exam_rates(commits, label, stage);
            let exam = if rates.is_empty() {
                "not examined".to_string()
            } else {
                ["a", "b", "c", "anchor"]
                    .iter()
                    .map(|t| rates.get(*t).map_or("-".into(), |r| format!("{r:.2}")))
                    .collect::<Vec<_>>()
                    .join(" / ")
            };
            let _ = writeln!(
                s,
                "  {:<11} {:<7} {:<34} {:<26} {}",
                stage,
                format!("{}/{}", p.summary["passed"], p.summary["total"]),
                by_family(&examples),
                exam,
                training(commits, label, stage)
            );
        }
        for pair in ["scratch", "experience"] {
            if let Some(c) = find(commits, &format!("{label}/{pair}/verdict")) {
                let _ = writeln!(
                    s,
                    "  verdict {pair}: {} ({}, {})",
                    c.summary["verdict"].as_str().unwrap_or("?"),
                    if c.summary["verified"].as_bool().unwrap_or(false) {
                        "signature verified"
                    } else {
                        "signature NOT verified"
                    },
                    if c.summary["promotes"].as_bool().unwrap_or(false) {
                        "promoted"
                    } else {
                        "not promoted"
                    },
                );
            }
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_probe_panel_shows_passes_by_family_and_a_solved_and_a_failed_transcript() {
        let summary = serde_json::json!({
            "passed": 1, "total": 2,
            "text": { "prompt": "Velmora is a town by", "continuation": " a river." },
            "examples": [
                { "task": "dev-00000-lookup", "instruction": "Write the code of pilme to answer.txt.",
                  "outcome": { "passed": false, "answer": "1407", "reason": "answer.txt holds \"1407\"",
                               "actions": ["read_file({\"path\":\"registry.txt\"})", "write_file({\"path\":\"answer.txt\",\"text\":\"1407\"})"] } },
                { "task": "dev-00001-count", "instruction": "How many lines of log.txt contain WARN?",
                  "outcome": { "passed": true, "answer": "3", "reason": "answer matches", "actions": ["say \"3\""] } },
            ],
        });
        let shown = probe("ouro/sft", &summary);
        assert!(
            shown.contains("1/2 development tasks solved (lookup 0/1, count 1/1)"),
            "{shown}"
        );
        let pass = shown.find("PASS dev-00001-count").unwrap();
        let fail = shown.find("fail dev-00000-lookup").unwrap();
        assert!(pass < fail, "the solved one first: {shown}");
        assert!(
            shown.contains("read_file({\"path\":\"registry.txt\"})"),
            "{shown}"
        );
        assert!(
            shown.contains("answer.txt: \"1407\"  (answer.txt holds \"1407\")"),
            "{shown}"
        );
    }
}
