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
//! the end one table across the stages. Read
//! from ledger summaries alone, so a reused stage shows the same panel.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use crate::ledger::Commit;
use serde_json::Value;

/// A task's family: the last segment of its id (`dev-00002-count`).
fn family(task: &str) -> &str {
    task.rsplit('-').next().unwrap_or(task)
}

/// The probed checkpoints of `label`, in the order this run probed them.
fn stages(commits: &[Commit], label: &str) -> Vec<String> {
    let prefix = format!("probe/{label}/");
    let mut out: Vec<String> = Vec::new();
    for c in commits {
        if let Some(stage) = c.unit.strip_prefix(&prefix).filter(|s| !s.contains('/')) {
            if !out.iter().any(|s| s == stage) {
                out.push(stage.to_string());
            }
        }
    }
    out
}

/// The development pass rate the aptitude's `N*` is reported for.
const APTITUDE_TARGET: f64 = 0.75;

/// The fit `s(N) = b + k log10(N / N0)` of the development pass rate over
/// fine-tune steps (least squares), from the probes after each fine-tune
/// segment: the slope `k` is the learning aptitude, and `N*` the steps the
/// fit puts at `target`.
fn aptitude(commits: &[Commit], label: &str, target: f64) -> Option<String> {
    let mut points: Vec<(f64, f64)> = Vec::new();
    let mut steps = 0.0;
    for stage in stages(commits, label) {
        if stage != "sft" && !stage.starts_with("sft-") {
            continue;
        }
        let t = find(commits, &format!("{label}/{stage}"))?;
        steps += t.summary["steps"].as_f64()?;
        let p = find(commits, &format!("probe/{label}/{stage}"))?;
        points.push((
            steps,
            p.summary["passed"].as_f64()? / p.summary["total"].as_f64()?.max(1.0),
        ));
    }
    if points.len() < 2 {
        return None;
    }
    let n0 = points[0].0;
    let xs: Vec<f64> = points.iter().map(|(n, _)| (n / n0).log10()).collect();
    let count = xs.len() as f64;
    let (mx, my) = (
        xs.iter().sum::<f64>() / count,
        points.iter().map(|p| p.1).sum::<f64>() / count,
    );
    let sxx: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
    let sxy: f64 = xs
        .iter()
        .zip(&points)
        .map(|(x, p)| (x - mx) * (p.1 - my))
        .sum();
    let k = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    let b = my - k * mx;
    let reach = if k > 0.0 {
        format!(
            "N* = {:.0} steps to {target:.2}",
            n0 * 10f64.powf((target - b) / k)
        )
    } else {
        "no rise, so no N*".to_string()
    };
    let shown: Vec<String> = points
        .iter()
        .map(|(n, s)| format!("{n:.0}:{s:.2}"))
        .collect();
    Some(format!("learning aptitude (dev pass rate over fine-tune steps {}): k = {k:.3} per decade, b = {b:.3} at N0 = {n0:.0}; {reach}", shown.join(" ")))
}

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
        "         answer: {}  ({})",
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
        "  probe {name}: {}/{} development tasks solved ({}); first reply a tool call in {}/{}",
        summary["passed"],
        summary["total"],
        by_family(&examples),
        summary["first_tool"],
        summary["total"]
    );
    let held = &summary["heldout"];
    let held_examples: Vec<Value> = held["examples"].as_array().cloned().unwrap_or_default();
    if !held_examples.is_empty() {
        let _ = writeln!(
            s,
            "  held-out phrasings and formats: {}/{} solved ({}); first reply a tool call in {}/{}",
            held["passed"],
            held["total"],
            by_family(&held_examples),
            held["first_tool"],
            held["total"]
        );
    }
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
    if let Some(example) = held_examples.first() {
        transcript(&mut s, example);
    }
    s
}

fn find<'a>(commits: &'a [Commit], unit: &str) -> Option<&'a Commit> {
    commits.iter().rev().find(|c| c.unit == unit)
}

/// Training throughput of the stage that made the checkpoint at `stage`.
fn training(commits: &[Commit], label: &str, stage: &str) -> String {
    let unit = match stage {
        "init" => return String::new(),
        "pretrained" => "pretrain",
        "midtrained" => "midtrain",
        "improved" => "improve",
        other => other,
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
/// family and the training that made it.
#[must_use]
pub fn summary(commits: &[Commit], labels: &[String]) -> String {
    let mut s = String::new();
    for label in labels {
        let _ = writeln!(
            s,
            "\n{label}: what each checkpoint can do\n  {:<11} {:<7} {:<9} {:<9} {:<40} training",
            "stage", "probe", "held-out", "1st tool", "by skill"
        );
        for stage in stages(commits, label) {
            let stage = stage.as_str();
            let Some(p) = find(commits, &format!("probe/{label}/{stage}")) else {
                continue;
            };
            let examples: Vec<Value> = p.summary["examples"]
                .as_array()
                .cloned()
                .unwrap_or_default();
            let h = &p.summary["heldout"];
            let _ = writeln!(
                s,
                "  {:<11} {:<7} {:<9} {:<9} {:<40} {}",
                stage,
                format!("{}/{}", p.summary["passed"], p.summary["total"]),
                format!("{}/{}", h["passed"], h["total"]),
                format!("{}/{}", p.summary["first_tool"], p.summary["total"]),
                by_family(&examples),
                training(commits, label, stage)
            );
        }
        // Retrieval against pretraining tokens.
        let mut tokens = 0u64;
        let mut curve = Vec::new();
        for c in commits {
            let Some(name) = c.unit.strip_prefix(&format!("retrieval/{label}/")) else {
                continue;
            };
            if let Some(t) = find(commits, &format!("{label}/{name}")) {
                tokens += t.summary["tokens"].as_u64().unwrap_or(0);
            }
            curve.push(format!(
                "{:.0}M tok: {}+{}/{}",
                tokens as f64 / 1e6,
                c.summary["listing"],
                c.summary["path"],
                2 * c.summary["per_kind"].as_u64().unwrap_or(0)
            ));
        }
        if let Some(line) = aptitude(commits, label, APTITUDE_TARGET) {
            let _ = writeln!(s, "  {line}");
        }
        if !curve.is_empty() {
            let _ = writeln!(
                s,
                "  retrieval while pretraining (listing+path exact): {}",
                curve.join(", ")
            );
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
            shown.contains("answer: \"1407\"  (answer.txt holds \"1407\")"),
            "{shown}"
        );
    }
}
