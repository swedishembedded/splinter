// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measured validation of continual learning
// from a user's own agent sessions for its clients. If your team needs
// expertise in proving what a nightly update keeps, replaces and drops, you
// can procure our services by sending an email to info@swedishembedded.com.

//! The results of a storyline as JSON and as Markdown: what each night did,
//! the table of items by runs, and each prediction beside the answers.

use std::fmt::Write as _;

use serde_json::{json, Value};

use crate::study::{Cell, RunData, Study};

/// The most characters of an answer a report quotes.
const QUOTE_CHARS: usize = 220;

fn count(value: &Value) -> usize {
    value.as_array().map_or(0, Vec::len)
}

/// A one-line quote of an answer.
fn quote(text: &str) -> String {
    let flat: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= QUOTE_CHARS {
        return flat;
    }
    let cut: String = flat.chars().take(QUOTE_CHARS).collect();
    format!("{cut}...")
}

/// What the claims stage and the gate did on a night, from the absorb report.
#[must_use]
pub fn night_summary(data: &RunData) -> Value {
    let Some(report) = &data.absorb else {
        return Value::Null;
    };
    let names: Vec<(String, String)> = report["intake"]["sessions"]
        .as_array()
        .map(|sessions| {
            sessions
                .iter()
                .map(|s| {
                    let path = s["path"].as_str().unwrap_or_default();
                    let stem = path
                        .rsplit('/')
                        .next()
                        .unwrap_or(path)
                        .trim_end_matches(".atif.json");
                    (
                        s["source"].as_str().unwrap_or_default().to_string(),
                        stem.to_string(),
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    let rulings: Vec<Value> = report["claims"]["rulings"]
        .as_array()
        .map(|rulings| {
            rulings
                .iter()
                .map(|r| {
                    let session = r["session"].as_str().unwrap_or_default();
                    let name = names
                        .iter()
                        .find(|(id, _)| id == session)
                        .map_or(session, |(_, n)| n.as_str());
                    json!({
                        "session": name,
                        "kind": r["kind"],
                        "statement": r["statement"],
                        "outcome": r["outcome"],
                        "supersedes": count(&r["supersedes"]),
                        "reason": r["reason_code"],
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    let stages: Value = data
        .record
        .as_ref()
        .and_then(|r| r["stages"].as_array())
        .map_or(Value::Null, |stages| {
            Value::Object(
                stages
                    .iter()
                    .filter_map(|s| {
                        Some((
                            s["stage"].as_str()?.to_string(),
                            json!(s["duration_ms"].as_f64()? / 1000.0),
                        ))
                    })
                    .collect(),
            )
        });
    json!({
        "run": data.run,
        "exit": data.absorb_exit,
        "absorb_seconds": data.absorb_seconds,
        "stage_seconds": stages,
        "error": report["error"],
        "stopped": report["stopped"],
        "proposals": report["claims"]["proposals"],
        "admitted": report["claims"]["admitted"],
        "superseded": report["claims"]["superseded"],
        "reinforced": report["claims"]["reinforced"],
        "refused": report["claims"]["refused"],
        "live": report["live"],
        "pending": count(&report["pending"]),
        "kits_built": report["kits"]["built"],
        "kits_reused": report["kits"]["reused"],
        "records_trained": report["dataset"]["trained"],
        "gate_passed": report["gate"]["passed"],
        "gained": count(&report["gate"]["gained"]),
        "regressed": count(&report["gate"]["regressed"]),
        "unanswered": count(&report["gate"]["unanswered"]),
        "release": report["release"],
        "rulings": rulings,
    })
}

/// The whole study as JSON.
#[must_use]
pub fn results_json(study: &Study) -> Value {
    let last = study.runs.last().map_or(0, |r| r.run);
    let table: Vec<Value> = study
        .story
        .items
        .iter()
        .map(|item| {
            let cells: Vec<Cell> = (0..=last)
                .filter_map(|run| study.cell(&item.id, run))
                .collect();
            json!({"item": item.id, "cells": cells})
        })
        .collect();
    let scenarios: Vec<Value> = study
        .story
        .scenarios
        .iter()
        .map(|s| {
            let verdicts: Vec<_> = s.checks.iter().map(|c| study.verdict(c)).collect();
            json!({"id": s.id, "title": s.title, "expectation": s.expectation, "checks": verdicts})
        })
        .collect();
    json!({
        "storyline": study.story.name,
        "runs_done": study.runs.len(),
        "nights": study.runs.iter().map(night_summary).collect::<Vec<_>>(),
        "drift": study.runs.iter().filter_map(|r| study.drift(r.run).map(|(c, n)| json!({"run": r.run, "changed": c, "compared": n}))).collect::<Vec<_>>(),
        "table": table,
        "scenarios": scenarios,
        "scans": study.runs.iter().filter_map(|r| r.scan.as_ref().map(|s| json!({"run": r.run, "scan": s}))).collect::<Vec<_>>(),
        "forgettings": study.runs.iter().filter_map(|r| r.forget.as_ref().map(|f| json!({"run": r.run, "forget": f}))).collect::<Vec<_>>(),
    })
}

fn nights_section(study: &Study, out: &mut String) {
    out.push_str("## Nights\n\n| run | absorb s | exit | proposals | admitted | superseded | reinforced | refused | live | records | gained | regressed | released | answers changed |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
    for data in study.runs.iter().filter(|r| r.run > 0) {
        let s = night_summary(data);
        let drift = study
            .drift(data.run)
            .map_or("-".to_string(), |(c, n)| format!("{c}/{n}"));
        let show = |v: &Value| {
            if v.is_null() {
                "-".to_string()
            } else {
                v.to_string().trim_matches('"').to_string()
            }
        };
        let _ = writeln!(
            out,
            "| {} | {:.0} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            data.run,
            data.absorb_seconds.unwrap_or(0.0),
            show(&s["exit"]),
            show(&s["proposals"]),
            show(&s["admitted"]),
            show(&s["superseded"]),
            show(&s["reinforced"]),
            show(&s["refused"]),
            show(&s["live"]),
            show(&s["records_trained"]),
            show(&s["gained"]),
            show(&s["regressed"]),
            if s["release"].is_null() { "no" } else { "yes" },
            drift,
        );
    }
    out.push('\n');
    for data in study.runs.iter().filter(|r| r.run > 0) {
        let s = night_summary(data);
        let _ = writeln!(out, "### Run {} ledger rulings\n", data.run);
        if let Some(rulings) = s["rulings"].as_array().filter(|r| !r.is_empty()) {
            for r in rulings {
                let _ = writeln!(
                    out,
                    "- `{}` {}: {} (supersedes {}{})",
                    r["session"].as_str().unwrap_or("?"),
                    r["outcome"].as_str().unwrap_or("?"),
                    r["statement"].as_str().unwrap_or("?"),
                    r["supersedes"],
                    r["reason"]
                        .as_str()
                        .map(|c| format!(", reason {c}"))
                        .unwrap_or_default(),
                );
            }
        } else {
            let _ = writeln!(
                out,
                "No claim was ruled on. stopped: {}, error: {}",
                s["stopped"], s["error"]
            );
        }
        out.push('\n');
    }
}

fn table_section(study: &Study, out: &mut String) {
    let last = study.runs.last().map_or(0, |r| r.run);
    out.push_str("## Items by runs\n\nEach cell counts the grades of the three wordings: T the right value alone, M it with another, S another value only, O no value. `!` marks a cell that is not the wanted outcome (the right value alone, or nothing when nothing is right). Run 0 is the base.\n\n| item |");
    for run in 0..=last {
        let _ = write!(out, " {run} |");
    }
    out.push_str("\n|---|");
    for _ in 0..=last {
        out.push_str("---|");
    }
    out.push('\n');
    for item in &study.story.items {
        let _ = write!(out, "| {} |", item.id);
        for run in 0..=last {
            match study.cell(&item.id, run) {
                Some(cell) => {
                    let _ = write!(
                        out,
                        " {}{} |",
                        cell.tally,
                        if cell.wanted { "" } else { "!" }
                    );
                }
                None => out.push_str("  |"),
            }
        }
        out.push('\n');
    }
    out.push('\n');
}

fn scenarios_section(study: &Study, out: &mut String) {
    out.push_str("## Scenarios\n\n");
    for scenario in &study.story.scenarios {
        let _ = writeln!(
            out,
            "### {}: {}\n\n**Expected.** {}\n",
            scenario.id, scenario.title, scenario.expectation
        );
        if !scenario.checks.is_empty() {
            out.push_str(
                "| item | after run | predicted | actual | held |\n|---|---|---|---|---|\n",
            );
            for check in &scenario.checks {
                let v = study.verdict(check);
                let held = match v.held {
                    Some(true) => "yes",
                    Some(false) => "NO",
                    None => "not run",
                };
                let _ = writeln!(
                    out,
                    "| {} | {} | {:?} | {} | {} |",
                    check.item,
                    check.night,
                    check.expect,
                    v.actual.unwrap_or_else(|| "-".into()),
                    held
                );
            }
            out.push('\n');
        }
        let mut runs: Vec<u32> = scenario.checks.iter().map(|c| c.night).collect();
        runs.extend(study.runs.last().map(|r| r.run));
        if scenario.checks.is_empty() {
            runs = study.runs.iter().map(|r| r.run).collect();
        }
        runs.sort_unstable();
        runs.dedup();
        for item in &scenario.items {
            for run in &runs {
                let answers = study.answers(item, *run);
                if answers.is_empty() {
                    continue;
                }
                let tally = study.cell(item, *run).map_or("-".to_string(), |c| c.tally);
                let _ = writeln!(out, "- `{item}` run {run} ({tally}):");
                for answer in answers {
                    let _ = writeln!(out, "  - \"{}\"", quote(&answer));
                }
            }
        }
        out.push('\n');
    }
}

/// The whole study as Markdown.
#[must_use]
pub fn results_markdown(study: &Study) -> String {
    let mut out = format!(
        "# Storyline `{}`: {} of {} runs done\n\n",
        study.story.name,
        study.runs.len(),
        study.story.nights.len() + 1
    );
    nights_section(study, &mut out);
    table_section(study, &mut out);
    scenarios_section(study, &mut out);
    for data in &study.runs {
        if let Some(scan) = &data.scan {
            let _ = writeln!(out, "Secret search after run {}: {}\n", data.run, scan);
        }
    }
    out
}
