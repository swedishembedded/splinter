// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements resumable, content-addressed experiment
// runs for its clients. If your team needs expertise in running long
// measurement campaigns that survive interruption, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Running a storyline one run at a time. Run 0 asks the base about every
//! item; run `n` writes night `n`'s sessions, carries out the forgettings
//! asked for before it, runs `splinter absorb` on them, records the ledger,
//! and asks every item taught so far of what is deployed afterwards. Every
//! file is written when its part is done and a part whose file exists is not
//! done again, so a run interrupted anywhere is resumed by running it again.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{anyhow, bail, Context as _};
use serde_json::{json, Value};

use crate::evaluate::{evaluate, Evaluation, POLICY_CONTEXT_TOKENS};
use crate::layout::Layout;
use crate::proc;
use crate::scan::scan_state;
use crate::story::{Forget, Night, Storyline};

/// The policy the evaluation asks: whatever `policy:default` resolves to.
pub const DEPLOYED: &str = "policy:default";

/// What the runner calls.
#[derive(Clone, Debug)]
pub struct Tools {
    /// The `splinter` binary that carries out the nights.
    pub splinter: PathBuf,
}

/// What a step did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    /// Every run was already done.
    Complete,
    /// This run was carried out.
    Ran(u32),
}

/// The first run not yet done, or `None` when all are.
#[must_use]
pub fn next_run(story: &Storyline, layout: &Layout) -> Option<u32> {
    (0..=story.nights.len() as u32).find(|run| !layout.run(*run).join("done.json").exists())
}

/// Carries out the next run that is not done.
///
/// # Errors
/// A command could not run, or the absorb failed where it should not have.
pub fn step(story: &Storyline, layout: &Layout, tools: &Tools) -> anyhow::Result<Step> {
    let Some(run) = next_run(story, layout) else {
        return Ok(Step::Complete);
    };
    let dir = layout.run(run);
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    if run == 0 {
        let items: Vec<_> = story.items.iter().collect();
        ask_all(layout, &dir.join("eval.json"), DEPLOYED, &items)?;
    } else {
        night(story, layout, tools, run, &dir)?;
    }
    write(&dir.join("done.json"), &json!({"run": run}))?;
    Ok(Step::Ran(run))
}

fn night(
    story: &Storyline,
    layout: &Layout,
    tools: &Tools,
    run: u32,
    dir: &Path,
) -> anyhow::Result<()> {
    let night = &story.nights[run as usize - 1];
    let sessions = layout.sessions(run);
    write_sessions(night, &sessions)?;
    if !night.forgets.is_empty() && !dir.join("forget.json").exists() {
        let done = night
            .forgets
            .iter()
            .map(|f| forget(tools, layout, f))
            .collect::<anyhow::Result<Vec<_>>>()?;
        write(&dir.join("forget.json"), &Value::Array(done))?;
    }
    if !dir.join("absorb.exit").exists() {
        absorb(story, layout, tools, night, &sessions, dir)?;
    }
    if !dir.join("ledger.json").exists() {
        let ledger = run_json(
            &mut splinter(tools, layout, &["claims", "ledger"]),
            &dir.join("splinter.stderr"),
        )?;
        write(&dir.join("ledger.json"), &ledger)?;
    }
    if let Some(scan) = &story.leak_scan {
        if !dir.join("scan.json").exists() {
            write(&dir.join("scan.json"), &scan_state(&layout.state(), scan)?)?;
        }
    }
    let absorbed: Value = read(&dir.join("absorb.json"))?;
    let items = story.introduced(run);
    ask_all(layout, &dir.join("eval.json"), DEPLOYED, &items)?;
    if let Some(reference) = candidate_reference(&absorbed) {
        ask_all(layout, &dir.join("candidate-eval.json"), &reference, &items)?;
    }
    Ok(())
}

/// The reference that asks the candidate a night trained but did not
/// release, so what the gate withheld can be read.
fn candidate_reference(absorbed: &Value) -> Option<String> {
    if !absorbed["release"].is_null() {
        return None;
    }
    let adapter = absorbed["candidate"]["adapter"].as_str()?;
    let from = absorbed["candidate"]["from"].as_str()?;
    let (checkpoint, tokens) = from
        .rsplit_once('@')
        .map_or((from, None), |(c, t)| (c, Some(t)));
    let limit = tokens.map(|t| format!("@{t}")).unwrap_or_default();
    Some(format!("{checkpoint}+{adapter}{limit}"))
}

fn write_sessions(night: &Night, dir: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    for session in &night.sessions {
        session.script.write(dir)?;
    }
    Ok(())
}

fn splinter(tools: &Tools, layout: &Layout, args: &[&str]) -> Command {
    let mut command = Command::new(&tools.splinter);
    command
        .arg("--state")
        .arg(layout.state())
        .args([
            "--policy-context-tokens",
            &POLICY_CONTEXT_TOKENS.to_string(),
            "--json",
        ])
        .args(args);
    command
}

fn run_json(command: &mut Command, log: &Path) -> anyhow::Result<Value> {
    let finished = proc::run(command, log)?;
    serde_json::from_str(&finished.stdout).with_context(|| {
        format!(
            "{command:?} (exit {}) printed no JSON: {}",
            finished.exit, finished.stdout
        )
    })
}

fn absorb(
    story: &Storyline,
    layout: &Layout,
    tools: &Tools,
    night: &Night,
    sessions: &Path,
    dir: &Path,
) -> anyhow::Result<()> {
    let sessions = sessions.display().to_string();
    let mut args = vec!["absorb", sessions.as_str()];
    args.extend(story.absorb_args.iter().map(String::as_str));
    let finished = proc::run(
        &mut splinter(tools, layout, &args),
        &dir.join("absorb.stderr"),
    )?;
    append(
        &dir.join("absorb.attempts"),
        &format!("{:.1}\n", finished.seconds),
    )?;
    let report: Value = serde_json::from_str(&finished.stdout).with_context(|| {
        format!(
            "absorb of night {} (exit {}) printed no JSON: {}",
            night.number, finished.exit, finished.stdout
        )
    })?;
    if let Some(error) = report["error"].as_str() {
        if !night.sessions.is_empty() {
            bail!("absorb of night {} failed: {error}", night.number);
        }
    }
    write(&dir.join("absorb.json"), &report)?;
    if let Some(id) = report["run"].as_str() {
        if let Ok(record) = run_json(
            &mut splinter(tools, layout, &["runs", "show", id]),
            &dir.join("splinter.stderr"),
        ) {
            write(&dir.join("run.json"), &record)?;
        }
    }
    std::fs::write(dir.join("absorb.exit"), format!("{}\n", finished.exit))
        .context("writing absorb.exit")?;
    Ok(())
}

fn forget(tools: &Tools, layout: &Layout, forget: &Forget) -> anyhow::Result<Value> {
    let log = layout.root().join("forget.stderr");
    let ledger = run_json(&mut splinter(tools, layout, &["claims", "ledger"]), &log)?;
    let live = ledger["live"].as_array().cloned().unwrap_or_default();
    let matching: Vec<&Value> = live
        .iter()
        .filter(|claim| {
            let statement = claim["statement"]
                .as_str()
                .unwrap_or_default()
                .to_lowercase();
            forget
                .terms
                .iter()
                .all(|t| statement.contains(&t.to_lowercase()))
        })
        .collect();
    let [claim] = matching.as_slice() else {
        return Ok(json!({
            "item": forget.item,
            "terms": forget.terms,
            "error": format!("{} live claims hold the terms", matching.len()),
        }));
    };
    let id = claim["claim"]
        .as_str()
        .ok_or_else(|| anyhow!("a live claim without an id"))?;
    let result = run_json(
        &mut splinter(tools, layout, &["claims", "forget", id]),
        &log,
    )?;
    Ok(json!({
        "item": forget.item,
        "claim": id,
        "statement": claim["statement"],
        "result": result,
    }))
}

fn ask_all(
    layout: &Layout,
    target: &Path,
    policy: &str,
    items: &[&crate::item::Item],
) -> anyhow::Result<()> {
    if target.exists() {
        return Ok(());
    }
    let evaluation: Evaluation = evaluate(&layout.state(), policy, items)?;
    write(target, &serde_json::to_value(evaluation)?)
}

fn write(path: &Path, value: &Value) -> anyhow::Result<()> {
    let partial = path.with_extension("partial");
    std::fs::write(&partial, serde_json::to_string_pretty(value)?)
        .with_context(|| format!("writing {}", partial.display()))?;
    std::fs::rename(&partial, path).with_context(|| format!("moving {} into place", path.display()))
}

fn read(path: &Path) -> anyhow::Result<Value> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

fn append(path: &Path, text: &str) -> anyhow::Result<()> {
    use std::io::Write as _;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("opening {}", path.display()))?;
    file.write_all(text.as_bytes())
        .with_context(|| format!("writing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_withheld_candidate_is_asked_through_its_adapter_and_context_limit() {
        let report = json!({
            "release": null,
            "candidate": {"adapter": "/s/a.safetensors", "from": "local:/m/Qwen3-8B@8192"},
        });
        assert_eq!(
            candidate_reference(&report).as_deref(),
            Some("local:/m/Qwen3-8B+/s/a.safetensors@8192")
        );
        assert_eq!(
            candidate_reference(&json!({"release": "r", "candidate": {}})),
            None
        );
    }
}
