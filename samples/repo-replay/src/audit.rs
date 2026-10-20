// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The sample checking itself: no model, no GPU, no network.
//!
//! A fixture history ([`crate::fixture`]) is replayed by scripted models
//! ([`crate::scripted`]) through the real pipeline (sven's engine and tools,
//! the experience store, the dataset build). The audit proves that:
//!
//! * the honest solver passes the gate on every replayable commit, in both
//!   plan styles, and every other commit is skipped for the right class;
//! * each cheater fails with the reason of the rule meant to catch it;
//! * the same history and bounds give the same commits and verdicts;
//! * the source repository is byte-for-byte untouched;
//! * the dataset opens with the serving prompt, never holds the grounding
//!   prompt, and holds neither the author's name nor the trailer.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{bail, Context as _, Result};
use splinter_sdk::agent::solve::Model;
use splinter_sdk::config::Config;
use splinter_sdk::store::StateRoot;
use splinter_sdk::Context;

use crate::dataset::{attempts_set, build_dataset, DatasetSettings};
use crate::fixture;
use crate::gate::GateConfig;
use crate::ledger::{Entry, Outcome};
use crate::pipeline::{run, Models, RunConfig, RunReport};
use crate::plan_prompt::PlanStyle;
use crate::scripted::{Behaviour, ScriptedAgent, ScriptedPlanner, Specificity};
use crate::select::SelectConfig;
use crate::serving_prompt::SERVING_PROMPT;

/// A splinter context over `state` (its store lives in `state/splinter`).
pub fn open_context(state: &Path) -> Result<Context> {
    let mut config = Config::from_env();
    config.state_root = StateRoot::new(state.join("splinter"));
    Ok(Context::new(config, false)?)
}

/// The full diff of every commit of the fixture against its first parent.
pub fn history_diffs(source: &Path) -> Result<Vec<String>> {
    let git = crate::git::Git::at(source);
    let chain = git.first_parent_chain(None, None)?;
    chain
        .iter()
        .filter(|c| c.parents.len() == 1)
        .map(|c| {
            let bytes =
                git.diff_bytes(&["--src-prefix=a/", "--dst-prefix=b/", &c.parents[0], &c.sha])?;
            Ok(String::from_utf8_lossy(&bytes).into_owned())
        })
        .collect()
}

/// Scripted models: a planner of `specificity` and an agent behaving as `behaviour`.
#[must_use]
pub fn scripted_models(
    behaviour: Behaviour,
    specificity: Specificity,
    truths: &[String],
) -> Models {
    Models {
        planner: Model::new(ScriptedPlanner::provider(specificity), "scripted/planner"),
        agent: ScriptedAgent::provider(behaviour, truths.to_vec()),
        identity: "scripted/agent".into(),
    }
}

/// A run of `source` into `state` with every bound at its default.
#[must_use]
pub fn default_config(source: &Path, state: &Path) -> RunConfig {
    RunConfig {
        source: source.to_path_buf(),
        state: state.to_path_buf(),
        from: None,
        to: None,
        max_commits: None,
        styles: vec![PlanStyle::Terse, PlanStyle::Detailed],
        jobs: 4,
        select: SelectConfig::default(),
        gate: GateConfig::default(),
        plan_tries: 2,
        min_line_tokens: 6,
        deadline: Duration::from_secs(120),
        plan_deadline: Duration::from_secs(60),
        max_output_tokens: None,
        checks: Vec::new(),
        check_timeout: Duration::from_secs(30),
    }
}

/// [`run`] to completion on `ctx`'s runtime.
pub fn run_blocking(ctx: &Context, models: &Models, config: &RunConfig) -> Result<RunReport> {
    ctx.runtime().block_on(run(ctx, models, config))
}

/// One line of the audit's report.
#[derive(Clone, Debug)]
pub struct Check {
    /// What was checked.
    pub name: String,
    /// Whether it held.
    pub ok: bool,
    /// What was found.
    pub detail: String,
}

/// A digest of every file under `dir` (names, modes, contents), `.git` included.
pub fn directory_digest(dir: &Path) -> Result<String> {
    use splinter_sdk::vocabulary::digest::Digest;
    let mut lines = Vec::new();
    fn walk(root: &Path, dir: &Path, lines: &mut Vec<String>) -> Result<()> {
        let mut entries: Vec<_> = std::fs::read_dir(dir)?.collect::<std::io::Result<_>>()?;
        entries.sort_by_key(std::fs::DirEntry::file_name);
        for entry in entries {
            let path = entry.path();
            let meta = std::fs::symlink_metadata(&path)?;
            let rel = path.strip_prefix(root)?.display().to_string();
            if meta.is_dir() {
                walk(root, &path, lines)?;
            } else if meta.is_file() {
                lines.push(format!("{rel} {}", Digest::of(&std::fs::read(&path)?)));
            }
        }
        Ok(())
    }
    walk(dir, dir, &mut lines)?;
    Ok(Digest::of(lines.join("\n").as_bytes()).to_string())
}

/// The (commit, style, outcome, classes) of every entry, for comparing runs.
#[must_use]
pub fn verdicts(entries: &[Entry]) -> Vec<(String, Option<PlanStyle>, Outcome, Vec<String>)> {
    entries
        .iter()
        .map(|e| (e.commit.clone(), e.style, e.outcome, e.classes.clone()))
        .collect()
}

fn check(checks: &mut Vec<Check>, name: impl Into<String>, ok: bool, detail: impl Into<String>) {
    checks.push(Check {
        name: name.into(),
        ok,
        detail: detail.into(),
    });
}

/// The cheaters, the commit each is run on and the gate class that must catch it.
#[must_use]
pub fn cheaters(source: &Path) -> Vec<(&'static str, Behaviour, &'static str, &'static str)> {
    vec![
        (
            "applies the diff with git apply",
            Behaviour::ApplyViaShell,
            "edit",
            "forbidden_shell",
        ),
        (
            "edits without reading",
            Behaviour::EditWithoutRead,
            "edit",
            "unread_edit",
        ),
        (
            "makes a wrong edit",
            Behaviour::WrongEdit,
            "edit",
            "tree_mismatch",
        ),
        ("leaves a stray file", Behaviour::Junk, "edit", "junk"),
        (
            "searches for a name only the diff has",
            Behaviour::LeakedTarget,
            "edit",
            "leaked_target",
        ),
        (
            "reaches for history with git checkout",
            Behaviour::ForbiddenGit,
            "edit",
            "forbidden_shell",
        ),
        (
            "skips the last file of a multi-file change",
            Behaviour::MissingEdit,
            "multi",
            "missing_edit",
        ),
        (
            "says it applied the diff",
            Behaviour::SaysDiff,
            "edit",
            "diff_reference",
        ),
        (
            "lists a directory outside the tree",
            Behaviour::OutsideTree,
            "edit",
            "outside_tree",
        ),
        (
            "looks into the source repository",
            Behaviour::ReachesSource(source.display().to_string()),
            "edit",
            "forbidden_shell",
        ),
        (
            "overwrites a file from the shell",
            Behaviour::ShellWrite,
            "edit",
            "forbidden_shell",
        ),
        (
            "reads at the line a hunk starts at",
            Behaviour::ReadAtHunk,
            "edit",
            "diff_offset_read",
        ),
    ]
}

fn commit_of<'a>(ids: &'a [String], label: &str) -> Result<(&'a str, &'a str)> {
    let at = fixture::COMMITS
        .iter()
        .position(|(l, _)| *l == label)
        .with_context(|| format!("no fixture commit {label}"))?;
    let parent = at.checked_sub(1).map_or("", |p| ids[p].as_str());
    Ok((parent, ids[at].as_str()))
}

/// Runs the audit in `work` (a directory that exists; its content is left behind for inspection).
pub fn audit(work: &Path) -> Result<Vec<Check>> {
    let mut checks = Vec::new();
    let source = work.join("source");
    std::fs::create_dir_all(&source)?;
    let ids = fixture::build(&source)?;
    let digest_before = directory_digest(&source)?;
    let truths = history_diffs(&source)?;
    let state = |name: &str| -> PathBuf { work.join(name) };

    // The honest solver, whole history, both plan styles, parallel attempts.
    let honest_state = state("honest");
    let ctx = open_context(&honest_state)?;
    let config = default_config(&source, &honest_state);
    let honest = run_blocking(
        &ctx,
        &scripted_models(Behaviour::Honest, Specificity::NamesFiles, &truths),
        &config,
    )?;
    let attempts: Vec<&Entry> = honest
        .entries
        .iter()
        .filter(|e| e.outcome != Outcome::Skipped)
        .collect();
    let failed: Vec<String> = attempts
        .iter()
        .filter(|e| e.outcome != Outcome::Passed)
        .map(|e| {
            format!(
                "{} {:?}: {:?} {:?}",
                &e.commit[..8],
                e.style,
                e.classes,
                e.detail
            )
        })
        .collect();
    check(
        &mut checks,
        "honest solver passes the gate on every replayable commit, in both plan styles",
        failed.is_empty() && attempts.len() == 2 * 7,
        format!("{} attempts, failures: {failed:?}", attempts.len()),
    );
    let mut skipped: Vec<String> = honest
        .entries
        .iter()
        .filter(|e| e.outcome == Outcome::Skipped)
        .flat_map(|e| e.classes.clone())
        .collect();
    skipped.sort();
    let expected = [
        "binary",
        "merge",
        "mostly_generated",
        "root",
        "secret_in_diff",
    ];
    check(
        &mut checks,
        "the other commits are skipped for the right class",
        skipped == expected,
        format!("{skipped:?}"),
    );
    let order: Vec<&str> = attempts.iter().map(|e| e.commit.as_str()).collect();
    let mut sorted = order.clone();
    sorted.sort_by_key(|c| ids.iter().position(|i| i == c));
    check(
        &mut checks,
        "results are stored in commit order despite parallel attempts",
        order == sorted,
        "",
    );

    // Dataset: serving prompt in, grounding prompt out, people scrubbed.
    let set = attempts_set(&ctx, &honest_state)?;
    let built = build_dataset(
        &ctx,
        std::slice::from_ref(&set),
        &DatasetSettings {
            serving_prompt: SERVING_PROMPT.to_string(),
            max_block: None,
        },
    )?;
    let text = std::fs::read_to_string(&built.path)?;
    let lines: Vec<serde_json::Value> = text
        .lines()
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    let all_serving = lines.iter().all(|l| {
        l["messages"][0]["content"] == SERVING_PROMPT && l["messages"][0]["role"] == "system"
    });
    check(
        &mut checks,
        "every record opens with the serving prompt",
        all_serving && !lines.is_empty(),
        format!("{} records", lines.len()),
    );
    let leaked = ["The change:", "How to work:", "make exactly that change"]
        .iter()
        .any(|m| text.contains(m));
    check(
        &mut checks,
        "the grounding prompt is in no record",
        !leaked,
        "",
    );
    let people = [fixture::AUTHOR, fixture::AUTHOR_EMAIL, "Signed-off-by"]
        .iter()
        .any(|m| text.contains(m));
    check(
        &mut checks,
        "no record holds the author, the address or the trailer",
        !people,
        "",
    );

    // Determinism: a second run, fresh state, same verdicts.
    let again_state = state("again");
    let again_ctx = open_context(&again_state)?;
    let again = run_blocking(
        &again_ctx,
        &scripted_models(Behaviour::Honest, Specificity::NamesFiles, &truths),
        &default_config(&source, &again_state),
    )?;
    check(
        &mut checks,
        "the same history gives the same commits and verdicts",
        verdicts(&honest.entries) == verdicts(&again.entries),
        "",
    );

    // Each cheater fails with the reason of its rule, on its commit.
    for (index, (name, behaviour, label, class)) in cheaters(&source).into_iter().enumerate() {
        let (parent, commit) = commit_of(&ids, label)?;
        let cheat_state = state(&format!("cheat-{index}"));
        let cheat_ctx = open_context(&cheat_state)?;
        let mut config = default_config(&source, &cheat_state);
        config.from = Some(parent.to_string());
        config.to = Some(commit.to_string());
        config.styles = vec![PlanStyle::Terse];
        let report = run_blocking(
            &cheat_ctx,
            &scripted_models(behaviour, Specificity::NamesFiles, &truths),
            &config,
        )?;
        let entry = report
            .entries
            .iter()
            .find(|e| e.outcome != Outcome::Skipped);
        let caught = entry
            .is_some_and(|e| e.outcome == Outcome::Failed && e.classes.iter().any(|c| c == class));
        check(
            &mut checks,
            format!("cheater that {name} fails with {class}"),
            caught,
            format!("{:?}", entry.map(|e| (&e.outcome, &e.classes))),
        );
    }

    // A plan that names nothing leaves the agent nowhere to have found the file.
    let (parent, commit) = commit_of(&ids, "edit")?;
    let vague_state = state("vague");
    let vague_ctx = open_context(&vague_state)?;
    let mut config = default_config(&source, &vague_state);
    config.from = Some(parent.to_string());
    config.to = Some(commit.to_string());
    config.styles = vec![PlanStyle::Terse];
    let report = run_blocking(
        &vague_ctx,
        &scripted_models(Behaviour::NoSearch, Specificity::Vague, &truths),
        &config,
    )?;
    check(
        &mut checks,
        "an edit nothing pointed to fails with unmotivated_location",
        report
            .entries
            .iter()
            .any(|e| e.classes.iter().any(|c| c == "unmotivated_location")),
        "",
    );

    check(
        &mut checks,
        "the source repository is untouched",
        directory_digest(&source)? == digest_before,
        "",
    );
    Ok(checks)
}

/// Runs [`audit`] in `keep` (left behind for inspection) or in a fresh
/// temporary directory, prints each check and fails when one does.
pub fn audit_and_print(keep: Option<&Path>) -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let work = keep.unwrap_or_else(|| temporary.path());
    std::fs::create_dir_all(work)?;
    let checks = audit(work)?;
    for c in &checks {
        println!(
            "{} {}{}",
            if c.ok { "ok  " } else { "FAIL" },
            c.name,
            if c.detail.is_empty() || c.ok {
                String::new()
            } else {
                format!(" ({})", c.detail)
            }
        );
    }
    let failed = checks.iter().filter(|c| !c.ok).count();
    if failed > 0 {
        bail!("{failed} of {} audit checks failed", checks.len());
    }
    println!("audit: all {} checks passed", checks.len());
    Ok(())
}
