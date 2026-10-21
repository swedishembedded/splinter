// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Re-gating: the current gate applied again to attempts that already passed.
//!
//! A rule added after attempts were stored (as `unjustified_content` was)
//! says nothing about them unless they are judged again. The model is not
//! asked anything: the stored trajectory, the stored plan, the commit's diff
//! and the parent tree are everything the gate reads for an attempt whose
//! tree matched. An attempt the new gate fails gets a new ledger line (it
//! supersedes the pass, so the report and the splits count it as failed, and a
//! `--redo-class` run can do it again) and a failing verdict next to the old
//! pass in the store (equal-strength verdicts that disagree decide nothing, so
//! the dataset leaves it out).
//!
//! Attempts that needed corrections are not re-gated: the calls made before
//! each correction are not stored, and the gate needs them.

use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;

use anyhow::{Context as _, Result};
use serde_json::json;
use splinter_sdk::vocabulary::annotation::{
    Annotation, AnnotationBody, Outcome as Judged, Strength,
};
use splinter_sdk::vocabulary::digest::Digest;
use splinter_sdk::vocabulary::experience::ExperienceId;
use splinter_sdk::Context;

use crate::attempt::{CommitPlan, TARGET_PREFIX};
use crate::budget::{hunks_of, Budget};
use crate::gate::{evaluate, GateConfig, GateInput, Verdict};
use crate::git::{mirror, Git};
use crate::ledger::{self, Entry, Outcome};
use crate::pipeline::{layout, producer};
use crate::plan::changed_lines;
use crate::scrub::Scrubber;
use crate::select::{prepare, SelectConfig, Selected};
use crate::trace::Trace;
use crate::tree::{read_tree, vocabulary};
use splinter_sdk::sandbox::scratch::Claim;

/// What to re-gate and with which gate.
#[derive(Clone, Debug)]
pub struct RegateConfig {
    /// The repository the state was made from (its history is read, never written).
    pub source: PathBuf,
    /// The state directory holding the ledger and the store.
    pub state: PathBuf,
    /// Selection bounds the attempts were made under.
    pub select: SelectConfig,
    /// Gate bounds.
    pub gate: GateConfig,
    /// Tool calls by commit size.
    pub budget: Budget,
    /// Whether to record the new verdicts (false: judge and report only).
    pub write: bool,
}

/// An attempt the current gate fails.
#[derive(Debug)]
pub struct Regated {
    /// The ledger entry that stood (a pass).
    pub was: Entry,
    /// What the current gate says.
    pub verdict: Verdict,
}

/// What re-gating found.
#[derive(Debug, Default)]
pub struct RegateReport {
    /// Passed attempts judged again.
    pub examined: usize,
    /// Passed attempts needing corrections, not judged.
    pub not_judged_corrected: usize,
    /// Passed attempts whose commit or experience could not be read (with why).
    pub unreadable: Vec<(String, String)>,
    /// Passes the current gate fails.
    pub failed: Vec<Regated>,
}

impl RegateReport {
    /// Passes the current gate keeps.
    #[must_use]
    pub fn kept(&self) -> usize {
        self.examined - self.failed.len()
    }

    /// How many of the failed passes carry each class.
    #[must_use]
    pub fn by_class(&self) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for f in &self.failed {
            for class in f.verdict.classes() {
                *counts.entry(class.to_string()).or_insert(0) += 1;
            }
        }
        counts
    }
}

/// The attempt tree paths the agent was told, found in what it did.
fn attempt_roots(trajectory: &serde_json::Value) -> Vec<String> {
    let text = trajectory.to_string();
    let found = regex::Regex::new(r"/[^\s\x22':;\\]*/attempt-[A-Za-z0-9]+/tree")
        .unwrap_or_else(|_| unreachable!("a fixed pattern"));
    let mut roots: Vec<String> = found
        .find_iter(&text)
        .map(|m| m.as_str().to_string())
        .collect();
    roots.sort();
    roots.dedup();
    roots
}

/// Judges every passed attempt of `config.state` again.
pub fn regate(ctx: &Context, config: &RegateConfig) -> Result<RegateReport> {
    let state = config
        .state
        .canonicalize()
        .context("resolving the state directory")?;
    let parts = layout(&state);
    let git = mirror(&config.source, &parts.mirror)?;
    let chain = git.first_parent_chain(None, None)?;
    let scrubber = Scrubber::new(
        chain
            .iter()
            .flat_map(|c| [c.author_name.as_str(), c.author_email.as_str()]),
    );
    let forbidden: Vec<String> = [&config.source, &parts.mirror]
        .into_iter()
        .flat_map(|p| [Some(p.clone()), p.canonicalize().ok()])
        .flatten()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();

    let entries = ledger::read(&state)?;
    let passed: Vec<&Entry> = ledger::latest(&entries)
        .into_iter()
        .filter(|e| e.outcome == Outcome::Passed)
        .collect();
    let mut report = RegateReport::default();
    // Per commit, read once for both plan styles.
    let mut commits: BTreeMap<String, Option<(CommitPlan, HashSet<String>)>> = BTreeMap::new();

    for entry in passed {
        if entry.corrections > 0 {
            report.not_judged_corrected += 1;
            continue;
        }
        let unreadable = |why: String| (entry.commit.clone(), why);
        let known = commits.entry(entry.commit.clone()).or_insert_with(|| {
            read_commit(
                &git,
                &chain,
                &scrubber,
                config,
                &parts.scratch,
                &entry.commit,
            )
            .ok()
        });
        let Some((commit, parent_words)) = known else {
            report.unreadable.push(unreadable(
                "the commit is not replayable under these bounds".into(),
            ));
            continue;
        };
        let Some(hex) = entry.experience.as_deref() else {
            report
                .unreadable
                .push(unreadable("the ledger names no experience".into()));
            continue;
        };
        let id = ExperienceId(
            Digest::from_content_hex(hex)
                .with_context(|| format!("{hex:?} is not an experience id"))?,
        );
        let experience = match ctx.experiences().get(&id) {
            Ok(e) => e,
            Err(e) => {
                report
                    .unreadable
                    .push(unreadable(format!("reading the experience: {e}")));
                continue;
            }
        };
        let trace = Trace::of(&experience.trajectory);
        let roots = attempt_roots(&serde_json::to_value(&experience.trajectory)?);
        let calls = config
            .budget
            .calls(commit.changes.len(), hunks_of(&commit.diff));
        let verdict = evaluate(
            &GateInput {
                changes: &commit.changes,
                plan: &experience.instruction,
                parent_words,
                diff: &commit.diff,
                added_text: &changed_lines(&commit.diff).join("\n"),
                trace: &trace,
                residual: &[],
                forbidden_locations: &forbidden,
                attempt_roots: &roots,
                stopped: None,
                guidance: &[],
            },
            &GateConfig {
                max_tool_calls: calls,
                ..config.gate
            },
        );
        report.examined += 1;
        if !verdict.passed {
            report.failed.push(Regated {
                was: entry.clone(),
                verdict,
            });
        }
    }

    if config.write {
        for f in &report.failed {
            record(ctx, &state, f)?;
        }
    }
    Ok(report)
}

/// The commit as a run sees it and the words its parent tree holds.
fn read_commit(
    git: &Git,
    chain: &[crate::git::CommitMeta],
    scrubber: &Scrubber,
    config: &RegateConfig,
    scratch_root: &std::path::Path,
    sha: &str,
) -> Result<(CommitPlan, HashSet<String>)> {
    let meta = chain
        .iter()
        .find(|m| m.sha == sha)
        .context("not on the first-parent chain")?;
    let Selected::Replay(plan) = prepare(git, meta, scrubber, &config.select)? else {
        anyhow::bail!("skipped by selection");
    };
    let claim = Claim::create(scratch_root, TARGET_PREFIX)?;
    let tree = claim.path().join("tree");
    std::fs::create_dir(&tree)?;
    git.export_tree(&plan.parent, &tree)?;
    let words = vocabulary(&read_tree(&tree)?);
    Ok((*plan, words))
}

/// Writes the new verdict: a ledger line that supersedes the pass, and a
/// failing verdict on the experience.
fn record(ctx: &Context, state: &std::path::Path, found: &Regated) -> Result<()> {
    let was = &found.was;
    let classes: Vec<String> = found
        .verdict
        .classes()
        .into_iter()
        .map(String::from)
        .collect();
    let id = was
        .experience
        .as_deref()
        .context("a passed attempt without an experience")?;
    ctx.experiences().annotate(&Annotation {
        experience: ExperienceId(Digest::from_content_hex(id)?),
        producer: producer(),
        body: AnnotationBody::Verdict {
            outcome: Judged::Fail,
            strength: Strength::Executable,
            evidence: json!({
                "regated": true,
                "classes": classes,
                "reasons": found.verdict.reasons,
                "commit": was.commit,
            }),
        },
    })?;
    ledger::append(
        state,
        &Entry {
            outcome: Outcome::Failed,
            classes,
            detail: found
                .verdict
                .reasons
                .first()
                .map(|r| format!("regated: {}", r.detail)),
            ..was.clone()
        },
    )
}
