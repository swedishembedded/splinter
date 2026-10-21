// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Temporal train, eval and test datasets, and readable traces.
//!
//! The first-parent chain of the run's range is cut by position: the oldest
//! `train` percent of its commits are train, the next `eval` percent eval,
//! the rest (the newest) test. The cut is over every commit of the range, not
//! only the replayed ones, so the boundaries do not move with the pass rate
//! or with how far a resumable run has got. The cut is splinter's own
//! `temporal_bands`: both plan styles of a commit land in the same part, and
//! the split has a content address (`split_digest` in `splits.json`).
//!
//! Commits that are not each other's group can still repeat each other (a
//! change and its revert, a cherry-pick): the passed attempts of different
//! parts whose target changes share their lines are reported as pairs, the
//! build fails with them unless `--drop-overlaps` leaves the training-side
//! record out, and what was left out is counted in `splits.json`.
//!
//! Each split's dataset is splinter's own `sft-step` build over that split's
//! attempts; the boundary commits and counts are written to `splits.json`,
//! and every attempt (plan, verdict, reasons, the agent's calls and words, the
//! full trajectory) is written as one JSON file under `traces/<split>/`,
//! positives and negatives alike, so a reader can open any trace directly.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

use anyhow::{bail, Context as _, Result};
use serde_json::json;
use splinter_sdk::data::partition::Unit;
use splinter_sdk::data::split::{
    content_overlaps, temporal_bands, DataSplit, Member, Overlap, OverlapRule, Part,
};
use splinter_sdk::vocabulary::digest::Digest;
use splinter_sdk::vocabulary::experience::ExperienceId;
use splinter_sdk::Context;

use crate::dataset::{attempts_set_filtered, build_dataset, CorrectionRange, DatasetSettings};
use crate::git::{CommitMeta, Git};
use crate::ledger::{self, Outcome};
use crate::pipeline::layout;
use crate::trace::Trace;

/// The split names in time order.
pub const SPLITS: [&str; 3] = ["train", "eval", "test"];

/// The name a part goes by in this sample's output.
fn split_name(part: Part) -> &'static str {
    match part {
        Part::Train => "train",
        Part::Validation => "eval",
        Part::Test => "test",
    }
}

/// How alike the target changes of two commits must be to count as repeating
/// each other: four fifths of their changed lines in common.
const OVERLAP_RULE: OverlapRule = OverlapRule {
    min_similarity: 0.8,
    min_features: 3,
};

/// Shortest changed line (trimmed, without its sign) that identifies a change;
/// braces and blank lines occur in every diff.
const MIN_FEATURE_CHARS: usize = 8;

/// What to do with passed attempts of different splits that repeat each
/// other's target change.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Overlaps {
    /// Fail the build and name the pairs.
    #[default]
    Refuse,
    /// Leave out the record of the split nearer training and count it.
    Drop,
}

/// How the splits are cut and filled.
#[derive(Clone, Copy, Debug)]
pub struct SplitOptions {
    /// The oldest this percent of the commits are train.
    pub train_percent: usize,
    /// The next this percent of the commits are eval, the rest test.
    pub eval_percent: usize,
    /// Which attempts the datasets admit by the corrections they needed.
    pub corrections: CorrectionRange,
    /// What to do with attempts that repeat each other across splits.
    pub overlaps: Overlaps,
}

impl Default for SplitOptions {
    fn default() -> Self {
        Self {
            train_percent: 80,
            eval_percent: 10,
            corrections: CorrectionRange::default(),
            overlaps: Overlaps::Refuse,
        }
    }
}

/// The boundaries of the three bands over `total` commits at positions
/// `0..total`: the last position of train and the last of eval, each half a
/// position short of the next band so that no position sits on a boundary.
#[must_use]
pub fn band_edges(total: usize, train_percent: usize, eval_percent: usize) -> (f64, f64) {
    let edge = |end: usize| end as f64 - 0.5;
    (
        edge(total * train_percent / 100),
        edge(total * (train_percent + eval_percent) / 100),
    )
}

/// The part of the commit at `position` of `total`.
fn band_of(position: usize, total: usize, options: &SplitOptions) -> Part {
    let (train_until, eval_until) = band_edges(total, options.train_percent, options.eval_percent);
    let at = position as f64;
    if at <= train_until {
        Part::Train
    } else if at <= eval_until {
        Part::Validation
    } else {
        Part::Test
    }
}

/// The lines a change touches, without their signs and indentation, as a
/// set: a change and its revert (the same lines the other way round), a copy
/// and a cherry-pick of it share them. Lines too short to identify anything
/// are left out.
#[must_use]
pub fn target_features(diff: &str) -> BTreeSet<String> {
    diff.lines()
        .filter(|l| !l.starts_with("+++") && !l.starts_with("---"))
        .filter_map(|l| l.strip_prefix('+').or_else(|| l.strip_prefix('-')))
        .map(str::trim)
        .filter(|l| l.chars().count() >= MIN_FEATURE_CHARS)
        .map(str::to_owned)
        .collect()
}

/// The target-change features of the commit `meta`, from the mirror.
fn commit_features(git: &Git, meta: &CommitMeta) -> Result<BTreeSet<String>> {
    let Some(parent) = meta.parents.first() else {
        return Ok(BTreeSet::new());
    };
    Ok(target_features(&git.diff(parent, &meta.sha, &[])?))
}

/// The three-band split of the stored attempts of `state` over `chain`, and
/// the pairs of passed attempts across its parts that repeat each other.
struct Planned {
    split: DataSplit,
    overlaps: Vec<Overlap>,
}

/// One unit per stored attempt (its experience id), grouped by commit and
/// dated by the commit's position in `chain`.
fn members(entries: &[ledger::Entry], chain: &[CommitMeta]) -> Vec<Member> {
    let position: HashMap<&str, usize> = chain
        .iter()
        .enumerate()
        .map(|(at, c)| (c.sha.as_str(), at))
        .collect();
    entries
        .iter()
        .filter_map(|e| {
            let at = *position.get(e.commit.as_str())?;
            Some(Member {
                unit: Unit {
                    id: e.experience.clone()?,
                    group: e.commit.clone(),
                    stratum: e.style.map_or("-", |s| s.name()).to_owned(),
                },
                source: "history".into(),
                at: at as f64,
            })
        })
        .collect()
}

fn plan(
    git: &Git,
    entries: &[ledger::Entry],
    chain: &[CommitMeta],
    options: &SplitOptions,
) -> Result<Planned> {
    let (train_until, eval_until) =
        band_edges(chain.len(), options.train_percent, options.eval_percent);
    let units = members(entries, chain);
    let dataset = Digest::of(&serde_json::to_vec(
        &units.iter().map(|m| &m.unit.id).collect::<Vec<_>>(),
    )?);
    let split = temporal_bands(&units, train_until, eval_until, 0, &dataset)
        .context("cutting the splits")?;

    let meta: HashMap<&str, &CommitMeta> = chain.iter().map(|c| (c.sha.as_str(), c)).collect();
    let mut by_commit: HashMap<&str, BTreeSet<String>> = HashMap::new();
    let mut features = BTreeMap::new();
    for e in entries.iter().filter(|e| e.outcome == Outcome::Passed) {
        let (Some(id), Some(commit)) = (e.experience.as_ref(), meta.get(e.commit.as_str())) else {
            continue;
        };
        if !by_commit.contains_key(e.commit.as_str()) {
            by_commit.insert(&e.commit, commit_features(git, commit)?);
        }
        features.insert(id.clone(), by_commit[e.commit.as_str()].clone());
    }
    let overlaps = content_overlaps(&split, &features, &OVERLAP_RULE);
    Ok(Planned { split, overlaps })
}

/// The commit and style of the stored attempt `experience`.
fn describe(entries: &[ledger::Entry], experience: &str) -> serde_json::Value {
    let entry = entries
        .iter()
        .find(|e| e.experience.as_deref() == Some(experience));
    json!({
        "commit": entry.map(|e| e.commit.as_str()),
        "style": entry.and_then(|e| e.style).map(|s| s.name()),
    })
}

/// Writes the three datasets, `splits.json` and the traces under `out`.
pub fn write_splits(
    ctx: &Context,
    state: &Path,
    out: &Path,
    settings: &DatasetSettings,
    options: &SplitOptions,
) -> Result<serde_json::Value> {
    let state = state.canonicalize()?;
    let (from, to) = ledger::last_range(&state)?;
    let git = Git::at(layout(&state).mirror);
    let chain = git.first_parent_chain(from.as_deref(), to.as_deref())?;
    let total = chain.len();
    let entries = ledger::read(&state)?;
    let Planned {
        mut split,
        overlaps,
    } = plan(&git, &entries, &chain, options)?;

    let overlap_report: Vec<serde_json::Value> = overlaps
        .iter()
        .map(|o| {
            json!({
                "earlier": describe(&entries, &o.a), "earlier_split": split_name(o.part_a),
                "later": describe(&entries, &o.b), "later_split": split_name(o.part_b),
                "similarity": o.similarity,
            })
        })
        .collect();
    match options.overlaps {
        Overlaps::Drop => split = split.without_overlaps(&overlaps),
        Overlaps::Refuse if !overlaps.is_empty() => {
            let pairs: Vec<String> = overlap_report
                .iter()
                .take(5)
                .map(|p| p.to_string())
                .collect();
            bail!(
                "{} pair(s) of passed attempts in different splits duplicate or nearly duplicate \
                 each other's target change (first: {}); pass --drop-overlaps to leave the \
                 training-side record out",
                overlaps.len(),
                pairs.join(", ")
            );
        }
        Overlaps::Refuse => {}
    }

    // The split each stored attempt is in, by experience id.
    let part_of: HashMap<&str, Part> = Part::ALL
        .iter()
        .flat_map(|p| split.ids(*p).iter().map(move |id| (id.as_str(), *p)))
        .collect();

    let mut summary = serde_json::Map::new();
    for part in Part::ALL {
        let name = split_name(part);
        // The commits of the range in this band, replayed or not: the cut does
        // not move with how many of them were.
        let in_order: Vec<&str> = chain
            .iter()
            .enumerate()
            .filter(|(at, _)| band_of(*at, total, options) == part)
            .map(|(_, c)| c.sha.as_str())
            .collect();
        let mut entry = json!({
            "commits": in_order.len(),
            "first": in_order.first(),
            "last": in_order.last(),
        });
        let keep = |e: &ledger::Entry| {
            e.experience
                .as_deref()
                .is_some_and(|id| part_of.get(id) == Some(&part))
                && options.corrections.admits(e)
        };
        match attempts_set_filtered(ctx, &state, name, &keep) {
            Ok(set) => {
                let built = match build_dataset(ctx, &[set], settings) {
                    Ok(built) => built,
                    // A split whose attempts all failed the gate has nothing to train on.
                    Err(e) => {
                        entry["records"] = json!(0);
                        entry["note"] = json!(format!("{e:#}"));
                        summary.insert(name.to_string(), entry);
                        continue;
                    }
                };
                let dir = out.join(name);
                std::fs::create_dir_all(&dir)?;
                std::fs::copy(&built.path, dir.join("dataset.jsonl"))?;
                let manifest = splinter_sdk::data::manifest_path(&built.path);
                if manifest.exists() {
                    std::fs::copy(&manifest, dir.join("dataset.manifest.json"))?;
                }
                entry["records"] = json!(built.records);
                entry["dataset"] = json!(built.dataset.to_string());
                entry["excluded"] = json!(built
                    .excluded
                    .iter()
                    .map(|(k, v)| (format!("{k:?}"), *v))
                    .collect::<BTreeMap<_, _>>());
            }
            Err(_) => entry["records"] = json!(0),
        }
        summary.insert(name.to_string(), entry);
    }

    // One readable file per attempt; one a split left out is kept under `excluded`.
    let store = ctx.experiences();
    for (n, e) in entries.iter().enumerate() {
        let Some(hex) = e.experience.as_deref() else {
            continue;
        };
        if !chain.iter().any(|c| c.sha == e.commit) {
            continue;
        }
        let split_dir = part_of.get(hex).map_or("excluded", |p| split_name(*p));
        let id = ExperienceId(
            Digest::from_content_hex(hex)
                .with_context(|| format!("{hex:?} is not an experience id"))?,
        );
        let experience = store.get(&id)?;
        let notes = store.annotations(&id)?.annotations;
        let trace = Trace::of(&experience.trajectory);
        let calls: Vec<_> = trace
            .calls
            .iter()
            .map(|c| json!({"step": c.step, "tool": c.tool, "args": c.args, "result": c.result}))
            .collect();
        let dir = out.join("traces").join(split_dir);
        std::fs::create_dir_all(&dir)?;
        let style = e.style.map_or("-", |s| s.name());
        let file = dir.join(format!("{n:05}-{}-{style}.json", &e.commit[..10]));
        std::fs::write(
            file,
            serde_json::to_vec_pretty(&json!({
                "commit": e.commit, "style": style, "outcome": e.outcome, "classes": e.classes,
                "detail": e.detail, "changed_lines": e.changed_lines, "files": e.files,
                "tool_calls": e.tool_calls, "seconds": e.seconds, "checks": e.checks,
                "plan": experience.instruction, "final_output": experience.final_output,
                "verdict": notes, "calls": calls, "said": trace.said,
                "trajectory": experience.trajectory,
            }))?,
        )?;
    }

    let summary = json!({
        "total_commits": total,
        "train_percent": options.train_percent,
        "eval_percent": options.eval_percent,
        "split_digest": split.digest()?.to_string(),
        "excluded": split.excluded,
        "overlaps": overlap_report,
        "splits": summary,
    });
    std::fs::create_dir_all(out)?;
    std::fs::write(
        out.join("splits.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    Ok(summary)
}
