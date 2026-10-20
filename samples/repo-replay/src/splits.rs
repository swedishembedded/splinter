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
//! or with how far a resumable run has got. Each split's dataset is splinter's
//! own `sft-step` build over that split's attempts; the boundary commits and
//! counts are written to `splits.json`, and every attempt (plan, verdict,
//! reasons, the agent's calls and words, the full trajectory) is written as
//! one JSON file under `traces/<split>/`, positives and negatives alike, so a
//! reader can open any trace directly.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context as _, Result};
use serde_json::json;
use splinter_sdk::vocabulary::digest::Digest;
use splinter_sdk::vocabulary::experience::ExperienceId;
use splinter_sdk::Context;

use crate::dataset::{attempts_set_where, build_dataset, DatasetSettings};
use crate::git::Git;
use crate::ledger;
use crate::pipeline::layout;
use crate::trace::Trace;

/// The split names in time order.
pub const SPLITS: [&str; 3] = ["train", "eval", "test"];

/// The split of the commit at `position` of `total`, with `train` and `eval`
/// percent of the commits (the rest is test).
#[must_use]
pub fn split_of(position: usize, total: usize, train: usize, eval: usize) -> &'static str {
    let train_end = total * train / 100;
    let eval_end = total * (train + eval) / 100;
    if position < train_end {
        "train"
    } else if position < eval_end {
        "eval"
    } else {
        "test"
    }
}

/// Writes the three datasets, `splits.json` and the traces under `out`.
pub fn write_splits(
    ctx: &Context,
    state: &Path,
    out: &Path,
    settings: &DatasetSettings,
    train: usize,
    eval: usize,
) -> Result<serde_json::Value> {
    let state = state.canonicalize()?;
    let (from, to) = ledger::last_range(&state)?;
    let git = Git::at(layout(&state).mirror);
    let chain = git.first_parent_chain(from.as_deref(), to.as_deref())?;
    let total = chain.len();
    let split_by_commit: HashMap<&str, &'static str> = chain
        .iter()
        .enumerate()
        .map(|(i, c)| (c.sha.as_str(), split_of(i, total, train, eval)))
        .collect();

    let mut summary = serde_json::Map::new();
    for split in SPLITS {
        let commits: Vec<&str> = chain
            .iter()
            .map(|c| c.sha.as_str())
            .filter(|c| split_by_commit[c] == split)
            .collect();
        let mut entry = json!({
            "commits": commits.len(),
            "first": commits.first(),
            "last": commits.last(),
        });
        let keep = |c: &str| split_by_commit.get(c) == Some(&split);
        match attempts_set_where(ctx, &state, split, &keep) {
            Ok(set) => {
                let built = build_dataset(ctx, &[set], settings)?;
                let dir = out.join(split);
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
                    .collect::<std::collections::BTreeMap<_, _>>());
            }
            Err(_) => entry["records"] = json!(0),
        }
        summary.insert(split.to_string(), entry);
    }

    // One readable file per attempt.
    let store = ctx.experiences();
    for (n, e) in ledger::read(&state)?.iter().enumerate() {
        let (Some(hex), Some(split)) = (
            e.experience.as_deref(),
            split_by_commit.get(e.commit.as_str()),
        ) else {
            continue;
        };
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
        let dir = out.join("traces").join(split);
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

    let summary = json!({"total_commits": total, "train_percent": train, "eval_percent": eval, "splits": summary});
    std::fs::create_dir_all(out)?;
    std::fs::write(
        out.join("splits.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    Ok(summary)
}
