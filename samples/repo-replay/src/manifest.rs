// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The manifest of a built dataset: what it was made from, what is in it, what
//! was left out and why, and whether the trainer will take it. Written beside
//! `splits.json` as `dataset-manifest.json`, so a reader (or a trainer's
//! preflight) can judge the dataset without opening a record.
//!
//! Everything here is computed from what the run stored: the ledger (counts,
//! reasons, scrub totals), the stored experiences (the documents' digests) and
//! the dataset files. No model is called and no GPU is used.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{Context as _, Result};
use serde_json::{json, Value};
use splinter_sdk::data::split::Part;
use splinter_sdk::datasets::examples_in;
use splinter_sdk::model::TokenCounter;
use splinter_sdk::train::trainable;
use splinter_sdk::vocabulary::build::BuildInfo;
use splinter_sdk::Context;

use crate::dataset::DatasetSettings;
use crate::ledger::{self, Entry, Outcome};
use crate::scrub::ScrubCounts;

/// File name of the manifest, beside `splits.json`.
pub const FILE: &str = "dataset-manifest.json";

/// What the manifest is composed from.
pub struct Sources<'a> {
    /// The run's state directory.
    pub state: &'a Path,
    /// Where `splits.json` and the datasets were written.
    pub out: &'a Path,
    /// What the ledger holds.
    pub entries: &'a [Entry],
    /// The split of each stored attempt, by experience id (absent: left out).
    pub part_of: &'a BTreeMap<String, Part>,
    /// The `splits.json` summary.
    pub summary: &'a Value,
    /// The build of the binary that made the dataset.
    pub build: &'a BuildInfo,
    /// The serving prompt and the length bound the dataset was built with.
    pub settings: &'a DatasetSettings,
}

/// Composes the manifest, writes it to [`FILE`] under `out` and returns it.
pub fn write(ctx: &Context, sources: &Sources<'_>) -> Result<Value> {
    let manifest = compose(ctx, sources)?;
    std::fs::write(
        sources.out.join(FILE),
        serde_json::to_vec_pretty(&manifest)?,
    )
    .with_context(|| format!("writing {}", sources.out.join(FILE).display()))?;
    Ok(manifest)
}

fn compose(ctx: &Context, s: &Sources<'_>) -> Result<Value> {
    let bias = ledger::bias(s.entries);
    let mut splits = serde_json::Map::new();
    for part in Part::ALL {
        let name = crate::splits::split_name(part);
        splits.insert(name.into(), split_section(ctx, s, part, name)?);
    }
    Ok(json!({
        "provenance": provenance(ctx, s)?,
        "split_digest": s.summary["split_digest"],
        "splits": splits,
        "left_out_of_every_split": s.summary["excluded"],
        "content_overlaps": s.summary["overlaps"].as_array().map_or(0, Vec::len),
        "skipped_commits": bias.skipped,
        "failed_attempts_by_reason": bias.reasons,
        "supervisor_messages_refused": bias.supervisor_refusals,
        "infrastructure_errors": bias.errors,
        "scrubbed": scrub_summary(s.entries),
    }))
}

/// Where the dataset comes from.
fn provenance(ctx: &Context, s: &Sources<'_>) -> Result<Value> {
    let runs: Vec<Value> = splinter_sdk::store::jsonl::read(&s.state.join("runs.jsonl"))?;
    let first = runs.first().cloned().unwrap_or(Value::Null);
    // The range the splits were cut over is the last run's.
    let last = runs.last().cloned().unwrap_or(Value::Null);
    let caller = &last["settings"]["caller"];
    // The documents the models were told, pinned by digest on each stored attempt.
    let mut documents = BTreeSet::new();
    let store = ctx.experiences();
    for id in s.part_of.keys().take(256) {
        let id = splinter_sdk::vocabulary::experience::ExperienceId(
            splinter_sdk::vocabulary::digest::Digest::from_content_hex(id)
                .with_context(|| format!("{id:?} is not an experience id"))?,
        );
        documents.extend(
            store
                .get(&id)?
                .provenance
                .prompt_digests
                .iter()
                .map(ToString::to_string),
        );
    }
    Ok(json!({
        "source": Path::new(first["source"].as_str().unwrap_or_default()).file_name()
            .map(|n| n.to_string_lossy().into_owned()),
        "from": last["from"],
        "to": last["to"],
        "runs": runs.len(),
        "git_version": first["git_version"],
        "revisions": s.build.revisions,
        "dirty": s.build.dirty,
        "binary_digest": s.build.binary_digest,
        "harness_documents": documents,
        "model": caller["model"],
        "seed": caller["seed"],
        "sampling": {
            "temperature": caller["temperature"], "top_p": caller["top_p"], "top_k": caller["top_k"],
            "thinking": caller["thinking"],
        },
        "serving_prompt_digest": splinter_sdk::vocabulary::digest::Digest::of(s.settings.serving_prompt.as_bytes()),
    }))
}

/// One split: its records by type and style, their token lengths, and whether
/// the trainer takes its dataset.
fn split_section(ctx: &Context, s: &Sources<'_>, part: Part, name: &str) -> Result<Value> {
    let in_part = |e: &&Entry| {
        e.experience
            .as_ref()
            .is_some_and(|id| s.part_of.get(id) == Some(&part))
    };
    let passed: Vec<&Entry> = s
        .entries
        .iter()
        .filter(in_part)
        .filter(|e| e.outcome == Outcome::Passed)
        .collect();
    let mut by_style: BTreeMap<&str, usize> = BTreeMap::new();
    for e in &passed {
        *by_style
            .entry(e.style.map_or("-", |style| style.name()))
            .or_default() += 1;
    }
    let summary = &s.summary["splits"][name];
    let mut section = json!({
        "attempts": s.entries.iter().filter(in_part).count(),
        "records": summary["records"],
        "first_try": passed.iter().filter(|e| e.corrections == 0).count(),
        "corrected": passed.iter().filter(|e| e.corrections > 0).count(),
        "by_style": by_style,
        "left_out_of_the_dataset": summary["excluded"],
    });
    let file = s.out.join(name).join("dataset.jsonl");
    if file.exists() {
        section["token_lengths"] = token_lengths(ctx, &file, s.settings.max_block)?;
        section["trainer"] = trainer_preflight(ctx, summary, &file, &s.settings.serving_prompt)?;
    }
    Ok(section)
}

/// Token lengths of the records of `file`: counted by the policy's tokenizer
/// when its directory holds one, else by the characters-per-token ratio the
/// builder falls back to; the counter used is named.
fn token_lengths(ctx: &Context, file: &Path, max_block: Option<usize>) -> Result<Value> {
    let counter = TokenCounter::for_model(&ctx.config().policy_base).ok();
    let mut lengths: Vec<usize> = std::fs::read_to_string(file)?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| match &counter {
            Some(c) => c.count(line),
            None => splinter_sdk::data::chars_as_tokens(line),
        })
        .collect();
    lengths.sort_unstable();
    let nearest_rank = |p: usize| {
        lengths
            .len()
            .checked_sub(1)
            .map(|last| lengths[(last * p).div_ceil(100)])
    };
    Ok(json!({
        "counter": if counter.is_some() { "tokenizer of the policy base" } else { "characters (no tokenizer at the policy base)" },
        "policy_base": ctx.config().policy_base.file_name().map(|n| n.to_string_lossy().into_owned()),
        "records": lengths.len(),
        "min": lengths.first(),
        "p50": nearest_rank(50),
        "p90": nearest_rank(90),
        "max": lengths.last(),
        "mean": (!lengths.is_empty()).then(|| lengths.iter().sum::<usize>() / lengths.len()),
        "max_block": max_block,
        "over_max_block": max_block.map(|m| lengths.iter().filter(|n| **n > m).count()),
    }))
}

/// What the trainer needs of the dataset, checked without a device: the
/// stored dataset is one brain trains and by which regime, the examples it
/// would read, and every record opens with the serving prompt.
fn trainer_preflight(ctx: &Context, summary: &Value, file: &Path, serving: &str) -> Result<Value> {
    let Some(id) = summary["dataset"].as_str() else {
        return Ok(json!({"accepts": false, "why": "no dataset was built"}));
    };
    let wrong_prompt = std::fs::read_to_string(file)?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter(|line| {
            let record: Value = serde_json::from_str(line).unwrap_or(Value::Null);
            record["messages"][0]["role"] != "system" || record["messages"][0]["content"] != serving
        })
        .count();
    let examples = examples_in(file).with_context(|| format!("reading {}", file.display()))?;
    Ok(match trainable(ctx, id) {
        Ok((stored, regime)) => json!({
            "accepts": wrong_prompt == 0,
            "dataset": stored.id.to_string(),
            "regime": format!("{regime:?}"),
            "examples": examples,
            "records_not_under_the_serving_prompt": wrong_prompt,
        }),
        Err(refusal) => json!({"accepts": false, "why": refusal.to_string()}),
    })
}

/// What the scrubber removed from the stored attempts, as counts.
fn scrub_summary(entries: &[Entry]) -> Value {
    let stored: Vec<&Entry> = entries.iter().filter(|e| e.experience.is_some()).collect();
    let total = stored
        .iter()
        .fold(ScrubCounts::default(), |sum, e| sum.plus(e.scrubbed));
    json!({
        "attempts": stored.len(),
        "trailers": total.trailers,
        "people": total.people,
        "secrets": total.secrets,
    })
}
