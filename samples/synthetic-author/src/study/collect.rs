// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the measurement of fine-tuned language
// models for its clients. If your team needs expertise in sizing and
// analysing a paired evaluation with honest statistics, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Reads what the stages wrote for the report: the study's manifests and the
//! outcomes they bind, and nothing else.
//!
//! Every outcome file is checked against the manifest of its slot and every
//! manifest against the protocol and the examination frozen for the author,
//! so a report can only be made of outcomes that answer the frozen items under
//! the model their manifest names. What is simply not there (an arm not yet
//! examined, an author whose corpus is not written) is reported as missing;
//! what contradicts a manifest is an error.

use std::path::Path;

use anyhow::Context as _;

use super::analysis::{AuthorInput, Measured, MeasuredSampled, Unmeasured};
use super::layout::Study;
use crate::arms::read_outcomes;
use crate::binding::{items_digest, recorded, verify_items, verify_outcomes};
use crate::lab::{exam_items_of, outcomes_path};
use crate::sampled::{key as sampled_key, read_sampled_file, sampled_path, verify_sampled};

/// What was read for the report.
pub struct Collected {
    /// The authors with a corpus, each with the arms that were measured.
    pub authors: Vec<AuthorInput>,
    /// What the protocol asks for and is missing.
    pub unmeasured: Vec<Unmeasured>,
}

/// Reads the study in `study`.
///
/// # Errors
/// A manifest was written under another protocol, the examination drawn from
/// an author's corpus is not the one frozen, or outcomes contradict their
/// manifest, the frozen items, or the adapter their run trained.
pub fn collect(study: &Study) -> anyhow::Result<Collected> {
    let mut authors = Vec::new();
    let mut unmeasured = Vec::new();
    for plan in &study.protocol().authors {
        match study.author_manifest(plan.seed)? {
            None => unmeasured.push(Unmeasured {
                author: Some(plan.seed),
                what: "corpus".into(),
                why: "no study.json: `study corpus` has not finished for this author".into(),
            }),
            Some(_) => {
                let (input, missing) = collect_author(study, plan.seed)
                    .with_context(|| format!("author {}", plan.seed))?;
                authors.push(input);
                unmeasured.extend(missing);
            }
        }
    }
    Ok(Collected {
        authors,
        unmeasured,
    })
}

fn collect_author(study: &Study, seed: u64) -> anyhow::Result<(AuthorInput, Vec<Unmeasured>)> {
    let protocol = study.protocol();
    let plan = study.author_plan(seed)?;
    let manifest = study.require_author_manifest(seed)?;
    let cache = study.cache(seed)?;
    let author = crate::manifest::author(cache.dir(), Some(seed))?;
    let items = exam_items_of(
        &cache,
        &author.persona,
        author.seed,
        protocol.corpus.articles,
        protocol.exam.per_cell,
        protocol.exam.version,
    )?;
    let digest = items_digest(&items).to_string();
    anyhow::ensure!(
        digest == manifest.exam.digest,
        "the examination drawn from the corpus is {digest} and the one frozen is {}",
        manifest.exam.digest
    );
    let mut input = AuthorInput {
        seed,
        name: manifest.author_name.clone(),
        corpus: manifest.corpus.clone(),
        survivors: manifest.survivors.clone(),
        items,
        persona: author.persona,
        measured: Vec::new(),
        sampled: Vec::new(),
    };
    let mut missing = Vec::new();
    let mut lack = |slot: &str, arm: &str, why: String| {
        missing.push(Unmeasured {
            author: Some(seed),
            what: format!("{slot}/{arm}"),
            why,
        });
    };
    for key in protocol.arm_keys()? {
        let seeds: Vec<Option<u64>> = if key.is_adapter() {
            protocol
                .seeds_for(plan, key)
                .into_iter()
                .map(Some)
                .collect()
        } else {
            vec![None]
        };
        for training_seed in seeds {
            let slot = key.slot(training_seed.unwrap_or_default());
            let slotted = study.cache(seed)?.with_exam_slot(&slot);
            let exam_dir = slotted.exam_dir().to_path_buf();
            let arm = key.arm();
            let adapter_digest = match (key.is_adapter(), training_seed) {
                (true, Some(s)) => study
                    .run_manifest(seed, key.briefings(), s)?
                    .map(|run| (run.adapter_digest, run.base_digest)),
                _ => None,
            };
            let path = outcomes_path(&slotted, arm);
            if path.exists() {
                let outcomes = read_outcomes(&path)?;
                let record = recorded(&exam_dir, arm.name())?.with_context(|| {
                    format!(
                        "{}: outcomes without a manifest for arm {}",
                        path.display(),
                        arm.name()
                    )
                })?;
                verify_outcomes(&record, &outcomes, &path)?;
                verify_items(&record, &input.items, &outcomes, &path)?;
                check_bound(
                    &record.items_digest,
                    &manifest.exam.digest,
                    adapter_digest.as_ref(),
                    &record.model,
                    &path,
                )?;
                match Measured::covering(&key.name(), training_seed, outcomes, &input.items) {
                    Some(measured) => input.measured.push(measured),
                    None => lack(&slot, &key.name(), "answered in part".into()),
                }
            } else {
                lack(&slot, &key.name(), "no outcomes".into());
            }
            let sampled = sampled_path(&slotted, arm.name());
            if sampled.exists() {
                let outcomes = read_sampled_file(&sampled)?;
                let record = recorded(&exam_dir, &sampled_key(arm.name()))?.with_context(|| {
                    format!("{}: sampled outcomes without a manifest", sampled.display())
                })?;
                verify_sampled(&record, &input.items, &outcomes, &sampled)?;
                check_bound(
                    &record.items_digest,
                    &manifest.exam.digest,
                    adapter_digest.as_ref(),
                    &record.model,
                    &sampled,
                )?;
                if let Some(measured) =
                    MeasuredSampled::covering(&key.name(), training_seed, outcomes, &input.items)
                {
                    input.sampled.push(measured);
                }
            }
        }
    }
    Ok((input, missing))
}

/// Checks that an arm's record is for the frozen items and, for an adapter,
/// for the adapter and base its training run recorded.
fn check_bound(
    items_digest: &str,
    frozen: &str,
    run: Option<&(String, String)>,
    model: &crate::binding::ModelIdentity,
    source: &Path,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        items_digest == frozen,
        "{}: asked items {items_digest} and the examination frozen for the author is {frozen}",
        source.display()
    );
    if let Some((adapter, base)) = run {
        anyhow::ensure!(
            model.adapter_digest.as_deref() == Some(adapter.as_str()) && &model.base_digest == base,
            "{}: answered by a model other than the adapter its training run recorded",
            source.display()
        );
    }
    Ok(())
}
