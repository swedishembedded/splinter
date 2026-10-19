// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the measurement of fine-tuned language
// models for its clients. If your team needs expertise in sizing and
// analysing a paired evaluation with honest statistics, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The examination stage: every arm of the protocol put to its model on the
//! author's frozen examination.
//!
//! The arms that have no adapter share one loaded base and one directory of
//! outcomes (`exam/shared/`); each adapter is examined in a directory of its
//! own (`exam/tuned-s<seed>/`), so no adapter's outcomes can be mistaken for
//! another's. Every directory has the examination manifest the existing `exam`
//! command writes, binding outcomes to the items and the model.

use std::path::Path;
use std::sync::Arc;

use anyhow::Context as _;
use splinter_sdk::agent::sven::model::ModelProvider;
use splinter_sdk::model::answer::{Answerer, Decoding};

use super::arm_key::{ArmKey, SHARED_SLOT};
use super::layout::Study;
use super::train::finished;
use crate::binding::{file_digest, items_digest, Base, ModelIdentity};
use crate::exam_cmd::CONTEXT_TOKENS;
use crate::lab::{exam_items_of, run_arm, ExamRun};
use crate::sampled::{self, run_arm_sampled, Requested};

/// A loaded model that answers the examination.
pub trait Examinee {
    /// Answers greedily.
    fn greedy(&self) -> Arc<dyn ModelProvider>;
    /// Answers by sampling at `temperature` with `seed`.
    ///
    /// # Errors
    /// The sampled model cannot be set up.
    fn sampled(&self, temperature: f32, seed: u64) -> anyhow::Result<Arc<dyn ModelProvider>>;
    /// How many requests it answers at once: an in-process model answers
    /// one, a batching server as many as it has decode slots.
    fn slots(&self) -> usize {
        1
    }
}

/// Loads the model the examination is put to, with an adapter or without.
pub trait Loader {
    /// The base model with `adapter` attached, if one is given.
    ///
    /// # Errors
    /// The model cannot be loaded.
    fn load(&self, adapter: Option<&Path>) -> anyhow::Result<Box<dyn Examinee>>;
}

/// Loads the base checkpoint with brain.
pub struct BrainLoader<'a> {
    /// The checkpoint directory of the base model.
    pub base: &'a Path,
}

struct Loaded(Answerer);

impl Examinee for Loaded {
    fn greedy(&self) -> Arc<dyn ModelProvider> {
        self.0.provider()
    }

    fn sampled(&self, temperature: f32, seed: u64) -> anyhow::Result<Arc<dyn ModelProvider>> {
        Ok(self.0.provider_with(Decoding::Sampled {
            temperature,
            seed: Some(seed),
        }))
    }
}

impl Loader for BrainLoader<'_> {
    fn load(&self, adapter: Option<&Path>) -> anyhow::Result<Box<dyn Examinee>> {
        Ok(Box::new(Loaded(Answerer::load_with(
            self.base,
            adapter,
            Some(CONTEXT_TOKENS),
            "study",
            Decoding::Greedy,
        )?)))
    }
}

/// What a call of the stage asked.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Examined {
    /// The slot and arm of each examination that was run or resumed, with the
    /// number of items it has outcomes for.
    pub arms: Vec<(String, String, usize)>,
}

/// Puts the frozen examination of the author with seed `author_seed` to every
/// arm the protocol asks, and with `sampled` also to the sampled decoding the
/// protocol names. Items already answered are not asked again.
///
/// # Errors
/// The author's corpus stage or an adapter's training has not finished; the
/// examination drawn from the corpus is not the one frozen; an adapter file
/// is not the one trained; the outcomes of an arm were made under another
/// manifest; or the model fails.
pub fn run(
    study: &Study,
    author_seed: u64,
    base: &Base,
    loader: &dyn Loader,
    sampled: bool,
) -> anyhow::Result<Examined> {
    run_arms(study, author_seed, base, loader, sampled, &[])
}

/// [`run`], restricted to the arms named in `only` (every arm the protocol
/// asks when it is empty); a model whose arms are all left out is not loaded.
///
/// # Errors
/// As [`run`]; also a name in `only` that is not an arm the protocol asks.
pub fn run_arms(
    study: &Study,
    author_seed: u64,
    base: &Base,
    loader: &dyn Loader,
    sampled: bool,
    only: &[String],
) -> anyhow::Result<Examined> {
    let protocol = study.protocol();
    let plan = study.author_plan(author_seed)?;
    let frozen = study.require_author_manifest(author_seed)?;
    let cache = study.cache(author_seed)?;
    let author = crate::manifest::author(cache.dir(), Some(author_seed))?;
    let items = exam_items_of(
        &cache,
        &author.persona,
        author.seed,
        protocol.corpus.articles,
        protocol.exam.per_cell,
        protocol.exam.version,
    )?;
    let now = items_digest(&items).to_string();
    anyhow::ensure!(
        now == frozen.exam.digest,
        "author {author_seed}: the examination drawn from the corpus is {now} and the one frozen is {}",
        frozen.exam.digest
    );
    let base_digest = base.digest().to_string();
    let secondary = match (sampled, protocol.decoding.secondary) {
        (true, Some(plan)) => Some(plan),
        (true, None) => anyhow::bail!("the protocol names no sampled decoding"),
        (false, _) => None,
    };

    // Everything an adapter needs exists before any model is loaded.
    let mut keys = protocol.arm_keys()?;
    if let Some(unknown) = only
        .iter()
        .find(|name| !keys.iter().any(|k| k.name() == name.as_str()))
    {
        anyhow::bail!("{unknown:?} is not an arm the protocol asks");
    }
    if !only.is_empty() {
        keys.retain(|k| only.iter().any(|name| name.as_str() == k.name()));
    }
    let mut adapters = Vec::new();
    for key in keys.iter().copied().filter(|k| k.is_adapter()) {
        let wanted = protocol.seeds_for(plan, key);
        let runs = finished(study, author_seed, key)?;
        for seed in wanted {
            let (_, run) = runs.iter().find(|(s, _)| *s == seed).with_context(|| {
                format!(
                    "author {author_seed}: {} with training seed {seed} is not trained: run `study train` first",
                    key.name()
                )
            })?;
            let file = study.root().join(&run.adapter);
            anyhow::ensure!(
                file_digest(&file).with_context(|| format!("hashing {}", file.display()))?
                    == run.adapter_digest,
                "{} is not the adapter its manifest records (digest {})",
                file.display(),
                run.adapter_digest
            );
            anyhow::ensure!(
                run.base_digest == base_digest,
                "{} was trained on another base than {}",
                key.name(),
                base.path().display()
            );
            adapters.push((key, seed, file, run.adapter_digest.clone()));
        }
    }

    let mut examined = Examined::default();
    let mut ask =
        |key: ArmKey, slot: &str, adapter_digest: Option<String>, model: &dyn Examinee| {
            let slotted = study.cache(author_seed)?.with_exam_slot(slot);
            let run = ExamRun {
                articles: protocol.corpus.articles,
                per_cell: protocol.exam.per_cell,
                max_tokens: protocol.exam.max_tokens,
                model: ModelIdentity {
                    base_digest: base_digest.clone(),
                    adapter_digest,
                    decoding: "greedy".to_string(),
                    // Greedy decoding takes the likeliest token and draws nothing.
                    decoding_seed: None,
                },
                exam: protocol.exam.version,
                adopt: false,
                slots: model.slots(),
            };
            let outcomes = run_arm(
                &slotted,
                &author.persona,
                key.arm(),
                &model.greedy(),
                &items,
                &run,
            )?;
            examined
                .arms
                .push((slot.to_string(), key.name(), outcomes.len()));
            eprintln!(
                "author {author_seed} {slot} {}: {} items answered",
                key.name(),
                outcomes.len()
            );
            if let Some(plan) = secondary {
                let how = sampled::resolve(
                    &slotted,
                    key.arm(),
                    Requested {
                        temperature: plan.temperature as f32,
                        draws: Some(plan.draws),
                        run_seed: None,
                    },
                )?;
                let provider_for = |seed| model.sampled(how.temperature, seed);
                run_arm_sampled(
                    &slotted,
                    &author.persona,
                    key.arm(),
                    &provider_for,
                    &items,
                    &run,
                    &how,
                )?;
            }
            anyhow::Ok(())
        };

    let plain: Vec<ArmKey> = keys.iter().copied().filter(|k| !k.is_adapter()).collect();
    if !plain.is_empty() {
        let model = loader.load(None)?;
        for key in plain {
            ask(key, SHARED_SLOT, None, model.as_ref())?;
        }
    }
    for (key, seed, file, digest) in adapters {
        let model = loader.load(Some(&file))?;
        ask(key, &key.slot(seed), Some(digest), model.as_ref())?;
    }
    Ok(examined)
}
