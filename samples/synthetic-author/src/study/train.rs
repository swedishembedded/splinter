// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the measurement of fine-tuned language
// models for its clients. If your team needs expertise in training runs whose
// every setting is recorded and repeatable, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The training stage: one adapter for each training seed of each author, and
//! for the ablations the records they ask for, under the protocol's settings.
//!
//! Nothing is trained before the author's corpus stage has frozen the
//! examination, and the examination is drawn again from the corpus first: an
//! adapter is never trained on a corpus that is not the one its examination
//! was frozen for. A run that has a manifest is checked, not redone; the
//! stage refuses to retrain it under other settings.

use std::path::Path;

use anyhow::Context as _;

use super::arm_key::ArmKey;
use super::layout::{RunManifest, Study};
use crate::binding::{base_digest, file_digest, items_digest};
use crate::dataset::{path as dataset_path, Briefings};
use crate::lab::exam_items_of;
use crate::manifest::code_revision;
use crate::training::{dataset_digest, fine_tune, Job, Settings, Trained};

/// What trains an adapter. The stage decides what to train and records what
/// came of it; this is the part that needs a device.
pub trait Trainer {
    /// Trains the adapter `job` describes and returns what came of it.
    ///
    /// # Errors
    /// The training fails.
    fn train(&self, job: &Job<'_>) -> anyhow::Result<Trained>;
}

/// Trains with brain, as the `train` command does.
pub struct Brain;

impl Trainer for Brain {
    fn train(&self, job: &Job<'_>) -> anyhow::Result<Trained> {
        fine_tune(job)
    }
}

/// What a call of the stage did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Trainings {
    /// Adapters trained by this call.
    pub trained: usize,
    /// Adapters that already had a manifest and were left alone.
    pub already: usize,
}

/// The settings the protocol trains with, for one adapter.
fn settings_for(study: &Study, seed: u64, dataset: &Path) -> anyhow::Result<Settings> {
    let t = &study.protocol().training;
    Ok(Settings {
        seed,
        steps: t.steps,
        rank: t.rank,
        alpha: t.alpha as f32,
        learning_rate: Some(t.learning_rate as f32),
        records_per_step: t.records_per_step,
        bf16: t.bf16,
        dataset_digest: dataset_digest(dataset)
            .with_context(|| format!("hashing {}", dataset.display()))?,
        briefings: Briefings::Both,
    })
}

/// Trains the adapters of the author with seed `author_seed` that have no
/// manifest yet: for every adapter arm the protocol asks, every training seed
/// it is trained at.
///
/// # Errors
/// The author's corpus stage has not run; the corpus or its records differ
/// from what the examination was frozen for; a run's manifest contradicts the
/// protocol, the base or the settings; or the training fails (the runs that
/// finished before it keep their manifests, and running the stage again
/// continues with the rest).
pub fn run(
    study: &Study,
    author_seed: u64,
    base: &Path,
    trainer: &dyn Trainer,
) -> anyhow::Result<Trainings> {
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
        "author {author_seed}: the examination drawn from the corpus is {now} and the one frozen is {}: \
         no adapter is trained on a corpus that is not the one its examination was frozen for",
        frozen.exam.digest
    );
    let base_digest = base_digest(base).with_context(|| format!("hashing {}", base.display()))?;
    let mut done = Trainings::default();
    for key in protocol.arm_keys()?.into_iter().filter(|k| k.is_adapter()) {
        for training_seed in protocol.seeds_for(plan, key) {
            let record = frozen
                .datasets
                .iter()
                .find(|d| d.briefings == key.briefings())
                .with_context(|| {
                    format!("author {author_seed} has no records for {}", key.name())
                })?;
            let dataset = dataset_path(&cache, key.briefings());
            let mut settings = settings_for(study, training_seed, &dataset)?;
            settings.briefings = key.briefings();
            anyhow::ensure!(
                file_digest(&dataset)? == record.digest,
                "{}: the records were changed after the corpus stage froze the examination",
                dataset.display()
            );
            if let Some(kept) = study.run_manifest(author_seed, key.briefings(), training_seed)? {
                let differs: Vec<&str> = [
                    (kept.base_digest != base_digest, "base"),
                    (kept.settings != settings, "settings"),
                    (kept.exam_digest != frozen.exam.digest, "examination"),
                ]
                .into_iter()
                .filter_map(|(differs, name)| differs.then_some(name))
                .collect();
                anyhow::ensure!(
                    differs.is_empty(),
                    "{} was trained with another {}; a trained adapter is never retrained under other \
                     settings, use another study directory",
                    key.name(),
                    differs.join(" and "),
                );
                done.already += 1;
                continue;
            }
            eprintln!(
                "training {} of author {author_seed} with seed {training_seed}",
                key.name()
            );
            let attempt = study.run_dir(author_seed, key.briefings(), training_seed);
            let trained = trainer.train(&Job {
                base,
                dataset: &dataset,
                attempt: &attempt,
                settings: &settings,
            })?;
            let adapter = trained
                .adapter
                .strip_prefix(study.root())
                .unwrap_or(&trained.adapter)
                .to_path_buf();
            study.write_run_manifest(
                key.briefings(),
                &RunManifest {
                    protocol_digest: study.digest().to_string(),
                    author_seed,
                    training_seed,
                    base_digest: base_digest.clone(),
                    exam_digest: frozen.exam.digest.clone(),
                    adapter_digest: file_digest(&trained.adapter)?,
                    adapter,
                    settings,
                    records: trained.records,
                    held_out_before: trained.held_out_before,
                    held_out_after: trained.held_out_after,
                    seconds: trained.seconds,
                    code_revision: code_revision().to_string(),
                },
            )?;
            done.trained += 1;
        }
    }
    Ok(done)
}

/// The training seeds and arms of an author that have a finished run, for
/// the stages that follow.
///
/// # Errors
/// A run's manifest cannot be read or was written under another protocol.
pub fn finished(
    study: &Study,
    author_seed: u64,
    key: ArmKey,
) -> anyhow::Result<Vec<(u64, RunManifest)>> {
    let plan = study.author_plan(author_seed)?;
    let mut runs = Vec::new();
    for training_seed in study.protocol().seeds_for(plan, key) {
        if let Some(run) = study.run_manifest(author_seed, key.briefings(), training_seed)? {
            runs.push((training_seed, run));
        }
    }
    Ok(runs)
}
