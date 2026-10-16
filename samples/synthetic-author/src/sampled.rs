// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled validation of learning systems
// against ground truth that cannot have been seen in pretraining, for its
// clients. If your team needs expertise in measuring how stable a model's
// answers are under sampling, you can procure our services by sending an
// email to info@swedishembedded.com.

//! The examination under sampled decoding.
//!
//! Greedy decoding gives an item one answer per set of weights. To see how far
//! a ranking of arms depends on that, an arm can also be asked each item K
//! times at a temperature. Every draw has its own seed, derived from one run
//! seed and the place of the item, and is recorded with its outcome, so a
//! draw can be repeated from the run seed in the manifest. The outcomes sit
//! beside the greedy ones under their own record.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use splinter_sdk::agent::sven::model::ModelProvider;
use splinter_sdk::store::jsonl;
use splinter_sdk::vocabulary::rng::SplitMix64;

use crate::arms::answer;
use crate::binding::{bind, item_digest, recorded, verify_items, ArmRecord};
use crate::cache::Cache;
use crate::exam::Item;
use crate::lab::{arm_record, system_of, Arm, ExamRun};
use crate::persona::Persona;

/// How many times each item is asked when the caller does not say.
pub const DEFAULT_DRAWS: usize = 8;

/// Keeps the seeds of the draws apart from every other stream derived from a
/// seed the caller passes.
const DRAW_SALT: u64 = 0x5341_4d50_4c45_0004;

/// How an arm is asked under sampled decoding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampledDecoding {
    /// The softmax temperature of every draw.
    pub temperature: f32,
    /// How many times each item is asked.
    pub draws: usize,
    /// The seed every draw's own seed is derived from.
    pub run_seed: u64,
}

impl SampledDecoding {
    /// The decoding as a manifest names it.
    #[must_use]
    pub fn name(&self) -> String {
        format!("sample:{}", self.temperature)
    }
}

/// The seed of draw `draw` (from zero) of the item at `place` in the
/// examination: a pure function of the run seed and the place, so a draw can
/// be repeated from the manifest and no two draws share a stream.
#[must_use]
pub fn draw_seed(run_seed: u64, place: usize, draw: usize) -> u64 {
    SplitMix64::new(run_seed ^ DRAW_SALT ^ ((place as u64) << 24) ^ draw as u64).next_u64()
}

/// One draw of one item.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Draw {
    /// The seed the model decoded this draw with.
    pub seed: u64,
    /// Whether the call named the option the beliefs determine.
    pub correct: bool,
    /// The tool called and its arguments, if a call was made.
    pub call: Option<Value>,
}

/// What a model did with one item over all its draws.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SampledOutcome {
    /// The item.
    pub id: String,
    /// Digest of the item as it was asked.
    pub item: String,
    /// Digest of the model that answered.
    pub model: String,
    /// The draws, in order.
    pub draws: Vec<Draw>,
}

impl SampledOutcome {
    /// The share of the draws that were right; `None` for an item with no draw.
    #[must_use]
    pub fn share_right(&self) -> Option<f64> {
        (!self.draws.is_empty()).then(|| {
            self.draws.iter().filter(|d| d.correct).count() as f64 / self.draws.len() as f64
        })
    }
}

/// What a caller asks of a sampled run; what it leaves out is read from the
/// record of the run or decided by [`resolve`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Requested {
    /// The softmax temperature of every draw.
    pub temperature: f32,
    /// How many times each item is asked.
    pub draws: Option<usize>,
    /// The seed every draw's own seed is derived from.
    pub run_seed: Option<u64>,
}

/// The sampled decoding for `arm` of the corpus in `cache`: what the arm's
/// record already says when it was begun (a seed or a draw count passed now
/// must agree, which the binding enforces when the run starts), else what is
/// requested, and for the seed a fresh one printed so passing it back repeats
/// the draws.
///
/// # Errors
/// The record cannot be read.
pub fn resolve(cache: &Cache, arm: Arm, asked: Requested) -> anyhow::Result<SampledDecoding> {
    let kept = recorded(&cache.dir().join("exam"), &key(arm.name()))?;
    let kept_seed = kept.as_ref().and_then(|r| r.model.decoding_seed);
    let kept_draws = kept.as_ref().and_then(|r| r.draws);
    Ok(SampledDecoding {
        temperature: asked.temperature,
        draws: asked.draws.or(kept_draws).unwrap_or(DEFAULT_DRAWS),
        run_seed: asked
            .run_seed
            .or(kept_seed)
            .unwrap_or_else(|| splinter_sdk::seed::resolve("decoding-seed", None)),
    })
}

/// The key under which the sampled run of `arm` is recorded and named on disk.
#[must_use]
pub fn key(arm: &str) -> String {
    format!("{arm}-sampled")
}

/// Where the sampled outcomes of `arm` are kept.
#[must_use]
pub fn sampled_path(cache: &Cache, arm: &str) -> PathBuf {
    cache.dir().join("exam").join(format!("{}.jsonl", key(arm)))
}

/// The sampled outcomes in `path`, each item once.
///
/// # Errors
/// The file cannot be read or a line is not an outcome.
pub fn read_sampled_file(path: &Path) -> anyhow::Result<Vec<SampledOutcome>> {
    Ok(jsonl::read_unique(path, |o: &SampledOutcome| o.id.clone())?)
}

/// The sampled outcomes kept for `arm`.
///
/// # Errors
/// As [`read_sampled_file`].
pub fn read_sampled(cache: &Cache, arm: Arm) -> anyhow::Result<Vec<SampledOutcome>> {
    read_sampled_file(&sampled_path(cache, arm.name()))
}

/// The record of the sampled run of `arm`: what the greedy run's record would
/// be, with the decoding and the number of draws of `how`.
///
/// # Errors
/// As [`arm_record`].
pub fn sampled_record(
    cache: &Cache,
    persona: &Persona,
    arm: Arm,
    items: &[Item],
    run: &ExamRun,
    how: &SampledDecoding,
) -> anyhow::Result<ArmRecord> {
    anyhow::ensure!(how.draws > 0, "a sampled run asks each item at least once");
    let mut record = arm_record(cache, persona, arm, items, run)?;
    record.arm = key(arm.name());
    record.model.decoding = how.name();
    record.model.decoding_seed = Some(how.run_seed);
    record.draws = Some(how.draws);
    Ok(record)
}

/// Puts the examination to `arm` as `how` says: every item `how.draws` times,
/// each draw by the model `provider_for` makes for the seed of that draw,
/// appending an outcome to the arm's sampled file as each item is done. Items
/// already in it are not asked again. Returns every outcome there is when it
/// ends, in order.
///
/// # Errors
/// The outcomes were made under another record (the differing fields are
/// named), the model failed, or the outcomes cannot be written.
pub fn run_arm_sampled(
    cache: &Cache,
    persona: &Persona,
    arm: Arm,
    provider_for: &dyn Fn(u64) -> Arc<dyn ModelProvider>,
    items: &[Item],
    run: &ExamRun,
    how: &SampledDecoding,
) -> anyhow::Result<Vec<SampledOutcome>> {
    let path = sampled_path(cache, arm.name());
    let dir = path.parent().context("a directory")?;
    std::fs::create_dir_all(dir)?;
    let record = sampled_record(cache, persona, arm, items, run, how)?;
    bind(dir, &path, &record, run.adopt)?;
    let model = record.model.digest().to_string();
    let system = system_of(cache, persona, arm)?;
    let mut done = read_sampled_file(&path)?;
    let runtime = tokio::runtime::Runtime::new()?;
    let log = jsonl::Log::open(&path)?;
    for (place, item) in items.iter().enumerate() {
        if done.iter().any(|o| o.id == item.id) {
            continue;
        }
        let mut draws = Vec::with_capacity(how.draws);
        for draw in 0..how.draws {
            let seed = draw_seed(how.run_seed, place, draw);
            let provider = provider_for(seed);
            let answered =
                runtime.block_on(answer(&provider, &system(item), item, run.max_tokens))?;
            draws.push(Draw {
                seed,
                correct: answered.correct,
                call: answered.call,
            });
        }
        let outcome = SampledOutcome {
            id: item.id.clone(),
            item: item_digest(item).to_string(),
            model: model.clone(),
            draws,
        };
        log.append(&outcome)?;
        done.push(outcome);
    }
    Ok(done)
}

/// Checks that `outcomes`, read from `source`, were made by the model `record`
/// names for the items `items` now are.
///
/// # Errors
/// The items differ from the record's, an outcome names another model, or an
/// outcome answers another item than its id names now.
pub fn verify_sampled(
    record: &ArmRecord,
    items: &[Item],
    outcomes: &[SampledOutcome],
    source: &Path,
) -> anyhow::Result<()> {
    verify_items(record, items, &[], source)?;
    let model = record.model.digest();
    for outcome in outcomes {
        anyhow::ensure!(
            outcome.model == model.as_str(),
            "{}: outcome {} was made by model {:?} and the manifest of arm {} says {model}",
            source.display(),
            outcome.id,
            outcome.model,
            record.arm,
        );
        if let Some(item) = items.iter().find(|i| i.id == outcome.id) {
            let digest = item_digest(item);
            anyhow::ensure!(
                outcome.item == digest.as_str(),
                "{}: outcome {} answers item {:?} and the item that id names now is {digest}",
                source.display(),
                outcome.id,
                outcome.item,
            );
        }
    }
    Ok(())
}
