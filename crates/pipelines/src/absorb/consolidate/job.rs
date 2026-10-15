// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning that keeps accumulating
// capabilities generation after generation, for its clients. If your team
// needs expertise in consolidating what a model learned into its own
// weights, you can procure our services by sending an email to
// info@swedishembedded.com.

//! One consolidation: the weights, the gate and the release.
//!
//! A [`Job`] says which claims are taken into which base, and which release
//! the checkpoint replaces. `full` trains every weight of the base on the
//! records of the claims being taken in, beside a bounded sample of the
//! records of the claims the base already holds and the base's own answers;
//! `merge` folds the release in use's adapter into the base's weights and
//! trains nothing. The checkpoint is gated against the release in use
//! ([`super::gate`]) and released as a full checkpoint, the base of the
//! generation it opens; the claims it holds are recorded in the ledger as the
//! trained set of that release, which is what the next generation retains.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::Serialize;
use splinter_agent::CancelToken;
use splinter_core::claim::{ClaimId, TrainedClaims};
use splinter_core::dataset::DatasetId;
use splinter_core::digest::Digest;
use splinter_core::release::ReleaseId;
use splinter_core::rng::Seed;
use splinter_core::terms::{combine_stated, Distribution, Terms};
use splinter_core::training::{Regime, RehearsalSample, TrainingSummary};
use splinter_eval::claim_gate::ClaimGate;
use splinter_eval::gate::GateConfig;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::{io, OrchestratorError};
use splinter_orchestrator::generation::{Consolidation, ConsolidationMode, GenerationMark};
use splinter_orchestrator::releases::{
    Provenance, ReleaseGate, ReleaseManifest, ReleasedArtifact, StoredRelease, RELEASE_FORMAT,
};
use splinter_store::artifacts::ArtifactSpec;

use super::gate::{decide, Decision};
use crate::absorb::dataset::{assemble, Assembled};
use crate::absorb::generation::{records_of, replay_budget, retention_sample, Generation};
use crate::absorb::kit::{ClaimKit, Live};
use crate::absorb::sealed::SealedProbes;
use crate::learn::records_at_share;
use crate::rehearsal::{rehearse_on, RehearseRequest, Rehearsed};
use crate::release::arm_on;
use crate::train::{
    auto_records_per_step, plan_training, ModelCheckpoint, Rehearse, TrainRequest, Trainer, Tuning,
    DEFAULT_FULL_LEARNING_RATE, DEFAULT_LORA_RANK,
};

/// How a consolidation is carried out, whichever command asks for it.
#[derive(Clone, Debug, Serialize)]
pub struct Params {
    /// Train every weight, or fold the adapter in.
    pub mode: ConsolidationMode,
    /// The share of the training draws that are the base's own answers.
    pub rehearsal_share: f64,
    /// The seed the rehearsal's form tasks are built with; `None` when no
    /// rehearsal is asked for.
    pub rehearsal_seed: Option<Seed>,
    /// The share of the training draws that are records of the claims the
    /// base already holds.
    pub retention_replay_share: f64,
    /// The most passes over the records.
    pub epochs: u32,
    /// The step budget, in place of `epochs`.
    pub steps: Option<u32>,
    /// The learning rate ([`DEFAULT_FULL_LEARNING_RATE`] when `None`) and how
    /// the run is watched.
    pub tuning: Tuning,
    /// The serve check's bounds.
    pub gate: GateConfig,
    /// Stop at the checkpoint: do not release it.
    pub no_release: bool,
}

/// What is consolidated into what.
pub(crate) struct Job<'a> {
    /// The alias the release moves along.
    pub(crate) alias: &'a str,
    /// The release consolidated: the one the checkpoint is gated against and
    /// descends from.
    pub(crate) champion: &'a StoredRelease,
    /// The release the alias points at now, which the checkpoint replaces:
    /// the champion, unless an earlier release is consolidated.
    pub(crate) replaces: &'a ReleaseId,
    /// The generation whose base is trained from. The champion's own, but
    /// for a rebuild, which starts from an earlier base.
    pub(crate) from: &'a Generation,
    /// The claims being taken into the weights.
    pub(crate) train: &'a [Live],
    /// The claims the base already holds.
    pub(crate) retention: &'a [Live],
    /// Whether this retrains from an earlier base to take a claim out.
    pub(crate) rebuilt: bool,
    /// The dead claims a rebuild takes out of the weights.
    pub(crate) dead: &'a [ClaimId],
    /// The kits of every claim named.
    pub(crate) kits: &'a BTreeMap<ClaimId, ClaimKit>,
    /// The system prompt the policy answers under.
    pub(crate) system: Option<&'a str>,
    /// Probes no training record may contain.
    pub(crate) sealed: &'a SealedProbes,
    /// How.
    pub(crate) params: &'a Params,
    /// Stops the consolidation.
    pub(crate) cancel: CancelToken,
}

/// What a consolidation did.
#[derive(Clone, Debug, Serialize)]
pub struct Consolidated {
    /// Train every weight, or fold the adapter in.
    pub mode: ConsolidationMode,
    /// The release in use, which the checkpoint replaces.
    pub champion: ReleaseId,
    /// The generation of the release in use.
    pub from_epoch: u32,
    /// The generation the checkpoint opens.
    pub epoch: u32,
    /// The consolidation the checkpoint was trained from; `None` is the
    /// configured base.
    pub from_base: Option<ReleaseId>,
    /// The claims taken into the weights.
    pub closing: Vec<ClaimId>,
    /// The claims the base already held.
    pub retained: Vec<ClaimId>,
    /// Whether it trained again from an earlier base to take a claim out.
    pub rebuilt: bool,
    /// The claims the weights held that are no longer live, which a rebuild
    /// takes out.
    pub dead: Vec<ClaimId>,
    /// The records trained on (full).
    pub dataset: Option<Assembled>,
    /// Records of the retained claims replayed beside them (full).
    pub replayed_records: usize,
    /// The base's own answers rehearsed beside them (full).
    pub rehearsal: Option<Rehearsed>,
    /// Optimizer steps completed (full).
    pub steps_completed: u32,
    /// The training loss before the first step (full).
    pub initial_loss: Option<f32>,
    /// The last step's training loss (full).
    pub final_loss: Option<f32>,
    /// `sha256:<hex>` of the checkpoint's weights.
    pub weights_digest: Option<String>,
    /// The gate's decision with every count.
    pub gate: Option<ClaimGate>,
    /// The release written, when the gate passed.
    pub release: Option<ReleaseId>,
    /// Why the consolidation ended before a release, if it did.
    pub stopped: Option<String>,
}

impl Consolidated {
    fn begin(job: &Job<'_>) -> Self {
        let ids = |claims: &[Live]| claims.iter().map(|(id, _)| id.clone()).collect();
        Self {
            mode: job.params.mode,
            champion: job.champion.id.clone(),
            from_epoch: job.champion.epoch(),
            epoch: job.champion.epoch() + 1,
            from_base: job.from.base_release.clone(),
            closing: ids(job.train),
            retained: ids(job.retention),
            rebuilt: job.rebuilt,
            dead: job.dead.to_vec(),
            dataset: None,
            replayed_records: 0,
            rehearsal: None,
            steps_completed: 0,
            initial_loss: None,
            final_loss: None,
            weights_digest: None,
            gate: None,
            release: None,
            stopped: None,
        }
    }
}

/// The weights a consolidation made, with what they were trained from.
struct Made {
    checkpoint: ModelCheckpoint,
    datasets: Vec<DatasetId>,
    rehearsal: Option<RehearsalSample>,
    steps: u32,
    records: usize,
    /// The directory of working files, removed when the job ends.
    work: PathBuf,
}

/// Carries out `job`: the weights, the gate, and the release when the gate
/// passes and one is asked for.
pub(crate) fn run(
    ctx: &Context,
    job: &Job<'_>,
    trainer: &dyn Trainer,
) -> Result<Consolidated, OrchestratorError> {
    let mut report = Consolidated::begin(job);
    if job.train.is_empty() {
        report.stopped = Some("no live claim is taken into the weights".into());
        return Ok(report);
    }
    let made = match job.params.mode {
        ConsolidationMode::Full => full(ctx, job, trainer, &mut report)?,
        ConsolidationMode::Merge => merge(ctx, job, trainer)?,
    };
    let Some(made) = made else {
        return Ok(report);
    };
    let outcome = judge_and_release(ctx, job, &made, &mut report);
    // The checkpoint is kept as an artifact by now, or not wanted.
    let _ = std::fs::remove_dir_all(&made.work);
    outcome?;
    Ok(report)
}

fn full(
    ctx: &Context,
    job: &Job<'_>,
    trainer: &dyn Trainer,
    report: &mut Consolidated,
) -> Result<Option<Made>, OrchestratorError> {
    let params = job.params;
    let from_ref = arm_on(ctx.config(), &job.from.base, None);
    let own = records_of(job.kits, job.train);
    let held = records_of(job.kits, job.retention);
    let replayed = retention_sample(
        job.kits,
        job.retention,
        replay_budget(own, held, params.retention_replay_share),
    );
    report.replayed_records = replayed.len();
    let Some(assembled) = assemble(ctx, job.train, &replayed, job.kits, job.system, job.sealed)?
    else {
        report.stopped = Some("no claim has a record to train on".into());
        return Ok(None);
    };
    report.dataset = Some(assembled.clone());
    let rehearsal_seed = params
        .rehearsal_seed
        .filter(|_| params.rehearsal_share > 0.0);
    let rehearsal = if let Some(seed) = rehearsal_seed {
        let rehearsed = rehearse_on(
            ctx,
            &from_ref,
            &RehearseRequest {
                records: records_at_share(assembled.trained, params.rehearsal_share).max(1),
                seed,
                deadline: None,
                cancel: job.cancel.clone(),
            },
        )?;
        if let Some((position, leak)) = job.sealed.check_file(&rehearsed.dataset.path)? {
            return Err(OrchestratorError::Refused(format!(
                "rehearsal record {position} {leak}: the rehearsal is refused"
            )));
        }
        let id = rehearsed.dataset.dataset.to_string();
        report.rehearsal = Some(rehearsed);
        Some(Rehearse {
            dataset: id,
            share: params.rehearsal_share,
        })
    } else {
        None
    };
    let per_step = params
        .tuning
        .records_per_step
        .unwrap_or_else(|| auto_records_per_step(assembled.trained));
    let steps = params.steps.unwrap_or_else(|| {
        let records = u64::try_from(assembled.trained).unwrap_or(u64::MAX);
        let steps = (records * u64::from(params.epochs)).div_ceil(u64::from(per_step.max(1)));
        u32::try_from(steps.max(1)).unwrap_or(u32::MAX)
    });
    let planned = plan_training(
        ctx,
        &TrainRequest {
            datasets: vec![assembled.dataset.to_string()],
            rehearsal,
            from: from_ref,
            replay_fraction: 0.0,
            steps: Some(steps),
            rank: DEFAULT_LORA_RANK,
            beta: None,
            tuning: Tuning {
                records_per_step: Some(per_step),
                eval_every: Some(0),
                learning_rate: Some(
                    params
                        .tuning
                        .learning_rate
                        .unwrap_or(DEFAULT_FULL_LEARNING_RATE),
                ),
                ..params.tuning
            },
        },
    )?;
    let work = planned.plan.dir.clone();
    // A full fine-tune loads its own copy of the base: the device holds no
    // other while it trains.
    ctx.release_bases();
    let trained = trainer.train_full(ctx, &planned.plan, &job.cancel);
    let checkpoint = match trained {
        Ok(checkpoint) => checkpoint,
        Err(e) => {
            let _ = std::fs::remove_dir_all(&work);
            return Err(e);
        }
    };
    report.steps_completed = checkpoint.steps_completed;
    report.initial_loss = checkpoint.initial_loss;
    report.final_loss = checkpoint.final_loss;
    let rehearsal = match &planned.plan.rehearsal {
        Some(r) => {
            let kept = ctx.artifacts().put_file(
                &r.fit,
                &ArtifactSpec::new("rehearsal", "splinter-consolidate").with_extension(".jsonl"),
            )?;
            Some(RehearsalSample {
                dataset: r.dataset.id.clone(),
                share: r.share,
                trained: r.trained,
                monitored: r.monitored,
                digest: Some(kept.digest),
            })
        }
        None => None,
    };
    Ok(Some(Made {
        checkpoint,
        datasets: vec![assembled.dataset],
        rehearsal,
        steps,
        records: assembled.trained + assembled.held_out,
        work,
    }))
}

fn merge(
    ctx: &Context,
    job: &Job<'_>,
    trainer: &dyn Trainer,
) -> Result<Option<Made>, OrchestratorError> {
    let adapter = job.champion.adapter()?;
    if job.champion.base_release() != job.from.base_release {
        return Err(OrchestratorError::Refused(format!(
            "release {} sits on another base than the one named: its adapter cannot be folded \
             into it",
            job.champion.id
        )));
    }
    let work = ctx
        .root()
        .work()
        .join("consolidate")
        .join(splinter_store::new_id_with_prefix("merge"));
    std::fs::create_dir_all(&work).map_err(io(&work))?;
    ctx.release_bases();
    let checkpoint = match trainer.merge(ctx, &job.from.base, adapter, &work.join("checkpoint")) {
        Ok(checkpoint) => checkpoint,
        Err(e) => {
            let _ = std::fs::remove_dir_all(&work);
            return Err(e);
        }
    };
    Ok(Some(Made {
        checkpoint,
        datasets: Vec::new(),
        rehearsal: None,
        steps: 0,
        records: job.champion.manifest.training.records,
        work,
    }))
}

fn judge_and_release(
    ctx: &Context,
    job: &Job<'_>,
    made: &Made,
    report: &mut Consolidated,
) -> Result<(), OrchestratorError> {
    let claims: Vec<Live> = job.train.iter().chain(job.retention).cloned().collect();
    let champion_generation = Generation::of_release(ctx, job.champion)?;
    let mixed_in = [made.rehearsal.as_ref().and_then(|r| r.digest.as_ref())];
    let gate = decide(
        ctx,
        &Decision {
            checkpoint: &made.checkpoint.dir,
            weights_digest: &made.checkpoint.weights_digest,
            datasets: &made.datasets,
            mixed_in: &mixed_in,
            champion: job.champion,
            champion_base: &champion_generation.base,
            claims: &claims,
            kits: job.kits,
            config: &job.params.gate,
            cancel: &job.cancel,
        },
    )?;
    report.weights_digest = Some(made.checkpoint.weights_digest.clone());
    let passed = gate.passed;
    report.gate = Some(gate.clone());
    if !passed {
        report.stopped = Some(
            "the gate refused the consolidated checkpoint: the current release stays in use".into(),
        );
        return Ok(());
    }
    if job.params.no_release {
        report.stopped = Some("no release was asked for: the checkpoint is not kept".into());
        return Ok(());
    }
    report.release = Some(publish(ctx, job, made, gate, report)?);
    Ok(())
}

/// The architecture a checkpoint directory states, as brain names it.
fn architecture_of(checkpoint: &std::path::Path) -> String {
    std::fs::read_to_string(checkpoint.join("config.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .and_then(|config| config.get("model_type")?.as_str().map(str::to_string))
        .unwrap_or_else(|| "qwen3".to_string())
}

fn publish(
    ctx: &Context,
    job: &Job<'_>,
    made: &Made,
    gate: ClaimGate,
    report: &Consolidated,
) -> Result<ReleaseId, OrchestratorError> {
    let kept = ctx.artifacts().put_file(
        &made.checkpoint.weights,
        &ArtifactSpec::new("checkpoint", "brain-trainer")
            .with_extension(".safetensors")
            .with_sha256(),
    )?;
    let expected = Digest::parse(&made.checkpoint.weights_digest)
        .map_err(|e| OrchestratorError::Train(format!("brain's checkpoint digest: {e}")))?;
    if kept.sha256.as_ref() != Some(&expected) {
        return Err(OrchestratorError::Train(format!(
            "brain reported the checkpoint as {expected} but its file hashes to {}",
            kept.sha256
                .as_ref()
                .map_or_else(|| "nothing".to_string(), ToString::to_string)
        )));
    }
    // A merge trained nothing: it was made from what the release it folds in
    // was made from.
    let datasets = if made.datasets.is_empty() {
        job.champion.manifest.datasets.clone()
    } else {
        made.datasets.clone()
    };
    let candidate = splinter_store::new_id_with_prefix("consolidation");
    let digests: Vec<Digest> = datasets.iter().map(|d| d.0.clone()).collect();
    ctx.workspace().record_external_candidate(
        &candidate,
        &digests,
        "sft",
        Some(&job.champion.id.0),
    )?;
    let terms = terms_of(ctx, &datasets, job.champion)?;
    let snapshots = datasets
        .iter()
        .map(|id| ctx.datasets().get(id).map(|d| d.manifest.dataset))
        .collect::<Result<Vec<_>, _>>()?;
    let manifest = ReleaseManifest {
        format: RELEASE_FORMAT.into(),
        artifact: ReleasedArtifact::FullCheckpoint {
            architecture: architecture_of(&made.checkpoint.dir),
            checkpoint_digest: expected,
            checkpoint_artifact: kept.digest,
        },
        parent: Some(job.champion.id.clone()),
        candidate,
        datasets,
        replay: None,
        rehearsal: made.rehearsal.clone(),
        training: TrainingSummary {
            from: arm_on(ctx.config(), &job.from.base, None).to_string(),
            steps: made.steps,
            rank: 0,
            records: made.records,
            regime: Regime::Sft,
            base_score: None,
            tuned_score: None,
            preference: None,
            curve: None,
            record: serde_json::json!({
                "consolidation": report.mode,
                "steps_completed": made.checkpoint.steps_completed,
                "initial_loss": made.checkpoint.initial_loss,
                "final_loss": made.checkpoint.final_loss,
                "replayed_records": report.replayed_records,
            }),
            terms: Some(terms.clone()),
        },
        gate: ReleaseGate::Claims {
            report: Box::new(gate),
        },
        metrics: None,
        terms,
        distribution: Distribution::Restricted,
        provenance: Provenance {
            dataset_snapshots: snapshots,
            ..Provenance::default()
        },
        generation: Some(GenerationMark {
            epoch: report.epoch,
            base: job.from.base_release.clone(),
            consolidation: Some(Consolidation {
                mode: report.mode,
                from: job.champion.id.clone(),
                closing: report.closing.clone(),
                retained: report.retained.clone(),
                rebuilt: report.rebuilt,
            }),
        }),
        created_at: ctx.clock().utc_now(),
    };
    let store = ctx.releases();
    let stored = store.put(&manifest)?;
    store.move_alias(
        job.alias,
        Some(job.replaces),
        &stored.id,
        &ctx.clock().utc_now(),
    )?;
    ctx.repin_policy(job.alias);
    let held: Vec<ClaimId> = report
        .closing
        .iter()
        .chain(&report.retained)
        .cloned()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    ctx.claims().record_trained(&TrainedClaims {
        release: stored.id.clone(),
        claims: held,
    })?;
    Ok(stored.id)
}

/// The terms of what the checkpoint was made from: the datasets' and the
/// release in use's, the most restrictive of each axis; unknown when none
/// stated any.
fn terms_of(
    ctx: &Context,
    datasets: &[DatasetId],
    champion: &StoredRelease,
) -> Result<Terms, OrchestratorError> {
    let stored = datasets
        .iter()
        .map(|id| ctx.datasets().get(id))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(combine_stated(
        stored
            .iter()
            .map(|d| d.manifest.terms.as_ref())
            .chain(std::iter::once(Some(&champion.manifest.terms))),
    )
    .unwrap_or_else(|| Terms::unknown("unstated")))
}
