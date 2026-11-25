// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements online reinforcement learning that trains a
// served policy on its own rollouts while it runs, for its clients. If your
// team needs expertise in asynchronous policy optimisation, you can procure
// our services by sending an email to info@swedishembedded.com.

//! `train.online`: a candidate trained online on an executable task
//! collection (`dataset.rlvr`'s, pinned).
//!
//! The training engine runs a streaming policy-gradient job that serves each
//! version it publishes; the agent runtime rolls the collection's tasks out
//! against that served policy, `group_size` attempts per group, every
//! rollout graded by its task's verifier; the online controller
//! ([`crate::online`]) keeps rollouts in flight, filters groups that cannot
//! teach, bounds staleness and token backpressure, and streams each
//! qualifying group to the job, recording every group in the run state. A
//! controller killed mid-stage resumes from that ledger: no finished rollout
//! runs again, and no group is trained twice (the job deduplicates by key,
//! and a resumed job answers its committed keys as duplicates). A job that
//! restarts under the stage is started again under the same resume key and
//! the controller sends it everything not committed.
//!
//! A stage may start from a candidate (`start`, an adapter of its base,
//! such as a `model.train` stage's): the learner's run continues it and
//! publishes from it.
//!
//! The product holds the candidate (`candidate.json`, the adapter of the
//! newest version a durable checkpoint holds, load-checked by serving it),
//! `report.json` (the online statistics: reward over every scored group,
//! zero-variance groups included, the zero-std share, drops and filters,
//! backpressure, the learner's wait and staleness, tokens generated and
//! trained, utilisation and useful throughput) and `experiences.jsonl`, one
//! line per committed group.

use std::time::Instant;

use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use splinter_core::clock::unix_now;
use splinter_core::factory::execution::{FileRef, Isolation};
use splinter_core::factory::group::GroupState;
use splinter_core::factory::identity::StageIdentity;
use splinter_core::factory::operation::{
    ArtifactKind, OperationDescriptor, Port, Reproducibility, OPERATION_SCHEMA,
};
use splinter_core::factory::product::{CandidateKind, ModelCandidate, CANDIDATE_SCHEMA};
use splinter_core::revision::{ModelDigests, REVISIONS_FILE};
use splinter_store::factory::products::Staging;

use super::training::CANDIDATE_FILE;
use super::{io, stage_error, write_json, Factory, RunError, LEASE_SECONDS};
use crate::online::driver::{run, Ended, OnlineError, Report, RunLedger, Seams};
use crate::online::seams::{
    LearnerTelemetry, RolloutsRequest, RoundRobin, StreamLearner, SystemRunClock,
};
use crate::online::Settings;
use crate::plan::PlannedStage;
use crate::registry::OperationSpec;
use crate::training::{sha256, OnlineTrain};

/// The operation's name.
pub const ONLINE: &str = "train.online";

/// How many times a restarted job is started again before the stage fails.
const RESTARTS: u32 = 8;

fn one() -> u32 {
    1
}
fn eight() -> u32 {
    8
}
fn temperature_one() -> f32 {
    1.0
}
fn clipped() -> String {
    "clipped".into()
}
fn two() -> u32 {
    2
}

/// `train.online`'s parameters.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OnlineParams {
    /// The model role of the base the policy is trained from.
    pub base: String,
    /// Groups the run produces in all, its tasks in turn.
    pub groups: u64,
    /// Rollouts per group.
    #[serde(default = "eight")]
    pub group_size: u32,
    /// Rollouts in flight at once at most.
    #[serde(default = "eight")]
    pub max_concurrent_attempts: u32,
    /// Versions a group's oldest request may lag the published policy.
    #[serde(default = "one_u64")]
    pub max_staleness: u64,
    /// Policy tokens a rollout is assumed to sample until one has finished.
    pub rollout_tokens: u64,
    /// Policy tokens an optimiser step trains.
    pub step_tokens: u64,
    /// Policy tokens the learner's queue holds at most (at least a step).
    pub queue_tokens: u64,
    /// Steps between two published versions.
    #[serde(default = "one")]
    pub publish_every: u32,
    /// Optimiser steps after which the learner stops.
    #[serde(default)]
    pub max_updates: Option<u32>,
    /// The estimator: `clipped` (the clipped surrogate) or `score-centred`
    /// (centred over the sampler's top-K, `top_logprobs` of them).
    #[serde(default = "clipped")]
    pub estimator: String,
    /// The sampling temperature; the whole distribution is sampled.
    #[serde(default = "temperature_one")]
    pub temperature: f32,
    /// The sampler's top-K asked of every request.
    #[serde(default)]
    pub top_logprobs: Option<u32>,
    /// Steps between the learner's durable checkpoints.
    #[serde(default)]
    pub checkpoint_every: Option<u32>,
    /// Infrastructure failures a rollout is retried after.
    #[serde(default = "two")]
    pub max_failures: u32,
    /// The seed every rollout's draws derive from.
    pub seed: u64,
    /// The learner's other policy-gradient settings, as brain names them.
    #[serde(default)]
    pub params: Map<String, Value>,
}

fn one_u64() -> u64 {
    1
}

/// The `train.online` operation.
pub struct Online;

fn port(name: &str, kind: ArtifactKind) -> Port {
    Port {
        name: name.into(),
        kind,
        many: false,
        optional: false,
    }
}

impl OnlineParams {
    fn of(value: &Value) -> Result<Self, String> {
        let p: Self = serde_json::from_value(value.clone()).map_err(|e| e.to_string())?;
        p.settings("check").validate().map_err(|e| e.to_string())?;
        if p.queue_tokens < p.step_tokens {
            return Err(
                "queue_tokens is below step_tokens: the learner could never take a step".into(),
            );
        }
        if !(p.temperature.is_finite() && p.temperature > 0.0) {
            return Err("temperature is not a number above zero".into());
        }
        match (p.estimator.as_str(), p.top_logprobs) {
            ("clipped", _) => {}
            ("score-centred", Some(k)) if k > 0 => {}
            ("score-centred", _) => return Err(
                "the score-centred estimator centres over the sampler's top-K: set top_logprobs"
                    .into(),
            ),
            (other, _) => {
                return Err(format!(
                    "unknown estimator {other:?} (clipped, score-centred)"
                ))
            }
        }
        if p.groups == 0 || p.publish_every == 0 {
            return Err("groups and publish_every are at least 1".into());
        }
        Ok(p)
    }

    fn settings(&self, namespace: &str) -> Settings {
        Settings {
            namespace: namespace.to_string(),
            group_size: self.group_size,
            max_staleness: self.max_staleness,
            rollout_slots: self.max_concurrent_attempts,
            tokens_per_step: self.step_tokens,
            rollout_tokens: self.rollout_tokens,
            step_window: 8,
            max_failures: self.max_failures,
            send_retry_ms: 1_000,
            poll_ms: 200,
        }
    }

    /// The learner's settings beside the stream: the estimator and what the
    /// recipe named.
    fn learner_params(&self) -> Map<String, Value> {
        let mut params = self.params.clone();
        params.insert("estimator".into(), json!(self.estimator));
        params
    }
}

impl OperationSpec for Online {
    fn descriptor(&self) -> OperationDescriptor {
        OperationDescriptor {
            schema: OPERATION_SCHEMA.into(),
            name: ONLINE.into(),
            revision: 1,
            summary: "a candidate trained online on its own rollouts of an executable collection"
                .into(),
            inputs: vec![
                port("tasks", ArtifactKind::Tasks),
                Port {
                    optional: true,
                    ..port("start", ArtifactKind::Model)
                },
            ],
            outputs: vec![port("model", ArtifactKind::Model)],
            parameters: serde_json::to_value(schema_for!(OnlineParams)).unwrap_or(Value::Null),
            reproducibility: Reproducibility::Nondeterministic,
            isolation: Isolation::Any,
        }
    }

    fn check(&self, params: &Value) -> Result<(), String> {
        OnlineParams::of(params).map(|_| ())
    }

    fn model_roles(&self, params: &Value) -> Vec<String> {
        OnlineParams::of(params)
            .map(|p| vec![p.base])
            .unwrap_or_default()
    }
}

/// The stage's inputs by content: its tasks, and the candidate it starts
/// from when it starts from one.
fn inputs_of(
    factory: &Factory<'_>,
    stage: &PlannedStage,
) -> Result<Vec<(String, splinter_core::digest::Digest)>, RunError> {
    let mut inputs = Vec::new();
    for key in std::iter::once(tasks_key(stage)?).chain(stage.source("start").map(str::to_owned)) {
        inputs.push((key.clone(), factory.product(&key)?.1.digest()));
    }
    Ok(inputs)
}

/// The adapter the stage's policy starts from: the `start` candidate's,
/// which must be an adapter of the stage's own base.
fn start_adapter(
    factory: &Factory<'_>,
    stage: &PlannedStage,
    base: &crate::recipe::ModelRef,
) -> Result<Option<std::path::PathBuf>, RunError> {
    let Some(key) = stage.source("start") else {
        return Ok(None);
    };
    let view = factory.view(key);
    let candidate = super::training::candidate_of(&view)?;
    if candidate.base != base.to_string() || candidate.kind != CandidateKind::Adapter {
        return Err(stage_error(
            &stage.key,
            format!(
                "start {key} is a {:?} candidate of {}, not an adapter of the base {base}",
                candidate.kind, candidate.base
            ),
        ));
    }
    let adapter = candidate
        .artifacts
        .iter()
        .find(|a| a.path.ends_with(".safetensors"))
        .ok_or_else(|| stage_error(&stage.key, format!("start {key} holds no adapter file")))?;
    Ok(Some(view.join(&adapter.path)))
}

fn tasks_key(stage: &PlannedStage) -> Result<String, RunError> {
    stage
        .source("tasks")
        .map(str::to_owned)
        .ok_or_else(|| stage_error(&stage.key, "trains on no tasks"))
}

fn identity_parts(
    factory: &Factory<'_>,
    stage: &PlannedStage,
) -> Result<(OnlineParams, String), RunError> {
    let params = OnlineParams::of(&stage.params).map_err(|e| stage_error(&stage.key, e))?;
    let trainer = factory
        .services
        .trainer
        .ok_or_else(|| stage_error(&stage.key, "no trainer is configured"))?;
    let rollouts = factory
        .services
        .online
        .ok_or_else(|| stage_error(&stage.key, "nothing runs online rollouts here"))?;
    Ok((
        params,
        format!("{} | {}", trainer.identity(), rollouts.identity()),
    ))
}

pub(crate) fn identity(
    factory: &Factory<'_>,
    stage: &PlannedStage,
) -> Result<StageIdentity, RunError> {
    let (params, engines) = identity_parts(factory, stage)?;
    let base = factory.resolved.model(&params.base).ok_or_else(|| {
        stage_error(
            &stage.key,
            format!("model role {} is not bound", params.base),
        )
    })?;
    let parameters = json!({"params": params, "base": base.to_string()});
    Ok(super::identity_of(
        &stage.versioned_operation(),
        &stage.key,
        parameters,
        inputs_of(factory, stage)?,
        &engines,
        Some(params.seed),
    ))
}

/// What the stage's learner and controller came to.
struct Trained {
    report: Report,
    telemetry: LearnerTelemetry,
    learner: Box<dyn StreamLearner>,
    restarts: u32,
}

pub(crate) fn produce(
    factory: &Factory<'_>,
    stage: &PlannedStage,
    identity: &StageIdentity,
    staging: &Staging,
) -> Result<(), RunError> {
    let started = Instant::now();
    let (params, _) = identity_parts(factory, stage)?;
    let fail = |e: String| stage_error(&stage.key, e);
    let trainer = factory
        .services
        .trainer
        .ok_or_else(|| fail("no trainer".into()))?;
    let rollouts = factory
        .services
        .online
        .ok_or_else(|| fail("no online rollouts".into()))?;
    let base = factory
        .resolved
        .model(&params.base)
        .cloned()
        .ok_or_else(|| fail(format!("model role {} is not bound", params.base)))?;
    let key = tasks_key(stage)?;
    let tasks = factory.view(&key);
    let digest = identity.digest();
    let hex = digest.hex();
    // One namespace per stage identity: a changed stage never resumes
    // another's groups.
    let tag = format!("{}-{}", stage.key, &hex[..12]);
    let publish = format!("splinter/online:{}:{}:latest", stage.key, &hex[..12]);
    let scratch = factory.work(&stage.key).join("rollouts");
    std::fs::create_dir_all(&scratch).map_err(io(&scratch))?;
    let learner_params = params.learner_params();
    let start = start_adapter(factory, stage, &base)?;
    let mut restarts = 0;
    let trained = loop {
        let mut learner = trainer
            .stream(&OnlineTrain {
                base: &base,
                params: &learner_params,
                resume_key: &tag,
                checkpoint_every: params.checkpoint_every,
                publish_adapter: &publish,
                publish_every: params.publish_every,
                step_tokens: params.step_tokens,
                queue_tokens: params.queue_tokens,
                max_staleness: params.max_staleness,
                max_steps: params.max_updates,
                top_k: params
                    .top_logprobs
                    .filter(|_| params.estimator == "score-centred"),
                start: start.as_deref(),
                job_place: &factory.work(&stage.key).join("stream-job.json"),
            })
            .map_err(|e| fail(e.to_string()))?;
        let served = learner.served_as();
        let mut outcome: Option<Result<Report, OnlineError>> = None;
        rollouts
            .run(
                &RolloutsRequest {
                    tasks: &tasks,
                    policy: &served,
                    temperature: params.temperature,
                    top_logprobs: params.top_logprobs,
                    seed: params.seed,
                    scratch: &scratch,
                },
                &mut |executor, task_ids| {
                    let selector = RoundRobin::new(task_ids.to_vec(), params.groups);
                    let mut selector = match selector {
                        Ok(selector) => selector,
                        Err(e) => {
                            outcome = Some(Err(OnlineError::Control(
                                crate::online::control::ControlError(e.to_string()),
                            )));
                            return;
                        }
                    };
                    let mut ledger = RunLedger::new(
                        &factory.state,
                        factory.fence.clone(),
                        LEASE_SECONDS,
                        unix_now(),
                    );
                    let cancel = || factory.cancelled().is_err();
                    outcome = Some(run(
                        params.settings(&tag),
                        Seams {
                            executor,
                            learner: learner.as_mut(),
                            selector: &mut selector,
                            ledger: &mut ledger,
                            clock: &SystemRunClock,
                            cancel: &cancel,
                        },
                    ));
                },
            )
            .map_err(fail)?;
        match outcome {
            Some(Ok(report)) if report.ended == Ended::Cancelled => {
                return Err(RunError::Cancelled)
            }
            Some(Ok(report)) => {
                break Trained {
                    report,
                    telemetry: learner.telemetry(),
                    learner,
                    restarts,
                }
            }
            Some(Err(OnlineError::Learner(e))) if e.restarted && restarts < RESTARTS => {
                restarts += 1;
                eprintln!(
                    "{}: the learner restarted ({}); resuming against it",
                    stage.key, e.message
                );
            }
            Some(Err(e)) => return Err(fail(e.to_string())),
            None => return Err(fail("the rollouts never ran".into())),
        }
    };
    let mut trained = trained;
    let candidate = candidate(factory, stage, &params, &base, &mut trained, staging)?;
    // The code, servers, data and configuration of the run, with the base and
    // tokenizer the learner named: in the product and in the report.
    let revisions = factory.stage_revisions(identity, served_models(&trained.telemetry));
    write_json(&staging.path().join(REVISIONS_FILE), &revisions)?;
    let mut report = report(&trained, started);
    report["revisions"] = json!(revisions);
    write_json(&staging.path().join("report.json"), &report)?;
    write_experiences(factory, &tag, staging)?;
    write_json(&staging.path().join(CANDIDATE_FILE), &candidate)
}

/// The candidate: the newest version a checkpoint holds, served once.
fn candidate(
    factory: &Factory<'_>,
    stage: &PlannedStage,
    params: &OnlineParams,
    base: &crate::recipe::ModelRef,
    trained: &mut Trained,
    staging: &Staging,
) -> Result<ModelCandidate, RunError> {
    let fail = |e: String| stage_error(&stage.key, e);
    let (version, _) = trained
        .telemetry
        .committed_version()
        .ok_or_else(|| fail("the learner committed no published version".into()))?;
    let name = "adapter.safetensors";
    let adapter = staging.path().join(name);
    trained
        .learner
        .final_adapter(&adapter)
        .map_err(|e| fail(e.to_string()))?;
    let trainer = factory
        .services
        .trainer
        .ok_or_else(|| fail("no trainer".into()))?;
    // The load check: the candidate counts only once the trainer's server lists it.
    let served_as = trainer
        .serve(base, Some(&adapter))
        .map_err(|e| fail(e.to_string()))?;
    Ok(ModelCandidate {
        schema: CANDIDATE_SCHEMA.into(),
        name: stage.key.clone(),
        kind: CandidateKind::Adapter,
        base: base.to_string(),
        artifacts: vec![FileRef {
            path: name.into(),
            digest: sha256(&adapter).map_err(io(&adapter))?,
        }],
        dataset: factory.product(&tasks_key(stage)?)?.1.digest(),
        method: ONLINE.into(),
        params: json!({"params": params, "version": version}),
        trainer: trainer.identity(),
        seed: params.seed,
        report: json!({"version": version, "steps": trained.telemetry.steps}),
        served_as: Some(served_as),
        load_checked: true,
    })
}

/// The base and tokenizer the learner's versions name: each base digest a
/// published version's lineage names, with the tokenizer the run streamed
/// against; none when the server named neither.
fn served_models(telemetry: &LearnerTelemetry) -> Vec<ModelDigests> {
    let mut models: Vec<ModelDigests> = telemetry
        .published
        .iter()
        .filter_map(|p| p.lineage.as_ref())
        .map(|lineage| ModelDigests {
            base_digest: Some(lineage.base_digest.clone()),
            tokenizer_digest: telemetry.tokenizer_digest.clone(),
        })
        .collect();
    if models.is_empty() && telemetry.tokenizer_digest.is_some() {
        models.push(ModelDigests {
            base_digest: None,
            tokenizer_digest: telemetry.tokenizer_digest.clone(),
        });
    }
    models.sort();
    models.dedup();
    models
}

#[allow(clippy::cast_precision_loss)]
fn report(trained: &Trained, started: Instant) -> Value {
    let stats = &trained.report.stats;
    let telemetry = &trained.telemetry;
    let wall = started.elapsed().as_secs_f64();
    let generated = stats.tokens_generated;
    let trained_tokens = telemetry.trained_tokens;
    json!({
        "reward_mean": stats.reward.mean(),
        "reward_std": stats.reward.std(),
        "frac_zero_std": stats.frac_zero_std(),
        "reward_zero_std": stats.zero_std_reward.mean(),
        "groups": {
            "started": stats.groups_started, "scored": stats.groups_scored, "sent": stats.groups_sent,
            "applied": stats.groups_applied, "committed": stats.groups_committed,
            "dropped_stale": stats.dropped_stale, "dropped_by_learner": stats.dropped_by_learner,
            "filtered": {"zero_variance": stats.filtered_zero_variance, "not_exact": stats.filtered_not_exact,
                         "unscored": stats.filtered_unscored},
            "resends": stats.resends,
        },
        "rollouts": {
            "finished": stats.rollouts_finished, "unscored": stats.rollouts_unscored,
            "not_exact": stats.rollouts_not_exact, "truncated": stats.rollouts_truncated,
            "failures": stats.rollout_failures, "abandoned": stats.rollouts_abandoned,
        },
        "backpressure_seconds": stats.backpressure_ms as f64 / 1000.0,
        "learner": telemetry,
        "staleness_at_send": stats.staleness,
        "tokens_generated": generated,
        "tokens_trained": trained_tokens,
        // How much of what was generated was trained on, and how fast useful
        // tokens were made: unmeasured is absent, never zero.
        "experience_utilisation": (generated > 0).then(|| trained_tokens as f64 / generated as f64),
        "useful_tokens_per_second": (wall > 0.0).then(|| trained_tokens as f64 / wall),
        "wall_seconds": wall,
        "learner_restarts": trained.restarts,
        // Which run-level identity served each version: the weights every
        // rollout of that version was drawn from, named across servers.
        "versions": telemetry.published,
    })
}

/// One line per committed group of the stage's namespace.
fn write_experiences(
    factory: &Factory<'_>,
    namespace: &str,
    staging: &Staging,
) -> Result<(), RunError> {
    let prefix = format!("{namespace}:");
    let mut text = String::new();
    for entry in factory.state.group_entries()? {
        let group = &entry.group;
        if group.state != GroupState::Committed || !group.key.as_str().starts_with(&prefix) {
            continue;
        }
        let rollouts: Vec<Value> = entry
            .rollouts
            .iter()
            .map(|r| {
                json!({"index": r.index, "reward": r.outcome.as_ref().and_then(|o| o.reward),
                       "tokens": r.outcome.as_ref().map(|o| o.tokens),
                       "span": r.outcome.as_ref().map(|o| o.span),
                       "exact": r.outcome.as_ref().map(|o| o.exact)})
            })
            .collect();
        let line = json!({"group": group.key, "task": group.task, "staleness": group.staleness, "rollouts": rollouts});
        text.push_str(&line.to_string());
        text.push('\n');
    }
    let path = staging.path().join("experiences.jsonl");
    std::fs::write(&path, text).map_err(io(&path))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(extra: Value) -> Value {
        let mut p = json!({"base": "policy", "groups": 4, "rollout_tokens": 64, "step_tokens": 64,
                           "queue_tokens": 128, "seed": 1});
        p.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        p
    }

    #[test]
    fn parameters_that_could_not_train_are_refused_by_name() {
        assert!(Online.check(&params(json!({}))).is_ok());
        for (bad, why) in [
            (json!({"queue_tokens": 32}), "queue_tokens"),
            (json!({"estimator": "score-centred"}), "top_logprobs"),
            (json!({"estimator": "ppo"}), "unknown estimator"),
            (json!({"group_size": 1}), "group_size"),
            (json!({"temperature": 0.0}), "temperature"),
            (json!({"surprise": 1}), "unknown field"),
        ] {
            let refused = Online.check(&params(bad.clone())).unwrap_err();
            assert!(refused.contains(why), "{bad}: {refused}");
        }
        assert!(Online
            .check(&params(
                json!({"estimator": "score-centred", "top_logprobs": 8})
            ))
            .is_ok());
        assert_eq!(Online.model_roles(&params(json!({}))), ["policy"]);
    }
}
