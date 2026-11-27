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
//! trained, utilisation and useful throughput), `experiences.jsonl`, one
//! line per committed group, and `groups.jsonl`, the log of every group's
//! transitions the stage wrote as it ran ([`crate::online::group_log`]).

use std::time::Instant;

use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use splinter_core::factory::execution::{FileRef, Isolation};
use splinter_core::factory::group::GroupState;
use splinter_core::factory::identity::StageIdentity;
use splinter_core::factory::operation::{
    ArtifactKind, OperationDescriptor, Port, Reproducibility, OPERATION_SCHEMA,
};
use splinter_core::factory::product::{CandidateKind, ModelCandidate, CANDIDATE_SCHEMA};
use splinter_core::revision::{ModelDigests, REVISIONS_FILE};
use splinter_store::factory::products::Staging;

use super::online_train::{self, Trained};
use super::online_worker::{self, OnlineAgent};
use super::training::CANDIDATE_FILE;
use super::{io, stage_error, write_json, Factory, RunError};
use crate::online::frontier::SelectorKind;
use crate::online::group_log::GROUPS_FILE;
use crate::online::seams::{LearnerTelemetry, OnlineRollouts};
use crate::online::watchdog::WatchdogSettings;
use crate::online::Settings;
use crate::plan::PlannedStage;
use crate::registry::OperationSpec;
use crate::training::sha256;

/// The operation's name.
pub const ONLINE: &str = "train.online";

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
    /// The most tokens a rollout's reply may sample: one that reaches it ends truncated (counted, never trained)
    /// instead of running on to the end of the served context. Unset, a reply may fill the context.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,
    /// Steps between the learner's durable checkpoints.
    #[serde(default)]
    pub checkpoint_every: Option<u32>,
    /// Infrastructure failures a rollout is retried after.
    #[serde(default = "two")]
    pub max_failures: u32,
    /// The seed every rollout's draws derive from.
    pub seed: u64,
    /// How each group's task is drawn: `round_robin` (the tasks in turn) or
    /// `frontier` (in proportion to each task's learning signal under a
    /// posterior on its pass rate; a task the policy passes reliably
    /// retires, and the run ends early when every task has).
    #[serde(default, skip_serializing_if = "SelectorKind::is_default")]
    pub selector: SelectorKind,
    /// Train only rollouts that one policy version decoded from their first
    /// request to their last; a rollout that straddles a publication is
    /// scored and counted (`rollouts.version_mixed`), never trained. Size
    /// `publish_every` so a version outlives a rollout, or most rollouts
    /// straddle one.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub pin_versions: bool,
    /// Run every request of an episode on the version it started on: the
    /// learner serves its newest versions under ids of their own and each
    /// rollout asks the one that was newest when it started, so its span is
    /// one version (needs a brain that retains versions).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub pin_episodes: bool,
    /// Watch the training and roll it back when it moves the policy the
    /// wrong way: invariants pause intake, drift asks for an evaluation,
    /// and a paired evaluation of the newest version against the anchor on
    /// the held-out `monitor` tasks (the `confirm` tasks before a promotion)
    /// decides. Unset, nothing watches.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub watchdog: Option<WatchdogSettings>,
    /// Run each rollout on the execution worker, in its task's own
    /// environment, with this agent: the tools, configuration and sandbox
    /// the task declares, exactly as an attempt pass runs it. Unset, each
    /// rollout is a session in this process in a copy of its task's tree
    /// (a task that gives its agent tools is refused there).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<OnlineAgent>,
    /// The learner's other policy-gradient settings, as brain names them (not `thinking`: the rollouts are drawn
    /// with thinking off, and the learner always trains that mode).
    #[serde(default)]
    pub params: Map<String, Value>,
}

/// The most versions a learner is asked to keep served under ids of their
/// own (brain bounds it at 16: each holds a slot of its adapter bank).
const RETAINED_VERSIONS: u32 = 16;

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
        if p.params.contains_key("thinking") {
            return Err("params.thinking: the rollouts are drawn with thinking off and the learner trains that mode; \
                 it is not a setting of an online run"
                .into());
        }
        if p.max_output_tokens == Some(0) {
            return Err("max_output_tokens is at least 1".into());
        }
        if let Some(agent) = &p.agent {
            agent.check()?;
        }
        if let Some(watchdog) = &p.watchdog {
            watchdog.validate().map_err(|e| format!("watchdog: {e}"))?;
        }
        if p.groups == 0 || p.publish_every == 0 {
            return Err("groups and publish_every are at least 1".into());
        }
        Ok(p)
    }

    /// How many of its newest versions the learner keeps served under ids
    /// of their own: enough for a pinned episode to stay on its version as
    /// long as its group may lag (and for the watchdog to evaluate them);
    /// none when nothing addresses a version.
    pub(super) fn retained_versions(&self) -> Option<u32> {
        (self.pin_episodes || self.watchdog.is_some()).then(|| {
            u32::try_from(self.max_staleness.saturating_add(2))
                .unwrap_or(u32::MAX)
                .min(RETAINED_VERSIONS)
        })
    }

    pub(super) fn settings(&self, namespace: &str) -> Settings {
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
            epoch: 0,
            pin_versions: self.pin_versions,
            pin_episodes: self.pin_episodes,
        }
    }

    /// The learner's settings beside the stream: the estimator and what the
    /// recipe named.
    pub(super) fn learner_params(&self) -> Map<String, Value> {
        let mut params = self.params.clone();
        params.insert("estimator".into(), json!(self.estimator));
        // The rollouts ask the policy for no reasoning block, so each answer was sampled after the template's closed
        // think block; the learner renders the prompts it scores them in the same mode.
        params.insert("thinking".into(), json!(false));
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
                // The watchdog's held-out tasks: the monitor pool its
                // evaluations draw from, the confirm set read before a
                // promotion.
                Port {
                    optional: true,
                    ..port("monitor", ArtifactKind::Tasks)
                },
                Port {
                    optional: true,
                    ..port("confirm", ArtifactKind::Tasks)
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
    let held_out = ["monitor", "confirm"].map(|port| stage.source(port).map(str::to_owned));
    let params = OnlineParams::of(&stage.params).map_err(|e| stage_error(&stage.key, e))?;
    if params.watchdog.is_some() && held_out.iter().any(Option::is_none) {
        return Err(stage_error(
            &stage.key,
            "a watchdog evaluates on held-out tasks: give the stage monitor and confirm inputs",
        ));
    }
    for key in std::iter::once(tasks_key(stage)?)
        .chain(stage.source("start").map(str::to_owned))
        .chain(held_out.into_iter().flatten())
    {
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
    let rollouts = match &params.agent {
        Some(agent) => {
            let arm = online_worker::arm(factory, stage, &tasks_key(stage)?, agent, params.seed)?;
            online_worker::identity(factory, &arm)?
        }
        None => factory
            .services
            .online
            .ok_or_else(|| stage_error(&stage.key, "nothing runs online rollouts here"))?
            .identity(),
    };
    Ok((params, format!("{} | {rollouts}", trainer.identity())))
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
    let groups_log = factory.work(&stage.key).join(GROUPS_FILE);
    std::fs::create_dir_all(&scratch).map_err(io(&scratch))?;
    let start = start_adapter(factory, stage, &base)?;
    let arm = match &params.agent {
        Some(agent) => Some(online_worker::arm(
            factory,
            stage,
            &key,
            agent,
            params.seed,
        )?),
        None => None,
    };
    let worker = match &arm {
        Some(arm) => Some(online_worker::WorkerRollouts::new(
            factory,
            arm,
            &base,
            factory.work(&stage.key).join("worker"),
        )?),
        None => None,
    };
    let rollouts: &dyn OnlineRollouts = match &worker {
        Some(worker) => worker,
        None => factory
            .services
            .online
            .ok_or_else(|| fail("no online rollouts".into()))?,
    };
    let held_out = match (stage.source("monitor"), stage.source("confirm")) {
        (Some(monitor), Some(confirm)) => Some(online_train::HeldOut {
            monitor: factory.view(monitor),
            confirm: factory.view(confirm),
        }),
        _ => None,
    };
    let trained = online_train::train(&online_train::Stage {
        factory,
        stage,
        params: &params,
        base: &base,
        trainer,
        rollouts,
        tasks: &tasks,
        tag: &tag,
        publish: &publish,
        scratch: &scratch,
        groups_log: &groups_log,
        start: start.as_deref(),
        held_out,
    })?;
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
    let logged = staging.path().join(GROUPS_FILE);
    std::fs::copy(&groups_log, &logged).map_err(io(&logged))?;
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
                         "unscored": stats.filtered_unscored, "version_mixed": stats.filtered_version_mixed},
            "rolled_back": stats.groups_rolled_back,
            "resends": stats.resends,
        },
        "rollouts": {
            "finished": stats.rollouts_finished, "unscored": stats.rollouts_unscored,
            "not_exact": stats.rollouts_not_exact, "truncated": stats.rollouts_truncated,
            "truncations": stats.truncations,
            "version_mixed": stats.rollouts_version_mixed,
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
        "selector": trained.selector,
        "watchdog": trained.watchdog,
        // Which run-level identity served each version: the weights every
        // rollout of that version was drawn from, named across servers.
        "versions": telemetry.published,
        // Training health: each step's figures as the learner reported them,
        // the steps whose update it skipped, and what each policy version's
        // rollouts came to (rates over what was measured; absent when nothing was).
        "health": {
            "steps": telemetry.step_health,
            "skipped_steps": telemetry.skipped_steps,
            "versions": trained.series.summary(),
        },
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
            (json!({"params": {"thinking": true}}), "thinking"),
            (json!({"max_output_tokens": 0}), "max_output_tokens"),
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

    #[test]
    fn the_learner_trains_the_mode_the_rollouts_are_drawn_in() {
        // The rollouts ask a local server for no reasoning block, so the policy was sampled after the template's
        // closed think block: the learner must render each answer's prompt the same way (no-think mode), or every
        // record's prompt ids disagree with the context it would be scored in.
        let p = OnlineParams::of(&params(json!({"params": {"rank": 16}}))).unwrap();
        let learner = p.learner_params();
        assert_eq!(learner["thinking"], json!(false));
        assert_eq!(learner["rank"], json!(16));
        assert_eq!(learner["estimator"], json!("clipped"));
    }
}
