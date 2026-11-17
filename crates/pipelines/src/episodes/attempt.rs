// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that turn practice task folders
// into graded, replayable agent experience for its clients. If your team
// needs expertise in on-policy experience collection for agent training you
// can procure our services by sending an email to info@swedishembedded.com.

//! One attempt: found recorded, or run, graded and recorded.

use serde::Serialize;
use splinter_agent::solve::{Model, SolveOptions};
use splinter_agent::work::WorkProtocol;
use splinter_agent::CancelToken;
use splinter_core::digest::{canonical_json, Digest};
use splinter_core::experience::Provenance;
use splinter_core::model_ref::ModelRef;
use splinter_eval::verifiers::verify_and_annotate;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_sandbox::ResolvedEnvironment;

use super::grade::{grade, trace_summary, usage};
use super::{
    exact_sampling, EpisodeError, EpisodeOutcome, EpisodeRecord, EpisodeSettings, EpisodeUsage,
    PolicyIdentity, PracticeTask, TraceSummary, RECORD_FORMAT,
};
use crate::attempt_store::store_attempt;
use crate::solving::{conclusion_label, harness_protocol};
use crate::workspace_solve::{solve_attempt, WorkspaceInputs};

/// The document class of an episode record.
const RECORD_CLASS: &str = "practice_episode";

/// The pointers that name the record of each attempt, by its key.
const POINTER_PREFIX: &str = "practice-episode/";

/// An attempt to run, or to find recorded.
pub(super) struct Planned<'a> {
    pub(super) task: &'a PracticeTask,
    pub(super) attempt: u32,
    pub(super) key: Digest,
}

/// What became of an attempt.
pub(super) enum Done {
    /// It ran now.
    Ran(EpisodeRecord),
    /// A finished earlier run had recorded it.
    Reused(EpisodeRecord),
    /// The run was cancelled before it began.
    NotStarted,
}

/// What every attempt of a run shares.
pub(super) struct Run<'a> {
    pub(super) ctx: &'a Context,
    pub(super) reference: &'a ModelRef,
    pub(super) model: &'a Model,
    pub(super) policy: &'a PolicyIdentity,
    pub(super) config: &'a Digest,
    pub(super) settings: &'a EpisodeSettings,
    pub(super) cancel: &'a CancelToken,
}

/// What an attempt's key is the digest of.
#[derive(Serialize)]
struct KeyOf<'a> {
    format: &'static str,
    task: &'a Digest,
    source: &'a Digest,
    policy: &'a PolicyIdentity,
    config: &'a Digest,
    seed: u64,
    attempt: u32,
}

/// The content key of attempt `attempt` at `task`.
pub(super) fn key(
    task: &PracticeTask,
    policy: &PolicyIdentity,
    config: &Digest,
    seed: u64,
    attempt: u32,
) -> Result<Digest, EpisodeError> {
    let bytes = canonical_json(&KeyOf {
        format: RECORD_FORMAT,
        task: &task.task,
        source: &task.source,
        policy,
        config,
        seed,
        attempt,
    })
    .map_err(|e| EpisodeError::Settings(format!("an attempt's key: {e}")))?;
    Ok(Digest::of(&bytes))
}

/// The seed of the draws of attempt `attempt` at `task` in a run seeded
/// `seed`: a function of the three alone.
fn attempt_seed(seed: u64, task: &Digest, attempt: u32) -> u64 {
    let digest = Digest::of(format!("{seed}/{task}/{attempt}").as_bytes());
    // The first 16 hex digits of the address: 64 bits.
    u64::from_str_radix(&digest.hex()[..16], 16).unwrap_or_default()
}

impl Run<'_> {
    /// Attempt `planned`: its record when a finished run left one, else the
    /// record of running it now.
    pub(super) fn attempt(&self, planned: &Planned<'_>) -> Result<Done, EpisodeError> {
        let pointer = format!("{POINTER_PREFIX}{}", planned.key.hex());
        let earlier = self.recorded(&pointer)?;
        if let Some((_, record)) = &earlier {
            if !matches!(record.outcome, EpisodeOutcome::Infrastructure { .. }) {
                return Ok(Done::Reused(record.clone()));
            }
        }
        if self.cancel.is_cancelled() {
            return Ok(Done::NotStarted);
        }
        // An attempt the run's cancel stopped is not finished: it has no
        // record, and a rerun runs it again.
        let Some(record) = self.run(planned)? else {
            return Ok(Done::NotStarted);
        };
        let workspace = self.ctx.workspace();
        let id = workspace
            .put_document(RECORD_CLASS, &record)
            .map_err(OrchestratorError::from)?;
        workspace
            .move_pointer(
                &pointer,
                earlier.as_ref().map(|(value, _)| value.as_str()),
                id.as_str(),
                &self.ctx.clock().utc_now(),
            )
            .map_err(OrchestratorError::from)?;
        workspace.commit().map_err(OrchestratorError::from)?;
        Ok(Done::Ran(record))
    }

    /// The record `pointer` names, with the pointer's value.
    fn recorded(&self, pointer: &str) -> Result<Option<(String, EpisodeRecord)>, EpisodeError> {
        let workspace = self.ctx.workspace();
        let Some((_, value)) = workspace
            .pointer(pointer)
            .map_err(OrchestratorError::from)?
        else {
            return Ok(None);
        };
        let id = Digest::parse(&value).map_err(|e| {
            EpisodeError::Settings(format!(
                "the episode pointer {pointer} names {value:?}: {e}"
            ))
        })?;
        let record = workspace
            .get_document::<EpisodeRecord>(RECORD_CLASS, &id)
            .map_err(OrchestratorError::from)?
            .ok_or_else(|| {
                EpisodeError::Settings(format!("the episode record {id} is not stored"))
            })?;
        Ok(Some((value, record)))
    }

    /// Runs `planned` now: the episode, its experience, its verdicts; `None`
    /// when the run's cancel stopped the episode.
    fn run(&self, planned: &Planned<'_>) -> Result<Option<EpisodeRecord>, EpisodeError> {
        let ctx = self.ctx;
        let seed = attempt_seed(self.settings.seed, &planned.task.task, planned.attempt);
        let (model, seeded) = match ctx.resampled(self.reference, exact_sampling(seed))? {
            Some(model) => (model, true),
            None => (self.model.clone(), false),
        };
        let mut record = EpisodeRecord {
            format: RECORD_FORMAT.into(),
            key: planned.key.clone(),
            task: planned.task.task.clone(),
            name: planned.task.name.clone(),
            source: Some(planned.task.source.clone()),
            attempt: planned.attempt,
            seed,
            seeded,
            policy: self.policy.clone(),
            config: self.config.clone(),
            experience: None,
            conclusion: None,
            outcome: EpisodeOutcome::Infrastructure {
                detail: String::new(),
            },
            reward: None,
            usage: EpisodeUsage::default(),
            trace: TraceSummary::default(),
        };
        let infrastructure = |mut record: EpisodeRecord, detail: String| {
            record.outcome = EpisodeOutcome::Infrastructure { detail };
            Ok(Some(record))
        };
        let task = ctx
            .tasks()
            .get(&planned.task.task)
            .map_err(OrchestratorError::from)?;
        crate::sealed::admit_task(ctx, &task)?;
        let env = match ctx.environments().for_record(&task.environment) {
            Ok(ResolvedEnvironment::Workspace(env)) => env,
            Ok(_) => {
                return Err(EpisodeError::Malformed {
                    directory: planned.task.directory.clone(),
                    reason: "its task is not a change to a workspace".into(),
                })
            }
            Err(e) => return infrastructure(record, e.to_string()),
        };
        let inputs = match WorkspaceInputs::of(ctx, &env, &task) {
            Ok(inputs) => inputs,
            Err(why) => return infrastructure(record, why),
        };
        let mut options = SolveOptions::new(self.settings.deadline);
        options.max_output_tokens = self.settings.max_output_tokens;
        options.cancel = Some(self.cancel.clone());
        options.want_logprobs = true;
        let options = model.solving(options);
        let protocol = WorkProtocol {
            max_tool_calls: self.settings.max_tool_calls,
            prompt: ctx.config().coding_prompt,
            ..harness_protocol()
        };
        let solved = ctx.block_on(solve_attempt(
            &task, &env, &inputs, &model, options, &protocol,
        ));
        if self.cancel.is_cancelled() {
            return Ok(None);
        }
        let (solution, report) = match solved {
            Ok(solved) => solved,
            Err(e) => return infrastructure(record, e.to_string()),
        };
        record.conclusion = Some(conclusion_label(solution.conclusion, &report));
        record.trace = trace_summary(&solution.trajectory);
        record.usage = usage(&solution.trajectory, &report);
        let provenance = Provenance {
            policy: self.policy.adapter.as_ref().map(Digest::to_string),
            attempt: Some(planned.attempt),
            ..Provenance::new(model.identity.clone(), ctx.clock())
        };
        let store = ctx.experiences();
        let id = store_attempt(&store, task.clone(), solution, provenance, report)?;
        ctx.workspace().commit().map_err(OrchestratorError::from)?;
        let experience = store.get(&id).map_err(OrchestratorError::from)?;
        record.experience = Some(id);
        // The agent is done: the verifiers grade what it left, outside its
        // reach. A verifier that could not run is a verifier error.
        record.outcome = match crate::verify::verifiers_for(ctx, &task, &[], None) {
            Ok(verifiers) => match verify_and_annotate(&store, &verifiers, &task, &experience) {
                Ok(verification) => grade(&task, &verification),
                Err(e) => EpisodeOutcome::VerifierError {
                    detail: e.to_string(),
                },
            },
            Err(e) => EpisodeOutcome::VerifierError {
                detail: e.to_string(),
            },
        };
        record.reward = record.outcome.reward();
        Ok(Some(record))
    }
}
