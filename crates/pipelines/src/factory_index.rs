// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements experience databases that index what many
// learning experiments did, for its clients. If your team needs expertise in
// learning from agent experience across experiments, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The experience database as an optional index over learning-factory runs.
//!
//! [`index_run`] reads a run's committed rollout products and files every
//! trial that left a trajectory as an experience: the trajectory under a
//! task of the task repository (`bench` environment: attempted elsewhere,
//! never re-graded here), with the factory's grade as an executable
//! verdict, and one experience set per pass. A trial whose grade is not a
//! verdict (the infrastructure decided it) is filed with no verdict, never a
//! fail. Indexing is idempotent: experiences and verdicts are content
//! addressed, so indexing a run again files nothing new. The index never
//! decides anything about the run; the run's own products stay the record.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Serialize;
use serde_json::json;
use splinter_core::annotation::{Annotation, AnnotationBody, Outcome, Producer, Strength};
use splinter_core::clock::SystemClock;
use splinter_core::experience::{Environment, Experience, Provenance, Task};
use splinter_core::factory::product::TrialResult;
use splinter_core::kinds::BENCH_TASK;
use splinter_factory::run::rollout::trials_of;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_store::experiences::{ExperienceSet, SetId};
use splinter_store::factory::products::ProductStore;

/// The producer of the verdicts an index files.
pub const PRODUCER: &str = "splinter/factory";

/// What indexing one pass filed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct IndexedPass {
    /// The pass's product view (`rollouts/<pass>`, `eval/_arms/<arm>`).
    pub pass: String,
    /// Its experience set; `None` when no trial left a trajectory.
    pub experience_set: Option<SetId>,
    /// Experiences filed, by verdict (`pass`, `fail`, `abstain`).
    pub verdicts: BTreeMap<String, usize>,
    /// Trials left out, each with why.
    pub skipped: Vec<(String, String)>,
}

fn refused(why: impl ToString) -> OrchestratorError {
    OrchestratorError::Refused(why.to_string())
}

/// The trial's verdict: counted trials pass or fail, the rest have none.
fn outcome(trial: &TrialResult) -> Outcome {
    match (trial.grade.counted, trial.verified_success()) {
        (true, true) => Outcome::Pass,
        (true, false) => Outcome::Fail,
        (false, _) => Outcome::Abstain,
    }
}

fn experience(view: &Path, trial: &TrialResult) -> Result<Experience, String> {
    let trace = trial.trace.as_ref().ok_or("the trial left no trajectory")?;
    let bytes =
        std::fs::read(view.join(&trace.path)).map_err(|e| format!("{}: {e}", trace.path))?;
    let trajectory: atif::Trajectory =
        serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", trace.path))?;
    let instruction = trajectory
        .steps
        .iter()
        .find(|s| s.source == atif::StepOrigin::User)
        .and_then(|s| s.message.as_text().map(str::to_owned))
        .ok_or("the trajectory has no user step with text: the instruction is unknown")?;
    // What is the task's, not the trial's: another trial of the task, by any
    // agent, has the same task address.
    let spec = json!({ "task": { "id": trial.task, "digest": trial.task_digest } });
    let task = Task::new(
        BENCH_TASK,
        Vec::new(),
        Environment::new(Environment::BENCH, spec),
        instruction,
        Vec::new(),
    )
    .map_err(|e| e.to_string())?;
    let solver = format!(
        "factory/{}@{}/{}",
        trial.agent.name,
        trial.agent.version.as_deref().unwrap_or("?"),
        trial.model.as_deref().unwrap_or("own-model")
    );
    let mut provenance = Provenance::new(solver, &SystemClock);
    // The trial's own start, so the same trial is the same experience
    // whenever it is indexed.
    if let Some(started) = &trial.started_at {
        provenance.created_at.clone_from(started);
    } else {
        provenance.created_at = format!("slot {} attempt {}", trial.slot, trial.attempt);
    }
    provenance.attempt = Some(trial.attempt);
    Experience::new(task, trajectory, None, provenance).map_err(|e| e.to_string())
}

/// Index every committed rollout product of the factory run in `run`.
///
/// # Errors
/// The run's products cannot be read, or the store refuses a write.
pub fn index_run(ctx: &Context, run: &Path) -> Result<Vec<IndexedPass>, OrchestratorError> {
    splinter_factory::run::require_run(run).map_err(refused)?;
    let products = ProductStore::new(run, Box::new(SystemClock));
    let store = ctx.experiences();
    let mut out = Vec::new();
    for manifest in products.list().map_err(refused)? {
        let is_rollout =
            manifest.name.starts_with("rollouts/") || manifest.name.starts_with("eval/_arms/");
        if !is_rollout {
            continue;
        }
        let view = run.join(&manifest.name);
        let trials = trials_of(&view).map_err(refused)?;
        let (mut members, mut verdicts, mut skipped) = (Vec::new(), BTreeMap::new(), Vec::new());
        for trial in &trials {
            let experience = match experience(&view, trial) {
                Ok(experience) => experience,
                Err(why) => {
                    skipped.push((trial.slot.to_string(), why));
                    continue;
                }
            };
            let id = store.put(&experience)?;
            let verdict = outcome(trial);
            store.annotate(&Annotation {
                experience: id.clone(),
                producer: Producer { name: PRODUCER.into(), version: "1".into() },
                body: AnnotationBody::Verdict {
                    outcome: verdict,
                    strength: Strength::Executable,
                    evidence: json!({ "grade": trial.grade, "slot": trial.slot, "guided": trial.guided }),
                },
            })?;
            *verdicts
                .entry(format!("{verdict:?}").to_lowercase())
                .or_default() += 1;
            if !members.contains(&id) {
                members.push(id);
            }
        }
        let experience_set = if members.is_empty() {
            None
        } else {
            Some(store.put_set(&ExperienceSet {
                name: format!("factory trials of {}", manifest.name),
                members,
            })?)
        };
        out.push(IndexedPass {
            pass: manifest.name.clone(),
            experience_set,
            verdicts,
            skipped,
        });
    }
    Ok(out)
}
