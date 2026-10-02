// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements durable, content-addressed experience
// stores for learning agents, for its clients. If your team needs expertise
// in training-data lineage or crash-safe storage, you can procure our
// services by sending an email to info@swedishembedded.com.

//! An experience as the graph the experience database reads.
//!
//! The experience itself is stored whole, so it reads back exactly as it was
//! written. Beside it, each experience is an attempt at its task: a decision
//! for every agent step, with what the agent was shown before it. The attempt
//! is what the database's recipes select, pair and group, and what evidence
//! about the experience is recorded against. A small `projection` entity
//! names the attempt and the decisions of an experience, so evidence can find
//! them again.
//!
//! The graph is derived from the experience and says nothing the experience
//! does not.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::json;
use splinter_expdb::ingest::Collector;
use splinter_expdb::model::{Action, PolicyRef, ReproLevel, State, TaskDefinition, TaskInstance};
use splinter_expdb::{ContentId, RecordId};
use sven_sdk::atif::StepOrigin;

use crate::digest::Digest;
use crate::error::StoreError;
use crate::experience::{Experience, ExperienceId};
use crate::workspace::Workspace;

/// The class of the entity that names an experience's attempt.
pub(crate) const PROJECTION: &str = "projection";

/// Where an experience sits in the graph.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Projected {
    /// The experience this describes.
    pub(crate) experience: ExperienceId,
    /// Its attempt record.
    pub(crate) attempt: RecordId,
    /// The decision record of each agent step, by the step's ATIF id.
    pub(crate) decisions: BTreeMap<u64, RecordId>,
}

/// The address of the projection entity of `id`.
pub(crate) fn projection_address(id: &ExperienceId) -> Digest {
    let mut bytes = PROJECTION.as_bytes().to_vec();
    bytes.push(0);
    bytes.extend_from_slice(id.as_str().as_bytes());
    Digest::of(&bytes)
}

/// Records `experience` as an attempt in the collector's graph.
pub(crate) fn project(
    collector: &mut Collector,
    id: &ExperienceId,
    experience: &Experience,
) -> splinter_expdb::Result<Projected> {
    let steps = &experience.trajectory.steps;
    // Contents first: a large one goes to the blob store, which the run below
    // borrows the collector away from.
    let mut shown = Vec::with_capacity(steps.len());
    for step in steps {
        let text =
            serde_json::to_vec(&step.message).map_err(|source| splinter_expdb::Error::Encode {
                what: "step message",
                source,
            })?;
        let observation = match &step.observation {
            Some(observation) => Some(collector.content(
                &serde_json::to_vec(observation).map_err(|source| {
                    splinter_expdb::Error::Encode {
                        what: "step observation",
                        source,
                    }
                })?,
            )?),
            None => None,
        };
        shown.push((collector.content(&text)?, observation));
    }

    let task = &experience.task;
    let definition = TaskDefinition {
        name: task.kind.clone(),
        description: format!("{} tasks", task.kind),
        domain: task.kind.clone(),
    };
    let instance = TaskInstance {
        definition: definition.id()?,
        params: json!({ "task": task.id }),
        environment: None,
    };
    let address = task
        .id
        .content_id()
        .unwrap_or_else(|| ContentId::of(task.id.as_str().as_bytes()));
    let initial = State::new(
        BTreeMap::from([("task".to_owned(), address)]),
        ReproLevel::Approximate,
    );
    let provenance = &experience.provenance;
    let policy = PolicyRef::new(
        &provenance.solver,
        provenance.policy.as_deref().unwrap_or("-"),
    );
    let mut run = collector.start_attempt(
        &definition,
        &instance,
        &initial,
        &policy,
        provenance.attempt.map(u64::from),
    )?;
    let attempt = run.attempt();

    let mut decisions = BTreeMap::new();
    let mut pending = None;
    for (step, (message, observation)) in steps.iter().zip(shown) {
        match step.source {
            StepOrigin::System | StepOrigin::User => {
                pending = Some(run.observe(message)?);
            }
            StepOrigin::Agent => {
                let mut decision = run.decision();
                if let Some(seen) = pending.take() {
                    decision = decision.observation(seen);
                }
                let name = step
                    .tool_calls
                    .as_ref()
                    .and_then(|calls| calls.first())
                    .map_or_else(|| "message".to_owned(), |call| call.function_name.clone());
                // What the step said and called, not the whole step: the
                // experience holds that, once, and what a decision needs is
                // enough to tell it from another and to find it again.
                let arguments = serde_json::to_value(json!({
                    "step_id": step.step_id,
                    "message": message,
                    "tool_calls": step.tool_calls,
                }))
                .map_err(|source| splinter_expdb::Error::Encode {
                    what: "step",
                    source,
                })?;
                let made = decision.commit(Action::new(&name, arguments))?;
                decisions.insert(step.step_id, made.id);
                if let Some(content) = observation {
                    pending = Some(run.observe(content)?);
                }
            }
        }
    }
    Ok(Projected {
        experience: id.clone(),
        attempt,
        decisions,
    })
}

/// Where the graph holds `id`.
pub(crate) fn projection_of(
    workspace: &Workspace,
    id: &ExperienceId,
) -> Result<Projected, StoreError> {
    let entity = workspace
        .find(PROJECTION, &projection_address(id))?
        .ok_or_else(|| StoreError::UnknownExperience(id.clone()))?;
    serde_json::from_value(entity.value).map_err(|e| StoreError::UndecodableObject {
        what: format!("the graph of {id}"),
        reason: e.to_string(),
    })
}
