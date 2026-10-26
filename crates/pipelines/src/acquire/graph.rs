// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that turn the documentation of an
// unfamiliar technical domain into a graph of skills they then acquire, for
// its clients. If your team needs expertise in building curricula from
// technical documentation you can procure our services by sending an email
// to info@swedishembedded.com.

//! The skill graph of an acquisition: planned from the sources, kept
//! versioned.
//!
//! The planner role (the generator when no planner is named) is shown the
//! sections of the sources a window at a time and proposes skills; what the
//! sections bear out is kept ([`splinter_knowledge::skills::admit`]). Where
//! no model proposes anything that survives - none is configured, or what it
//! said was refused - the graph is read from the sources by rule
//! ([`splinter_knowledge::skills::structural`]). Either way the graph is
//! validated, then kept as a content-addressed document and named by a
//! pointer that moves, compare-and-set, to each revision; planning again over
//! the same sources with a different result is a new revision naming the
//! one before.

use std::time::Instant;

use serde::Serialize;
use splinter_agent::skill_planner::{propose_skills, SkillPlanRequest};
use splinter_agent::CancelToken;
use splinter_core::digest::Digest;
use splinter_core::source::SourceId;
use splinter_knowledge::skills::{admit, section_index, structural, Proposed, Refusal, SkillGraph};
use splinter_store::error::StoreError;

use crate::tasks::remaining;
use splinter_core::model_ref::ModelRef;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// Sections shown to the planner in one request.
pub const SECTIONS_PER_PLAN_REQUEST: usize = 12;

/// How long one planner request may take.
pub const PLAN_REQUEST_DEADLINE: std::time::Duration = std::time::Duration::from_secs(600);

const GRAPH: &str = "skill_graph";

/// How the graph came to be.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "by", rename_all = "snake_case")]
pub enum PlannedBy {
    /// A planner model proposed it and the sources bore it out.
    Planner {
        /// The model's identity.
        model: String,
        /// Requests made.
        requests: usize,
        /// Requests with no usable reply.
        unanswered: usize,
    },
    /// Read from the sources by rule.
    Structure {
        /// Why the planner's graph was not used, when one was asked for.
        reason: Option<String>,
    },
    /// Taken from an earlier acquisition run's record, not planned.
    Reused {
        /// The run whose graph this is.
        run: String,
    },
}

/// What planning made.
#[derive(Clone, Debug, Serialize)]
pub struct Planned {
    /// The graph.
    pub graph: SkillGraph,
    /// Its address.
    pub address: Digest,
    /// How it came to be.
    pub by: PlannedBy,
    /// What admission refused of a planner's proposal, as `(skill, why)`.
    pub refused: Vec<(String, Refusal)>,
}

fn store_error(e: StoreError) -> OrchestratorError {
    e.into()
}

/// The graph of `sources` for `objective`; see the module documentation.
pub fn plan(
    ctx: &Context,
    objective: &str,
    sources: &[SourceId],
    planner: Option<&ModelRef>,
    deadline: Option<Instant>,
    cancel: &CancelToken,
) -> Result<Planned, OrchestratorError> {
    let index = section_index(&ctx.sources(), sources).map_err(store_error)?;
    let mut by = None;
    let mut refused = Vec::new();
    let mut graph = None;
    if let Some(planner) = planner {
        let model = ctx.model(planner)?;
        let mut proposed = Proposed::default();
        let (mut requests, mut unanswered) = (0, 0);
        for window in index
            .iter()
            .enumerate()
            .collect::<Vec<_>>()
            .chunks(SECTIONS_PER_PLAN_REQUEST)
        {
            if cancel.is_cancelled() {
                return Err(OrchestratorError::Cancelled);
            }
            if deadline.is_some_and(|d| Instant::now() >= d) {
                break;
            }
            let shown: Vec<(usize, String)> = window
                .iter()
                .map(|(n, (_, body))| (*n, body.clone()))
                .collect();
            requests += 1;
            let asked = ctx.block_on(propose_skills(
                &model,
                &SkillPlanRequest {
                    objective,
                    sections: &shown,
                    deadline: remaining(deadline, PLAN_REQUEST_DEADLINE),
                    max_output_tokens: None,
                    cancel: Some(cancel.clone()),
                },
            ));
            match asked {
                Ok(reply) => proposed.skills.extend(reply.skills),
                Err(_) => unanswered += 1,
            }
        }
        let admitted = admit(&proposed, &index, objective, sources);
        refused = admitted.refused;
        if admitted.graph.skills.is_empty() || admitted.graph.validate().is_err() {
            by = Some(PlannedBy::Structure {
                reason: Some(format!(
                    "the planner's proposal left no skill the sections bear out ({requests} \
                     request(s), {unanswered} unanswered, {} refusal(s))",
                    refused.len()
                )),
            });
        } else {
            by = Some(PlannedBy::Planner {
                model: model.identity.clone(),
                requests,
                unanswered,
            });
            graph = Some(admitted.graph);
        }
    }
    let graph = match graph {
        Some(graph) => graph,
        None => structural(&ctx.sources(), sources, objective).map_err(store_error)?,
    };
    graph
        .validate()
        .map_err(|e| OrchestratorError::Refused(format!("the skill graph is not valid: {e}")))?;
    if graph.skills.is_empty() {
        return Err(OrchestratorError::Refused(
            "the sources state no skill: no section has a heading and a body to learn from".into(),
        ));
    }
    let (graph, address) = keep(ctx, graph)?;
    Ok(Planned {
        graph,
        address,
        by: by.unwrap_or(PlannedBy::Structure { reason: None }),
        refused,
    })
}

/// The pointer a graph for `objective` is named by.
fn pointer_of(objective: &str) -> String {
    format!(
        "acquire-graph-{}",
        &Digest::of(objective.trim().as_bytes()).hex()[..24]
    )
}

/// Keeps `graph` as the newest revision for its objective: the same graph
/// again is the revision already kept; a different one names it as its
/// parent. The graph as kept and its address.
fn keep(ctx: &Context, mut graph: SkillGraph) -> Result<(SkillGraph, Digest), OrchestratorError> {
    let pointer = pointer_of(&graph.objective);
    let encode = |graph: &SkillGraph| -> Result<Digest, OrchestratorError> {
        let stored = ctx.workspace().put_document(GRAPH, graph)?;
        Digest::parse(stored.as_str()).map_err(|e| {
            OrchestratorError::Refused(format!("the skill graph was kept unreadably: {e}"))
        })
    };
    let current = ctx.workspace().pointer(&pointer)?.map(|(_, value)| value);
    if let Some(value) = &current {
        let previous = Digest::parse(value)
            .map_err(|e| OrchestratorError::Refused(format!("{pointer} is corrupt: {e}")))?;
        if let Some(kept) = ctx
            .workspace()
            .get_document::<SkillGraph>(GRAPH, &previous)?
        {
            let mut same = graph.clone();
            same.revision = kept.revision;
            same.parent.clone_from(&kept.parent);
            if same == kept {
                return Ok((kept, previous));
            }
            graph.revision = kept.revision + 1;
            graph.parent = Some(previous);
        }
    }
    let address = encode(&graph)?;
    ctx.workspace().move_pointer(
        &pointer,
        current.as_deref(),
        address.as_str(),
        &ctx.clock().utc_now(),
    )?;
    Ok((graph, address))
}
