// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What the experience store holds, for a person to inspect: the sets
//! stages hand each other, an experience with everything said about it,
//! the graph of relations - retries, critiques, preferences - around one,
//! and whether an experience's code calls replay to what it observed
//! ([`replay`]).

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::Serialize;
use splinter_core::annotation::{Annotation, AnnotationBody, RelationKind, Strength};
use splinter_core::experience::{Experience, ExperienceId};
use splinter_store::decision::decide;
use splinter_store::experiences::SetId;

use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_orchestrator::ids;

pub use splinter_agent::replay::{CallReplay, ReplayedCall};

/// The most experiences a relation graph shows.
pub const MAX_GRAPH_NODES: usize = 500;

/// One experience set, as `experiences list` shows it.
#[derive(Clone, Debug, Serialize)]
pub struct SetSummary {
    /// Its id.
    pub id: SetId,
    /// Its name.
    pub name: String,
    /// Experiences in it.
    pub experiences: usize,
}

/// What `experiences list` reports.
#[derive(Clone, Debug, Serialize)]
pub struct SetList {
    /// Every stored experience set, in id order.
    pub experience_sets: Vec<SetSummary>,
}

/// Every stored experience set.
pub fn list(ctx: &Context) -> Result<SetList, OrchestratorError> {
    let store = ctx.experiences();
    let experience_sets = store
        .list_sets()?
        .into_iter()
        .map(|id| {
            let set = store.get_set(&id)?;
            Ok(SetSummary {
                experiences: set.members.len(),
                name: set.name,
                id,
            })
        })
        .collect::<Result<_, OrchestratorError>>()?;
    Ok(SetList { experience_sets })
}

/// Where an experience's verdicts leave it.
#[derive(Clone, Debug, Serialize)]
pub struct DecisionView {
    /// `pass`, `fail`, or `undecided`.
    pub decision: &'static str,
    /// The strength of the verdicts that decided it.
    pub strength: Option<Strength>,
}

fn decision_of(notes: &[Annotation]) -> DecisionView {
    match decide(notes) {
        Some(d) => DecisionView {
            decision: if d.passed { "pass" } else { "fail" },
            strength: Some(d.strength),
        },
        None => DecisionView {
            decision: "undecided",
            strength: None,
        },
    }
}

/// One experience, as a set lists it or a graph shows it.
#[derive(Clone, Debug, Serialize)]
pub struct ExperienceLine {
    /// Its id.
    pub id: ExperienceId,
    /// Its task's kind.
    pub kind: String,
    /// The model that produced it.
    pub solver: String,
    /// Where its verdicts leave it.
    #[serde(flatten)]
    pub decision: DecisionView,
}

fn line(id: &ExperienceId, experience: &Experience, notes: &[Annotation]) -> ExperienceLine {
    ExperienceLine {
        id: id.clone(),
        kind: experience.task.kind.clone(),
        solver: experience.provenance.solver.clone(),
        decision: decision_of(notes),
    }
}

/// One relation between two experiences: `from <relation> to`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct Edge {
    /// The experience the relation is recorded on.
    pub from: ExperienceId,
    /// The relation, read as "from <relation> to".
    pub relation: RelationKind,
    /// The other experience.
    pub to: ExperienceId,
}

/// What `experiences show` reports.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "shows", rename_all = "snake_case")]
pub enum ExperienceShow {
    /// An experience set and its experiences.
    Set {
        /// Its id.
        id: SetId,
        /// Its name.
        name: String,
        /// Its experiences, in order.
        experiences: Vec<ExperienceLine>,
    },
    /// One experience, whole, with its annotations.
    Experience {
        /// Its id.
        id: ExperienceId,
        /// Where its verdicts leave it.
        decision: DecisionView,
        /// The experience.
        experience: Box<Experience>,
        /// Everything said about it, in append order.
        annotations: Vec<Annotation>,
    },
    /// The relations around a set's experiences or one experience.
    Graph {
        /// The experiences, in id order.
        nodes: Vec<ExperienceLine>,
        /// The relations between them.
        edges: Vec<Edge>,
        /// Whether the graph was cut at [`MAX_GRAPH_NODES`].
        truncated: bool,
    },
}

/// The experience set or experience `id` names; with `graph`, the
/// relations around it.
pub fn show(ctx: &Context, id: &str, graph: bool) -> Result<ExperienceShow, OrchestratorError> {
    let store = ctx.experiences();
    let (start, set) = match resolve_set(ctx, id) {
        Ok(set_id) => {
            let set = store.get_set(&set_id)?;
            (set.members.clone(), Some((set_id, set)))
        }
        Err(OrchestratorError::NotFound { .. }) => (vec![resolve_experience(ctx, id)?], None),
        Err(e) => return Err(e),
    };
    if graph {
        return relation_graph(ctx, start);
    }
    if let Some((set_id, set)) = set {
        let experiences = set
            .members
            .iter()
            .map(|member| {
                let experience = store.get(member)?;
                let notes = store.annotations(member)?.annotations;
                Ok(line(member, &experience, &notes))
            })
            .collect::<Result<_, OrchestratorError>>()?;
        return Ok(ExperienceShow::Set {
            id: set_id,
            name: set.name,
            experiences,
        });
    }
    let id = start
        .into_iter()
        .next()
        .ok_or_else(|| OrchestratorError::NotFound {
            what: "experience",
            id: id.to_string(),
        })?;
    let annotations = store.annotations(&id)?.annotations;
    Ok(ExperienceShow::Experience {
        decision: decision_of(&annotations),
        experience: Box::new(store.get(&id)?),
        annotations,
        id,
    })
}

/// The experiences related to `start`, directly or through others, in
/// either direction, and the relations between them.
fn relation_graph(
    ctx: &Context,
    start: Vec<ExperienceId>,
) -> Result<ExperienceShow, OrchestratorError> {
    let store = ctx.experiences();
    // Relations are recorded on the experience they start from; finding
    // the ones that end at an experience means reading every log once.
    let mut edges_of: BTreeMap<ExperienceId, BTreeSet<Edge>> = BTreeMap::new();
    for id in store.list()? {
        for note in store.annotations(&id)?.annotations {
            if let AnnotationBody::Relation { kind, other } = note.body {
                let edge = Edge {
                    from: id.clone(),
                    relation: kind,
                    to: other.clone(),
                };
                edges_of.entry(id.clone()).or_default().insert(edge.clone());
                edges_of.entry(other).or_default().insert(edge);
            }
        }
    }
    let mut seen: BTreeSet<ExperienceId> = BTreeSet::new();
    let mut queue: VecDeque<ExperienceId> = start.into_iter().collect();
    let mut edges = BTreeSet::new();
    let mut truncated = false;
    while let Some(id) = queue.pop_front() {
        if seen.contains(&id) {
            continue;
        }
        if seen.len() >= MAX_GRAPH_NODES {
            truncated = true;
            break;
        }
        seen.insert(id.clone());
        for edge in edges_of.get(&id).into_iter().flatten() {
            edges.insert(edge.clone());
            for next in [&edge.from, &edge.to] {
                if !seen.contains(next) {
                    queue.push_back(next.clone());
                }
            }
        }
    }
    let nodes = seen
        .iter()
        .map(|id| {
            let experience = store.get(id)?;
            let notes = store.annotations(id)?.annotations;
            Ok(line(id, &experience, &notes))
        })
        .collect::<Result<_, OrchestratorError>>()?;
    Ok(ExperienceShow::Graph {
        nodes,
        edges: edges
            .into_iter()
            .filter(|e| seen.contains(&e.from) && seen.contains(&e.to))
            .collect(),
        truncated,
    })
}

/// The stored experience set `id` (or a unique prefix of it) names.
pub fn resolve_set(ctx: &Context, id: &str) -> Result<SetId, OrchestratorError> {
    let stored = ctx.experiences().list_sets()?.into_iter().map(|s| s.0);
    Ok(SetId(ids::resolve("experience set", id, stored)?))
}

/// The stored experience `id` (or a unique prefix of it) names.
pub fn resolve_experience(ctx: &Context, id: &str) -> Result<ExperienceId, OrchestratorError> {
    let stored = ctx.experiences().list()?.into_iter().map(|e| e.0);
    Ok(ExperienceId(ids::resolve("experience", id, stored)?))
}

/// One experience replayed.
#[derive(Clone, Debug, Serialize)]
pub struct ExperienceReplay {
    /// The experience.
    pub id: ExperienceId,
    /// The environment it was replayed in, by kind.
    pub environment: String,
    /// Whether no call diverged.
    pub reproduced: bool,
    /// Its code calls, in order.
    pub calls: Vec<ReplayedCall>,
}

/// What `experiences replay` reports.
#[derive(Clone, Debug, Serialize)]
pub struct Replayed {
    /// Whether every experience reproduced.
    pub reproduced: bool,
    /// Each experience, in the order named.
    pub experiences: Vec<ExperienceReplay>,
}

/// Replays the code calls of the experience `id` names, or of every member
/// of the experience set it names, each in the environment it records
/// ([`splinter_agent::replay`]). An environment that no longer resolves to
/// the recorded one is refused.
pub fn replay(ctx: &Context, id: &str) -> Result<Replayed, OrchestratorError> {
    let store = ctx.experiences();
    let members = match resolve_set(ctx, id) {
        Ok(set) => store.get_set(&set)?.members,
        Err(OrchestratorError::NotFound { .. }) => vec![resolve_experience(ctx, id)?],
        Err(e) => return Err(e),
    };
    let mut experiences = Vec::with_capacity(members.len());
    for member in members {
        let experience = store.get(&member)?;
        let environment = ctx.environments().for_record(&experience.environment)?;
        let replay = splinter_agent::replay::replay(&experience, &environment)?;
        experiences.push(ExperienceReplay {
            id: member,
            environment: experience.environment.kind.clone(),
            reproduced: replay.reproduced(),
            calls: replay.calls,
        });
    }
    Ok(Replayed {
        reproduced: experiences.iter().all(|e| e.reproduced),
        experiences,
    })
}
