// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The lineage graph, derived from what each store records: every stored
//! artifact becomes a node, and every reference one record makes to
//! another an edge (the table in the parent module). Nothing here writes.
//!
//! A reference to an artifact whose record is not stored (a candidate
//! directory removed, a span into content never captured here) still
//! becomes a node, labelled as missing, so the walk shows the gap instead
//! of hiding it. A record that is stored but fails its verification is an
//! error: lineage over a corrupt store is not reported as if it were whole.

use splinter_core::annotation::{AnnotationBody, Outcome, RelationKind};
use splinter_core::experience::{Environment, Experience, Span, Task};
use splinter_core::source::Origin;
use splinter_store::experiences::StoreError;
use splinter_store::sources::SourceStore;

use crate::train::{candidate_ids, load_candidate};
use splinter_core::training::ReplaySample;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_orchestrator::releases::StoredRelease;

use super::graph::Graph;
use super::{NodeKind, Relation};

/// The most characters of a text a label quotes.
const EXCERPT_CHARS: usize = 100;

/// The graph of everything under `ctx`'s state root.
pub(super) fn load(ctx: &Context) -> Result<Graph, OrchestratorError> {
    let mut graph = Graph::default();
    sources(ctx, &mut graph)?;
    tasks(ctx, &mut graph)?;
    experiences(ctx, &mut graph)?;
    datasets(ctx, &mut graph)?;
    candidates(ctx, &mut graph)?;
    releases(ctx, &mut graph)?;
    answers(ctx, &mut graph)?;
    Ok(graph)
}

/// `text` quoted on one line, cut after [`EXCERPT_CHARS`] characters.
fn excerpt(text: &str) -> String {
    let mut chars = text.chars();
    let head: String = chars.by_ref().take(EXCERPT_CHARS).collect();
    if chars.next().is_some() {
        format!("{head:?}...")
    } else {
        format!("{head:?}")
    }
}

fn missing(kind: &str) -> impl FnOnce() -> String + '_ {
    move || format!("{kind} (no record of it under the state root)")
}

fn sources(ctx: &Context, graph: &mut Graph) -> Result<(), OrchestratorError> {
    let store = ctx.sources();
    for id in store.list()? {
        let source = store.get_source(&id)?;
        let origin = match &source.origin {
            Origin::Document { path } | Origin::Repository { path, .. } => path.clone(),
            Origin::Command { argv, .. } => argv.join(" "),
        };
        graph.record(
            id.as_str(),
            NodeKind::Source,
            format!("{} {origin}", source.kind()),
        );
        for part in &source.parts {
            graph.mention(part.content.as_str(), NodeKind::Content, || {
                format!(
                    "part {:?}: {} bytes of {}",
                    part.name, part.bytes, part.media_type
                )
            });
            graph.link(part.content.as_str(), Relation::PartOf, id.as_str());
        }
    }
    Ok(())
}

/// Adds the span node for `span` and its edge to the source part or the
/// content it indexes; returns its id.
fn span(store: &SourceStore, graph: &mut Graph, span: &Span) -> Result<String, OrchestratorError> {
    let id = format!("span:{}:{}-{}", span.source.hex(), span.start, span.end);
    if graph.contains(&id) {
        return Ok(id);
    }
    let text = match store.read_span(span) {
        Ok(bytes) => excerpt(&String::from_utf8_lossy(&bytes)),
        Err(
            e @ (StoreError::UnknownBlob(_)
            | StoreError::UnknownSource(_)
            | StoreError::UnknownPart { .. }
            | StoreError::SpanPart { .. }
            | StoreError::SpanOutOfRange { .. }),
        ) => format!("(its bytes do not resolve: {e})"),
        Err(e) => return Err(e.into()),
    };
    let range = format!("[{}, {})", span.start, span.end);
    match &span.part {
        Some(part) => {
            graph.record(
                &id,
                NodeKind::Span,
                format!("{} {range}: {text}", part.name),
            );
            graph.mention(part.source.as_str(), NodeKind::Source, missing("source"));
            graph.link(&id, Relation::SpanOf, part.source.as_str());
        }
        None => {
            graph.record(&id, NodeKind::Span, format!("content {range}: {text}"));
            graph.mention(span.source.as_str(), NodeKind::Content, missing("content"));
            graph.link(&id, Relation::SpanOf, span.source.as_str());
        }
    }
    Ok(id)
}

/// Adds `task` and its evidence.
fn task(store: &SourceStore, graph: &mut Graph, task: &Task) -> Result<(), OrchestratorError> {
    let id = task.task.id.as_str();
    graph.record(
        id,
        NodeKind::Task,
        format!("{}: {}", task.task.kind, excerpt(&task.instruction)),
    );
    for evidence in &task.evidence {
        let span = span(store, graph, evidence)?;
        graph.link(id, Relation::Evidence, &span);
    }
    Ok(())
}

fn tasks(ctx: &Context, graph: &mut Graph) -> Result<(), OrchestratorError> {
    let sources = ctx.sources();
    let store = ctx.tasks();
    for id in store.list()? {
        task(&sources, graph, &store.get(&id)?)?;
    }
    for id in store.list_sets()? {
        let set = store.get_set(&id)?;
        let set_id = id.0.as_str();
        graph.record(
            set_id,
            NodeKind::TaskSet,
            format!("task set {:?}, {} task(s)", set.name, set.members.len()),
        );
        for member in &set.members {
            graph.mention(member.task.as_str(), NodeKind::Task, missing("task"));
            graph.link(set_id, Relation::Member, member.task.as_str());
        }
    }
    Ok(())
}

fn relation(kind: RelationKind) -> Relation {
    match kind {
        RelationKind::PreferredOver => Relation::PreferredOver,
        RelationKind::RetryOf => Relation::RetryOf,
        RelationKind::CritiqueOf => Relation::CritiqueOf,
        RelationKind::RevisionOf => Relation::RevisionOf,
        RelationKind::VariantOf => Relation::VariantOf,
    }
}

fn experience_label(experience: &Experience) -> String {
    let answer = experience
        .final_output
        .as_deref()
        .map_or("no answer".to_string(), excerpt);
    format!("{} answered {answer}", experience.provenance.solver)
}

fn experiences(ctx: &Context, graph: &mut Graph) -> Result<(), OrchestratorError> {
    let sources = ctx.sources();
    let store = ctx.experiences();
    for id in store.list()? {
        let experience = store.get(&id)?;
        let exp = id.as_str();
        graph.record(exp, NodeKind::Experience, experience_label(&experience));
        let attempted = experience.task.id.as_str();
        if !graph.contains(attempted) {
            task(&sources, graph, &experience.to_task())?;
        }
        graph.link(exp, Relation::Attempts, attempted);
        let environment = &experience.environment;
        let snapshot = environment
            .snapshot
            .clone()
            .unwrap_or_else(|| Environment::snapshot_of(&environment.kind, &environment.spec));
        graph.mention(snapshot.as_str(), NodeKind::Environment, || {
            format!("environment {}", environment.kind)
        });
        graph.link(exp, Relation::RanIn, snapshot.as_str());
        let solver = format!("model:{}", experience.provenance.solver);
        graph.mention(&solver, NodeKind::Model, || {
            experience.provenance.solver.clone()
        });
        graph.link(exp, Relation::SolvedBy, &solver);
        for (index, note) in store.annotations(&id)?.annotations.iter().enumerate() {
            match &note.body {
                AnnotationBody::Verdict {
                    outcome, strength, ..
                } => {
                    let verdict = format!("verdict:{}:{index}", id.hex());
                    let outcome = match outcome {
                        Outcome::Pass => "pass",
                        Outcome::Fail => "fail",
                        Outcome::Abstain => "abstain",
                    };
                    let producer =
                        format!("producer:{}@{}", note.producer.name, note.producer.version);
                    graph.record(
                        &verdict,
                        NodeKind::Verdict,
                        format!("{outcome} ({strength:?}) by {}", note.producer.name),
                    );
                    graph.link(&verdict, Relation::VerdictOn, exp);
                    graph.mention(&producer, NodeKind::Producer, || {
                        format!("{} {}", note.producer.name, note.producer.version)
                    });
                    graph.link(&verdict, Relation::ProducedBy, &producer);
                }
                AnnotationBody::Relation { kind, other } => {
                    graph.mention(other.as_str(), NodeKind::Experience, missing("experience"));
                    graph.link(exp, relation(*kind), other.as_str());
                }
                AnnotationBody::StepLabel { .. } => {}
            }
        }
    }
    for id in store.list_sets()? {
        let set = store.get_set(&id)?;
        let set_id = id.0.as_str();
        graph.record(
            set_id,
            NodeKind::ExperienceSet,
            format!(
                "experience set {:?}, {} experience(s)",
                set.name,
                set.members.len()
            ),
        );
        for member in &set.members {
            graph.mention(member.as_str(), NodeKind::Experience, missing("experience"));
            graph.link(set_id, Relation::Member, member.as_str());
        }
    }
    Ok(())
}

fn datasets(ctx: &Context, graph: &mut Graph) -> Result<(), OrchestratorError> {
    let store = ctx.datasets();
    for id in store.list()? {
        let manifest = store.get(&id)?.manifest;
        let dataset = id.0.as_str();
        graph.record(
            dataset,
            NodeKind::Dataset,
            format!(
                "{} dataset, {} record(s)",
                manifest.view, manifest.counts.records
            ),
        );
        for experience in &manifest.experiences {
            graph.mention(
                experience.as_str(),
                NodeKind::Experience,
                missing("experience"),
            );
            graph.link(dataset, Relation::ProjectedFrom, experience.as_str());
        }
        for task in &manifest.tasks {
            graph.mention(task.as_str(), NodeKind::Task, missing("task"));
            graph.link(dataset, Relation::ProjectedFrom, task.as_str());
        }
        for content in &manifest.sources {
            graph.mention(content.as_str(), NodeKind::Content, missing("content"));
            graph.link(dataset, Relation::ProjectedFrom, content.as_str());
        }
    }
    Ok(())
}

/// Adds `replay`'s node and its edges; returns its id, `None` when nothing
/// was drawn (then there is no replay file to name).
fn replay(graph: &mut Graph, replay: &ReplaySample) -> Option<String> {
    let id = replay.digest.as_ref()?.as_str().to_string();
    graph.record(
        &id,
        NodeKind::Replay,
        format!(
            "{} replayed record(s), {} of each earlier release, seed {}",
            replay.records, replay.fraction, replay.seed
        ),
    );
    for source in &replay.sources {
        graph.mention(
            source.release.0.as_str(),
            NodeKind::Release,
            missing("release"),
        );
        graph.link(&id, Relation::SampledFrom, source.release.0.as_str());
    }
    Some(id)
}

/// The edges a candidate and its release share: datasets, parent, replay
/// and adapter.
fn training(
    graph: &mut Graph,
    id: &str,
    datasets: &[splinter_core::dataset::DatasetId],
    parent: Option<&splinter_core::release::ReleaseId>,
    sample: Option<&ReplaySample>,
    adapter: &str,
) {
    for dataset in datasets {
        graph.mention(dataset.0.as_str(), NodeKind::Dataset, missing("dataset"));
        graph.link(id, Relation::TrainedOn, dataset.0.as_str());
    }
    if let Some(parent) = parent {
        graph.mention(parent.0.as_str(), NodeKind::Release, missing("release"));
        graph.link(id, Relation::TrainedFrom, parent.0.as_str());
    }
    if let Some(sample) = sample.and_then(|s| replay(graph, s)) {
        graph.link(id, Relation::Replayed, &sample);
    }
    graph.mention(adapter, NodeKind::Adapter, || "adapter".into());
    graph.link(id, Relation::Adapter, adapter);
}

fn candidates(ctx: &Context, graph: &mut Graph) -> Result<(), OrchestratorError> {
    for id in candidate_ids(ctx)? {
        let candidate = load_candidate(ctx, &id)?;
        graph.record(
            &id,
            NodeKind::Candidate,
            format!(
                "candidate trained from {} on {} record(s)",
                candidate.from, candidate.records
            ),
        );
        training(
            graph,
            &id,
            &candidate.datasets,
            candidate.parent.as_ref(),
            candidate.replay.as_ref(),
            &candidate.adapter_digest,
        );
    }
    Ok(())
}

fn releases(ctx: &Context, graph: &mut Graph) -> Result<(), OrchestratorError> {
    let store = ctx.releases();
    let aliases = store.aliases()?;
    for id in store.list()? {
        let StoredRelease { manifest, .. } = store.get(&id)?;
        let release = id.0.as_str();
        let named: Vec<&str> = aliases
            .iter()
            .filter(|(_, target)| **target == id)
            .map(|(name, _)| name.as_str())
            .collect();
        let aliased = if named.is_empty() {
            String::new()
        } else {
            format!(" [{}]", named.join(", "))
        };
        graph.record(
            release,
            NodeKind::Release,
            format!(
                "release of {}, {}{aliased}",
                manifest.candidate, manifest.created_at
            ),
        );
        graph.mention(
            &manifest.candidate,
            NodeKind::Candidate,
            missing("candidate"),
        );
        graph.link(release, Relation::ReleaseOf, &manifest.candidate);
        training(
            graph,
            release,
            &manifest.datasets,
            manifest.parent.as_ref(),
            manifest.replay.as_ref(),
            manifest.adapter_digest.as_str(),
        );
    }
    Ok(())
}

fn answers(ctx: &Context, graph: &mut Graph) -> Result<(), OrchestratorError> {
    let store = ctx.answers();
    for id in store.list()? {
        let record = store.get(&id)?;
        let answer = id.as_str();
        graph.record(
            answer,
            NodeKind::Answer,
            format!(
                "{} answered {}",
                excerpt(&record.question),
                excerpt(&record.answer)
            ),
        );
        if let Some(release) = &record.release {
            graph.mention(release.0.as_str(), NodeKind::Release, missing("release"));
            graph.link(answer, Relation::AnsweredWith, release.0.as_str());
        }
        let model = format!("model:{}", record.model);
        graph.mention(&model, NodeKind::Model, || record.model.clone());
        graph.link(answer, Relation::AnsweredBy, &model);
        if let Some(source) = &record.open_book {
            graph.mention(source.as_str(), NodeKind::Source, missing("source"));
            graph.link(answer, Relation::OpenBook, source.as_str());
        }
    }
    Ok(())
}
