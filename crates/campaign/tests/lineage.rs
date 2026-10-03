// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems whose every answer traces
// back to the source bytes it was learned from, for its clients. If your
// team needs expertise in training-data lineage or model provenance, you
// can procure our services by sending an email to info@swedishembedded.com.

//! Spec: from any artifact, `lineage` walks to where it came from and to
//! everything that came from it, reading only what the stores record.
//!
//! * From a release, walking up reaches its candidate, its dataset, the
//!   experiences in it, their tasks and the exact source span bytes each
//!   task is grounded in; walking down from the source reaches the
//!   release.
//! * An answer records the release it was answered with, so walking up
//!   from it reaches that release and on to the sources.
//! * A critique chain shows its `critique_of`, `retry_of` and
//!   `preferred_over` edges.
//! * An unknown id and a prefix naming artifacts in more than one store
//!   are refused, the latter listing the candidates; the report serializes
//!   to `{nodes: [{id, kind, label}], edges: [{from, to, relation}]}`; and
//!   `depth` bounds the walk.
//!
//! The pipeline is `learn` with the scripted policy and the trainer double
//! the learn specs use; the release is written straight into the release
//! store from the learned candidate, since what the gate decides is the
//! release specs' concern and lineage reads only the manifest.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use common::gate::{gate_context, Brain, FakeTrainer, BASE_BYTES, NOW};
use common::manual::{manual_policy, BAUD_QUESTION, BAUD_QUOTE, MANUAL};
use common::Scratch;
use splinter_agent::solve::Model;
use splinter_campaign::ask::ask;
use splinter_campaign::learn::{learn, LearnRequest, Learned};
use splinter_campaign::lineage::{lineage, Direction, Lineage, LineageRequest, NodeKind, Relation};
use splinter_campaign::model_ref::ModelRef;
use splinter_campaign::release::gate::{Check, GateConfig, GateReport};
use splinter_campaign::release::{ReleaseId, ReleaseManifest, RELEASE_FORMAT};
use splinter_campaign::train::{Candidate, TrainingSummary};
use splinter_campaign::{CampaignError, Context};
use splinter_core::clock::FixedClock;
use splinter_core::digest::Digest;
use splinter_core::experience::{Environment, ExperienceId, Task};
use splinter_core::source::{CapturedSource, Origin, PartContent, SourceId};
use splinter_views::DatasetId;

/// What the learned fixture holds.
struct Fixture {
    source: SourceId,
    dataset: DatasetId,
    candidate: Candidate,
    release: ReleaseId,
}

/// Learns the manual (released nothing), then writes a release of the
/// learned candidate and points `default` at it.
fn learned(test: &str) -> (Scratch, Context, Fixture) {
    let (scratch, ctx) = gate_context(test, Brain::Missing);
    ctx.add_model(
        ModelRef::policy_default(),
        Model::new(Arc::new(manual_policy()), common::POLICY),
    );
    let manual = scratch.0.join("manual.md");
    std::fs::write(&manual, MANUAL).unwrap();
    let Learned::Ran(run) = learn(
        &ctx,
        &LearnRequest {
            sources: vec![manual.display().to_string()],
            no_release: true,
            // The scripted policy answers each task the same way every
            // time: keep every task.
            no_frontier: true,
            ..LearnRequest::default()
        },
        &FakeTrainer::knowing(&[]),
    )
    .unwrap() else {
        panic!("a learn that is not a dry run runs");
    };
    let report = run.report;
    let source = report.sources[0].id.clone();
    let dataset = report.dataset.unwrap().dataset;
    let candidate = report.candidate.unwrap();
    let release = write_release(&ctx, &candidate);
    (
        scratch,
        ctx,
        Fixture {
            source,
            dataset,
            candidate,
            release,
        },
    )
}

/// Stores a release of `candidate` and points `default` at it.
fn write_release(ctx: &Context, candidate: &Candidate) -> ReleaseId {
    let unmeasured = "written by the lineage spec";
    let manifest = ReleaseManifest {
        format: RELEASE_FORMAT.into(),
        base_model: "Qwen/Qwen3-0.6B".into(),
        base_digest: Digest::sha256_of(BASE_BYTES),
        adapter_digest: Digest::parse(&candidate.adapter_digest).unwrap(),
        adapter_artifact: candidate.adapter_artifact.clone(),
        parent: candidate.parent.clone(),
        candidate: candidate.candidate.clone(),
        datasets: candidate.datasets.clone(),
        replay: candidate.replay.clone(),
        training: TrainingSummary {
            from: candidate.from.clone(),
            steps: candidate.steps,
            rank: candidate.rank,
            records: candidate.records,
            regime: candidate.regime,
            base_score: candidate.base_score,
            tuned_score: candidate.tuned_score,
            preference: candidate.preference.clone(),
            record: serde_json::json!({ "trainer": "fake" }),
        },
        gate: GateReport::new(
            GateConfig::default(),
            Check::unmeasured(unmeasured),
            Check::unmeasured(unmeasured),
            Check::unmeasured(unmeasured),
            Check::unmeasured(unmeasured),
        ),
        created_at: NOW.into(),
    };
    let store = ctx.releases();
    let stored = store.put(&manifest).unwrap();
    store.move_alias("default", None, &stored.id, NOW).unwrap();
    stored.id
}

fn walk(ctx: &Context, id: &str, direction: Direction, depth: Option<usize>) -> Lineage {
    lineage(
        ctx,
        &LineageRequest {
            id: id.to_string(),
            direction,
            depth,
        },
    )
    .unwrap()
}

fn ids_of(lineage: &Lineage, kind: NodeKind) -> BTreeSet<String> {
    lineage
        .nodes
        .iter()
        .filter(|n| n.kind == kind)
        .map(|n| n.id.clone())
        .collect()
}

fn has_edge(lineage: &Lineage, from: &str, relation: Relation, to: &str) -> bool {
    lineage
        .edges
        .iter()
        .any(|e| e.from == from && e.relation == relation && e.to == to)
}

#[test]
fn from_a_release_up_to_the_source_bytes_and_from_the_source_down_to_the_release() {
    let (_scratch, ctx, fixture) = learned("lineage-release");
    let release = fixture.release.to_string();
    let up = walk(&ctx, &release, Direction::Up, None);
    assert_eq!(up.nodes[0].id, release, "the root comes first");
    assert_eq!(up.nodes[0].kind, NodeKind::Release);

    let dataset = fixture.dataset.to_string();
    assert!(has_edge(&up, &release, Relation::TrainedOn, &dataset));
    assert!(has_edge(
        &up,
        &release,
        Relation::ReleaseOf,
        &fixture.candidate.candidate
    ));
    assert!(has_edge(
        &up,
        &release,
        Relation::Adapter,
        &fixture.candidate.adapter_digest
    ));

    // The database's own record of the release's provenance and the graph the
    // command prints say the same: one record of lineage, two readings.
    let manifest = ctx.datasets().get(&fixture.dataset).unwrap().manifest;
    let trace = ctx
        .workspace()
        .trace_release(&fixture.release.0)
        .unwrap()
        .unwrap();
    assert_eq!(trace.candidates, vec![fixture.candidate.candidate.clone()]);
    assert_eq!(trace.datasets, vec![fixture.dataset.0.clone()]);
    assert!(
        trace.attempts >= manifest.experiences.len(),
        "every experience of the dataset is an attempt it traces back to"
    );
    assert_eq!(
        ids_of(&up, NodeKind::Dataset)
            .into_iter()
            .collect::<Vec<_>>(),
        vec![fixture.dataset.to_string()]
    );

    // The dataset's experiences, as its manifest names them, and their
    // tasks.
    assert_eq!(manifest.experiences.len(), 2);
    let experiences = ids_of(&up, NodeKind::Experience);
    let tasks = ids_of(&up, NodeKind::Task);
    for experience in &manifest.experiences {
        assert!(experiences.contains(experience.as_str()), "{experience}");
        assert!(has_edge(
            &up,
            &dataset,
            Relation::ProjectedFrom,
            experience.as_str()
        ));
        let recorded = ctx.experiences().get(experience).unwrap();
        assert!(tasks.contains(recorded.task.id.as_str()));
    }

    // The baud-rate task's evidence resolves to exactly the bytes the span
    // node shows, in the source it names.
    let baud = manifest
        .experiences
        .iter()
        .map(|id| ctx.experiences().get(id).unwrap())
        .find(|e| e.instruction == BAUD_QUESTION)
        .unwrap();
    let span = &baud.evidence[0];
    let bytes = String::from_utf8(ctx.sources().read_span(span).unwrap()).unwrap();
    assert_eq!(bytes, BAUD_QUOTE);
    let span_node = up
        .nodes
        .iter()
        .find(|n| {
            n.kind == NodeKind::Span
                && has_edge(&up, baud.task.id.as_str(), Relation::Evidence, &n.id)
        })
        .unwrap();
    assert!(
        span_node.label.contains(&format!("{:?}", BAUD_QUOTE)),
        "{}",
        span_node.label
    );
    assert!(
        span_node
            .label
            .contains(&format!("[{}, {})", span.start, span.end)),
        "{}",
        span_node.label
    );
    assert!(span_node.label.contains("manual.md"), "{}", span_node.label);
    let source = fixture.source.to_string();
    assert!(has_edge(&up, &span_node.id, Relation::SpanOf, &source));
    assert_eq!(
        ids_of(&up, NodeKind::Source),
        BTreeSet::from([source.clone()])
    );

    // Down from the source: through the span, task, experience and dataset
    // to the candidate and the release.
    let down = walk(&ctx, &source, Direction::Down, None);
    for reached in [
        &span_node.id,
        &dataset,
        &fixture.candidate.candidate,
        &release,
    ] {
        assert!(
            down.nodes.iter().any(|n| &n.id == reached),
            "{reached} not reached"
        );
    }
    assert!(
        down.nodes.iter().all(|n| n.kind != NodeKind::Environment),
        "walking down never climbs back up"
    );
}

#[test]
fn an_answer_walks_up_to_the_release_it_was_answered_with_and_on_to_the_sources() {
    let (_scratch, ctx, fixture) = learned("lineage-answer");
    // A fresh command: `default` is resolved to the release now.
    let asking = Context::new(ctx.config().clone(), false)
        .unwrap()
        .with_clock(Box::new(FixedClock::new(NOW)))
        .with_model(
            ModelRef::policy_default(),
            Model::new(Arc::new(manual_policy()), common::POLICY),
        );
    let answer = ask(&asking, BAUD_QUESTION, None, &ModelRef::policy_default()).unwrap();
    assert_eq!(answer.answer, "115200 baud");
    assert_eq!(answer.release, Some(fixture.release.clone()));

    let up = walk(&ctx, answer.id.as_str(), Direction::Up, None);
    assert_eq!(up.nodes[0].kind, NodeKind::Answer);
    assert!(
        up.nodes[0].label.contains(BAUD_QUESTION),
        "{}",
        up.nodes[0].label
    );
    let release = fixture.release.to_string();
    assert!(has_edge(
        &up,
        answer.id.as_str(),
        Relation::AnsweredWith,
        &release
    ));
    assert!(ids_of(&up, NodeKind::Source).contains(fixture.source.as_str()));

    let down = walk(&ctx, &release, Direction::Down, None);
    assert!(ids_of(&down, NodeKind::Answer).contains(answer.id.as_str()));
}

#[test]
fn a_critique_chain_shows_its_relations() {
    let (_scratch, ctx, fixture) = learned("lineage-critique");
    let up = walk(&ctx, &fixture.release.to_string(), Direction::Up, None);
    let retry = up
        .edges
        .iter()
        .find(|e| e.relation == Relation::RetryOf)
        .expect("the repaired answer in the dataset retries the failed one");
    let failed = retry.to.clone();
    assert!(has_edge(&up, &retry.from, Relation::PreferredOver, &failed));

    let around = walk(&ctx, &failed, Direction::Down, None);
    let relations: BTreeSet<Relation> = around
        .edges
        .iter()
        .filter(|e| e.to == failed)
        .map(|e| e.relation)
        .collect();
    for relation in [
        Relation::CritiqueOf,
        Relation::RetryOf,
        Relation::PreferredOver,
    ] {
        assert!(relations.contains(&relation), "{relation:?}: {relations:?}");
    }
    let critique = around
        .edges
        .iter()
        .find(|e| e.relation == Relation::CritiqueOf)
        .unwrap();
    let critic = ExperienceId(Digest::parse(&critique.from).unwrap());
    assert!(ctx.experiences().contains(&critic).unwrap());
}

#[test]
fn ids_are_refused_when_unknown_or_ambiguous_and_the_report_has_its_documented_shape() {
    let (_scratch, ctx, fixture) = learned("lineage-shape");
    let unknown = lineage(
        &ctx,
        &LineageRequest {
            id: "0000000000".into(),
            direction: Direction::Both,
            depth: None,
        },
    )
    .unwrap_err();
    assert!(
        matches!(unknown, CampaignError::NotFound { .. }) && unknown.is_refusal(),
        "{unknown}"
    );

    // The documented JSON: nodes of {id, kind, label}, edges of {from, to,
    // relation} between them, the root first.
    let both = walk(&ctx, fixture.dataset.0.hex(), Direction::Both, None);
    let json: serde_json::Value =
        serde_json::from_str(&serde_json::to_string(&both).unwrap()).unwrap();
    let keys = |v: &serde_json::Value| -> Vec<String> {
        let mut keys: Vec<String> = v.as_object().unwrap().keys().cloned().collect();
        keys.sort();
        keys
    };
    assert_eq!(keys(&json), ["edges", "nodes"]);
    assert_eq!(json["nodes"][0]["id"], fixture.dataset.to_string());
    assert_eq!(json["nodes"][0]["kind"], "dataset");
    let nodes: BTreeSet<&str> = json["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| {
            assert_eq!(keys(n), ["id", "kind", "label"]);
            n["id"].as_str().unwrap()
        })
        .collect();
    assert_eq!(nodes.len(), both.nodes.len(), "each node once");
    for edge in json["edges"].as_array().unwrap() {
        assert_eq!(keys(edge), ["from", "relation", "to"]);
        assert!(nodes.contains(edge["from"].as_str().unwrap()));
        assert!(nodes.contains(edge["to"].as_str().unwrap()));
    }
    let relations: BTreeSet<&str> = json["edges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["relation"].as_str().unwrap())
        .collect();
    assert!(
        relations.contains("projected_from") && relations.contains("trained_on"),
        "up and down: {relations:?}"
    );

    // Depth 1: the release and what it names directly, nothing further.
    let release = fixture.release.to_string();
    let shallow = walk(&ctx, &release, Direction::Up, Some(1));
    assert!(shallow.edges.iter().all(|e| e.from == release));
    assert!(ids_of(&shallow, NodeKind::Dataset).contains(fixture.dataset.0.as_str()));
    assert!(ids_of(&shallow, NodeKind::Experience).is_empty());
    let up = shallow.up.as_ref().unwrap();
    assert!(
        up.children.iter().any(|c| c.cut),
        "a node with more beyond the depth says so"
    );
}

#[test]
fn a_prefix_naming_artifacts_in_two_stores_is_refused_with_the_candidates() {
    let scratch = Scratch::new("lineage-ambiguous");
    let ctx = Context::new(common::config(&scratch), false).unwrap();
    let clock = FixedClock::new(NOW);
    // A source and a task whose ids share their first four hex digits.
    let mut sources: HashMap<String, CapturedSource> = HashMap::new();
    for i in 0..2000 {
        let captured = CapturedSource::new(
            Origin::Document {
                path: format!("/docs/{i}.md"),
            },
            vec![PartContent {
                name: "doc.md".into(),
                media_type: "text/markdown".into(),
                bytes: format!("document {i}").into_bytes(),
            }],
            &clock,
        )
        .unwrap();
        sources.insert(captured.source().id.hex()[..4].to_string(), captured);
    }
    let (source, task) = (0..100_000)
        .find_map(|i| {
            let task = Task::new(
                "recall",
                Vec::new(),
                Environment::closed_book(),
                format!("question {i}"),
                Vec::new(),
            )
            .unwrap();
            sources
                .get(&task.task.id.hex()[..4])
                .map(|source| (source.clone(), task))
        })
        .unwrap();
    let source_id = ctx.sources().put_source(&source).unwrap();
    let task_id = ctx.tasks().put(&task).unwrap();

    let prefix = &task_id.hex()[..4];
    let refused = lineage(
        &ctx,
        &LineageRequest {
            id: prefix.into(),
            direction: Direction::Both,
            depth: None,
        },
    )
    .unwrap_err();
    assert!(refused.is_refusal(), "{refused}");
    let CampaignError::AmbiguousArtifact { candidates, .. } = &refused else {
        panic!("{refused}");
    };
    let listed = refused.to_string();
    for (kind, id) in [("source", source_id.as_str()), ("task", task_id.as_str())] {
        assert!(
            candidates
                .iter()
                .any(|c| c.contains(id) && c.contains(kind)),
            "{listed}"
        );
        assert!(listed.contains(id), "{listed}");
    }
    // More of the id names one.
    let one = walk(&ctx, task_id.as_str(), Direction::Up, None);
    assert_eq!(one.nodes[0].kind, NodeKind::Task);
}
