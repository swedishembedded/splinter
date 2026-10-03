// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning pipelines that turn verified agent
// experience into training data, for its clients. If your team needs
// expertise in continual learning or agent evaluation, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: the experience path end to end, with no network and no model.
//!
//! source in the source store -> denoise task -> a solve through sven ->
//! experience in the store -> formal verdict appended -> SFT-final view -> a
//! chat dataset the trainer's parser accepts; the task's evidence resolves
//! back to the source's bytes. The solver is a scripted in-test model behind
//! sven's `ModelProvider` seam.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::path::PathBuf;
use std::sync::Arc;

use splinter_core::annotation::{Annotation, AnnotationBody, Outcome, Producer, Strength};
use splinter_core::clock::FixedClock;
use splinter_core::experience::{Digest, Experience, Privileged, PrivilegedKind, Provenance, Task};
use splinter_core::prompt::SYSTEM_PROMPT;
use splinter_core::source::{CapturedSource, Origin, PartContent};
use splinter_knowledge::denoise::{Denoise, GENERATOR};
use splinter_lab::denoise::FormalVerifier;
use splinter_lab::verifiers::annotation;
use splinter_store::experiences::ExperienceStore;
use splinter_store::sources::SourceStore;
use splinter_store::StateRoot;
use splinter_views::{write_dataset, Corpus, SftFinal, View, WriteOptions};
use sven_sdk::model::{CompletionRequest, ModelProvider, ResponseEvent, ResponseStream};
use sven_sdk::Engine;

const SOURCE: &str = "An experience is immutable and content addressed. An annotation \
    is appended beside it and never rewrites it. Reward is derived from the annotations, \
    and every training set is projected from experiences through a view.";

/// A privileged item only the teacher sees, distinct from anything the
/// student is shown or answers.
const TEACHER_NOTE: &str = "TEACHER-ONLY: the passage comes from the store's design notes";

const SOLVER: &str = "scripted/solver-1";

/// A model that answers every request with one fixed reply.
struct Scripted(String);

#[async_trait::async_trait]
impl ModelProvider for Scripted {
    fn name(&self) -> &str {
        "scripted"
    }
    fn model_name(&self) -> &str {
        "solver-1"
    }
    async fn complete(&self, _req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        Ok(Box::pin(futures::stream::iter(vec![
            Ok(ResponseEvent::TextDelta(self.0.clone())),
            Ok(ResponseEvent::Done),
        ])))
    }
}

struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The denoise task over `SOURCE` stored in `sources`, with a teacher-only
/// note beside its reference.
fn task(sources: &SourceStore) -> Task {
    let captured = CapturedSource::new(
        Origin::Document {
            path: "/notes/experience.md".into(),
        },
        vec![PartContent {
            name: "experience.md".into(),
            media_type: "text/markdown".into(),
            bytes: SOURCE.as_bytes().to_vec(),
        }],
        &FixedClock::new("2026-09-30T07:00:00.000Z"),
    )
    .unwrap();
    let id = sources.put_source(&captured).unwrap();
    let generated = Denoise::new(11)
        .generate(
            &sources.get_source(&id).unwrap(),
            "experience.md",
            &sources.read_part(&id, "experience.md").unwrap(),
        )
        .unwrap();
    let mut privileged = generated.privileged;
    privileged.push(Privileged {
        kind: PrivilegedKind::Hint,
        content: TEACHER_NOTE.into(),
        span: None,
    });
    Task::new(
        generated.task.kind,
        generated.evidence,
        generated.environment,
        generated.instruction,
        privileged,
    )
    .unwrap()
}

fn reference(task: &Task) -> String {
    task.privileged
        .iter()
        .find(|p| p.kind == PrivilegedKind::Reference)
        .map(|p| p.content.clone())
        .unwrap()
}

/// Solves `task` through sven with a model that replies `answer`.
async fn solve(task: Task, answer: &str) -> Experience {
    let engine = Engine::builder()
        .model_provider(Arc::new(Scripted(answer.into())))
        .build()
        .unwrap();
    let mut agent = engine.agent("agent");
    let outcome = agent.send(&task.instruction).await.unwrap();
    let mut provenance = Provenance::new(SOLVER, &FixedClock::new("2026-09-30T08:00:00.000Z"));
    provenance.generator = Some(GENERATOR.into());
    Experience::new(task, agent.trajectory(), Some(outcome.reply), provenance).unwrap()
}

#[tokio::test]
async fn a_source_becomes_a_verified_dataset_through_the_experience_store() {
    let scratch = Scratch(
        std::env::temp_dir().join(format!("splinter-experience-path-{}", std::process::id())),
    );
    let root = StateRoot::new(&scratch.0);
    let store = ExperienceStore::new(&splinter_store::workspace::Workspace::at(&root));
    let sources = SourceStore::new(&splinter_store::workspace::Workspace::at(&root));
    let task = task(&sources);
    let answer = reference(&task);
    assert_eq!(
        sources.read_span(&task.evidence[0]).unwrap(),
        answer.as_bytes(),
        "the evidence resolves to the passage the reference restores"
    );

    // A solve that restores the passage.
    let passed = solve(task.clone(), &answer).await;
    assert!(
        !passed.trajectory.steps.is_empty(),
        "sven recorded the solve"
    );
    let id = store.put(&passed).unwrap();
    assert_eq!(
        store.put(&passed).unwrap(),
        id,
        "identical content is one experience"
    );
    assert_eq!(
        store.get(&id).unwrap(),
        passed,
        "the trajectory round-trips"
    );
    assert_eq!(store.list().unwrap().len(), 1);

    store
        .annotate(&annotation(&FormalVerifier::new(), &task, &passed).unwrap())
        .unwrap();
    let view = SftFinal::new(Strength::Formal);
    let projection = view
        .project(&Corpus::load(&store, std::slice::from_ref(&id)).unwrap())
        .unwrap();
    assert_eq!(projection.records.len(), 1);

    let dataset = write_dataset(
        &scratch.0.join("sft.jsonl"),
        &projection,
        WriteOptions::default(),
    )
    .unwrap();
    let text = std::fs::read_to_string(&dataset.path).unwrap();
    assert_eq!(dataset.digest, Digest::of(text.as_bytes()));
    let line: serde_json::Value = serde_json::from_str(text.trim_end()).unwrap();
    assert_eq!(line["messages"][0]["role"], "system");
    assert_eq!(line["messages"][0]["content"], SYSTEM_PROMPT);
    assert_eq!(line["messages"][0]["train"], false);
    assert_eq!(line["messages"][1]["role"], "user");
    assert_eq!(line["messages"][1]["content"], task.instruction.as_str());
    assert_eq!(line["messages"][1]["train"], false);
    assert_eq!(line["messages"][2]["role"], "assistant");
    assert_eq!(line["messages"][2]["content"], answer.as_str());
    assert_eq!(line["messages"][2]["train"], true);
    assert!(
        !text.contains("TEACHER-ONLY"),
        "privileged content leaked: {text}"
    );
    assert!(
        !task.instruction.contains(&answer),
        "the student's turn never shows the reference"
    );

    // A solve with the wrong answer yields a verdict and no record.
    let failed = solve(task.clone(), "a wrong answer").await;
    let failed_id = store.put(&failed).unwrap();
    assert_ne!(failed_id, id);
    store
        .annotate(&annotation(&FormalVerifier::new(), &task, &failed).unwrap())
        .unwrap();
    let notes = store.annotations(&failed_id).unwrap().annotations;
    assert!(matches!(
        notes[..],
        [Annotation {
            body: AnnotationBody::Verdict {
                outcome: Outcome::Fail,
                ..
            },
            ..
        }]
    ));
    let projection = view
        .project(&Corpus::load(&store, std::slice::from_ref(&failed_id)).unwrap())
        .unwrap();
    assert!(projection.records.is_empty());

    // Re-grading appends; the view changes, the experience does not.
    let before = store.get(&id).unwrap();
    store
        .annotate(&Annotation {
            experience: id.clone(),
            producer: Producer {
                name: "executable-check".into(),
                version: "1".into(),
            },
            body: AnnotationBody::Verdict {
                outcome: Outcome::Fail,
                strength: Strength::Executable,
                evidence: serde_json::json!({"reason": "stricter check"}),
            },
        })
        .unwrap();
    let notes = store.annotations(&id).unwrap().annotations;
    assert_eq!(notes.len(), 2);
    let projection = view
        .project(&Corpus::load(&store, std::slice::from_ref(&id)).unwrap())
        .unwrap();
    assert!(projection.records.is_empty());
    assert_eq!(store.get(&id).unwrap(), before, "never rewritten");
}
