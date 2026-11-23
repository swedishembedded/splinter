// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model evaluations whose every verdict is
// produced by code that names itself, for its clients. If your team needs
// expertise in rigorous answer verification, you can procure our services by
// sending an email to info@swedishembedded.com.

//! `answer.collect`: a closed-book task (one `task.generate` admitted from a
//! real document) answered in process, through the one solve, and graded by
//! the verifiers its kind names: the same verdicts, under the same producer,
//! as wherever an answer is graded.

use std::sync::{Arc, Mutex};

use serde_json::Value;
use splinter_agent::solve::Model;
use splinter_core::factory::execution::ModelBinding;
use splinter_factory::execute::Models;
use splinter_factory::resolve::resolve;
use splinter_factory::run::{Factory, Services};
use sven_sdk::model::{CompletionRequest, ModelProvider, ResponseEvent, ResponseStream};

use crate::fakes::{FakeBackend, Outcome};
use crate::repo::{connections, Repo};

/// A model that gives the answer it is set to.
struct Answering(Mutex<String>);

#[async_trait::async_trait]
impl ModelProvider for Answering {
    fn name(&self) -> &str {
        "scripted"
    }
    fn model_name(&self) -> &str {
        "student-1"
    }
    async fn complete(&self, _: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let reply = self.0.lock().unwrap().clone();
        Ok(Box::pin(futures::stream::iter(vec![
            Ok(ResponseEvent::TextDelta(reply)),
            Ok(ResponseEvent::Done),
        ])))
    }
}

struct Student(Arc<Answering>);

impl Models for Student {
    fn model(&self, _: &ModelBinding) -> Result<Model, String> {
        Ok(Model::new(self.0.clone(), "scripted/student-1"))
    }
}

const RECIPE: &str = "schema: splinter.recipe/v2
name: recall
inputs:
  tasks: { kind: files, path: closed }
models:
  student: \"hosted:org/student\"
stages:
  - id: answered
    uses: answer.collect
    foreach: $inputs.tasks
    inputs: { task: $item }
    with: { model: student }
";

/// The stated verdict of the one answer to the fixture task, as `answer`.
fn verdict(answer: &str) -> String {
    let repo = Repo::new();
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/it/fixtures/closed-book/recall.json");
    repo.write(
        "closed/recall/task.json",
        &std::fs::read_to_string(fixture).unwrap(),
    );
    let resolved = resolve(&repo.root, RECIPE, connections()).unwrap();
    let backend = FakeBackend::with(Outcome::Pass);
    let student = Student(Arc::new(Answering(Mutex::new(answer.into()))));
    let services = Services {
        backend: &backend,
        trainer: None,
        models: Some(&student),
        variables: None,
    };
    let dir = tempfile::tempdir().unwrap();
    Factory::open(&resolved, dir.path(), services, "test")
        .unwrap()
        .run()
        .unwrap();
    let verdicts: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("answered/recall/verdicts-0.json")).unwrap(),
    )
    .unwrap();
    assert!(dir
        .path()
        .join("answered/recall/experience-0.json")
        .is_file());
    let stated = verdicts
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["producer"]["name"] == "splinter-lab/reference-stated:lenient")
        .expect("the kind's stated verdict");
    stated["body"]["outcome"].as_str().unwrap().to_owned()
}

#[test]
fn a_closed_book_answer_is_graded_by_its_kinds_verifiers() {
    assert_eq!(
        verdict("A fork correction continues it instead of starting the task afresh."),
        "pass"
    );
    assert_eq!(verdict("It deletes the attempt and moves on."), "fail");
}
