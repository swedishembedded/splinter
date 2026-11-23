// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements structured extraction from documents with
// language models whose every answer is checked against a schema, for its
// clients. If your team needs expertise in reliable model-driven data
// pipelines, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The `agent` operation: one bounded call per item for a value of the
//! stage's JSON Schema. A reply that breaks the schema goes back to the
//! model for correction; an item the model never answers validly is
//! rejected, with the reason; the value and the call's record are the
//! item's product, and a run that resumes asks again for nothing.

use std::path::Path;
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use splinter_agent::solve::Model;
use splinter_core::factory::execution::ModelBinding;
use splinter_core::factory::items::{ItemIndex, ITEMS_FILE};
use splinter_factory::execute::Models;
use splinter_factory::resolve::{resolve, Resolved};
use splinter_factory::run::{Factory, RunError, Services};
use sven_sdk::model::{CompletionRequest, ModelProvider, ResponseEvent, ResponseStream};

use crate::fakes::{FakeBackend, Outcome as Trial};
use crate::repo::{connections, Repo};

/// A model that answers by what the request is about: a note on alpha first
/// with a value of the wrong shape, then a right one; a note on zeta always
/// with the wrong shape.
#[derive(Default)]
struct Scripted {
    requests: Mutex<Vec<String>>,
}

#[async_trait::async_trait]
impl ModelProvider for Scripted {
    fn name(&self) -> &str {
        "scripted"
    }
    fn model_name(&self) -> &str {
        "writer-1"
    }
    async fn complete(&self, request: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let text = serde_json::to_string(&request.messages).unwrap_or_default();
        let mut requests = self.requests.lock().unwrap();
        requests.push(text.clone());
        let alpha_asked = requests.iter().filter(|r| r.contains("alpha")).count();
        let reply = if text.contains("alpha") && alpha_asked > 1 {
            r#"{"fact": "alpha comes first"}"#
        } else {
            r#"{"fact": 3}"#
        };
        Ok(Box::pin(futures::stream::iter(vec![
            Ok(ResponseEvent::TextDelta(reply.into())),
            Ok(ResponseEvent::Done),
        ])))
    }
}

/// The run's models: every binding served by the one scripted model.
struct Fake {
    provider: Arc<Scripted>,
    bound: Mutex<Vec<ModelBinding>>,
}

impl Models for Fake {
    fn model(&self, binding: &ModelBinding) -> Result<Model, String> {
        self.bound.lock().unwrap().push(binding.clone());
        Ok(Model::new(self.provider.clone(), "scripted/writer-1"))
    }
}

const RECIPE: &str = "schema: splinter.recipe/v2
name: facts
inputs:
  notes: { kind: files, path: notes }
models:
  writer: \"hosted:org/writer\"
stages:
  - id: facts
    uses: agent
    foreach: $inputs.notes
    concurrency: 2
    inputs: { input: $item }
    with: { instruction: prompts/extract.md, output_schema: schemas/fact.json, model: writer,
            repairs: 2 }
";

fn repo() -> Repo {
    let repo = Repo::new();
    repo.write("notes/a.md", "alpha is the first letter");
    repo.write("notes/z.md", "zeta is a late letter");
    repo.write("prompts/extract.md", "State the one fact the note gives.");
    repo.write(
        "schemas/fact.json",
        r#"{"type": "object", "properties": {"fact": {"type": "string"}},
            "required": ["fact"], "additionalProperties": false}"#,
    );
    repo
}

fn run(resolved: &Resolved, dir: &Path, models: &Fake) -> Result<(), RunError> {
    let backend = FakeBackend::with(Trial::Pass);
    let services = Services {
        backend: &backend,
        trainer: None,
        models: Some(models),
        variables: None,
    };
    let mut factory = Factory::open(resolved, dir, services, "test")?;
    let ran = factory.run().map(|_| ());
    drop(factory);
    ran
}

#[test]
fn an_agent_stage_answers_each_item_in_its_schema_or_rejects_it() {
    let repo = repo();
    let resolved = resolve(&repo.root, RECIPE, connections()).unwrap();
    let models = Fake {
        provider: Arc::new(Scripted::default()),
        bound: Mutex::new(Vec::new()),
    };
    let dir = tempfile::tempdir().unwrap();
    run(&resolved, dir.path(), &models).unwrap();

    let index: ItemIndex = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("facts").join(ITEMS_FILE)).unwrap(),
    )
    .unwrap();
    assert_eq!(index.items.len(), 1);
    assert_eq!(index.items[0].id, "a.md");
    assert_eq!(index.rejected.len(), 1);
    assert_eq!(index.rejected[0].id, "z.md");
    let output: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("facts/a.md/output.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(output, json!({ "fact": "alpha comes first" }));
    let record: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("facts/a.md/call.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(record["model"], "scripted/writer-1");

    let requests = models.provider.requests.lock().unwrap().clone();
    let alpha: Vec<&String> = requests.iter().filter(|r| r.contains("alpha")).collect();
    assert_eq!(alpha.len(), 2, "one correction");
    assert!(
        alpha[0].contains("State the one fact") && alpha[0].contains("\\\"fact\\\""),
        "the brief carries the instruction and the schema: {}",
        alpha[0]
    );
    assert!(
        models
            .bound
            .lock()
            .unwrap()
            .iter()
            .all(|b| b.connection == "hosted" && b.model == "org/writer"),
        "the role's binding"
    );

    // Resumed, nothing is asked again.
    let asked = requests.len();
    run(&resolved, dir.path(), &models).unwrap();
    assert_eq!(models.provider.requests.lock().unwrap().len(), asked);
}

#[test]
fn an_agent_stage_is_refused_without_a_readable_schema_or_a_bound_role() {
    let repo = repo();
    repo.write("schemas/broken.json", "{ not json");
    let refused = |recipe: String| {
        resolve(&repo.root, &recipe, connections())
            .unwrap_err()
            .0
            .join("\n")
    };
    let broken = refused(RECIPE.replace("schemas/fact.json", "schemas/broken.json"));
    assert!(broken.contains("schemas/broken.json"), "{broken}");
    let missing = refused(RECIPE.replace("schemas/fact.json", "schemas/none.json"));
    assert!(missing.contains("schemas/none.json"), "{missing}");
    let unbound = refused(RECIPE.replace("model: writer,", "model: reader,"));
    assert!(unbound.contains("reader"), "{unbound}");
}

#[test]
fn a_run_without_models_is_refused_before_any_item_runs() {
    let repo = repo();
    let resolved = resolve(&repo.root, RECIPE, connections()).unwrap();
    let backend = FakeBackend::with(Trial::Pass);
    let services = Services {
        backend: &backend,
        trainer: None,
        models: None,
        variables: None,
    };
    let dir = tempfile::tempdir().unwrap();
    let error = Factory::open(&resolved, dir.path(), services, "test")
        .and_then(|mut f| f.run().map(|_| ()))
        .unwrap_err()
        .to_string();
    assert!(error.contains("model"), "{error}");
}
