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
use splinter_factory::run::{Factory, ItemRetry, RunError, Services};
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
        online: None,
        build: splinter_core::build::BuildInfo::this_executable(),
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
        online: None,
        build: splinter_core::build::BuildInfo::this_executable(),
    };
    let dir = tempfile::tempdir().unwrap();
    let error = Factory::open(&resolved, dir.path(), services, "test")
        .and_then(|mut f| f.run().map(|_| ()))
        .unwrap_err()
        .to_string();
    assert!(error.contains("model"), "{error}");
}

/// A model that answers every note, except that it fails the first
/// `transient` asks about zeta as an unreachable server does, and every ask
/// about zeta when `down`.
struct Flaky {
    transient: usize,
    down: bool,
    zeta_asked: Mutex<usize>,
    requests: Mutex<Vec<String>>,
}

impl Flaky {
    fn new(transient: usize, down: bool) -> Arc<Self> {
        Arc::new(Self {
            transient,
            down,
            zeta_asked: Mutex::new(0),
            requests: Mutex::new(Vec::new()),
        })
    }
}

#[async_trait::async_trait]
impl ModelProvider for Flaky {
    fn name(&self) -> &str {
        "flaky"
    }
    fn model_name(&self) -> &str {
        "writer-1"
    }
    async fn complete(&self, request: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let text = serde_json::to_string(&request.messages).unwrap_or_default();
        self.requests.lock().unwrap().push(text.clone());
        if text.contains("zeta") {
            let mut asked = self.zeta_asked.lock().unwrap();
            *asked += 1;
            if self.down {
                anyhow::bail!("HTTP 400 Bad Request: the model is not served here");
            }
            if *asked <= self.transient {
                anyhow::bail!("no response headers from brain after 600 s");
            }
        }
        Ok(Box::pin(futures::stream::iter(vec![
            Ok(ResponseEvent::TextDelta(r#"{"fact": "a letter"}"#.into())),
            Ok(ResponseEvent::Done),
        ])))
    }
}

struct Serving(Arc<Flaky>);

impl Models for Serving {
    fn model(&self, _: &ModelBinding) -> Result<Model, String> {
        Ok(Model::new(self.0.clone(), "flaky/writer-1"))
    }
}

/// Retries quick enough for a test: three asks, a millisecond apart.
fn quick() -> ItemRetry {
    ItemRetry {
        attempts: 3,
        first_pause: std::time::Duration::from_millis(1),
    }
}

fn run_with(resolved: &Resolved, dir: &Path, models: &dyn Models) -> Result<(), RunError> {
    let backend = FakeBackend::with(Trial::Pass);
    let services = Services {
        backend: &backend,
        trainer: None,
        models: Some(models),
        variables: None,
        online: None,
        build: splinter_core::build::BuildInfo::this_executable(),
    };
    let mut factory = Factory::open(resolved, dir, services, "test")?.with_item_retry(quick());
    let ran = factory.run().map(|_| ());
    drop(factory);
    ran
}

fn index_of(dir: &Path) -> ItemIndex {
    serde_json::from_str(&std::fs::read_to_string(dir.join("facts").join(ITEMS_FILE)).unwrap())
        .unwrap()
}

#[test]
fn an_item_that_fails_transiently_is_asked_again_and_the_stage_completes() {
    let repo = repo();
    let resolved = resolve(&repo.root, RECIPE, connections()).unwrap();
    let flaky = Flaky::new(2, false);
    let dir = tempfile::tempdir().unwrap();
    run_with(&resolved, dir.path(), &Serving(flaky.clone())).unwrap();
    assert_eq!(
        *flaky.zeta_asked.lock().unwrap(),
        3,
        "two failures, then the answer"
    );
    let index = index_of(dir.path());
    assert_eq!(index.items.len(), 2, "{index:?}");
}

#[test]
fn a_persistent_item_failure_fails_the_stage_naming_it_and_a_resume_asks_only_for_it() {
    let repo = repo();
    let resolved = resolve(&repo.root, RECIPE, connections()).unwrap();
    let dir = tempfile::tempdir().unwrap();
    // Transient past every retry: the other item still completes.
    let flaky = Flaky::new(usize::MAX, false);
    let error = run_with(&resolved, dir.path(), &Serving(flaky.clone()))
        .unwrap_err()
        .to_string();
    assert!(error.contains("z.md") && !error.contains("a.md"), "{error}");
    assert_eq!(*flaky.zeta_asked.lock().unwrap(), 3, "bounded: {error}");
    // A failure that asking again cannot pass is not retried.
    let down = Flaky::new(0, true);
    let other = tempfile::tempdir().unwrap();
    let error = run_with(&resolved, other.path(), &Serving(down.clone()))
        .unwrap_err()
        .to_string();
    assert!(error.contains("z.md"), "{error}");
    assert_eq!(*down.zeta_asked.lock().unwrap(), 1, "{error}");

    // Resumed with the server back: only the failed item is asked for.
    let healthy = Flaky::new(0, false);
    run_with(&resolved, dir.path(), &Serving(healthy.clone())).unwrap();
    let asked = healthy.requests.lock().unwrap().clone();
    assert!(
        !asked.is_empty() && asked.iter().all(|r| r.contains("zeta")),
        "only zeta is asked again: {asked:?}"
    );
    assert_eq!(index_of(dir.path()).items.len(), 2);
}

#[test]
fn a_stage_that_tolerates_failed_items_records_them_as_absent() {
    let repo = repo();
    let tolerant = RECIPE.replace(
        "    concurrency: 2\n",
        "    concurrency: 2\n    tolerate_failed_items: 1\n",
    );
    let resolved = resolve(&repo.root, &tolerant, connections()).unwrap();
    let dir = tempfile::tempdir().unwrap();
    run_with(&resolved, dir.path(), &Serving(Flaky::new(0, true))).unwrap();
    let index = index_of(dir.path());
    assert_eq!(index.items.len(), 1);
    assert_eq!(index.rejected.len(), 1);
    assert_eq!(index.rejected[0].id, "z.md");
    assert!(index.rejected[0].reason.contains("failed"), "{index:?}");

    // A share is a bound too, and past it the stage fails.
    let strict = RECIPE.replace(
        "    concurrency: 2\n",
        "    concurrency: 2\n    tolerate_failed_items: 0.4\n",
    );
    let resolved = resolve(&repo.root, &strict, connections()).unwrap();
    let other = tempfile::tempdir().unwrap();
    let error = run_with(&resolved, other.path(), &Serving(Flaky::new(0, true)))
        .unwrap_err()
        .to_string();
    assert!(error.contains("z.md"), "{error}");
}
