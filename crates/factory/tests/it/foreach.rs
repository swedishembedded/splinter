// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements data pipelines that fan work out over large
// corpora and survive crashes without redoing finished items, for its
// clients. If your team needs expertise in resumable data factories, you can
// procure our services by sending an email to info@swedishembedded.com.

//! `foreach`: a stage runs its operation once per item of a collection, at
//! most `concurrency` at once; every item is committed on its own, so a
//! crashed run resumes without repeating a finished item and a changed item
//! recomputes only itself; the stage's product is a deterministic reduction
//! listing every item produced and every item the operation rejected.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::Value;
use splinter_core::factory::execution::Isolation;
use splinter_core::factory::items::{ItemIndex, ITEMS_FILE};
use splinter_core::factory::operation::{
    ArtifactKind, OperationDescriptor, Port, Reproducibility, OPERATION_SCHEMA,
};
use splinter_factory::execute::{Call, Execute, Outcome};
use splinter_factory::registry::{OperationSpec, Registry};
use splinter_factory::resolve::{resolve_with, Resolved};
use splinter_factory::run::{fork, Factory, RunError, Services};

use crate::fakes::{FakeBackend, Outcome as Trial};
use crate::repo::{connections, Repo};

/// What the test operations did, shared with the test.
#[derive(Default)]
struct Log {
    /// Executions per item (file name).
    calls: Mutex<BTreeMap<String, usize>>,
    running: AtomicUsize,
    most_at_once: AtomicUsize,
    /// Items that fail (once each) instead of producing.
    fail_once: Mutex<Vec<String>>,
}

impl Log {
    fn calls(&self) -> BTreeMap<String, usize> {
        self.calls.lock().unwrap().clone()
    }
}

/// `text.upper`: one text file in, the same file upper-cased out; a text
/// that says REJECT is rejected.
struct Upper(Arc<Log>);

/// `text.join`: every item of a collection, joined in its index's order.
struct Join;

fn port(name: &str) -> Port {
    Port {
        name: name.into(),
        kind: ArtifactKind::Files,
        many: false,
        optional: false,
    }
}

fn descriptor(name: &str, input: &str, output: &str) -> OperationDescriptor {
    OperationDescriptor {
        schema: OPERATION_SCHEMA.into(),
        name: name.into(),
        revision: 1,
        summary: name.into(),
        inputs: vec![port(input)],
        outputs: vec![port(output)],
        parameters: serde_json::json!({ "type": "object", "additionalProperties": false }),
        reproducibility: Reproducibility::Deterministic,
        isolation: Isolation::Any,
    }
}

fn no_params(params: &Value) -> Result<(), String> {
    match params.as_object() {
        Some(map) if map.is_empty() => Ok(()),
        _ => Err("takes no parameters".into()),
    }
}

impl OperationSpec for Upper {
    fn descriptor(&self) -> OperationDescriptor {
        descriptor("text.upper", "text", "upper")
    }
    fn check(&self, params: &Value) -> Result<(), String> {
        no_params(params)
    }
    fn executor(&self) -> Option<&dyn Execute> {
        Some(self)
    }
}

impl Execute for Upper {
    fn implementation(&self) -> String {
        "test upper 1".into()
    }

    fn execute(&self, call: &Call<'_>) -> Result<Outcome, String> {
        let input = &call.inputs["text"][0];
        let name = input.file_name().unwrap().to_string_lossy().into_owned();
        let log = &self.0;
        *log.calls.lock().unwrap().entry(name.clone()).or_default() += 1;
        let now = log.running.fetch_add(1, Ordering::SeqCst) + 1;
        log.most_at_once.fetch_max(now, Ordering::SeqCst);
        std::thread::sleep(std::time::Duration::from_millis(30));
        log.running.fetch_sub(1, Ordering::SeqCst);
        let mut failing = log.fail_once.lock().unwrap();
        if let Some(at) = failing.iter().position(|f| *f == name) {
            failing.remove(at);
            return Err(format!("{name}: the model endpoint went away"));
        }
        let text = std::fs::read_to_string(input).map_err(|e| e.to_string())?;
        if text.contains("REJECT") {
            return Ok(Outcome::Rejected("asks to be rejected".into()));
        }
        std::fs::write(call.out.join(&name), text.to_uppercase()).map_err(|e| e.to_string())?;
        Ok(Outcome::Produced)
    }
}

impl OperationSpec for Join {
    fn descriptor(&self) -> OperationDescriptor {
        descriptor("text.join", "parts", "joined")
    }
    fn check(&self, params: &Value) -> Result<(), String> {
        no_params(params)
    }
    fn executor(&self) -> Option<&dyn Execute> {
        Some(self)
    }
}

impl Execute for Join {
    fn implementation(&self) -> String {
        "test join 1".into()
    }

    fn execute(&self, call: &Call<'_>) -> Result<Outcome, String> {
        let parts = &call.inputs["parts"][0];
        let index: ItemIndex = serde_json::from_str(
            &std::fs::read_to_string(parts.join(ITEMS_FILE)).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let mut joined = String::new();
        for item in &index.items {
            let dir = parts.join(&item.path);
            let mut files: Vec<_> = std::fs::read_dir(&dir)
                .map_err(|e| e.to_string())?
                .map(|e| e.unwrap().path())
                .collect();
            files.sort();
            for file in files {
                joined += &std::fs::read_to_string(file).map_err(|e| e.to_string())?;
                joined.push('\n');
            }
        }
        std::fs::write(call.out.join("joined.txt"), joined).map_err(|e| e.to_string())?;
        Ok(Outcome::Produced)
    }
}

const RECIPE: &str = "schema: splinter.recipe/v2
name: shout
inputs:
  books: { kind: files, path: books }
stages:
  - id: upper
    uses: text.upper
    foreach: $inputs.books
    concurrency: 2
    inputs: { text: $item }
  - id: joined
    uses: text.join
    inputs: { parts: $stages.upper.upper }
outputs: { joined: $stages.joined.joined }
";

fn registry(log: &Arc<Log>) -> Registry {
    splinter_factory::ops::registry()
        .with(Upper(Arc::clone(log)))
        .with(Join)
}

fn books(repo: &Repo) {
    repo.write("books/a.txt", "alpha");
    repo.write("books/b.txt", "beta");
    repo.write("books/c.txt", "REJECT me");
    repo.write("books/d.txt", "delta");
}

fn resolved(repo: &Repo, recipe: &str, log: &Arc<Log>) -> Resolved {
    resolve_with(&repo.root, recipe, connections(), registry(log)).unwrap()
}

fn run(resolved: &Resolved, dir: &Path) -> Result<(), RunError> {
    let backend = FakeBackend::with(Trial::Pass);
    let services = Services {
        backend: &backend,
        trainer: None,
        models: None,
        variables: None,
    };
    let mut factory = Factory::open(resolved, dir, services, "test")?;
    let ran = factory.run().map(|_| ());
    drop(factory);
    ran
}

#[test]
fn a_fan_out_runs_each_item_at_most_n_at_once_and_reduces_them_in_order() {
    let repo = Repo::new();
    books(&repo);
    let log = Arc::new(Log::default());
    let dir = tempfile::tempdir().unwrap();
    run(&resolved(&repo, RECIPE, &log), dir.path()).unwrap();

    assert!(log.calls().values().all(|n| *n == 1), "{:?}", log.calls());
    assert!(log.most_at_once.load(Ordering::SeqCst) <= 2);

    let index: ItemIndex = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("upper").join(ITEMS_FILE)).unwrap(),
    )
    .unwrap();
    let produced: Vec<&str> = index.items.iter().map(|i| i.id.as_str()).collect();
    assert_eq!(produced, ["a.txt", "b.txt", "d.txt"]);
    assert_eq!(index.rejected.len(), 1);
    assert_eq!(index.rejected[0].id, "c.txt");
    assert!(index.rejected[0].reason.contains("asks to be rejected"));
    assert_eq!(
        std::fs::read_to_string(dir.path().join("joined/joined.txt")).unwrap(),
        "ALPHA\nBETA\nDELTA\n"
    );
}

#[test]
fn a_crashed_fan_out_resumes_without_repeating_a_finished_item() {
    let repo = Repo::new();
    books(&repo);
    let log = Arc::new(Log::default());
    log.fail_once.lock().unwrap().push("b.txt".into());
    let sequential = RECIPE.replace("concurrency: 2", "concurrency: 1");
    let resolved = resolved(&repo, &sequential, &log);
    let dir = tempfile::tempdir().unwrap();

    let error = run(&resolved, dir.path()).unwrap_err().to_string();
    assert!(error.contains("b.txt"), "{error}");
    assert_eq!(log.calls().get("c.txt"), None, "nothing after the failure");

    run(&resolved, dir.path()).unwrap();
    assert_eq!(
        log.calls(),
        BTreeMap::from([
            ("a.txt".into(), 1),
            ("b.txt".into(), 2),
            ("c.txt".into(), 1),
            ("d.txt".into(), 1),
        ])
    );
}

#[test]
fn one_changed_item_recomputes_only_itself() {
    let repo = Repo::new();
    books(&repo);
    let log = Arc::new(Log::default());
    let first = tempfile::tempdir().unwrap();
    run(&resolved(&repo, RECIPE, &log), first.path()).unwrap();

    repo.write("books/d.txt", "delta, revised");
    let second = tempfile::tempdir().unwrap();
    fork(first.path(), second.path()).unwrap();
    run(&resolved(&repo, RECIPE, &log), second.path()).unwrap();

    assert_eq!(
        log.calls(),
        BTreeMap::from([
            ("a.txt".into(), 1),
            ("b.txt".into(), 1),
            ("c.txt".into(), 1),
            ("d.txt".into(), 2),
        ])
    );
    assert_eq!(
        std::fs::read_to_string(second.path().join("joined/joined.txt")).unwrap(),
        "ALPHA\nBETA\nDELTA, REVISED\n"
    );
}

#[test]
fn a_fan_out_is_refused_where_it_cannot_run() {
    let repo = Repo::new();
    books(&repo);
    let log = Arc::new(Log::default());
    let refused = |recipe: &str| {
        resolve_with(&repo.root, recipe, connections(), registry(&log))
            .unwrap_err()
            .0
            .join("\n")
    };
    let unbounded = refused(&RECIPE.replace("concurrency: 2", "concurrency: 0"));
    assert!(unbounded.contains("concurrency"), "{unbounded}");
    let alone = refused(&RECIPE.replace(
        "    uses: text.join\n",
        "    uses: text.join\n    concurrency: 2\n",
    ));
    assert!(
        alone.contains("concurrency") && alone.contains("foreach"),
        "{alone}"
    );
    let built_in = refused(
        "schema: splinter.recipe/v2
name: x
inputs:
  books: { kind: files, path: books }
stages:
  - id: tasks
    uses: tasks.pin
    foreach: $inputs.books
    inputs: { collection: $item }
",
    );
    assert!(built_in.contains("once per item"), "{built_in}");
}
