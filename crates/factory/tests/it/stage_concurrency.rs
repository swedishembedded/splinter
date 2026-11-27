// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements experiment controllers that keep a fleet
// busy by running every independent step of a pipeline at once, for its
// clients. If your team needs expertise in concurrent, resumable
// machine-learning pipelines, you can procure our services by sending an
// email to info@swedishembedded.com.

//! The controller runs a recipe's ready set: every stage whose inputs are
//! committed starts, at most a bound at once. A dependency waits for its
//! inputs, a bound of one is the sequential controller, a failure starts
//! nothing new while the running stages finish and every failure is
//! reported, and what a run commits does not depend on the order its stages
//! ran in. Attempt passes, trainings and evaluations at once:
//! `concurrent_passes`.

use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;
use splinter_core::factory::execution::Isolation;
use splinter_core::factory::operation::{
    ArtifactKind, OperationDescriptor, Port, Reproducibility, OPERATION_SCHEMA,
};
use splinter_factory::execute::{Call, Execute, Outcome};
use splinter_factory::registry::{OperationSpec, Registry};
use splinter_factory::resolve::{resolve_with, Resolved};
use splinter_factory::run::{
    fork, status, Factory, RunError, RunManifest, Services, DEFAULT_CONCURRENT_STAGES,
};

use crate::fakes::{FakeBackend, Outcome as Trial};
use crate::repo::{connections, Repo};

/// What the test operations did, shared with the test.
#[derive(Default)]
struct Log {
    /// When each stage (by its `name` parameter) ran: start and end.
    spans: Mutex<BTreeMap<String, Vec<(Instant, Instant)>>>,
    running: AtomicUsize,
    most_at_once: AtomicUsize,
    /// How long each stage takes, when not the default.
    millis: Mutex<BTreeMap<String, u64>>,
    /// The stages that fail instead of producing.
    failing: Mutex<BTreeSet<String>>,
}

impl Log {
    fn starts(&self) -> Vec<String> {
        let spans = self.spans.lock().unwrap();
        let mut all: Vec<(Instant, String)> = spans
            .iter()
            .flat_map(|(name, runs)| runs.iter().map(move |(start, _)| (*start, name.clone())))
            .collect();
        all.sort();
        all.into_iter().map(|(_, name)| name).collect()
    }

    fn span(&self, name: &str) -> (Instant, Instant) {
        self.spans.lock().unwrap()[name][0]
    }

    fn runs(&self, name: &str) -> usize {
        self.spans.lock().unwrap().get(name).map_or(0, Vec::len)
    }
}

/// `test.slow`: one text in, the text with the stage's name out, after a
/// pause the log sets.
struct Slow(Arc<Log>);

/// `test.gather`: every part, in the order bound.
struct Gather(Arc<Log>);

fn port(name: &str, many: bool) -> Port {
    Port {
        name: name.into(),
        kind: ArtifactKind::Files,
        many,
        optional: false,
    }
}

fn descriptor(name: &str, input: Port, output: &str) -> OperationDescriptor {
    OperationDescriptor {
        schema: OPERATION_SCHEMA.into(),
        name: name.into(),
        revision: 1,
        summary: name.into(),
        inputs: vec![input],
        outputs: vec![port(output, false)],
        parameters: serde_json::json!({
            "type": "object",
            "properties": { "name": { "type": "string" } },
            "required": ["name"],
            "additionalProperties": false
        }),
        reproducibility: Reproducibility::Deterministic,
        isolation: Isolation::Any,
    }
}

fn named(params: &Value) -> Result<(), String> {
    params
        .get("name")
        .and_then(Value::as_str)
        .map(|_| ())
        .ok_or_else(|| "names its stage".into())
}

/// Record one run of the stage `name` around `body`.
fn timed<T>(log: &Log, name: &str, body: impl FnOnce() -> T) -> T {
    let now = log.running.fetch_add(1, Ordering::SeqCst) + 1;
    log.most_at_once.fetch_max(now, Ordering::SeqCst);
    let start = Instant::now();
    let pause = log.millis.lock().unwrap().get(name).copied().unwrap_or(300);
    std::thread::sleep(Duration::from_millis(pause));
    let out = body();
    log.running.fetch_sub(1, Ordering::SeqCst);
    log.spans
        .lock()
        .unwrap()
        .entry(name.to_owned())
        .or_default()
        .push((start, Instant::now()));
    out
}

/// Every `.txt` file below `dir`, in path order.
fn texts(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(at) = stack.pop() {
        for entry in std::fs::read_dir(&at).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|x| x == "txt") {
                found.push(path);
            }
        }
    }
    found.sort();
    found
}

fn name_of(call: &Call<'_>) -> String {
    call.params["name"].as_str().unwrap().to_owned()
}

impl OperationSpec for Slow {
    fn descriptor(&self) -> OperationDescriptor {
        descriptor("test.slow", port("text", false), "out")
    }
    fn check(&self, params: &Value) -> Result<(), String> {
        named(params)
    }
    fn executor(&self) -> Option<&dyn Execute> {
        Some(self)
    }
}

impl Execute for Slow {
    fn implementation(&self) -> String {
        "test slow 1".into()
    }

    fn execute(&self, call: &Call<'_>) -> Result<Outcome, String> {
        let name = name_of(call);
        timed(&self.0, &name, || {
            if self.0.failing.lock().unwrap().contains(&name) {
                return Err(format!("{name}: the endpoint went away"));
            }
            let text = texts(&call.inputs["text"][0])
                .iter()
                .map(|f| std::fs::read_to_string(f).map_err(|e| e.to_string()))
                .collect::<Result<Vec<_>, _>>()?
                .join(" ");
            std::fs::write(
                call.out.join(format!("{name}.txt")),
                format!("{text} {name}"),
            )
            .map_err(|e| e.to_string())?;
            Ok(Outcome::Produced)
        })
    }
}

impl OperationSpec for Gather {
    fn descriptor(&self) -> OperationDescriptor {
        descriptor("test.gather", port("parts", true), "gathered")
    }
    fn check(&self, params: &Value) -> Result<(), String> {
        named(params)
    }
    fn executor(&self) -> Option<&dyn Execute> {
        Some(self)
    }
}

impl Execute for Gather {
    fn implementation(&self) -> String {
        "test gather 1".into()
    }

    fn execute(&self, call: &Call<'_>) -> Result<Outcome, String> {
        let name = name_of(call);
        timed(&self.0, &name, || {
            let mut gathered = String::new();
            for part in &call.inputs["parts"] {
                for file in texts(part) {
                    gathered += &std::fs::read_to_string(file).map_err(|e| e.to_string())?;
                    gathered.push('\n');
                }
            }
            std::fs::write(call.out.join("gathered.txt"), gathered).map_err(|e| e.to_string())?;
            Ok(Outcome::Produced)
        })
    }
}

/// Three independent stages and one that gathers them; `d` depends on `a`
/// alone.
const FAN: &str = "schema: splinter.recipe/v2
name: fan
inputs:
  seed: { kind: files, path: seed }
stages:
  - id: a
    uses: test.slow
    inputs: { text: $inputs.seed }
    with: { name: a }
  - id: b
    uses: test.slow
    inputs: { text: $inputs.seed }
    with: { name: b }
  - id: c
    uses: test.slow
    inputs: { text: $inputs.seed }
    with: { name: c }
  - id: d
    uses: test.slow
    inputs: { text: $stages.a.out }
    with: { name: d }
  - id: gathered
    uses: test.gather
    inputs: { parts: [$stages.a.out, $stages.b.out, $stages.c.out] }
    with: { name: gathered }
outputs: { gathered: $stages.gathered.gathered }
";

fn fan(log: &Arc<Log>) -> (Repo, Resolved) {
    let repo = Repo::new();
    repo.write("seed/seed.txt", "seed");
    let registry: Registry = splinter_factory::ops::registry()
        .with(Slow(Arc::clone(log)))
        .with(Gather(Arc::clone(log)));
    let resolved = resolve_with(&repo.root, FAN, connections(), registry).unwrap();
    (repo, resolved)
}

pub(crate) fn bound(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

fn run_fan(resolved: &Resolved, dir: &Path, at_once: usize) -> Result<RunManifest, RunError> {
    let backend = FakeBackend::with(Trial::Pass);
    let services = Services {
        backend: &backend,
        trainer: None,
        models: None,
        variables: None,
        online: None,
        build: splinter_core::build::BuildInfo::this_executable(),
    };
    let mut factory =
        Factory::open(resolved, dir, services, "test")?.with_concurrent_stages(bound(at_once));
    factory.run()
}

#[test]
fn independent_stages_run_together_up_to_the_bound_and_a_dependency_waits_for_its_inputs() {
    for at_once in [2, 3] {
        let log = Arc::new(Log::default());
        let (_repo, resolved) = fan(&log);
        let dir = tempfile::tempdir().unwrap();
        let manifest = run_fan(&resolved, dir.path(), at_once).unwrap();

        assert_eq!(
            log.most_at_once.load(Ordering::SeqCst),
            at_once,
            "the bound is {at_once}"
        );
        // Each stage ran once, and the manifest lists them in plan order.
        for name in ["a", "b", "c", "d", "gathered"] {
            assert_eq!(log.runs(name), 1, "{name}");
        }
        let keys: Vec<&str> = manifest.stages.iter().map(|s| s.key.as_str()).collect();
        assert_eq!(keys, ["a", "b", "c", "d", "gathered"]);
        // A stage starts only once every stage it consumes has finished.
        let gathered = log.span("gathered").0;
        for input in ["a", "b", "c"] {
            assert!(
                log.span(input).1 <= gathered,
                "gathered began before {input} ended"
            );
        }
        assert!(log.span("a").1 <= log.span("d").0, "d began before a ended");
        assert_eq!(
            std::fs::read_to_string(dir.path().join("gathered/gathered.txt")).unwrap(),
            "seed a\nseed b\nseed c\n"
        );
        // The report says how many ran at once.
        let report = std::fs::read_to_string(dir.path().join("report.md")).unwrap();
        assert!(
            report.contains(&format!("{at_once} at once")),
            "the report does not show the stages that ran together:\n{report}"
        );
    }
}

#[test]
fn a_bound_of_one_runs_every_stage_alone_in_plan_order() {
    let log = Arc::new(Log::default());
    // The last independent stage is the quickest: run together, it would end first.
    log.millis.lock().unwrap().insert("c".into(), 10);
    let (_repo, resolved) = fan(&log);
    let dir = tempfile::tempdir().unwrap();
    run_fan(&resolved, dir.path(), 1).unwrap();
    assert_eq!(log.most_at_once.load(Ordering::SeqCst), 1);
    assert_eq!(log.starts(), ["a", "b", "c", "d", "gathered"]);
}

#[test]
fn the_default_bound_runs_independent_stages_together() {
    assert!(DEFAULT_CONCURRENT_STAGES.get() >= 3);
    let log = Arc::new(Log::default());
    let (_repo, resolved) = fan(&log);
    let dir = tempfile::tempdir().unwrap();
    let backend = FakeBackend::with(Trial::Pass);
    let services = Services {
        backend: &backend,
        trainer: None,
        models: None,
        variables: None,
        online: None,
        build: splinter_core::build::BuildInfo::this_executable(),
    };
    Factory::open(&resolved, dir.path(), services, "test")
        .unwrap()
        .run()
        .unwrap();
    assert_eq!(log.most_at_once.load(Ordering::SeqCst), 3);
}

#[test]
fn a_failure_starts_no_new_stage_lets_the_running_ones_commit_and_every_failure_is_reported() {
    let log = Arc::new(Log::default());
    log.millis.lock().unwrap().insert("a".into(), 50);
    log.millis.lock().unwrap().insert("b".into(), 600);
    log.failing.lock().unwrap().insert("a".into());
    let (_repo, resolved) = fan(&log);
    let dir = tempfile::tempdir().unwrap();

    let error = run_fan(&resolved, dir.path(), 2).unwrap_err().to_string();
    assert!(error.contains("a: the endpoint went away"), "{error}");
    // b was running when a failed: it finished and is committed; nothing
    // else started.
    assert_eq!(log.runs("b"), 1);
    for never in ["c", "d", "gathered"] {
        assert_eq!(log.runs(never), 0, "{never} started after the failure");
    }
    let states: BTreeMap<String, String> = status(dir.path())
        .unwrap()
        .into_iter()
        .map(|s| (s.key, s.state))
        .collect();
    assert_eq!(states["a"], "failed");
    assert_eq!(states["b"], "completed");

    // Two stages failing together are both reported.
    log.failing.lock().unwrap().insert("c".into());
    log.millis.lock().unwrap().insert("a".into(), 300);
    log.millis.lock().unwrap().insert("c".into(), 300);
    let error = run_fan(&resolved, dir.path(), 3).unwrap_err().to_string();
    assert!(
        error.contains("a: the endpoint went away") && error.contains("c: the endpoint went away"),
        "{error}"
    );
    assert_eq!(log.runs("b"), 1, "a committed stage is reused");

    // Mended, the run finishes what is left and reuses the rest.
    log.failing.lock().unwrap().clear();
    let manifest = run_fan(&resolved, dir.path(), 3).unwrap();
    let computed: BTreeMap<&str, bool> = manifest
        .stages
        .iter()
        .map(|s| (s.key.as_str(), s.computed))
        .collect();
    assert!(!computed["b"]);
    assert!(computed["a"] && computed["c"] && computed["d"] && computed["gathered"]);
    assert_eq!(log.runs("b"), 1);
}

#[test]
fn what_a_run_commits_does_not_depend_on_the_order_its_stages_ran_in() {
    let log = Arc::new(Log::default());
    let (_repo, resolved) = fan(&log);
    let sequential = tempfile::tempdir().unwrap();
    let one = run_fan(&resolved, sequential.path(), 1).unwrap();

    // Run together, and in the reverse order of their ends.
    for (name, millis) in [("a", 500), ("b", 250), ("c", 10)] {
        log.millis.lock().unwrap().insert(name.into(), millis);
    }
    let together = tempfile::tempdir().unwrap();
    let three = run_fan(&resolved, together.path(), 3).unwrap();

    // Every stage's output is the same bytes. (Its `stage.json`, its
    // identity, names the products it consumed, and a product's digest names
    // when it was committed: two runs of a recipe differ there, however their
    // stages were ordered.)
    let files = |dir: &Path, key: &str| -> Vec<Value> {
        let manifest: Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join(key).join("manifest.json")).unwrap(),
        )
        .unwrap();
        manifest["files"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|f| f["path"] != "stage.json")
            .cloned()
            .collect()
    };
    for (a, b) in one.stages.iter().zip(&three.stages) {
        assert_eq!(a.key, b.key);
        assert_eq!(
            files(sequential.path(), &a.key),
            files(together.path(), &b.key)
        );
    }

    // A fork of the sequential run, run together, reuses every product
    // under the same identities.
    let forked = tempfile::tempdir().unwrap();
    fork(sequential.path(), forked.path()).unwrap();
    let again = run_fan(&resolved, forked.path(), 3).unwrap();
    assert!(
        again.stages.iter().all(|s| !s.computed),
        "{:?}",
        again.stages
    );
    let ids = |m: &RunManifest| -> Vec<_> {
        m.stages
            .iter()
            .map(|s| (s.key.clone(), s.stage.clone(), s.product.clone()))
            .collect()
    };
    assert_eq!(ids(&one), ids(&again));
}
