// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements experiment controllers that keep a fleet
// busy by running every independent step of a pipeline at once, for its
// clients. If your team needs expertise in concurrent, resumable
// machine-learning pipelines, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Attempt passes, trainings and evaluations run at once: passes interrupted
//! together each resume from their own slots and the status shows both
//! running, trainings hold the trainer one at a time while other stages run,
//! and evaluations sharing a baseline run it once.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;
use splinter_core::factory::attempt::AttemptReport;
use splinter_factory::backend::{BackendError, Batch, ExecutionBackend};
use splinter_factory::recipe::ModelRef;
use splinter_factory::resolve::resolve;
use splinter_factory::run::{status, Factory, Services};
use splinter_factory::training::{TrainError, TrainRequest, Trained, Trainer};

use crate::factory_run::decisive;
use crate::fakes::{FakeBackend, FakeTrainer, Outcome as Trial};
use crate::repo::{connections, Repo, RECIPE as TRAINING_RECIPE};
use crate::stage_concurrency::bound;

/// Two attempt stages over the same tasks that do not depend on each other.
// seed: a recipe must state every seed; the fake backend draws nothing from them.
const TWO_PASSES: &str = r#"
schema: splinter.recipe/v2
name: two-passes
inputs:
  course: { kind: files, path: collections/vim.yaml }
models:
  student: "brain:Qwen/Qwen3-4B"
  teacher: "hosted:deepseek/deepseek-v4.1-flash"
stages:
  - id: tasks
    uses: tasks.pin
    inputs: { collection: $inputs.course }
    with: { split: train }
  - id: local
    uses: attempt.collect
    inputs: { tasks: $stages.tasks.tasks }
    with: { agent: sven, model: student, environment: environments/vim.yaml, attempts: 3, seed: 11 }
  - id: remote
    uses: attempt.collect
    inputs: { tasks: $stages.tasks.tasks }
    with: { agent: sven, model: teacher, environment: environments/vim.yaml, attempts: 3, seed: 12 }
"#;

/// A backend whose batches, while armed, wait until `together` of them are
/// in flight, note what the run's status says then, report the first
/// `reported` attempts and are killed.
struct Interrupting {
    inner: FakeBackend,
    run: PathBuf,
    together: usize,
    reported: usize,
    armed: AtomicBool,
    in_flight: Mutex<usize>,
    arrived: Condvar,
    /// The stages the run's status showed running while the batches were in flight.
    seen_running: Mutex<BTreeSet<String>>,
}

impl Interrupting {
    /// Block until `together` batches are in flight; false after a while
    /// without them.
    fn rendezvous(&self) -> bool {
        let mut in_flight = self.in_flight.lock().unwrap();
        *in_flight += 1;
        self.arrived.notify_all();
        let (count, _) = self
            .arrived
            .wait_timeout_while(in_flight, Duration::from_secs(20), |n| *n < self.together)
            .unwrap();
        *count >= self.together
    }
}

impl ExecutionBackend for Interrupting {
    fn identity(&self) -> String {
        self.inner.identity()
    }

    fn execute(&self, batch: Batch<'_>) -> Result<Vec<AttemptReport>, BackendError> {
        if !self.armed.load(Ordering::SeqCst) {
            return self.inner.execute(batch);
        }
        if !self.rendezvous() {
            return Err(BackendError::Worker(
                "the other stage's batch never ran beside this one".into(),
            ));
        }
        for stage in status(&self.run).unwrap() {
            if stage.state == "running" {
                self.seen_running.lock().unwrap().insert(stage.key);
            }
        }
        let first = Batch {
            specs: &batch.specs[..self.reported],
            ..batch
        };
        self.inner.execute(first)?;
        Err(BackendError::Worker("killed".into()))
    }

    fn reconcile(&self, out: &Path) -> Result<Vec<AttemptReport>, BackendError> {
        self.inner.reconcile(out)
    }
}

#[test]
fn stages_interrupted_together_each_resume_from_their_own_slots() {
    let repo = Repo::new();
    let resolved = resolve(&repo.root, TWO_PASSES, connections()).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let run_dir = std::fs::canonicalize(dir.path()).unwrap();
    let backend = Interrupting {
        inner: FakeBackend::with(Trial::Pass),
        run: run_dir.clone(),
        together: 2,
        reported: 2,
        armed: AtomicBool::new(true),
        in_flight: Mutex::new(0),
        arrived: Condvar::new(),
        seen_running: Mutex::new(BTreeSet::new()),
    };
    let run = || {
        let services = Services {
            backend: &backend,
            trainer: None,
            models: None,
            variables: None,
            online: None,
            build: splinter_core::build::BuildInfo::this_executable(),
        };
        let mut factory =
            Factory::open(&resolved, &run_dir, services, "test")?.with_concurrent_stages(bound(3));
        factory.run()
    };

    let error = run().unwrap_err().to_string();
    assert!(
        error.contains("local") && error.contains("remote") && error.contains("killed"),
        "both interrupted stages are reported: {error}"
    );
    assert_eq!(
        *backend.seen_running.lock().unwrap(),
        BTreeSet::from(["local".to_owned(), "remote".to_owned()]),
        "the status shows both stages running"
    );
    assert_eq!(
        backend.inner.executed().len(),
        4,
        "two attempts of each pass reported"
    );

    backend.armed.store(false, Ordering::SeqCst);
    run().unwrap();
    for pass in ["local", "remote"] {
        let mut slots: Vec<(String, u32)> = backend
            .inner
            .executed()
            .into_iter()
            .filter(|s| s.slot.pass == pass)
            .map(|s| (s.task.id, s.slot.slot))
            .collect();
        let executed = slots.len();
        slots.sort();
        slots.dedup();
        assert_eq!(slots.len(), 6, "{pass}: every slot ran");
        assert_eq!(
            executed, 6,
            "{pass}: an attempt reported before the kill ran again"
        );
    }
}

/// A trainer that takes a while and counts how many of its runs overlap.
struct Counting {
    inner: FakeTrainer,
    training: AtomicUsize,
    most_training: AtomicUsize,
    /// Whether an evaluation's attempts ran while a training did.
    evaluated_while_training: AtomicBool,
    /// Whether an evaluation's attempts have begun.
    evaluating: AtomicBool,
}

/// Wait (a while) until `ready`.
fn wait_for(ready: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while !ready() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
}

impl Trainer for Counting {
    fn identity(&self) -> String {
        self.inner.identity()
    }
    fn resolve(&self, method: &str, params: &Value) -> Result<Value, TrainError> {
        self.inner.resolve(method, params)
    }
    fn measure(&self, target: &ModelRef, lines: &[String]) -> Result<Vec<u64>, TrainError> {
        self.inner.measure(target, lines)
    }
    fn train(&self, request: &TrainRequest<'_>) -> Result<Trained, TrainError> {
        let now = self.training.fetch_add(1, Ordering::SeqCst) + 1;
        self.most_training.fetch_max(now, Ordering::SeqCst);
        // The second training waits for the first candidate's evaluation,
        // which became ready with it, to begin beside it.
        if !self.inner.trained.lock().unwrap().is_empty() {
            wait_for(|| self.evaluating.load(Ordering::SeqCst));
        }
        let trained = self.inner.train(request);
        self.training.fetch_sub(1, Ordering::SeqCst);
        trained
    }
    fn serve(&self, base: &ModelRef, adapter: Option<&Path>) -> Result<String, TrainError> {
        self.inner.serve(base, adapter)
    }
}

/// The fake backend, noting when an evaluation's attempts run during a training.
struct Watching<'t> {
    inner: FakeBackend,
    trainer: &'t Counting,
}

impl ExecutionBackend for Watching<'_> {
    fn identity(&self) -> String {
        self.inner.identity()
    }
    fn execute(&self, batch: Batch<'_>) -> Result<Vec<AttemptReport>, BackendError> {
        if batch.specs.iter().any(|s| s.slot.pass.starts_with("eval-")) {
            // The first candidate's evaluation becomes ready with the second
            // training: wait (a while) for that training to be in flight,
            // unless every training is over already.
            self.trainer.evaluating.store(true, Ordering::SeqCst);
            wait_for(|| {
                self.trainer.training.load(Ordering::SeqCst) > 0
                    || self.trainer.inner.trained.lock().unwrap().len() == 2
            });
            if self.trainer.training.load(Ordering::SeqCst) > 0 {
                self.trainer
                    .evaluated_while_training
                    .store(true, Ordering::SeqCst);
            }
        }
        self.inner.execute(batch)
    }
    fn reconcile(&self, out: &Path) -> Result<Vec<AttemptReport>, BackendError> {
        self.inner.reconcile(out)
    }
}

#[test]
fn independent_trainings_hold_the_trainer_one_at_a_time_while_other_stages_run() {
    let repo = Repo::new();
    decisive(&repo);
    let resolved = resolve(&repo.root, TRAINING_RECIPE, connections()).unwrap();
    let trainer = Counting {
        inner: FakeTrainer::default(),
        training: AtomicUsize::new(0),
        most_training: AtomicUsize::new(0),
        evaluated_while_training: AtomicBool::new(false),
        evaluating: AtomicBool::new(false),
    };
    let backend = Watching {
        inner: FakeBackend {
            candidate: Some(Trial::Pass),
            ..FakeBackend::with(Trial::Fail)
        },
        trainer: &trainer,
    };
    backend
        .inner
        .script("train/001-first", &[Trial::Pass, Trial::Pass]);
    let dir = tempfile::tempdir().unwrap();
    let services = Services {
        backend: &backend,
        trainer: Some(&trainer),
        models: None,
        variables: None,
        online: None,
        build: splinter_core::build::BuildInfo::this_executable(),
    };
    Factory::open(&resolved, dir.path(), services, "test")
        .unwrap()
        .with_concurrent_stages(bound(4))
        .run()
        .unwrap();
    assert_eq!(trainer.inner.trained.lock().unwrap().len(), 2);
    assert_eq!(
        trainer.most_training.load(Ordering::SeqCst),
        1,
        "two trainings shared the trainer"
    );
    assert!(
        trainer.evaluated_while_training.load(Ordering::SeqCst),
        "the first candidate's evaluation waited for the second training"
    );
}

#[test]
fn evaluations_sharing_a_baseline_run_together_and_run_it_once() {
    let repo = Repo::new();
    decisive(&repo);
    // Two evaluations of one candidate under one plan: their baseline arm
    // is one stage identity, viewed under each.
    let recipe = format!(
        "{}{}{}",
        TRAINING_RECIPE.split("  - id: lora-r16\n").next().unwrap(),
        crate::repo::evaluation("lora-r8", "eval/dev.yaml", "student"),
        crate::repo::evaluation("lora-r8", "eval/dev.yaml", "student")
            .replace("id: eval-lora-r8", "id: eval-again"),
    );
    let resolved = resolve(&repo.root, &recipe, connections()).unwrap();
    let backend = FakeBackend {
        candidate: Some(Trial::Pass),
        ..FakeBackend::with(Trial::Fail)
    };
    backend.script("train/001-first", &[Trial::Pass, Trial::Pass]);
    let trainer = FakeTrainer::default();
    let dir = tempfile::tempdir().unwrap();
    let services = Services {
        backend: &backend,
        trainer: Some(&trainer),
        models: None,
        variables: None,
        online: None,
        build: splinter_core::build::BuildInfo::this_executable(),
    };
    let manifest = Factory::open(&resolved, dir.path(), services, "test")
        .unwrap()
        .with_concurrent_stages(bound(3))
        .run()
        .unwrap();
    assert!(manifest.stages.iter().any(|s| s.key == "eval-again"));
    let baseline: Vec<_> = backend
        .executed()
        .into_iter()
        .filter(|s| s.slot.pass.starts_with("baseline-"))
        .collect();
    // One held-out task, eight attempts: run once, for both evaluations.
    assert_eq!(baseline.len(), 8, "the shared baseline ran more than once");
    // Each evaluation judged its candidate against all eight.
    for key in ["eval-lora-r8", "eval-again"] {
        let result = splinter_factory::run::eval::result_of(&dir.path().join(key)).unwrap();
        assert_eq!(result.baseline.unwrap().trials, 8, "{key}");
    }
}
