// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that run model learning as one
// resumable, multi-generation campaign with a single promotion gate and
// rollback, for its clients. If your team needs expertise in self-improving
// training loops you can procure our services by sending an email to
// info@swedishembedded.com.

//! The world the campaign specs run in: a state root whose brain is the
//! scripted HTTP service (training is a real served job under a
//! `resume_key`), practice that writes the examiner's export of
//! deterministic trials (admitted by the real intake, curated by the real
//! dataset build), and an examiner that signs `bench.examination/1`
//! verdicts with a test key and keeps burn records on disk as bench does.
//! Each stand-in can be told to die at a chosen point, as a killed process
//! would: a panic in the middle of its work.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use splinter_agent::CancelToken;
use splinter_core::clock::FixedClock;
use splinter_eval::promotion::VerdictKey;
use splinter_model::scripted_endpoint::{ScriptedEndpoint, SCRIPTED_KEY};
use splinter_model::scripted_jobs::JobScript;
use splinter_model::served_train::ArtifactStore;
use splinter_model::BrainEndpoints;
use splinter_orchestrator::error::OrchestratorError;
use splinter_orchestrator::Context;
use splinter_pipelines::campaign::{
    CampaignController, CampaignError, CampaignSpec, ExamRequest, Examined, Examiner, Model, Parts,
    Rolled, RolloutRequest, Rollouts, Stopwatch, Trial,
};
use splinter_pipelines::train::{BrainTrainer, TrainPlan, Trainer};

use crate::common::bench::{export, Trial as BenchTrial};
use crate::common::{config, Scratch};

/// The practice tasks.
pub const TASKS: [&str; 4] = ["alpha", "beta", "gamma", "delta"];

/// The test key the examiner stand-in signs with.
pub fn key() -> VerdictKey {
    VerdictKey::from_bytes([42; 32])
}

/// A scratch state root whose brain is a scripted training service.
pub struct Rig {
    pub scratch: Scratch,
    pub ctx: Context,
    pub brain: ScriptedEndpoint,
}

impl Rig {
    /// A fresh rig under the scratch directory named `test`; the same name
    /// gives the same absolute paths.
    pub fn new(test: &str) -> Self {
        let scratch = Scratch::new(test);
        let store = scratch.0.join("artifacts");
        std::fs::create_dir_all(&store).unwrap();
        let brain = ScriptedEndpoint::start(vec![]).with_training(JobScript::over(store.clone()));
        let mut settings = config(&scratch);
        settings.brain = Some(
            BrainEndpoints::new(vec![brain.url()], SCRIPTED_KEY)
                .unwrap()
                .with_artifact_store(ArtifactStore::open(&store).unwrap()),
        );
        let ctx = Context::new(settings, false)
            .unwrap()
            .with_clock(Box::new(FixedClock::new("2026-11-18T09:00:00.000Z")));
        Self {
            scratch,
            ctx,
            brain,
        }
    }

    /// A campaign of `generations` over the policy base, seeded `seed`.
    pub fn spec(&self, name: &str, generations: u32, seed: u64) -> CampaignSpec {
        let mut spec = CampaignSpec::parse(&format!(
            "name = {name:?}\ngenerations = {generations}\nbase = \"local:{}\"\nseed = \"{seed}\"\n\
             [rollout]\nattempts = 2\nretire_at = 0.9\nretire_after = 4\n[train]\nrank = 4\nsteps = 2\n",
            self.ctx.config().policy_base.display()
        ))
        .unwrap();
        spec.train.records_per_step = Some(1);
        spec
    }
}

/// Whether a champion of `generation` passes attempt `attempt` of `task`: a
/// later champion passes more.
pub fn passes(generation: u32, task: &str, attempt: u32) -> bool {
    let rank = TASKS.iter().position(|t| *t == task).unwrap() as u32;
    rank + attempt <= generation + 2
}

/// Practice that writes the examiner's export of deterministic trials.
#[derive(Default)]
pub struct Practice {
    /// Rollouts run (not reused).
    pub ran: Cell<usize>,
    /// Die after writing this many trials of this generation's rollout,
    /// once.
    pub kill_at: Cell<Option<(u32, usize)>>,
    /// Every rollout's champion generation, in order.
    pub champions: RefCell<Vec<u32>>,
}

impl Rollouts for Practice {
    fn identity(&self) -> Value {
        json!({"practice": "scripted", "tasks": TASKS})
    }

    fn tasks(&self) -> Result<Vec<String>, CampaignError> {
        Ok(TASKS.iter().map(ToString::to_string).collect())
    }

    fn rollout(
        &mut self,
        _ctx: &Context,
        request: &RolloutRequest<'_>,
        _cancel: &CancelToken,
    ) -> Result<Rolled, CampaignError> {
        self.ran.set(self.ran.get() + 1);
        self.champions
            .borrow_mut()
            .push(request.champion.generation);
        let generation = request.champion.generation;
        let mut planned: Vec<(String, bool, &'static str, &'static str)> = Vec::new();
        for task in request.tasks {
            for attempt in 1..=request.attempts {
                let name: &'static str =
                    Box::leak(format!("g{}-{task}-{attempt}", request.generation).into_boxed_str());
                let id: &'static str = Box::leak(format!("org/{task}").into_boxed_str());
                planned.push((task.clone(), passes(generation, task, attempt), name, id));
            }
        }
        let trials = |n: usize| -> Vec<BenchTrial> {
            planned
                .iter()
                .take(n)
                .map(|(_, pass, name, id)| BenchTrial::new(name, id, Some("train"), *pass).scored())
                .collect()
        };
        let dir = request.dir.join("export");
        // A rollout starts over: what a killed attempt left is replaced.
        let _ = std::fs::remove_dir_all(&dir);
        if let Some((at, after)) = self.kill_at.get().filter(|(g, _)| *g == request.generation) {
            self.kill_at.set(None);
            export(&dir, &trials(after));
            panic!("killed mid-rollout of generation {at} after {after} trials");
        }
        export(&dir, &trials(planned.len()));
        Ok(Rolled {
            export: dir,
            trials: planned
                .into_iter()
                .map(|(task, pass, _, _)| Trial {
                    task,
                    reward: Some(if pass { 1.0 } else { 0.0 }),
                    passed: Some(pass),
                })
                .collect(),
            tokens: Some(15 * u64::from(request.attempts) * request.tasks.len() as u64),
        })
    }
}

/// What the examiner stand-in says about a generation's candidate: the dev
/// gain's interval and the retention bound.
#[derive(Clone, Copy, Debug)]
pub struct Says {
    pub dev: [f64; 2],
    pub retention_lower: f64,
}

/// A gain shown and retention kept.
pub const GOOD: Says = Says {
    dev: [0.05, 0.25],
    retention_lower: -0.01,
};

/// How a verdict is spoiled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Spoil {
    /// Signed under another key.
    Forged,
    /// About another candidate.
    OtherCandidate,
}

/// An examiner that signs aggregate verdicts with [`key`] and keeps a burn
/// record per sealed slice in its private directory, as bench does.
pub struct Exam {
    pub private: PathBuf,
    pub says: BTreeMap<u32, Says>,
    pub spoil: Option<(u32, Spoil)>,
    pub ran: Cell<usize>,
    key: VerdictKey,
}

impl Exam {
    pub fn new(private: &Path, says: &[Says]) -> Self {
        std::fs::create_dir_all(private).unwrap();
        Self {
            private: private.to_path_buf(),
            says: says
                .iter()
                .enumerate()
                .map(|(i, s)| (u32::try_from(i).unwrap() + 1, *s))
                .collect(),
            spoil: None,
            ran: Cell::new(0),
            key: key(),
        }
    }

    /// The burn record of `exam`: the first pair examined over it.
    fn burn(&self, exam: &str, pair: &Value) -> Value {
        let path = self.private.join(format!("burn-{exam}.json"));
        if let Ok(text) = std::fs::read_to_string(&path) {
            return serde_json::from_str(&text).unwrap();
        }
        let record = json!({"exam_sha256": exam, "candidate": pair["candidate"], "predecessor": pair["predecessor"]});
        std::fs::write(&path, record.to_string()).unwrap();
        record
    }

    fn verdict(
        &self,
        request: &ExamRequest<'_>,
        collection: &str,
        sealed: bool,
        interval: [f64; 2],
        lower: f64,
    ) -> Value {
        let mut candidate = request.candidate.adapter.clone();
        if self.spoil == Some((request.generation, Spoil::OtherCandidate)) {
            candidate = Some("sha256:another".into());
        }
        let pair = json!({"candidate": candidate, "predecessor": request.predecessor.adapter});
        let exam = format!("{collection}-exam");
        let doc = json!({
            "schema": "bench.examination/1", "role": "candidate", "valid": true, "invalid_reasons": [],
            "integrity_flags": [], "alpha": 0.05,
            "candidate": {"adapter_digest": candidate}, "predecessor": {"adapter_digest": request.predecessor.adapter},
            "collection": {"name": collection, "exam_sha256": exam, "items": 120, "sealed": sealed},
            "paired_gain": {"mean": (interval[0] + interval[1]) / 2.0, "interval": interval,
                            "one_sided_lower": lower, "families": 12},
            "burn": if sealed { self.burn(&exam, &pair) } else { Value::Null },
        });
        if self.spoil == Some((request.generation, Spoil::Forged)) && !sealed && collection == "dev"
        {
            VerdictKey::from_bytes([1; 32]).sign(&doc)
        } else {
            self.key.sign(&doc)
        }
    }
}

impl Examiner for Exam {
    fn identity(&self) -> Value {
        json!({"examiner": "scripted"})
    }

    fn slices(&self) -> Vec<String> {
        (1..=4).map(|g| format!("slice-{g}")).collect()
    }

    fn key(&self) -> &VerdictKey {
        &self.key
    }

    fn examine(
        &mut self,
        _ctx: &Context,
        request: &ExamRequest<'_>,
        _cancel: &CancelToken,
    ) -> Result<Examined, CampaignError> {
        self.ran.set(self.ran.get() + 1);
        assert!(
            request.training.is_file(),
            "the examiner is handed the training records to audit"
        );
        let says = self.says[&request.generation];
        let dev = self.verdict(request, "dev", false, says.dev, says.dev[0] + 0.01);
        let retention = self.verdict(
            request,
            "anchor",
            false,
            [says.retention_lower - 0.02, 0.03],
            says.retention_lower,
        );
        let sealed = self.verdict(request, request.slice, true, [0.0, 0.2], 0.02);
        let examined = Examined {
            dev: request.dir.join("dev.json"),
            retention: request.dir.join("retention.json"),
            sealed: request.dir.join("sealed.json"),
        };
        for (path, doc) in [
            (&examined.dev, dev),
            (&examined.retention, retention),
            (&examined.sealed, sealed),
        ] {
            std::fs::write(path, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
        }
        Ok(examined)
    }
}

/// brain's trainer, which dies once right after brain finished the job of
/// its `kill` generation's candidate: the job ran, Splinter never kept it.
#[derive(Default)]
pub struct Dying {
    pub kill: Cell<Option<String>>,
}

impl Trainer for Dying {
    fn train(
        &self,
        ctx: &Context,
        plan: &TrainPlan,
        cancel: &CancelToken,
    ) -> Result<splinter_model::train::Trained, OrchestratorError> {
        let trained = BrainTrainer.train(ctx, plan, cancel)?;
        if let Some(kill) = self.kill.take() {
            if plan.candidate.contains(&kill) {
                panic!(
                    "killed mid-training, after brain finished {}",
                    plan.candidate
                );
            }
            self.kill.set(Some(kill));
        }
        Ok(trained)
    }

    fn train_preference(
        &self,
        ctx: &Context,
        plan: &TrainPlan,
        cancel: &CancelToken,
    ) -> Result<splinter_model::train::TrainedPreference, OrchestratorError> {
        BrainTrainer.train_preference(ctx, plan, cancel)
    }
}

/// One second per reading: every stage attempt measures the same time.
#[derive(Default)]
pub struct Ticks(Cell<f64>);

impl Stopwatch for Ticks {
    fn seconds(&self) -> f64 {
        self.0.set(self.0.get() + 1.0);
        self.0.get()
    }
}

/// The stand-ins a campaign runs with.
pub struct Doubles {
    pub practice: Practice,
    pub exam: Exam,
    pub trainer: Dying,
    pub ticks: Ticks,
}

impl Doubles {
    pub fn new(rig: &Rig, says: &[Says]) -> Self {
        Self {
            practice: Practice::default(),
            exam: Exam::new(&rig.scratch.0.join("bench-private"), says),
            trainer: Dying::default(),
            ticks: Ticks::default(),
        }
    }

    /// The controller of `spec` over these stand-ins.
    pub fn open<'a>(
        &'a mut self,
        rig: &'a Rig,
        spec: CampaignSpec,
    ) -> Result<CampaignController<'a>, CampaignError> {
        CampaignController::open(
            &rig.ctx,
            spec,
            Parts {
                rollouts: &mut self.practice,
                examiner: &mut self.exam,
                trainer: &self.trainer,
                stopwatch: &self.ticks,
                gpus: 1,
                cancel: CancelToken::new(),
                progress: None,
            },
        )
    }
}

/// The champion's generation.
pub fn generation_of(model: &Model) -> u32 {
    model.generation
}
