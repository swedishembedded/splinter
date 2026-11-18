// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that run model learning as one
// continuous, resumable campaign with dynamic sampling, replay and a
// validation gate with rollback, for its clients. If your team needs expertise
// in on-policy training loops for language-model agents you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec of the campaign controller against a scripted world: tasks of fixed
//! difficulty, a policy whose skill follows its version, a trainer that
//! raises (or lowers) it by a set amount. The world has no model; what is
//! tested is the controller's behaviour: dynamic sampling, retirement,
//! dev-validation with rollback, resume after a crash, one controller per
//! campaign, power-first refusal, plateau switching and the ledger.
#![allow(clippy::unwrap_used)]

use std::cell::{Cell, RefCell};
use std::path::PathBuf;
use std::rc::Rc;

use serde_json::{json, Value};
use splinter_core::chat::WireMessage;
use splinter_core::clock::FixedClock;
use splinter_core::digest::Digest;
use splinter_data::RecordBody;
use splinter_eval::campaign::gate::DevResult;
use splinter_eval::campaign::pass_rate::Band;
use splinter_eval::campaign::plateau::PlateauRule;
use splinter_eval::campaign::sampling::AllFailStrategy;
use splinter_model::stats::ExactSignTest;
use splinter_pipelines::campaign::{
    AllFailHandler, CampaignConfig, CampaignController, CampaignError, CollectRequest,
    LearningStrategy, Parts, PolicyRef, Rollout, RolloutCollector, RolloutGroup, Stopwatch,
    TrainReport, TrainRequest, Trained, Trainer, Validator,
};
use splinter_store::campaign::Verdict;
use splinter_store::StateRoot;

/// The skill of the policy at a version.
fn skill(version: u64, gain: f64) -> f64 {
    0.3 + gain * version as f64
}

/// A task passes when the policy's skill beats its difficulty.
fn difficulty(task: &str) -> f64 {
    let n: u32 = task[1..].parse().unwrap();
    f64::from(n) / 20.0
}

/// A deterministic pseudo-random number in `[0, 1)` from a few integers.
fn noise(seed: u64, a: u64, b: u64) -> f64 {
    let mut x =
        seed ^ a.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ b.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 31;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 29;
    (x >> 11) as f64 / (1u64 << 53) as f64
}

#[derive(Default)]
struct Log {
    collected_versions: RefCell<Vec<u64>>,
    trained: Cell<usize>,
    last_records: RefCell<Vec<splinter_data::Record>>,
    validated: Cell<usize>,
    graded: Cell<usize>,
    requests: RefCell<Vec<(usize, Vec<String>)>>,
}

struct World {
    tasks: Vec<String>,
    gain: f64,
    log: Rc<Log>,
    privileged: bool,
    kl: Option<f64>,
    entropy: Option<f64>,
    fail_validation_once: Cell<bool>,
    flat_dev_for_even: bool,
}

impl World {
    fn new(tasks: usize, gain: f64) -> Self {
        Self::with_tasks((0..tasks).map(|i| format!("t{i:02}")).collect(), gain)
    }

    fn with_tasks(tasks: Vec<String>, gain: f64) -> Self {
        Self {
            tasks,
            gain,
            log: Rc::new(Log::default()),
            privileged: false,
            kl: Some(0.01),
            entropy: None,
            fail_validation_once: Cell::new(false),
            flat_dev_for_even: false,
        }
    }
}

struct Collector<'w>(&'w World);
struct Learner<'w> {
    world: &'w World,
    dir: PathBuf,
}
struct Exam<'w>(&'w World, usize);

impl RolloutCollector for Collector<'_> {
    fn describe(&self) -> Value {
        json!({"world": self.0.tasks.len(), "gain": self.0.gain})
    }
    fn tasks(&self) -> Vec<String> {
        self.0.tasks.clone()
    }
    fn set_policy(&mut self, _policy: &PolicyRef) -> Result<(), CampaignError> {
        Ok(())
    }
    fn collect(&mut self, request: &CollectRequest) -> Result<Vec<RolloutGroup>, CampaignError> {
        self.0
            .log
            .requests
            .borrow_mut()
            .push((request.attempts, request.tasks.clone()));
        self.0
            .log
            .collected_versions
            .borrow_mut()
            .push(request.policy.version);
        let s = skill(request.policy.version, self.0.gain);
        Ok(request
            .tasks
            .iter()
            .enumerate()
            .map(|(t, task)| RolloutGroup {
                task: task.clone(),
                rollouts: (0..request.attempts)
                    .map(|a| {
                        let p = 1.0 / (1.0 + (-(s - difficulty(task)) * 12.0).exp());
                        let passed = noise(request.seed, t as u64, a as u64) < p;
                        Rollout {
                            reward: f64::from(u8::from(passed)),
                            passed,
                            prompt: vec![WireMessage {
                                role: "user".into(),
                                content: format!("solve {task}"),
                                images: vec![],
                                tool_calls: vec![],
                                tool_call_id: None,
                                reasoning_content: None,
                                train: false,
                            }],
                            completion: format!("answer {a}"),
                            feedback: Some("wrong".into()),
                            privileged: self
                                .0
                                .privileged
                                .then(|| format!("{task} is solved by 42")),
                            policy_version: request.policy.version,
                            trace: None,
                        }
                    })
                    .collect(),
            })
            .collect())
    }
}

impl Trainer for Learner<'_> {
    fn describe(&self) -> Value {
        json!({"scripted": true})
    }
    fn train(&mut self, request: &TrainRequest) -> Result<Trained, CampaignError> {
        self.world.log.trained.set(self.world.log.trained.get() + 1);
        *self.world.log.last_records.borrow_mut() = request.records.clone();
        std::fs::create_dir_all(&request.out_dir).unwrap();
        let file = request.out_dir.join("adapter.safetensors");
        std::fs::write(&file, format!("adapter {}", self.world.log.trained.get())).unwrap();
        assert!(self.dir.exists());
        Ok(Trained {
            digest: Digest::of(format!("adapter-{}", self.world.log.trained.get()).as_bytes()),
            adapter: file,
            dir: request.out_dir.clone(),
            bytes: 16,
            report: TrainReport {
                steps: 4,
                kl: self.world.kl,
                clip_fraction: Some(0.1),
                skipped_nonfinite: 0,
                skipped_guard: 1,
                entropy: self.world.entropy,
            },
        })
    }
}

impl Validator for Exam<'_> {
    fn describe(&self) -> Value {
        json!({"dev": self.1})
    }
    fn units(&self) -> usize {
        self.1
    }
    fn dev_tasks(&self) -> Vec<String> {
        (0..self.1).map(|i| format!("d{i:03}")).collect()
    }
    fn validate_on(&mut self, p: &PolicyRef, dev: &[String]) -> Result<DevResult, CampaignError> {
        if self.0.fail_validation_once.replace(false) {
            return Err(CampaignError::Validator("the card fell over".into()));
        }
        self.0.log.validated.set(self.0.log.validated.get() + 1);
        self.0.log.graded.set(self.0.log.graded.get() + dev.len());
        // A candidate's version is its parent's plus one, but a gain of zero
        // or less makes its skill no better than the champion's.
        let s = skill(p.version, self.0.gain);
        Ok(DevResult {
            dev: dev
                .iter()
                .map(|task| {
                    let i: u32 = task[1..].parse().unwrap();
                    // The even environment's dev measure never moves.
                    let flat = self.0.flat_dev_for_even && i.is_multiple_of(2);
                    (task.clone(), s > f64::from(i) / self.1 as f64 && !flat)
                })
                .collect(),
            retention: Default::default(),
        })
    }
}

/// A stopwatch that advances by a fixed step at every reading.
struct Seconds {
    now: Cell<f64>,
    step: f64,
}

impl Seconds {
    fn still() -> Self {
        Self {
            now: Cell::new(0.0),
            step: 0.0,
        }
    }
}

impl Stopwatch for Seconds {
    fn seconds(&self) -> f64 {
        let now = self.now.get();
        self.now.set(now + self.step);
        now
    }
}

fn config(name: &str) -> CampaignConfig {
    let mut c = CampaignConfig::new(
        name,
        "a policy that is better each arm solves more dev tasks",
    );
    c.batch.target_groups = 4;
    c.batch.max_request = 20;
    c.calibration_batch = 10;
    c.target_effect = 0.3;
    c.expected_discordance = 0.5;
    c
}

fn workspace() -> (StateRoot, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    (StateRoot::new(dir.path().join("state")), dir)
}

struct Rig<'w> {
    collector: Collector<'w>,
    learner: Learner<'w>,
    exam: Exam<'w>,
}

impl<'w> Rig<'w> {
    fn new(world: &'w World, adapters: &std::path::Path, dev: usize) -> Self {
        Self {
            collector: Collector(world),
            learner: Learner {
                world,
                dir: adapters.to_path_buf(),
            },
            exam: Exam(world, dev),
        }
    }

    fn parts<'a>(&'a mut self, clock: &'a FixedClock, watch: &'a Seconds) -> Parts<'a> {
        Parts {
            collector: &mut self.collector,
            trainer: &mut self.learner,
            validator: &mut self.exam,
            handler: None,
            significance: &ExactSignTest,
            clock,
            stopwatch: watch,
            environments: None,
            progress: None,
        }
    }
}

const SEED: u64 = 424_242; // seed: the scripted world is a pure function of it

fn open<'a>(
    ws: &'a StateRoot,
    config: CampaignConfig,
    rig: &'a mut Rig<'_>,
    adapters: &std::path::Path,
) -> Result<CampaignController<'a>, CampaignError> {
    let clock = Box::leak(Box::new(FixedClock::new("2026-11-13T09:00:00.000Z")));
    let watch = Box::leak(Box::new(Seconds::still()));
    CampaignController::open(
        ws,
        config,
        SEED,
        rig.parts(clock, watch),
        adapters.to_path_buf(),
    )
}

#[test]
fn an_arm_trains_only_on_informative_groups_and_the_gate_adopts_a_better_policy() {
    let (ws, dir) = workspace();
    let world = World::new(20, 0.15);
    let adapters = dir.path().join("adapters");
    let mut rig = Rig::new(&world, &adapters, 120);
    let mut controller = open(&ws, config("c1"), &mut rig, &adapters).unwrap();
    let report = controller.run(1).unwrap();
    let arm = &report.arms[0];
    assert!(
        matches!(arm.verdict, Verdict::Accepted { .. }),
        "{:?}",
        arm.verdict
    );
    let decision = arm.decision.as_ref().unwrap();
    assert!(decision.wins > decision.losses);

    let attempts = controller.state().rates.counts(&world.tasks);
    assert_eq!(
        attempts.unmeasured, 0,
        "calibration measured every task: {attempts:?}"
    );
    // Every record trained on carries an advantage: no zero-spread group got in.
    let records = world.log.last_records.borrow();
    assert!(!records.is_empty());
    for record in records.iter() {
        let RecordBody::Chat { advantage, .. } = &record.body else {
            panic!()
        };
        assert!(advantage.is_some_and(|a| a != 0.0));
    }
    assert!(arm.signals.unwrap().informative_fraction.is_some());
    assert_eq!(controller.state().policy.version, 1);
    assert_eq!(world.log.collected_versions.borrow().last(), Some(&0));
}

#[test]
fn easy_tasks_are_retired_for_good_after_calibration() {
    let (ws, dir) = workspace();
    let world = World::new(20, 0.15);
    let adapters = dir.path().join("adapters");
    let mut rig = Rig::new(&world, &adapters, 120);
    let mut controller = open(&ws, config("c2"), &mut rig, &adapters).unwrap();
    controller.run(1).unwrap();
    let rates = &controller.state().rates;
    // Difficulty 0 and 0.05 are far below the 0.3 skill of the base.
    assert!(rates.is_retired("t00"), "{:?}", rates.rate("t00"));
    assert_eq!(
        rates.band("t19"),
        Band::Cliff,
        "difficulty 0.95 is never solved"
    );
}

#[test]
fn a_candidate_that_does_not_beat_the_champion_is_rolled_back_and_the_champion_stays() {
    let (ws, dir) = workspace();
    let world = World::new(20, 0.0);
    let adapters = dir.path().join("adapters");
    let mut rig = Rig::new(&world, &adapters, 120);
    let mut controller = open(&ws, config("c3"), &mut rig, &adapters).unwrap();
    let report = controller.run(1).unwrap();
    assert!(matches!(report.arms[0].verdict, Verdict::RolledBack { .. }));
    assert_eq!(controller.state().policy.version, 0);
    assert!(controller.state().policy.adapter.is_none());
    let ledger = controller.ledger().unwrap();
    let last = ledger.last().unwrap();
    assert!(matches!(last.verdict, Verdict::RolledBack { .. }));
    assert_eq!(last.discoveries, Some(0.0));
}

#[test]
fn the_ledger_records_hypothesis_digest_cost_and_the_numbers_behind_the_verdict() {
    let (ws, dir) = workspace();
    let world = World::new(20, 0.15);
    let adapters = dir.path().join("adapters");
    let mut rig = Rig::new(&world, &adapters, 120);
    let mut controller = open(&ws, config("c4"), &mut rig, &adapters).unwrap();
    controller.run(1).unwrap();
    let ledger = controller.ledger().unwrap();
    assert_eq!(ledger.len(), 2, "running, then the verdict");
    let state_files = std::fs::read_dir(dir.path().join("state/campaigns"))
        .unwrap()
        .count();
    assert_eq!(state_files, 1, "the whole campaign is one pack");
    assert_eq!(ledger[0].verdict, Verdict::Running);
    let done = &ledger[1];
    assert!(done.hypothesis.contains("a policy that is better each arm"));
    assert_eq!(done.config_digest, controller.state().config_digest);
    assert_eq!(done.cost.gpu_hours, Some(0.0), "the stopwatch never moved");
    assert!(done.cost.rollouts.unwrap() > 0);
    for key in [
        "informative_fraction",
        "kl",
        "clip_fraction",
        "dev_wins",
        "dev_p_value",
        "planned_mde",
    ] {
        assert!(
            done.measurements.contains_key(key),
            "{key} in {:?}",
            done.measurements
        );
    }
    assert!(
        !done.measurements.contains_key("entropy"),
        "no trace, so entropy is unmeasured and absent, not zero"
    );
}

#[test]
fn the_trainers_entropy_and_guard_skips_are_in_the_ledger_when_the_rollouts_carry_no_trace() {
    let (ws, dir) = workspace();
    let mut world = World::new(20, 0.15);
    world.entropy = Some(0.7);
    let adapters = dir.path().join("adapters");
    let mut rig = Rig::new(&world, &adapters, 120);
    let mut controller = open(&ws, config("c4b"), &mut rig, &adapters).unwrap();
    controller.run(1).unwrap();
    let done = &controller.ledger().unwrap()[1];
    assert_eq!(done.measurements.get("entropy"), Some(&0.7));
    assert_eq!(done.measurements.get("steps_skipped_guard"), Some(&1.0));
}

#[test]
fn a_crash_after_training_resumes_at_the_gate_without_training_again() {
    let (ws, dir) = workspace();
    let world = World::new(20, 0.15);
    world.fail_validation_once.set(true);
    let adapters = dir.path().join("adapters");
    let mut rig = Rig::new(&world, &adapters, 120);
    {
        let mut controller = open(&ws, config("c5"), &mut rig, &adapters).unwrap();
        let failed = controller.run(1);
        assert!(
            matches!(failed, Err(CampaignError::Validator(_))),
            "{failed:?}"
        );
        assert!(controller.state().pending.is_some());
    }
    assert_eq!(world.log.trained.get(), 1);
    // A new process: a new controller over the same store.
    let mut rig = Rig::new(&world, &adapters, 120);
    let mut controller = open(&ws, config("c5"), &mut rig, &adapters).unwrap();
    assert!(
        controller.state().pending.is_some(),
        "the pending candidate was saved"
    );
    let report = controller.run(1).unwrap();
    assert!(matches!(report.arms[0].verdict, Verdict::Accepted { .. }));
    assert_eq!(world.log.trained.get(), 1, "training was not repeated");
    assert_eq!(controller.state().arms, 1);
    assert!(controller.state().pending.is_none());
}

#[test]
fn two_controllers_cannot_both_advance_one_campaign() {
    let (ws, dir) = workspace();
    let world = World::new(20, 0.15);
    let adapters = dir.path().join("adapters");
    let mut rig_a = Rig::new(&world, &adapters, 120);
    let mut rig_b = Rig::new(&world, &adapters, 120);
    let mut first = open(&ws, config("c6"), &mut rig_a, &adapters).unwrap();
    let mut second = open(&ws, config("c6"), &mut rig_b, &adapters).unwrap();
    first.run(1).unwrap();
    let late = second.run(1);
    assert!(
        matches!(
            late,
            Err(CampaignError::Store(
                splinter_store::campaign::CampaignError::Conflict { .. }
            ))
        ),
        "{late:?}"
    );
}

#[test]
fn a_campaign_is_continued_as_it_began_a_changed_configuration_is_refused() {
    let (ws, dir) = workspace();
    let world = World::new(20, 0.15);
    let adapters = dir.path().join("adapters");
    let mut rig = Rig::new(&world, &adapters, 120);
    open(&ws, config("c7"), &mut rig, &adapters).unwrap();
    let mut changed = config("c7");
    changed.attempts_per_group = 12;
    let mut rig = Rig::new(&world, &adapters, 120);
    let refused = open(&ws, changed, &mut rig, &adapters).err().unwrap();
    assert!(
        refused.to_string().contains("another configuration"),
        "{refused}"
    );
}

#[test]
fn an_arm_whose_exam_cannot_see_the_target_is_refused_before_it_costs_anything() {
    let (ws, dir) = workspace();
    let world = World::new(20, 0.15);
    let adapters = dir.path().join("adapters");
    let mut rig = Rig::new(&world, &adapters, 6);
    let mut controller = open(&ws, config("c8"), &mut rig, &adapters).unwrap();
    let refused = controller.run(1);
    assert!(
        matches!(refused, Err(CampaignError::Refused(_))),
        "{refused:?}"
    );
    assert_eq!(world.log.trained.get(), 0);
    assert!(
        world.log.collected_versions.borrow().is_empty(),
        "nothing was rolled out"
    );
    let ledger = controller.ledger().unwrap();
    assert!(matches!(ledger[0].verdict, Verdict::Refused { .. }));
}

#[test]
fn a_kl_out_of_bounds_halts_the_campaign_and_adopts_nothing() {
    let (ws, dir) = workspace();
    let mut world = World::new(20, 0.15);
    world.kl = Some(5.0);
    let adapters = dir.path().join("adapters");
    let mut rig = Rig::new(&world, &adapters, 120);
    let mut controller = open(&ws, config("c9"), &mut rig, &adapters).unwrap();
    let report = controller.run(3).unwrap();
    assert_eq!(report.arms.len(), 1, "the campaign stopped");
    assert!(matches!(report.arms[0].verdict, Verdict::Halted { .. }));
    assert!(report.arms[0].stop);
    assert_eq!(controller.state().policy.version, 0);
    assert_eq!(
        world.log.validated.get(),
        0,
        "no candidate reached the gate"
    );
}

#[test]
fn a_pool_with_no_spread_trains_nothing_and_says_why() {
    let (ws, dir) = workspace();
    // Skill 0.3: difficulty 0 and 0.05 are always solved (retired after
    // calibration), 0.9 and 0.95 never: the cliff.
    let world = World::with_tasks(["t00", "t01", "t18", "t19"].map(String::from).to_vec(), 0.0);
    let adapters = dir.path().join("adapters");
    let mut rig = Rig::new(&world, &adapters, 120);
    let mut c = config("c10");
    c.batch.max_rounds = 2;
    let mut controller = open(&ws, c, &mut rig, &adapters).unwrap();
    let report = controller.run(1).unwrap();
    assert!(matches!(report.arms[0].verdict, Verdict::RolledBack { .. }));
    assert_eq!(world.log.trained.get(), 0);
}

#[test]
fn groups_that_all_failed_are_distilled_when_the_strategy_says_so() {
    let (ws, dir) = workspace();
    let mut world = World::new(20, 0.15);
    world.privileged = true;
    let adapters = dir.path().join("adapters");
    let mut rig = Rig::new(&world, &adapters, 120);
    let mut c = config("c11");
    c.all_fail = AllFailStrategy::CliffDistill;
    c.ladder = vec![LearningStrategy::PolicyGradient];
    let mut controller = open(&ws, c, &mut rig, &adapters).unwrap();
    let report = controller.run(1).unwrap();
    assert!(
        report.arms[0].records.distilled > 0,
        "{:?}",
        report.arms[0].records
    );
    let records = world.log.last_records.borrow();
    assert!(records.iter().any(|r| matches!(&r.body, RecordBody::Chat { teacher_context, .. } if !teacher_context.is_empty())));
}

/// A teacher that answers every all-failed task with one verified record.
struct Teacher {
    asked: usize,
}

impl AllFailHandler for Teacher {
    fn teach(
        &mut self,
        groups: &[&RolloutGroup],
    ) -> Result<Vec<splinter_data::Record>, CampaignError> {
        self.asked += groups.len();
        Ok(groups
            .iter()
            .map(|g| {
                let turn = |role: &str, content: &str, train: bool| WireMessage {
                    role: role.into(),
                    content: content.into(),
                    images: Vec::new(),
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                    reasoning_content: None,
                    train,
                };
                splinter_data::Record {
                    body: RecordBody::chat(vec![
                        turn("user", &g.task, false),
                        turn("assistant", "the verified answer", true),
                    ]),
                    metadata: splinter_data::RecordMetadata {
                        experiences: Vec::new(),
                        task: None,
                        sources: Vec::new(),
                        group: None,
                        split: None,
                        view: "campaign-taught".into(),
                        objective: splinter_data::Objective::Sft,
                    },
                }
            })
            .collect())
    }
    fn decompose(&mut self, _groups: &[&RolloutGroup]) -> Result<Vec<String>, CampaignError> {
        Ok(Vec::new())
    }
}

#[test]
fn groups_that_all_failed_are_trained_on_the_teachers_verified_answers() {
    let (ws, dir) = workspace();
    // Skill 0.3 never solves difficulty 0.9 or 0.95: every group fails, so
    // the teacher's verified answers are the arm's only training signal.
    let world = World::with_tasks(["t18", "t19"].map(String::from).to_vec(), 0.15);
    let adapters = dir.path().join("adapters");
    let mut rig = Rig::new(&world, &adapters, 120);
    let mut teacher = Teacher { asked: 0 };
    let mut c = config("c-teacher");
    c.all_fail = AllFailStrategy::Teacher;
    c.batch.max_rounds = 2;
    let clock = FixedClock::new("2026-11-13T09:00:00.000Z");
    let watch = Seconds::still();
    let mut parts = rig.parts(&clock, &watch);
    parts.handler = Some(&mut teacher);
    let mut controller = CampaignController::open(&ws, c, SEED, parts, adapters.clone()).unwrap();
    let report = controller.run(1).unwrap();
    let arm = &report.arms[0];
    assert!(arm.records.taught > 0, "{:?}", arm.records);
    assert_eq!(world.log.trained.get(), 1, "{:?}", arm.verdict);
    let records = world.log.last_records.borrow();
    assert!(records.iter().any(|r| r.metadata.view == "campaign-taught"));
}

#[test]
fn replay_carries_an_accepted_arms_verified_data_into_later_arms_only() {
    let (ws, dir) = workspace();
    let world = World::new(20, 0.15);
    let adapters = dir.path().join("adapters");
    let mut rig = Rig::new(&world, &adapters, 120);
    let mut c = config("c12");
    c.replay.per_batch_record = 1.0;
    let mut controller = open(&ws, c, &mut rig, &adapters).unwrap();
    let first = controller.run(1).unwrap();
    assert_eq!(
        first.arms[0].records.replay, 0,
        "nothing earlier to replay in arm 1"
    );
    assert!(!controller.state().replay.is_empty());
    let second = controller.run(1).unwrap();
    assert!(
        second.arms[0].records.replay > 0,
        "{:?}",
        second.arms[0].records
    );
}

#[test]
fn a_plateau_moves_the_campaign_to_the_next_strategy_and_superseded_checkpoints_are_collected() {
    let (ws, dir) = workspace();
    let world = World::new(20, 0.0);
    let adapters = dir.path().join("adapters");
    let mut rig = Rig::new(&world, &adapters, 120);
    let mut c = config("c13");
    c.ladder = vec![
        LearningStrategy::PolicyGradient,
        LearningStrategy::PositiveOnly,
    ];
    c.plateau = PlateauRule {
        window: 1,
        fraction_of_peak: 0.5,
    };
    // Hours must be spent for a rate to exist.
    let mut controller = {
        let clock = Box::leak(Box::new(FixedClock::new("2026-11-13T09:00:00.000Z")));
        let watch = Box::leak(Box::new(Seconds {
            now: Cell::new(0.0),
            step: 1800.0,
        }));
        CampaignController::open(&ws, c, SEED, rig.parts(clock, watch), adapters.clone()).unwrap()
    };
    let report = controller.run(3).unwrap();
    assert_eq!(report.arms[0].strategy, "policy-gradient");
    assert!(
        report.arms.iter().any(|a| a.strategy == "positive-only"),
        "{:?}",
        report.arms.iter().map(|a| a.strategy).collect::<Vec<_>>()
    );
    let state = controller.state();
    assert!(
        !state.dropped.is_empty(),
        "rolled-back candidates were collected"
    );
    assert!(state.dropped.iter().all(|d| d.bytes == 16));
    for checkpoint in &state.checkpoints {
        assert!(checkpoint.path.exists());
    }
}

#[test]
fn the_repair_strategy_adds_wrong_attempt_feedback_fix_records_to_the_batch() {
    let (ws, dir) = workspace();
    let world = World::new(20, 0.15);
    let adapters = dir.path().join("adapters");
    let mut rig = Rig::new(&world, &adapters, 120);
    let mut c = config("c14");
    c.ladder = vec![LearningStrategy::Repair];
    let mut controller = open(&ws, c, &mut rig, &adapters).unwrap();
    let report = controller.run(1).unwrap();
    assert!(
        report.arms[0].records.repair > 0,
        "{:?}",
        report.arms[0].records
    );
    let records = world.log.last_records.borrow();
    let repairs: Vec<_> = records
        .iter()
        .filter(|r| r.metadata.view == "campaign-repair")
        .collect();
    assert!(!repairs.is_empty());
    for repair in repairs {
        let RecordBody::Chat {
            messages,
            advantage,
            ..
        } = &repair.body
        else {
            panic!()
        };
        assert_eq!(*advantage, None);
        assert_eq!(
            messages.iter().filter(|m| m.train).count(),
            1,
            "only the fix is supervised"
        );
        assert_eq!(messages[2].content, "wrong");
    }
}

#[test]
fn a_task_is_attempted_once_per_batch_however_often_the_batch_is_refilled() {
    let (ws, dir) = workspace();
    let world = World::new(20, 0.15);
    let adapters = dir.path().join("adapters");
    let mut rig = Rig::new(&world, &adapters, 120);
    let mut c = config("c15");
    c.attempts_per_group = 6; // calibration asks for 8, so the two are told apart
    c.batch.target_groups = 12;
    c.batch.max_request = 20;
    let mut controller = open(&ws, c, &mut rig, &adapters).unwrap();
    controller.run(1).unwrap();
    let batch_requests: Vec<String> = world
        .log
        .requests
        .borrow()
        .iter()
        .filter(|(attempts, _)| *attempts == 6)
        .flat_map(|(_, tasks)| tasks.clone())
        .collect();
    let unique: std::collections::BTreeSet<&String> = batch_requests.iter().collect();
    assert_eq!(batch_requests.len(), unique.len(), "{batch_requests:?}");
}

mod environments;
mod evidence;
mod served;
