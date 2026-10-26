// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that turn the documentation of an
// unfamiliar technical domain into a graph of skills they then acquire, for
// its clients. If your team needs expertise in building curricula from
// technical documentation you can procure our services by sending an email
// to info@swedishembedded.com.

//! Spec: an acquisition round can train every weight from the incumbent
//! checkpoint, can let the incumbent write the next round's tasks, and
//! reports what it explored and what that cost.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use common::gate::FakeTrainer;
use common::zork::{authoring, context, request, topics_in, write_manual, Learner};
use splinter_agent::solve::Model;
use splinter_agent::CancelToken;
use splinter_model::train::{Trained, TrainedPreference};
use splinter_orchestrator::{Context, OrchestratorError};
use splinter_pipelines::acquire::{acquire, AcquireRequest, Outcome};
use splinter_pipelines::release::{arm, arm_on};
use splinter_pipelines::train::{ModelCheckpoint, TrainPlan, Trainer, DEFAULT_FULL_LEARNING_RATE};
use splinter_store::runs::read_run;

/// Training every weight as a test double: the checkpoint it writes is a
/// directory, served as `local:<directory>` by a model that knows what its
/// base knew and the topics of the plan's records (none for a dud).
struct FullLearner {
    dud: bool,
    /// Numbers its checkpoints give the training tasks they write.
    writes: std::ops::Range<usize>,
    plans: Mutex<Vec<TrainPlan>>,
    known: Mutex<BTreeMap<PathBuf, Vec<String>>>,
}

impl FullLearner {
    fn new(dud: bool) -> Self {
        Self {
            dud,
            writes: 20..28,
            plans: Mutex::new(Vec::new()),
            known: Mutex::new(BTreeMap::new()),
        }
    }
}

impl Trainer for FullLearner {
    fn train(
        &self,
        _: &Context,
        _: &TrainPlan,
        _: &CancelToken,
    ) -> Result<Trained, OrchestratorError> {
        Err(OrchestratorError::Refused("an adapter was trained".into()))
    }

    fn train_preference(
        &self,
        _: &Context,
        _: &TrainPlan,
        _: &CancelToken,
    ) -> Result<TrainedPreference, OrchestratorError> {
        Err(OrchestratorError::Refused("a preference run".into()))
    }

    fn train_full(
        &self,
        ctx: &Context,
        plan: &TrainPlan,
        _: &CancelToken,
    ) -> Result<ModelCheckpoint, OrchestratorError> {
        self.plans.lock().unwrap().push(plan.clone());
        let dir = plan.dir.join("checkpoint");
        std::fs::create_dir_all(&dir).unwrap();
        let weights = dir.join("model.safetensors");
        std::fs::write(&weights, b"weights").unwrap();
        let mut known = self.known.lock().unwrap();
        let mut topics = known.get(&plan.base).cloned().unwrap_or_default();
        if !self.dud {
            topics.extend(topics_in(plan));
        }
        let names: Vec<&str> = topics.iter().map(String::as_str).collect();
        ctx.add_model(
            arm_on(ctx.config(), &dir, None),
            Model::new(
                Arc::new(authoring(self.writes.clone(), &names)),
                "scripted/checkpoint",
            ),
        );
        known.insert(dir.clone(), topics);
        Ok(ModelCheckpoint {
            dir,
            weights,
            weights_digest: "sha256:00".into(),
            steps_completed: plan.steps,
            initial_loss: Some(2.0),
            final_loss: Some(1.0),
        })
    }
}

fn full_request(manual: &Path) -> AcquireRequest {
    AcquireRequest {
        full_fine_tune: true,
        train_precision: Some("state=int8,sr=on".into()),
        ..request(manual)
    }
}

#[test]
fn a_full_fine_tune_round_starts_from_the_incumbent_checkpoint_and_an_accepted_one_becomes_it() {
    let (scratch, ctx) = context("acquire-full");
    let manual = write_manual(&scratch);
    let trainer = FullLearner::new(false);
    let report = acquire(&ctx, &full_request(&manual), &trainer)
        .unwrap()
        .report;

    assert!(
        report.rounds.iter().all(|r| r.outcome == Outcome::Accepted),
        "{:#?}",
        report.rounds
    );
    let plans = trainer.plans.lock().unwrap();
    assert_eq!(plans.len(), 3);
    // The first round trains the start base; each later one the checkpoint
    // the round before it was accepted with.
    assert_eq!(plans[0].base, ctx.config().policy_base);
    for round in 1..3 {
        assert_eq!(
            plans[round].base,
            plans[round - 1].dir.join("checkpoint"),
            "round {} did not start from the incumbent",
            round + 1
        );
    }
    // Rounds trained on their own records beside a replay of the accepted
    // ones; the first has nothing to replay.
    assert!(plans[0].replay_file.is_none());
    assert!(plans[1].replay_file.is_some() && plans[2].replay_file.is_some());
    assert_eq!(plans[1].replay_share, Some(0.25));
    assert_eq!(plans[1].datasets.len(), 1);
    // Every weight at the full learning rate, at the precision asked for.
    for plan in plans.iter() {
        assert_eq!(plan.tuning.learning_rate, Some(DEFAULT_FULL_LEARNING_RATE));
        assert_eq!(plan.train_precision.as_deref(), Some("state=int8,sr=on"));
    }
    // The run ends on the last checkpoint, assessed against the start policy;
    // the checkpoints it superseded are gone.
    let assessment = report.assessment.as_ref().unwrap();
    assert!(
        assessment.accepted && assessment.improved == 3,
        "{assessment:#?}"
    );
    assert!(!plans[0].dir.exists() && !plans[1].dir.exists());
    assert!(plans[2].dir.join("checkpoint").exists());
    assert_eq!(
        report.rounds[2].checkpoint.as_deref(),
        Some(plans[2].dir.join("checkpoint").to_str().unwrap())
    );
    assert!(report.release.is_none());
}

#[test]
fn a_rejected_full_checkpoint_is_deleted_unless_candidates_are_kept() {
    for keep in [false, true] {
        let (scratch, ctx) = context(&format!("acquire-full-dud-{keep}"));
        let manual = write_manual(&scratch);
        let trainer = FullLearner::new(true);
        let request = AcquireRequest {
            keep_candidates: keep,
            ..full_request(&manual)
        };
        let report = acquire(&ctx, &request, &trainer).unwrap().report;
        let plans = trainer.plans.lock().unwrap();
        assert!(!plans.is_empty());
        assert!(report
            .rounds
            .iter()
            .all(|r| !matches!(r.outcome, Outcome::Accepted)));
        // A rejected round leaves the incumbent where it was: every round
        // starts from the start base.
        assert!(plans.iter().all(|p| p.base == ctx.config().policy_base));
        assert!(
            plans.iter().all(|p| p.dir.exists() == keep),
            "keep_candidates={keep}"
        );
    }
}

#[test]
fn a_full_fine_tune_that_the_release_gate_cannot_take_is_refused_up_front() {
    let (scratch, ctx) = context("acquire-full-refused");
    let manual = write_manual(&scratch);
    let refused = |request: AcquireRequest| {
        let e = acquire(&ctx, &request, &FullLearner::new(false)).unwrap_err();
        assert!(matches!(e, OrchestratorError::Refused(_)), "{e}");
        e.to_string()
    };
    let released = AcquireRequest {
        no_release: false,
        ..full_request(&manual)
    };
    assert!(refused(released).contains("release gate"));
    let precision = AcquireRequest {
        train_precision: Some("state=nonsense".into()),
        ..full_request(&manual)
    };
    assert!(refused(precision).contains("training precision"));
    let adapter_only = AcquireRequest {
        train_precision: Some("state=int8".into()),
        ..request(&manual)
    };
    assert!(refused(adapter_only).contains("full fine-tune"));
}

/// Training an adapter whose model also writes tasks, numbering them 20 to 27.
#[derive(Default)]
struct SelfWriter {
    plans: Mutex<Vec<TrainPlan>>,
}

impl Trainer for SelfWriter {
    fn train(
        &self,
        ctx: &Context,
        plan: &TrainPlan,
        cancel: &CancelToken,
    ) -> Result<Trained, OrchestratorError> {
        self.plans.lock().unwrap().push(plan.clone());
        let topics = topics_in(plan);
        let known: Vec<&str> = topics.iter().map(String::as_str).collect();
        let trained = FakeTrainer::knowing(&known).train(ctx, plan, cancel)?;
        ctx.add_model(
            arm(ctx.config(), Some(&trained.adapter)),
            Model::new(Arc::new(authoring(20..28, &known)), "scripted/writer"),
        );
        Ok(trained)
    }

    fn train_preference(
        &self,
        ctx: &Context,
        plan: &TrainPlan,
        cancel: &CancelToken,
    ) -> Result<TrainedPreference, OrchestratorError> {
        FakeTrainer::knowing(&[]).train_preference(ctx, plan, cancel)
    }
}

/// The numbers of the questions about `topic` in the plan's records.
fn numbers_in(plan: &TrainPlan, topic: &str) -> Vec<usize> {
    let mut found = Vec::new();
    for dataset in &plan.datasets {
        let text = std::fs::read_to_string(&dataset.path).unwrap();
        for part in text
            .split(&format!("What is the {topic} code number "))
            .skip(1)
        {
            let digits: String = part.chars().take_while(char::is_ascii_digit).collect();
            found.push(digits.parse().unwrap());
        }
    }
    found
}

#[test]
fn with_self_generation_the_accepted_candidate_writes_the_next_rounds_tasks_and_the_probes_stay_the_originals(
) {
    let (scratch, ctx) = context("acquire-self-generate");
    let manual = write_manual(&scratch);
    let trainer = SelfWriter::default();
    let report = acquire(
        &ctx,
        &AcquireRequest {
            self_generate: true,
            ..request(&manual)
        },
        &trainer,
    )
    .unwrap()
    .report;
    assert!(report.rounds.iter().all(|r| r.outcome == Outcome::Accepted));

    // Every round is written by the model that attempts it: the start
    // policy in round 1, the accepted candidate after.
    let first = &report.rounds[0];
    assert_eq!(first.generator, "policy:default");
    for round in &report.rounds {
        assert_eq!(round.generator, round.incumbent);
    }
    for round in &report.rounds[1..] {
        assert_ne!(round.generator, "policy:default");
    }
    assert_ne!(report.rounds[1].generator, report.rounds[2].generator);

    // The records bear it out: the original generator numbers its tasks 10
    // to 17, the trained one 20 to 27; and every probe is the original's.
    let plans = trainer.plans.lock().unwrap();
    assert!(numbers_in(&plans[0], "alpha")
        .iter()
        .all(|n| (10..18).contains(n)));
    assert!(numbers_in(&plans[1], "beta")
        .iter()
        .all(|n| (20..28).contains(n)));
    assert!(numbers_in(&plans[2], "gamma")
        .iter()
        .all(|n| (20..28).contains(n)));
    for probes in &report.probes {
        for digest in probes.dev.iter().chain(&probes.finals) {
            let task = ctx.tasks().get(digest).unwrap();
            let number: usize = task
                .instruction
                .rsplit(' ')
                .next()
                .unwrap()
                .trim_end_matches('?')
                .parse()
                .unwrap();
            assert!(number < 8, "{}", task.instruction);
        }
    }
}

#[test]
fn every_round_reports_what_it_explored_and_what_that_cost() {
    let (scratch, ctx) = context("acquire-yield");
    let manual = write_manual(&scratch);
    let recorded = acquire(&ctx, &request(&manual), &Learner).unwrap();
    let run = read_run(ctx.workspace(), &recorded.run).unwrap();
    for round in &recorded.report.rounds {
        // 8 tasks admitted, fewer than a round wants, so the generator is
        // asked again; it writes the same 8, refused whole, and the round
        // goes on with them: 2 attempts at each, none solved by a policy
        // that knows nothing, all of them taught.
        assert_eq!(
            (round.proposed, round.written, round.attempts, round.graded),
            (16, 8, 16, 16)
        );
        assert_eq!((round.verified_passes, round.worth_training), (0, 8));
        assert!(round.records.unwrap() > 0);
        assert!(round.solve.unwrap().wall_seconds >= 0.0);
        assert!(round.train.unwrap().wall_seconds >= 0.0);
        // The same fields are in the run record's stage for the round.
        let stage = run
            .stages
            .iter()
            .find(|s| s.stage == format!("round-{}", round.round))
            .unwrap();
        for key in [
            "incumbent",
            "generator",
            "proposed",
            "written",
            "attempts",
            "verified_passes",
            "worth_training",
            "records",
            "solve",
            "train",
        ] {
            assert!(stage.summary.get(key).is_some(), "{key}: {}", stage.summary);
        }
        assert_eq!(round.generator, "policy:default");
    }
}

#[test]
fn a_reference_teacher_teaches_with_the_tasks_own_answers_and_asks_no_model() {
    let model = common::zork::model();
    let prompts = model.prompts.clone();
    let (scratch, ctx) = common::scratch_context("acquire-reference-teacher", model, false);
    common::gate::put_base(&ctx);
    let manual = write_manual(&scratch);
    let report = acquire(
        &ctx,
        &AcquireRequest {
            reference_teacher: true,
            ..request(&manual)
        },
        &Learner,
    )
    .unwrap()
    .report;
    // The policy knows nothing, so every training record is a taught one.
    assert!(report.rounds.iter().any(|r| r.outcome == Outcome::Accepted));
    // No model was shown grounding material to teach with.
    assert!(prompts
        .lock()
        .unwrap()
        .iter()
        .all(|p| !p.contains("Reference material:")));
    // The demonstrations are the teacher's, said by the demonstrator, and
    // passed the same verifier as any answer.
    let demonstrations: Vec<_> = ctx
        .experiences()
        .list()
        .unwrap()
        .into_iter()
        .map(|id| ctx.experiences().get(&id).unwrap())
        .filter(|e| e.provenance.solver == splinter_agent::demonstration::DEMONSTRATOR)
        .collect();
    assert!(!demonstrations.is_empty());
    for experience in demonstrations {
        assert!(experience.provenance.teacher);
        let reference = experience
            .privileged
            .iter()
            .find(|p| p.kind == splinter_core::experience::PrivilegedKind::Reference)
            .unwrap();
        assert_eq!(
            experience.final_output.as_deref(),
            Some(reference.content.as_str())
        );
    }
}

#[test]
fn a_round_asks_a_generator_that_writes_one_task_a_request_again() {
    let (scratch, ctx) = common::scratch_context(
        "acquire-round-batches",
        common::zork::writing_tasks_in_batches(1),
        false,
    );
    common::gate::put_base(&ctx);
    let manual = write_manual(&scratch);
    let report = acquire(&ctx, &request(&manual), &Learner).unwrap().report;
    let first = &report.rounds[0];
    assert_eq!(
        first.written,
        splinter_pipelines::acquire::MAX_ROUND_WRITES,
        "{first:?}"
    );
    assert_eq!(first.outcome, Outcome::Accepted, "{first:?}");
}
