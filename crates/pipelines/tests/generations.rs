// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements generations of a code-editing agent that
// keep what earlier generations learned, for its clients. If your team needs
// expertise in measuring retention across successive fine-tunes of an
// agent, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: generations of a coding policy, each held to what the ones before it
//! had.
//!
//! * a generation measures the champion on every protected task before it
//!   trains, trains a candidate on the new admitted experience, measures both
//!   on the protected tasks and the new exam at the same settings, and
//!   releases the candidate through the gate when it keeps what the champion
//!   had; the ledger holds everything it was decided on;
//! * a candidate that loses protected families, or does not beat the champion
//!   on the new exam, is rejected by the limits before the gate is asked, the
//!   alias stays where it was, and the rejection is in the ledger;
//! * the full retention matrix has every generation, the base included, on
//!   every protected task, so forward transfer and backward transfer are
//!   measured, not inferred;
//! * the replay a generation trains with follows its policy: none, uniform,
//!   or a rebuild of a fresh adapter from the base on everything accumulated.
//!
//! The agents are scripted and training is a test double (see the fixtures
//! of `release_workspace`); the gate and the exams are the real ones.

#![allow(clippy::unwrap_used)]

mod common;

use std::path::Path;
use std::sync::Arc;

use common::fake_brain::fake_brain;
use common::gate::{anchor_file, put_base, Brain, FakeTrainer, NOW};
use common::repo::{call, module, module_of, project_of, request_of, tool_calls, Mender, Wording};
use common::{config, Scratch};
use splinter_agent::solve::Model;
use splinter_agent::CancelToken;
use splinter_core::clock::FixedClock;
use splinter_core::model_ref::ModelRef;
use splinter_core::rng::Seed;
use splinter_core::training::Regime;
use splinter_eval::continual::Measure;
use splinter_orchestrator::{Context, OrchestratorError};
use splinter_pipelines::curriculum::frontier::PassAtK;
use splinter_pipelines::generations::{
    read_ledger, retention_matrix, run_generation, GenerationReport, GenerationRequest, Limits,
    Outcome, ReplayChoice,
};
use splinter_pipelines::practice::PracticeRequest;
use splinter_pipelines::release::{anchor, arm};
use splinter_pipelines::train::{TrainPlan, Trainer};
use sven_sdk::model::{CompletionRequest, ModelProvider, ResponseEvent, ResponseStream};

/// Modules of the first project; each is a family of tasks.
const MODULES: usize = 9;
/// Families the first exam reserves: five discordant families are the
/// fewest at which a unanimous sign test is significant at the default level.
const EXAM_FAMILIES: usize = 5;
/// Modules of the second project: the first's, and ten more of which the
/// exam takes five.
const SECOND_MODULES: usize = MODULES + 10;

/// An agent that restores the modules below `limit` and leaves the rest.
struct UpTo {
    limit: usize,
}

#[async_trait::async_trait]
impl ModelProvider for UpTo {
    fn name(&self) -> &str {
        "scripted"
    }
    fn model_name(&self) -> &str {
        "up-to"
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let events = match (tool_calls(&req).len(), module_of(&request_of(&req))) {
            (0, Some(n)) if n < self.limit => call(
                "c-fix",
                "write_file",
                serde_json::json!({"path": format!("shop/tier{n}.py"), "text": module(n), "append": false}),
            ),
            _ => vec![ResponseEvent::TextDelta("done".into()), ResponseEvent::Done],
        };
        Ok(Box::pin(futures::stream::iter(
            events.into_iter().map(Ok).collect::<Vec<_>>(),
        )))
    }
}

/// A trainer whose candidates restore the modules below `limit` only: the
/// double of what a generation has learned so far.
struct Learner {
    inner: FakeTrainer,
    limit: usize,
    /// The text of each replay file a training was handed, read while the
    /// training ran: the candidate's directory is removed after it.
    replayed: std::sync::Mutex<Vec<String>>,
    /// The families of the records each training fitted (what the holdout
    /// rule leaves to train on), which a replay of a later one draws from.
    fitted: std::sync::Mutex<Vec<std::collections::BTreeSet<String>>>,
}

impl Learner {
    fn up_to(limit: usize) -> Self {
        Self {
            inner: FakeTrainer::mending(true),
            limit,
            replayed: std::sync::Mutex::new(Vec::new()),
            fitted: std::sync::Mutex::new(Vec::new()),
        }
    }

    fn plans(&self) -> Vec<TrainPlan> {
        self.inner.plans.lock().unwrap().clone()
    }
}

impl Trainer for Learner {
    fn train(
        &self,
        ctx: &Context,
        plan: &TrainPlan,
        cancel: &CancelToken,
    ) -> Result<splinter_model::train::Trained, OrchestratorError> {
        if let Some(file) = &plan.replay_file {
            self.replayed
                .lock()
                .unwrap()
                .push(std::fs::read_to_string(file).unwrap());
        }
        let lines: Vec<String> = plan
            .datasets
            .iter()
            .flat_map(|d| {
                std::fs::read_to_string(&d.path)
                    .unwrap()
                    .lines()
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .collect();
        let fitted = splinter_data::holdout::holdout_split_records(&lines)
            .map_or(lines.iter().collect(), |(train, _)| train);
        self.fitted.lock().unwrap().push(
            fitted
                .iter()
                .filter_map(|l| {
                    let r: serde_json::Value = serde_json::from_str(l).ok()?;
                    r["metadata"]["group"].as_str().map(str::to_string)
                })
                .collect(),
        );
        let trained = self.inner.train(ctx, plan, cancel)?;
        ctx.add_model(
            arm(ctx.config(), Some(&trained.adapter)),
            Model::new(Arc::new(UpTo { limit: self.limit }), "scripted/up-to"),
        );
        Ok(trained)
    }

    fn train_preference(
        &self,
        ctx: &Context,
        plan: &TrainPlan,
        cancel: &CancelToken,
    ) -> Result<splinter_model::train::TrainedPreference, OrchestratorError> {
        self.inner.train_preference(ctx, plan, cancel)
    }
}

fn context(test: &str) -> (Scratch, Context) {
    let scratch = Scratch::new(test);
    let mut config = config(&scratch);
    config.brain_binary = Some(fake_brain(&scratch.0, Brain::Honest));
    let base = arm(&config, None);
    let ctx = Context::new(config, false)
        .unwrap()
        .with_clock(Box::new(FixedClock::new(NOW)))
        .with_model(
            ModelRef::policy_default(),
            Model::new(common::repo::Agent::new(), "scripted/agent-1"),
        )
        .with_model(base, Model::new(Mender::new(false), "scripted/idle"));
    put_base(&ctx);
    anchor::freeze(&ctx, &[anchor_file(&scratch.0, 4)]).unwrap();
    (scratch, ctx)
}

fn generation(repo: &Path, modules_asked: usize, replay: ReplayChoice) -> GenerationRequest {
    GenerationRequest {
        experiment: "coding".into(),
        practice: PracticeRequest {
            repository: repo.to_path_buf(),
            revision: "HEAD".into(),
            max_tasks: Some(128),
            mutants_per_file: Some(2),
            exam_families: Some(modules_asked),
            // seed: which modules form the exam decides how many training
            // records the fixture leaves; the spec needs one split that
            // leaves enough.
            exam_seed: Some(Seed::new(1)),
            pass_at_k: PassAtK::new(2, None, None),
            ..PracticeRequest::default()
        },
        replay,
        limits: Limits::default(),
    }
}

fn first_project(scratch: &Scratch) -> std::path::PathBuf {
    let repo = scratch.0.join("first");
    project_of(&repo, &[], Wording::Short, MODULES);
    repo
}

fn second_project(scratch: &Scratch) -> std::path::PathBuf {
    let repo = scratch.0.join("second");
    project_of(&repo, &[], Wording::Long, SECOND_MODULES);
    repo
}

fn run(ctx: &Context, request: &GenerationRequest, trainer: &Learner) -> GenerationReport {
    run_generation(ctx, request, trainer, &CancelToken::new())
        .unwrap()
        .report
}

/// Two generations: the first learns the first project, the second the
/// second, whose ten new modules the first generation never saw.
fn two_generations(ctx: &Context, scratch: &Scratch, replay: ReplayChoice) -> (Learner, Learner) {
    let (first, second) = (Learner::up_to(MODULES), Learner::up_to(SECOND_MODULES));
    let accepted = run(
        ctx,
        &generation(&first_project(scratch), EXAM_FAMILIES, replay),
        &first,
    );
    assert!(
        matches!(accepted.record.outcome, Outcome::Accepted { .. }),
        "{:#?}",
        accepted.record
    );
    let accepted = run(
        ctx,
        &generation(&second_project(scratch), 2 * EXAM_FAMILIES, replay),
        &second,
    );
    assert!(
        matches!(accepted.record.outcome, Outcome::Accepted { .. }),
        "{:#?}",
        accepted.record
    );
    (first, second)
}

#[test]
fn a_candidate_that_keeps_what_the_champion_had_and_gains_on_the_new_exam_is_released_and_recorded()
{
    let (scratch, ctx) = context("generations-accept");
    two_generations(&ctx, &scratch, ReplayChoice::Uniform);

    let ledger = read_ledger(&ctx, "coding").unwrap();
    assert_eq!(ledger.len(), 2);
    let (first, second) = (&ledger[0], &ledger[1]);
    assert_eq!((first.generation, second.generation), (1, 2));
    assert_eq!(first.champion, None);
    assert_eq!(
        second.champion.as_ref(),
        Some(match &first.outcome {
            Outcome::Accepted { release } => release,
            other => panic!("{other:?}"),
        })
    );
    assert_eq!(
        ctx.releases().alias("default").unwrap().as_ref(),
        match &second.outcome {
            Outcome::Accepted { release } => Some(release),
            other => panic!("{other:?}"),
        }
    );

    // The second generation was decided on the champion and the candidate
    // measured on the same protected families and the new exam.
    let (champion, candidate) = (
        second.champion_row.as_ref().unwrap(),
        second.candidate_row.as_ref().unwrap(),
    );
    assert_eq!(champion.cells.len(), candidate.cells.len());
    assert!(champion
        .cells
        .iter()
        .zip(&candidate.cells)
        .all(|(a, b)| a.task == b.task && a.column == b.column));
    let assessment = second.assessment.as_ref().unwrap();
    assert!(assessment.regressions.is_empty(), "{assessment:#?}");
    assert!(assessment.new_gain.unwrap() > 0.4, "{assessment:#?}");
    assert!(
        first.assessment.as_ref().unwrap().mean_change.is_none(),
        "nothing was protected yet"
    );
}

#[test]
fn a_candidate_that_loses_protected_families_is_rejected_before_the_gate_and_the_alias_stays() {
    let (scratch, ctx) = context("generations-reject");
    let learner = Learner::up_to(MODULES);
    let accepted = run(
        &ctx,
        &generation(
            &first_project(&scratch),
            EXAM_FAMILIES,
            ReplayChoice::Uniform,
        ),
        &learner,
    );
    let Outcome::Accepted { release } = accepted.record.outcome else {
        panic!("{:#?}", accepted.record)
    };

    // The next candidate restores nothing: it loses every protected family
    // and does not gain on the new exam.
    let forgetful = Learner::up_to(0);
    let rejected = run(
        &ctx,
        &generation(
            &second_project(&scratch),
            2 * EXAM_FAMILIES,
            ReplayChoice::Uniform,
        ),
        &forgetful,
    );
    let Outcome::Rejected { reasons } = &rejected.record.outcome else {
        panic!("{:#?}", rejected.record)
    };
    assert!(
        reasons.iter().any(|r| r.starts_with("retention:")),
        "{reasons:?}"
    );
    assert!(
        reasons.iter().any(|r| r.starts_with("gain:")),
        "{reasons:?}"
    );
    assert!(
        reasons.iter().all(|r| !r.starts_with("gate:")),
        "the gate was not asked: {reasons:?}"
    );
    let candidate = rejected.record.candidate.as_ref().unwrap();
    assert!(ctx.releases().of_candidate(candidate).unwrap().is_none());
    assert_eq!(ctx.releases().alias("default").unwrap(), Some(release));
    assert_eq!(read_ledger(&ctx, "coding").unwrap().len(), 2);
}

#[test]
fn the_full_matrix_measures_every_generation_on_every_protected_task() {
    let (scratch, ctx) = context("generations-matrix");
    two_generations(&ctx, &scratch, ReplayChoice::Uniform);
    let work = splinter_eval::gate::AgenticConfig::default();
    let matrix = retention_matrix(&ctx, "default", work, &CancelToken::new()).unwrap();

    // The base, then the two releases.
    assert_eq!(matrix.rows.len(), 3);
    assert_eq!(matrix.matrix.generations(), 3);
    for row in &matrix.rows {
        assert_eq!(row.cells.len(), matrix.matrix.tasks().len());
        assert!(
            row.cells.iter().all(|c| c.verdict.is_some()),
            "every cell measured"
        );
    }
    let value = |m: &Measure| match m {
        Measure::Measured { value, .. } => *value,
        other => panic!("{other:?}"),
    };
    let metrics = &matrix.metrics;
    // The final model solves everything it was examined on.
    assert!((value(&metrics.acc) - 1.0).abs() < 1e-9);
    // Nothing earlier was lost when the second generation learned.
    assert!(value(&metrics.bwt).abs() < 1e-9);
    assert!(value(&metrics.forgetting).abs() < 1e-9);
    // Generation 1 already solved the second exam's old families and none
    // of its new ones, from a base that solves nothing: of the fifteen
    // families (five of the first exam, ten of the second) five were solved
    // before they were learned.
    assert!(
        (value(&metrics.forward_transfer) - 1.0 / 3.0).abs() < 1e-9,
        "{:?}",
        metrics.forward_transfer
    );
    assert!((value(&metrics.learning_gain) - 1.0).abs() < 1e-9);
}

#[test]
fn replay_follows_the_policy_and_a_rebuild_starts_a_fresh_adapter_from_the_base() {
    let plans = |replay: ReplayChoice, test: &str| -> (Learner, Learner) {
        let (scratch, ctx) = context(test);
        two_generations(&ctx, &scratch, replay)
    };

    // None replays nothing of the first generation's records.
    let (_, second) = plans(ReplayChoice::None, "generations-replay-none");
    let plan = &second.plans()[0];
    assert!(plan.replay_file.is_none(), "{plan:#?}");

    // Uniform replays some, and continues the champion.
    let (_, second) = plans(ReplayChoice::Uniform, "generations-replay-uniform");
    let plan = &second.plans()[0];
    assert!(plan.replay_file.is_some());
    assert!(plan.continue_from.is_some() && plan.parent.is_some());

    // A balanced replay gives each capability of the earlier records the same
    // share, as far as it has records.
    let (first, second) = plans(
        ReplayChoice::CapabilityBalanced,
        "generations-replay-balanced",
    );
    let replayed = second.replayed.lock().unwrap().clone();
    assert_eq!(
        replayed.len(),
        1,
        "the second generation was handed a replay"
    );
    let mut per_family = std::collections::BTreeMap::<String, usize>::new();
    for line in replayed[0].lines() {
        let record: serde_json::Value = serde_json::from_str(line).unwrap();
        *per_family
            .entry(record["metadata"]["group"].as_str().unwrap().to_string())
            .or_default() += 1;
    }
    let (most, least) = (
        per_family.values().max().unwrap(),
        per_family.values().min().unwrap(),
    );
    assert!(most - least <= 1, "{per_family:?}");
    // The replay is spread over as many capabilities as it has records and
    // the first generation trained on: a replay of one record is one family
    // however balanced the policy.
    let drawn: usize = per_family.values().sum();
    let available = first.fitted.lock().unwrap()[0].len();
    assert_eq!(per_family.len(), drawn.min(available), "{per_family:?}");

    // A rebuild trains every dataset accumulated on a fresh adapter, still a
    // child of the champion in the lineage, with nothing continued.
    let (_, second) = plans(ReplayChoice::RebuildFromBase, "generations-replay-rebuild");
    let plan = &second.plans()[0];
    assert_eq!(plan.datasets.len(), 2, "the first generation's and the new");
    assert!(plan.continue_from.is_none());
    assert!(plan.parent.is_some());
    assert!(plan.replay_file.is_none());
    assert!(plan.regime == Regime::Sft);
}
