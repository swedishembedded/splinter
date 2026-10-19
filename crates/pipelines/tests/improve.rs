// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the choice between retraining, a better
// harness, more search or no change at all, on measured cost-adjusted gain,
// for its clients. If your team needs expertise in deciding how an agent
// should be improved before spending a training budget, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: `improve` chooses among interventions by what each was measured to
//! do on the target capability, net of what it costs.
//!
//! * on three synthetic failure clusters, with outcomes drawn from a known
//!   world, the choice differs by cluster - a better harness where the model
//!   knows but the harness fails it, a teacher's demonstrations and training
//!   where knowledge is missing, more samples where search beats training -
//!   and its realised cost-adjusted value beats every fixed strategy;
//! * where nothing helps enough to pay for itself, nothing changes;
//! * an intervention whose mechanism is not available is reported as not
//!   measured, never as a zero gain;
//! * measured by solving real tasks with scripted agents, a trained
//!   candidate that mends what the incumbent cannot is chosen, and the
//!   decision is a recorded run.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use common::gate::{put_base, FakeTrainer};
use common::repo::{project, Mender};
use common::{config, Scratch};
use splinter_agent::solve::Model;
use splinter_agent::CancelToken;
use splinter_core::clock::FixedClock;
use splinter_core::digest::Digest;
use splinter_core::model_ref::ModelRef;
use splinter_eval::intervention::{Decision, Pricing, DEFAULT_CONFIDENCE_Z};
use splinter_orchestrator::error::OrchestratorError;
use splinter_orchestrator::Context;
use splinter_pipelines::curriculum::frontier::PassAtK;
use splinter_pipelines::improve::{
    fixed_comparison, improve, ArmRunner, Evaluation, ImprovementExperiment, InterventionSpec,
    Measured, ProtocolSpec, SolveArms,
};
use splinter_pipelines::practice::{practice, PracticeRequest};
use splinter_store::runs::{read_run, RunStatus};
use splinter_store::tasks::TaskSetId;

/// A failure cluster of the synthetic world: the probability a task is
/// solved by the incumbent and under each intervention.
struct Cluster {
    name: &'static str,
    incumbent: f64,
    samples: f64,
    harness: f64,
    sft: f64,
    teacher_sft: f64,
}

const FAMILIES: usize = 8;
const TASKS_PER_FAMILY: usize = 10;

/// What a use of the incumbent costs, in seconds.
const USE_SECONDS: f64 = 1.0;
const SAMPLES: usize = 8;
const SFT_SECONDS: f64 = 3000.0;
const TEACHER_SFT_SECONDS: f64 = 4500.0;

const CLUSTERS: [Cluster; 4] = [
    // The model knows how: its harness cuts the work short.
    Cluster {
        name: "bad-harness",
        incumbent: 0.2,
        samples: 0.3,
        harness: 0.8,
        sft: 0.3,
        teacher_sft: 0.35,
    },
    // The model lacks the knowledge: only learning it helps.
    Cluster {
        name: "missing-knowledge",
        incumbent: 0.05,
        samples: 0.1,
        harness: 0.05,
        sft: 0.45,
        teacher_sft: 0.65,
    },
    // Many samples find what training barely moves.
    Cluster {
        name: "search-beats-training",
        incumbent: 0.3,
        samples: 0.85,
        harness: 0.32,
        sft: 0.4,
        teacher_sft: 0.45,
    },
    // Already good: nothing is worth its cost.
    Cluster {
        name: "already-good",
        incumbent: 0.9,
        samples: 0.92,
        harness: 0.9,
        sft: 0.9,
        teacher_sft: 0.91,
    },
];

/// The uniform draw that decides whether task `task` of family `family` of
/// `cluster` is solved: the same for every arm, so arms are paired, and a
/// function of the names alone.
fn draw(cluster: &str, family: usize, task: usize) -> f64 {
    let digest = Digest::of(format!("{cluster}/{family}/{task}").as_bytes());
    let top = u32::from_str_radix(&digest.hex()[..8], 16).unwrap();
    f64::from(top) / f64::from(u32::MAX)
}

/// The synthetic world as an arm runner: each family's pass rate under an
/// arm, minus the incumbent's, over the same tasks.
struct World<'a>(&'a Cluster);

impl World<'_> {
    fn probability(&self, label: &str) -> Option<f64> {
        let c = self.0;
        Some(match label {
            "no-change" => c.incumbent,
            "samples-8" => c.samples,
            "longer-protocol" => c.harness,
            "sft" => c.sft,
            "teacher-sft" => c.teacher_sft,
            _ => return None,
        })
    }
}

impl ArmRunner for World<'_> {
    fn measure(
        &mut self,
        _: &ImprovementExperiment,
        spec: &InterventionSpec,
    ) -> Result<Measured, OrchestratorError> {
        let label = spec.label();
        let p = self
            .probability(&label)
            .ok_or_else(|| OrchestratorError::Refused(format!("no such arm {label}")))?;
        let deltas = (0..FAMILIES)
            .map(|family| {
                let solved = |p: f64| {
                    (0..TASKS_PER_FAMILY)
                        .filter(|t| draw(self.0.name, family, *t) < p)
                        .count() as f64
                        / TASKS_PER_FAMILY as f64
                };
                solved(p) - solved(self.0.incumbent)
            })
            .collect();
        let (one_time_cost, per_use_cost) = match spec {
            InterventionSpec::MoreSamples { samples } => {
                (0.0, (*samples as f64 - 1.0) * USE_SECONDS)
            }
            InterventionSpec::Protocol { .. } => (0.0, 0.5 * USE_SECONDS),
            InterventionSpec::Candidate {
                one_time_seconds, ..
            } => (*one_time_seconds, 0.0),
            _ => (0.0, 0.0),
        };
        Ok(Measured {
            deltas,
            one_time_cost,
            per_use_cost,
            experience_set: None,
            note: None,
        })
    }
}

fn pricing() -> Pricing {
    Pricing {
        value_per_success: 60.0,
        expected_uses: 200.0,
        confidence_z: DEFAULT_CONFIDENCE_Z,
        min_units: 4,
        budget: None,
    }
}

fn interventions() -> Vec<InterventionSpec> {
    let candidate = |label: &str, seconds| InterventionSpec::Candidate {
        label: label.into(),
        model: "local:qwen3-8b+adapter".parse().unwrap(),
        one_time_seconds: seconds,
    };
    vec![
        InterventionSpec::NoChange,
        InterventionSpec::MoreSamples { samples: SAMPLES },
        InterventionSpec::Protocol {
            label: "longer-protocol".into(),
            protocol: ProtocolSpec {
                follow_ups: 6,
                max_tool_calls: Some(200),
                max_repeated_results: Some(5),
            },
        },
        candidate("sft", SFT_SECONDS),
        candidate("teacher-sft", TEACHER_SFT_SECONDS),
        InterventionSpec::Unavailable {
            label: "search".into(),
            reason: "sven's search facility has not landed".into(),
        },
    ]
}

fn experiment(tasks: TaskSetId, interventions: Vec<InterventionSpec>) -> ImprovementExperiment {
    ImprovementExperiment {
        target_capability: "repair failing tests".into(),
        target_tasks: tasks,
        incumbent: ModelRef::policy_default(),
        evaluation: Evaluation {
            pass_at_k: PassAtK::new(2, None, None),
            protocol: ProtocolSpec::default(),
        },
        interventions,
        pricing: pricing(),
    }
}

fn context(test: &str) -> (Scratch, Context) {
    let scratch = Scratch::new(test);
    let ctx = Context::new(config(&scratch), false)
        .unwrap()
        .with_clock(Box::new(FixedClock::new("2026-10-19T08:00:00.000Z")))
        .with_model(
            ModelRef::policy_default(),
            Model::new(Mender::new(false), "scripted/idle-1"),
        );
    put_base(&ctx);
    (scratch, ctx)
}

fn some_tasks(ctx: &Context, scratch: &Scratch) -> TaskSetId {
    let repo = scratch.0.join("repo");
    project(&repo, &[]);
    let recorded = practice(
        ctx,
        &PracticeRequest {
            repository: repo,
            revision: "HEAD".into(),
            max_tasks: Some(32),
            mutants_per_file: Some(1),
            exam_families: Some(0),
            pass_at_k: PassAtK::new(2, None, None),
            budget: Some(std::time::Duration::from_secs(1)),
            no_release: true,
            ..PracticeRequest::default()
        },
        &FakeTrainer::knowing(&[]),
    )
    .unwrap();
    recorded.report.tasks.unwrap().task_set.unwrap()
}

/// What an arm is worth in the world, as it is and not as measured: the gain
/// in the cluster's probability over every use, priced, less its cost.
fn realised(cluster: &Cluster, label: &str) -> f64 {
    let p = World(cluster).probability(label).unwrap();
    let cost = match label {
        "samples-8" => (SAMPLES as f64 - 1.0) * USE_SECONDS * pricing().expected_uses,
        "longer-protocol" => 0.5 * USE_SECONDS * pricing().expected_uses,
        "sft" => SFT_SECONDS,
        "teacher-sft" => TEACHER_SFT_SECONDS,
        _ => 0.0,
    };
    pricing().expected_uses * pricing().value_per_success * (p - cluster.incumbent) - cost
}

#[test]
fn the_choice_differs_by_failure_cluster_and_beats_every_fixed_strategy() {
    let (scratch, ctx) = context("improve-clusters");
    let tasks = some_tasks(&ctx, &scratch);
    let mut chosen = Vec::new();
    let mut reports = Vec::new();
    for cluster in &CLUSTERS {
        let recorded = improve(
            &ctx,
            &experiment(tasks.clone(), interventions()),
            &mut World(cluster),
        )
        .unwrap();
        let label = match &recorded.report.verdict.decision {
            Decision::Apply { label, .. } => label.clone(),
            Decision::NoChange { .. } => "no-change".to_string(),
        };
        chosen.push((cluster.name, label));
        reports.push(recorded.report);
    }
    assert_eq!(
        chosen,
        [
            ("bad-harness", "longer-protocol".to_string()),
            ("missing-knowledge", "teacher-sft".to_string()),
            ("search-beats-training", "samples-8".to_string()),
            ("already-good", "no-change".to_string()),
        ]
    );

    // Measured, the per-cluster choice is worth at least as much as every
    // fixed strategy over the clusters the strategy was measured in.
    let comparison = fixed_comparison(&reports.iter().collect::<Vec<_>>());
    assert!(comparison.beats_every_fixed_strategy, "{comparison:#?}");
    assert!(comparison.strategies.contains_key("teacher-sft"));
    assert!(
        !comparison.strategies.contains_key("search"),
        "an arm never measured is no strategy"
    );
    let best_fixed = comparison
        .strategies
        .values()
        .filter(|s| s.experiments == CLUSTERS.len())
        .map(|s| s.net)
        .fold(f64::MIN, f64::max);
    assert!(comparison.selected_net > best_fixed, "{comparison:#?}");

    // Realised value over all clusters, against each strategy that applies
    // one intervention to everything.
    let selected: f64 = CLUSTERS
        .iter()
        .zip(&chosen)
        .map(|(cluster, (_, label))| realised(cluster, label))
        .sum();
    for fixed in [
        "samples-8",
        "longer-protocol",
        "sft",
        "teacher-sft",
        "no-change",
    ] {
        let total: f64 = CLUSTERS.iter().map(|c| realised(c, fixed)).sum();
        assert!(
            selected > total,
            "choosing per cluster ({selected}) must beat always {fixed} ({total})"
        );
    }
}

#[test]
fn an_intervention_without_a_mechanism_is_not_measured_and_never_scored_as_zero() {
    let (scratch, ctx) = context("improve-unavailable");
    let tasks = some_tasks(&ctx, &scratch);
    let recorded = improve(
        &ctx,
        &experiment(tasks, interventions()),
        &mut World(&CLUSTERS[2]),
    )
    .unwrap();
    let search = recorded
        .report
        .arms
        .iter()
        .find(|arm| arm.label == "search")
        .unwrap();
    assert!(search.measured.is_none());
    assert!(search.note.as_deref().unwrap().contains("not measured"));
    assert!(recorded
        .report
        .verdict
        .scores
        .iter()
        .all(|score| score.label != "search"));
}

#[test]
fn an_experiment_that_cannot_be_answered_is_refused_before_anything_is_recorded() {
    let (scratch, ctx) = context("improve-refused");
    let tasks = some_tasks(&ctx, &scratch);
    for bad in [
        experiment(tasks.clone(), Vec::new()),
        experiment(
            tasks.clone(),
            vec![InterventionSpec::NoChange, InterventionSpec::NoChange],
        ),
        experiment(
            tasks.clone(),
            vec![InterventionSpec::MoreSamples { samples: 1 }],
        ),
    ] {
        let refused = improve(&ctx, &bad, &mut World(&CLUSTERS[0])).unwrap_err();
        assert!(refused.is_refusal(), "{refused}");
    }
}

#[test]
fn a_trained_candidate_that_mends_what_the_incumbent_cannot_is_measured_by_solving_and_chosen() {
    let (scratch, ctx) = context("improve-solving");
    let ctx = ctx.with_model(
        "local:strong".parse().unwrap(),
        Model::new(Mender::new(true), "scripted/strong-1"),
    );
    let tasks = some_tasks(&ctx, &scratch);
    let interventions = vec![
        InterventionSpec::NoChange,
        InterventionSpec::MoreSamples { samples: 3 },
        InterventionSpec::Candidate {
            label: "trained".into(),
            model: "local:strong".parse().unwrap(),
            one_time_seconds: 1.0,
        },
    ];
    let mut arms = SolveArms::new(&ctx, CancelToken::new());
    let recorded = improve(&ctx, &experiment(tasks, interventions), &mut arms).unwrap();

    let report = &recorded.report;
    match &report.verdict.decision {
        Decision::Apply { label, .. } => assert_eq!(label, "trained"),
        Decision::NoChange { reason } => panic!("nothing was chosen: {reason}"),
    }
    let measured = |label: &str| {
        report
            .arms
            .iter()
            .find(|a| a.label == label)
            .and_then(|a| a.measured.clone())
            .unwrap()
    };
    // The idle incumbent solves nothing: more of its samples gain nothing,
    // the candidate gains everything in every family.
    assert!(measured("samples-3").deltas.iter().all(|d| *d == 0.0));
    let trained = measured("trained");
    assert!(trained.deltas.len() >= 4, "{trained:?}");
    assert!(trained.deltas.iter().all(|d| *d > 0.99), "{trained:?}");
    assert!(report.arms.iter().any(|a| a.experience_set.is_some()));

    let run = read_run(ctx.workspace(), &recorded.run).unwrap();
    assert_eq!(run.status, RunStatus::Completed);
}
