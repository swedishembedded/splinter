// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements paired, family-clustered comparisons of a
// trained code-editing agent against its untrained base, for its clients. If
// your team needs expertise in showing that fine-tuning an agent beat its
// base on held-out repository tasks, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec: `experiences compare` pairs two experience sets by task, counts a
//! family once, applies the store's decision rule, and leaves attempts that
//! say nothing about the model - exposed to what only the grader holds, or
//! never graded because the harness failed - out of every number while
//! counting them.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use atif::{AgentProfile, Trajectory};
use common::gate::{gate_context, Brain};
use serde_json::json;
use splinter_core::annotation::{Annotation, AnnotationBody, Outcome, Producer, Strength};
use splinter_core::clock::FixedClock;
use splinter_core::experience::{Environment, Experience, Provenance, Task};
use splinter_core::rng::Seed;
use splinter_eval::exposure::EXPOSURE_PRODUCER;
use splinter_orchestrator::Context;
use splinter_pipelines::compare::compare;
use splinter_store::experiences::{ExperienceSet, SetId};

fn task(family: &str, n: usize) -> Task {
    Task::new(
        "workspace-change",
        Vec::new(),
        Environment::closed_book(),
        format!("Fix the fault {n} in {family}."),
        Vec::new(),
    )
    .unwrap()
    .with_concepts([family])
    .unwrap()
}

/// How an attempt was graded.
#[derive(Clone, Copy)]
enum Graded {
    Pass,
    Fail,
    /// The exposure audit failed it, whatever else was said.
    Exposed,
    /// No verifier decided it: the harness could not run.
    Undecided,
}

fn attempt(
    ctx: &Context,
    task: &Task,
    solver: &str,
    graded: Graded,
) -> splinter_core::experience::ExperienceId {
    let profile = AgentProfile {
        name: "t".into(),
        version: "1".into(),
        model_name: None,
        tool_definitions: None,
        extra: None,
    };
    let experience = Experience::new(
        task.clone(),
        Trajectory::new("ATIF-v1.7", profile),
        Some("patch".into()),
        Provenance::new(solver, &FixedClock::new("2026-10-16T00:00:00.000Z")),
    )
    .unwrap();
    let id = ctx.experiences().put(&experience).unwrap();
    let note = |producer: &str, outcome, strength, evidence| Annotation {
        experience: id.clone(),
        producer: Producer {
            name: producer.into(),
            version: "1".into(),
        },
        body: AnnotationBody::Verdict {
            outcome,
            strength,
            evidence,
        },
    };
    let notes = match graded {
        Graded::Pass => vec![note(
            "tests",
            Outcome::Pass,
            Strength::Executable,
            json!({}),
        )],
        Graded::Fail => vec![note(
            "tests",
            Outcome::Fail,
            Strength::Executable,
            json!({}),
        )],
        Graded::Exposed => vec![
            note("tests", Outcome::Pass, Strength::Executable, json!({})),
            note(
                EXPOSURE_PRODUCER,
                Outcome::Fail,
                Strength::Constraint,
                json!({ "exposed": [{ "kind": "hidden_path" }] }),
            ),
        ],
        Graded::Undecided => Vec::new(),
    };
    for n in &notes {
        ctx.experiences().annotate(n).unwrap();
    }
    id
}

fn arm(ctx: &Context, solver: &str, attempts: &[(&Task, Graded)]) -> SetId {
    let members = attempts
        .iter()
        .map(|(task, graded)| attempt(ctx, task, solver, *graded))
        .collect();
    ctx.experiences()
        .put_set(&ExperienceSet {
            name: solver.into(),
            members,
        })
        .unwrap()
}

#[test]
fn two_arms_are_paired_by_task_a_family_counts_once_and_exposed_or_unrun_attempts_are_counted_apart(
) {
    let (_scratch, ctx) = gate_context("compare-sets", Brain::Honest);
    let (a1, a2, b1, b2, c1) = (
        task("a.py", 1),
        task("a.py", 2),
        task("b.py", 1),
        task("b.py", 2),
        task("c.py", 1),
    );
    use Graded::{Exposed, Fail, Pass, Undecided};
    let candidate = arm(
        &ctx,
        "candidate",
        &[
            (&a1, Pass),
            (&a2, Exposed),
            (&b1, Pass),
            (&b2, Pass),
            (&c1, Fail),
        ],
    );
    let baseline = arm(
        &ctx,
        "baseline",
        &[
            (&a1, Fail),
            (&a2, Pass),
            (&b1, Pass),
            (&b2, Undecided),
            (&c1, Pass),
        ],
    );

    // seed: the interval seed only draws the bootstrap; the asserted counts, p-value and agreement do not depend on it, and a fixed one keeps the interval output reproducible.
    let seed = Seed::new(5);
    let cmp = compare(&ctx, &candidate, &baseline, seed).unwrap();
    // Five tasks; a2 (exposed) and b2 (never graded) have no verdict in
    // both arms, so three are paired: a win, a tie and a loss.
    assert_eq!((cmp.tasks, cmp.paired, cmp.unpaired), (5, 3, 2));
    assert_eq!(
        (cmp.candidate.exposed, cmp.candidate.harness_failed),
        (1, 0)
    );
    assert_eq!((cmp.baseline.exposed, cmp.baseline.harness_failed), (0, 1));
    // The exposed attempt is in no denominator: the candidate was graded on
    // four attempts and passed three of them.
    assert_eq!((cmp.candidate.graded, cmp.baseline.graded), (4, 4));
    assert_eq!(cmp.candidate.competence.counts.attempted, 4);
    assert_eq!(cmp.candidate.competence.counts.verified, 3);
    assert_eq!(cmp.candidate.competence.quality.value, Some(0.75));
    // One family won, one tied, one lost; the sign test counts the two
    // discordant families: one candidate win in two, p = 3/4.
    assert_eq!(
        (cmp.clusters.wins, cmp.clusters.losses, cmp.clusters.ties),
        (1, 1, 1)
    );
    assert_eq!(cmp.sign_test.discordant, 2);
    assert_eq!(cmp.sign_test.candidate_wins, 1);
    assert!(
        (cmp.sign_test.p_value - 0.75).abs() < 1e-12,
        "{:?}",
        cmp.sign_test
    );
    // The verdicts agree on one paired task of three.
    assert!((cmp.agreement.unwrap() - 1.0 / 3.0).abs() < 1e-12);

    let family = |name: &str| cmp.families.iter().find(|f| f.family == name).unwrap();
    assert_eq!(cmp.families.len(), 3);
    let a = family("a.py");
    assert_eq!((a.paired, a.candidate_right, a.baseline_right), (1, 1, 0));
    assert_eq!((a.wins, a.losses, a.ties), (1, 0, 0));
    let c = family("c.py");
    assert_eq!((c.wins, c.losses, c.ties), (0, 1, 0));

    // The same sets and seed give the same report, and it is plain data.
    let again = compare(&ctx, &candidate, &baseline, seed).unwrap();
    assert_eq!(
        serde_json::to_string(&cmp).unwrap(),
        serde_json::to_string(&again).unwrap()
    );
}
