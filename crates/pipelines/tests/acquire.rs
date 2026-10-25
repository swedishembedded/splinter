// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that turn the documentation of an
// unfamiliar technical domain into a graph of skills they then acquire, for
// its clients. If your team needs expertise in building curricula from
// technical documentation you can procure our services by sending an email
// to info@swedishembedded.com.

//! Spec: `learn "<objective>" --source <documentation>` turns a body of
//! documentation into a measured, trained and assessed set of skills.
//!
//! * the documentation becomes a skill graph (skills grounded in sections,
//!   prerequisites between them), by rule or from a planner model whose
//!   proposal is admitted only as far as the text bears it out, versioned;
//! * every skill has probes written from its own sections, split into dev
//!   probes that measure the policy as rounds go and final probes that only
//!   the closing assessment touches; none is ever trained on;
//! * the baseline competence is measured before any training, rounds train
//!   the skill expected to benefit most (prerequisites first), a candidate is
//!   accepted only when its skill improved and nothing measured before fell
//!   past the bound, and the competence graph is kept revision by revision;
//! * a run is accepted only when every assessed skill reaches the target on
//!   the final probes, against the policy it began with; a candidate that
//!   learns nothing is rejected and the policy is exactly what it was;
//! * a web page is fetched once, converted to Markdown, and traced from the
//!   URL to the source.
//!
//! The policy, the generator, the teacher and the planner are one scripted
//! model; training is a test double whose candidate knows the topics of what
//! it was trained on.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::io::{Read, Write};
use std::net::TcpListener;

use common::gate::FakeTrainer;
use common::zork::{context, model, request, write_manual, Learner};
use common::{scratch_context, Scripted};
use serde_json::json;
use splinter_agent::CancelToken;
use splinter_core::model_ref::ModelRef;
use splinter_knowledge::skills::Refusal;
use splinter_model::train::{Trained, TrainedPreference};
use splinter_orchestrator::{Context, OrchestratorError};
use splinter_pipelines::acquire::{acquire, intake, plan, Outcome, PlannedBy};
use splinter_pipelines::train::{TrainPlan, Trainer};
use splinter_store::runs::{read_run, RunStatus};

#[test]
fn documentation_becomes_skills_probes_rounds_and_an_assessment_that_decided_nothing() {
    let (scratch, ctx) = context("acquire");
    let manual = write_manual(&scratch);
    let recorded = acquire(&ctx, &request(&manual), &Learner).unwrap();
    let report = &recorded.report;

    // The graph is read by rule: a skill a register, the third using the first.
    assert!(
        matches!(report.planned.by, PlannedBy::Structure { .. }),
        "{:?}",
        report.planned.by
    );
    let ids: Vec<&str> = report
        .planned
        .graph
        .skills
        .iter()
        .map(|s| s.id.as_str())
        .collect();
    assert_eq!(ids, ["alpha-register", "beta-register", "gamma-register"]);
    assert_eq!(
        report
            .planned
            .graph
            .get("gamma-register")
            .unwrap()
            .prerequisites,
        ["alpha-register"]
    );
    report.planned.graph.validate().unwrap();

    // Every skill has probes of both halves, and a training task repeats none.
    assert_eq!(report.probes.len(), 3);
    for probes in &report.probes {
        assert_eq!(
            (probes.dev.len(), probes.finals.len()),
            (3, 3),
            "{probes:#?}"
        );
    }

    // The baseline is measured before any training, on a policy that knows
    // nothing.
    let (baseline, _) = &report.baseline;
    assert_eq!(baseline.revision, 1);
    assert_eq!(baseline.skills.len(), 3);
    assert!(baseline
        .skills
        .iter()
        .all(|s| s.passes == 0 && s.graded == 3));

    // The prerequisite is trained first, then the others; every candidate is
    // accepted and the competence graph keeps each revision.
    let trained: Vec<&str> = report.rounds.iter().map(|r| r.skill.as_str()).collect();
    assert_eq!(
        trained,
        ["alpha-register", "beta-register", "gamma-register"]
    );
    assert!(
        report.rounds.iter().all(|r| r.outcome == Outcome::Accepted),
        "{:#?}",
        report.rounds
    );
    let (competence, _) = &report.competence;
    assert!(
        competence
            .skills
            .iter()
            .all(|s| s.passes == 3 && s.graded == 3),
        "{competence:#?}"
    );
    assert_eq!(
        competence.revision,
        1 + 3 * 3,
        "every accepted round updates every measured skill"
    );

    // The closing assessment is on the final probes, against the policy the
    // run began with.
    let assessment = report.assessment.as_ref().unwrap();
    assert!(
        assessment.accepted && assessment.improved == 3,
        "{assessment:#?}"
    );
    for skill in &assessment.skills {
        assert_eq!(
            (skill.baseline.passes, skill.final_score.passes),
            (0, 3),
            "{skill:#?}"
        );
        assert!(skill.interval.unwrap().0 > 0.4);
    }
    assert!(report.accepted);
    assert_eq!(
        report.ended,
        "every skill is mastered or has had its rounds"
    );
    let run = read_run(ctx.workspace(), &recorded.run).unwrap();
    assert_eq!(run.status, RunStatus::Completed);
    let stages: Vec<&str> = run.stages.iter().map(|s| s.stage.as_str()).collect();
    assert!(
        stages.contains(&"round-1") && stages.contains(&"assessment"),
        "{stages:?}"
    );

    // Nothing a probe asks was trained on: every probe is a question about
    // a code below ten, and every record is about a code of ten or more.
    let probe_ids: std::collections::BTreeSet<_> = report
        .probes
        .iter()
        .flat_map(|p| p.dev.iter().chain(&p.finals).cloned())
        .collect();
    for id in ctx.experiences().list().unwrap() {
        let experience = ctx.experiences().get(&id).unwrap();
        if probe_ids.contains(&experience.task.id) {
            assert!(!experience.provenance.teacher, "a teacher answered a probe");
        }
    }
}

#[test]
fn a_candidate_that_learns_nothing_is_rejected_and_the_policy_is_exactly_what_it_was() {
    let (scratch, ctx) = context("acquire-rejected");
    let manual = write_manual(&scratch);
    // Training that teaches nothing: its candidate knows no topic.
    struct Dud;
    impl Trainer for Dud {
        fn train(
            &self,
            ctx: &Context,
            plan: &TrainPlan,
            cancel: &CancelToken,
        ) -> Result<Trained, OrchestratorError> {
            FakeTrainer::knowing(&[]).train(ctx, plan, cancel)
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
    let report = acquire(&ctx, &request(&manual), &Dud).unwrap().report;
    assert!(!report.rounds.is_empty());
    // The first round on a skill is rejected; the generator, asked again,
    // can write nothing the first round did not (its tasks are not made twice),
    // so the second is skipped. Neither is accepted.
    let rejected = report
        .rounds
        .iter()
        .filter(|r| matches!(&r.outcome, Outcome::Rejected { reasons } if reasons.iter().any(|why| why.contains("did not improve"))))
        .count();
    assert_eq!(rejected, 3, "{:#?}", report.rounds);
    assert!(report.rounds.iter().all(|r| matches!(
        r.outcome,
        Outcome::Rejected { .. } | Outcome::Skipped { .. }
    )));
    // Every skill had its two rounds and none was accepted: the competence is
    // what it was, the rounds are counted, and the run is not accepted.
    let (competence, _) = &report.competence;
    assert!(
        competence
            .skills
            .iter()
            .all(|s| s.passes == 0 && s.rounds == 2),
        "{competence:#?}"
    );
    assert!(!report.accepted && report.release.is_none());
    assert_eq!(
        report.ended,
        "every skill is mastered or has had its rounds"
    );
}

#[test]
fn a_planner_is_believed_only_as_far_as_the_text_bears_it_out() {
    let planner = Scripted::new(|prompt| {
        if !prompt.contains("You plan what to learn") {
            return String::new();
        }
        json!({ "skills": [
            { "id": "alpha", "title": "the alpha register", "sections": [1], "symbols": ["alpha"],
              "requirements": [{ "statement": "it returns a code number for each unit",
                                 "quote": "It returns a code number for each unit of the bank" }],
              "constraints": [{ "statement": "never reuse", "quote": "The controller must never reuse a code." }],
              "prerequisites": [] },
            { "id": "beta", "title": "the beta register", "sections": [2], "symbols": ["beta", "not-in-the-text"],
              "requirements": [{ "statement": "invented", "quote": "a sentence nobody wrote" }],
              "prerequisites": ["alpha"] },
            { "id": "gamma", "title": "the gamma register", "sections": [3], "symbols": ["gamma"],
              "requirements": [{ "statement": "it returns a code number",
                                 "quote": "It returns a code number for each unit of the bank" }],
              "prerequisites": ["alpha", "ghost", "gamma"] }
        ] })
        .to_string()
    });
    let (scratch, ctx) = scratch_context("acquire-planner", planner, false);
    let manual = write_manual(&scratch);
    let sources: Vec<_> = intake(&ctx, &[manual.display().to_string()])
        .unwrap()
        .into_iter()
        .map(|s| s.source)
        .collect();
    let planned = plan(
        &ctx,
        "know the registers",
        &sources,
        Some(&ModelRef::policy_default()),
        None,
        &CancelToken::new(),
    )
    .unwrap();
    assert!(
        matches!(planned.by, PlannedBy::Planner { .. }),
        "{:?}",
        planned.by
    );
    let ids: Vec<&str> = planned.graph.skills.iter().map(|s| s.id.as_str()).collect();
    assert_eq!(ids, ["alpha", "gamma"], "beta states nothing the text says");
    let gamma = planned.graph.get("gamma").unwrap();
    assert_eq!(
        gamma.prerequisites,
        ["alpha"],
        "a skill that is none, and itself, are no prerequisites"
    );
    let reasons: Vec<&Refusal> = planned.refused.iter().map(|(_, r)| r).collect();
    assert!(reasons
        .iter()
        .any(|r| matches!(r, Refusal::QuoteNotFound { .. })));
    assert!(reasons.iter().any(|r| matches!(r, Refusal::NoRequirement)));
    assert!(reasons.iter().any(
        |r| matches!(r, Refusal::PrerequisiteDropped { prerequisite } if prerequisite == "ghost")
    ));
    // The same plan again is the revision already kept; a different graph
    // for the same objective is the next, naming it.
    let again = plan(
        &ctx,
        "know the registers",
        &sources,
        Some(&ModelRef::policy_default()),
        None,
        &CancelToken::new(),
    )
    .unwrap();
    assert_eq!(
        (again.address.clone(), again.graph.revision),
        (planned.address.clone(), 1)
    );
    let by_rule = plan(
        &ctx,
        "know the registers",
        &sources,
        None,
        None,
        &CancelToken::new(),
    )
    .unwrap();
    assert_eq!(by_rule.graph.revision, 2);
    assert_eq!(by_rule.graph.parent, Some(planned.address));
}

/// Serves `body` once at `127.0.0.1`, returning the address.
fn serve_once(body: &'static str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request = [0u8; 2048];
        let _ = stream.read(&mut request).unwrap();
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(response.as_bytes()).unwrap();
    });
    format!("http://{address}/manual.html")
}

#[test]
fn a_web_page_is_fetched_converted_to_markdown_and_traced_from_its_url() {
    let (_scratch, ctx) = scratch_context("acquire-url", model(), false);
    let url = serve_once(
        "<html><body><h2><code>alpha</code> register</h2><p>The <code>alpha</code> register returns a code number for each unit of its bank and never reuses one.</p></body></html>",
    );
    let taken = intake(&ctx, std::slice::from_ref(&url)).unwrap();
    let fetched = taken[0].fetched.as_ref().unwrap();
    assert_eq!(fetched.url, url);
    assert!(fetched.converted);
    let source = ctx.sources().get_source(&taken[0].source).unwrap();
    let text =
        String::from_utf8(ctx.sources().read_blob(&source.parts[0].content).unwrap()).unwrap();
    assert!(text.starts_with("## `alpha` register"), "{text}");
    assert!(text.contains("returns a code number"), "{text}");
}

#[test]
fn a_local_latin_1_page_is_decoded_converted_and_traced_like_a_fetched_one() {
    let (scratch, ctx) = scratch_context("acquire-local-html", model(), false);
    let page = scratch.0.join("manual.html");
    std::fs::write(
        &page,
        b"<html><head><meta http-equiv=\"content-type\" content=\"text/html; charset=iso-8859-1\"></head><body><h2><code>alpha</code> register</h2><p>The <code>alpha</code> register returns a code number \xa9 for each unit of its bank.</p></body></html>",
    )
    .unwrap();
    let taken = intake(&ctx, &[page.display().to_string()]).unwrap();
    let fetched = taken[0].fetched.as_ref().unwrap();
    assert_eq!(
        fetched.url,
        format!("file://{}", page.canonicalize().unwrap().display())
    );
    assert!(fetched.converted);
    let source = ctx.sources().get_source(&taken[0].source).unwrap();
    let text =
        String::from_utf8(ctx.sources().read_blob(&source.parts[0].content).unwrap()).unwrap();
    assert!(text.starts_with("## `alpha` register"), "{text}");
    assert!(text.contains("code number \u{a9} for each"), "{text}");
}

#[test]
fn a_structural_graph_is_read_by_rule_without_asking_the_planner() {
    let asked = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counted = asked.clone();
    let planner = Scripted::new(move |prompt| {
        if prompt.contains("You plan what to learn") {
            counted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
        String::new()
    });
    let (scratch, ctx) = scratch_context("acquire-structural", planner, false);
    let manual = write_manual(&scratch);
    let structural = splinter_pipelines::acquire::AcquireRequest {
        structural_graph: true,
        ..request(&manual)
    };
    // Whatever becomes of the run, the graph is the sources' own.
    let _ = acquire(&ctx, &structural, &Learner);
    assert_eq!(asked.load(std::sync::atomic::Ordering::SeqCst), 0);
    let _ = acquire(&ctx, &request(&manual), &Learner);
    assert!(asked.load(std::sync::atomic::Ordering::SeqCst) > 0);
}

#[test]
fn a_run_without_probes_says_why_the_generator_wrote_none() {
    let unusable = Scripted::new(|_| "I would write some tasks about registers.".into());
    let (scratch, ctx) = scratch_context("acquire-no-probes", unusable, false);
    let manual = write_manual(&scratch);
    let refused = acquire(
        &ctx,
        &splinter_pipelines::acquire::AcquireRequest {
            structural_graph: true,
            ..request(&manual)
        },
        &Learner,
    )
    .unwrap_err()
    .to_string();
    assert!(refused.contains("no skill has enough probes"), "{refused}");
    // The replies' fate is named with its count: `(refused: <reason> <n>, ...)`.
    let named = refused
        .split_once("(refused: ")
        .map(|(_, rest)| rest.trim_end_matches(')'))
        .unwrap_or_else(|| panic!("no refusals named: {refused}"));
    assert!(
        named.split(", ").all(|reason| reason
            .rsplit_once(' ')
            .is_some_and(|(why, n)| !why.is_empty() && n.parse::<usize>().is_ok())),
        "{refused}"
    );
}
