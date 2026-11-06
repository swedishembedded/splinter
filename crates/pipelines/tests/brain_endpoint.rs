// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that reach their model
// only through a serving API, so that many agent attempts share one
// continuously batched inference service, for its clients. If your team
// needs expertise in running agent evaluations against an inference
// service, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: with a brain endpoint, a local model reference is requested from the
//! endpoint, and the attempts of a solve run on it at once.
//!
//! * Several attempts are in flight together (the endpoint sees their
//!   overlap), and the experiences recorded are the ones the same solve at
//!   width one records: the same tasks, attempts and answers, in task order.
//! * A policy the endpoint does not serve is refused by its name before any
//!   attempt is made; nothing falls back to loading it here.
//! * The sampling of a run rides the requests: the configured temperature
//!   reaches the endpoint.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::time::Duration;

use common::gate::fact;
use common::{config, Scratch};
use splinter_agent::CancelToken;
use splinter_core::model_ref::ModelRef;
use splinter_model::scripted_endpoint::{Card, ScriptedEndpoint, SCRIPTED_KEY};
use splinter_model::BrainEndpoints;
use splinter_orchestrator::Context;
use splinter_pipelines::solving::{solve_tasks, SamplingChoice, SolveRequest};
use splinter_store::tasks::{TaskEntry, TaskSet, TaskSetId};

/// What the scripted endpoint serves, as brain's store names it.
const SERVED: &str = "Qwen/Qwen3-0.6B";

fn tasks(ctx: &Context, n: usize) -> TaskSetId {
    let members = (0..n)
        .map(|i| TaskEntry {
            task: ctx.tasks().put(&fact("alpha", i)).unwrap(),
            generator: None,
            prompt: None,
            variant_of: None,
            subject: None,
        })
        .collect();
    ctx.tasks()
        .put_set(&TaskSet {
            name: "endpoint".into(),
            members,
        })
        .unwrap()
}

/// A context whose local models are requested from `server`, `width`
/// attempts at a time.
fn context(scratch: &Scratch, server: &ScriptedEndpoint, width: usize) -> Context {
    let mut settings = config(scratch);
    settings.brain = Some(
        BrainEndpoints::new(vec![server.url()], SCRIPTED_KEY)
            .unwrap()
            .with_attempt_concurrency(width),
    );
    Context::new(settings, false).unwrap()
}

/// What an experience recorded of its attempt: the task, which attempt it
/// was, and what the solver answered.
type Recorded = (String, Option<u32>, Option<String>);

/// Solves six tasks twice each, with the policy requested from a scripted
/// endpoint `width` attempts at a time: the most requests the endpoint saw
/// together, and what the experience set records, in order.
fn solve_through_endpoint(test: &str, width: usize) -> (usize, Vec<Recorded>) {
    let scratch = Scratch::new(test);
    let server = ScriptedEndpoint::start(vec![Card::base(SERVED)])
        .with_delay(Duration::from_millis(150))
        .with_reply(|seen| format!("answer to {}", seen.last_user.len()));
    let ctx = context(&scratch, &server, width);
    let set = tasks(&ctx, 6);
    let solved = solve_tasks(
        &ctx,
        &SolveRequest {
            task_set: &set,
            solver: &ModelRef::policy_default(),
            attempts: 2,
            sampling: SamplingChoice::Own,
            teacher: false,
            system: None,
            work: splinter_agent::work::WorkProtocol::default(),
            deadline: None,
            harness: None,
            cancel: CancelToken::new(),
            resume: None,
        },
    )
    .unwrap();
    assert!(solved.skipped.is_empty(), "{:?}", solved.skipped);
    let recorded = ctx
        .experiences()
        .get_set(&solved.experience_set)
        .unwrap()
        .members
        .iter()
        .map(|id| {
            let experience = ctx.experiences().get(id).unwrap();
            (
                experience.task.id.to_string(),
                experience.provenance.attempt,
                experience.final_output,
            )
        })
        .collect();
    (server.peak_in_flight(), recorded)
}

#[test]
fn attempts_overlap_on_the_endpoint_and_record_what_the_sequential_run_records() {
    let (sequential_peak, sequential) = solve_through_endpoint("endpoint-narrow", 1);
    let (concurrent_peak, concurrent) = solve_through_endpoint("endpoint-wide", 6);

    assert_eq!(sequential_peak, 1, "width one asks one request at a time");
    assert!(
        concurrent_peak >= 4,
        "attempts overlap on the endpoint: at most {concurrent_peak} were in flight together"
    );
    assert_eq!(sequential.len(), 12, "one experience per attempt");
    assert_eq!(
        concurrent, sequential,
        "the same tasks, attempts and answers, in the same order"
    );
}

#[test]
fn a_policy_the_endpoint_does_not_serve_is_refused_by_name_and_never_run_here() {
    let scratch = Scratch::new("endpoint-unserved");
    let server = ScriptedEndpoint::start(vec![Card::base("Qwen/Qwen3-1.7B")]);
    let ctx = context(&scratch, &server, 4);
    let set = tasks(&ctx, 2);

    let refused = solve_tasks(
        &ctx,
        &SolveRequest {
            task_set: &set,
            solver: &ModelRef::policy_default(),
            attempts: 1,
            sampling: SamplingChoice::Own,
            teacher: false,
            system: None,
            work: splinter_agent::work::WorkProtocol::default(),
            deadline: None,
            harness: None,
            cancel: CancelToken::new(),
            resume: None,
        },
    )
    .unwrap_err()
    .to_string();

    assert!(refused.contains(SERVED), "names the model: {refused}");
    assert!(
        refused.contains("Qwen/Qwen3-1.7B"),
        "names what is served: {refused}"
    );
    assert!(server.seen().is_empty(), "no attempt was made");
}

#[test]
fn the_configured_sampling_reaches_the_endpoint_with_every_attempt() {
    let scratch = Scratch::new("endpoint-sampling");
    let server = ScriptedEndpoint::start(vec![Card::base(SERVED)]);
    let mut settings = config(&scratch);
    settings.brain = Some(BrainEndpoints::new(vec![server.url()], SCRIPTED_KEY).unwrap());
    settings.served_temperature = Some(0.6);
    settings.served_top_k = Some(40);
    let ctx = Context::new(settings, false).unwrap();
    let set = tasks(&ctx, 1);

    let solved = solve_tasks(
        &ctx,
        &SolveRequest {
            task_set: &set,
            solver: &ModelRef::policy_default(),
            attempts: 1,
            sampling: SamplingChoice::Own,
            teacher: false,
            system: None,
            work: splinter_agent::work::WorkProtocol::default(),
            deadline: None,
            harness: None,
            cancel: CancelToken::new(),
            resume: None,
        },
    )
    .unwrap();

    let seen = server.seen();
    assert_eq!(seen.len(), 1, "{solved:?}");
    assert!((seen[0].temperature.unwrap() - 0.6).abs() < 1e-6);
    assert_eq!(seen[0].top_k, Some(40));
}
