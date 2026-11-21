// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning agents that acquire a capability
// from a document or a tool and prove it with evidence, for its clients. If
// your team needs expertise in continual learning or agent evaluation, you
// can procure our services by sending an email to info@swedishembedded.com.

//! Spec: a model reached over an API is asked up to the configured width at
//! once, a model a brain server decodes as wide as its slots, and the
//! experience set is the same either way: one member per attempt, in the
//! order of the task set.
//!
//! How wide a solve asks is read without a clock: each request waits at a
//! barrier as wide as the width the spec expects, so the barrier fills only
//! if that many requests are in flight together, and a request beyond the
//! width shows as a peak above it. A barrier that never fills is a defect,
//! reported after a long liveness bound (never reached by a correct solve,
//! however loaded the machine) instead of hanging the suite.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use crate::common;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use common::gate::fact;
use common::{config, Scratch};
use splinter_agent::solve::Model;
use splinter_agent::CancelToken;
use splinter_core::model_ref::ModelRef;
use splinter_orchestrator::Context;
use splinter_pipelines::solving::{solve_tasks, SamplingChoice, SolveRequest};
use splinter_store::tasks::{TaskEntry, TaskSet, TaskSetId};
use sven_sdk::model::{CompletionRequest, ModelProvider, ResponseEvent, ResponseStream};

/// How long a request waits for its barrier to fill before the test calls
/// the solve too narrow: a liveness bound for a defect, not a timing
/// assumption (a correct solve fills it as soon as its requests start).
const STALLED_AFTER: Duration = Duration::from_secs(120);

/// A model that counts how many requests are waiting on it at once, and
/// holds each one until as many as `together` are waiting (or, without a
/// barrier, for `hold`).
struct Counting {
    waiting: AtomicUsize,
    most: AtomicUsize,
    together: Option<tokio::sync::Barrier>,
    hold: Duration,
    stalled: AtomicBool,
}

impl Counting {
    /// Each request waits until `width` are in flight together.
    fn at_barrier(width: usize) -> Self {
        Self {
            waiting: AtomicUsize::new(0),
            most: AtomicUsize::new(0),
            together: Some(tokio::sync::Barrier::new(width)),
            hold: Duration::ZERO,
            stalled: AtomicBool::new(false),
        }
    }

    /// Each request takes at least `hold` to answer.
    fn taking(hold: Duration) -> Self {
        Self {
            together: None,
            hold,
            ..Self::at_barrier(1)
        }
    }
}

#[async_trait::async_trait]
impl ModelProvider for Counting {
    fn name(&self) -> &str {
        "counting"
    }
    fn model_name(&self) -> &str {
        "counting-1"
    }
    async fn complete(&self, _: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let now = self.waiting.fetch_add(1, Ordering::SeqCst) + 1;
        self.most.fetch_max(now, Ordering::SeqCst);
        if let Some(barrier) = &self.together {
            if tokio::time::timeout(STALLED_AFTER, barrier.wait())
                .await
                .is_err()
            {
                self.stalled.store(true, Ordering::SeqCst);
            }
        }
        tokio::time::sleep(self.hold).await;
        self.waiting.fetch_sub(1, Ordering::SeqCst);
        Ok(Box::pin(futures::stream::iter(vec![
            Ok(ResponseEvent::TextDelta("an answer".into())),
            Ok(ResponseEvent::Done),
        ])))
    }
}

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
            name: "concurrency".into(),
            members,
        })
        .unwrap()
}

/// Solves six tasks twice each with a model behind `reference` that holds
/// each request until `expected` are in flight; the most that waited on it
/// at once, and the tasks the experience set's members answer, in order.
fn solve_with(
    test: &str,
    reference: ModelRef,
    width: usize,
    expected: usize,
) -> (usize, Vec<String>, Vec<String>) {
    let scratch = Scratch::new(test);
    let mut settings = config(&scratch);
    settings.remote_concurrency = width;
    let ctx = Context::new(settings, true).unwrap();
    let model = Arc::new(Counting::at_barrier(expected));
    ctx.add_model(
        reference.clone(),
        Model::new(model.clone(), "counting/model"),
    );
    let set = tasks(&ctx, 6);
    let solved = solve_tasks(
        &ctx,
        &SolveRequest {
            task_set: &set,
            solver: &reference,
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
    let wanted: Vec<String> = ctx
        .tasks()
        .get_set(&set)
        .unwrap()
        .members
        .iter()
        .map(|m| m.task.to_string())
        .collect();
    let answered: Vec<String> = ctx
        .experiences()
        .get_set(&solved.experience_set)
        .unwrap()
        .members
        .iter()
        .map(|id| ctx.experiences().get(id).unwrap().task.id.to_string())
        .collect();
    assert!(
        !model.stalled.load(Ordering::SeqCst),
        "{expected} requests were never in flight together (at most {})",
        model.most.load(Ordering::SeqCst)
    );
    (model.most.load(Ordering::SeqCst), wanted, answered)
}

#[test]
fn a_remote_model_is_asked_up_to_the_configured_width_at_once() {
    let remote: ModelRef = "remote:test/model".parse().unwrap();
    let (most, wanted, answered) = solve_with("remote-wide", remote, 3, 3);
    assert_eq!(most, 3, "three requests are in flight together");
    let expected: Vec<String> = wanted.iter().flat_map(|t| [t.clone(), t.clone()]).collect();
    assert_eq!(answered.len(), 12, "one member per attempt");
    assert_eq!(
        answered
            .iter()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>(),
        wanted
            .iter()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>()
    );
    // Members follow the task set's order, attempts of a task together.
    let mut collapsed = answered.clone();
    collapsed.dedup();
    assert_eq!(collapsed, wanted, "{expected:?} vs {answered:?}");
}

#[test]
fn a_local_model_is_asked_as_wide_as_the_brain_server_decodes_together() {
    let local: ModelRef = "local:/models/base".parse().unwrap();
    let (most, ..) = solve_with("local-wide", local, 3, 12);
    // Every attempt of the solve (12) fits the server's slots, so all are in flight together.
    assert_eq!(
        most, 12,
        "requests overlap instead of queueing one at a time"
    );
}

#[test]
fn a_model_is_handed_to_a_stage_with_the_width_it_is_asked_at() {
    let scratch = Scratch::new("model-slots");
    let mut settings = config(&scratch);
    settings.remote_concurrency = 5;
    let ctx = Context::new(settings, true).unwrap();
    let model = Arc::new(Counting::taking(Duration::ZERO));
    let remote: ModelRef = "remote:test/model".parse().unwrap();
    let local: ModelRef = "local:/models/base".parse().unwrap();
    for reference in [&remote, &local] {
        ctx.add_model(
            reference.clone(),
            Model::new(model.clone(), "counting/model"),
        );
    }
    assert_eq!(ctx.model(&remote).unwrap().slots(), 5);
    assert_eq!(
        ctx.model(&local).unwrap().slots(),
        splinter_model::server::DEFAULT_SERVE_SLOTS as usize
    );
    // A model whose owner says how many requests it decodes together keeps
    // that width, whatever its reference.
    let served: ModelRef = "local:/models/served".parse().unwrap();
    ctx.add_model(
        served.clone(),
        Model::new(model, "counting/served").with_slots(16),
    );
    assert_eq!(ctx.model(&served).unwrap().slots(), 16);
}

#[test]
fn every_solved_attempt_keeps_the_facts_the_experience_cannot() {
    use splinter_core::annotation::AnnotationBody;
    let scratch = Scratch::new("attempt-facts");
    let ctx = Context::new(config(&scratch), true).unwrap();
    let reference: ModelRef = "remote:test/model".parse().unwrap();
    let model = Arc::new(Counting::taking(Duration::from_millis(60)));
    ctx.add_model(reference.clone(), Model::new(model, "counting/model"));
    let set = tasks(&ctx, 2);
    let solved = solve_tasks(
        &ctx,
        &SolveRequest {
            task_set: &set,
            solver: &reference,
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
    let members = ctx
        .experiences()
        .get_set(&solved.experience_set)
        .unwrap()
        .members;
    assert_eq!(members.len(), 2);
    for id in &members {
        let reports: Vec<_> = ctx
            .experiences()
            .annotations(id)
            .unwrap()
            .annotations
            .into_iter()
            .filter_map(|n| match n.body {
                AnnotationBody::AttemptReport(report) => Some(report),
                _ => None,
            })
            .collect();
        assert_eq!(reports.len(), 1);
        // perf-number: the scripted model's configured delay, not a measurement
        // The model takes at least 60 ms to answer (a lower bound, whatever
        // the load); an answer in one reply has no tool calls to count, and
        // nothing else was measured.
        assert!(reports[0].wall_seconds.unwrap() >= 0.05);
        assert_eq!(reports[0].tool_calls, None);
        assert_eq!(reports[0].gpu_seconds, None);
    }
}
