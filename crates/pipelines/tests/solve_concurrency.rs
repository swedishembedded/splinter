// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning agents that acquire a capability
// from a document or a tool and prove it with evidence, for its clients. If
// your team needs expertise in continual learning or agent evaluation, you
// can procure our services by sending an email to info@swedishembedded.com.

//! Spec: a model reached over an API is asked several tasks at once, a model
//! on the local device one at a time, and the experience set is the same
//! either way: one member per attempt, in the order of the task set.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
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

/// A model that takes a while to answer, and counts how many are waiting on
/// it at once.
struct Slow {
    waiting: AtomicUsize,
    most: AtomicUsize,
}

#[async_trait::async_trait]
impl ModelProvider for Slow {
    fn name(&self) -> &str {
        "slow"
    }
    fn model_name(&self) -> &str {
        "slow-1"
    }
    async fn complete(&self, _: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let now = self.waiting.fetch_add(1, Ordering::SeqCst) + 1;
        self.most.fetch_max(now, Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(60)).await;
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

/// Solves six tasks twice each with `slow` as `reference`; the most that
/// waited on it at once, and the tasks the experience set's members answer,
/// in order.
fn solve_with(test: &str, reference: ModelRef, width: usize) -> (usize, Vec<String>, Vec<String>) {
    let scratch = Scratch::new(test);
    let mut settings = config(&scratch);
    settings.remote_concurrency = width;
    let ctx = Context::new(settings, true).unwrap();
    let slow = Arc::new(Slow {
        waiting: AtomicUsize::new(0),
        most: AtomicUsize::new(0),
    });
    ctx.add_model(reference.clone(), Model::new(slow.clone(), "slow/model"));
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
    (slow.most.load(Ordering::SeqCst), wanted, answered)
}

#[test]
fn a_remote_model_is_asked_up_to_the_configured_width_at_once() {
    let remote: ModelRef = "remote:test/model".parse().unwrap();
    let (most, wanted, answered) = solve_with("remote-wide", remote, 3);
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
fn a_model_on_the_local_device_is_asked_one_request_at_a_time() {
    let local: ModelRef = "local:/models/base".parse().unwrap();
    let (most, ..) = solve_with("local-narrow", local, 3);
    assert_eq!(most, 1);
}

#[test]
fn every_solved_attempt_keeps_the_facts_the_experience_cannot() {
    use splinter_core::annotation::AnnotationBody;
    let scratch = Scratch::new("attempt-facts");
    let ctx = Context::new(config(&scratch), true).unwrap();
    let reference: ModelRef = "remote:test/model".parse().unwrap();
    let slow = Arc::new(Slow {
        waiting: AtomicUsize::new(0),
        most: AtomicUsize::new(0),
    });
    ctx.add_model(reference.clone(), Model::new(slow, "slow/model"));
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
        // The model takes 60 ms to answer; an answer in one reply has no
        // tool calls to count, and nothing else was measured.
        assert!(reports[0].wall_seconds.unwrap() >= 0.05);
        assert_eq!(reports[0].tool_calls, None);
        assert_eq!(reports[0].gpu_seconds, None);
    }
}
