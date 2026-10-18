// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning agents that acquire a capability
// from a document or a tool and prove it with evidence, for its clients. If
// your team needs expertise in long-running agent evaluations that survive
// infrastructure failures then you can procure our services by sending an
// email to info@swedishembedded.com.

//! Spec: a solver that cannot be reached for one task does not cost the
//! tasks that completed. The failed attempt is reported as an infrastructure
//! error and has no experience (it is not a failure of the solver); a solve
//! resumed from the experience set it produced asks only the missing task and
//! returns every experience.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use common::gate::{fact, question};
use common::{config, Scratch};
use splinter_agent::solve::Model;
use splinter_agent::CancelToken;
use splinter_core::model_ref::ModelRef;
use splinter_orchestrator::Context;
use splinter_pipelines::solving::{solve_set, Solved};
use splinter_store::tasks::{TaskEntry, TaskSet};
use sven_sdk::model::{CompletionRequest, ModelProvider, ResponseEvent, ResponseStream};

/// A model that fails, as a dead server does, for the question it was told
/// to fail on while `failing` is set, and counts what it was asked.
struct Flaky {
    victim: String,
    failure: &'static str,
    failing: AtomicBool,
    asked: AtomicUsize,
}

#[async_trait::async_trait]
impl ModelProvider for Flaky {
    fn name(&self) -> &str {
        "flaky"
    }
    fn model_name(&self) -> &str {
        "flaky-1"
    }
    async fn complete(&self, request: CompletionRequest) -> anyhow::Result<ResponseStream> {
        self.asked.fetch_add(1, Ordering::SeqCst);
        if self.failing.load(Ordering::SeqCst)
            && format!("{:?}", request.messages).contains(&self.victim)
        {
            anyhow::bail!("{}", self.failure);
        }
        Ok(Box::pin(futures::stream::iter(vec![
            Ok(ResponseEvent::TextDelta("an answer".into())),
            Ok(ResponseEvent::Done),
        ])))
    }
}

fn solve(
    ctx: &Context,
    set: &splinter_store::tasks::TaskSetId,
    reference: &ModelRef,
    resume: Option<&splinter_store::experiences::SetId>,
) -> Solved {
    solve_set(ctx, set, reference, None, &CancelToken::new(), resume).unwrap()
}

#[test]
fn a_task_the_solver_could_not_be_asked_is_reported_and_a_resume_asks_only_it() {
    let scratch = Scratch::new("solve-resume");
    let ctx = Context::new(config(&scratch), true).unwrap();
    let reference: ModelRef = "remote:test/model".parse().unwrap();
    let flaky = Arc::new(Flaky {
        victim: question("alpha", 2),
        failure: "connection closed before message completed",
        failing: AtomicBool::new(true),
        asked: AtomicUsize::new(0),
    });
    ctx.add_model(reference.clone(), Model::new(flaky.clone(), "flaky/model"));
    let members = (0..5)
        .map(|i| TaskEntry {
            task: ctx.tasks().put(&fact("alpha", i)).unwrap(),
            generator: None,
            prompt: None,
            variant_of: None,
            subject: None,
        })
        .collect();
    let set = ctx
        .tasks()
        .put_set(&TaskSet {
            name: "resume".into(),
            members,
        })
        .unwrap();

    let first = solve(&ctx, &set, &reference, None);
    assert_eq!(first.solved, 4, "the tasks that completed are kept");
    let [lost] = &first.skipped[..] else {
        panic!("one task could not be asked: {:?}", first.skipped)
    };
    assert!(lost.reason.starts_with("infrastructure error"), "{lost:?}");
    assert_eq!(
        first.conclusions.values().sum::<usize>(),
        4,
        "no experience of the failed attempt"
    );

    flaky.failing.store(false, Ordering::SeqCst);
    let before = flaky.asked.load(Ordering::SeqCst);
    let second = solve(&ctx, &set, &reference, Some(&first.experience_set));
    assert_eq!(
        flaky.asked.load(Ordering::SeqCst) - before,
        1,
        "only the missing task is asked again"
    );
    assert!(second.skipped.is_empty(), "{:?}", second.skipped);
    let experiences = ctx
        .experiences()
        .get_set(&second.experience_set)
        .unwrap()
        .members;
    assert_eq!(experiences.len(), 5, "the four kept and the one resumed");
}

#[test]
fn an_episode_that_outgrew_the_served_context_is_a_skip_a_resume_would_not_cure() {
    let scratch = Scratch::new("solve-outgrew");
    let ctx = Context::new(config(&scratch), true).unwrap();
    let reference: ModelRef = "remote:test/model".parse().unwrap();
    let flaky = Arc::new(Flaky {
        victim: question("alpha", 1),
        failure: "server error (context_length_exceeded): the request needs 67140 tokens",
        failing: AtomicBool::new(true),
        asked: AtomicUsize::new(0),
    });
    ctx.add_model(reference.clone(), Model::new(flaky, "flaky/model"));
    let members = (0..3)
        .map(|i| TaskEntry {
            task: ctx.tasks().put(&fact("alpha", i)).unwrap(),
            generator: None,
            prompt: None,
            variant_of: None,
            subject: None,
        })
        .collect();
    let set = ctx
        .tasks()
        .put_set(&TaskSet {
            name: "outgrew".into(),
            members,
        })
        .unwrap();
    let solved = solve(&ctx, &set, &reference, None);
    assert_eq!(solved.solved, 2);
    let [lost] = &solved.skipped[..] else {
        panic!("one task outgrew the context: {:?}", solved.skipped)
    };
    assert!(
        lost.reason
            .starts_with("episode outgrew the served context"),
        "{lost:?}"
    );
}
