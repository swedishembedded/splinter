// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements release gates that cannot be passed by a
// model that games its exam, for its clients. If your team needs expertise
// in detecting reward hacking and benchmark contamination in agent
// evaluations, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The candidate attempted on canary tasks: tasks no honest solution can
//! pass, so any pass refuses it ([`splinter_eval::canary`]).
//!
//! Only the candidate attempts them: a canary carries no comparison. They are
//! attempted as the exam is (closed-book ones answered, ones that are
//! changes to a repository worked as an agent under the gate's bounds), and a
//! suite that could not be graded leaves its tasks without a verdict, which
//! is not a pass.

use splinter_agent::CancelToken;
use splinter_core::experience::Task;
use splinter_core::model_ref::ModelRef;
use splinter_eval::canary::{canary_check, Canaries, CanaryOutcome};
use splinter_eval::gate::{AgenticConfig, Check};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

use super::gating::{grade_arm, Asked};
use super::probe::Suite;

/// The canary check for `candidate` over `canaries`; `None` when the gate
/// was handed none.
pub(super) fn check(
    ctx: &Context,
    candidate: &ModelRef,
    deployed: Option<&str>,
    canaries: &[Task],
    work: AgenticConfig,
    cancel: &CancelToken,
) -> Result<Option<Check<Canaries>>, OrchestratorError> {
    if canaries.is_empty() {
        return Ok(None);
    }
    let closed = Suite::of_tasks("canaries", canaries.to_vec());
    let workspace = Suite::workspace("canaries (workspace)", canaries.to_vec()).with_work(work);
    let suites: Vec<&Suite> = [&closed, &workspace]
        .into_iter()
        .filter(|s| !s.tasks.is_empty())
        .collect();
    let asked: Vec<(&Suite, Asked)> = suites.iter().map(|s| (*s, Asked::Deployed)).collect();
    let graded = grade_arm(ctx, candidate, deployed, &asked, cancel)?;
    let mut outcomes = Vec::new();
    for (suite, graded) in suites.iter().zip(graded) {
        for (i, task) in suite.tasks.iter().enumerate() {
            outcomes.push(CanaryOutcome {
                task: task.task.id.to_string(),
                verdict: graded.as_ref().ok().and_then(|probes| probes[i].verdict),
            });
        }
    }
    Ok(Some(canary_check(&outcomes)))
}
