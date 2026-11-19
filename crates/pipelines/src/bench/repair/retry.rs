// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the loop from a failed agent attempt to a
// verified corrected one, for its clients. If your team needs expertise in
// turning agent failures into on-policy training data you can procure our
// services by sending an email to info@swedishembedded.com.

//! The retry: the student's own model samples again from the last good
//! step, under the practice run's sampling and a seed of the repair's own.
//!
//! A pass is the policy's own success from the state it reached, the most
//! on-policy repair there is: no other model wrote a step of it. The seed is
//! derived from the repair's key, so a rerun of the same repair asks for the
//! same sample, and another sample of the same failure for another.

use std::path::Path;
use std::time::Duration;

use serde_json::{json, Value};
use splinter_agent::fork_at::POLICY;
use splinter_agent::solve::{Model, SolveOptions};
use splinter_agent::sven::model::SamplingParams;
use splinter_agent::CancelToken;
use splinter_core::digest::Digest;
use splinter_core::experience::Provenance;
use splinter_orchestrator::context::Context;

use super::super::practice::BenchCheckout;
use super::fork::{finish, fork_and_grade, Continuation};
use super::{Attempted, Case, Intervention, InterventionKind, Notes};

/// The retry: the student's model continues after the last good step.
pub struct Retry<'a> {
    /// The examiner that lays out and grades the workspace.
    pub checkout: &'a BenchCheckout,
    /// The student's own model.
    pub student: Model,
    /// The time one continuation may take.
    pub deadline: Duration,
    /// Stops a continuation.
    pub cancel: CancelToken,
}

/// The seed a retry keyed `key` samples under.
#[must_use]
pub fn retry_seed(key: &str) -> u64 {
    let digest = Digest::of(format!("retry:{key}").as_bytes());
    u64::from_str_radix(&digest.hex()[..16], 16).unwrap_or_default()
}

/// The sampling the practice run at `run` recorded (`report.json`'s
/// `sampling`), with `seed` in place of the run's.
fn sampling(run: &Path, seed: u64) -> (SamplingParams, Value) {
    let recorded = std::fs::read(run.join("report.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .and_then(|report| report.get("sampling").cloned())
        .filter(Value::is_object);
    let number = |name: &str| recorded.as_ref().and_then(|r| r.get(name)?.as_f64());
    let params = SamplingParams {
        temperature: number("temperature"),
        top_p: number("top_p"),
        top_k: recorded
            .as_ref()
            .and_then(|r| r.get("top_k")?.as_u64())
            .and_then(|k| u32::try_from(k).ok()),
        seed: Some(seed),
    };
    // What was asked, and whether the rest was the run's or the server's.
    let record = json!({
        "seed": seed,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "top_k": params.top_k,
        "from": if recorded.is_some() { "practice_run" } else { "server_default" },
    });
    (params, record)
}

impl Intervention for Retry<'_> {
    fn name(&self) -> &'static str {
        InterventionKind::Retry.as_str()
    }

    fn actor(&self) -> &str {
        &self.student.identity
    }

    fn attempt(&self, ctx: &Context, case: &Case<'_>) -> Result<Attempted, String> {
        let (params, record) = sampling(case.run, retry_seed(case.key));
        let mut options = SolveOptions::new(self.deadline);
        options.cancel = Some(self.cancel.clone());
        let mut options = self.student.solving(options);
        options.request_sampling = Some(params);
        let forked = fork_and_grade(
            ctx,
            self.checkout,
            case,
            Continuation {
                model: &self.student,
                role: POLICY,
                options,
                briefing: None,
                turns: &[],
                after_turn: &mut |_, _| true,
            },
        )?;
        let notes = Notes {
            sampling: Some(record),
            ..Notes::default()
        };
        let provenance = Provenance::new(self.student.identity.clone(), ctx.clock());
        finish(case, self.name(), forked, notes, provenance)
    }
}
