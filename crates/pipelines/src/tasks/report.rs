// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model-driven task generation that turns
// any source into verifiable training tasks, for its clients. If your team
// needs expertise in synthetic task generation or verifier design, you can
// procure our services by sending an email to info@swedishembedded.com.

//! What the tasks stage reports.

use std::collections::BTreeMap;

use serde::Serialize;
use splinter_core::rng::Seed;
use splinter_knowledge::tasks::{CallOutcome, CallRecord};
use splinter_store::tasks::TaskSetId;

use super::{reason_name, InventReport};

/// Admitted and rejected proposals of one kind.
#[derive(Clone, Debug, Default, Serialize)]
pub struct KindTally {
    /// Tasks admitted.
    pub admitted: usize,
    /// Proposals rejected.
    pub rejected: usize,
}

/// One proposal turned away, and exactly why.
#[derive(Clone, Debug, Serialize)]
pub struct RejectionNote {
    /// The kind it was proposed as.
    pub kind: String,
    /// The reason, as [`TasksGenerated::rejected`] counts it.
    pub reason: String,
    /// What failed: for a malformed reply, the reply itself.
    pub detail: String,
}

/// One request put to the generator model.
#[derive(Clone, Debug, Serialize)]
pub struct CallNote {
    /// The kind of task asked for.
    pub kind: String,
    /// The digest of what the model was shown.
    pub prompt: String,
    /// The size of what the model was shown, in bytes.
    pub prompt_bytes: usize,
    /// How long the request took, repairs included, in milliseconds.
    pub elapsed_ms: u64,
    /// Tasks in the model's reply, when it replied in the shape asked for.
    pub candidates: Option<usize>,
    /// Why there was no usable reply, as [`TasksGenerated::rejected`] names
    /// it.
    pub declined: Option<String>,
}

impl From<&CallRecord> for CallNote {
    fn from(call: &CallRecord) -> Self {
        let (candidates, declined) = match call.outcome {
            CallOutcome::Replied { candidates } => (Some(candidates), None),
            CallOutcome::Declined { rejection } => (None, Some(reason_name(&rejection))),
        };
        Self {
            kind: call.kind.clone(),
            prompt: call.prompt.to_string(),
            prompt_bytes: call.prompt_bytes,
            elapsed_ms: u64::try_from(call.elapsed.as_millis()).unwrap_or(u64::MAX),
            candidates,
            declined,
        }
    }
}

/// What the tasks stage reports.
#[derive(Clone, Debug, Serialize)]
pub struct TasksGenerated {
    /// The task set (`solve <task_set>`).
    pub task_set: TaskSetId,
    /// The seed the denoise tasks were built with; `None` when none were
    /// asked for. Generating again from the same parts with it yields the
    /// same denoise tasks.
    pub denoise_seed: Option<Seed>,
    /// Tasks in it.
    pub tasks: usize,
    /// Text parts generated from.
    pub parts: usize,
    /// Text parts across the sources, generated from or not.
    pub parts_total: usize,
    /// How long generation ran, in seconds.
    pub elapsed_secs: u64,
    /// How long generating from every part would take at the pace measured,
    /// in seconds: `None` when every part was covered.
    pub full_coverage_secs: Option<u64>,
    /// Admitted and rejected, by kind.
    pub per_kind: BTreeMap<String, KindTally>,
    /// Rejections, by reason.
    pub rejected: BTreeMap<String, usize>,
    /// Every rejection of a model's proposal, with what failed.
    pub rejections: Vec<RejectionNote>,
    /// Every request put to the generator model, in the order made.
    pub calls: Vec<CallNote>,
    /// Single sections generated from, beside the sources' parts.
    pub sections: usize,
    /// What the invent harness yielded, when it wrote the tasks.
    pub invent: Option<InventReport>,
    /// Why generation stopped before every part was covered, if it did.
    pub stopped: Option<String>,
    /// The kinds generation gave up on and why: asked for until their
    /// proposals were refused too often to be worth another request.
    pub dropped: BTreeMap<String, String>,
}
