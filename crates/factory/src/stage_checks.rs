// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements experiment recipes that are refused before
// they spend anything when a stage cannot do what it says, for its clients.
// If your team needs expertise in machine-learning experiment orchestration,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Why a stage's parameters are refused, as rules over the parameters
//! alone: what [`crate::resolve`] reports beside what reading the
//! repository's files finds.

use crate::eval_plan::EvalPlan;
use crate::ops::AttemptParams;

/// Why an evaluation under `plan` (at `path`) has no one source of tasks:
/// its plan's collection, the stage's pinned tasks, or its examiner's.
pub(crate) fn task_source_problem(plan: &EvalPlan, path: &str, given: bool) -> Option<String> {
    match (&plan.collection, &plan.authority, given) {
        (Some(_), _, true) => Some(format!(
            "{path} names its collection, and the stage gives tasks too"
        )),
        (None, None, false) => Some(format!(
            "{path} names no collection: bind the stage's tasks"
        )),
        (None, Some(_), true) => Some(format!("{path} is protected: its examiner holds its tasks")),
        _ => None,
    }
}

/// Why an attempt stage cannot continue failed attempts: only sven
/// continues a conversation, unhinted (the instruction is the conversation's
/// first turn) and unplayed, as its own agent and not a harness.
pub(crate) fn fork_refusals(params: &AttemptParams) -> Vec<&'static str> {
    let mut out = Vec::new();
    if params.agent != "sven" {
        out.push("continues failed attempts, which only the sven agent does");
    }
    if params.hint.is_some() {
        out.push("continues failed attempts, whose instruction is already given: it takes no hint");
    }
    if params.harness.is_some() {
        out.push("continues failed attempts as the agent's own conversation, never a harness's");
    }
    if params.interactive {
        out.push("continues failed attempts, which is never interactive");
    }
    out
}

/// Why an attempt stage cannot run its harness: only sven runs one, told
/// its model; a harness is never played and carries its own instructions.
pub(crate) fn harness_refusals(params: &AttemptParams) -> Vec<&'static str> {
    let mut out = Vec::new();
    if params.agent != "sven" {
        out.push("runs a harness, which only the sven agent runs");
    }
    if params.model.is_none() {
        out.push("runs a harness, which is told its model: name a model role");
    }
    if params.interactive {
        out.push("runs a harness, which is never interactive");
    }
    if params.system_prompt.is_some() {
        out.push("runs a harness, which carries its own instructions: it takes no system_prompt");
    }
    out
}
