// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agent solvers whose every run is replayable
// evidence, for its clients. If your team needs expertise in agent
// environments or learning from agent experience, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Replay: an experience's code calls run again in the environment it
//! records, each result compared with the one its trajectory observed.
//!
//! An experience is evidence only if what it observed can be observed
//! again. Every [`RUN_CODE`] call in the trajectory is re-run, in order,
//! in an environment whose record is exactly the experience's (the
//! snapshot pins runtime, sandbox and limits), and its [`CodeResult`] is
//! compared field by field with the one recorded as that call's
//! observation. A call is [`CallReplay::Reproduced`] when every field is
//! equal, [`CallReplay::Diverged`] naming the fields that are not
//! otherwise - a program that reads the clock or a random source, a
//! runtime that changed under an unchanged record, or a trajectory edited
//! after it was recorded. A call with nothing to compare (its arguments
//! were not a code call, so the tool refused it, or its observation is not
//! a code result) is [`CallReplay::NotReplayable`], with why.

use serde::Serialize;
use splinter_sandbox::{CodeCall, CodeResult, ResolvedEnvironment, SandboxError};
use splinter_store::experience::{Environment, Experience};
use sven_sdk::atif::{MessageBody, ObservationEntry};

use crate::run_code::RUN_CODE;

/// How one recorded call replayed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum CallReplay {
    /// The same result as recorded.
    Reproduced,
    /// A different result: the fields that differ.
    Diverged {
        /// The [`CodeResult`] fields that differ, by name.
        fields: Vec<&'static str>,
    },
    /// Nothing to compare, and why.
    NotReplayable {
        /// Why.
        reason: String,
    },
}

/// One recorded call, replayed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ReplayedCall {
    /// The call's id in the trajectory.
    pub call_id: String,
    /// How it replayed.
    #[serde(flatten)]
    pub replay: CallReplay,
}

/// An experience's calls, replayed in order.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Replay {
    /// Every [`RUN_CODE`] call the trajectory records.
    pub calls: Vec<ReplayedCall>,
}

impl Replay {
    /// Whether no call diverged.
    #[must_use]
    pub fn reproduced(&self) -> bool {
        !self
            .calls
            .iter()
            .any(|c| matches!(c.replay, CallReplay::Diverged { .. }))
    }
}

/// Why an experience could not be replayed.
#[derive(Debug, thiserror::Error)]
pub enum ReplayError {
    /// The environment offered is not the one the experience records.
    #[error("the experience was recorded in {recorded:?}, not {offered:?}")]
    EnvironmentMismatch {
        /// The experience's environment kind and snapshot.
        recorded: Box<Environment>,
        /// The offered one's.
        offered: Box<Environment>,
    },
    /// The sandbox could not run a call.
    #[error(transparent)]
    Sandbox(#[from] SandboxError),
}

/// Replays `experience`'s code calls in `environment`; see the module
/// documentation. A closed-book experience has no calls to replay.
pub fn replay(
    experience: &Experience,
    environment: &ResolvedEnvironment,
) -> Result<Replay, ReplayError> {
    let offered = environment.record()?;
    if offered != experience.environment {
        return Err(ReplayError::EnvironmentMismatch {
            recorded: Box::new(experience.environment.clone()),
            offered: Box::new(offered),
        });
    }
    let steps = &experience.trajectory.steps;
    let observations: Vec<&ObservationEntry> = steps
        .iter()
        .filter_map(|s| s.observation.as_ref())
        .flat_map(|o| o.results.iter())
        .collect();
    let mut calls = Vec::new();
    for call in steps
        .iter()
        .flat_map(|s| s.tool_calls.iter().flatten())
        .filter(|c| c.function_name == RUN_CODE)
    {
        let observed = observations
            .iter()
            .find(|o| o.source_call_id.as_deref() == Some(call.tool_call_id.as_str()));
        let replay = match (environment, recorded(observed)) {
            (_, Err(reason)) => CallReplay::NotReplayable { reason },
            (ResolvedEnvironment::ClosedBook, Ok(_)) => CallReplay::NotReplayable {
                reason: "a closed-book environment runs no code".into(),
            },
            (ResolvedEnvironment::Runtime(runtime), Ok(recorded)) => {
                match serde_json::from_value::<CodeCall>(call.arguments.clone()) {
                    Err(e) => CallReplay::NotReplayable {
                        reason: format!("its arguments are not a code call: {e}"),
                    },
                    Ok(code_call) => compare(&recorded, &runtime.run(&code_call)?),
                }
            }
        };
        calls.push(ReplayedCall {
            call_id: call.tool_call_id.clone(),
            replay,
        });
    }
    Ok(Replay { calls })
}

/// The code result `observed` records.
fn recorded(observed: Option<&&ObservationEntry>) -> Result<CodeResult, String> {
    let observed = observed.ok_or("the trajectory records no result for it")?;
    let text = observed
        .content
        .as_ref()
        .and_then(MessageBody::as_text)
        .ok_or("its result is not text")?;
    serde_json::from_str(text).map_err(|_| format!("its result is not a code result: {text}"))
}

fn compare(recorded: &CodeResult, replayed: &CodeResult) -> CallReplay {
    let mut fields = Vec::new();
    let mut differs = |name: &'static str, equal: bool| {
        if !equal {
            fields.push(name);
        }
    };
    differs("stdout", recorded.stdout == replayed.stdout);
    differs("stderr", recorded.stderr == replayed.stderr);
    differs("exit_code", recorded.exit_code == replayed.exit_code);
    differs("signal", recorded.signal == replayed.signal);
    differs("timed_out", recorded.timed_out == replayed.timed_out);
    differs(
        "stdout_truncated",
        recorded.stdout_truncated == replayed.stdout_truncated,
    );
    differs(
        "stderr_truncated",
        recorded.stderr_truncated == replayed.stderr_truncated,
    );
    if fields.is_empty() {
        CallReplay::Reproduced
    } else {
        CallReplay::Diverged { fields }
    }
}
