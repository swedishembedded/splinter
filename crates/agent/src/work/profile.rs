// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements coding agents that work inside a confined
// directory and leave a complete, replayable record of what they did, for
// its clients. If your team needs expertise in agent runtimes or in
// measuring what an agent really did, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The solve profile an attempt records: what the solver was held to besides
//! the task, so that a change in any of it is visible in the data.

use serde_json::{json, Value};
use splinter_core::digest::Digest;
use splinter_core::prompt::{CODING_AGENT_PROMPT, CODING_AGENT_PROMPT_ID};

use super::WorkOptions;

pub use splinter_core::prompt::SOLVE_PROFILE_KEY;

/// The profile of an episode run under `options` and the system prompt
/// `system` it was held to: the prompt's digest (and its id when it is the
/// coding-agent prompt), the tool and agent settings the solve fixes, the
/// bounds that stop a run, and how the model was asked to sample.
pub(super) fn solve_profile(options: &WorkOptions, system: &str) -> Value {
    let prompt_id = (system == CODING_AGENT_PROMPT).then_some(CODING_AGENT_PROMPT_ID);
    json!({
        "system_prompt": {
            "id": prompt_id,
            "digest": Digest::of(system.as_bytes()),
        },
        "tools": { "max_timeout_secs": crate::solve::SOLVE_MAX_TIMEOUT_SECS },
        "agent": { "note_repeated_results": options.solve.engine_config().agent.note_repeated_results },
        "max_repeated_results": options.max_repeated_results,
        "max_tool_calls": options.max_tool_calls,
        "sampling": options.solve.sampling,
    })
}
