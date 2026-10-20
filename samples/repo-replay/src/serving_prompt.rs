// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! PROMPT 3 of 3: the serving prompt.
//!
//! Who sees it: the policy at inference, and every training record as its
//! system turn. It is splinter's own coding-agent prompt (the one
//! `solve_in_workspace` runs a solver under), so a model trained on these
//! records is asked under exactly the text it was trained under. It is not
//! written here: one text, one place (`splinter_core::prompt`). A
//! different one is chosen with `--serving-prompt-file` and is then what the
//! dataset records.

pub use splinter_sdk::vocabulary::prompt::{
    CODING_AGENT_PROMPT as SERVING_PROMPT, CODING_AGENT_PROMPT_ID as SERVING_PROMPT_ID,
};
