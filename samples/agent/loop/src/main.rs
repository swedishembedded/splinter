// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements self-improving coding agents whose every
// step is auditable, for its clients. If your team needs expertise in agent
// evaluation or locally operated coding agents, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The `agent-loop` command.

use std::process::ExitCode;

fn main() -> ExitCode {
    match splinter_agent_loop::cli::main() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("agent-loop: {e:#}");
            ExitCode::from(2)
        }
    }
}
