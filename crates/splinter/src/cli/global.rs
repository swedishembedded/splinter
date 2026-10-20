// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The flags every command takes: where state lives, how output is printed,
//! and the knobs of the models it reaches.

use std::path::PathBuf;

use clap::{ArgAction, Args};
use splinter_sdk::config::{DEFAULT_REMOTE_CONCURRENCY, DEFAULT_SOLVE_DEADLINE};
use splinter_sdk::vocabulary::prompt::CodingPrompt;

/// Flags every command takes, meaning the same everywhere.
#[derive(Clone, Debug, Default, Args)]
pub struct Global {
    /// Which hardware models run on.
    #[command(flatten)]
    pub device: splinter_sdk::device::DeviceFlags,
    /// The state root (default: $SPLINTER_STATE, else ~/.splinter).
    #[arg(long, global = true, value_name = "DIR")]
    pub state: Option<PathBuf>,
    /// Print machine-readable JSON on stdout instead of text.
    #[arg(long, global = true)]
    pub json: bool,
    /// Report more as work goes (each stage's full summary).
    #[arg(short, long, global = true, action = ArgAction::Count)]
    pub verbose: u8,
    /// Permit models reached over the network (remote:...); without it, and
    /// without SPLINTER_ALLOW_REMOTE=1, every model runs locally.
    #[arg(long, global = true)]
    pub allow_remote: bool,
    /// Limit the policy model's context to this many tokens (default: the
    /// largest its checkpoint supports); for a policy too large for the
    /// card at that size. Another model is limited by its reference,
    /// `local:<checkpoint>@<tokens>`.
    #[arg(long, global = true, value_name = "TOKENS", value_parser = clap::value_parser!(u32).range(1..))]
    pub policy_context_tokens: Option<u32>,
    /// How many requests to a model reached over an API may be in flight at
    /// once (a model run on this machine's device is asked one at a time).
    #[arg(long, global = true, value_name = "N", value_parser = clap::value_parser!(u32).range(1..),
        help = format!("How many requests to a model reached over an API may be in flight at \
                        once [default: {DEFAULT_REMOTE_CONCURRENCY}]"))]
    pub remote_concurrency: Option<u32>,
    /// The wall-clock bound of one task's solve, a guard for a stuck episode.
    #[arg(long, global = true, value_name = "SECONDS", value_parser = clap::value_parser!(u64).range(1..),
        help = format!("How many seconds one task's solve may take on the wall clock \
                        [default: {}]", DEFAULT_SOLVE_DEADLINE.as_secs()))]
    pub solve_deadline: Option<u64>,
    /// The most bytes kept of each output stream of a project's test run
    /// (practice, solve, eval): past it the middle is elided and the head
    /// and tail kept, so a verbose report is still read from its summary.
    #[arg(long, global = true, value_name = "BYTES",
        value_parser = clap::value_parser!(u64).range(1024..),
        help = format!("Bytes kept of each output stream of a test run [default: {}]",
            splinter_sdk::config::WORKSPACE_OUTPUT_CAP_BYTES))]
    pub test_output_cap: Option<u64>,
    /// Let a project's commands (the solver's shell, the test runs) see the
    /// whole filesystem where the kernel cannot confine them to their tree.
    /// Without it such a run is refused; the environment record says when it
    /// was allowed.
    #[arg(long, global = true)]
    pub allow_unconfined_workspace: bool,
    /// Let a project's commands (the solver's shell, the test runs) use the
    /// network. Without it they cannot connect anywhere, a model server on
    /// the loopback included; the environment record says when it was
    /// allowed.
    #[arg(long, global = true)]
    pub allow_workspace_network: bool,
    /// Bound the processes and threads one project command may start. Part
    /// of the environment record, so tasks generated without it name a
    /// different environment.
    #[arg(long, global = true, value_name = "N",
        value_parser = clap::value_parser!(u64).range(1..))]
    pub workspace_max_processes: Option<u64>,
    /// A variable every project command runs with (for example
    /// `PYTHONDONTWRITEBYTECODE=1`, so a Python project leaves no bytecode
    /// caches in the tree the solver reads); repeat for more. Part of the
    /// environment record, so tasks generated with it name a different
    /// environment than tasks generated without it.
    #[arg(long, global = true, value_name = "NAME=VALUE", value_parser = workspace_variable)]
    pub workspace_env: Vec<(String, String)>,
    /// A directory searched before the system's for the interpreters code
    /// runs in (`lua` is Lua 5.1 and no other: `make deps/lua` builds the
    /// pinned one); repeat for more.
    #[arg(long, global = true, value_name = "DIR")]
    pub runtime_dir: Vec<PathBuf>,
    /// The sampling temperature, 0 to 2, asked of a model reached through
    /// sven's providers (the provider's own default when not given). The
    /// same flag sets the pass@k sampling of a verb that measures it; when
    /// given twice, the last wins.
    #[arg(long, global = true, value_name = "T", value_parser = served_temperature)]
    pub temperature: Option<f32>,
    /// Nucleus sampling (`top_p`, above 0 up to 1) asked of a model reached
    /// through sven's providers; the server's own default when not given.
    #[arg(long, global = true, value_name = "P", value_parser = served_top_p)]
    pub top_p: Option<f32>,
    /// Sampling among the K most likely tokens (`top_k`) asked of a model
    /// reached through sven's providers; the server's own default when not
    /// given.
    #[arg(long, global = true, value_name = "K",
        value_parser = clap::value_parser!(u32).range(1..))]
    pub top_k: Option<u32>,
    /// The seed every request of a model reached through sven's providers
    /// carries, so that a sampled attempt is reproducible; recorded with the
    /// attempt. The server draws its own when not given.
    #[arg(long, global = true, value_name = "SEED")]
    pub sampling_seed: Option<u64>,
    /// Let a model served on this machine reason before it answers (the
    /// request says so); without it the server is asked for no reasoning
    /// block. Recorded with the attempt.
    #[arg(long, global = true)]
    pub thinking: bool,
    /// With --thinking: the most tokens the model may think in one response;
    /// the server closes the thought there and the model goes on to its
    /// answer. Recorded with the attempt. Unbounded when not given.
    #[arg(long, global = true, value_name = "N", requires = "thinking",
        value_parser = clap::value_parser!(u32).range(1..))]
    pub thinking_budget: Option<u32>,
    /// The coding-agent system prompt a solver of a repository works under:
    /// `coding-agent-1` or `coding-agent-2`. The registered protocol's prompt
    /// when not given; the prompt's id and digest are recorded with each
    /// attempt.
    #[arg(long, global = true, value_name = "ID", value_parser = coding_prompt)]
    pub coding_prompt: Option<CodingPrompt>,
}

fn coding_prompt(id: &str) -> Result<CodingPrompt, String> {
    CodingPrompt::from_id(id).ok_or_else(|| {
        let known: Vec<&str> = CodingPrompt::ALL.iter().map(|p| p.id()).collect();
        format!("{id:?} is not a coding-agent prompt ({})", known.join(", "))
    })
}

fn served_top_p(text: &str) -> Result<f32, String> {
    match text.parse::<f32>() {
        Ok(p) if p > 0.0 && p <= 1.0 => Ok(p),
        _ => Err(format!("{text:?} is not a top-p above 0 and up to 1")),
    }
}

fn served_temperature(text: &str) -> Result<f32, String> {
    match text.parse::<f32>() {
        Ok(t) if (0.0..=2.0).contains(&t) => Ok(t),
        _ => Err(format!("{text:?} is not a temperature from 0 to 2")),
    }
}

/// `NAME=VALUE`, a variable for a project's commands: a name of letters,
/// digits and underscores that does not start with a digit, and not one the
/// sandbox sets for each command itself (`HOME`, `TMPDIR`), whose value would
/// be replaced.
fn workspace_variable(text: &str) -> Result<(String, String), String> {
    let (name, value) = text
        .split_once('=')
        .ok_or_else(|| format!("{text:?} is not NAME=VALUE"))?;
    let valid = !name.is_empty()
        && !name.starts_with(|c: char| c.is_ascii_digit())
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    if !valid {
        return Err(format!("{name:?} is not a variable name"));
    }
    if matches!(name, "HOME" | "TMPDIR") {
        return Err(format!("{name} is set by the sandbox for each command"));
    }
    Ok((name.to_string(), value.to_string()))
}
