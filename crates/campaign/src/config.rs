// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Splinter's configuration, resolved from the environment in one place.
//!
//! Everything below this module takes its settings as values; nothing else
//! in Splinter reads the environment. A test or a second campaign builds a
//! [`Config`] directly and never touches process state.

use std::path::{Path, PathBuf};

use splinter_store::StateRoot;

/// The model Splinter trains by default, as brain's model store names it.
pub const DEFAULT_POLICY_MODEL: &str = "Qwen/Qwen3-0.6B";

/// The settings every command reads.
#[derive(Clone, Debug)]
pub struct Config {
    /// Where runs, datasets, attempts and the promotion pointer live.
    pub state_root: StateRoot,
    /// The local weights served when a command names none.
    pub default_local_weights: PathBuf,
    /// Key for models reached through OpenRouter.
    pub openrouter_api_key: Option<String>,
    /// Key for any other OpenAI-compatible endpoint (a served brain, say).
    pub brain_api_key: Option<String>,
}

impl Config {
    /// The configuration this process runs with:
    ///
    /// * `SPLINTER_STATE`, else `$HOME/.sven/splinter`;
    /// * `BRAIN_QWEN_WEIGHTS`, else [`DEFAULT_POLICY_MODEL`] in brain's model
    ///   store under `$HOME/.local/share/brain/models`;
    /// * `AGENT_OPENROUTER_KEY` and `BRAIN_API_KEY`.
    #[must_use]
    pub fn from_env() -> Self {
        let var = |name: &str| std::env::var(name).ok().filter(|v| !v.is_empty());
        let home = PathBuf::from(var("HOME").unwrap_or_else(|| ".".into()));
        Self {
            state_root: var("SPLINTER_STATE")
                .map(StateRoot::new)
                .unwrap_or_else(|| StateRoot::under_home(&home)),
            default_local_weights: var("BRAIN_QWEN_WEIGHTS")
                .map(PathBuf::from)
                .unwrap_or_else(|| default_model_store(&home).join(DEFAULT_POLICY_MODEL)),
            openrouter_api_key: var("AGENT_OPENROUTER_KEY"),
            brain_api_key: var("BRAIN_API_KEY"),
        }
    }
}

/// brain's model store for a user whose home directory is `home`.
fn default_model_store(home: &Path) -> PathBuf {
    home.join(".local/share/brain/models")
}

/// Widens sven's stream watchdog for a local model.
///
/// The watchdog declares a connection dead after 300 s of silence between
/// chunks - a guard for a REMOTE wire going stale. A local provider is silent
/// for a different reason: its prefill takes tens of seconds before the
/// first chunk, and no chunk in between is honest to invent. The attempt's
/// own wall-clock limit is the bound that matters there, so the watchdog is
/// raised to it. sven reads this setting only from the environment, so it is
/// set here, before any engine turn runs.
pub fn allow_slow_local_prefill(timeout_secs: u64) {
    std::env::set_var("SVEN_STREAM_CHUNK_TIMEOUT_SECS", timeout_secs.to_string());
}
