// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Splinter's configuration, resolved from the environment in one place.
//!
//! Everything below this module takes its settings as values; nothing else
//! in Splinter reads the environment. A test or a second campaign builds a
//! [`Config`] directly and never touches process state.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use splinter_knowledge::capture::DEFAULT_ENV_ALLOWLIST;
use splinter_store::StateRoot;

/// The model Splinter trains by default, as brain's model store names it.
pub const DEFAULT_POLICY_MODEL: &str = "Qwen/Qwen3-0.6B";

/// The settings every command reads.
#[derive(Clone, Debug)]
pub struct Config {
    /// Where every store, run and candidate lives.
    pub state_root: StateRoot,
    /// brain's model store: where `local:<org>/<model>` names a checkpoint.
    pub model_store: PathBuf,
    /// The checkpoint `policy:default` serves.
    pub policy_base: PathBuf,
    /// Key for models reached through OpenRouter.
    pub openrouter_api_key: Option<String>,
    /// Key for any other OpenAI-compatible endpoint (a served brain, say).
    pub brain_api_key: Option<String>,
    /// Whether models reached over the network may be used without
    /// `--allow-remote`: the configuration's one network opt-in.
    pub allow_remote: bool,
    /// What a captured command sees of this process's environment: the
    /// capture allowlist's variables, as they are set.
    pub command_env: BTreeMap<String, String>,
    /// The directory a captured command runs in: where Splinter was
    /// started.
    pub working_dir: PathBuf,
}

impl Config {
    /// The configuration this process runs with:
    ///
    /// * `SPLINTER_STATE`, else `$HOME/.sven/splinter`;
    /// * `BRAIN_MODELS_DIR`, else `$HOME/.local/share/brain/models`, as
    ///   brain's model store;
    /// * `BRAIN_QWEN_WEIGHTS`, else [`DEFAULT_POLICY_MODEL`] in that store,
    ///   as the policy's base;
    /// * `AGENT_OPENROUTER_KEY` and `BRAIN_API_KEY`;
    /// * `SPLINTER_ALLOW_REMOTE` set to `1` or `true` as the network
    ///   opt-in;
    /// * the capture allowlist's variables, and the current directory, for
    ///   command sources.
    #[must_use]
    pub fn from_env() -> Self {
        let var = |name: &str| std::env::var(name).ok().filter(|v| !v.is_empty());
        let home = PathBuf::from(var("HOME").unwrap_or_else(|| ".".into()));
        let model_store = var("BRAIN_MODELS_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| default_model_store(&home));
        Self {
            state_root: var("SPLINTER_STATE")
                .map(StateRoot::new)
                .unwrap_or_else(|| StateRoot::under_home(&home)),
            policy_base: var("BRAIN_QWEN_WEIGHTS")
                .map(PathBuf::from)
                .unwrap_or_else(|| model_store.join(DEFAULT_POLICY_MODEL)),
            model_store,
            openrouter_api_key: var("AGENT_OPENROUTER_KEY"),
            brain_api_key: var("BRAIN_API_KEY"),
            allow_remote: var("SPLINTER_ALLOW_REMOTE")
                .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true")),
            command_env: DEFAULT_ENV_ALLOWLIST
                .iter()
                .filter_map(|name| var(name).map(|value| (name.to_string(), value)))
                .collect(),
            working_dir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        }
    }
}

/// brain's model store for a user whose home directory is `home`.
fn default_model_store(home: &Path) -> PathBuf {
    home.join(".local/share/brain/models")
}

/// Widens sven's stream watchdog for a local model.
///
/// The watchdog declares a connection dead after sven's default idle limit
/// between chunks - a guard for a REMOTE wire going stale. A local provider
/// is silent for a different reason: its prefill takes tens of seconds
/// before the first chunk, and no chunk in between is honest to invent. The
/// run's own deadline is the bound that matters there, so the watchdog is
/// raised to it. sven reads this setting only from the environment, so it
/// is set here, before any engine turn runs.
pub fn allow_slow_local_prefill(timeout_secs: u64) {
    std::env::set_var("SVEN_STREAM_CHUNK_TIMEOUT_SECS", timeout_secs.to_string());
}
