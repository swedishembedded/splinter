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

/// The model Splinter trains by default, as brain's model store names it:
/// the size whose base serves and whose adapters train, at agent-trajectory
/// lengths, on one 24 GB GPU.
pub const DEFAULT_POLICY_MODEL: &str = "Qwen/Qwen3-0.6B";

/// How many requests to a model reached over an API are in flight at once,
/// when the configuration names no width.
pub const DEFAULT_REMOTE_CONCURRENCY: usize = 4;

/// The settings every command reads.
#[derive(Clone, Debug)]
pub struct Config {
    /// Where every store, run and candidate lives.
    pub state_root: StateRoot,
    /// brain's model store: where `local:<org>/<model>` names a checkpoint.
    pub model_store: PathBuf,
    /// The checkpoint `policy:default` serves.
    pub policy_base: PathBuf,
    /// The policy's context limit in tokens; `None` is the largest the
    /// checkpoint supports. Set by `--policy-context-tokens` when that does
    /// not fit the card.
    pub policy_context_tokens: Option<u32>,
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
    /// The `brain` executable a release is checked to serve on; `None`
    /// when there is none, and the release gate's serve check is then
    /// unmeasured, which blocks the release.
    pub brain_binary: Option<PathBuf>,
    /// The model reference that reads a sentence at the front door, when it
    /// is not the policy: a small policy reads a sentence worse than a
    /// larger model would. Parsed where it is used, so a bad value is
    /// refused by name when a sentence is read.
    pub front_door_model: Option<String>,
    /// The model reference of a stronger model than the policy that plans a
    /// learning run, writes its tasks and teaches what the policy cannot
    /// answer, when the command names none of them. Parsed where it is used.
    pub assistant_model: Option<String>,
    /// A model reference that judges answers in place of the assistant: the
    /// judge is another model than the one that wrote what it grades.
    pub judge_model: Option<String>,
    /// Hold a base the policy trains at bf16: for a base too large for the
    /// card at fp32, which is a fact about the machine, not the run.
    pub bf16_base: bool,
    /// How long a learning run may take when its command names no budget, as
    /// `--budget` writes it (`6h`). Parsed where it is used, so a bad value
    /// is refused by name when a run starts. A run on a corpus too large to
    /// read has no end without one.
    pub default_budget: Option<String>,
    /// How many requests to a model reached over an API may be in flight at
    /// once. A model run on this machine's device is asked one at a time.
    pub remote_concurrency: usize,
    /// Let local models reason before they answer, and a model served by
    /// brain on this machine too. Off by default: a model whose template
    /// opens a reasoning block is asked with it closed, so agent work,
    /// probes and exams get the answer at once. Models reached over an API
    /// are not affected.
    pub thinking: bool,
    /// The fewest controls a judge is measured on before its verdicts count:
    /// a judge measured on fewer is not trusted, and one measured on more is
    /// known better. A small corpus gives few controls (a task has one only
    /// where another family of sources lends a wrong answer); a user who
    /// knows that may accept a thinner measurement.
    pub min_calibration_controls: usize,
}

/// [`Config::min_calibration_controls`] when nothing names one.
pub const DEFAULT_MIN_CALIBRATION_CONTROLS: usize = 16;

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
    ///   command sources;
    /// * `SPLINTER_BRAIN_BIN`, else the first `brain` executable on `PATH`,
    ///   as the binary a release must serve on;
    /// * `SPLINTER_FRONT_DOOR_MODEL`, a model reference that reads
    ///   sentences instead of the policy;
    /// * `SPLINTER_ASSISTANT_MODEL`, a model reference that plans, writes
    ///   tasks and teaches in place of the policy;
    /// * `SPLINTER_JUDGE_MODEL`, a model reference that judges in place of the
    ///   assistant;
    /// * `SPLINTER_BF16_BASE` set to `1` or `true` to train with the base
    ///   held at bf16;
    /// * `SPLINTER_MIN_CALIBRATION_CONTROLS`, the fewest controls a judge is
    ///   measured on before its verdicts count;
    /// * `SPLINTER_BUDGET`, how long a learning run may take when its command
    ///   names no `--budget`;
    /// * `SPLINTER_REMOTE_CONCURRENCY`, how many requests to a model reached
    ///   over an API may be in flight at once ([`DEFAULT_REMOTE_CONCURRENCY`]).
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
            policy_context_tokens: None,
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
            brain_binary: var("SPLINTER_BRAIN_BIN")
                .map(PathBuf::from)
                .or_else(|| var("PATH").and_then(|path| find_executable(&path, "brain"))),
            front_door_model: var("SPLINTER_FRONT_DOOR_MODEL"),
            assistant_model: var("SPLINTER_ASSISTANT_MODEL"),
            judge_model: var("SPLINTER_JUDGE_MODEL"),
            bf16_base: var("SPLINTER_BF16_BASE")
                .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true")),
            default_budget: var("SPLINTER_BUDGET"),
            thinking: var("SPLINTER_THINKING")
                .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true")),
            min_calibration_controls: var("SPLINTER_MIN_CALIBRATION_CONTROLS")
                .and_then(|v| v.parse().ok())
                .filter(|n| *n > 0)
                .unwrap_or(DEFAULT_MIN_CALIBRATION_CONTROLS),
            remote_concurrency: var("SPLINTER_REMOTE_CONCURRENCY")
                .and_then(|v| v.parse().ok())
                .filter(|n| *n > 0)
                .unwrap_or(DEFAULT_REMOTE_CONCURRENCY),
        }
    }
}

/// The first file named `name` in the directories of the `PATH`-style list
/// `path` that is executable.
fn find_executable(path: &str, name: &str) -> Option<PathBuf> {
    std::env::split_paths(path)
        .map(|dir| dir.join(name))
        .find(|candidate| is_executable(candidate))
}

fn is_executable(path: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        path.metadata()
            .is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
    }
    #[cfg(not(unix))]
    {
        path.is_file()
    }
}

/// brain's model store for a user whose home directory is `home`.
fn default_model_store(home: &Path) -> PathBuf {
    home.join(".local/share/brain/models")
}
