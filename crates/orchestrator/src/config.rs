// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Splinter's configuration: settings as values, with defaults that need no
//! setting.
//!
//! [`Config::from_env`] reads only what the environment says about the
//! machine and the user: the home directory, `PATH` (where `brain` is), and
//! the variables a captured command or a docker client is handed through.
//! No setting is read from it: every choice is a flag of the command line,
//! which fills in this value. Nothing else in Splinter reads the
//! environment; a test or a second campaign builds a [`Config`] directly and
//! never touches process state.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use splinter_core::build::BuildInfo;
use splinter_core::prompt::CodingPrompt;
use splinter_knowledge::capture::DEFAULT_ENV_ALLOWLIST;
use splinter_store::StateRoot;

/// The model Splinter trains by default, as brain's model store names it:
/// the size whose base serves and whose adapters train, at agent-trajectory
/// lengths, on one 24 GB GPU.
pub const DEFAULT_POLICY_MODEL: &str = "Qwen/Qwen3-0.6B";

/// How many requests to a model reached over an API are in flight at once,
/// when the configuration names no width.
pub const DEFAULT_REMOTE_CONCURRENCY: usize = 4;

/// How long one task's solve may take when the configuration names no bound.
pub const DEFAULT_SOLVE_DEADLINE: Duration = Duration::from_secs(300);

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
    /// `None` takes the strongest chat model in the model store that fits a
    /// card (see `roles::discovered_assistant`).
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
    /// How long one task's solve may take on the wall clock. A bound for a
    /// stuck episode, not a measure of the solver: under a loaded server it
    /// must leave room for the queue, or a task is failed for the load.
    pub solve_deadline: Duration,
    /// The coding-agent system prompt a solver of a repository works under
    /// (see [`CodingPrompt`]); the registered protocol's prompt unless a run
    /// asks for another, and recorded in each attempt's solve profile.
    pub coding_prompt: CodingPrompt,
    /// Let local models reason before they answer, and a model served by
    /// brain on this machine too. Off by default: a model whose template
    /// opens a reasoning block is asked with it closed, so agent work,
    /// probes and exams get the answer at once. Models reached over an API
    /// are not affected.
    pub thinking: bool,
    /// The sampling temperature asked of a model reached through sven's
    /// providers, a model served on this machine included; `None` leaves the
    /// provider's own default. Greedy decoding (temperature 0) can fall into
    /// repeating one action for as long as the run lasts, so a served model
    /// that works as an agent wants a temperature above zero.
    pub served_temperature: Option<f32>,
    /// Nucleus sampling (`top_p`) asked of a served model; `None` leaves the
    /// server's.
    pub served_top_p: Option<f32>,
    /// `top_k` asked of a served model; `None` leaves the server's.
    pub served_top_k: Option<u32>,
    /// The seed every request of a served model carries, so a sampled
    /// attempt is reproducible and its seed is part of its record; `None`
    /// leaves the server to draw one.
    pub served_seed: Option<u64>,
    /// The most tokens a served model may think in one response, when
    /// [`Config::thinking`] lets it; `None` leaves the thought unbounded.
    pub served_thinking_budget: Option<u32>,
    /// The fewest controls a judge is measured on before its verdicts count:
    /// a judge measured on fewer is not trusted, and one measured on more is
    /// known better. A small corpus gives few controls (a task has one only
    /// where another family of sources lends a wrong answer); a user who
    /// knows that may accept a thinner measurement.
    pub min_calibration_controls: usize,
    /// Where a workspace's commands run and what they are given.
    pub workspace: WorkspaceSettings,
    /// What this process was built from and started beside; handed in by the
    /// command line, which knows its own build. Recorded in every run and
    /// release.
    pub build: BuildInfo,
}

/// Where the commands of a workspace (a project's tests, a solver's shell)
/// run, and what they are given. A container is used when both a docker
/// client and an image are named; otherwise the commands run directly on the
/// host, bounded but not isolated, and every environment record says so.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkspaceSettings {
    /// The `docker` client executable.
    pub docker: Option<PathBuf>,
    /// The image commands run in, pinned by digest (`name@sha256:...`).
    /// Checked when a workspace is first needed, so a bad value is refused by
    /// name where it is used.
    pub image: Option<String>,
    /// What the docker client needs to reach its daemon: the variables of
    /// [`DOCKER_CLIENT_ENV`] that are set.
    pub docker_env: BTreeMap<String, String>,
    /// Where trees are laid out and archives are made; `None` is a directory
    /// under the state root's scratch space.
    pub scratch_root: Option<PathBuf>,
    /// The cargo home commands that build Rust projects use, so they can
    /// build offline from a registry that was fetched beforehand. `None`
    /// leaves the commands without one.
    pub cargo_home: Option<PathBuf>,
    /// The most bytes kept of each output stream of a workspace command (a
    /// test run's report); past it the middle is elided and the head and
    /// tail are kept. The command line's `--test-output-cap`.
    pub output_cap_bytes: u64,
    /// Run workspace commands with the caller's whole filesystem where the
    /// kernel cannot confine them. Without it such a workspace is refused:
    /// a solver that can read the whole disk can read the answer.
    pub allow_unconfined: bool,
    /// Let workspace commands use the network. Without it they cannot
    /// connect anywhere, a model server on the loopback included: a solver
    /// that can reach one can reach whatever it serves. The environment
    /// record says when it was allowed.
    pub allow_network: bool,
    /// How many processes and threads a workspace command may start; `None`
    /// is no bound. Part of the environment's snapshot.
    pub max_processes: Option<u64>,
    /// Variables every workspace command is run with besides the ones the
    /// sandbox sets, the command line's `--workspace-env`. Part of the
    /// environment's snapshot, so a task generated with one names another
    /// environment than a task generated without it.
    pub env: BTreeMap<String, String>,
    /// The address space, in bytes, one process of a workspace command may
    /// map (`RLIMIT_AS`); `None` keeps the default of 32 GiB. Part of the
    /// environment's snapshot.
    pub memory_bytes: Option<u64>,
    /// The most memory, in bytes, the processes of one workspace command may
    /// hold together; `None` keeps the default of 24 GiB (or the figure of
    /// `memory_bytes` when that is named) and `Some(0)` names no bound, the
    /// environment of the tasks stored before the bound existed. Part of the
    /// environment's snapshot when set.
    pub tree_memory_bytes: Option<u64>,
    /// Directories searched before the system's for the interpreters code
    /// runs in: where a pinned build of a language (Lua 5.1, built from its
    /// verified source) is put first. Empty is the system's search path.
    pub runtime_dirs: Vec<PathBuf>,
}

impl Default for WorkspaceSettings {
    fn default() -> Self {
        Self {
            docker: None,
            image: None,
            docker_env: BTreeMap::new(),
            scratch_root: None,
            cargo_home: None,
            output_cap_bytes: WORKSPACE_OUTPUT_CAP_BYTES,
            allow_unconfined: false,
            allow_network: false,
            max_processes: None,
            env: BTreeMap::new(),
            memory_bytes: None,
            tree_memory_bytes: None,
            runtime_dirs: Vec::new(),
        }
    }
}

/// The most bytes kept of each output stream of one workspace command, when
/// the command line names none: the verbose report of a project's whole test
/// suite runs to hundreds of KiB (115 KiB for sqlglot's 1301 tests). A run
/// holds at most two streams of this size in memory (2 MiB), against the
/// gigabytes its tree and test process already use, so concurrent runs are
/// not made unsafe by it.
pub const WORKSPACE_OUTPUT_CAP_BYTES: u64 = 1024 * 1024;

/// The variables a docker client may need to reach its daemon.
pub const DOCKER_CLIENT_ENV: &[&str] = &[
    "PATH",
    "HOME",
    "DOCKER_HOST",
    "DOCKER_CONFIG",
    "DOCKER_CONTEXT",
    "DOCKER_TLS_VERIFY",
    "DOCKER_CERT_PATH",
    "XDG_RUNTIME_DIR",
];

/// [`Config::min_calibration_controls`] when nothing names one.
pub const DEFAULT_MIN_CALIBRATION_CONTROLS: usize = 16;

impl Config {
    /// The configuration this process runs with before its command line
    /// names anything: every setting at its default, and the facts of this
    /// machine and user it needs:
    ///
    /// * `$HOME`, under which the state root (`~/.splinter`) lives;
    /// * brain's model store, as brain resolves it, with the policy's base
    ///   [`DEFAULT_POLICY_MODEL`] in it;
    /// * the first `brain` executable on `PATH`, the binary a release must
    ///   serve on;
    /// * the capture allowlist's variables, and the current directory, for
    ///   command sources;
    /// * the [`DOCKER_CLIENT_ENV`] variables a docker client needs.
    #[must_use]
    pub fn from_env() -> Self {
        let var = |name: &str| std::env::var(name).ok().filter(|v| !v.is_empty());
        let home = PathBuf::from(var("HOME").unwrap_or_else(|| ".".into()));
        let model_store = splinter_model::store::default_dir()
            .unwrap_or_else(|| home.join(".local/share/brain/models"));
        Self {
            state_root: StateRoot::under_home(&home),
            policy_base: model_store.join(DEFAULT_POLICY_MODEL),
            policy_context_tokens: None,
            model_store,
            allow_remote: false,
            command_env: DEFAULT_ENV_ALLOWLIST
                .iter()
                .filter_map(|name| var(name).map(|value| (name.to_string(), value)))
                .collect(),
            working_dir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            brain_binary: var("PATH").and_then(|path| find_executable(&path, "brain")),
            front_door_model: None,
            assistant_model: None,
            judge_model: None,
            bf16_base: false,
            default_budget: None,
            thinking: false,
            served_temperature: None,
            served_top_p: None,
            served_top_k: None,
            served_seed: None,
            served_thinking_budget: None,
            min_calibration_controls: DEFAULT_MIN_CALIBRATION_CONTROLS,
            remote_concurrency: DEFAULT_REMOTE_CONCURRENCY,
            solve_deadline: DEFAULT_SOLVE_DEADLINE,
            coding_prompt: CodingPrompt::default(),
            workspace: WorkspaceSettings {
                docker_env: DOCKER_CLIENT_ENV
                    .iter()
                    .filter_map(|name| var(name).map(|value| (name.to_string(), value)))
                    .collect(),
                ..WorkspaceSettings::default()
            },
            build: BuildInfo::default(),
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
