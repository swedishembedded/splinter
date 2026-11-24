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

use crate::cluster::{ClusterAllocation, ClusterSettings, SCRATCH_VARS, SLURM_VARS};

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
    /// How the learning factory reaches model providers and executes
    /// attempts.
    pub factory: FactorySettings,
    /// How jobs are submitted and reconciled.
    pub jobs: JobSettings,
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
    /// `None` leaves the policy to play the assistant's roles.
    pub assistant_model: Option<String>,
    /// A model reference that judges answers in place of the assistant: the
    /// judge is another model than the one that wrote what it grades.
    pub judge_model: Option<String>,
    /// Hold a base the policy trains at bf16: for a base too large for the
    /// card at fp32, which is a fact about the machine, not the run.
    pub bf16_base: bool,
    /// How many requests to a model reached over an API may be in flight at
    /// once. A model run on this machine's device is asked one at a time.
    pub remote_concurrency: usize,
    /// The brain endpoints every local model reference (`local:...`,
    /// `policy:...`) is requested from; `None` starts a server of this
    /// process's own ([`crate::context::Runtime::brain`]).
    /// With endpoints, attempts are asked as many at once as they say
    /// ([`splinter_model::BrainEndpoints::attempt_concurrency`]).
    pub brain: Option<splinter_model::BrainEndpoints>,
    /// The compute a brain server this process starts may schedule
    /// (`brain --device`); brain's own choice when `None`. Without
    /// [`Config::brain`], local models are requested from that server.
    pub brain_device: Option<String>,
    /// How a brain server this process starts drives the hardware
    /// (`brain --backend`); brain's choice when `None`.
    pub brain_backend: Option<String>,
    /// How many requests a brain server this process starts decodes together
    /// (`--max-batch`), which is also how many attempts are asked of it at
    /// once; [`splinter_model::server::DEFAULT_SERVE_SLOTS`] when `None`.
    pub brain_max_batch: Option<u32>,
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
    /// The batch allocation this process runs inside, and how it uses its
    /// nodes: brain servers on each of them, and a job's ranks spread over
    /// them.
    pub cluster: ClusterSettings,
}

/// What starts a job's workers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LauncherKind {
    /// One step on the allocation's nodes when this process runs inside an
    /// allocation that spans nodes ([`LauncherKind::SlurmStep`]), else this
    /// machine ([`LauncherKind::Local`]).
    #[default]
    Auto,
    /// Every rank runs as a detached process on this machine.
    Local,
    /// The job is submitted to Slurm's batch system.
    Slurm,
    /// Every rank runs as a task of one step of the allocation this process
    /// runs inside, on its usable nodes.
    SlurmStep,
}

impl LauncherKind {
    /// The kind a setting names: `auto`, `local`, `slurm` or `slurm-step`.
    ///
    /// An unknown name is refused by name, so a mistyped flag is reported
    /// before anything is submitted.
    pub fn parse(name: &str) -> Result<LauncherKind, LauncherKindError> {
        match name.trim() {
            "auto" => Ok(LauncherKind::Auto),
            "local" => Ok(LauncherKind::Local),
            "slurm" => Ok(LauncherKind::Slurm),
            "slurm-step" => Ok(LauncherKind::SlurmStep),
            unknown => Err(LauncherKindError {
                name: unknown.to_string(),
            }),
        }
    }

    /// The name this kind is configured and recorded under.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            LauncherKind::Auto => "auto",
            LauncherKind::Local => "local",
            LauncherKind::Slurm => "slurm",
            LauncherKind::SlurmStep => "slurm-step",
        }
    }
}

/// A launcher kind the configuration does not have.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("unknown job launcher {name:?}: the kinds are `auto`, `local`, `slurm` and `slurm-step`")]
pub struct LauncherKindError {
    /// The name that was given.
    pub name: String,
}

/// What a job is submitted to when [`LauncherKind::Slurm`] is selected: the
/// scheduler's programs, and any arguments every submission passes.
///
/// The programs are values, never the process environment, so a test can
/// point the launcher at stand-in programs. A path is required for each: a
/// missing one is refused by name when the launcher is built, rather than
/// failing a submission later. `sbatch` receives [`SlurmSettings::args`] before
/// its own arguments, for a partition or an account.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SlurmSettings {
    /// The `sbatch` program that submits a job.
    pub sbatch: Option<PathBuf>,
    /// The `squeue` program that reports a job's state.
    pub squeue: Option<PathBuf>,
    /// The `scancel` program that stops a job.
    pub scancel: Option<PathBuf>,
    /// Arguments every `sbatch` submission passes before the script.
    pub args: Vec<String>,
}

/// How a process submits work that runs outside it: who the controller says it
/// is, where each attempt's directory is written, the program that runs a
/// worker and records its exit, and what starts the workers.
///
/// The owner names the controller on every submission and every lease it
/// takes, so two controllers can be told apart. It defaults to the machine's
/// host name. The attempt root defaults to `jobs` under the state root; the
/// local launcher writes each attempt there, and the Slurm launcher keeps each
/// attempt's batch script and scheduler job id there. The supervisor is the
/// executable that understands the supervisor protocol; the command line sets
/// it to itself, and a caller that leaves it unset launches workers directly,
/// so their exit is known only while the submitting process lives. The launcher
/// kind decides what starts the workers: this machine by default, or Slurm with
/// [`JobSettings::slurm`] set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JobSettings {
    /// The controller's name, recorded with every job it submits.
    pub owner: String,
    /// Where attempt directories are written; `None` is `jobs` under the state
    /// root.
    pub attempt_root: Option<PathBuf>,
    /// The executable that runs a worker and records its exit; `None` starts
    /// workers directly.
    pub supervisor: Option<PathBuf>,
    /// What starts a job's workers.
    pub launcher: LauncherKind,
    /// The Slurm programs, used when [`JobSettings::launcher`] is
    /// [`LauncherKind::Slurm`].
    pub slurm: SlurmSettings,
}

impl Default for JobSettings {
    fn default() -> Self {
        Self {
            owner: default_owner(),
            attempt_root: None,
            supervisor: None,
            launcher: LauncherKind::Auto,
            slurm: SlurmSettings::default(),
        }
    }
}

impl JobSettings {
    /// The directory attempt directories are written under for `state_root`.
    #[must_use]
    pub fn attempt_root(&self, state_root: &StateRoot) -> PathBuf {
        self.attempt_root
            .clone()
            .unwrap_or_else(|| state_root.path().join("jobs"))
    }
}

/// The controller name when none is configured: this machine's host name.
fn default_owner() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname")
        .ok()
        .map(|name| name.trim().to_owned())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "local".to_owned())
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

/// What the learning factory needs from the machine it runs on.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FactorySettings {
    /// The machine's one configuration file: how models are reached and
    /// what the task repository's operators are granted (`$SPLINTER_CONFIG`,
    /// else `$XDG_CONFIG_HOME/splinter/config.yaml`, else
    /// `~/.config/splinter/config.yaml`).
    pub config: PathBuf,
    /// The execution worker (`$SPLINTER_HARBOR_WORKER`, else the first
    /// `splinter-harbor-worker` on `PATH`).
    pub harbor_worker: Option<PathBuf>,
}

/// [`Config::min_calibration_controls`] when nothing names one.
pub const DEFAULT_MIN_CALIBRATION_CONTROLS: usize = 16;

impl Config {
    /// The configuration this process runs with before its command line
    /// names anything: every setting at its default, and the facts of this
    /// machine and user it needs:
    ///
    /// * the home directory (`$HOME`, else the account's), under which the
    ///   state root (`~/.splinter`) lives; with none, the paths under it are
    ///   left relative and [`crate::context::Runtime::new`] refuses them;
    /// * brain's model store, as brain resolves it, with the policy's base
    ///   [`DEFAULT_POLICY_MODEL`] in it;
    /// * the first `brain` executable on `PATH`, the binary a release must
    ///   serve on;
    /// * the capture allowlist's variables, and the current directory, for
    ///   command sources;
    /// * the host name, as the default name of the job controller;
    /// * the [`DOCKER_CLIENT_ENV`] variables a docker client needs.
    ///
    /// The batch allocation is not read here: a program that uses it asks
    /// [`detect_cluster`] (the command line does, through its device flags),
    /// so a test or an embedding that builds a configuration runs on this
    /// machine alone even inside an allocation.
    #[must_use]
    pub fn from_env() -> Self {
        // `HOME`, else the account's home directory. With neither, every
        // path under it stays relative, which the runtime refuses rather
        // than writing state into whatever directory it was started in.
        let home = std::env::home_dir()
            .filter(|home| home.is_absolute())
            .unwrap_or_default();
        Self::from_vars(|name| std::env::var(name).ok(), &home)
    }

    /// [`Config::from_env`] with the environment read through `lookup` and
    /// the home directory given: the precedence of every variable, with an
    /// empty value taken as unset, apart from this process's environment.
    #[must_use]
    pub fn from_vars(lookup: impl Fn(&str) -> Option<String>, home: &Path) -> Self {
        let var = |name: &str| lookup(name).filter(|v| !v.is_empty());
        // Where `brain pull` puts a checkpoint when nothing names a store.
        let model_store = var("XDG_DATA_HOME").map_or_else(
            || home.join(".local/share/brain/models"),
            |data| PathBuf::from(data).join("brain/models"),
        );
        let config_home =
            var("XDG_CONFIG_HOME").map_or_else(|| home.join(".config"), PathBuf::from);
        let factory = FactorySettings {
            config: var("SPLINTER_CONFIG")
                .map_or_else(|| config_home.join("splinter/config.yaml"), PathBuf::from),
            harbor_worker: var("SPLINTER_HARBOR_WORKER")
                .map(PathBuf::from)
                .or_else(|| {
                    var("PATH").and_then(|path| find_executable(&path, "splinter-harbor-worker"))
                }),
        };
        Self {
            state_root: StateRoot::under_home(home),
            factory,
            jobs: JobSettings::default(),
            policy_base: model_store.join(DEFAULT_POLICY_MODEL),
            policy_context_tokens: None,
            model_store,
            allow_remote: false,
            command_env: DEFAULT_ENV_ALLOWLIST
                .iter()
                .filter_map(|name| var(name).map(|value| (name.to_string(), value)))
                .collect(),
            working_dir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            brain_binary: BRAIN_DEFAULTS
                .get()
                .and_then(|d| d.binary.clone())
                .or_else(|| var("PATH").and_then(|path| find_executable(&path, "brain"))),
            front_door_model: None,
            assistant_model: None,
            judge_model: None,
            bf16_base: false,
            thinking: false,
            served_temperature: None,
            served_top_p: None,
            served_top_k: None,
            served_seed: None,
            served_thinking_budget: None,
            min_calibration_controls: DEFAULT_MIN_CALIBRATION_CONTROLS,
            remote_concurrency: DEFAULT_REMOTE_CONCURRENCY,
            brain: BRAIN_DEFAULTS.get().and_then(|d| d.endpoints.clone()),
            brain_device: BRAIN_DEFAULTS.get().and_then(|d| d.device.clone()),
            brain_backend: BRAIN_DEFAULTS.get().and_then(|d| d.backend.clone()),
            brain_max_batch: BRAIN_DEFAULTS.get().and_then(|d| d.max_batch),
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
            cluster: BRAIN_DEFAULTS
                .get()
                .map(|d| d.cluster.clone())
                .unwrap_or_default(),
        }
    }
}

/// The credential a connection names (`token: env:NAME`), read from this
/// process's environment when a model is reached in process; `None` when
/// it is unset or empty. Credentials are read here, the one place the
/// environment is, and only when they are used.
#[must_use]
pub fn credential(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

/// The batch allocation this process runs inside, as the scheduler's
/// [`SLURM_VARS`] describe it, with the `srun` and `scancel` on `PATH` that
/// start and stop work on its nodes. Outside an allocation it detects none; a
/// value that does not parse is kept as the error it is and reported when the
/// allocation is used.
#[must_use]
pub fn detect_cluster() -> ClusterSettings {
    let var = |name: &str| std::env::var(name).ok().filter(|v| !v.is_empty());
    ClusterSettings {
        detected: ClusterAllocation::from_vars(
            |name| SLURM_VARS.contains(&name).then(|| var(name)).flatten(),
            Some(default_owner()).filter(|host| host != "local"),
        ),
        srun: var("PATH").and_then(|path| find_executable(&path, "srun")),
        scancel: var("PATH").and_then(|path| find_executable(&path, "scancel")),
        // What the site announces to this step is what it announces to every
        // step: a task prolog sets it alike on each node. Whether the
        // directory exists and can be written on a node is checked there,
        // when a step is started.
        announced_scratch: SCRATCH_VARS.iter().find_map(|name| {
            var(name)
                .filter(|value| value.starts_with('/'))
                .map(|value| ((*name).to_string(), PathBuf::from(value)))
        }),
        ..ClusterSettings::default()
    }
}

/// What a program's command line says about brain before any [`Config`] is
/// built: the compute and endpoints it was given. A program that builds its
/// [`Config`] in several places (a sample's commands each do) states them once
/// and every [`Config::from_env`] after carries them.
#[derive(Clone, Debug, Default)]
pub struct BrainDefaults {
    /// [`Config::brain_device`].
    pub device: Option<String>,
    /// [`Config::brain_backend`].
    pub backend: Option<String>,
    /// [`Config::brain_max_batch`].
    pub max_batch: Option<u32>,
    /// [`Config::brain`].
    pub endpoints: Option<splinter_model::BrainEndpoints>,
    /// [`Config::brain_binary`], when the command line names one.
    pub binary: Option<PathBuf>,
    /// [`Config::cluster`].
    pub cluster: ClusterSettings,
}

static BRAIN_DEFAULTS: std::sync::OnceLock<BrainDefaults> = std::sync::OnceLock::new();

/// States the process's brain settings; the first statement stands.
///
/// # Errors
/// The settings were already stated.
pub fn set_brain_defaults(defaults: BrainDefaults) -> Result<(), &'static str> {
    BRAIN_DEFAULTS
        .set(defaults)
        .map_err(|_| "the process's brain settings were already stated")
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_launcher_kind_is_parsed_by_name_and_an_unknown_one_is_refused() {
        assert_eq!(LauncherKind::parse("local").unwrap(), LauncherKind::Local);
        assert_eq!(LauncherKind::parse("slurm").unwrap(), LauncherKind::Slurm);
        assert_eq!(LauncherKind::parse(" slurm ").unwrap(), LauncherKind::Slurm);
        assert_eq!(LauncherKind::Local.as_str(), "local");
        assert_eq!(LauncherKind::Slurm.as_str(), "slurm");

        let error = LauncherKind::parse("kubernetes").unwrap_err();
        assert_eq!(error.name, "kubernetes");
        assert!(
            error.to_string().contains("kubernetes"),
            "the refusal names the value: {error}"
        );
        assert!(error.to_string().contains("`slurm-step`"), "{error}");
        assert_eq!(
            LauncherKind::parse("slurm-step").unwrap(),
            LauncherKind::SlurmStep
        );
        assert_eq!(LauncherKind::parse("auto").unwrap().as_str(), "auto");
    }

    #[test]
    fn jobs_are_launched_locally_until_a_slurm_configuration_is_given() {
        let jobs = JobSettings::default();
        assert_eq!(
            jobs.launcher,
            LauncherKind::Auto,
            "local outside an allocation"
        );
        assert_eq!(jobs.slurm, SlurmSettings::default());
        assert_eq!(jobs.slurm.sbatch, None);
        assert_eq!(jobs.slurm.squeue, None);
        assert_eq!(jobs.slurm.scancel, None);
        assert!(jobs.slurm.args.is_empty());
    }
}

/// Spec: the configuration a process starts with is read from its
/// environment by fixed precedence, and an empty variable is an unset one.
#[cfg(test)]
mod env_spec {
    use super::*;

    fn vars<'a>(set: &'a [(&'a str, String)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |name| set.iter().find(|(n, _)| *n == name).map(|(_, v)| v.clone())
    }

    /// A directory of the test's own, under which every path it names lies.
    fn scratch(test: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("splinter-config-{test}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn text(path: &Path) -> String {
        path.to_str().unwrap().to_string()
    }

    fn program(dir: &Path, name: &str, executable: bool) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        std::fs::create_dir_all(dir).unwrap();
        let file = dir.join(name);
        std::fs::write(&file, "#!/bin/sh\n").unwrap();
        let mode = if executable { 0o755 } else { 0o644 };
        std::fs::set_permissions(&file, std::fs::Permissions::from_mode(mode)).unwrap();
        file
    }

    #[test]
    fn the_configuration_file_is_named_then_under_xdg_then_under_home() {
        let root = scratch("file");
        let home = root.join("home");
        let named = root.join("s.yaml");
        let xdg = root.join("xdg");
        let config = |set: &[(&str, String)]| Config::from_vars(vars(set), &home).factory.config;
        assert_eq!(
            config(&[
                ("SPLINTER_CONFIG", text(&named)),
                ("XDG_CONFIG_HOME", text(&xdg))
            ]),
            named
        );
        assert_eq!(
            config(&[("XDG_CONFIG_HOME", text(&xdg))]),
            xdg.join("splinter/config.yaml")
        );
        assert_eq!(
            config(&[
                ("SPLINTER_CONFIG", String::new()),
                ("XDG_CONFIG_HOME", String::new())
            ]),
            home.join(".config/splinter/config.yaml"),
            "an empty variable is an unset one"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_model_store_and_the_policy_base_follow_xdg_data_home_else_home() {
        let root = scratch("store");
        let home = root.join("home");
        let data = root.join("data");
        let under_xdg = Config::from_vars(vars(&[("XDG_DATA_HOME", text(&data))]), &home);
        assert_eq!(under_xdg.model_store, data.join("brain/models"));
        assert_eq!(
            under_xdg.policy_base,
            data.join("brain/models").join(DEFAULT_POLICY_MODEL)
        );
        let under_home = Config::from_vars(vars(&[]), &home);
        assert_eq!(
            under_home.model_store,
            home.join(".local/share/brain/models")
        );
        assert_eq!(under_home.state_root, StateRoot::under_home(&home));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_worker_is_named_else_the_first_executable_one_on_path() {
        let dir = scratch("worker");
        let (first, second) = (dir.join("a"), dir.join("b"));
        program(&first, "splinter-harbor-worker", false);
        let found = program(&second, "splinter-harbor-worker", true);
        let path = std::env::join_paths([&first, &second]).unwrap();
        let path = path.to_str().unwrap().to_string();
        let named = dir.join("w");
        let worker =
            |set: &[(&str, String)]| Config::from_vars(vars(set), &dir).factory.harbor_worker;
        assert_eq!(
            worker(&[("PATH", path.clone())]),
            Some(found),
            "a file that is not executable is passed over"
        );
        assert_eq!(
            worker(&[("SPLINTER_HARBOR_WORKER", text(&named)), ("PATH", path)]),
            Some(named)
        );
        assert_eq!(worker(&[]), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn only_the_allowlisted_variables_are_captured_and_empty_ones_are_not() {
        let root = scratch("capture");
        let set = [
            ("PATH", text(&root.join("bin"))),
            ("HOME", String::new()),
            ("DOCKER_HOST", "unix:d.sock".to_string()),
            ("SECRET_TOKEN", "x".to_string()),
        ];
        let config = Config::from_vars(vars(&set), &root);
        assert_eq!(config.command_env.get("PATH"), Some(&set[0].1));
        assert!(!config.command_env.contains_key("HOME"), "empty is unset");
        assert!(!config.command_env.contains_key("SECRET_TOKEN"));
        assert!(config
            .command_env
            .keys()
            .all(|k| DEFAULT_ENV_ALLOWLIST.contains(&k.as_str())));
        assert_eq!(
            config
                .workspace
                .docker_env
                .get("DOCKER_HOST")
                .map(String::as_str),
            Some("unix:d.sock")
        );
        assert!(!config.workspace.docker_env.contains_key("SECRET_TOKEN"));
        let _ = std::fs::remove_dir_all(&root);
    }
}
