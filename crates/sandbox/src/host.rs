// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements bounded execution of model-written code for
// its clients. If your team needs expertise in agent sandboxing or process
// supervision, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The process sandbox: the runtime runs directly on the host, bounded but
//! **not isolated**.
//!
//! Each call gets a fresh directory under the scratch root (its working
//! directory, `HOME` and `TMPDIR`), removed when the call ends; a command
//! in a directory tree runs there instead, with only `HOME` and `TMPDIR`
//! in such a fresh directory; exactly
//! the environment the sandbox was built with; its standard input from the
//! call; a wall-clock timeout that kills its whole process group; output
//! caps; and kernel resource limits (CPU time, address space, file size)
//! set in the child before the runtime starts. The number of processes a
//! call starts can be bounded by a budget on top of the user's own task
//! count when the call starts ([`ProcessSandbox::max_processes`]; unbounded
//! unless set, and the record says which).
//!
//! A code call is bounded only. A command in a directory tree
//! ([`Sandbox::run_command`], how a workspace is worked in) is also
//! confined by the kernel: it reaches the system (read only), its tree and
//! its private directory and nothing else of the filesystem, and no network
//! (TCP by Landlock, every internet socket by seccomp) unless [`ProcessSandbox::allow_network`] says so. Its
//! environment holds no variable named like a secret even when the
//! allowlist names one ([`is_secret_name`]). A kernel that cannot confine
//! a command refuses it. The environment records `isolation: "none"` all
//! the same (code calls are not confined), so an experience produced here
//! is never mistaken for a container-isolated one. `/proc` stays readable
//! (system tools need it), so the supervisor makes itself non-dumpable with
//! [`crate::process::protect_supervisor_environment`]; that is what keeps its
//! `/proc/<pid>/environ` unreadable to its commands.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::backend::{
    io, Backend, CallDir, CodeCall, CodeResult, CommandCall, Isolation, Sandbox, SandboxError,
    SandboxSpec,
};
use crate::landlock::Confinement;
use crate::limits::Limits;
use crate::process::{environment_from, run, ProcessSpec};
use crate::runtime::{resolve_on_host, ResolvedRuntime, RuntimeSpec};

/// The `PATH` of a call unless the caller passes its own through the
/// allowlist; also where runtimes are looked up.
pub const DEFAULT_SEARCH_PATH: &str = "/usr/local/bin:/usr/bin:/bin";

/// The `LANG` of a call unless the caller passes its own.
pub const DEFAULT_LANG: &str = "C.UTF-8";

/// The `TERM` of a call unless the caller passes its own: no colour, no
/// cursor control, so output is plain text.
pub const DEFAULT_TERM: &str = "dumb";

/// What every command may read and run: the system's own directories.
const SYSTEM_ROOTS: [&str; 9] = [
    "/usr", "/lib", "/lib32", "/lib64", "/bin", "/sbin", "/etc", "/opt", "/proc",
];

/// Whether a variable of this name carries a credential: an API key, token,
/// secret, password, authentication socket or the like. Such a variable is
/// never passed to a command, whatever the allowlist says, so a
/// misconfigured allowlist cannot hand the solver the supervisor's keys.
#[must_use]
pub fn is_secret_name(name: &str) -> bool {
    const MARKERS: [&str; 8] = [
        "KEY",
        "TOKEN",
        "SECRET",
        "PASSWORD",
        "PASSWD",
        "CREDENTIAL",
        "AUTH",
        "COOKIE",
    ];
    const PREFIXES: [&str; 6] = ["AWS_", "GOOGLE_", "AZURE_", "GITHUB_", "GH_", "HF_"];
    let upper = name.to_ascii_uppercase();
    MARKERS.iter().any(|m| upper.contains(m)) || PREFIXES.iter().any(|p| upper.starts_with(p))
}

/// Variables of the environment that name a toolchain's home directory.
const TOOLCHAIN_HOMES: [&str; 2] = ["CARGO_HOME", "RUSTUP_HOME"];

/// The directory a program directory belongs to: `/venv` for `/venv/bin`,
/// the directory itself otherwise, and nothing for the filesystem root.
fn toolchain_root(program_dir: &Path) -> Option<PathBuf> {
    let dir = if program_dir.is_file() {
        program_dir.parent()?
    } else {
        program_dir
    };
    let root = if dir.file_name().is_some_and(|n| n == "bin") {
        dir.parent()?
    } else {
        dir
    };
    (root.parent().is_some()).then(|| root.to_path_buf())
}

/// Variables a call always gets set to its own directory.
const PER_CALL_VARIABLES: [&str; 2] = ["HOME", "TMPDIR"];

/// Runs a runtime directly on the host, in a fresh directory per call,
/// under [`Limits`]. **Not isolation**: see the module documentation.
#[derive(Clone, Debug)]
pub struct ProcessSandbox {
    scratch_root: PathBuf,
    env: BTreeMap<String, String>,
    limits: Limits,
    allow_unconfined: bool,
    allow_network: bool,
    landlock: bool,
}

impl ProcessSandbox {
    /// A sandbox whose calls get their directories under `scratch_root`
    /// and run under `limits`, with an environment of `PATH`, `LANG` and
    /// `TERM` ([`DEFAULT_SEARCH_PATH`], [`DEFAULT_LANG`], [`DEFAULT_TERM`])
    /// plus the variables of `caller_env` (the caller's environment, as the
    /// caller reads it) named in `allowlist`, which override those
    /// defaults. `HOME` and `TMPDIR` are always the call's own directory.
    #[must_use]
    pub fn new(
        scratch_root: impl Into<PathBuf>,
        caller_env: impl IntoIterator<Item = (String, String)>,
        allowlist: &[&str],
        limits: Limits,
    ) -> Self {
        let mut env: BTreeMap<String, String> = [
            ("PATH", DEFAULT_SEARCH_PATH),
            ("LANG", DEFAULT_LANG),
            ("TERM", DEFAULT_TERM),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
        env.extend(
            environment_from(caller_env, allowlist)
                .into_iter()
                .filter(|(name, _)| !is_secret_name(name)),
        );
        for name in PER_CALL_VARIABLES {
            env.remove(name);
        }
        Self {
            scratch_root: scratch_root.into(),
            env,
            limits: Limits {
                max_processes: None,
                ..limits
            },
            allow_unconfined: false,
            allow_network: false,
            landlock: crate::landlock::abi_version().is_some(),
        }
    }

    /// Lets commands run with the caller's whole filesystem where the kernel
    /// cannot confine them (and when `allow` is set, everywhere). Without
    /// it a command is confined to its tree and a private directory, and is
    /// refused ([`SandboxError::Unconfined`]) where that is impossible. The
    /// environment record says so when it is set.
    #[must_use]
    pub fn allow_unconfined(mut self, allow: bool) -> Self {
        self.allow_unconfined = allow;
        self
    }

    /// Bounds the processes and threads of every call to `budget` beyond the
    /// ones the user runs when the call starts (see [`Limits::max_processes`]);
    /// `None` (the default) is no bound. The record carries it, so it is part
    /// of the environment's identity: a sandbox built without it is a
    /// different environment from one built with it.
    #[must_use]
    pub fn max_processes(mut self, budget: Option<u64>) -> Self {
        self.limits.max_processes = budget;
        self
    }

    /// Lets commands use the network. Off by default: a command cannot
    /// connect anywhere (a brain server on the loopback included) and a
    /// kernel that cannot deny the network refuses to run it. The
    /// environment record says so when it is set.
    #[must_use]
    pub fn allow_network(mut self, allow: bool) -> Self {
        self.allow_network = allow;
        self
    }

    /// Behaves as on a kernel without Landlock, for a test of the refusal.
    #[doc(hidden)]
    #[must_use]
    pub fn without_landlock(mut self) -> Self {
        self.landlock = false;
        self
    }

    /// What a command in `tree`, with `home` as its private directory, may
    /// reach: the system, the directories its `PATH` and toolchain
    /// variables name (read only), its tree, its private directory and the
    /// device files (read and write). Nothing else: no home directory, no
    /// other tree, no state root.
    fn command_confinement(&self, tree: &Path, home: &Path) -> Confinement {
        let mut read_only: Vec<PathBuf> = SYSTEM_ROOTS.iter().map(PathBuf::from).collect();
        for entry in self.search_path().split(':').filter(|e| !e.is_empty()) {
            read_only.extend(toolchain_root(Path::new(entry)));
            read_only.extend(toolchain_root(
                &Path::new(entry)
                    .join("python3")
                    .canonicalize()
                    .unwrap_or_default(),
            ));
        }
        let mut read_write = vec![
            tree.to_path_buf(),
            home.to_path_buf(),
            PathBuf::from("/dev"),
        ];
        for name in TOOLCHAIN_HOMES {
            // A toolchain's home is written to (cargo locks its registry).
            read_write.extend(self.env.get(name).map(PathBuf::from));
        }
        Confinement {
            read_only,
            read_write,
            allow_network: self.allow_network,
        }
    }

    /// The limits every call runs under.
    #[must_use]
    pub fn limits(&self) -> &Limits {
        &self.limits
    }

    fn search_path(&self) -> &str {
        self.env
            .get("PATH")
            .map_or(DEFAULT_SEARCH_PATH, String::as_str)
    }

    /// The bounded run of `argv` in `cwd`, with `HOME` and `TMPDIR` set to
    /// `home`.
    fn process_spec(
        &self,
        argv: Vec<String>,
        cwd: &Path,
        home: &str,
        stdin: Option<&String>,
    ) -> ProcessSpec {
        let mut env = self.env.clone();
        for name in PER_CALL_VARIABLES {
            env.insert(name.to_string(), home.to_string());
        }
        ProcessSpec {
            argv,
            cwd: cwd.to_path_buf(),
            env,
            stdin: stdin.map(|s| s.as_bytes().to_vec()),
            timeout: self.limits.wall_time(),
            output_cap: self.limits.output_cap(),
            resources: Some(self.limits.resources()),
            confinement: None,
        }
    }

    fn execute(&self, spec: &ProcessSpec) -> Result<CodeResult, SandboxError> {
        let output = run(spec)?;
        // The kernel sends SIGXCPU when the CPU-time limit is spent.
        let timed_out = output.timed_out || output.signal == Some(libc::SIGXCPU);
        Ok(CodeResult::from_output(&output, timed_out))
    }

    fn run_in(
        &self,
        dir: &CallDir,
        runtime: &ResolvedRuntime,
        call: &CodeCall,
    ) -> Result<CodeResult, SandboxError> {
        let file = dir.write_code(runtime, &call.code)?;
        let (Some(file_arg), Some(dir_arg)) = (file.to_str(), dir.path().to_str()) else {
            return Err(io(&file)(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "the scratch root is not a UTF-8 path",
            )));
        };
        let argv = runtime.spec.argv_for(&runtime.executable, file_arg);
        self.execute(&self.process_spec(argv, dir.path(), dir_arg, call.stdin.as_ref()))
    }
}

impl Sandbox for ProcessSandbox {
    fn isolation(&self) -> Isolation {
        Isolation::None
    }

    fn spec(&self) -> SandboxSpec {
        SandboxSpec {
            backend: Backend::Process,
            isolation: Isolation::None,
            limits: self.limits,
            env: self.env.clone(),
            unconfined_commands: self.allow_unconfined,
            network_allowed: self.allow_network,
        }
    }

    fn resolve(&self, runtime: &RuntimeSpec) -> Result<ResolvedRuntime, SandboxError> {
        resolve_on_host(runtime, self.search_path(), &self.env)
    }

    fn run(&self, runtime: &ResolvedRuntime, call: &CodeCall) -> Result<CodeResult, SandboxError> {
        let dir = CallDir::create(&self.scratch_root)?;
        let result = self.run_in(&dir, runtime, call);
        let removed = dir.remove();
        let result = result?;
        removed?;
        Ok(result)
    }

    /// Runs the command with `dir` as its working directory. `HOME` and
    /// `TMPDIR` name a fresh directory of the call under the scratch root,
    /// removed afterwards, so what the command keeps in `dir` is only what
    /// it means to leave there.
    fn run_command(&self, dir: &Path, call: &CommandCall) -> Result<CodeResult, SandboxError> {
        if !self.allow_unconfined && !self.landlock {
            return Err(SandboxError::Unconfined);
        }
        let home = CallDir::create(&self.scratch_root)?;
        let result = match home.path().to_str() {
            Some(home_arg) => {
                let mut spec =
                    self.process_spec(call.argv.clone(), dir, home_arg, call.stdin.as_ref());
                if !self.allow_unconfined {
                    spec.confinement = Some(self.command_confinement(dir, home.path()));
                }
                self.execute(&spec)
            }
            None => Err(io(home.path())(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "the scratch root is not a UTF-8 path",
            ))),
        };
        let removed = home.remove();
        let result = result?;
        removed?;
        Ok(result)
    }
}
