// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements container-isolated execution of
// model-written code for its clients. If your team needs expertise in agent
// sandboxing, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The container sandbox: each call runs in a fresh container of an image
//! pinned by digest.
//!
//! `docker run --rm` with no network, a read-only root filesystem, a
//! writable `/work` in memory (the working directory, `HOME` and `TMPDIR`),
//! an unprivileged user with every capability dropped, the call's code
//! mounted read-only, and the [`Limits`] as the container's memory, process
//! and `ulimit` limits plus a CPU share. The image is named as
//! `repository@sha256:<digest>`, never by a tag, so the snapshot's image
//! digest is exactly what ran; the image must be present locally for a call
//! (it is pulled, if missing, when the runtime is resolved).
//!
//! The wall-clock limit covers the whole `docker run`, container start
//! included; a call that outlives it has its client killed and its
//! container removed.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use splinter_store::digest::Digest;

use crate::backend::{
    io, Backend, CallDir, CodeCall, CodeResult, Isolation, Sandbox, SandboxError, SandboxSpec,
    CODE_FILE_STEM,
};
use crate::host::{DEFAULT_LANG, DEFAULT_SEARCH_PATH, DEFAULT_TERM};
use crate::limits::Limits;
use crate::process::{run, ProcessOutput, ProcessSpec};
use crate::runtime::{version_from, ResolvedRuntime, RuntimeError, RuntimeSpec};

/// The default CPU share of a call's container, in thousandths of a CPU.
pub const DEFAULT_CPU_MILLIS: u32 = 1000;

/// The user a call runs as inside its container: `nobody`.
pub const CONTAINER_USER: &str = "65534:65534";

/// The writable directory of a call inside its container.
pub const WORK_DIR: &str = "/work";

/// Where a call's code file is mounted inside its container.
pub const CODE_DIR: &str = "/code";

/// How long a docker command other than a call (probing the daemon,
/// resolving a runtime, which may pull the image, removing a container)
/// may take.
const CLIENT_TIMEOUT: Duration = Duration::from_secs(300);

/// The most bytes kept of a docker command's own output.
const CLIENT_OUTPUT_CAP: usize = 64 * 1024;

/// The exit status `docker run` reports when docker itself, not the
/// program in the container, failed.
const DOCKER_FAILED: i32 = 125;

/// The exit status `docker run` reports when the command is not found in
/// the image.
const COMMAND_NOT_FOUND: i32 = 127;

/// A container image pinned by digest: `repository@sha256:<hex>`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ImageRef {
    /// The repository, with its registry when it has one.
    pub repository: String,
    /// The image's content digest.
    pub digest: Digest,
}

/// Why a string is not a digest-pinned image reference.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{value:?} is not an image pinned by digest (repository@sha256:<64 hex>, no tag)")]
pub struct ImageRefError {
    /// The rejected text.
    pub value: String,
}

impl ImageRef {
    /// Parses `repository@sha256:<hex>`, refusing a reference with no
    /// digest, and one with a tag (which a digest makes meaningless and a
    /// reader could mistake for what ran).
    pub fn parse(text: &str) -> Result<Self, ImageRefError> {
        let refused = || ImageRefError {
            value: text.to_string(),
        };
        let (repository, digest) = text.rsplit_once('@').ok_or_else(refused)?;
        let last_component = repository.rsplit('/').next().unwrap_or(repository);
        if repository.is_empty() || last_component.is_empty() || last_component.contains(':') {
            return Err(refused());
        }
        Ok(Self {
            repository: repository.to_string(),
            digest: Digest::parse(digest).map_err(|_| refused())?,
        })
    }
}

impl std::fmt::Display for ImageRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}@{}", self.repository, self.digest)
    }
}

impl TryFrom<String> for ImageRef {
    type Error = ImageRefError;
    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::parse(&value)
    }
}

impl From<ImageRef> for String {
    fn from(image: ImageRef) -> Self {
        image.to_string()
    }
}

/// Runs each call in a fresh container of a pinned image.
#[derive(Clone, Debug)]
pub struct ContainerSandbox {
    docker: PathBuf,
    client_env: BTreeMap<String, String>,
    image: ImageRef,
    scratch_root: PathBuf,
    limits: Limits,
    cpu_millis: u32,
    env: BTreeMap<String, String>,
}

impl ContainerSandbox {
    /// A sandbox that runs `docker` (with `client_env`, the docker client's
    /// own environment: what it needs to reach its daemon) to start
    /// containers of `image`, stages each call's code under `scratch_root`,
    /// and applies `limits` and [`DEFAULT_CPU_MILLIS`]. Inside the
    /// container a call sees `PATH`, `LANG` and `TERM` as the process
    /// sandbox sets them, and `HOME` and `TMPDIR` as [`WORK_DIR`].
    #[must_use]
    pub fn new(
        docker: PathBuf,
        client_env: BTreeMap<String, String>,
        image: ImageRef,
        scratch_root: impl Into<PathBuf>,
        limits: Limits,
    ) -> Self {
        let env = [
            ("PATH", DEFAULT_SEARCH_PATH),
            ("LANG", DEFAULT_LANG),
            ("TERM", DEFAULT_TERM),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
        Self {
            docker,
            client_env,
            image,
            scratch_root: scratch_root.into(),
            limits,
            cpu_millis: DEFAULT_CPU_MILLIS,
            env,
        }
    }

    /// The same sandbox with a CPU share of `cpu_millis` thousandths of a
    /// CPU per call.
    #[must_use]
    pub fn with_cpu_millis(mut self, cpu_millis: u32) -> Self {
        self.cpu_millis = cpu_millis;
        self
    }

    /// The image every call runs in.
    #[must_use]
    pub fn image(&self) -> &ImageRef {
        &self.image
    }

    /// Whether `docker` can reach its daemon with `client_env`: `Ok`, or
    /// what docker said when it could not.
    pub fn reachable(docker: &Path, client_env: &BTreeMap<String, String>) -> Result<(), String> {
        let argv = vec![docker.display().to_string(), "info".into()];
        let spec = ProcessSpec::new(
            argv,
            "/",
            client_env.clone(),
            CLIENT_TIMEOUT,
            CLIENT_OUTPUT_CAP,
        );
        match run(&spec) {
            Ok(out) if out.exit_code == Some(0) => Ok(()),
            Ok(out) => Err(summary(&out)),
            Err(e) => Err(e.to_string()),
        }
    }

    fn client(&self, argv: Vec<String>, timeout: Duration) -> ProcessSpec {
        let mut full = vec![self.docker.display().to_string()];
        full.extend(argv);
        ProcessSpec::new(
            full,
            "/",
            self.client_env.clone(),
            timeout,
            CLIENT_OUTPUT_CAP,
        )
    }

    /// The flags every container of this sandbox is started with.
    fn container_flags(&self) -> Vec<String> {
        let l = &self.limits;
        let cpus = format!("{}.{:03}", self.cpu_millis / 1000, self.cpu_millis % 1000);
        let mut flags: Vec<String> = [
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            CONTAINER_USER,
            "--workdir",
            WORK_DIR,
        ]
        .into_iter()
        .map(str::to_string)
        .collect();
        flags.extend([
            "--tmpfs".into(),
            format!("{WORK_DIR}:rw,exec,nosuid,size={}", l.file_size_bytes),
            "--memory".into(),
            l.memory_bytes.to_string(),
            "--memory-swap".into(),
            l.memory_bytes.to_string(),
            "--cpus".into(),
            cpus,
            "--ulimit".into(),
            format!("cpu={}:{}", l.cpu_seconds, l.cpu_seconds.saturating_add(1)),
            "--ulimit".into(),
            format!("fsize={0}:{0}", l.file_size_bytes),
            "--ulimit".into(),
            "core=0:0".into(),
        ]);
        if let Some(max) = l.max_processes {
            flags.extend(["--pids-limit".into(), max.to_string()]);
        }
        let mut env = self.env.clone();
        env.insert("HOME".into(), WORK_DIR.into());
        env.insert("TMPDIR".into(), WORK_DIR.into());
        for (name, value) in env {
            flags.extend(["--env".into(), format!("{name}={value}")]);
        }
        flags
    }

    /// The `docker` arguments of one call named `name` that runs
    /// `runtime` on the host file `code`, reading standard input when
    /// `stdin` is set.
    fn call_argv(
        &self,
        name: &str,
        runtime: &ResolvedRuntime,
        code: &str,
        stdin: bool,
    ) -> Vec<String> {
        let target = format!("{CODE_DIR}/{CODE_FILE_STEM}.{}", runtime.spec.extension);
        let mut argv = self.container_flags();
        argv.extend([
            "--name".into(),
            name.to_string(),
            "--pull".into(),
            "never".into(),
            "--mount".into(),
            format!("type=bind,source={code},target={target},readonly"),
        ]);
        if stdin {
            argv.push("--interactive".into());
        }
        argv.push(self.image.to_string());
        argv.extend(runtime.spec.argv_for(&runtime.executable, &target));
        argv
    }

    fn run_in(
        &self,
        dir: &CallDir,
        runtime: &ResolvedRuntime,
        call: &CodeCall,
    ) -> Result<CodeResult, SandboxError> {
        use std::os::unix::fs::PermissionsExt;
        let file = dir.write_code(runtime, &call.code)?;
        // The container's unprivileged user reads the mounted file.
        std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o644))
            .map_err(io(&file))?;
        let (Some(code), Some(name)) = (
            file.to_str(),
            dir.path().file_name().and_then(|n| n.to_str()),
        ) else {
            return Err(io(&file)(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "the scratch root is not a UTF-8 path",
            )));
        };
        let name = name.to_string();
        let mut spec = self.client(
            self.call_argv(&name, runtime, code, call.stdin.is_some()),
            self.limits.wall_time(),
        );
        spec.output_cap = self.limits.output_cap();
        spec.stdin = call.stdin.as_ref().map(|s| s.as_bytes().to_vec());
        let output = run(&spec)?;
        if output.timed_out {
            // Killing the client leaves the container running.
            let removed =
                run(&self.client(vec!["rm".into(), "--force".into(), name], CLIENT_TIMEOUT))?;
            if removed.exit_code != Some(0) {
                return Err(SandboxError::Container {
                    what: "removing a timed-out container",
                    detail: summary(&removed),
                });
            }
        } else if output.exit_code == Some(DOCKER_FAILED) {
            return Err(SandboxError::Container {
                what: "starting a call",
                detail: summary(&output),
            });
        }
        // `docker run` reports a program killed by a signal as 128 plus the
        // signal; SIGXCPU is the CPU-time limit.
        let timed_out = output.timed_out || output.exit_code == Some(128 + libc::SIGXCPU);
        Ok(CodeResult::from_output(&output, timed_out))
    }
}

impl Sandbox for ContainerSandbox {
    fn isolation(&self) -> Isolation {
        Isolation::Container
    }

    fn spec(&self) -> SandboxSpec {
        SandboxSpec {
            backend: Backend::Container {
                image: self.image.clone(),
                cpu_millis: self.cpu_millis,
            },
            isolation: Isolation::Container,
            limits: self.limits,
            env: self.env.clone(),
        }
    }

    /// Resolves `runtime` inside the image: the program by name on the
    /// image's search path, its version as the probe reports it there. The
    /// image digest pins the executable, so no executable digest is taken.
    fn resolve(&self, runtime: &RuntimeSpec) -> Result<ResolvedRuntime, SandboxError> {
        let version = match runtime.version_argv_for(&runtime.program) {
            None => None,
            Some(probe) => {
                let mut argv = self.container_flags();
                argv.push(self.image.to_string());
                argv.extend(probe);
                let output = run(&self.client(argv, CLIENT_TIMEOUT)).map_err(|source| {
                    RuntimeError::ProbeFailed {
                        runtime: runtime.name.clone(),
                        source,
                    }
                })?;
                match output.exit_code {
                    Some(COMMAND_NOT_FOUND) => {
                        return Err(RuntimeError::ExecutableNotFound {
                            runtime: runtime.name.clone(),
                            program: runtime.program.clone(),
                            search_path: format!("the PATH of {}", self.image),
                        }
                        .into())
                    }
                    Some(DOCKER_FAILED) => {
                        return Err(SandboxError::Container {
                            what: "probing a runtime",
                            detail: summary(&output),
                        })
                    }
                    _ => Some(version_from(&runtime.name, &output)?),
                }
            }
        };
        Ok(ResolvedRuntime {
            spec: runtime.clone(),
            executable: runtime.program.clone(),
            version,
            executable_digest: None,
        })
    }

    fn run(&self, runtime: &ResolvedRuntime, call: &CodeCall) -> Result<CodeResult, SandboxError> {
        let dir = CallDir::create(&self.scratch_root)?;
        let result = self.run_in(&dir, runtime, call);
        let removed = dir.remove();
        let result = result?;
        removed?;
        Ok(result)
    }
}

/// What a docker command said when it failed.
fn summary(output: &ProcessOutput) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr.bytes);
    format!(
        "exit code {:?}: {}",
        output.exit_code,
        stderr.lines().next().unwrap_or("").trim()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::RuntimeRegistry;

    #[test]
    fn a_call_runs_without_network_on_a_read_only_root_of_the_pinned_image() {
        let image = ImageRef::parse(&format!("python@sha256:{}", "b".repeat(64))).unwrap();
        let limits = Limits::default();
        let sandbox = ContainerSandbox::new(
            PathBuf::from("docker"),
            BTreeMap::new(),
            image.clone(),
            "/scratch",
            limits,
        );
        let spec = RuntimeRegistry::builtin().get("python3").unwrap().clone();
        let runtime = ResolvedRuntime {
            executable: spec.program.clone(),
            spec,
            version: Some("Python 3".into()),
            executable_digest: None,
        };
        let argv = sandbox.call_argv("splinter-call-x", &runtime, "/scratch/x/main.py", true);
        let joined = argv.join(" ");
        for expected in [
            "run --rm --network none --read-only".to_string(),
            "--cap-drop ALL".to_string(),
            format!("--memory {}", limits.memory_bytes),
            format!("--pids-limit {}", limits.max_processes.unwrap_or_default()),
            "--cpus 1.000".to_string(),
            "type=bind,source=/scratch/x/main.py,target=/code/main.py,readonly".to_string(),
            format!("--interactive {image} python3 -I /code/main.py"),
        ] {
            assert!(joined.contains(&expected), "{expected} missing: {joined}");
        }
        assert!(!joined.contains(":latest"));
    }
}
