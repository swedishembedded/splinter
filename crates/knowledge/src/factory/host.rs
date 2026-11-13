// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verifiers that are run, attacked and
// re-run before a model is allowed to learn from them, for its clients. If
// your team needs expertise in admission testing of training environments,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Running a task folder on the host, the way a container would run it.
//!
//! Admission has to run each environment's reference, its verifier and the
//! attacks on it without a container runtime. A [`Session`] lays the folder
//! out in a private scratch directory (the workspace, `tests/`, `solution/`
//! and a logs directory), runs the environment's reset script when it has
//! one, and hands the three container paths to every script through
//! `APP_DIR`, `TESTS_DIR`, `SOLUTION_DIR` and `LOGS_DIR`. Scripts written
//! for this format address those variables with the container paths as
//! their defaults (`${APP_DIR:-/app}`), so the same files run unchanged in
//! a container.
//!
//! A verifier that crashes writes no reward. That is a different fact from a
//! reward of zero: [`Verdict::NoReward`] is an infrastructure failure, never
//! a wrong answer, and admission treats it as a failure of the verifier.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use splinter_core::digest::Digest;
use splinter_sandbox::host::DEFAULT_SEARCH_PATH;
use splinter_sandbox::process::{run, ProcessError, ProcessOutput, ProcessSpec, ResourceLimits};

use super::artifact::{Instance, ENVIRONMENT_DIR, RESET_SCRIPT, SOLVE_SCRIPT, TEST_SCRIPT};
use super::folder::{Folder, FolderError};

/// The most output of a script kept.
const OUTPUT_CAP: usize = 64 * 1024;

/// What a verifier writes, relative to the logs directory.
const REWARD_FILE: &str = "reward.json";

/// How scripts run: where they find programs and how long they may take.
#[derive(Clone, Debug)]
pub struct Host {
    path: String,
    timeout: Duration,
    limits: ResourceLimits,
}

impl Default for Host {
    fn default() -> Self {
        Self::with_path(DEFAULT_SEARCH_PATH)
    }
}

impl Host {
    /// A host whose scripts find programs on `path` (a `PATH` value). The
    /// caller supplies it: nothing here reads the process environment.
    #[must_use]
    pub fn with_path(path: &str) -> Self {
        Self {
            path: path.to_string(),
            timeout: Duration::from_secs(60),
            limits: ResourceLimits {
                cpu_seconds: 60,
                address_space_bytes: 8 << 30,
                file_size_bytes: 64 << 20,
                max_processes: None,
                tree_memory_bytes: None,
            },
        }
    }

    /// This host with a different time limit per script.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Runs `argv` in `cwd` with `env` beside the host's own variables
    /// (`PATH`, `LC_ALL`, `PYTHONDONTWRITEBYTECODE`), bounded in time and
    /// resources.
    pub fn exec(
        &self,
        cwd: &Path,
        argv: Vec<String>,
        env: BTreeMap<String, String>,
    ) -> Result<Ran, HostError> {
        let mut vars = BTreeMap::from([
            ("PATH".to_string(), self.path.clone()),
            ("LC_ALL".to_string(), "C".to_string()),
            ("PYTHONDONTWRITEBYTECODE".to_string(), "1".to_string()),
        ]);
        vars.extend(env);
        let mut spec = ProcessSpec::new(argv, cwd, vars, self.timeout, OUTPUT_CAP);
        spec.resources = Some(self.limits);
        Ok(ran(&run(&spec)?))
    }

    /// Lays `instance` out in a fresh scratch directory and resets it.
    pub fn open(&self, instance: &Instance) -> Result<Session, HostError> {
        let scratch = tempfile::tempdir().map_err(|source| HostError::Io {
            path: "scratch".into(),
            source,
        })?;
        let root = scratch.path().to_path_buf();
        for dir in ["home", "tmp", "logs"] {
            std::fs::create_dir_all(root.join(dir)).map_err(|source| HostError::Io {
                path: dir.into(),
                source,
            })?;
        }
        instance.workspace().write_to(&root.join("app"))?;
        for (label, prefix) in [("tests", "tests"), ("solution", "solution")] {
            let mut part = Folder::new();
            for (path, entry) in instance.folder().under(prefix) {
                part.put(
                    &path[prefix.len() + 1..],
                    entry.bytes.clone(),
                    entry.executable,
                )?;
            }
            part.write_to(&root.join(label))?;
        }
        let session = Session {
            host: self.clone(),
            _scratch: scratch,
            root,
            has_reset: instance.folder().get(RESET_SCRIPT).is_some(),
        };
        session.reset()?;
        Ok(session)
    }
}

/// Why a script could not be run at all. A script that fails is an outcome.
#[derive(Debug, thiserror::Error)]
pub enum HostError {
    /// The disk refused.
    #[error("{path}: {source}")]
    Io {
        /// What was being touched.
        path: String,
        /// The error.
        source: std::io::Error,
    },
    /// A folder could not be laid out or read back.
    #[error(transparent)]
    Folder(#[from] FolderError),
    /// The process could not be run.
    #[error(transparent)]
    Process(#[from] ProcessError),
    /// The environment's reset script failed.
    #[error("reset failed ({exit}): {tail}")]
    Reset {
        /// How it ended.
        exit: String,
        /// The end of its output.
        tail: String,
    },
}

/// What a verifier said about the state it was run on.
#[derive(Clone, Debug, PartialEq)]
pub enum Verdict {
    /// A reward vector: every value in `0..=1`, with `reward` the scalar.
    Graded(BTreeMap<String, f64>),
    /// The verifier wrote no reward: it crashed or timed out. An
    /// infrastructure failure, not a wrong answer.
    NoReward {
        /// How the script ended.
        exit: String,
        /// The end of its output.
        tail: String,
    },
    /// The verifier wrote a reward that is not a flat object of numbers in
    /// `0..=1` with a `reward` entry.
    Malformed(String),
}

impl Verdict {
    /// The scalar reward, when the verifier graded.
    #[must_use]
    pub fn reward(&self) -> Option<f64> {
        match self {
            Self::Graded(vector) => vector.get("reward").copied(),
            _ => None,
        }
    }

    /// Whether the verifier graded this state as solved: a scalar reward of
    /// exactly one. Partial credit is not a pass.
    #[must_use]
    pub fn solved(&self) -> bool {
        self.reward().is_some_and(|r| r >= 1.0)
    }
}

/// How a script ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ran {
    /// The exit code, `None` when a signal or the timeout ended it.
    pub exit_code: Option<i32>,
    /// What it wrote to standard output.
    pub stdout: String,
    /// What it wrote to standard error.
    pub stderr: String,
}

impl Ran {
    /// Whether it exited with status zero.
    #[must_use]
    pub fn succeeded(&self) -> bool {
        self.exit_code == Some(0)
    }

    fn describe(&self) -> String {
        self.exit_code
            .map_or_else(|| "killed".to_string(), |c| format!("exit {c}"))
    }

    fn tail(&self) -> String {
        let text = format!("{}{}", self.stdout, self.stderr);
        let start = text.len().saturating_sub(600);
        let start = text
            .char_indices()
            .map(|(i, _)| i)
            .find(|i| *i >= start)
            .unwrap_or(text.len());
        text[start..].to_string()
    }
}

fn ran(output: &ProcessOutput) -> Ran {
    Ran {
        exit_code: if output.timed_out {
            None
        } else {
            output.exit_code
        },
        stdout: String::from_utf8_lossy(&output.stdout.bytes).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr.bytes).into_owned(),
    }
}

/// One instance laid out in a scratch directory.
pub struct Session {
    host: Host,
    _scratch: tempfile::TempDir,
    root: PathBuf,
    has_reset: bool,
}

impl Session {
    /// The workspace directory.
    #[must_use]
    pub fn app_dir(&self) -> PathBuf {
        self.root.join("app")
    }

    fn logs_dir(&self) -> PathBuf {
        self.root.join("logs")
    }

    fn script(&self, argv: Vec<String>, cwd: &Path) -> Result<Ran, HostError> {
        let dir = |name: &str| self.root.join(name).display().to_string();
        let env = BTreeMap::from([
            ("HOME".to_string(), dir("home")),
            ("TMPDIR".to_string(), dir("tmp")),
            ("APP_DIR".to_string(), dir("app")),
            ("TESTS_DIR".to_string(), dir("tests")),
            ("SOLUTION_DIR".to_string(), dir("solution")),
            ("LOGS_DIR".to_string(), dir("logs")),
        ]);
        self.host.exec(cwd, argv, env)
    }

    /// Whether the environment has a reset script.
    #[must_use]
    pub fn has_reset(&self) -> bool {
        self.has_reset
    }

    /// Rebuilds the workspace's state from its sources, for an environment
    /// that has a reset script; nothing otherwise.
    pub fn reset(&self) -> Result<(), HostError> {
        if !self.has_reset {
            return Ok(());
        }
        let name = RESET_SCRIPT.trim_start_matches(&format!("{ENVIRONMENT_DIR}/"));
        let outcome = self.script(vec!["sh".into(), name.into()], &self.app_dir())?;
        if outcome.succeeded() {
            Ok(())
        } else {
            Err(HostError::Reset {
                exit: outcome.describe(),
                tail: outcome.tail(),
            })
        }
    }

    /// Runs the reference solution in the workspace.
    pub fn solve(&self) -> Result<Ran, HostError> {
        let script = self
            .root
            .join("solution")
            .join(SOLVE_SCRIPT.trim_start_matches("solution/"));
        self.script(
            vec!["sh".into(), script.display().to_string()],
            &self.app_dir(),
        )
    }

    /// Runs `argv` in the workspace.
    pub fn run_in_workspace(&self, argv: &[&str]) -> Result<Ran, HostError> {
        self.script(
            argv.iter().map(ToString::to_string).collect(),
            &self.app_dir(),
        )
    }

    /// Runs the verifier on the workspace as it is now.
    pub fn verify(&self) -> Result<Verdict, HostError> {
        let logs = self.logs_dir();
        let reward = logs.join(REWARD_FILE);
        if reward.exists() {
            std::fs::remove_file(&reward).map_err(|source| HostError::Io {
                path: reward.display().to_string(),
                source,
            })?;
        }
        let script = self
            .root
            .join("tests")
            .join(TEST_SCRIPT.trim_start_matches("tests/"));
        let outcome = self.script(
            vec!["sh".into(), script.display().to_string()],
            &self.app_dir(),
        )?;
        let Ok(text) = std::fs::read_to_string(&reward) else {
            return Ok(Verdict::NoReward {
                exit: outcome.describe(),
                tail: outcome.tail(),
            });
        };
        Ok(parse_reward(&text))
    }

    /// Replaces or creates a workspace file, as a solver would.
    pub fn write(&self, path: &str, bytes: &[u8]) -> Result<(), HostError> {
        let mut check = Folder::new();
        check.put(path, bytes, false)?; // refuses a path that leaves the workspace
        let target = self.app_dir().join(path);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent).map_err(|source| HostError::Io {
                path: parent.display().to_string(),
                source,
            })?;
        }
        std::fs::write(&target, bytes).map_err(|source| HostError::Io {
            path: target.display().to_string(),
            source,
        })
    }

    /// Removes a workspace file; whether it was there.
    pub fn remove(&self, path: &str) -> bool {
        std::fs::remove_file(self.app_dir().join(path)).is_ok()
    }

    /// The workspace as it is now.
    pub fn workspace(&self) -> Result<Folder, HostError> {
        Ok(Folder::read_from(&self.app_dir())?)
    }

    /// Puts the workspace back to `snapshot`: files it lacks are written,
    /// files it does not have are removed. Cheaper than opening a new
    /// session when only workspace files changed.
    pub fn restore(&self, snapshot: &Folder) -> Result<(), HostError> {
        let now = self.workspace()?;
        for path in now.paths().filter(|p| snapshot.get(p).is_none()) {
            self.remove(path);
        }
        for (path, entry) in snapshot.entries() {
            if now.get(path) != Some(entry) {
                self.write(path, &entry.bytes)?;
                if entry.executable {
                    use std::os::unix::fs::PermissionsExt;
                    let target = self.app_dir().join(path);
                    std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o755))
                        .map_err(|source| HostError::Io {
                            path: target.display().to_string(),
                            source,
                        })?;
                }
            }
        }
        Ok(())
    }

    /// The content address of the workspace as it is now.
    pub fn workspace_digest(&self) -> Result<Digest, HostError> {
        Ok(self.workspace()?.digest())
    }
}

fn parse_reward(text: &str) -> Verdict {
    let parsed: Result<BTreeMap<String, f64>, _> = serde_json::from_str(text);
    let vector = match parsed {
        Ok(vector) => vector,
        Err(e) => return Verdict::Malformed(format!("{e}")),
    };
    if !vector.contains_key("reward") {
        return Verdict::Malformed("no `reward` entry".into());
    }
    if let Some((name, value)) = vector
        .iter()
        .find(|(_, v)| !v.is_finite() || !(0.0..=1.0).contains(*v))
    {
        return Verdict::Malformed(format!("`{name}` is {value}, outside 0..=1"));
    }
    Verdict::Graded(vector)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // a panic is the failure report
mod tests {
    use super::*;
    use crate::factory::testing::{meta, parts};

    fn instance_with(test: &str, solve: &str) -> Instance {
        let mut parts = parts();
        parts.tests.put_script("test.sh", test).unwrap();
        parts.solution.put_script("solve.sh", solve).unwrap();
        Instance::assemble(meta(), parts).unwrap()
    }

    const GRADER: &str = "#!/bin/sh\nmkdir -p \"${LOGS_DIR:-/logs/verifier}\"\nif [ \"$(cat \"${APP_DIR:-/app}/answer.txt\")\" = 42 ]; then r=1; else r=0; fi\nprintf '{\"reward\": %s}' \"$r\" > \"${LOGS_DIR:-/logs/verifier}/reward.json\"\n";

    #[test]
    fn the_reference_solves_and_the_untouched_workspace_does_not() {
        let instance = instance_with(
            GRADER,
            "#!/bin/sh\necho 42 > \"${APP_DIR:-/app}/answer.txt\"\n",
        );
        let session = Host::default().open(&instance).unwrap();
        assert!(!session.verify().unwrap().solved());
        assert!(session.solve().unwrap().succeeded());
        let verdict = session.verify().unwrap();
        assert!(verdict.solved(), "{verdict:?}");
    }

    #[test]
    fn a_verifier_that_crashes_is_not_a_zero() {
        let instance = instance_with("#!/bin/sh\nexit 3\n", "#!/bin/sh\n");
        let verdict = Host::default().open(&instance).unwrap().verify().unwrap();
        assert!(matches!(verdict, Verdict::NoReward { .. }), "{verdict:?}");
        assert_eq!(verdict.reward(), None);
    }

    #[test]
    fn a_reward_outside_the_unit_interval_or_without_a_scalar_is_malformed() {
        assert!(matches!(
            parse_reward("{\"reward\": 2}"),
            Verdict::Malformed(_)
        ));
        assert!(matches!(
            parse_reward("{\"exact\": 1}"),
            Verdict::Malformed(_)
        ));
        assert!(matches!(parse_reward("[1]"), Verdict::Malformed(_)));
        assert!(matches!(
            parse_reward("{\"reward\": 0.5, \"x\": 1}"),
            Verdict::Graded(_)
        ));
        assert!(!parse_reward("{\"reward\": 0.5}").solved());
    }

    #[test]
    fn a_workspace_file_cannot_be_written_outside_the_workspace() {
        let instance = instance_with(GRADER, "#!/bin/sh\n");
        let session = Host::default().open(&instance).unwrap();
        assert!(session.write("../escape", b"x").is_err());
        assert!(session.write("sub/ok.txt", b"x").is_ok());
    }
}
