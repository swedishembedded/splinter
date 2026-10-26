// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements interchange of agent benchmark tasks between
// task formats for its clients. If your team needs expertise in importing
// containerised benchmark tasks into reproducible, verifiable task sets, you
// can procure our services by sending an email to info@swedishembedded.com.

//! Harbor tasks as workspace tasks, and back.
//!
//! A Harbor task is a directory: `instruction.md`, `task.toml`,
//! `environment/` (a Dockerfile and the files it copies), `tests/` with a
//! `test.sh` that writes a reward, and optionally `solution/solve.sh`. A
//! workspace task is the same thing in splinter's terms: a request, a tree,
//! hidden files and commands that must fail before and pass after, and,
//! sometimes, the change that solves it.
//!
//! **Import** goes through the authored-fixture path, so an imported task is
//! held to the bar every workspace task is: its check fails on the tree the
//! solver is given, on every run, the request neither quotes nor points at
//! the hidden files, and, when an oracle change exists, the verifier passes
//! it. What the host cannot honour is refused with the reason: a Dockerfile
//! with `RUN`, a prebuilt image, a Windows task, a multi-step task. A task
//! splinter itself exported carries its check in `[metadata.splinter]` and
//! is rebuilt exactly; any other task is graded by its own `test.sh`, run
//! through a runner that maps `/tests`, `/logs/verifier` and the working
//! directory to host paths and passes only on a reward of exactly 1.
//!
//! **Export** writes a task Harbor can run in a container, with the check's
//! commands in a generated `test.sh`. Harbor's verifier does not enforce the
//! protected or pinned paths of the check; splinter's does.
//!
//! **Discovery** lists the tasks under a directory and whether each is
//! importable, writing nothing: the external task repository is a versioned
//! source read at a revision, never a database splinter keeps.

mod dockerfile;
mod export;
mod script;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use splinter_core::clock::Clock;
use splinter_eval::verifiers::hidden_strength::HiddenStrength;
use splinter_eval::verifiers::test_report::TestExpectation;
use splinter_sandbox::CommandCall;

use super::files::{copy_dir, copy_file, files_under, io, read_files, refuse};
use super::{Fixture, Held, WorkspaceError, WorkspaceGeneration, WorkspaceGenerator};

/// The schema name `[metadata.splinter]` carries.
pub const SPLINTER_SCHEMA: &str = "splinter.harbor.v1";

/// The files of an environment directory that describe the build, not the
/// repository it copies.
fn is_build_file(name: &str) -> bool {
    name == "Dockerfile"
        || name.starts_with("Dockerfile.")
        || name == ".dockerignore"
        || name.starts_with("docker-compose")
}

/// What an export records so an import rebuilds the task exactly.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SplinterMeta {
    /// Always [`SPLINTER_SCHEMA`].
    pub schema: String,
    /// The task's family.
    pub family: String,
    /// The commands that must fail before and pass after, as argv.
    pub fail_to_pass: Vec<Vec<String>>,
    /// The commands that must pass before and after, as argv.
    pub pass_to_pass: Vec<Vec<String>>,
    /// Paths a change may not touch.
    pub protected: Vec<String>,
    /// Paths that must stay byte-identical.
    #[serde(default)]
    pub pinned: Vec<String>,
    /// The tests each command must be seen to run.
    #[serde(default)]
    pub expectations: Vec<TestExpectation>,
    /// What a pass is worth, when measured.
    #[serde(default)]
    pub hidden_strength: Option<HiddenStrength>,
}

/// A Harbor task directory, read and found importable.
#[derive(Clone, Debug)]
pub struct HarborTask {
    /// The task's directory.
    pub dir: PathBuf,
    /// Its name: `[task] name` of `task.toml`, else the directory's.
    pub name: String,
    /// The request.
    pub instruction: String,
    /// The working directory its environment copies the repository into.
    pub workdir: String,
    config: toml::Table,
    plan: dockerfile::Plan,
}

impl HarborTask {
    /// The task in `dir`, when the host can honour it.
    ///
    /// # Errors
    ///
    /// [`WorkspaceError::Fixture`] naming why not.
    pub fn read(dir: &Path) -> Result<Self, WorkspaceError> {
        let text = |name: &str| {
            let path = dir.join(name);
            std::fs::read_to_string(&path).map_err(io(&path))
        };
        files_under(dir)?;
        let instruction = text("instruction.md")?.trim().to_string();
        if instruction.is_empty() {
            return refuse(dir, "instruction.md is empty");
        }
        let config: toml::Table =
            text("task.toml")?
                .parse()
                .map_err(|e: toml::de::Error| WorkspaceError::Fixture {
                    path: dir.join("task.toml"),
                    reason: e.to_string(),
                })?;
        if config.get("steps").is_some() {
            return refuse(dir, "it is a multi-step task");
        }
        let table = |key: &str| config.get(key).and_then(toml::Value::as_table);
        let environment = table("environment");
        let setting = |key: &str| environment.and_then(|e| e.get(key));
        if setting("os").and_then(toml::Value::as_str) == Some("windows") {
            return refuse(dir, "it is a Windows task");
        }
        if setting("docker_image").is_some() {
            return refuse(
                dir,
                "its environment is a prebuilt image the host cannot unpack",
            );
        }
        if !dir.join("tests/test.sh").is_file() {
            return refuse(
                dir,
                "tests/test.sh is missing (only shell verifiers run on the host)",
            );
        }
        let dockerfile = dir.join("environment/Dockerfile");
        if !dockerfile.is_file() {
            return refuse(dir, "environment/Dockerfile is missing");
        }
        let plan = dockerfile::parse(
            &std::fs::read_to_string(&dockerfile).map_err(io(&dockerfile))?,
            setting("workdir").and_then(toml::Value::as_str),
        )
        .or_else(|why| refuse(dir, why))?;
        let Some(workdir) = plan.workdir.clone().filter(|w| w != "/") else {
            return refuse(
                dir,
                "the environment sets no WORKDIR to copy the repository into",
            );
        };
        if plan.copies.is_empty() {
            return refuse(dir, "the environment copies no files");
        }
        let name = table("task")
            .and_then(|t| t.get("name"))
            .and_then(toml::Value::as_str)
            .map(str::to_string)
            .or_else(|| dir.file_name().map(|n| n.to_string_lossy().into_owned()))
            .unwrap_or_default();
        Ok(Self {
            dir: dir.to_path_buf(),
            name,
            instruction,
            workdir,
            config,
            plan,
        })
    }

    /// What `[metadata.splinter]` records, when the task was exported by
    /// splinter.
    ///
    /// # Errors
    ///
    /// [`WorkspaceError::Fixture`] when it is there and is not the schema.
    pub fn splinter(&self) -> Result<Option<SplinterMeta>, WorkspaceError> {
        let Some(value) = self
            .config
            .get("metadata")
            .and_then(toml::Value::as_table)
            .and_then(|m| m.get("splinter"))
        else {
            return Ok(None);
        };
        let meta: SplinterMeta =
            value
                .clone()
                .try_into()
                .map_err(|e: toml::de::Error| WorkspaceError::Fixture {
                    path: self.dir.join("task.toml"),
                    reason: format!("[metadata.splinter] is not {SPLINTER_SCHEMA}: {e}"),
                })?;
        if meta.schema != SPLINTER_SCHEMA {
            return refuse(
                &self.dir,
                format!(
                    "[metadata.splinter] is {}, not {SPLINTER_SCHEMA}",
                    meta.schema
                ),
            );
        }
        Ok(Some(meta))
    }

    /// A string of the task's `[metadata]`.
    fn metadata_text(&self, key: &str) -> Option<&str> {
        self.config
            .get("metadata")
            .and_then(toml::Value::as_table)
            .and_then(|m| m.get(key))
            .and_then(toml::Value::as_str)
    }

    /// The repository the environment copies, laid out in `into`.
    fn lay_tree(&self, into: &Path) -> Result<(), WorkspaceError> {
        let environment = self.dir.join("environment");
        for copy in &self.plan.copies {
            let target = dockerfile::under(&self.workdir, &copy.destination)
                .or_else(|why| refuse(&self.dir, why))?;
            for source in &copy.sources {
                let from = if source == "." {
                    environment.clone()
                } else {
                    environment.join(source)
                };
                let name = from.file_name().map(PathBuf::from).unwrap_or_default();
                if from.is_dir() {
                    copy_dir(&from, &into.join(&target), is_build_file)?;
                } else if from.is_file() {
                    let to = if copy.into_directory {
                        into.join(&target).join(&name)
                    } else {
                        into.join(&target)
                    };
                    copy_file(&from, &to)?;
                } else {
                    return refuse(&self.dir, format!("COPY source {source} does not exist"));
                }
            }
        }
        Ok(())
    }
}

fn argv(words: &[String]) -> CommandCall {
    CommandCall {
        argv: words.to_vec(),
        stdin: None,
    }
}

impl Fixture {
    /// `task` as an authored fixture, its repository laid out under `work`.
    /// A task splinter exported is rebuilt exactly from its metadata; any
    /// other is graded by its own `tests/test.sh` through the runner.
    ///
    /// # Errors
    ///
    /// [`WorkspaceError`] for a task that cannot be laid out or whose
    /// metadata is not splinter's.
    pub fn harbor(task: &HarborTask, work: &Path) -> Result<Self, WorkspaceError> {
        let tree = work.join("tree");
        task.lay_tree(&tree)?;
        let tests = task.dir.join("tests");
        if let Some(meta) = task.splinter()? {
            let hidden = read_files(&tests.join("overlay"))?;
            let reference = std::fs::read_to_string(task.dir.join("solution/reference.patch")).ok();
            return Ok(Self {
                family: meta.family,
                instruction: task.instruction.clone(),
                tree,
                hidden,
                fail_to_pass: meta.fail_to_pass.iter().map(|w| argv(w)).collect(),
                pass_to_pass: meta.pass_to_pass.iter().map(|w| argv(w)).collect(),
                protected: meta.protected,
                reference,
                expectations: meta.expectations,
                pinned: meta.pinned,
                hidden_strength: meta.hidden_strength,
                process: None,
            });
        }
        let mut hidden = BTreeMap::new();
        for (file, bytes) in read_files(&tests)? {
            let bytes = match String::from_utf8(bytes) {
                Ok(text) => script::placeholders(&text, &task.workdir).into_bytes(),
                Err(binary) => binary.into_bytes(),
            };
            hidden.insert(format!("{}/tests/{file}", script::HIDDEN_DIR), bytes);
        }
        let runner = format!("{}/run.sh", script::HIDDEN_DIR);
        hidden.insert(runner.clone(), script::runner().into_bytes());
        if tree.join(script::HIDDEN_DIR).exists() {
            return refuse(
                &task.dir,
                format!("the repository has a {} directory", script::HIDDEN_DIR),
            );
        }
        Ok(Self {
            family: task
                .metadata_text("category")
                .map_or_else(|| task.name.clone(), str::to_string),
            instruction: task.instruction.clone(),
            tree,
            hidden,
            fail_to_pass: vec![argv(&["sh".to_string(), runner])],
            pass_to_pass: Vec::new(),
            protected: vec![script::HIDDEN_DIR.to_string()],
            reference: None,
            expectations: Vec::new(),
            pinned: Vec::new(),
            hidden_strength: None,
            process: None,
        })
    }
}

/// What to do about a task's `solution/solve.sh`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Oracle {
    /// Do not run it.
    Skip,
    /// Run it; a task whose solution does not run on the host is imported
    /// without a reference.
    Try,
    /// Run it; a task without a solution that runs and passes is refused.
    Require,
}

/// A task found under a directory.
#[derive(Clone, Debug, Serialize)]
pub struct Discovered {
    /// The task's directory.
    pub dir: PathBuf,
    /// Its name, when it could be read.
    pub name: Option<String>,
    /// Why it cannot be imported; `None` when it can.
    pub refusal: Option<String>,
}

/// The Harbor tasks under `root` (each a directory holding a `task.toml`),
/// in path order, with whether each can be imported. Nothing is written.
///
/// # Errors
///
/// [`WorkspaceError::Io`] when `root` cannot be read.
pub fn discover(root: &Path) -> Result<Vec<Discovered>, WorkspaceError> {
    fn walk(dir: &Path, found: &mut Vec<PathBuf>) -> Result<(), WorkspaceError> {
        if dir.join("task.toml").is_file() {
            found.push(dir.to_path_buf());
            return Ok(());
        }
        let mut children: Vec<PathBuf> = std::fs::read_dir(dir)
            .map_err(io(dir))?
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
            .map(|e| e.path())
            .filter(|p| {
                !p.file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with('.'))
            })
            .collect();
        children.sort();
        for child in children {
            walk(&child, found)?;
        }
        Ok(())
    }
    let mut dirs = Vec::new();
    walk(root, &mut dirs)?;
    Ok(dirs
        .into_iter()
        .map(|dir| match HarborTask::read(&dir) {
            Ok(task) => Discovered {
                dir,
                name: Some(task.name),
                refusal: None,
            },
            Err(error) => Discovered {
                dir,
                name: None,
                refusal: Some(error.to_string()),
            },
        })
        .collect())
}

impl WorkspaceGenerator {
    /// `task` as a workspace task: its repository staged, its check held to
    /// the bar of every workspace task and, with `oracle`, its solution run.
    ///
    /// # Errors
    ///
    /// [`WorkspaceError`] when the task cannot be laid out, or the oracle is
    /// required and does not hold.
    pub fn import_harbor(
        &self,
        task: &HarborTask,
        oracle: Oracle,
        held: &Held,
        clock: &dyn Clock,
    ) -> Result<WorkspaceGeneration, WorkspaceError> {
        let work = self.temp_dir("splinter-harbor-")?;
        let mut fixture = Fixture::harbor(task, work.path())?;
        if fixture.reference.is_none() && oracle != Oracle::Skip {
            match self.oracle_change(task, &fixture) {
                Ok(Some(patch)) => fixture.reference = Some(patch),
                Ok(None) if oracle == Oracle::Require => {
                    return refuse(&task.dir, "it has no solution/solve.sh to run");
                }
                Err(why) if oracle == Oracle::Require => return refuse(&task.dir, why),
                Ok(None) | Err(_) => {}
            }
        }
        self.import_fixture(&fixture, held, clock)
    }

    /// The change `solution/solve.sh` makes to the repository, read back as
    /// a diff; `None` when the task has no solution.
    fn oracle_change(
        &self,
        task: &HarborTask,
        fixture: &Fixture,
    ) -> Result<Option<String>, String> {
        let solution = task.dir.join("solution");
        let script_path = solution.join("solve.sh");
        if !script_path.is_file() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(&script_path)
            .map_err(|e| format!("the oracle: reading solve.sh: {e}"))?;
        let scripts = [
            (
                "solve.sh",
                script::oracle_placeholders(&text, &task.workdir),
            ),
            ("run.sh", script::oracle_runner()),
        ];
        let call = CommandCall {
            argv: vec!["sh".into(), ".oracle/run.sh".into()],
            stdin: None,
        };
        self.solution_change(&fixture.tree, &solution, &scripts, &call)
            .map(Some)
    }
}
