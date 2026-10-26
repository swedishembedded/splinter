// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agent-driven training-task generation that
// is judged by what the tasks do when run, for its clients. If your team
// needs expertise in synthetic task generation then you can procure our
// services by sending an email to info@swedishembedded.com.

//! Tasks invented by the flywheel's invent harness.
//!
//! In place of asking a generator model for the model kinds' tasks, the
//! tasks stage runs the invent harness (see [`splinter_agent::flywheel`]) once
//! per window of a source's text, on one workspace that persists across the
//! stage so a later window sees the tasks the earlier ones made. Each run's
//! new commits are task directories; each becomes a workspace task through
//! [`WorkspaceGenerator::import_task_directory`], held to the bar of every
//! workspace task: its tests fail on the tree the solver gets and pass its
//! reference solution's change. A directory that is not laid out as the
//! contract says, or whose task fails that bar, is refused with the reason
//! and counted; the stage's report carries the yield invent itself logged
//! (admitted, rejected, duplicate, failed) beside it, so an idea lost on
//! either side is visible.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Serialize;
use splinter_agent::flywheel::{
    DataFile, Invent, InventRequest, InventWorkspace, SvenLaunch, Yield,
};
use splinter_agent::CancelToken;
use splinter_core::kinds::WORKSPACE_CHANGE;
use splinter_knowledge::tasks::workspace::{
    Held, TaskDirectory, WorkspaceGenerator, WorkspacePlan, WORKSPACE_GENERATOR,
};
use splinter_knowledge::tasks::SourceText;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

use super::batch::{Batch, Provenance};
use super::{remaining, spread, MAX_WINDOWS_PER_PART, SECTIONS_PER_REQUEST};
use crate::history::generator;
use crate::practice::held::held_tasks;

/// The directory under the state root that keeps the workspaces and the
/// trajectories of every harness run.
const HARNESS_RUNS: &str = "task-harness";

/// What the tasks are for when the learner states no goal.
const DEFAULT_BRIEF: &str = "Tasks for an engineer who must master the material in data/ \
well enough to be handed real problems that use it: each task is a problem someone would \
actually meet, whose solution a program can check.";

/// The flywheel harnesses and the model server they run against: the invent
/// harness that writes the model kinds' tasks, the solve harness that
/// solves tasks that bring no process of their own, and, when asked for,
/// the validate harness that decides which passing attempts are kept. Each
/// is run as `sven harness run` against a model server that is already
/// running.
#[derive(Clone, Debug, Serialize)]
pub struct TaskHarness {
    /// The invent harness directory.
    pub invent: PathBuf,
    /// The generic solve harness (a directory holding `harness.yaml`, or the
    /// document), which solves a task that has no process of its own and is
    /// the template invent adapts a task's process from.
    pub solve: PathBuf,
    /// The validate harness directory; attempts are not validated without it.
    pub validate: Option<PathBuf>,
    /// The `sven` executable that runs them.
    pub sven: PathBuf,
    /// The OpenAI-compatible server the harnesses' agents talk to.
    pub endpoint: String,
    /// The model they ask for there, as `provider/name`.
    pub model: String,
    /// The file holding the server's API key (`{"brain": key}`), when it
    /// needs one.
    pub key_file: Option<PathBuf>,
    /// The most angles, and so new tasks, one invent run works on.
    pub angles: usize,
    /// How long one harness run may take, in seconds. A run of invent takes
    /// tens of minutes: this is the run's own deadline, never the
    /// generation request's.
    pub run_timeout_secs: u64,
}

impl TaskHarness {
    /// How the harnesses are launched, keeping their runs under `runs`.
    pub(crate) fn launch(&self, runs: PathBuf) -> SvenLaunch {
        SvenLaunch {
            sven: self.sven.clone(),
            model: self.model.clone(),
            endpoint: self.endpoint.clone(),
            key_file: self.key_file.clone(),
            first_byte_timeout: None,
            runs,
        }
    }

    /// The directory the runs and workspaces of a stage are kept under.
    pub(crate) fn runs_root(ctx: &Context) -> PathBuf {
        ctx.root().path().join(HARNESS_RUNS)
    }

    /// The generic solve harness as a document.
    ///
    /// # Errors
    ///
    /// [`OrchestratorError::Refused`] when it cannot be read.
    pub(crate) fn solve_document(&self) -> Result<String, OrchestratorError> {
        let file = if self.solve.is_dir() {
            self.solve.join("harness.yaml")
        } else {
            self.solve.clone()
        };
        std::fs::read_to_string(&file).map_err(|e| {
            OrchestratorError::Refused(format!(
                "cannot read the solve harness {}: {e}",
                file.display()
            ))
        })
    }

    /// How long one run may take.
    pub(crate) fn run_timeout(&self) -> Duration {
        Duration::from_secs(self.run_timeout_secs)
    }
}

/// A task directory not made into a task, and why.
#[derive(Clone, Debug, Serialize)]
pub struct RefusedDirectory {
    /// The task directory.
    pub directory: PathBuf,
    /// Why: the missing piece, or the bar the task did not meet.
    pub reason: String,
}

/// One invent run, as the report keeps it.
#[derive(Clone, Debug, Serialize)]
pub struct InventRunNote {
    /// The text part the window is of.
    pub part: String,
    /// The window's sections.
    pub sections: Vec<usize>,
    /// What the run logged.
    pub yielded: Yield,
    /// Task directories the run committed.
    pub committed: usize,
    /// Tasks admitted from them.
    pub admitted: usize,
    /// The status sven reported.
    pub status: Option<String>,
    /// The conclusion sven reported.
    pub conclusion: Option<String>,
    /// Why the run was killed instead of ending, when it was.
    pub killed: Option<String>,
}

/// What the invent runs of a stage yielded.
#[derive(Clone, Debug, Default, Serialize)]
pub struct InventReport {
    /// Where the stage's tasks repository (with `log.jsonl`) is kept.
    pub workspace: PathBuf,
    /// Every run, in order.
    pub runs: Vec<InventRunNote>,
    /// The outcome of every idea, summed over the runs.
    pub yielded: Yield,
    /// Task directories committed that were not made into tasks.
    pub refused: Vec<RefusedDirectory>,
}

/// The invent harness, run once per window of a stage's sources on one
/// persisting workspace, with the tasks it commits admitted into a batch.
pub(super) struct Inventor {
    invent: Invent,
    workspace: InventWorkspace,
    generator: WorkspaceGenerator,
    held: Held,
    brief: String,
    angles: usize,
    run_timeout: Duration,
    report: InventReport,
}

impl Inventor {
    pub(super) fn new(
        ctx: &Context,
        harness: &TaskHarness,
        goal: Option<&str>,
        cancel: &CancelToken,
    ) -> Result<Self, OrchestratorError> {
        let root = TaskHarness::runs_root(ctx);
        let stage = root.join(format!("stage-{}", uuid_like()));
        let workspace = InventWorkspace::create(&stage.join("workspace"))
            .map_err(|e| OrchestratorError::Refused(e.to_string()))?;
        let invent = Invent::new(
            harness.launch(stage.join("runs")),
            harness.invent.clone(),
            harness.solve.clone(),
        )
        .with_tool_dirs(ctx.environments().runtime_dirs().to_vec())
        .with_cancel(cancel.clone());
        Ok(Self {
            invent,
            workspace,
            generator: generator(ctx, WorkspacePlan::default())?,
            held: held_tasks(ctx)?,
            brief: goal
                .filter(|g| !g.trim().is_empty())
                .map_or_else(|| DEFAULT_BRIEF.to_string(), str::to_string),
            angles: harness.angles,
            run_timeout: harness.run_timeout(),
            report: InventReport {
                workspace: stage.join("workspace"),
                ..InventReport::default()
            },
        })
    }

    /// Runs invent over the windows of `text` and admits the tasks it
    /// commits into `batch`. No run starts after the batch's deadline, and
    /// none runs past it.
    pub(super) fn part(
        &mut self,
        ctx: &Context,
        text: &SourceText,
        batch: &mut Batch,
    ) -> Result<(), OrchestratorError> {
        let positions: Vec<usize> = (0..text.sections().len()).collect();
        let windows = spread(
            positions.chunks(SECTIONS_PER_REQUEST).collect(),
            Some(MAX_WINDOWS_PER_PART),
        );
        for window in windows {
            let left = remaining(batch.deadline, self.run_timeout);
            if left.is_zero() {
                batch.expired = true;
                return Ok(());
            }
            self.window(ctx, text, window, left, batch)?;
        }
        Ok(())
    }

    fn window(
        &mut self,
        ctx: &Context,
        text: &SourceText,
        window: &[usize],
        deadline: Duration,
        batch: &mut Batch,
    ) -> Result<(), OrchestratorError> {
        let name = text.part().name.clone();
        let data = [DataFile {
            name: format!(
                "{}-{}.md",
                file_stem(&name),
                window
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("-")
            ),
            text: window
                .iter()
                .filter_map(|&position| text.section_text(position))
                .collect::<Vec<_>>()
                .join("\n\n"),
        }];
        let invented = ctx
            .block_on(self.invent.run(
                &self.workspace,
                &InventRequest {
                    data: &data,
                    brief: &self.brief,
                    count: self.angles,
                    deadline,
                },
            ))
            .map_err(|e| OrchestratorError::Refused(format!("the invent harness: {e}")))?;
        let sealed = crate::sealed::families(ctx)?;
        let mut admitted = 0;
        for dir in &invented.tasks {
            match self.admit(ctx, dir, &sealed) {
                Ok(tasks) => {
                    admitted += tasks.len();
                    for (task, key) in tasks {
                        let provenance = Provenance {
                            generator: WORKSPACE_GENERATOR,
                            prompt: Some(key),
                            subject: None,
                        };
                        batch.admit(ctx, WORKSPACE_CHANGE, task, provenance)?;
                    }
                }
                Err(reason) => {
                    batch.reject(WORKSPACE_CHANGE, &reason);
                    self.report.refused.push(RefusedDirectory {
                        directory: dir.clone(),
                        reason,
                    });
                }
            }
        }
        self.report.yielded.add(invented.yielded);
        self.report.runs.push(InventRunNote {
            part: name,
            sections: window.to_vec(),
            yielded: invented.yielded,
            committed: invented.tasks.len(),
            admitted,
            status: invented.run.status,
            conclusion: invented.run.conclusion,
            killed: invented.run.killed,
        });
        Ok(())
    }

    /// The task `dir` is, with the key that identifies it across runs; the
    /// reason when it is none.
    fn admit(
        &mut self,
        ctx: &Context,
        dir: &Path,
        sealed: &[crate::sealed::SealedFamily],
    ) -> Result<
        Vec<(
            splinter_core::experience::Task,
            splinter_core::digest::Digest,
        )>,
        String,
    > {
        let directory = TaskDirectory::read(dir).map_err(|e| e.to_string())?;
        let generation = self
            .generator
            .import_task_directory(&directory, &self.held, ctx.clock())
            .map_err(|e| e.to_string())?;
        if let Some((reason, _)) = generation.refused.iter().next() {
            return Err(format!(
                "{}: {}",
                directory.name,
                super::reason_name(reason)
            ));
        }
        let mut tasks = Vec::new();
        for made in generation.admitted {
            if let Some(why) = crate::sealed::refuses_task(sealed, &made.task) {
                return Err(why);
            }
            self.held.hold(&made);
            tasks.push((made.task, made.key));
        }
        Ok(tasks)
    }

    pub(super) fn into_report(self) -> InventReport {
        self.report
    }
}

/// A file name stem for a part named `name`.
fn file_stem(name: &str) -> String {
    let stem: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    let stem = stem.trim_matches('-');
    if stem.is_empty() {
        "part".into()
    } else {
        stem.chars().take(60).collect()
    }
}

/// A name no other stage of any run has.
fn uuid_like() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    format!("{nanos:x}")
}
