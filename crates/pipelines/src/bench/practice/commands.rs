// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that turn an examiner's practice
// material into graded agent experience for its clients. If your team needs
// expertise in on-policy experience collection for agent training you can
// procure our services by sending an email to info@swedishembedded.com.

//! The examiner's commands a practice run is made of, each run from its
//! checkout: its model gateway for the length of the run, the staging of
//! the material under the policy's profile, Harbor running the staged job,
//! and the export of the graded trials.

use std::fs::File;
use std::io::{BufRead, BufReader, Write as _};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

use serde::Deserialize;

use super::material::Practice;

/// The line the gateway prints once it serves.
const SERVING: &str = "bench gateway: serving";

/// The environment variable the policy profile names its upstream key by,
/// set for the gateway's process only.
pub(crate) const KEY_VARIABLE: &str = "BRAIN_KEY";

/// The examiner's checkout and the Harbor that runs its jobs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BenchCheckout {
    /// The checkout; `./bench` is run from it.
    pub dir: PathBuf,
    /// The `harbor` program that imports the examiner's sandboxes and agent
    /// adapters (the checkout's `.venv/bin/harbor` unless told otherwise).
    pub harbor: PathBuf,
    /// The sven build the profile's agent runs, given to the examiner as
    /// `BENCH_SVEN_BUILD`; `None` leaves the environment's.
    pub sven_build: Option<PathBuf>,
}

impl BenchCheckout {
    /// The checkout at `dir`, running its own Harbor.
    #[must_use]
    pub fn new(dir: PathBuf) -> Self {
        Self {
            harbor: dir.join(".venv").join("bin").join("harbor"),
            dir,
            sven_build: None,
        }
    }

    pub(crate) fn bench(&self) -> Command {
        let mut command = Command::new(self.dir.join("bench"));
        command.current_dir(&self.dir);
        if let Some(build) = &self.sven_build {
            command.env("BENCH_SVEN_BUILD", build);
        }
        command
    }
}

/// A command's output, kept in `log`: stdout and stderr appended.
fn logged(command: &mut Command, what: &str, log: &Path) -> Result<(), String> {
    let file = File::options()
        .create(true)
        .append(true)
        .open(log)
        .map_err(|e| format!("{}: {e}", log.display()))?;
    let copy = file
        .try_clone()
        .map_err(|e| format!("{}: {e}", log.display()))?;
    let status = command
        .stdout(file)
        .stderr(copy)
        .status()
        .map_err(|e| format!("starting the examiner to {what}: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "the examiner could not {what} ({status}); its output is in {}",
            log.display()
        ))
    }
}

/// The examiner's model gateway, serving the policy's profile until it is
/// dropped.
pub(super) struct Gateway {
    child: Child,
}

impl Gateway {
    /// Starts `bench gateway serve --profile <profile> --port <port>` with the
    /// upstream key in its environment, and returns once it says it serves;
    /// everything it prints goes to `log`.
    pub(super) fn start(
        checkout: &BenchCheckout,
        profile: &Path,
        port: u16,
        key: &str,
        log: &Path,
    ) -> Result<Self, String> {
        let mut command = checkout.bench();
        command
            .args(["gateway", "serve", "--profile"])
            .arg(profile)
            .args(["--port", &port.to_string()])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        if !key.is_empty() {
            command.env(KEY_VARIABLE, key);
        }
        let mut child = command
            .spawn()
            .map_err(|e| format!("starting the examiner's gateway: {e}"))?;
        let stderr = child.stderr.take().ok_or("the gateway's stderr")?;
        let mut file = File::create(log).map_err(|e| format!("{}: {e}", log.display()))?;
        let (ready, serving) = std::sync::mpsc::channel();
        // The gateway logs as long as it runs: its stderr is copied to the
        // log until it exits, and the line that says it serves is passed on.
        std::thread::spawn(move || {
            let mut said = false;
            for line in BufReader::new(stderr).lines() {
                let Ok(line) = line else { break };
                let _ = writeln!(file, "{line}");
                if !said && line.starts_with(SERVING) {
                    said = true;
                    let _ = ready.send(());
                }
            }
        });
        let mut gateway = Self { child };
        if serving.recv().is_err() {
            let status = gateway.child.wait().map_err(|e| e.to_string())?;
            return Err(format!(
                "the examiner's gateway stopped before it served ({status}); its output is in {}",
                log.display()
            ));
        }
        Ok(gateway)
    }
}

impl Drop for Gateway {
    /// Stops the gateway as its owner does (SIGTERM), so it removes its
    /// state file, and waits for it.
    fn drop(&mut self) {
        let stopped = Command::new("kill")
            .args(["-TERM", &self.child.id().to_string()])
            .status()
            .is_ok_and(|s| s.success());
        if !stopped {
            let _ = self.child.kill();
        }
        let _ = self.child.wait();
    }
}

/// Stages `practice`'s collection again under `profile` into `out`
/// (`bench task practice`): the committed train side, or the same draw,
/// with the reference-knowledge collection `knowledge` mounted when given.
pub(super) fn stage(
    checkout: &BenchCheckout,
    practice: &Practice,
    profile: &Path,
    knowledge: Option<&Path>,
    out: &Path,
    log: &Path,
) -> Result<(), String> {
    let mut command = checkout.bench();
    command
        .args(["task", "practice", &practice.manifest.collection])
        .args(["--environment", &practice.manifest.environment])
        .arg("--profile")
        .arg(profile)
        .arg("--out")
        .arg(out);
    if let Some(draw) = practice.draw().map_err(|e| e.to_string())? {
        command
            .args(["--count", &draw.count.to_string()])
            .args(["--seed", &draw.seed.to_string()]);
        narrow(&mut command, draw.rung, &draw.families);
    }
    for capability in practice
        .selected_capabilities()
        .map_err(|e| e.to_string())?
    {
        command.args(["--capability", &capability]);
    }
    for (name, path) in &practice.tools {
        command.arg("--tool").arg(format!("{name}={path}"));
    }
    if let Some(knowledge) = knowledge {
        command.arg("--knowledge").arg(knowledge);
    }
    logged(&mut command, "stage the practice material", log)
}

/// Narrows a draw to `rung` and `families` (`--rung`, `--family`), as the
/// examiner records them in practice.json.
fn narrow<R>(command: &mut Command, rung: Option<R>, families: &[String])
where
    R: std::fmt::Display,
{
    if let Some(rung) = rung {
        command.args(["--rung", &rung.to_string()]);
    }
    for family in families {
        command.args(["--family", family]);
    }
}

/// A fresh draw of practice material from a collection's generator, kept
/// to the tasks that exercise one of the given capabilities.
#[derive(Clone, Copy, Debug)]
pub struct PracticeDraw<'a> {
    /// The collection, as the examiner names it (`tasks/legal`).
    pub collection: &'a str,
    /// The sandbox the material is staged for (`userns`).
    pub environment: &'a str,
    /// Tasks drawn before the selection.
    pub count: usize,
    /// The draw's seed.
    pub seed: u64,
    /// The rung of the generator's difficulty ladder to draw at; the
    /// generator's own when absent.
    pub rung: Option<i64>,
    /// The families to draw from; all that respond to the rung when empty.
    pub families: &'a [String],
    /// The capability ids a kept task exercises one of; none keeps every
    /// task drawn.
    pub capabilities: &'a [String],
}

impl BenchCheckout {
    /// Draws `draw` into `out` (`bench task practice --count --seed
    /// --capability`), the examiner logging to `log`, and returns the
    /// directory it wrote (`<out>/<collection's last part>`). The examiner
    /// refuses sealed material, withheld lineages, an unknown capability and
    /// a selection that keeps nothing.
    ///
    /// # Errors
    /// The examiner refused or failed; the message names its log.
    pub fn draw_practice(
        &self,
        draw: &PracticeDraw<'_>,
        out: &Path,
        log: &Path,
    ) -> Result<PathBuf, String> {
        let mut command = self.bench();
        command
            .args(["task", "practice", draw.collection])
            .args(["--environment", draw.environment])
            .args(["--count", &draw.count.to_string()])
            .args(["--seed", &draw.seed.to_string()]);
        narrow(&mut command, draw.rung, draw.families);
        command.arg("--out").arg(out);
        for capability in draw.capabilities {
            command.args(["--capability", capability]);
        }
        logged(&mut command, "draw practice material", log)?;
        let name = draw
            .collection
            .rsplit('/')
            .find(|part| !part.is_empty())
            .unwrap_or(draw.collection);
        Ok(out.join(name))
    }
}

/// How a Harbor job is run: under which name, how many attempts of each
/// task, how many at once, and which tasks (`None` for all).
pub(super) struct HarborRun<'a> {
    pub(super) config: &'a Path,
    /// The staged task folders' directory, the dataset the config names.
    pub(super) tasks_dir: &'a Path,
    pub(super) job_name: &'a str,
    pub(super) attempts: u32,
    pub(super) jobs: usize,
    pub(super) tasks: Option<&'a [String]>,
}

/// Runs (or resumes) the staged job `run.config` as Harbor job
/// `run.job_name`.
pub(super) fn harbor(
    checkout: &BenchCheckout,
    run: &HarborRun<'_>,
    log: &Path,
) -> Result<(), String> {
    let mut command = Command::new(&checkout.harbor);
    command
        .current_dir(&checkout.dir)
        .args(["run", "-c"])
        .arg(run.config)
        .args(["--job-name", run.job_name])
        .args(["--n-attempts", &run.attempts.to_string()])
        .args(["--n-concurrent", &run.jobs.to_string()]);
    if let Some(tasks) = run.tasks {
        // Harbor filters by name only over a dataset named on its command
        // line: the one the config names.
        command.arg("--path").arg(run.tasks_dir);
        for task in tasks {
            command.args(["--include-task-name", task]);
        }
    }
    logged(&mut command, "run the practice job in Harbor", log)
}

/// A trial the export did not write, and why.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub(super) struct Skipped {
    pub(super) trial: String,
    pub(super) reason: String,
}

#[derive(Deserialize)]
struct ExportSummary {
    #[serde(default)]
    skipped: Vec<Skipped>,
}

#[derive(Deserialize)]
struct ExportResult {
    ok: bool,
    summary: ExportSummary,
}

/// Exports the graded trials of the staged job directory `job` into `out`
/// (`bench export`, idempotent) and returns the trials it did not export.
pub(super) fn export(
    checkout: &BenchCheckout,
    job: &Path,
    out: &Path,
    log: &Path,
) -> Result<Vec<Skipped>, String> {
    let output = checkout
        .bench()
        .args(["--output", "json", "export"])
        .arg(job)
        .arg("--out")
        .arg(out)
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("starting the examiner to export the trials: {e}"))?;
    let _ = std::fs::write(log, &output.stderr);
    if !output.status.success() {
        return Err(format!(
            "the examiner could not export the trials ({}); its output is in {}",
            output.status,
            log.display()
        ));
    }
    let result: ExportResult = serde_json::from_slice(&output.stdout)
        .map_err(|e| format!("the examiner's export report is not one this reader knows: {e}"))?;
    if !result.ok {
        return Err("the examiner reported its export as not ok".into());
    }
    Ok(result.summary.skipped)
}
