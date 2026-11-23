// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning factories that take a model from
// agent rollouts to an evaluated, released candidate, for its clients. If
// your team needs expertise in model fine-tuning pipelines, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The learning factory's commands: argument handling and output only; the
//! work is `splinter_sdk::factory`'s.

use std::path::{Path, PathBuf};
use std::process::Command as Process;

use serde::Serialize;
use splinter_sdk::factory::backend::HarborWorker;
use splinter_sdk::factory::resolution::Resolved;
use splinter_sdk::factory::run::{
    self, Factory, RunError, RunManifest, Services, LOCK_FILE, REPORT_FILE,
};
use splinter_sdk::factory::training::BrainTrainer;
use splinter_sdk::factory::{resolve_recipe, resolve_run, trainer, worker};
use splinter_sdk::{Config, Error};

use crate::cli::{
    Command, EvalArgs, ExportArgs, ForkArgs, InspectArgs, PlayArgs, RecipeArgs, RunArgs,
    RunStatusArgs, VerifyArgs,
};
use crate::output::{emit, Report};
use crate::session::Exit;

fn refused(e: impl ToString) -> Error {
    Error::Refused(e.to_string())
}

/// Run `command` when it is the factory's (a factory verb, or `status`,
/// `verify` or `eval` given a run directory); hand it back otherwise.
pub fn route(
    config: &Config,
    json: bool,
    command: Command,
) -> Result<Result<Exit, Error>, Box<Command>> {
    Ok(match command {
        Command::Status(RunStatusArgs { run: Some(run) }) => status(json, &run),
        Command::Validate(args) => plan(config, json, &args, false),
        Command::Plan(args) => plan(config, json, &args, true),
        Command::Generate(args) => generate(config, &args.collection),
        Command::Run(args) => run_recipe(config, json, &args),
        Command::Resume(args) => resume(config, json, &args.run),
        Command::Cancel(args) => cancel(&args.run),
        Command::Inspect(args) => inspect(config, &args),
        Command::Fork(args) => fork(config, json, &args),
        Command::Report(args) => report(&args.run),
        Command::Export(args) => export(&args),
        Command::Play(args) => play(config, json, &args),
        Command::Verify(VerifyArgs { experience_set, .. })
            if is_run(Path::new(&experience_set)) =>
        {
            verify(json, Path::new(&experience_set))
        }
        Command::Eval(EvalArgs {
            model: Some(run),
            plan: Some(plan),
            out: Some(out),
            ..
        }) if is_run(Path::new(&run)) => eval(config, json, Path::new(&run), &plan, &out),
        Command::Eval(EvalArgs { plan: Some(_), .. }) => Err(refused(
            "--plan evaluates a factory run's candidates: give the run directory as REF",
        )),
        other => return Err(Box::new(other)),
    })
}

/// Whether `path` is a factory run directory.
#[must_use]
pub fn is_run(path: &Path) -> bool {
    path.join(LOCK_FILE).is_file()
}

/// The controller's name on its lease: this host and process.
fn owner(config: &Config) -> String {
    format!("{}:{}", config.jobs.owner, std::process::id())
}

/// What `validate` and `plan` show.
#[derive(Serialize)]
struct PlanReport<'a> {
    recipe: &'a str,
    tasks: usize,
    stages: &'a [splinter_sdk::factory::plan::PlannedStage],
}

impl Report for PlanReport<'_> {
    fn human(&self) -> String {
        if self.stages.is_empty() {
            return format!("{}: valid; {} tasks\n", self.recipe, self.tasks);
        }
        let mut out = format!(
            "{}: {} tasks, {} stages\n",
            self.recipe,
            self.tasks,
            self.stages.len()
        );
        for stage in self.stages {
            let after = if stage.depends_on.is_empty() {
                String::new()
            } else {
                format!(" (after {})", stage.depends_on.join(", "))
            };
            out.push_str(&format!("  {}{after}\n      {}\n", stage.key, stage.work));
        }
        out
    }
}

/// `validate` and `plan`.
pub fn plan(
    config: &Config,
    json: bool,
    args: &RecipeArgs,
    show_stages: bool,
) -> Result<Exit, Error> {
    let resolved = resolve_recipe(config, &args.recipe).map_err(refused)?;
    let stages = if show_stages {
        resolved.plan.stages.as_slice()
    } else {
        &[]
    };
    emit(
        json,
        &PlanReport {
            recipe: &resolved.recipe.name,
            tasks: resolved.collection.collection.tasks.len(),
            stages,
        },
    );
    Ok(Exit::Ok)
}

/// What a run printed when it ended.
#[derive(Serialize)]
struct Ran<'a> {
    run: &'a Path,
    manifest: &'a RunManifest,
}

impl Report for Ran<'_> {
    fn human(&self) -> String {
        let mut out = format!("run {} of {}\n", self.run.display(), self.manifest.recipe);
        for stage in &self.manifest.stages {
            let how = if stage.computed { "computed" } else { "reused" };
            out.push_str(&format!("  {:<28} {how}\n", stage.key));
        }
        out.push_str(&format!(
            "report: {}\n",
            self.run.join(REPORT_FILE).display()
        ));
        out
    }
}

fn services(config: &Config) -> Result<(HarborWorker, Option<BrainTrainer>), Error> {
    Ok((worker(config).map_err(refused)?, trainer(config)))
}

/// Run `resolved` into `dir`.
fn execute(config: &Config, json: bool, resolved: &Resolved, dir: &Path) -> Result<Exit, Error> {
    let needs_trainer = resolved.recipe.training.is_some() || resolved.recipe.eval.is_some();
    let (backend, brain) = services(config)?;
    if needs_trainer && brain.is_none() {
        return Err(refused("the recipe trains or evaluates; give brain with --brain-endpoint (and --brain-artifact-store, --brain-adapter-dir)"));
    }
    let services = Services {
        backend: &backend,
        trainer: brain.as_ref().map(|b| b as _),
    };
    let mut factory =
        Factory::open(resolved, dir, services, &owner(config)).map_err(|e| match e {
            RunError::InputsChanged(_) | RunError::State(_) => refused(e),
            other => refused(other),
        })?;
    match factory.run() {
        Ok(manifest) => {
            emit(
                json,
                &Ran {
                    run: dir,
                    manifest: &manifest,
                },
            );
            Ok(Exit::Ok)
        }
        Err(RunError::Cancelled) => {
            eprintln!(
                "splinter: the run was cancelled; `splinter resume {}` continues it",
                dir.display()
            );
            Ok(Exit::Failed)
        }
        Err(e) => {
            eprintln!(
                "splinter: {e}\nwhat is committed stays; `splinter resume {}` continues",
                dir.display()
            );
            Ok(Exit::Failed)
        }
    }
}

/// `run`.
pub fn run_recipe(config: &Config, json: bool, args: &RunArgs) -> Result<Exit, Error> {
    let resolved = resolve_recipe(config, &args.recipe).map_err(refused)?;
    execute(config, json, &resolved, &args.out)
}

/// `resume`.
pub fn resume(config: &Config, json: bool, dir: &Path) -> Result<Exit, Error> {
    let resolved = resolve_run(config, dir).map_err(refused)?;
    execute(config, json, &resolved, dir)
}

/// `cancel`.
pub fn cancel(dir: &Path) -> Result<Exit, Error> {
    run::request_cancel(dir).map_err(refused)?;
    eprintln!(
        "asked the controller of {} to stop at its next stage or batch",
        dir.display()
    );
    Ok(Exit::Ok)
}

/// `fork`.
pub fn fork(config: &Config, json: bool, args: &ForkArgs) -> Result<Exit, Error> {
    let resolved = resolve_recipe(config, &args.recipe).map_err(refused)?;
    let reused = run::fork(&args.run, &args.out).map_err(refused)?;
    eprintln!("reusing {reused} products of {}", args.run.display());
    execute(config, json, &resolved, &args.out)
}

/// `eval RUN --plan PLAN --out DIR`: the run's recipe with its evaluation
/// plan replaced, forked from the run.
pub fn eval(
    config: &Config,
    json: bool,
    dir: &Path,
    plan: &Path,
    out: &Path,
) -> Result<Exit, Error> {
    let resolved =
        splinter_sdk::factory::resolve_run_under_plan(config, dir, plan).map_err(refused)?;
    let reused = run::fork(dir, out).map_err(refused)?;
    eprintln!("reusing {reused} products of {}", dir.display());
    execute(config, json, &resolved, out)
}

/// A run's stages as the run state records them.
#[derive(Serialize)]
struct StatusReport(Vec<run::StageStatus>);

impl Report for StatusReport {
    fn human(&self) -> String {
        let mut out = String::new();
        for stage in &self.0 {
            out.push_str(&format!("{:<30} {}", stage.key, stage.state));
            if !stage.slots.is_empty() {
                let slots: Vec<String> = stage
                    .slots
                    .iter()
                    .map(|(state, n)| format!("{n} {state}"))
                    .collect();
                out.push_str(&format!("  slots: {}", slots.join(", ")));
            }
            if let Some(error) = &stage.error {
                out.push_str(&format!("\n    {error}"));
            }
            out.push('\n');
        }
        out
    }
}

/// `status RUN`.
pub fn status(json: bool, dir: &Path) -> Result<Exit, Error> {
    emit(json, &StatusReport(run::status(dir).map_err(refused)?));
    Ok(Exit::Ok)
}

/// What `verify` found.
#[derive(Serialize)]
struct Verified {
    problems: Vec<String>,
}

impl Report for Verified {
    fn human(&self) -> String {
        if self.problems.is_empty() {
            "every product matches its blobs and its view\n".into()
        } else {
            self.problems.iter().map(|p| format!("{p}\n")).collect()
        }
    }
}

/// `verify RUN`.
pub fn verify(json: bool, dir: &Path) -> Result<Exit, Error> {
    let problems = run::verify(dir).map_err(refused)?;
    let failed = !problems.is_empty();
    emit(json, &Verified { problems });
    Ok(if failed { Exit::Failed } else { Exit::Ok })
}

/// `report RUN`.
pub fn report(dir: &Path) -> Result<Exit, Error> {
    let path = dir.join(REPORT_FILE);
    let text = std::fs::read_to_string(&path).map_err(|e| {
        refused(format!(
            "{}: {e} (a run writes its report when it finishes)",
            path.display()
        ))
    })?;
    print!("{text}");
    Ok(Exit::Ok)
}

/// `inspect`.
pub fn inspect(config: &Config, args: &InspectArgs) -> Result<Exit, Error> {
    let stages: Vec<run::StageStatus> = run::status(&args.run)
        .map_err(refused)?
        .into_iter()
        .filter(|s| {
            args.stage
                .as_ref()
                .is_none_or(|prefix| s.key.starts_with(prefix.as_str()))
        })
        .collect();
    if let Some(viewer) = &args.viewer {
        if viewer != "harbor" {
            return Err(refused(format!("viewer {viewer} is not one of: harbor")));
        }
        return harbor_view(config, &args.run, &stages);
    }
    for stage in &stages {
        println!("== {} ({})", stage.key, stage.state);
        let report = args.run.join(&stage.key).join("report.md");
        if let Ok(text) = std::fs::read_to_string(&report) {
            println!("{text}");
        }
    }
    Ok(Exit::Ok)
}

/// Open the native trials of the completed `stages` in Harbor's viewer.
fn harbor_view(config: &Config, dir: &Path, stages: &[run::StageStatus]) -> Result<Exit, Error> {
    let mut jobs: Vec<PathBuf> = Vec::new();
    for stage in stages.iter().filter(|s| s.state == "completed") {
        let batches = dir.join(&stage.key).join("batches");
        for entry in std::fs::read_dir(&batches).into_iter().flatten().flatten() {
            let harbor = entry.path().join("harbor");
            if harbor.is_dir() {
                jobs.push(harbor);
            }
        }
    }
    let Some(first) = jobs.first() else {
        return Err(refused("no completed stage of those holds Harbor trials"));
    };
    // Harbor's viewer is installed beside the worker.
    let program = config
        .factory
        .harbor_worker
        .as_ref()
        .and_then(|w| w.parent().map(|d| d.join("harbor")))
        .filter(|p| p.is_file())
        .unwrap_or_else(|| PathBuf::from("harbor"));
    if jobs.len() > 1 {
        eprintln!(
            "{} batches; opening {} (the others: {:?})",
            jobs.len(),
            first.display(),
            &jobs[1..]
        );
    }
    let status = Process::new(&program)
        .arg("view")
        .arg(first)
        .arg("--jobs")
        .status()
        .map_err(refused)?;
    Ok(if status.success() {
        Exit::Ok
    } else {
        Exit::Failed
    })
}

/// `export`: the run's products, views, state and record as one
/// `tar.zst`; the unfinished work and staging areas are left out.
pub fn export(args: &ExportArgs) -> Result<Exit, Error> {
    if !is_run(&args.run) {
        return Err(refused(format!(
            "{} is not a factory run",
            args.run.display()
        )));
    }
    let status = Process::new("tar")
        .arg("--zstd")
        .arg("-cf")
        .arg(&args.out)
        .arg("--exclude=./work")
        .arg("--exclude=./staging")
        .arg("--exclude=./pending")
        .arg("-C")
        .arg(&args.run)
        .arg(".")
        .status()
        .map_err(|e| refused(format!("tar: {e}")))?;
    if !status.success() {
        return Ok(Exit::Failed);
    }
    eprintln!(
        "wrote {}; `splinter verify` checks it once unpacked",
        args.out.display()
    );
    Ok(Exit::Ok)
}

/// `play`: one session of the run's interactive pass on this terminal, then
/// the run continued so the session reaches what depends on the pass.
pub fn play(config: &Config, json: bool, args: &PlayArgs) -> Result<Exit, Error> {
    let resolved = resolve_run(config, &args.run).map_err(refused)?;
    let (backend, brain) = services(config)?;
    let services = Services {
        backend: &backend,
        trainer: brain.as_ref().map(|b| b as _),
    };
    let trial = {
        let mut factory =
            Factory::open(&resolved, &args.run, services, &owner(config)).map_err(refused)?;
        factory
            .play(&args.pass, &args.task, args.continue_from)
            .map_err(refused)?
    };
    eprintln!(
        "session {} of {}: {:?}, {}",
        trial.slot.slot,
        args.pass,
        trial.grade.status,
        if trial.verified_success() {
            "a verified success"
        } else {
            "not a success"
        }
    );
    if args.no_run {
        return Ok(Exit::Ok);
    }
    execute(config, json, &resolved, &args.run)
}

/// `generate`.
pub fn generate(config: &Config, collection: &Path) -> Result<Exit, Error> {
    splinter_sdk::factory::collection::generate(&config.working_dir, &collection.to_string_lossy())
        .map_err(refused)?;
    Ok(Exit::Ok)
}
