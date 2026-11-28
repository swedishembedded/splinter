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
use splinter_sdk::factory::backend::{ExecutionBackend, NoWorker};
use splinter_sdk::factory::resolution::Resolved;
use splinter_sdk::factory::run::{self, Factory, RunError, RunManifest, Services, REPORT_FILE};
use splinter_sdk::factory::templates;
use splinter_sdk::factory::training::BrainTrainer;
use splinter_sdk::factory::{
    host_variable, resolve_recipe, resolve_run, trainer, worker, HostModels,
};
use splinter_sdk::{Config, Error};

use crate::cli::{
    Command, EvalArgs, ExportArgs, ForkArgs, InitArgs, InspectArgs, ObserveArgs, PlayArgs,
    ProbeArgs, RecipeArgs, RunArgs, RunStatusArgs, VerifyArgs,
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
        Command::Init(args) => init(json, &args),
        Command::Templates => {
            emit(json, &Templates(templates::TEMPLATES));
            Ok(Exit::Ok)
        }
        Command::Validate(args) => plan(config, json, &args, false),
        Command::Plan(args) => plan(config, json, &args, true),
        Command::Generate(args) => generate(config, &args.collection),
        Command::Run(args) => run_recipe(config, json, &args),
        Command::Resume(args) => resume(config, json, &args.run),
        Command::Cancel(args) => cancel(&args.run),
        Command::Inspect(args) => inspect(config, &args),
        Command::View(args) => crate::view_session::view(&args),
        Command::Fork(args) => fork(config, json, &args),
        Command::Observe(args) => observe(json, &args),
        Command::Probe(args) => probe(config, json, &args),
        Command::Report(args) => report(&args.run),
        Command::Export(args) => export(&args),
        Command::Play(args) => play(config, json, &args),
        Command::Verify(VerifyArgs { run }) => verify(json, &run),
        Command::Eval(EvalArgs {
            run: Some(run),
            plan: Some(plan),
            out: Some(out),
            ..
        }) => eval(config, json, &run, &plan, &out),
        Command::Eval(EvalArgs { run: Some(_), .. }) => Err(refused(
            "a factory run is evaluated under another plan: give --plan PLAN --out DIR",
        )),
        other => return Err(Box::new(other)),
    })
}

/// What `templates` shows.
struct Templates(&'static [templates::Template]);

impl Serialize for Templates {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let mut seq = serializer.serialize_seq(Some(self.0.len()))?;
        for t in self.0 {
            seq.serialize_element(&serde_json::json!({ "name": t.name, "summary": t.summary }))?;
        }
        seq.end()
    }
}

impl Report for Templates {
    fn human(&self) -> String {
        self.0
            .iter()
            .map(|t| format!("{:<20} {}\n", t.name, t.summary))
            .collect()
    }
}

/// What `init` wrote.
#[derive(Serialize)]
struct Initialised {
    template: String,
    dir: PathBuf,
    files: Vec<PathBuf>,
}

impl Report for Initialised {
    fn human(&self) -> String {
        let recipes: Vec<String> = self
            .files
            .iter()
            .filter_map(|f| f.strip_prefix(&self.dir).ok())
            .filter(|f| f.starts_with("recipes"))
            .map(|f| format!("  splinter validate {}\n", f.display()))
            .collect();
        format!(
            "{}: wrote {} files of the {} template; its models are reached through the \
             connections of connections.yaml there or the machine's `splinter config`. \
             From there:\n{}",
            self.dir.display(),
            self.files.len(),
            self.template,
            recipes.concat()
        )
    }
}

fn init(json: bool, args: &InitArgs) -> Result<Exit, Error> {
    let files = templates::init(&args.template, &args.dir).map_err(refused)?;
    emit(
        json,
        &Initialised {
            template: args.template.clone(),
            dir: args.dir.clone(),
            files,
        },
    );
    Ok(Exit::Ok)
}

/// The controller's name on its lease: this host and process
/// ([`run::controller_owner`]).
fn owner(config: &Config) -> String {
    run::controller_owner(&config.jobs.owner)
}

/// What `validate` and `plan` show.
#[derive(Serialize)]
struct PlanReport<'a> {
    recipe: &'a str,
    tasks: usize,
    /// The pins whose tasks the run compiles: counted when it runs.
    compiled: usize,
    stages: &'a [splinter_sdk::factory::plan::PlannedStage],
}

impl Report for PlanReport<'_> {
    fn human(&self) -> String {
        let tasks = match (self.tasks, self.compiled) {
            (_, 0) => format!("{} tasks", self.tasks),
            (0, _) => "the tasks the run compiles".to_owned(),
            (n, _) => format!("{n} tasks and the tasks the run compiles"),
        };
        if self.stages.is_empty() {
            return format!("{}: valid; {tasks}\n", self.recipe);
        }
        let mut out = format!("{}: {tasks}, {} stages\n", self.recipe, self.stages.len());
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
            recipe: resolved.name(),
            tasks: resolved
                .collections
                .values()
                .map(|c| c.collection.tasks.len())
                .sum(),
            compiled: resolved
                .plan
                .stages
                .iter()
                .filter(|s| s.operation == splinter_sdk::factory::ops::names::PIN)
                .filter(|s| !resolved.collections.contains_key(&s.key))
                .count(),
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

/// The execution worker (or, on a machine that offers none, a backend that
/// refuses with why, for a recipe that executes no attempts) and the trainer.
fn services(config: &Config) -> (Box<dyn ExecutionBackend>, Option<BrainTrainer>) {
    let backend: Box<dyn ExecutionBackend> = match worker(config) {
        Ok(worker) => Box::new(worker),
        Err(why) => Box::new(NoWorker(why.to_string())),
    };
    (backend, trainer(config))
}

/// Run `resolved` into `dir`.
fn execute(config: &Config, json: bool, resolved: &Resolved, dir: &Path) -> Result<Exit, Error> {
    match carry_out(config, resolved, dir)? {
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

/// Carries out `resolved` in the run directory `dir` with this machine's
/// worker and trainer: the run's manifest, or why it stopped.
fn carry_out(
    config: &Config,
    resolved: &Resolved,
    dir: &Path,
) -> Result<Result<RunManifest, RunError>, Error> {
    // Training, evaluating and serving a champion go through brain.
    let needs_trainer = !resolved.champions.is_empty()
        || resolved.plan.stages.iter().any(|s| {
            s.operation == splinter_sdk::factory::ops::names::TRAIN
                || s.operation == splinter_sdk::factory::ops::names::EVALUATION
        });
    let online = resolved
        .plan
        .stages
        .iter()
        .any(|s| s.operation == splinter_sdk::factory::run::online::ONLINE);
    let (backend, brain) = services(config);
    if (needs_trainer || online) && brain.is_none() {
        return Err(refused(
            "the recipe trains or evaluates; give brain with --brain-endpoint (and --brain-artifact-store, --brain-adapter-dir)",
        ));
    }
    // An online stage's rollouts run here, as sven sessions against the
    // version its learner serves.
    let context = online
        .then(|| splinter_sdk::Context::new(config.clone(), false))
        .transpose()?;
    let rollouts = context
        .as_ref()
        .and_then(|ctx| splinter_sdk::factory::online_rollouts(config, ctx));
    // The servers an online run's watchdog evaluates on, each with the in-process rollouts that reach it.
    let eval_rollouts: Vec<_> = match (&context, &config.brain_eval) {
        (Some(ctx), Some(eval)) => eval
            .urls()
            .iter()
            .filter_map(|url| {
                let mut reaching = config.clone();
                reaching.brain = eval.clone().reaching_only(url);
                splinter_sdk::factory::online_rollouts(&reaching, ctx)
            })
            .collect(),
        _ => Vec::new(),
    };
    let evaluation = config
        .brain_eval
        .clone()
        .filter(|_| online)
        .map(|endpoints| run::EvaluationServers {
            endpoints,
            rollouts: eval_rollouts.iter().map(|r| r as _).collect(),
        });
    let services = Services {
        backend: backend.as_ref(),
        trainer: brain.as_ref().map(|b| b as _),
        models: Some(&HostModels),
        variables: Some(&host_variable),
        online: rollouts.as_ref().map(|r| r as _),
        build: &config.build,
    };
    let mut factory =
        Factory::open(resolved, dir, services, &owner(config)).map_err(|e| match e {
            RunError::InputsChanged(_) | RunError::State(_) => refused(e),
            other => refused(other),
        })?;
    if let Some(bound) = config.factory.concurrent_stages {
        factory = factory.with_concurrent_stages(bound);
    }
    if let Some(servers) = &evaluation {
        factory = factory.with_evaluation_servers(servers);
    }
    Ok(factory.run())
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
    let sources: Vec<&Path> = args.runs.iter().map(PathBuf::as_path).collect();
    let forked = run::fork_from(&sources, &args.out).map_err(refused)?;
    report_fork(&forked, &sources);
    execute(config, json, &resolved, &args.out)
}

/// Says what a fork brought over from each source; copying is said, as it
/// is the slow path, and so is a stage two sources computed apart, as what
/// is downstream of it in the later one is not reused.
fn report_fork(forked: &run::Forked, sources: &[&Path]) {
    for (source, products) in sources.iter().zip(&forked.per_source) {
        eprintln!("reusing {products} products of {}", source.display());
    }
    eprintln!("{} blobs linked", forked.blobs_linked);
    if forked.blobs_copied > 0 {
        eprintln!(
            "warning: {} blobs could not be linked and were copied (is the output on another file system than its sources?)",
            forked.blobs_copied,
        );
    }
    for (stage, source) in &forked.conflicts {
        eprintln!(
            "warning: {} holds another product of stage {stage} than the one reused; the stages it computed downstream of {stage} are not reused (fork the sources from a common run so they share it)",
            source.display()
        );
    }
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
    let forked = run::fork(dir, out).map_err(refused)?;
    report_fork(&forked, &[dir]);
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

/// `observe RUN [RUN ...]`.
pub fn observe(json: bool, args: &ObserveArgs) -> Result<Exit, Error> {
    use splinter_sdk::factory::observe::{self, system::Utilisation, Options};
    let utilisation = match &args.utilisation {
        Some(path) => {
            let text = std::fs::read_to_string(path)
                .map_err(|e| refused(format!("{}: {e}", path.display())))?;
            Some(
                Utilisation::parse(&text, args.utilisation_utc_offset)
                    .map_err(|e| refused(format!("{}: {e}", path.display())))?,
            )
        }
        None => None,
    };
    let options = Options {
        seed: splinter_sdk::seed::resolve("seed", args.seed),
        utilisation,
    };
    let runs = args
        .runs
        .iter()
        .map(|run| observe::observe(run, &options).map_err(refused))
        .collect::<Result<Vec<_>, _>>()?;
    emit(json, &crate::observe_output::Observed(runs));
    Ok(Exit::Ok)
}

/// `probe RUN --knob NAME=VALUE --out DIR`: the run's recipe with one knob
/// set, forked from the run and carried out, both observed and compared.
/// Exits 1 when the knob did not move its variable (a defect of the knob).
pub fn probe(config: &Config, json: bool, args: &ProbeArgs) -> Result<Exit, Error> {
    use splinter_sdk::factory::observe::{probe, Options};
    let (knob, value) = splinter_sdk::factory::knobs::assignment(&args.knob).map_err(refused)?;
    let options = Options {
        seed: splinter_sdk::seed::resolve("seed", args.seed),
        utilisation: None,
    };
    let report = probe::probe(
        &args.run,
        &args.out,
        &knob,
        &value,
        &options,
        |repository, text| {
            splinter_sdk::factory::resolve_text(config, repository, text).map_err(|e| e.to_string())
        },
        |resolved, out| match carry_out(config, resolved, out) {
            Ok(Ok(_)) => Ok(()),
            Ok(Err(e)) => Err(format!(
                "the probe run stopped: {e}; `splinter resume {}` continues it",
                out.display()
            )),
            Err(e) => Err(e.to_string()),
        },
    )
    .map_err(refused)?;
    let defect = report.defect;
    emit(json, &crate::observe_output::Probed(report));
    Ok(if defect { Exit::Failed } else { Exit::Ok })
}

/// `report RUN`.
pub fn report(dir: &Path) -> Result<Exit, Error> {
    run::require_run(dir).map_err(refused)?;
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
    let all = run::status(&args.run).map_err(refused)?;
    let keys: Vec<String> = all.iter().map(|s| s.key.clone()).collect();
    let stages: Vec<run::StageStatus> = all
        .into_iter()
        .filter(|s| {
            args.stage
                .as_ref()
                .is_none_or(|prefix| s.key.starts_with(prefix.as_str()))
        })
        .collect();
    if let (Some(prefix), true) = (&args.stage, stages.is_empty()) {
        return Err(refused(format!(
            "no stage of {} begins with {prefix:?}; its stages: {}",
            args.run.display(),
            keys.join(", ")
        )));
    }
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
    run::require_run(&args.run).map_err(refused)?;
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
    // A played session runs on the execution worker.
    let backend = worker(config).map_err(refused)?;
    let brain = trainer(config);
    let services = Services {
        backend: &backend,
        trainer: brain.as_ref().map(|b| b as _),
        models: Some(&HostModels),
        variables: Some(&host_variable),
        online: None,
        build: &config.build,
    };
    let trial = {
        let mut factory =
            Factory::open(&resolved, &args.run, services, &owner(config)).map_err(refused)?;
        factory
            .play(&args.stage, &args.task, args.continue_from)
            .map_err(refused)?
    };
    eprintln!(
        "session {} of {}: {:?}, {}",
        trial.slot.slot,
        args.stage,
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

/// What `index` filed.
#[derive(Serialize)]
pub struct Indexed(pub Vec<splinter_sdk::factory_index::IndexedPass>);

impl Report for Indexed {
    fn human(&self) -> String {
        self.0
            .iter()
            .map(|p| {
                let set = p
                    .experience_set
                    .as_ref()
                    .map_or_else(|| "no set".to_owned(), ToString::to_string);
                format!(
                    "{:<28} {set} {:?}, {} skipped\n",
                    p.pass,
                    p.verdicts,
                    p.skipped.len()
                )
            })
            .collect()
    }
}
