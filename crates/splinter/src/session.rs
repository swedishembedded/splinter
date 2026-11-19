// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One invocation's work: a command run against the controller, a sentence
//! through the front door, or a REPL of sentences - each line handled
//! exactly like `splinter "<line>"`, on one context, so the policy loads
//! once.

use std::io::{BufRead, IsTerminal, Write};
use std::sync::Arc;

use clap::Parser;
use serde_json::json;
use splinter_sdk::ask::{ask, ask_retrieving, RetrievingQuestion};
use splinter_sdk::canaries;
use splinter_sdk::critique::{critique_set, CritiqueRequest};
use splinter_sdk::doctor::doctor;
use splinter_sdk::eval::{evaluate, EvalRequest};
use splinter_sdk::exam::{examine, ExamineRequest};
use splinter_sdk::experiences::{self, resolve_set};
use splinter_sdk::experiment;
use splinter_sdk::judge::{calibrate_judge, measure_judge};
use splinter_sdk::lineage::{lineage, LineageRequest};
use splinter_sdk::measure::gate::GateConfig;
use splinter_sdk::rehearsal::{rehearse, RehearseRequest, DEFAULT_REHEARSAL_SHARE};
use splinter_sdk::release::{self, ReleaseRequest};
use splinter_sdk::rerank::ModelReranker;
use splinter_sdk::retrieval::{candidates_for, library_of, Rerank, Retrieval};
use splinter_sdk::router::{interpret, Routed};
use splinter_sdk::runs::{self, record};
use splinter_sdk::sources::{self, SourceTarget};
use splinter_sdk::state;
use splinter_sdk::status::status;
use splinter_sdk::tasks::resolve_set as resolve_task_set;
use splinter_sdk::train::{train_recorded, Rehearse, TrainRequest, Tuning};
use splinter_sdk::verify::{verify_set, Grading, Judge, Judging};
use splinter_sdk::vocabulary::build::BuildInfo;
use splinter_sdk::vocabulary::model_ref::ModelRef;
use splinter_sdk::vocabulary::terms::Distribution;
use splinter_sdk::{Config, Context, Error, Splinter};

use crate::absorb_session;
use crate::baseline_report;
use crate::cli::{
    Cli, Command, DatasetCommand, EvalArgs, EvalCommand, ExperiencesCommand, ExperimentCommand,
    Global, JudgeCommand, ReleaseCommand, RetrieveArgs, RunsCommand, SourceCommand, StateCommand,
};
use crate::compare_session;
use crate::conversation_dataset;
use crate::exam_session;
use crate::harness_session;
use crate::learn_output;
use crate::output::{self, emit, shell_words};
use crate::practice_request::practice_request;
use crate::solve_session;
use crate::tasks_session;
use crate::train_output::Trained;

/// How an invocation ended; see the crate documentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Exit {
    /// Done.
    Ok,
    /// The work failed, or stopped short of what was asked.
    Failed,
    /// Refused before anything ran.
    Refused,
    /// A sentence was asked back.
    Asked,
}

impl From<Exit> for std::process::ExitCode {
    fn from(exit: Exit) -> Self {
        Self::from(match exit {
            Exit::Ok => 0,
            Exit::Failed => 1,
            Exit::Refused => 2,
            Exit::Asked => 3,
        })
    }
}

/// See the module documentation.
pub struct Session {
    splinter: Splinter,
    global: Global,
    /// What this binary was built from, for `baseline`.
    build: BuildInfo,
}

impl Session {
    /// A session over `config` with the command line's global flags.
    pub fn new(config: Config, global: Global) -> Result<Self, Error> {
        let verbose = global.verbose > 0;
        let build = config.build.clone();
        let splinter = Splinter::builder(config)
            .allow_remote(global.allow_remote)
            .progress(Arc::new(move |stage, summary| {
                if verbose {
                    eprintln!("[{stage}] {summary}");
                } else {
                    eprintln!("[{stage}] {}", learn_output::stage_line(stage, summary));
                }
            }))
            .build()?;
        Ok(Self {
            splinter,
            global,
            build,
        })
    }

    /// The context of one command: its own pins on the shared runtime, so
    /// each sentence of a REPL resolves `policy:<alias>` afresh.
    fn context(&self) -> Context {
        self.splinter.context()
    }

    /// Runs `command`, prints its report, and says how it ended.
    pub fn command(&self, command: Command) -> Exit {
        match self.dispatch(command) {
            Ok(exit) => exit,
            Err(e) => {
                output::error(self.global.json, &e);
                if e.is_refusal() {
                    Exit::Refused
                } else {
                    Exit::Failed
                }
            }
        }
    }

    /// Handles `sentence` through the front door, never asking back
    /// interactively.
    pub fn sentence(&self, sentence: &str) -> Exit {
        self.front_door(sentence, None)
    }

    /// Reads sentences until the input ends, each handled like
    /// `splinter "<line>"`; at a terminal, a question back is answered
    /// there. Ends with the worst exit of any line.
    pub fn repl(&self) -> Exit {
        let stdin = std::io::stdin();
        let interactive = stdin.is_terminal() && !self.global.json;
        let mut worst = Exit::Ok;
        let mut lines = stdin.lock().lines();
        loop {
            if interactive {
                print!("splinter> ");
                let _ = std::io::stdout().flush();
            }
            let Some(Ok(line)) = lines.next() else {
                break;
            };
            if line.trim().is_empty() {
                continue;
            }
            let answer = interactive.then_some(&mut lines);
            worst = worst.max(self.front_door(line.trim(), answer));
        }
        worst
    }

    /// One sentence: classified, then run, asked back or refused. With
    /// `answers`, a question back is answered from them.
    fn front_door(
        &self,
        sentence: &str,
        answers: Option<&mut std::io::Lines<std::io::StdinLock<'_>>>,
    ) -> Exit {
        let routed = match interpret(&self.context(), sentence) {
            Ok(routed) => routed,
            Err(e) => {
                output::error(self.global.json, &e);
                return if e.is_refusal() {
                    Exit::Refused
                } else {
                    Exit::Failed
                };
            }
        };
        match &routed {
            Routed::Execute(intent) => self.run_words(&intent.command_line()),
            Routed::Refuse(_) => {
                output::routed(self.global.json, &routed);
                Exit::Refused
            }
            Routed::Clarify(question) => {
                output::routed(self.global.json, &routed);
                let Some(answers) = answers else {
                    return Exit::Asked;
                };
                if question.candidates.is_empty() {
                    return Exit::Asked;
                }
                print!(
                    "run which? [1-{}, empty for none] ",
                    question.candidates.len()
                );
                let _ = std::io::stdout().flush();
                let choice = answers
                    .next()
                    .and_then(Result::ok)
                    .and_then(|a| a.trim().parse::<usize>().ok())
                    .and_then(|n| n.checked_sub(1))
                    .and_then(|n| question.candidates.get(n));
                match choice {
                    Some(chosen) => self.run_words(&chosen.command),
                    None => Exit::Asked,
                }
            }
        }
    }

    /// Runs the command `words` names (the arguments after `splinter`).
    fn run_words(&self, words: &[String]) -> Exit {
        eprintln!("splinter {}", shell_words(words));
        let argv = std::iter::once("splinter".to_string()).chain(words.iter().cloned());
        match Cli::try_parse_from(argv) {
            Ok(Cli {
                command: Some(command),
                ..
            }) => self.command(command),
            Ok(_) => {
                eprintln!("splinter: the sentence did not map to a command");
                Exit::Refused
            }
            Err(e) => {
                eprintln!("splinter: the sentence mapped to an invalid command: {e}");
                Exit::Refused
            }
        }
    }

    fn dispatch(&self, command: Command) -> Result<Exit, Error> {
        let context = self.context();
        let ctx = &context;
        let json = self.global.json;
        match command {
            Command::Learn(args) => {
                return crate::learn_session::learn_command(&self.splinter, ctx, json, *args)
            }
            Command::Improve(args) => {
                return crate::improve_session::improve_command(ctx, json, args)
            }
            Command::Campaign(command) => {
                return crate::campaign_session::campaign(&self.splinter, ctx, json, command)
            }
            Command::Generation(command) => {
                return crate::improve_session::generation_command(
                    &self.splinter,
                    ctx,
                    json,
                    command,
                )
            }
            Command::Practice(args) => {
                let release_asked = !args.no_release;
                let practiced = self.splinter.practice(&practice_request(*args)?)?;
                emit(json, &practiced);
                return Ok(if practiced.report.finished(release_asked) {
                    Exit::Ok
                } else {
                    Exit::Failed
                });
            }
            Command::Corrections(args) => {
                return crate::corrections_session::corrections_command(ctx, json, *args);
            }
            Command::Objects(command) => crate::objects_output::run(ctx, json, command)?,
            Command::Env(command) => crate::env_output::run(ctx, json, command)?,
            Command::Ask(args) => {
                let retrieve = &args.retrieval;
                if retrieve.sources.is_empty() {
                    emit(
                        json,
                        &ask(ctx, &args.question, args.open_book.as_deref(), &args.policy)?,
                    );
                } else {
                    if args.open_book.is_some() {
                        return Err(Error::Refused(
                            "--open-book shows one whole source and --retrieve passages of \
                             several: name one of them"
                                .into(),
                        ));
                    }
                    let embedder = ctx.embedder()?;
                    let reader = reader_of(ctx, retrieve)?;
                    let rerank = reader.as_ref().map(|reader| Rerank {
                        reranker: reader,
                        candidates: candidates_for(retrieve.passages),
                    });
                    emit(
                        json,
                        &ask_retrieving(
                            ctx,
                            &RetrievingQuestion {
                                question: &args.question,
                                sources: &retrieve.sources,
                                passages: retrieve.passages,
                                rerank,
                                policy: &args.policy,
                            },
                            &*embedder,
                        )?,
                    );
                }
            }
            Command::Baseline(args) => {
                return baseline_report::run(&self.build, &args, json).map_err(Error::Refused)
            }
            Command::Status => emit(json, &status(ctx)?),
            Command::Doctor => {
                let report = doctor(ctx);
                emit(json, &report);
                if !report.ready() {
                    return Ok(Exit::Failed);
                }
            }
            Command::Source(SourceCommand::Add {
                target,
                usage_policy,
            }) => {
                let target = SourceTarget::from_args(&target)?;
                let terms = usage_policy.map(|policy| policy.terms(policy.as_str()));
                emit(
                    json,
                    &record(ctx, "source add", &target, |_| {
                        sources::add_with_terms(ctx, &target, terms)
                    })?,
                );
            }
            Command::Experiment(ExperimentCommand::Check { plan, catalog }) => {
                let plan = experiment::ExperimentPlan::load(&plan, catalog.as_deref())?;
                // A validated plan seals its final exam in the state root, so
                // a run that is not given the plan cannot read it later.
                splinter_sdk::sealed::record(ctx, &plan)?;
                emit(json, &plan);
            }
            Command::Source(SourceCommand::List) => emit(json, &sources::list(ctx)?),
            Command::Source(SourceCommand::Show { id }) => emit(json, &sources::show(ctx, &id)?),
            Command::Absorb(args) => {
                return absorb_session::absorb(ctx, self.splinter.trainer(), json, *args)
            }
            Command::Consolidate(args) => {
                return absorb_session::consolidate(ctx, self.splinter.trainer(), json, *args)
            }
            Command::Session(command) => return absorb_session::session(ctx, json, command),
            Command::Claims(command) => return absorb_session::claims(ctx, json, command),
            Command::Tasks(command) => return tasks_session::tasks(ctx, json, command),
            Command::Solve(args) => return solve_session::solve(ctx, json, &args),
            Command::Harness(command) => return harness_session::harness(ctx, json, command),
            Command::Verify(args) => {
                let set = resolve_set(ctx, &args.experience_set)?;
                let judge = args
                    .judge
                    .as_ref()
                    .map(|j| Judge::load(ctx, j))
                    .transpose()?;
                let arguments = json!({ "experience_set": set, "judge": args.judge });
                let verified = record(ctx, "verify", &arguments, |run| {
                    verify_set(
                        ctx,
                        &set,
                        judge.as_ref().map_or(Grading::ActiveJudge, Grading::Judge),
                        &run.cancel_token(),
                    )
                })?;
                emit(json, &verified);
            }
            Command::Critique(args) => {
                let set = resolve_set(ctx, &args.experience_set)?;
                let solver = ModelRef::policy_default();
                let arguments =
                    json!({ "experience_set": set, "critic": args.critic, "retry": args.retry });
                let critiqued = record(ctx, "critique", &arguments, |run| {
                    critique_set(
                        ctx,
                        &CritiqueRequest {
                            set: &set,
                            critic: &args.critic,
                            solver: &solver,
                            retries: args.retry,
                            deadline: None,
                            cancel: run.cancel_token(),
                        },
                    )
                })?;
                emit(json, &critiqued);
            }
            Command::Judge(JudgeCommand::Calibrate { labelled, judge }) => {
                let arguments = json!({ "labelled": labelled, "judge": judge });
                let calibrated = record(ctx, "judge calibrate", &arguments, |_| {
                    calibrate_judge(ctx, &labelled, &judge)
                })?;
                emit(json, &calibrated);
            }
            Command::Judge(JudgeCommand::Measure { tasks, judge, fit }) => {
                let set = resolve_task_set(ctx, &tasks)?;
                let judging = if fit {
                    Judging::Fit
                } else {
                    Judging::Reference
                };
                let arguments = json!({ "tasks": tasks, "judge": judge, "fit": fit });
                let measured = record(ctx, "judge measure", &arguments, |_| {
                    measure_judge(ctx, &set, &judge, judging)
                })?;
                emit(json, &measured);
            }
            Command::Experiences(ExperiencesCommand::List) => emit(json, &experiences::list(ctx)?),
            Command::Experiences(ExperiencesCommand::Show { id, graph }) => {
                emit(json, &experiences::show(ctx, &id, graph)?);
            }
            Command::Experiences(ExperiencesCommand::Replay { id }) => {
                emit(json, &experiences::replay(ctx, &id)?);
            }
            Command::Experiences(ExperiencesCommand::IngestHarbor { task_set, trials }) => {
                let set = resolve_task_set(ctx, &task_set)?;
                let arguments = json!({ "task_set": set, "trials": trials });
                let ingested = record(ctx, "experiences ingest-harbor", &arguments, |_| {
                    splinter_sdk::harbor::ingest_harbor_trials(ctx, &set, &trials)
                })?;
                emit(json, &ingested);
            }
            Command::Experiences(ExperiencesCommand::IngestBench {
                exports,
                assume_set,
            }) => {
                let options = splinter_sdk::bench::IngestOptions {
                    assume_set: assume_set.map(Into::into),
                };
                let arguments = json!({ "exports": exports, "assume_set": assume_set.map(|s| format!("{s:?}")) });
                let ingested = record(ctx, "experiences ingest-bench", &arguments, |_| {
                    splinter_sdk::bench::ingest_bench_trials(ctx, &exports, options)
                })?;
                emit(json, &ingested);
                // A refused trial did not become an experience.
                if !ingested.report.refused.is_empty() {
                    return Ok(Exit::Failed);
                }
            }
            Command::Experiences(ExperiencesCommand::Diagnose(args)) => {
                let report = splinter_sdk::bench::diagnose::diagnose_dirs(&args.dirs);
                emit(json, &report);
                if !report.refused.is_empty() {
                    return Ok(Exit::Failed);
                }
            }
            Command::Experiences(ExperiencesCommand::Capabilities(args)) => {
                use splinter_sdk::bench::capabilities::{
                    capability_report, CapabilityRequest, StoredCapabilities,
                };
                let report = capability_report(&CapabilityRequest {
                    dirs: &args.dirs,
                    labels: &args.labels,
                    policy: args.policy.as_deref(),
                });
                let stored = StoredCapabilities::keep(ctx, report)?;
                emit(json, &stored);
                if !stored.report.refused.is_empty() {
                    return Ok(Exit::Failed);
                }
            }
            Command::Experiences(ExperiencesCommand::Repair(args)) => {
                use splinter_sdk::bench::repair::{
                    repair, RepairRequest, DEFAULT_BUDGET, DEFAULT_MAX_REPAIRS,
                    DEFAULT_REPAIR_DEADLINE, DEFAULT_REPAIR_JOBS,
                };
                let checkout =
                    splinter_sdk::bench::practice::BenchCheckout::new(args.bench.clone());
                let seed = splinter_sdk::seed::resolve("seed", args.seed.map(|s| s.value()));
                let arguments = json!({ "runs": args.runs, "teacher": args.teacher.to_string(),
                    "student": args.student.as_ref().map(ToString::to_string),
                    "proposer": args.proposer, "knowledge": args.knowledge,
                    "interventions": args.interventions, "budget": args.budget,
                    "seed": seed, "per_task": args.per_task, "samples": args.samples,
                    "classes": args.classes.iter().map(|c| c.as_str()).collect::<Vec<_>>() });
                let report = record(ctx, "experiences repair", &arguments, |run| {
                    repair(
                        ctx,
                        &RepairRequest {
                            runs: &args.runs,
                            checkout: &checkout,
                            teacher: &args.teacher,
                            student: args.student.as_ref(),
                            proposer: args.proposer,
                            knowledge: &args.knowledge,
                            interventions: &args.interventions,
                            budget: args.budget.unwrap_or(DEFAULT_BUDGET),
                            max: args.max.unwrap_or(DEFAULT_MAX_REPAIRS),
                            jobs: args.jobs.unwrap_or(DEFAULT_REPAIR_JOBS),
                            deadline: DEFAULT_REPAIR_DEADLINE,
                            classes: &args.classes,
                            per_task: args.per_task,
                            samples: args.samples.unwrap_or(1),
                            seed,
                            cancel: run.cancel_token(),
                        },
                    )
                })?;
                emit(json, &report);
            }
            Command::Experiences(ExperiencesCommand::Import(args)) => {
                let source = splinter_sdk::store::StateRoot::new(args.from.clone());
                let arguments = json!({ "from": args.from, "sets": args.sets });
                let imported = record(ctx, "experiences import", &arguments, |_| {
                    splinter_sdk::set_import::import_sets(ctx, &source, &args.sets)
                })?;
                emit(json, &imported);
            }
            Command::Experiences(ExperiencesCommand::IngestHf(args)) => {
                crate::hub_session::ingest_bundle(ctx, json, args)?;
            }
            Command::Experiences(ExperiencesCommand::Compare {
                candidate,
                baseline,
                interval_seed,
                strata,
            }) => compare_session::compare_sets(
                ctx,
                json,
                &candidate,
                &baseline,
                interval_seed,
                strata.cutoffs(),
            )?,
            Command::Dataset(DatasetCommand::Build(args)) => {
                crate::dataset_build::run(ctx, json, *args)?;
            }
            Command::Dataset(command) => {
                if !conversation_dataset::dispatch(ctx, json, command)? {
                    return Ok(Exit::Failed);
                }
            }
            Command::Goal(command) => {
                if !crate::goal_session::run(json, command)? {
                    return Ok(Exit::Refused);
                }
            }
            Command::Train(args) => {
                let request = TrainRequest {
                    datasets: args.datasets,
                    rehearsal: args.rehearsal.map(|dataset| Rehearse {
                        dataset,
                        share: args.rehearsal_share.unwrap_or(DEFAULT_REHEARSAL_SHARE),
                    }),
                    from: args.from,
                    replay_fraction: args.replay_fraction,
                    replay_policy: Default::default(),
                    steps: args.steps,
                    rank: args.rank,
                    beta: args.beta,
                    tuning: {
                        let tuning = args.optimiser.applied_to(Tuning {
                            records_per_step: args.records_per_step,
                            eval_every: args.monitoring.eval_every,
                            patience: args.monitoring.patience,
                            monitor_share: args.monitoring.monitor_share,
                            keep_evaluations: args.monitoring.keep_evaluations,
                            max_block: args.max_block,
                            seed: args.seed,
                            base_precision: args.base_precision,
                            policy_gradient: args.policy_gradient.options(args.temperature),
                            pairs: args.pairs.settings(),
                            ..Tuning::default()
                        });
                        match args.recipe {
                            Some(recipe) => recipe.apply(tuning),
                            None => tuning,
                        }
                    },
                };
                let candidate = record(ctx, "train", &request, |run| {
                    train_recorded(ctx, run, &request, self.splinter.trainer())
                })?;
                emit(json, &Trained::new(candidate, args.memorisation_ratio));
            }
            Command::Rehearse(args) => {
                let arguments =
                    json!({ "records": args.records, "rehearsal_seed": args.rehearsal_seed });
                let rehearsed = record(ctx, "rehearse", &arguments, |run| {
                    rehearse(
                        ctx,
                        &RehearseRequest {
                            records: args.records,
                            seed: run.seed("rehearsal-seed", args.rehearsal_seed)?,
                            deadline: None,
                            cancel: run.cancel_token(),
                        },
                    )
                })?;
                emit(json, &rehearsed);
            }
            Command::Release(args) => {
                if matches!(args.command, Some(ReleaseCommand::List)) {
                    emit(json, &release::list(ctx)?);
                    return Ok(Exit::Ok);
                }
                let Some(candidate) = args.candidate else {
                    return Err(Error::Refused(
                        "name a candidate to release, or `release list`".into(),
                    ));
                };
                if let Some(judge) = args.judge {
                    ctx.set_judge(judge);
                }
                let canaries = args
                    .canaries
                    .map(|id| canaries::load(ctx, &resolve_task_set(ctx, &id)?))
                    .transpose()?
                    .unwrap_or_default();
                let request = ReleaseRequest {
                    alias: args.alias,
                    canaries,
                    distribution: if args.unrestricted {
                        Distribution::Unrestricted
                    } else {
                        Distribution::Restricted
                    },
                    interval_seed: args.interval_seed,
                    gate: args.gate.plan().applied_to(GateConfig::default()),
                    ..ReleaseRequest::new(candidate)
                };
                let released = record(ctx, "release", &request, |run| {
                    release::release_recorded(ctx, run, &request)
                })?;
                emit(json, &released);
                // A blocked release did not do what was asked.
                if released.report.release.is_none() {
                    return Ok(Exit::Failed);
                }
            }
            Command::Rollback { alias } => {
                let arguments = json!({ "alias": alias });
                emit(
                    json,
                    &record(ctx, "rollback", &arguments, |_| {
                        release::rollback(ctx, &alias)
                    })?,
                );
            }
            Command::Eval(EvalArgs {
                command: Some(EvalCommand::Matrix { alias }),
                ..
            }) => emit(json, &release::continual::matrix(ctx, &alias)?),
            Command::Eval(args) => {
                if let Some(judge) = args.judge {
                    ctx.set_judge(judge);
                }
                let request = EvalRequest {
                    model: args.model,
                    suite: args.suite,
                    freeze: args.freeze,
                };
                emit(json, &evaluate(ctx, &request)?);
            }
            Command::ExamReport(command) => exam_session::report(ctx, json, command)?,
            Command::ExamSet(command) => exam_session::set(ctx, json, command)?,
            Command::Select(args) => emit(json, &exam_session::select(ctx, &args)?),
            Command::Exam(args) if args.exam_set.is_some() => {
                emit(json, &exam_session::powered(ctx, &args)?);
            }
            Command::Exam(args) => {
                let retrieve = &args.retrieval;
                let arguments = json!({
                    "candidate": args.candidate,
                    "judge": args.judge,
                    "prompt": args.prompt,
                    "retrieve": retrieve.sources,
                    "passages": retrieve.passages,
                    "reranker": retrieve.reranker,
                    "interval_seed": args.interval_seed,
                });
                // The library is made (or read back) before the exam starts:
                // it needs the embedding model, which the exam's queries use
                // too.
                let embedder;
                let library;
                let reader;
                let retrieval = if retrieve.sources.is_empty() {
                    None
                } else {
                    embedder = ctx.embedder()?;
                    library = library_of(ctx, &retrieve.sources, &*embedder)?.1;
                    reader = reader_of(ctx, retrieve)?;
                    Some(Retrieval {
                        library: &library,
                        embedder: &*embedder,
                        passages: retrieve.passages,
                        rerank: reader.as_ref().map(|reader| Rerank {
                            reranker: reader,
                            candidates: candidates_for(retrieve.passages),
                        }),
                    })
                };
                emit(
                    json,
                    &record(ctx, "exam", &arguments, |run| {
                        examine(
                            ctx,
                            &ExamineRequest {
                                candidate: &args.candidate,
                                base: None,
                                judge: args.judge.as_ref(),
                                prompted: args.prompt.as_deref(),
                                retrieval: retrieval.as_ref(),
                                interval_seed: run.seed("interval-seed", args.interval_seed)?,
                            },
                            &run.cancel_token(),
                        )
                    })?,
                );
            }
            Command::State(StateCommand::Status) => emit(json, &state::storage(ctx)?),
            Command::State(StateCommand::Maintain { collect }) => {
                emit(json, &state::maintain(ctx, collect)?);
            }
            Command::State(StateCommand::Unpin { holder }) => {
                emit(json, &state::unpin(ctx, &holder)?);
            }
            Command::State(StateCommand::Verify { deep }) => {
                let verified = state::verify(ctx, deep)?;
                emit(json, &verified);
                return Ok(if verified.is_sound() {
                    Exit::Ok
                } else {
                    Exit::Failed
                });
            }
            Command::State(StateCommand::Repair { from, accept_loss }) => {
                let repaired = state::repair(ctx, from, accept_loss)?;
                emit(json, &repaired);
                return Ok(if repaired.unresolved == 0 {
                    Exit::Ok
                } else {
                    Exit::Failed
                });
            }
            Command::State(StateCommand::Archive {
                file,
                no_artifacts,
                since,
            }) => emit(json, &state::archive(ctx, &file, no_artifacts, since)?),
            Command::State(StateCommand::Pack) => emit(json, &state::pack(ctx)?),
            Command::State(StateCommand::Restore { files }) => {
                emit(json, &state::restore(ctx, &files)?);
            }
            Command::Runs(RunsCommand::List) => emit(json, &runs::list(ctx)?),
            Command::Runs(RunsCommand::Show { id }) => emit(json, &runs::show(ctx, &id)?),
            Command::Runs(RunsCommand::Cancel { id }) => emit(json, &runs::cancel(ctx, &id)?),
            Command::Jobs(args) => return crate::jobs_session::jobs(&self.splinter, json, args),
            Command::CliReference(args) => return Ok(crate::cli::reference::run(&args)),
            Command::Publish(args) => crate::hub_session::publish_bundle(ctx, json, args)?,
            Command::Lineage(args) => {
                let request = LineageRequest {
                    direction: args.direction(),
                    id: args.id,
                    depth: args.depth,
                };
                emit(json, &lineage(ctx, &request)?);
            }
        }
        Ok(Exit::Ok)
    }
}

/// The model that reads retrieval's candidates, when one was named.
fn reader_of<'a>(
    ctx: &'a Context,
    retrieve: &RetrieveArgs,
) -> Result<Option<ModelReranker<'a>>, Error> {
    retrieve
        .reranker
        .as_ref()
        .map(|reference| ModelReranker::new(ctx, reference))
        .transpose()
}
