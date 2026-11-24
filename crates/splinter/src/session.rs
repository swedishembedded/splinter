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
use splinter_sdk::critique::{critique_set, CritiqueRequest};
use splinter_sdk::doctor::doctor;
use splinter_sdk::exam::{examine, ExamineRequest};
use splinter_sdk::experiences::{self, resolve_set};
use splinter_sdk::experiment;
use splinter_sdk::judge::{calibrate_judge, measure_judge};
use splinter_sdk::lineage::{lineage, LineageRequest};
use splinter_sdk::release;
use splinter_sdk::retrieval::{candidates_for, library_of, Rerank, Retrieval};
use splinter_sdk::router::{interpret, Routed};
use splinter_sdk::runs::{self, record};
use splinter_sdk::sources::{self, SourceTarget};
use splinter_sdk::state;
use splinter_sdk::status::status;
use splinter_sdk::tasks::resolve_set as resolve_task_set;
use splinter_sdk::verify::Judging;
use splinter_sdk::vocabulary::build::BuildInfo;
use splinter_sdk::vocabulary::model_ref::ModelRef;
use splinter_sdk::{Config, Context, Error, Splinter};

use crate::absorb_session;
use crate::baseline_report;
use crate::cli::{
    Cli, Command, EvalArgs, EvalCommand, ExperiencesCommand, ExperimentCommand, Global,
    JudgeCommand, ReleaseCommand, RunsCommand, SourceCommand, StateCommand,
};
use crate::compare_session;
use crate::conversation_dataset;
use crate::exam_session;
use crate::output::{self, emit, shell_words};
use crate::tasks_session;

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
                    eprintln!(
                        "[{stage}] {}",
                        crate::stage_output::stage_line(stage, summary)
                    );
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
        let command = match crate::factory_session::route(ctx.config(), json, command) {
            Ok(done) => return done,
            Err(other) => *other,
        };
        match command {
            Command::Improve(args) => {
                return crate::improve_session::improve_command(ctx, json, args);
            }
            Command::Corrections(args) => {
                return crate::corrections_session::corrections_command(ctx, json, *args);
            }
            Command::Objects(command) => crate::objects_output::run(ctx, json, command)?,
            Command::Env(command) => crate::env_output::run(ctx, json, command)?,
            Command::Ask(args) => return crate::ask_session::ask_command(ctx, json, &args),
            Command::Baseline(args) => {
                return baseline_report::run(&self.build, &args, json).map_err(Error::Refused);
            }
            Command::Status(_) => emit(json, &status(ctx)?),
            Command::Index(args) => {
                let indexed = splinter_sdk::factory_index::index_run(ctx, &args.run)?;
                emit(json, &crate::factory_session::Indexed(indexed));
            }
            Command::Init(_)
            | Command::Templates
            | Command::Validate(_)
            | Command::Plan(_)
            | Command::Generate(_)
            | Command::Run(_)
            | Command::Resume(_)
            | Command::Cancel(_)
            | Command::Inspect(_)
            | Command::Fork(_)
            | Command::Report(_)
            | Command::Export(_)
            | Command::Verify(_)
            | Command::Play(_) => {
                unreachable!("the factory's commands are routed before this match")
            }
            Command::Serve => {
                let serving = splinter_sdk::serving::start(ctx)?;
                emit(json, &serving);
                use std::io::Write as _;
                let _ = std::io::stdout().flush();
                splinter_sdk::serving::hold_until_interrupted();
            }
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
                return absorb_session::absorb(ctx, self.splinter.trainer(), json, *args);
            }
            Command::Consolidate(args) => {
                return absorb_session::consolidate(ctx, self.splinter.trainer(), json, *args);
            }
            Command::Session(command) => return absorb_session::session(ctx, json, command),
            Command::Claims(command) => return absorb_session::claims(ctx, json, command),
            Command::Tasks(command) => return tasks_session::tasks(ctx, json, command),
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
            Command::Experiences(ExperiencesCommand::CompareArms(args)) => {
                crate::compare_session::compare_arms(ctx, json, *args)?;
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
            Command::Release(ReleaseCommand::List) => emit(json, &release::list(ctx)?),
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
            Command::Eval(args) => crate::eval_session::eval(ctx, json, &args)?,
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
                    reader = crate::ask_session::reader_of(ctx, retrieve)?;
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
            // Runs in `main` before a session is built: it needs none.
            Command::Config(_) => unreachable!("`splinter config` runs before a session"),
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
