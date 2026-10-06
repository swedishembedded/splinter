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
use splinter_sdk::agent::CancelToken;
use splinter_sdk::ask::{ask, ask_retrieving, RetrievingQuestion};
use splinter_sdk::critique::{critique_set, CritiqueRequest};
use splinter_sdk::curriculum::frontier::{measure, MeasureRequest};
use splinter_sdk::datasets::{build, export, BuildRequest, VoiceBuild};
use splinter_sdk::describe::DescribeRequest;
use splinter_sdk::eval::{evaluate, EvalRequest};
use splinter_sdk::exam::{examine, ExamineRequest};
use splinter_sdk::experiences::{self, resolve_set};
use splinter_sdk::judge::{calibrate_judge, measure_judge};
use splinter_sdk::learn::{learn, Learned};
use splinter_sdk::lineage::{lineage, LineageRequest};
use splinter_sdk::rehearsal::{rehearse, RehearseRequest, DEFAULT_REHEARSAL_SHARE, REHEARSAL_SEED};
use splinter_sdk::release::{self, ReleaseRequest};
use splinter_sdk::rerank::ModelReranker;
use splinter_sdk::retrieval::{candidates_for, library_of, Rerank, Retrieval};
use splinter_sdk::router::{interpret, Routed};
use splinter_sdk::runs::{self, record};
use splinter_sdk::solving::solve_set;
use splinter_sdk::sources::{self, SourceTarget};
use splinter_sdk::state;
use splinter_sdk::status::status;
use splinter_sdk::tasks::{self, check_kinds, resolve_set as resolve_task_set};
use splinter_sdk::train::{train, Rehearse, TrainRequest, Tuning};
use splinter_sdk::variants;
use splinter_sdk::verify::{verify_set, Grading, Judge, Judging};
use splinter_sdk::vocabulary::model_ref::ModelRef;
use splinter_sdk::vocabulary::terms::Distribution;
use splinter_sdk::{Config, Context, Error, Splinter};

use crate::cli::{
    Cli, Command, DatasetCommand, ExperiencesCommand, Global, JudgeCommand, ReleaseCommand,
    RetrieveArgs, RunsCommand, SourceCommand, StateCommand, TasksCommand,
};
use crate::exam_session;
use crate::learn_output;
use crate::learn_request::learn_request;
use crate::output::{self, emit, shell_words};

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
}

impl Session {
    /// A session over `config` with the command line's global flags.
    pub fn new(config: Config, global: Global) -> Result<Self, Error> {
        let verbose = global.verbose > 0;
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
        Ok(Self { splinter, global })
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
                let release_asked = !args.no_release;
                let request = learn_request(*args);
                let learned = learn(ctx, &request, self.splinter.trainer())?;
                emit(json, &learned);
                let finished = match &learned {
                    Learned::Planned(_) => true,
                    Learned::Ran(run) => run.report.finished(release_asked),
                };
                return Ok(if finished { Exit::Ok } else { Exit::Failed });
            }
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
            Command::Status => emit(json, &status(ctx)?),
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
            Command::Source(SourceCommand::List) => emit(json, &sources::list(ctx)?),
            Command::Source(SourceCommand::Show { id }) => emit(json, &sources::show(ctx, &id)?),
            Command::Tasks(TasksCommand::Generate {
                sources: ids,
                kinds,
                generator,
                author,
            }) => {
                let kinds = check_kinds(&kinds)?;
                let ids = ids
                    .iter()
                    .map(|id| sources::resolve(ctx, id))
                    .collect::<Result<Vec<_>, _>>()?;
                let arguments = json!({
                    "sources": ids,
                    "kinds": kinds,
                    "generator": generator,
                    "author": author,
                });
                let generated = record(ctx, "tasks generate", &arguments, |run| {
                    tasks::generate(
                        ctx,
                        &tasks::Generation {
                            sources: &ids,
                            sections: &[],
                            kinds: &kinds,
                            generator: &generator,
                            goal: None,
                            author: author.as_deref(),
                            deadline: None,
                            cancel: run.cancel_token(),
                        },
                    )
                })?;
                emit(json, &generated);
            }
            Command::Tasks(TasksCommand::Variants {
                task_set,
                generator,
                per_task,
            }) => {
                let set = resolve_task_set(ctx, &task_set)?;
                let arguments =
                    json!({ "task_set": set, "generator": generator, "per_task": per_task });
                let varied = record(ctx, "tasks variants", &arguments, |run| {
                    variants::generate_variants(
                        ctx,
                        &variants::VariantsRequest {
                            task_set: &set,
                            generator: &generator,
                            per_task,
                            deadline: None,
                            cancel: run.cancel_token(),
                        },
                    )
                })?;
                emit(json, &varied);
            }
            Command::Tasks(TasksCommand::List) => emit(json, &tasks::list(ctx)?),
            Command::Tasks(TasksCommand::Show { id }) => emit(json, &tasks::show(ctx, &id)?),
            Command::Solve(args) if args.frontier => {
                let set = resolve_task_set(ctx, &args.task_set)?;
                let pass_at_k = args.pass_at_k.pass_at_k();
                let arguments = json!({
                    "task_set": set,
                    "solver": args.solver,
                    "teacher": args.teacher,
                    "frontier": true,
                    "pass_at_k": pass_at_k,
                });
                let measured = record(ctx, "solve", &arguments, |run| {
                    measure(
                        ctx,
                        &MeasureRequest {
                            task_set: &set,
                            solver: &args.solver,
                            teacher: args.teacher.as_ref(),
                            pass_at_k,
                            deadline: None,
                            cancel: run.cancel_token(),
                        },
                    )
                })?;
                emit(json, &measured);
            }
            Command::Solve(args) => {
                let set = resolve_task_set(ctx, &args.task_set)?;
                let arguments = json!({ "task_set": set, "solver": args.solver });
                let solved = record(ctx, "solve", &arguments, |run| {
                    solve_set(ctx, &set, &args.solver, None, &run.cancel_token())
                })?;
                emit(json, &solved);
            }
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
            Command::Dataset(DatasetCommand::Build {
                sets,
                view,
                strip,
                min_strength,
                system_prompt,
                export_only,
                limit,
                writer,
                token_limit,
                max_family_share,
                describe_with,
            }) => {
                let request = BuildRequest {
                    sets: sets
                        .iter()
                        .map(|id| resolve_set(ctx, id))
                        .collect::<Result<_, _>>()?,
                    view,
                    strip,
                    min_strength,
                    system_prompt,
                    export_only,
                    limit,
                    voice: VoiceBuild {
                        writer,
                        token_budget: token_limit,
                        max_family_share,
                        describe: describe_with.map(|generator| DescribeRequest {
                            generator,
                            deadline: None,
                            cancel: CancelToken::new(),
                        }),
                    },
                };
                emit(
                    json,
                    &record(ctx, "dataset build", &request, |_| build(ctx, &request))?,
                );
            }
            Command::Dataset(DatasetCommand::Export { id, out }) => {
                emit(json, &export(ctx, &id, &out)?);
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
                    steps: args.steps,
                    rank: args.rank,
                    beta: args.beta,
                    tuning: args.optimiser.applied_to(Tuning {
                        records_per_step: args.records_per_step,
                        eval_every: args.monitoring.eval_every,
                        patience: args.monitoring.patience,
                        monitor_share: args.monitoring.monitor_share,
                        seed: args.seed,
                        ..Tuning::default()
                    }),
                };
                let candidate = record(ctx, "train", &request, |run| {
                    train(ctx, &request, self.splinter.trainer(), &run.cancel_token())
                })?;
                emit(json, &candidate);
            }
            Command::Rehearse(args) => {
                let request = RehearseRequest {
                    records: args.records,
                    seed: REHEARSAL_SEED,
                    deadline: None,
                    cancel: CancelToken::new(),
                };
                let rehearsed = record(ctx, "rehearse", &request, |run| {
                    rehearse(
                        ctx,
                        &RehearseRequest {
                            cancel: run.cancel_token(),
                            ..request.clone()
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
                let request = ReleaseRequest {
                    alias: args.alias,
                    distribution: if args.unrestricted {
                        Distribution::Unrestricted
                    } else {
                        Distribution::Restricted
                    },
                    ..ReleaseRequest::new(candidate)
                };
                let released = record(ctx, "release", &request, |run| {
                    release::release(ctx, &request, &run.cancel_token())
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
            Command::State(StateCommand::Restore { files }) => {
                emit(json, &state::restore(ctx, &files)?);
            }
            Command::Runs(RunsCommand::List) => emit(json, &runs::list(ctx)?),
            Command::Runs(RunsCommand::Show { id }) => emit(json, &runs::show(ctx, &id)?),
            Command::Runs(RunsCommand::Cancel { id }) => emit(json, &runs::cancel(ctx, &id)?),
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
