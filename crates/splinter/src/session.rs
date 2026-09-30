// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One invocation's work: a command run against the controller, a sentence
//! through the front door, or a REPL of sentences - each line handled
//! exactly like `splinter "<line>"`, on one context, so the policy loads
//! once.

use std::io::{BufRead, IsTerminal, Write};

use clap::Parser;
use serde_json::json;
use splinter_campaign::ask::ask;
use splinter_campaign::critique::{critique_set, CritiqueRequest};
use splinter_campaign::datasets::{build, export, BuildRequest};
use splinter_campaign::eval::{evaluate, EvalRequest};
use splinter_campaign::experiences::{self, resolve_set};
use splinter_campaign::front_door::{interpret, Routed};
use splinter_campaign::judge::calibrate_judge;
use splinter_campaign::learn::{learn, LearnRequest, Learned};
use splinter_campaign::lineage::{lineage, LineageRequest};
use splinter_campaign::model_ref::ModelRef;
use splinter_campaign::release::{self, ReleaseRequest};
use splinter_campaign::runs::{self, record};
use splinter_campaign::solving::solve_set;
use splinter_campaign::sources::{self, SourceTarget};
use splinter_campaign::status::status;
use splinter_campaign::tasks::{self, check_kinds, resolve_set as resolve_task_set};
use splinter_campaign::train::{train, BrainTrainer, TrainRequest};
use splinter_campaign::verify::{verify_set, Judge};
use splinter_campaign::{CampaignError, Config, Context};

use crate::cli::{
    Cli, Command, DatasetCommand, ExperiencesCommand, Global, JudgeCommand, ReleaseCommand,
    RunsCommand, SourceCommand, TasksCommand,
};
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
    ctx: Context,
    global: Global,
}

impl Session {
    /// A session over `config` with the command line's global flags.
    pub fn new(config: Config, global: Global) -> Result<Self, CampaignError> {
        let verbose = global.verbose > 0;
        let ctx = Context::new(config, global.allow_remote)?.with_progress(Box::new(
            move |stage, summary| {
                if verbose {
                    eprintln!("[{stage}] {summary}");
                } else {
                    eprintln!("[{stage}] {}", output::stage_line(stage, summary));
                }
            },
        ));
        Ok(Self { ctx, global })
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
        let routed = match interpret(&self.ctx, sentence) {
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

    fn dispatch(&self, command: Command) -> Result<Exit, CampaignError> {
        let ctx = &self.ctx;
        let json = self.global.json;
        match command {
            Command::Learn(args) => {
                let request = LearnRequest {
                    sources: args.sources,
                    goal: args.goal,
                    kinds: args.kinds,
                    budget: args.budget,
                    dry_run: args.dry_run,
                    no_release: args.no_release,
                };
                let learned = learn(ctx, &request, &BrainTrainer)?;
                emit(json, &learned);
                let finished = match &learned {
                    Learned::Planned(_) => true,
                    Learned::Ran(run) => run.report.finished(!args.no_release),
                };
                return Ok(if finished { Exit::Ok } else { Exit::Failed });
            }
            Command::Ask(args) => {
                emit(
                    json,
                    &ask(ctx, &args.question, args.open_book.as_deref(), &args.policy)?,
                );
            }
            Command::Status => emit(json, &status(ctx)?),
            Command::Source(SourceCommand::Add { target }) => {
                let target = SourceTarget::from_args(&target)?;
                emit(
                    json,
                    &record(ctx, "source add", &target, |_| sources::add(ctx, &target))?,
                );
            }
            Command::Source(SourceCommand::List) => emit(json, &sources::list(ctx)?),
            Command::Source(SourceCommand::Show { id }) => emit(json, &sources::show(ctx, &id)?),
            Command::Tasks(TasksCommand::Generate {
                sources: ids,
                kinds,
                generator,
            }) => {
                let kinds = check_kinds(&kinds)?;
                let ids = ids
                    .iter()
                    .map(|id| sources::resolve(ctx, id))
                    .collect::<Result<Vec<_>, _>>()?;
                let arguments = json!({ "sources": ids, "kinds": kinds, "generator": generator });
                let generated = record(ctx, "tasks generate", &arguments, |run| {
                    tasks::generate(
                        ctx,
                        &tasks::Generation {
                            sources: &ids,
                            kinds: &kinds,
                            generator: &generator,
                            goal: None,
                            deadline: None,
                            cancel: run.cancel_token(),
                        },
                    )
                })?;
                emit(json, &generated);
            }
            Command::Tasks(TasksCommand::List) => emit(json, &tasks::list(ctx)?),
            Command::Tasks(TasksCommand::Show { id }) => emit(json, &tasks::show(ctx, &id)?),
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
                    verify_set(ctx, &set, judge.as_ref(), &run.cancel_token())
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
            Command::Experiences(ExperiencesCommand::List) => emit(json, &experiences::list(ctx)?),
            Command::Experiences(ExperiencesCommand::Show { id, graph }) => {
                emit(json, &experiences::show(ctx, &id, graph)?);
            }
            Command::Dataset(DatasetCommand::Build {
                sets,
                view,
                strip,
                min_strength,
                export_only,
            }) => {
                let request = BuildRequest {
                    sets: sets
                        .iter()
                        .map(|id| resolve_set(ctx, id))
                        .collect::<Result<_, _>>()?,
                    view,
                    strip,
                    min_strength,
                    export_only,
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
                    from: args.from,
                    replay_fraction: args.replay_fraction,
                    steps: args.steps,
                    rank: args.rank,
                };
                let candidate = record(ctx, "train", &request, |run| {
                    train(ctx, &request, &BrainTrainer, &run.cancel_token())
                })?;
                emit(json, &candidate);
            }
            Command::Release(args) => {
                if matches!(args.command, Some(ReleaseCommand::List)) {
                    emit(json, &release::list(ctx)?);
                    return Ok(Exit::Ok);
                }
                let Some(candidate) = args.candidate else {
                    return Err(CampaignError::Refused(
                        "name a candidate to release, or `release list`".into(),
                    ));
                };
                let request = ReleaseRequest {
                    alias: args.alias,
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
                let request = EvalRequest {
                    model: args.model,
                    suite: args.suite,
                    freeze: args.freeze,
                };
                emit(json, &evaluate(ctx, &request)?);
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
