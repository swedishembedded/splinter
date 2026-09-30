// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The command line's grammar: one verb per pipeline stage, the thing acted
//! on as a positional argument, one model-reference syntax, and four global
//! flags every command takes. Defaults come from the crates that own them.

use std::path::PathBuf;
use std::time::Duration;

use clap::{ArgAction, Args, Parser, Subcommand};
use splinter_campaign::critique::DEFAULT_RETRIES;
use splinter_campaign::datasets::{parse_strength, parse_strip, Strip, ViewName};
use splinter_campaign::learn::parse_budget;
use splinter_campaign::model_ref::ModelRef;
use splinter_campaign::train::{DEFAULT_LORA_RANK, DEFAULT_STEPS};
use splinter_store::annotation::Strength;

/// A learning agent with its own model. Tell it what to learn - a document,
/// a repository, a command's output - and it generates tasks from it,
/// solves and verifies them, and trains a candidate on what passed.
///
/// With no command, a REPL on the current policy: each line is handled
/// exactly like `splinter "<line>"`.
#[derive(Debug, Parser)]
#[command(
    name = "splinter",
    version,
    after_help = "Models are named policy:default, local:<checkpoint>[+<adapter>] or \
                  remote:<provider>/<name>; remote ones need --allow-remote. Every stage stores \
                  what it makes under the state root by content address, so any stage can be \
                  rerun or inspected alone."
)]
pub struct Cli {
    /// Flags every command takes.
    #[command(flatten)]
    pub global: Global,
    /// A sentence, classified by the policy into one of the commands below;
    /// an unclear or unconfident reading is asked back, never run.
    #[arg(value_name = "SENTENCE")]
    pub sentence: Option<String>,
    /// The command.
    #[command(subcommand)]
    pub command: Option<Command>,
}

/// Flags every command takes, meaning the same everywhere.
#[derive(Clone, Debug, Default, Args)]
pub struct Global {
    /// The state root (default: $SPLINTER_STATE, else ~/.sven/splinter).
    #[arg(long, global = true, value_name = "DIR")]
    pub state: Option<PathBuf>,
    /// Print machine-readable JSON on stdout instead of text.
    #[arg(long, global = true)]
    pub json: bool,
    /// Report more as work goes (each stage's full summary).
    #[arg(short, long, global = true, action = ArgAction::Count)]
    pub verbose: u8,
    /// Permit models reached over the network (remote:...); without it, and
    /// without SPLINTER_ALLOW_REMOTE=1, every model runs locally.
    #[arg(long, global = true)]
    pub allow_remote: bool,
}

/// The commands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Learn from sources: capture them, generate tasks, solve, verify,
    /// critique and retry failures, build an sft-final dataset and train a
    /// candidate on it. The candidate is reported, not released.
    Learn(LearnArgs),
    /// Answer a question with a model, closed-book unless --open-book.
    Ask(AskArgs),
    /// Show the policy in use, recent runs and what the stores hold.
    Status,
    /// Capture and inspect sources.
    #[command(subcommand)]
    Source(SourceCommand),
    /// Generate and inspect task sets.
    #[command(subcommand)]
    Tasks(TasksCommand),
    /// Solve every task of a task set, each in the environment it records.
    Solve(SolveArgs),
    /// Append verdicts to an experience set (re-grades without re-running).
    Verify(VerifyArgs),
    /// Critique the failed experiences of a set and retry them with it.
    Critique(CritiqueArgs),
    /// Measure a judge against labelled experiences.
    #[command(subcommand)]
    Judge(JudgeCommand),
    /// Inspect experience sets and experiences.
    #[command(subcommand)]
    Experiences(ExperiencesCommand),
    /// Build datasets from experience sets, and export them.
    #[command(subcommand)]
    Dataset(DatasetCommand),
    /// Train a candidate adapter on datasets; it is not released.
    Train(TrainArgs),
    /// List, inspect and cancel runs.
    #[command(subcommand)]
    Runs(RunsCommand),
}

/// A model reference, for clap.
fn model_ref(text: &str) -> Result<ModelRef, String> {
    text.parse()
        .map_err(|e: splinter_campaign::model_ref::RefError| e.to_string())
}

fn budget(text: &str) -> Result<Duration, String> {
    parse_budget(text).map_err(|e| e.to_string())
}

fn view(text: &str) -> Result<ViewName, String> {
    text.parse()
        .map_err(|e: splinter_campaign::CampaignError| e.to_string())
}

fn strip(text: &str) -> Result<Strip, String> {
    parse_strip(text).map_err(|e| e.to_string())
}

fn strength(text: &str) -> Result<Strength, String> {
    parse_strength(text).map_err(|e| e.to_string())
}

/// `learn`.
#[derive(Debug, Args)]
pub struct LearnArgs {
    /// Files, directories, cmd:<command> (quoted as one argument), or stored
    /// source ids.
    #[arg(required = true, value_name = "SOURCE")]
    pub sources: Vec<String>,
    /// What the learner is after; steers the tasks generated.
    #[arg(long, value_name = "TEXT")]
    pub goal: Option<String>,
    /// Task kinds, comma-separated (default: recall).
    #[arg(long, value_delimiter = ',', value_name = "K,..")]
    pub kinds: Vec<String>,
    /// Wall-clock time the whole run may take, in h, m and s: 30m, 2h, 1h30m.
    #[arg(long, value_parser = budget, value_name = "DUR")]
    pub budget: Option<Duration>,
    /// Report the plan and write nothing.
    #[arg(long)]
    pub dry_run: bool,
}

/// `ask`.
#[derive(Debug, Args)]
pub struct AskArgs {
    /// The question.
    pub question: String,
    /// Show the model this stored source's text with the question.
    #[arg(long, value_name = "SOURCE-ID")]
    pub open_book: Option<String>,
    /// The model asked.
    #[arg(long, value_parser = model_ref, default_value_t = ModelRef::policy_default(), value_name = "REF")]
    pub policy: ModelRef,
}

/// `source ...`.
#[derive(Debug, Subcommand)]
pub enum SourceCommand {
    /// Capture a file (a document), a directory (a repository) or
    /// cmd:COMMAND... (a command run), and store it. URLs are refused.
    Add {
        /// PATH, or cmd: followed by the command and its arguments.
        #[arg(
            required = true,
            num_args = 1..,
            trailing_var_arg = true,
            allow_hyphen_values = true,
            value_name = "PATH|cmd:COMMAND..."
        )]
        target: Vec<String>,
    },
    /// List the stored sources.
    List,
    /// Show one stored source.
    Show {
        /// The source's id, or a unique prefix of it.
        id: String,
    },
}

/// `tasks ...`.
#[derive(Debug, Subcommand)]
pub enum TasksCommand {
    /// Generate tasks from stored sources into a task set.
    Generate {
        /// The sources, by id or unique prefix.
        #[arg(required = true, value_name = "SOURCE-ID")]
        sources: Vec<String>,
        /// Task kinds, comma-separated.
        #[arg(long, required = true, value_delimiter = ',', value_name = "K,..")]
        kinds: Vec<String>,
        /// The model that writes the tasks.
        #[arg(long, value_parser = model_ref, default_value_t = ModelRef::policy_default(), value_name = "REF")]
        generator: ModelRef,
    },
    /// List the stored task sets.
    List,
    /// Show a task set's tasks, or one task.
    Show {
        /// A task set's or a task's id, or a unique prefix of it.
        id: String,
    },
}

/// `solve`.
#[derive(Debug, Args)]
pub struct SolveArgs {
    /// The task set, by id or unique prefix.
    #[arg(value_name = "TASKSET-ID")]
    pub task_set: String,
    /// The model that solves.
    #[arg(long, value_parser = model_ref, default_value_t = ModelRef::policy_default(), value_name = "REF")]
    pub solver: ModelRef,
}

/// `verify`.
#[derive(Debug, Args)]
pub struct VerifyArgs {
    /// The experience set, by id or unique prefix.
    #[arg(value_name = "EXPERIENCE-SET")]
    pub experience_set: String,
    /// A calibrated judge for the kinds a judge grades.
    #[arg(long, value_parser = model_ref, value_name = "REF")]
    pub judge: Option<ModelRef>,
}

/// `critique`.
#[derive(Debug, Args)]
pub struct CritiqueArgs {
    /// The experience set, by id or unique prefix.
    #[arg(value_name = "EXPERIENCE-SET")]
    pub experience_set: String,
    /// The model that critiques.
    #[arg(long, value_parser = model_ref, default_value_t = ModelRef::policy_default(), value_name = "REF")]
    pub critic: ModelRef,
    /// Retries per failed experience.
    #[arg(long, default_value_t = DEFAULT_RETRIES, value_name = "N")]
    pub retry: usize,
}

/// `judge ...`.
#[derive(Debug, Subcommand)]
pub enum JudgeCommand {
    /// Measure a judge on labelled experiences and keep the calibration
    /// `verify --judge` gates its verdicts by.
    Calibrate {
        /// JSON Lines: {"experience": "<id>", "label": "pass"|"fail"}.
        #[arg(value_name = "LABELLED-FILE")]
        labelled: PathBuf,
        /// The judge.
        #[arg(long, required = true, value_parser = model_ref, value_name = "REF")]
        judge: ModelRef,
    },
}

/// `experiences ...`.
#[derive(Debug, Subcommand)]
pub enum ExperiencesCommand {
    /// List the stored experience sets.
    List,
    /// Show an experience set, or one experience with its annotations.
    Show {
        /// An experience set's or an experience's id, or a unique prefix.
        id: String,
        /// Show the relations around it instead: retries, critiques,
        /// preferences.
        #[arg(long)]
        graph: bool,
    },
}

/// `dataset ...`.
#[derive(Debug, Subcommand)]
pub enum DatasetCommand {
    /// Project experience sets through a view into a stored dataset.
    Build {
        /// The experience sets, by id or unique prefix.
        #[arg(required = true, value_name = "EXPERIENCE-SET")]
        sets: Vec<String>,
        /// sft-final, sft-step, critic, preference, verifier, decision,
        /// retrieval, outcome, denoise or cpt.
        #[arg(long, required = true, value_parser = view, value_name = "VIEW")]
        view: ViewName,
        /// What the student sees: all (only the instruction), keep:K,..
        /// or mix:F.
        #[arg(long, value_parser = strip, value_name = "all|keep:K,..|mix:F")]
        strip: Option<Strip>,
        /// The weakest decision counted: executable, formal, consistency
        /// (default) or judged.
        #[arg(long, value_parser = strength, value_name = "STRENGTH")]
        min_strength: Option<Strength>,
        /// Write an objective brain cannot train in the export format.
        #[arg(long)]
        export_only: bool,
    },
    /// Copy a dataset and its manifest into a directory.
    Export {
        /// The dataset, by id or unique prefix.
        #[arg(value_name = "DATASET-ID")]
        id: String,
        /// The directory to copy into.
        #[arg(long, required = true, value_name = "DIR")]
        out: PathBuf,
    },
}

/// `train`.
#[derive(Debug, Args)]
pub struct TrainArgs {
    /// The datasets, by id or unique prefix; the newest records of the last
    /// are held out for scoring.
    #[arg(required = true, value_name = "DATASET-ID")]
    pub datasets: Vec<String>,
    /// The base to train, and an adapter on it to continue.
    #[arg(long, value_parser = model_ref, default_value_t = ModelRef::policy_default(), value_name = "REF")]
    pub from: ModelRef,
    /// A dataset mixed into training whole, never held out.
    #[arg(long, value_name = "DATASET-ID")]
    pub replay: Option<String>,
    /// Optimizer steps.
    #[arg(long, default_value_t = DEFAULT_STEPS, value_name = "N")]
    pub steps: u32,
    /// LoRA rank of a new adapter.
    #[arg(long, default_value_t = DEFAULT_LORA_RANK, value_name = "R")]
    pub rank: u32,
}

/// `runs ...`.
#[derive(Debug, Subcommand)]
pub enum RunsCommand {
    /// List the recorded runs.
    List,
    /// Show one run's record: arguments, stages, status, outputs.
    Show {
        /// The run's id.
        id: String,
    },
    /// Ask a run in progress to stop.
    Cancel {
        /// The run's id.
        id: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    fn parse(args: &[&str]) -> Result<Cli, clap::Error> {
        Cli::try_parse_from(std::iter::once("splinter").chain(args.iter().copied()))
    }

    fn command(args: &[&str]) -> Command {
        match parse(args) {
            Ok(Cli {
                command: Some(command),
                ..
            }) => command,
            other => panic!("{args:?} is not a command: {other:?}"),
        }
    }

    #[test]
    fn the_grammar_is_consistent() {
        Cli::command().debug_assert();
    }

    #[test]
    fn a_sentence_or_nothing_is_the_front_door() {
        let cli = parse(&["learn the docs in ./docs"]).unwrap();
        assert_eq!(cli.sentence.as_deref(), Some("learn the docs in ./docs"));
        assert!(cli.command.is_none());
        let repl = parse(&[]).unwrap();
        assert!(repl.sentence.is_none() && repl.command.is_none());
        assert!(
            parse(&["two", "sentences"]).is_err(),
            "a sentence is one argument"
        );
        let sentence = parse(&["--json", "what does X do?"]).unwrap();
        assert!(sentence.global.json && sentence.sentence.is_some());
    }

    #[test]
    fn each_verb_takes_what_it_acts_on_as_positionals() {
        let Command::Learn(learn) = command(&[
            "learn",
            "docs/",
            "cmd:make --help",
            "--goal",
            "flags",
            "--kinds",
            "recall,denoise",
            "--budget",
            "1h30m",
            "--dry-run",
        ]) else {
            panic!("learn");
        };
        assert_eq!(learn.sources, ["docs/", "cmd:make --help"]);
        assert_eq!(learn.kinds, ["recall", "denoise"]);
        assert_eq!(learn.budget, Some(Duration::from_secs(5400)));
        assert!(learn.dry_run);
        assert!(parse(&["learn"]).is_err(), "learn needs a source");
        assert!(parse(&["learn", "x", "--budget", "soon"]).is_err());

        let Command::Ask(ask) = command(&["ask", "what does X do?", "--open-book", "ab12"]) else {
            panic!("ask");
        };
        assert_eq!(ask.question, "what does X do?");
        assert_eq!(
            ask.policy,
            ModelRef::policy_default(),
            "closed-book on the policy"
        );
        assert!(matches!(command(&["status"]), Command::Status));

        let Command::Source(SourceCommand::Add { target }) =
            command(&["source", "add", "cmd:ls", "-la", "/tmp"])
        else {
            panic!("source add");
        };
        assert_eq!(target, ["cmd:ls", "-la", "/tmp"]);
        assert!(matches!(
            command(&["source", "list"]),
            Command::Source(SourceCommand::List)
        ));
        assert!(matches!(
            command(&["source", "show", "ab12"]),
            Command::Source(SourceCommand::Show { .. })
        ));

        let Command::Tasks(TasksCommand::Generate {
            sources,
            kinds,
            generator,
        }) = command(&[
            "tasks",
            "generate",
            "ab12",
            "cd34",
            "--kinds",
            "recall,construct",
        ])
        else {
            panic!("tasks generate");
        };
        assert_eq!((sources.len(), kinds.len()), (2, 2));
        assert_eq!(generator, ModelRef::policy_default());
        assert!(
            parse(&["tasks", "generate", "ab12"]).is_err(),
            "--kinds is required"
        );

        let Command::Solve(solve) =
            command(&["solve", "ab12", "--solver", "local:Qwen/Qwen3-1.7B"])
        else {
            panic!("solve");
        };
        assert_eq!(solve.solver.to_string(), "local:Qwen/Qwen3-1.7B");
        assert!(matches!(command(&["verify", "ab12"]), Command::Verify(v) if v.judge.is_none()));
        let Command::Critique(critique) = command(&["critique", "ab12", "--retry", "3"]) else {
            panic!("critique");
        };
        assert_eq!(critique.retry, 3);
        assert!(
            parse(&["judge", "calibrate", "labels.jsonl"]).is_err(),
            "--judge is required"
        );
        assert!(matches!(
            command(&["judge", "calibrate", "labels.jsonl", "--judge", "local:/j"]),
            Command::Judge(_)
        ));
        assert!(matches!(
            command(&["experiences", "show", "ab12", "--graph"]),
            Command::Experiences(ExperiencesCommand::Show { graph: true, .. })
        ));

        let Command::Dataset(DatasetCommand::Build {
            sets,
            view,
            strip,
            min_strength,
            export_only,
        }) = command(&[
            "dataset",
            "build",
            "ab12",
            "cd34",
            "--view",
            "preference",
            "--strip",
            "mix:0.5",
            "--min-strength",
            "formal",
            "--export-only",
        ])
        else {
            panic!("dataset build");
        };
        assert_eq!(sets.len(), 2);
        assert_eq!(view, ViewName::Preference);
        assert!(matches!(strip, Some(Strip::Mix { .. })));
        assert_eq!(min_strength, Some(Strength::Formal));
        assert!(export_only);
        assert!(parse(&["dataset", "build", "ab12", "--view", "sft"]).is_err());
        assert!(
            parse(&["dataset", "export", "ab12"]).is_err(),
            "--out is required"
        );

        let Command::Train(train) = command(&[
            "train",
            "ab12",
            "cd34",
            "--from",
            "local:./ckpt+a.safetensors",
            "--replay",
            "ef56",
            "--steps",
            "10",
            "--rank",
            "4",
        ]) else {
            panic!("train");
        };
        assert_eq!(train.datasets.len(), 2);
        assert_eq!((train.steps, train.rank), (10, 4));
        assert!(matches!(
            command(&["runs", "cancel", "run-1"]),
            Command::Runs(RunsCommand::Cancel { .. })
        ));
    }

    #[test]
    fn every_command_takes_the_global_flags() {
        for args in [
            &["status"][..],
            &["learn", "docs"],
            &["ask", "q"],
            &["source", "list"],
            &["tasks", "list"],
            &["solve", "ab12"],
            &["verify", "ab12"],
            &["critique", "ab12"],
            &["judge", "calibrate", "f", "--judge", "policy:default"],
            &["experiences", "list"],
            &["dataset", "export", "ab12", "--out", "d"],
            &["train", "ab12"],
            &["runs", "list"],
        ] {
            let flags = ["--json", "-v", "--allow-remote", "--state", "s"];
            let after: Vec<&str> = args.iter().chain(&flags).copied().collect();
            let before: Vec<&str> = flags.iter().chain(args).copied().collect();
            for with_flags in [after, before] {
                let cli = parse(&with_flags).unwrap_or_else(|e| panic!("{with_flags:?}: {e}"));
                assert!(cli.command.is_some(), "{with_flags:?}");
                assert!(cli.global.json && cli.global.allow_remote, "{with_flags:?}");
                assert_eq!(cli.global.state, Some(PathBuf::from("s")));
            }
        }
    }

    #[test]
    fn model_references_are_parsed_by_the_one_parser() {
        let refused = parse(&["solve", "ab12", "--solver", "Qwen3"]).unwrap_err();
        assert!(
            refused.to_string().contains("remote:<provider>/<name>"),
            "{refused}"
        );
        assert!(parse(&["ask", "q", "--policy", "policy:champion"]).is_err());
        // Parsing accepts a remote reference; using one needs the opt-in.
        assert!(parse(&["ask", "q", "--policy", "remote:openrouter/z-ai/glm"]).is_ok());
    }

    #[test]
    fn the_old_commands_are_gone() {
        for removed in [
            &["run", "--workspace", "w", "--task", "t"][..],
            &["resume", "--run", "r"],
            &["show", "--run", "r"],
            &["cancel", "--run", "r"],
            &["learn", "--run", "r"],
            &["train", "--dataset", "d.jsonl"],
            &["explore", "--file", "f.md", "--out", "o.jsonl"],
            &["ask", "--question", "q"],
            &["eval-facts", "--dataset", "d", "--out", "o"],
            &["facts", "--file", "f.md"],
        ] {
            assert!(parse(removed).is_err(), "{removed:?} still parses");
        }
    }
}
