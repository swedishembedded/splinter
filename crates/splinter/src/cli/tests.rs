// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The grammar's specs: every verb, its positionals and flags.

use super::*;
use clap::CommandFactory;
use splinter_sdk::lineage::Direction;

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
fn the_help_names_the_first_commands_to_run_and_the_worked_example() {
    let help = Cli::command().render_long_help().to_string();
    for first in [
        "splinter doctor",
        "splinter init",
        "splinter status",
        "splinter ask",
        "examples/first-run",
    ] {
        assert!(
            help.contains(first),
            "the help does not mention {first}:\n{help}"
        );
    }
}

#[test]
fn doctor_takes_no_arguments() {
    assert!(matches!(command(&["doctor"]), Command::Doctor));
    assert!(parse(&["doctor", "extra"]).is_err());
}

#[test]
fn a_sentence_or_nothing_is_the_front_door() {
    let cli = parse(&["summarise the docs in ./docs"]).unwrap();
    assert_eq!(
        cli.sentence.as_deref(),
        Some("summarise the docs in ./docs")
    );
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
    let Command::Ask(ask) = command(&["ask", "what does X do?", "--open-book", "ab12"]) else {
        panic!("ask");
    };
    assert_eq!(ask.question, "what does X do?");
    assert_eq!(
        ask.policy,
        ModelRef::policy_default(),
        "closed-book on the policy"
    );

    let Command::Source(SourceCommand::Add { target, .. }) =
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

    let Command::Tasks(TasksCommand::Variants {
        task_set,
        generator,
        per_task,
    }) = command(&["tasks", "variants", "ab12"])
    else {
        panic!("tasks variants");
    };
    assert_eq!(task_set, "ab12");
    assert_eq!(generator, ModelRef::policy_default());
    assert_eq!(per_task, splinter_sdk::variants::DEFAULT_VARIANTS_PER_TASK);
    assert!(matches!(
        command(&["tasks", "variants", "ab12", "--per-task", "5"]),
        Command::Tasks(TasksCommand::Variants { per_task: 5, .. })
    ));

    assert!(matches!(
        command(&["verify", "runs/r1"]),
        Command::Verify(v) if v.run == std::path::Path::new("runs/r1")
    ));
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

    assert!(matches!(
        command(&["experiences", "compare", "ab12", "cd34", "--interval-seed", "9"]),
        Command::Experiences(ExperiencesCommand::Compare { ref candidate, ref baseline, interval_seed: Some(_), .. })
            if candidate == "ab12" && baseline == "cd34"
    ));
    assert!(parse(&["experiences", "compare", "ab12"]).is_err());

    assert!(
        parse(&["dataset", "export", "ab12"]).is_err(),
        "--out is required"
    );

    assert!(matches!(
        command(&["release", "list"]),
        Command::Release(ReleaseCommand::List)
    ));
    assert!(matches!(
        command(&["rollback", "default"]),
        Command::Rollback { alias } if alias == "default"
    ));
    assert!(parse(&["rollback"]).is_err(), "rollback needs an alias");
    let Command::Eval(eval) = command(&["eval", "--freeze", "a.jsonl"]) else {
        panic!("eval --freeze");
    };
    assert_eq!((eval.run, eval.freeze.len()), (None, 1));
    let Command::Eval(eval) = command(&["eval", "runs/r1", "--plan", "p.yaml", "--out", "o"])
    else {
        panic!("eval of a run");
    };
    assert!(eval.run.is_some() && eval.plan.is_some());
    assert!(
        parse(&["eval", "--plan", "p.yaml", "--out", "o"]).is_err(),
        "a plan evaluates a run"
    );
    let Command::Eval(eval) = command(&["eval", "--suite", "probes.jsonl"]) else {
        panic!("eval --suite");
    };
    assert_eq!(eval.suite, Some("probes.jsonl".into()));
    assert_eq!(
        eval.policy.to_string(),
        "policy:default",
        "the default policy is graded"
    );
    let Command::Eval(eval) = command(&["eval", "--suite", "p.jsonl", "--policy", "local:m"])
    else {
        panic!("eval --suite --policy");
    };
    assert_eq!(eval.policy.to_string(), "local:m");
    for mixed in [
        &["eval", "runs/r1", "--suite", "p.jsonl"][..],
        &["eval", "--suite", "p.jsonl", "--freeze", "a.jsonl"],
        &["eval", "--policy", "local:m"],
    ] {
        assert!(parse(mixed).is_err(), "{mixed:?} parses");
    }
    let Command::Exam(exam) = command(&["exam", "c1", "--judge", "local:Qwen/Qwen3-8B"]) else {
        panic!("exam");
    };
    assert_eq!(exam.candidate, "c1");
    assert!(exam.judge.is_some());
    assert!(parse(&["exam"]).is_err(), "an exam needs a candidate");
    let Command::Exam(powered) = command(&[
        "exam",
        "c1",
        "--exam-set",
        "e1",
        "--resamples",
        "5",
        "--no-voice",
    ]) else {
        panic!("exam");
    };
    assert_eq!(powered.exam_set.as_deref(), Some("e1"));
    assert_eq!((powered.resamples, powered.no_voice), (Some(5), true));
    assert!(!powered.deployed_only);
    let Command::Exam(deployed) = command(&["exam", "c1", "--exam-set", "e1", "--deployed-only"])
    else {
        panic!("exam");
    };
    assert!(deployed.deployed_only);
    assert!(parse(&["exam", "c1", "--deployed-only"]).is_err());
    assert!(
        parse(&["exam", "c1", "--resamples", "3"]).is_err(),
        "resamples belong to a frozen exam"
    );
    assert!(parse(&["exam", "c1", "--exam-set", "e1", "--resamples", "0"]).is_err());
    let Command::ExamSet(ExamSetCommand::Create(created)) = command(&[
        "exam-set",
        "create",
        "./materials",
        "--families",
        "12",
        "--not-trained-by",
        "c1",
        "--not-trained-by",
        "c2",
    ]) else {
        panic!("exam-set create");
    };
    assert_eq!((created.families, created.not_trained_by.len()), (12, 2));
    assert_eq!(created.kinds, ["converse", "advise", "explain"]);
    assert!(matches!(
        command(&["runs", "cancel", "run-1"]),
        Command::Runs(RunsCommand::Cancel { .. })
    ));
    let Command::State(StateCommand::Maintain { collect }) =
        command(&["state", "maintain", "--collect"])
    else {
        panic!("state maintain");
    };
    assert!(collect);
    assert!(matches!(
        command(&["state", "maintain"]),
        Command::State(StateCommand::Maintain { collect: false })
    ));
    assert!(matches!(
        command(&["state", "unpin", "dataset-ab"]),
        Command::State(StateCommand::Unpin { .. })
    ));
    assert!(matches!(
        command(&["state", "verify"]),
        Command::State(StateCommand::Verify { deep: false })
    ));
    let Command::State(StateCommand::Repair { from, accept_loss }) = command(&[
        "state",
        "repair",
        "--from",
        "a.tar.zst",
        "--from",
        "other",
        "--accept-loss",
    ]) else {
        panic!("state repair");
    };
    assert_eq!(from, [PathBuf::from("a.tar.zst"), PathBuf::from("other")]);
    assert!(accept_loss);
    let Command::State(StateCommand::Archive {
        file,
        no_artifacts,
        since,
    }) = command(&["state", "archive", "new.tar.zst", "--since", "old.tar.zst"])
    else {
        panic!("state archive");
    };
    assert_eq!(
        (file, no_artifacts, since),
        (
            PathBuf::from("new.tar.zst"),
            false,
            Some(PathBuf::from("old.tar.zst"))
        )
    );
    let Command::State(StateCommand::Restore { files }) =
        command(&["state", "restore", "new.tar.zst", "old.tar.zst"])
    else {
        panic!("state restore");
    };
    assert_eq!(files.len(), 2);
    assert!(
        parse(&["state", "restore"]).is_err(),
        "restore needs an archive"
    );
    let Command::Lineage(lineage) = command(&["lineage", "ab12"]) else {
        panic!("lineage");
    };
    assert_eq!(
        (lineage.direction(), lineage.depth),
        (Direction::Both, None)
    );
    let Command::Lineage(lineage) = command(&["lineage", "ab12", "--up", "--depth", "2"]) else {
        panic!("lineage --up");
    };
    assert_eq!(
        (lineage.direction(), lineage.depth),
        (Direction::Up, Some(2))
    );
    assert!(parse(&["lineage", "ab12", "--up", "--down"]).is_err());
    assert!(parse(&["lineage"]).is_err(), "lineage needs an id");
}

#[test]
fn every_command_takes_the_global_flags() {
    for args in [
        &["status"][..],
        &["ask", "q"],
        &["source", "list"],
        &["tasks", "list"],
        &["verify", "runs/r1"],
        &["critique", "ab12"],
        &["judge", "calibrate", "f", "--judge", "policy:default"],
        &["experiences", "list"],
        &["dataset", "export", "ab12", "--out", "d"],
        &["release", "list"],
        &["rollback", "default"],
        &["eval"],
        &["eval", "--freeze", "a.jsonl"],
        &["eval", "--suite", "p.jsonl"],
        &["state", "status"],
        &["state", "maintain"],
        &["state", "maintain", "--collect"],
        &["state", "verify", "--deep"],
        &["state", "repair", "--from", "a.tar.zst"],
        &["state", "archive", "a.tar.zst"],
        &["state", "restore", "a.tar.zst"],
        &["runs", "list"],
        &["jobs", "list"],
        &["lineage", "ab12"],
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
    let refused = parse(&["ask", "q", "--policy", "Qwen3"]).unwrap_err();
    assert!(
        refused.to_string().contains("remote:<provider>/<name>"),
        "{refused}"
    );
    assert!(parse(&["ask", "q", "--policy", "policy:Champion"]).is_err());
    assert!(parse(&["ask", "q", "--policy", "policy:staging"]).is_ok());
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
        &["learn", "docs"],
        &["practice", "."],
        &["harness", "optimise", "ab12", "--baseline", "b.yaml"],
        &["train", "--dataset", "d.jsonl"],
        &["train", "ab12", "--replay", "cd34"],
        &["explore", "--file", "f.md", "--out", "o.jsonl"],
        &["ask", "--question", "q"],
        &["eval-facts", "--dataset", "d", "--out", "o"],
        &["facts", "--file", "f.md"],
        // The legacy pipeline verbs: the factory's operations do each.
        &["solve", "ab12"],
        &["rehearse", "--records", "120"],
        &["dataset", "build", "ab12"],
        &["tasks", "generate", "ab12", "--kinds", "recall"],
        &["release", "candidate-1"],
        &["eval", "policy:default", "--suite", "held-out"],
        &["verify", "ab12", "--judge", "local:/j"],
    ] {
        assert!(parse(removed).is_err(), "{removed:?} still parses");
    }
}

mod absorb;
mod env;
mod flags;
mod jobs;
mod measure;
