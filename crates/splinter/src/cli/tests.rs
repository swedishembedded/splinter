// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The grammar's specs: every verb, its positionals and flags.

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
    assert_eq!(per_task, DEFAULT_VARIANTS_PER_TASK);
    assert!(matches!(
        command(&["tasks", "variants", "ab12", "--per-task", "5"]),
        Command::Tasks(TasksCommand::Variants { per_task: 5, .. })
    ));

    let Command::Solve(solve) = command(&["solve", "ab12", "--solver", "local:Qwen/Qwen3-1.7B"])
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
        "--replay-fraction",
        "0.5",
        "--steps",
        "10",
        "--rank",
        "4",
        "--beta",
        "0.2",
    ]) else {
        panic!("train");
    };
    assert_eq!(train.datasets.len(), 2);
    assert_eq!((train.steps, train.rank, train.beta), (10, 4, Some(0.2)));
    assert_eq!(train.replay_fraction, 0.5);

    let Command::Release(release) = command(&["release", "candidate-1", "--alias", "staging"])
    else {
        panic!("release");
    };
    assert_eq!(release.candidate.as_deref(), Some("candidate-1"));
    assert_eq!(release.alias, "staging");
    assert!(release.command.is_none());
    let Command::Release(listing) = command(&["release", "list"]) else {
        panic!("release list");
    };
    assert!(matches!(listing.command, Some(ReleaseCommand::List)));
    assert_eq!(listing.alias, POLICY_DEFAULT);
    assert!(matches!(
        command(&["rollback", "default"]),
        Command::Rollback { alias } if alias == "default"
    ));
    assert!(parse(&["rollback"]).is_err(), "rollback needs an alias");
    let Command::Eval(eval) = command(&["eval", "policy:default"]) else {
        panic!("eval");
    };
    assert_eq!(eval.suite, SuiteChoice::HeldOut);
    let Command::Eval(eval) = command(&["eval", "--suite", "anchor", "--freeze", "a.jsonl"]) else {
        panic!("eval anchor");
    };
    assert_eq!((eval.model, eval.suite), (None, SuiteChoice::Anchor));
    let Command::Eval(eval) = command(&["eval", "c1", "--suite", "tasks.jsonl"]) else {
        panic!("eval file");
    };
    assert_eq!(eval.suite, SuiteChoice::File("tasks.jsonl".into()));
    let Command::Learn(learn) = command(&["learn", "docs", "--no-release"]) else {
        panic!("learn");
    };
    assert!(learn.no_release);
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
fn pass_at_k_is_asked_for_where_it_is_measured() {
    let Command::Solve(solve) = command(&[
        "solve",
        "ab12",
        "--frontier",
        "--k",
        "8",
        "--temperature",
        "1.0",
    ]) else {
        panic!("solve --frontier");
    };
    assert!(solve.frontier);
    let pass_at_k = solve.pass_at_k.pass_at_k();
    assert_eq!(pass_at_k.k, 8);
    assert_eq!(pass_at_k.sampling.map(|s| s.temperature), Some(1.0));
    let Command::Solve(plain) = command(&["solve", "ab12", "--frontier"]) else {
        panic!("solve --frontier");
    };
    assert_eq!(plain.pass_at_k.pass_at_k(), PassAtK::default());
    assert!(
        parse(&["solve", "ab12", "--k", "8"]).is_err(),
        "pass@k parameters need --frontier"
    );
    let Command::Solve(taught) =
        command(&["solve", "ab12", "--frontier", "--teacher", "local:./big"])
    else {
        panic!("solve --frontier --teacher");
    };
    assert_eq!(taught.teacher, Some("local:./big".parse().unwrap()));
    assert_eq!(plain.teacher, None, "the solver teaches by default");
    assert!(
        parse(&["solve", "ab12", "--teacher", "local:./big"]).is_err(),
        "a teacher needs --frontier"
    );

    let Command::Learn(learn) = command(&["learn", "docs", "--k", "6"]) else {
        panic!("learn --k");
    };
    assert!(!learn.no_frontier, "the frontier is measured by default");
    assert_eq!(learn.pass_at_k.pass_at_k().k, 6);
    let Command::Learn(learn) = command(&["learn", "docs", "--no-frontier"]) else {
        panic!("learn --no-frontier");
    };
    assert!(learn.no_frontier);
    assert!(parse(&["learn", "docs", "--no-frontier", "--k", "6"]).is_err());
    let Command::Learn(learn) = command(&["learn", "docs", "--teacher", "local:./big"]) else {
        panic!("learn --teacher");
    };
    assert_eq!(learn.teacher, Some("local:./big".parse().unwrap()));
    assert_eq!(
        learn.generator, None,
        "the policy writes the tasks by default"
    );
    let Command::Learn(learn) = command(&["learn", "docs", "--generator", "local:./big"]) else {
        panic!("learn --generator");
    };
    assert_eq!(learn.generator, Some("local:./big".parse().unwrap()));
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
        &["release", "list"],
        &["release", "c1"],
        &["rollback", "default"],
        &["eval", "c1", "--suite", "anchor"],
        &["state", "status"],
        &["state", "maintain"],
        &["state", "maintain", "--collect"],
        &["state", "verify", "--deep"],
        &["state", "repair", "--from", "a.tar.zst"],
        &["state", "archive", "a.tar.zst"],
        &["state", "restore", "a.tar.zst"],
        &["runs", "list"],
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
    let refused = parse(&["solve", "ab12", "--solver", "Qwen3"]).unwrap_err();
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
        &["train", "--dataset", "d.jsonl"],
        &["train", "ab12", "--replay", "cd34"],
        &["explore", "--file", "f.md", "--out", "o.jsonl"],
        &["ask", "--question", "q"],
        &["eval-facts", "--dataset", "d", "--out", "o"],
        &["facts", "--file", "f.md"],
    ] {
        assert!(parse(removed).is_err(), "{removed:?} still parses");
    }
}
