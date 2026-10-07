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

    let Command::Tasks(TasksCommand::Generate {
        sources,
        kinds,
        generator,
        author,
    }) = command(&[
        "tasks",
        "generate",
        "ab12",
        "cd34",
        "--kinds",
        "recall,construct",
        "--author",
        "Thomas Jefferson",
    ])
    else {
        panic!("tasks generate");
    };
    assert_eq!((sources.len(), kinds.len()), (2, 2));
    assert_eq!(generator, ModelRef::policy_default());
    assert_eq!(author.as_deref(), Some("Thomas Jefferson"));
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
        system_prompt,
        export_only,
        limit,
        writer,
        token_limit,
        max_family_share,
        describe_with,
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
        "--system-prompt",
        "You are a clerk.",
        "--export-only",
        "--limit",
        "5",
        "--writer",
        "The Writer",
        "--token-limit",
        "9000",
        "--max-family-share",
        "0.2",
        "--describe-with",
        "local:Qwen/Qwen3-8B",
    ])
    else {
        panic!("dataset build");
    };
    assert_eq!(
        (writer.as_deref(), token_limit, max_family_share),
        (Some("The Writer"), Some(9000), Some(0.2))
    );
    assert!(describe_with.is_some());
    assert_eq!(sets.len(), 2);
    assert_eq!(view, ViewName::Preference);
    assert!(matches!(strip, Some(Strip::Mix { .. })));
    assert_eq!(min_strength, Some(Strength::Formal));
    assert_eq!(system_prompt.as_deref(), Some("You are a clerk."));
    assert!(export_only);
    assert_eq!(limit, Some(5));
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
    assert_eq!(
        (train.steps, train.rank, train.beta),
        (Some(10), 4, Some(0.2))
    );
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
    let Command::Train(train) = command(&["train", "d1", "--lr", "0.0002"]) else {
        panic!("train");
    };
    assert_eq!(train.optimiser.lr, Some(0.0002));
    assert!(
        train.steps.is_none()
            && train.monitoring.eval_every.is_none()
            && train.monitoring.patience.is_none(),
        "the budget, the cadence and the patience follow the data unless named"
    );
    let Command::Train(train) = command(&[
        "train",
        "d1",
        "--eval-every",
        "0",
        "--patience",
        "2",
        "--monitor-share",
        "0.2",
    ]) else {
        panic!("train");
    };
    assert_eq!(
        (
            train.monitoring.eval_every,
            train.monitoring.patience,
            train.monitoring.monitor_share
        ),
        (Some(0), Some(2), Some(0.2))
    );
    assert!(
        parse(&["train", "d1", "--monitor-share", "0.6"]).is_err(),
        "more than half monitored is refused"
    );
    assert!(parse(&["train", "d1", "--monitor-share", "0"]).is_err());
    let Command::Learn(learn) =
        command(&["learn", "docs", "--patience", "0", "--monitor-share", "0.5"])
    else {
        panic!("learn");
    };
    assert_eq!(
        (learn.monitoring.patience, learn.monitoring.monitor_share),
        (Some(0), Some(0.5))
    );
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
    let Command::Learn(reserving) = command(&[
        "learn",
        "docs",
        "--exam-families",
        "0",
        "--exam-tasks-per-family",
        "50",
    ]) else {
        panic!("learn");
    };
    assert_eq!(
        (reserving.exam_families, reserving.exam_tasks_per_family),
        (Some(0), Some(50))
    );
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

#[test]
fn learn_can_distill_and_tune_the_training_it_runs() {
    let Command::Learn(plain) = command(&["learn", "docs"]) else {
        panic!("learn");
    };
    assert!(!plain.distill && !plain.bf16_base);
    assert_eq!(
        (plain.steps, plain.rank, plain.optimiser.lr),
        (None, None, None)
    );

    let Command::Learn(tuned) = command(&[
        "learn",
        "docs",
        "--distill",
        "--steps",
        "300",
        "--rank",
        "16",
        "--lr",
        "0.0002",
        "--bf16-base",
    ]) else {
        panic!("learn --distill");
    };
    assert!(tuned.distill && tuned.bf16_base);
    assert_eq!((tuned.steps, tuned.rank), (Some(300), Some(16)));
    assert_eq!(tuned.optimiser.lr, Some(0.0002));
    assert!(
        parse(&["learn", "docs", "--distill", "--k", "6"]).is_err(),
        "distilling makes no attempts to measure pass@k over"
    );
    assert!(
        parse(&["learn", "docs", "--steps", "0"]).is_err(),
        "no steps is no training"
    );
}

#[test]
fn a_rehearsal_is_built_by_its_own_verb_and_mixed_in_by_train() {
    let Command::Rehearse(args) = command(&["rehearse", "--records", "120"]) else {
        panic!("rehearse");
    };
    assert_eq!(args.records, 120);
    assert!(
        parse(&["rehearse"]).is_err(),
        "the size is the caller's to say"
    );
    assert!(parse(&["rehearse", "--records", "0"]).is_err());

    let Command::Train(plain) = command(&["train", "ds-1"]) else {
        panic!("train");
    };
    assert_eq!(
        (plain.rehearsal, plain.rehearsal_share, plain.seed),
        (None, None, None)
    );
    let Command::Train(mixed) = command(&[
        "train",
        "ds-1",
        "--rehearsal",
        "ds-2",
        "--rehearsal-share",
        "0.3",
        "--seed",
        "1338",
    ]) else {
        panic!("train --rehearsal");
    };
    assert_eq!(mixed.rehearsal.as_deref(), Some("ds-2"));
    assert_eq!((mixed.rehearsal_share, mixed.seed), (Some(0.3), Some(1338)));
    assert!(
        parse(&["train", "ds-1", "--rehearsal-share", "0.3"]).is_err(),
        "a share of nothing"
    );
    for bad in ["0", "1", "1.5"] {
        assert!(
            parse(&[
                "train",
                "ds-1",
                "--rehearsal",
                "ds-2",
                "--rehearsal-share",
                bad
            ])
            .is_err(),
            "{bad}"
        );
    }
    let Command::Learn(learn) = command(&["learn", "docs", "--rehearsal", "0"]) else {
        panic!("learn");
    };
    assert_eq!(learn.rehearsal, Some(0.0), "0 turns it off");
    assert!(parse(&["learn", "docs", "--rehearsal", "1"]).is_err());
}

#[test]
fn learn_names_the_model_that_plans() {
    let Command::Learn(plain) = command(&["learn", "docs"]) else {
        panic!("learn");
    };
    assert_eq!(plain.planner, None, "the generator plans by default");
    let Command::Learn(named) = command(&["learn", "docs", "--planner", "local:./big"]) else {
        panic!("learn --planner");
    };
    assert_eq!(named.planner, Some("local:./big".parse().unwrap()));
}

#[test]
fn passages_are_a_share_in_zero_to_one_not_zero() {
    let Command::Learn(learn) = command(&["learn", "docs", "--with-passages", "0.4"]) else {
        panic!("learn");
    };
    assert_eq!(learn.with_passages, Some(0.4));
    assert!(parse(&["learn", "docs", "--with-passages", "0"]).is_err());
    assert!(parse(&["learn", "docs", "--with-passages", "1.5"]).is_err());
    assert!(parse(&["learn", "docs", "--with-passages", "most"]).is_err());
}

#[test]
fn abstentions_are_taken_from_the_records_given_passages() {
    let Command::Learn(learn) = command(&[
        "learn",
        "docs",
        "--with-passages",
        "0.5",
        "--abstain",
        "0.1",
    ]) else {
        panic!("learn");
    };
    assert_eq!(learn.abstain, Some(0.1));
    assert!(parse(&["learn", "docs", "--abstain", "0.1"]).is_err());
}
