// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The grammar's specs for `jobs`: a submission takes its worker plan, the
//! wall clock and the attempt policy, and it takes the program it runs after
//! `--`, flags included.

use super::{command, parse, Command, JobsArgs, JobsCommand};
use splinter_sdk::vocabulary::checkpoint::JOB_PLAN_FILE;
use splinter_sdk::vocabulary::job::{PlanPolicy, JOB_PLAN_FLAG, JOB_RANK_FLAG};

#[test]
fn a_submission_takes_its_worker_plan_wall_clock_and_attempts() {
    assert!(matches!(
        command(&["jobs"]),
        Command::Jobs(JobsArgs { command: None })
    ));
    let Command::Jobs(JobsArgs {
        command: Some(JobsCommand::Submit(submit)),
    }) = command(&[
        "jobs",
        "submit",
        "--nodes",
        "2",
        "--ranks-per-node",
        "4",
        "--gpus-per-rank",
        "1",
        "--wall-clock",
        "60",
        "--attempts",
        "3",
        "--",
        "/bin/sh",
        "-c",
        "sleep 0.1",
    ])
    else {
        panic!("jobs submit");
    };
    assert_eq!(submit.command, ["/bin/sh", "-c", "sleep 0.1"]);
    let spec = submit.spec();
    assert_eq!(spec.workers.world_size(), 8);
    assert_eq!(
        (spec.workers.gpus_per_rank, spec.wall_clock_seconds),
        (1, 60)
    );
    assert_eq!(spec.retry.attempts, 3);
    assert!(parse(&["jobs", "submit"]).is_err(), "a job needs a program");
    assert!(
        parse(&["jobs", "submit", "--", "prog", "--attempts", "0"]).is_ok(),
        "everything after -- is the program's"
    );
    assert!(
        parse(&["jobs", "submit", "--attempts", "0", "--", "prog"]).is_err(),
        "a job allows at least one attempt"
    );
}

#[test]
fn a_distributed_submission_prepares_a_plan_for_every_rank() {
    let Command::Jobs(JobsArgs {
        command: Some(JobsCommand::Submit(submit)),
    }) = command(&[
        "jobs",
        "submit",
        "--distributed",
        "--nodes",
        "1",
        "--ranks-per-node",
        "2",
        "--gpus-per-rank",
        "1",
        "--",
        "brain",
        "qwen3",
        "finetune",
    ])
    else {
        panic!("jobs submit --distributed");
    };
    let spec = submit.spec();
    assert_eq!(spec.plan, Some(PlanPolicy::default()));
    assert_eq!(spec.workers.world_size(), 2);
    let for_rank = spec.command_for(1);
    assert_eq!(
        for_rank[for_rank.len() - 4..],
        [JOB_PLAN_FLAG, JOB_PLAN_FILE, JOB_RANK_FLAG, "1"]
    );

    // Without the option a single-rank job asks for no plan at all.
    let Command::Jobs(JobsArgs {
        command: Some(JobsCommand::Submit(plain)),
    }) = command(&["jobs", "submit", "--", "brain", "qwen3", "finetune"])
    else {
        panic!("jobs submit");
    };
    let plain = plain.spec();
    assert_eq!(plain.plan, None);
    assert!(!plain.command.iter().any(|part| part == JOB_PLAN_FLAG));
}
