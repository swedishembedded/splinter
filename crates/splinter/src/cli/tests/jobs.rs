// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The grammar's specs for `jobs`: a submission takes its worker plan, the
//! wall clock and the attempt policy, and it takes the program it runs after
//! `--`, flags included.

use super::{command, parse, Command, JobsArgs, JobsCommand};

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
