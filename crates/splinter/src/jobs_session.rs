// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements operator commands for work that outlives the
// process that submitted it, for its clients. If your team needs expertise in
// reporting detached job state or reconciling a restarted controller, you can
// procure our services by sending an email to info@swedishembedded.com.

//! `jobs ...`: submit, list, reconcile, show and cancel jobs through the SDK,
//! and print what each did; `jobs run-served` runs one action on the brain
//! servers.

use serde::Serialize;

use splinter_sdk::jobs::{JobSubmission, JobView};
use splinter_sdk::model::served_train::ServedJobReport;
use splinter_sdk::runs::{record, Recorded};
use splinter_sdk::served_job::ServedJobRequest;
use splinter_sdk::store::jobs::JobRecord;
use splinter_sdk::Context;
use splinter_sdk::{Error, Splinter};

use crate::cli::{JobsArgs, JobsCommand, RunServedArgs};
use crate::output::{self, shell_words, Report};
use crate::session::Exit;

/// Runs `jobs`.
pub fn jobs(splinter: &Splinter, json: bool, args: JobsArgs) -> Result<Exit, Error> {
    let Some(command) = args.command else {
        print!("{}", crate::cli::jobs_help());
        return Ok(Exit::Ok);
    };
    let jobs = splinter.jobs();
    match command {
        JobsCommand::Submit(args) => output::emit(json, &jobs.submit(&args.spec())?),
        JobsCommand::List => output::emit(json, &Listed { jobs: jobs.list()? }),
        JobsCommand::Reconcile => output::emit(
            json,
            &Reconciled {
                jobs: jobs.reconcile()?,
            },
        ),
        JobsCommand::Retry => output::emit(
            json,
            &Retried {
                jobs: jobs.retry()?,
            },
        ),
        JobsCommand::Show { id } => output::emit(json, &jobs.show(&id)?),
        JobsCommand::RunServed(_) => {
            unreachable!("jobs run-served runs on a context and is routed before this match")
        }
        JobsCommand::Cancel { id } => output::emit(
            json,
            &Cancelled {
                job: jobs.cancel(&id)?,
            },
        ),
    }
    Ok(Exit::Ok)
}

/// The jobs a list or a reconciliation reported.
#[derive(Debug, Serialize)]
pub struct Listed {
    /// The jobs, in specification order.
    pub jobs: Vec<JobRecord>,
}

/// The records a reconciliation changed.
#[derive(Debug, Serialize)]
pub struct Reconciled {
    /// The records, in specification order.
    pub jobs: Vec<JobRecord>,
}

/// The attempts a retry launched.
#[derive(Debug, Serialize)]
pub struct Retried {
    /// The records of the attempts started, in specification order.
    pub jobs: Vec<JobRecord>,
}

/// The job a cancel ended.
#[derive(Debug, Serialize)]
pub struct Cancelled {
    /// The job's record.
    pub job: JobRecord,
}

/// One job as a line: its digest, state, attempt, launcher and detail.
fn line(job: &JobRecord) -> String {
    format!(
        "{}  {:<12}  attempt {}  {:<8}  {}\n",
        job.spec.as_str(),
        job.state.as_str(),
        job.attempt,
        job.launcher,
        job.detail.as_deref().unwrap_or("")
    )
}

/// The lines every job report shares: where it was launched and how it ended.
fn detail(job: &JobRecord) -> String {
    let mut out = String::new();
    if let Some(launch) = &job.launch {
        out.push_str(&format!("  launch: {launch}\n"));
    }
    if let Some(code) = job.exit_code {
        out.push_str(&format!("  exit: {code}\n"));
    }
    out
}

fn listing(jobs: &[JobRecord], empty: &str) -> String {
    if jobs.is_empty() {
        return format!("{empty}\n");
    }
    jobs.iter().map(line).collect()
}

impl Report for JobSubmission {
    fn human(&self) -> String {
        let note = if self.launched {
            "launched"
        } else {
            "already submitted; nothing was launched"
        };
        let mut out = line(&self.job);
        out.push_str(&format!("  {note}\n"));
        out.push_str(&detail(&self.job));
        out
    }
}

impl Report for JobView {
    fn human(&self) -> String {
        let mut out = line(&self.job);
        match &self.spec {
            Some(spec) => {
                out.push_str(&format!("  command: {}\n", shell_words(&spec.command)));
                out.push_str(&format!(
                    "  workers: {} node(s) x {} rank(s), {} gpu(s) each\n",
                    spec.workers.nodes, spec.workers.ranks_per_node, spec.workers.gpus_per_rank
                ));
                if spec.wall_clock_seconds > 0 {
                    out.push_str(&format!("  wall clock: {}s\n", spec.wall_clock_seconds));
                }
                out.push_str(&format!("  attempts allowed: {}\n", spec.retry.attempts));
            }
            None => out.push_str("  specification: not stored\n"),
        }
        out.push_str(&detail(&self.job));
        out
    }
}

impl Report for Listed {
    fn human(&self) -> String {
        listing(&self.jobs, "no jobs submitted")
    }
}

impl Report for Reconciled {
    fn human(&self) -> String {
        listing(&self.jobs, "no unfinished job changed state")
    }
}

impl Report for Retried {
    fn human(&self) -> String {
        listing(&self.jobs, "no job was ready for another attempt")
    }
}

impl Report for Cancelled {
    fn human(&self) -> String {
        let mut out = line(&self.job);
        out.push_str(&detail(&self.job));
        out
    }
}

/// A served job's report, under the run that recorded it.
#[derive(Debug, Serialize)]
pub struct RanServed(pub Recorded<ServedJobReport>);

/// Runs `jobs run-served`: rank 0's step frames on stdout as JSON lines as
/// they come, then the report (with `--json`, as one more line, the one
/// that has a `run`).
pub fn run_served(ctx: &Context, json: bool, args: RunServedArgs) -> Result<Exit, Error> {
    let request = ServedJobRequest {
        action: args.action,
        params: args.params,
        artifacts: args.artifacts,
        out_dir: args.out,
    };
    let ran = record(ctx, "jobs run-served", &request, |run| {
        let cancel = run.cancel_token();
        splinter_sdk::served_job::run(ctx, &request, &cancel, &mut |frame| {
            println!("{frame}");
        })
    })?;
    let report = RanServed(ran);
    if json {
        match serde_json::to_string(&report) {
            Ok(line) => println!("{line}"),
            Err(e) => eprintln!("splinter: the report could not be written as JSON: {e}"),
        }
    } else {
        print!("{}", report.human());
    }
    Ok(Exit::Ok)
}

impl Report for RanServed {
    fn human(&self) -> String {
        let job = &self.0.report;
        let asked = &job.parallel.requested;
        let mut out = format!(
            "{}: completed in {:.1}s (run {})\n  spread: {} rank(s) over {} server(s), {} on the first\n",
            job.action, job.seconds, self.0.run, asked.world_size, asked.servers, asked.first_server_ranks
        );
        if let Some(seed) = job.seed {
            let how = if seed.drawn { " (drawn)" } else { "" };
            out.push_str(&format!("  seed: {}{how}\n", seed.value));
        }
        match &job.parallel.reported {
            Some(got) => out.push_str(&format!(
                "  parallelism: world {} over {}, transport {}\n",
                got.world_size,
                got.collective.as_deref().unwrap_or("unreported"),
                got.transport.as_deref().unwrap_or("unreported")
            )),
            None => out.push_str("  parallelism: not reported\n"),
        }
        for warning in job.parallel.warnings() {
            out.push_str(&format!("  WARNING {warning}\n"));
        }
        for field in ["steps_completed", "final_loss", "checkpoint_digest"] {
            if let Some(value) = job.result.get(field).filter(|v| !v.is_null()) {
                out.push_str(&format!("  {field}: {value}\n"));
            }
        }
        if let Some(first) = job.fetched.first() {
            let dir = first
                .path
                .parent()
                .map_or_else(String::new, |p| p.display().to_string());
            out.push_str(&format!(
                "  fetched: {} file(s), verified, under {dir}\n",
                job.fetched.len()
            ));
        }
        out
    }
}
