// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model releases proven to run on the
// plain serving stack before they ship, for its clients. If your team
// needs expertise in model deployment, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The serve check: a candidate is released only if plain `brain serve`
//! runs it.
//!
//! `brain serve --openai 127.0.0.1:<port> --adapter <candidate>` is started
//! with the policy's base checkpoint as its Qwen3 (`--checkpoint`) and
//! brain's model store (`--models-dir`, when it exists), in a runtime
//! directory of its own so no client of the user's finds it. Before it
//! binds, brain binds the adapter to the base and prints `brain serve:
//! <model> adapter=<id> digest=sha256:<hex>`; that digest must be the
//! candidate adapter's. Once
//! it reports ready (`--ready-file`), a sample of the held-out tasks is
//! re-answered through its OpenAI-compatible endpoint, with the key it
//! wrote (`--api-keys-out`), decoded greedily as the in-process arms
//! were ([`GREEDY_SAMPLING`]) so the two answers compare serving rather
//! than two draws and, as in-process, without a reasoning block, and
//! graded as in-process. A task is answered alike when the verdict is the
//! same and the final answer is the same text up to the numerical noise of
//! two processes decoding one model ([`alike`]) or, worded otherwise, says
//! the same thing ([`meaning::says_the_same`]: nearer in meaning to its own
//! in-process answer than to the answer to any other task, so the
//! comparison cannot call everything alike). A served model may sample, and
//! its kernels sum in another order, so the wording is not what must
//! survive; the meaning is. At least
//! [`gate::SERVE_AGREEMENT_PERCENT`] percent of the tasks must be. Each
//! task answered differently is reported with both answers. The server is
//! stopped when the check ends, however it ends.
//!
//! The server loads its own copy of the base, so every base this process
//! keeps resident is released before it starts. The answers are decided
//! only after the server has exited ([`ask_then_judge`]): the judge, a model
//! of its own, would otherwise run out of memory on the device the server
//! holds and every served verdict would be lost. A task the server gave no
//! answer to is not a disagreement: the check is not measured
//! ([`unanswered`]), and so is one whose server stopped answering, with what
//! the server last wrote. A judge may abstain on an answer; the verdict is
//! then undecided and the answers are compared by their text and meaning
//! ([`verdicts_compatible`]).
//!
//! A candidate that works in a workspace is served and asked the same way
//! ([`workspace`]): the same worker, with the served endpoint as its model,
//! over a sample of the exam, graded by the verifiers that graded the
//! in-process run; the candidate is offered to the server as the release
//! manifest a release step publishes, and the check passes when the served
//! verdicts agree with the in-process ones.
//!
//! Without a `brain` binary, or when it does not start, the check is not
//! measured - and the gate fails.

mod compare;
pub mod workspace;

use std::path::Path;
use std::time::Duration;

use splinter_agent::solve::Model;
use splinter_agent::CancelToken;
use splinter_model::local::GREEDY_SAMPLING;

use crate::brain_server::{self, recent_output, Failure, Launch, Offer, Running, Started};
use crate::release::probe::{answer_all, judge_all, Probe, Suite};
use compare::{agreements, unanswered};
use splinter_eval::gate::{self, Agreement, Check, Disagreement, Serve};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// What the serve check is asked to prove.
pub struct ServeCheck<'a> {
    /// The `brain` that serves; `None` when there is none to be found.
    pub binary: Option<&'a Path>,
    /// The base checkpoint the adapter sits on, or the consolidated
    /// checkpoint itself when there is no adapter.
    pub base: &'a Path,
    /// The candidate's adapter file; `None` for a consolidated checkpoint,
    /// which is served as the base with no adapter to bind.
    pub adapter: Option<&'a Path>,
    /// The digest the adapter must be served under; for a checkpoint with no
    /// adapter, the digest of its weights, which the server cannot report.
    pub adapter_digest: &'a str,
    /// The system prompt the candidate is asked under; `None` is the default.
    pub system: Option<&'a str>,
    /// The held-out tasks re-answered through the server. A suite of tasks
    /// that work in a workspace is asked by [`workspace::check`]; this one
    /// holds closed-book tasks.
    pub sample: &'a Suite,
    /// The candidate's in-process answers to them, one per task.
    pub in_process: &'a [Probe],
    /// How long the server may take to bind.
    pub startup: Duration,
}

/// Starts the `brain` of `request` serving its adapter on its base, and
/// re-answers its sample through it. An error is returned only for a
/// cancelled check; every other failure is the check's result.
///
/// A sample of tasks that work in a workspace is asked through
/// [`workspace::check`].
pub fn check(
    ctx: &Context,
    request: &ServeCheck<'_>,
    cancel: &CancelToken,
) -> Result<Check<Serve>, OrchestratorError> {
    if request.sample.workspace_tasks() > 0 {
        return workspace::check(ctx, request, cancel);
    }
    let Some(binary) = request.binary else {
        return Ok(Check::unmeasured(NO_BINARY));
    };
    // `brain serve` is another process loading its own copy of the base:
    // this one holds none while it runs.
    ctx.release_bases();
    let work = scratch(ctx)?;
    let result = serve_and_ask(
        ctx,
        request,
        &Spawn {
            binary,
            work: &work,
            models: &ctx.config().model_store,
        },
        cancel,
    );
    // The directory only held the keys and the ready marker.
    let _ = std::fs::remove_dir_all(&work);
    finished(result)
}

/// Why a check with no `brain` binary is not measured.
const NO_BINARY: &str = "no brain binary: none on PATH and no --brain-bin";

/// A fresh directory for one check's keys, ready marker and manifest.
fn scratch(ctx: &Context) -> Result<std::path::PathBuf, OrchestratorError> {
    let work = ctx.root().sandbox().join(format!(
        "serve-check-{}",
        splinter_store::new_id_with_prefix("gate")
    ));
    std::fs::create_dir_all(&work).map_err(splinter_orchestrator::error::io(&work))?;
    Ok(work)
}

/// A check's result, or the cancel that ended it.
fn finished(result: Result<Check<Serve>, Failure>) -> Result<Check<Serve>, OrchestratorError> {
    match result {
        Ok(check) => Ok(check),
        Err(Failure::Cancelled) => Err(OrchestratorError::Cancelled),
        Err(Failure::Unmeasured(why)) => Ok(Check::unmeasured(why)),
    }
}

/// How the server is started.
struct Spawn<'a> {
    binary: &'a Path,
    work: &'a Path,
    models: &'a Path,
}

/// The model the served endpoint is, asked as the candidate is: greedily,
/// under the prompt it was trained under, and as many requests at once as the
/// server was started to decode together.
fn served_model(started: &Started, system: Option<&str>) -> Result<Model, Failure> {
    let loaded = splinter_model::selection::served_model(
        &format!("http://{}/v1", started.address),
        &started.key,
        &started.model,
        Some(GREEDY_SAMPLING.temperature),
        None,
    )
    .map_err(|e| Failure::Unmeasured(format!("the served endpoint: {e}")))?;
    let served =
        Model::new(loaded.provider(), loaded.identity()).with_slots(started.slots as usize);
    Ok(match system {
        Some(system) => served.with_system(system),
        None => served,
    })
}

fn serve_and_ask(
    ctx: &Context,
    request: &ServeCheck<'_>,
    server: &Spawn<'_>,
    cancel: &CancelToken,
) -> Result<Check<Serve>, Failure> {
    let (sample, in_process) = (request.sample, request.in_process);
    let started = brain_server::start(
        &Launch {
            binary: server.binary,
            work: server.work,
            models: server.models,
            base: request.base,
            offer: request.adapter.map_or(Offer::Nothing, Offer::Adapter),
            expected_digest: request.adapter_digest,
            startup: request.startup,
            device: None,
            max_batch: None,
            kv_offload_gb: None,
            log: None,
        },
        cancel,
    )?;
    let mut measured = Serve {
        binary: server.binary.to_path_buf(),
        startup_line: started.startup_line.clone(),
        served_digest: started.served_digest.clone(),
        expected_digest: request.adapter_digest.to_string(),
        sampled: 0,
        agreed: 0,
        disagreed: Vec::new(),
        agreement: Agreement::Answers,
    };
    if !started.serves_candidate(request.adapter_digest) {
        return Ok(gate::serve(measured));
    }
    let served = served_model(&started, request.system)?;
    let Started {
        running,
        lines,
        mut kept,
        ..
    } = started;
    let answers = ask_then_judge(
        running,
        || {
            answer_all(ctx, &served, sample, cancel).map_err(|e| match e {
                OrchestratorError::Cancelled => Failure::Cancelled,
                other => Failure::Unmeasured(format!(
                    "asking the served candidate: {other}; brain serve said: {}",
                    recent_output(&lines, &mut kept)
                )),
            })
        },
        |answered| {
            judge_all(ctx, sample, answered).map_err(|e| match e {
                OrchestratorError::Cancelled => Failure::Cancelled,
                other => Failure::Unmeasured(format!("grading the served answers: {other}")),
            })
        },
    )?;
    measured.sampled = sample.tasks.len();
    let ids: Vec<String> = sample.tasks.iter().map(|t| t.task.id.to_string()).collect();
    if let Some(why) = unanswered(&ids, &answers, in_process) {
        return Err(Failure::Unmeasured(why));
    }
    let agree = agreements(ctx, &answers, in_process).map_err(Failure::Unmeasured)?;
    for (((task, served), local), agreed) in
        sample.tasks.iter().zip(&answers).zip(in_process).zip(agree)
    {
        if agreed {
            measured.agreed += 1;
        } else {
            measured.disagreed.push(Disagreement {
                task: task.task.id.to_string(),
                in_process: local.answer.clone(),
                served: served.answer.clone(),
                in_process_verdict: local.verdict,
                served_verdict: served.verdict,
            });
        }
    }
    Ok(gate::serve(measured))
}

/// Asks with the server running, stops it, then judges what it answered.
/// The judge is a model of its own that needs the device the server holds
/// until its process has exited: judged with the server alive, its load
/// runs out of device memory and every verdict is lost.
fn ask_then_judge<A, P>(
    server: Running,
    ask: impl FnOnce() -> Result<A, Failure>,
    judge: impl FnOnce(A) -> Result<P, Failure>,
) -> Result<P, Failure> {
    let answered = ask()?;
    drop(server);
    judge(answered)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    #[test]
    fn the_server_is_gone_before_its_answers_are_judged() {
        let child = Command::new("sleep").arg("60").spawn().unwrap();
        let pid = child.id();
        let alive = move || Path::new(&format!("/proc/{pid}")).exists();
        let judged = ask_then_judge(
            Running(child),
            || {
                assert!(alive(), "the server answers while it runs");
                Ok(7)
            },
            |answered| {
                assert!(!alive(), "the judge needs the device the server held");
                Ok(answered)
            },
        );
        assert!(matches!(judged, Ok(7)));
    }
}
