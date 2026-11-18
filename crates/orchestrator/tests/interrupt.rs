// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training clients that never leave a run
// holding a shared accelerator after their operator stopped them, for its
// clients. If your team needs expertise in cancellation across a client and
// a model service, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: Ctrl-C cancels the recorded runs in progress, through the token they
//! hand to every model job; with nothing in progress, or on a second signal,
//! the process exits; detached, it exits and leaves the jobs alone.
//!
//! Its own test binary: an interrupt is the process's, and a signal sent to
//! this process must reach no other test.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::sync::mpsc::channel;
use std::time::Duration;

use serde_json::json;
use splinter_orchestrator::interrupt::{handle_interrupts, interrupt, Interrupted, Interrupts};
use splinter_orchestrator::runs::record;
use splinter_orchestrator::{Config, Context, OrchestratorError};
use splinter_store::runs::{read_run, RunStatus};
use splinter_store::StateRoot;

#[test]
fn a_signal_cancels_the_run_in_progress_and_a_second_one_would_exit() {
    // Nothing in progress: a signal exits, detached or not.
    assert_eq!(interrupt(Interrupts::Cancel), Interrupted::Exiting);
    assert_eq!(interrupt(Interrupts::Detach), Interrupted::Exiting);

    let root = tempfile::tempdir().unwrap();
    let mut config = Config::from_env();
    config.state_root = StateRoot::new(root.path().join("state"));
    config.working_dir = root.path().to_path_buf();
    let ctx = Context::new(config, false).unwrap();
    let (told, heard) = channel();
    handle_interrupts(Interrupts::Cancel, move |what| told.send(what).unwrap()).unwrap();

    let (started, running) = channel::<()>();
    let operator = std::thread::spawn(move || {
        running.recv().unwrap();
        // From outside the process, as an operator's Ctrl-C is.
        let sent = std::process::Command::new("kill")
            .args(["-INT", &std::process::id().to_string()])
            .status()
            .unwrap();
        assert!(sent.success());
    });
    let ended = record(&ctx, "train", &json!({}), |run| {
        // A run is in progress: detached, a signal would leave it running.
        assert_eq!(interrupt(Interrupts::Detach), Interrupted::Detaching);
        let token = run.cancel_token();
        started.send(()).unwrap();
        // The job a model adapter follows stops when this token fires.
        for _ in 0..600 {
            if token.is_cancelled() {
                return Err(OrchestratorError::JobCancelled(
                    "brain job j at http://brain was cancelled".into(),
                ));
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        Ok(())
    });
    operator.join().unwrap();
    let Err(error) = ended else {
        panic!("the run was never interrupted");
    };
    assert!(error.to_string().contains("was cancelled"), "{error}");
    assert_eq!(heard.try_recv(), Ok(Interrupted::Cancelling));
    // A second signal now exits; detached, the first would have.
    assert_eq!(interrupt(Interrupts::Cancel), Interrupted::Exiting);
    let runs = splinter_orchestrator::runs::list(&ctx).unwrap().runs;
    assert_eq!(runs.len(), 1);
    let run = read_run(ctx.workspace(), &runs[0].id).unwrap();
    assert_eq!(run.status, RunStatus::Cancelled);
}
