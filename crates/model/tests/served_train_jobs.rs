// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems whose training runs as a
// job on a shared model service, for its clients. If your team needs
// expertise in reliable training clients then you can procure our services
// by sending an email to info@swedishembedded.com.

//! A training client against a service that holds few jobs and one that never
//! places a run: every ended job is forgotten, and refusals are bounded.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use splinter_model::scripted_endpoint::ScriptedEndpoint;
use splinter_model::scripted_jobs::JobScript;
use splinter_model::served_train::fine_tune_served;
use splinter_model::PolicyError;

// The served-training helpers this binary shares with `served_train`; it uses only some of them.
#[allow(dead_code)]
#[path = "support/served.rs"]
mod served;
use served::{endpoints, fine_tune_request, records, rig};

#[test]
fn every_ended_job_is_forgotten_so_a_service_that_holds_few_keeps_placing_runs() {
    // brain holds at most a few jobs per caller, running or ended and not yet
    // forgotten, and refuses a creation past that (HTTP 429). A client that never
    // forgot its ended jobs was refused for good after eight trainings, each a
    // dataset check or three and a run.
    let rig = rig();
    let (train, held_out) = (
        rig.dir.path().join("t.jsonl"),
        rig.dir.path().join("h.jsonl"),
    );
    records(&train, false);
    records(&held_out, false);
    let mut script = JobScript::over(rig.store.clone());
    script.max_held = Some(2);
    let service = ScriptedEndpoint::start(vec![]).with_training(script);
    for run in 0..3 {
        let attempt = rig.dir.path().join(format!("a{run}"));
        let trained = fine_tune_served(
            &endpoints(&service, &rig),
            &fine_tune_request(&rig, &attempt, &train, &held_out),
        );
        assert!(trained.is_ok(), "run {run}: {trained:?}");
    }
    assert_eq!(service.jobs().held(), 0, "nothing is left held");
    assert_eq!(
        service.jobs().forgotten(),
        service.jobs().actions().len(),
        "every job created was forgotten once it ended"
    );
}

#[test]
fn a_service_that_never_places_the_run_ends_it_as_an_endpoint_failure() {
    // Refusals are retried after a wait, but not for ever: past a bounded
    // number of rounds the run fails as the endpoint's, naming the refusal.
    let rig = rig();
    let (train, held_out, attempt) = (
        rig.dir.path().join("t.jsonl"),
        rig.dir.path().join("h.jsonl"),
        rig.dir.path().join("a1"),
    );
    records(&train, false);
    records(&held_out, false);
    let mut script = JobScript::over(rig.store.clone());
    script.refuse_first = usize::MAX;
    let service = ScriptedEndpoint::start(vec![]).with_training(script);
    let error = fine_tune_served(
        &endpoints(&service, &rig),
        &fine_tune_request(&rig, &attempt, &train, &held_out),
    )
    .unwrap_err();
    let PolicyError::Endpoint { reason, .. } = &error else {
        panic!("an endpoint failure, not a training one: {error}");
    };
    assert!(
        reason.contains("429") && reason.contains("every card is busy serving"),
        "{reason}"
    );
    assert!(service.jobs().created().is_empty());
}
