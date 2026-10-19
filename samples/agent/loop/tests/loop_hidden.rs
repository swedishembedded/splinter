// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements self-improving coding agents whose every
// step is auditable, for its clients. If your team needs expertise in agent
// evaluation or locally operated coding agents, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: what the supervisor keeps from the worker. A hidden check is named
//! to the worker and nothing more: not in its prompt, not in its tree, not in
//! the feedback of a failed attempt. A worker that goes looking for the
//! loop's records, where every contract names its checks, is rejected even
//! when its fix was right.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::time::Duration;

use common::*;
use serde_json::json;
use splinter_agent_loop::outcome::Status;
use splinter_agent_loop::run::{execute, Request};

#[test]
fn a_hidden_check_is_named_to_the_retry_and_nothing_more() {
    let f = Fixture::new();
    let m = model(Duration::ZERO, |req| {
        let retry = format!("{:?}", req.messages).contains("previous attempt failed");
        Ok(match (retry, tool_results(req)) {
            (false, 0) => write("calc.py", "def add(a, b):\n    return 0\n"),
            (true, 0) => write("calc.py", FIXED),
            _ => text("done"),
        })
    });
    let splinter = f.splinter(m.clone());
    let outcome = execute(
        &splinter,
        &f.loop_home(),
        Request::New(Box::new(f.contract(3))),
    )
    .unwrap();

    assert_eq!(outcome.status, Status::Accepted, "{outcome:#?}");
    let requests = m.requests.lock().unwrap();
    let everything = requests
        .iter()
        .map(|r| format!("{:?}", r.messages))
        .collect::<String>();
    let retry = requests
        .iter()
        .map(|r| format!("{:?}", r.messages))
        .find(|p| p.contains("previous attempt failed"))
        .unwrap();
    assert!(retry.contains("acceptance_failed: unit"), "{retry}");
    for secret in ["sealed-check-marker", "AssertionError", "test_calc"] {
        assert!(
            !everything.contains(secret),
            "the worker was shown {secret:?} of a check it must not see"
        );
    }
}

#[test]
fn nothing_of_a_hidden_check_is_in_the_workers_tree_or_prompt() {
    let f = Fixture::new();
    // The worker looks at everything its tree holds, then fixes the bug.
    let m = model(Duration::ZERO, |req| {
        Ok(match tool_results(req) {
            0 => call(
                "c0",
                "shell",
                json!({"description": "look around", "shell_command": "ls -aR; grep -r accept . || true"}),
            ),
            1 => write("calc.py", FIXED),
            _ => text("done"),
        })
    });
    let splinter = f.splinter(m.clone());
    let outcome = execute(
        &splinter,
        &f.loop_home(),
        Request::New(Box::new(f.contract(1))),
    )
    .unwrap();

    assert_eq!(outcome.status, Status::Accepted, "{outcome:#?}");
    let seen = m
        .requests
        .lock()
        .unwrap()
        .iter()
        .map(|r| format!("{:?}", r.messages))
        .collect::<String>();
    assert!(seen.contains("calc.py"), "the worker saw its tree");
    for secret in ["sealed-check-marker", "contract.json"] {
        assert!(!seen.contains(secret), "{secret:?} reached the worker");
    }
}

#[test]
fn a_worker_that_goes_looking_for_the_loops_records_is_rejected_even_when_it_fixed_the_bug() {
    let f = Fixture::new();
    let records = f.loop_home().root().display().to_string();
    let snoop = model(Duration::ZERO, move |req| {
        Ok(match tool_results(req) {
            0 => write("calc.py", FIXED),
            1 => call(
                "c1",
                "shell",
                json!({"description": "look", "shell_command": format!("ls {records}")}),
            ),
            _ => text("done"),
        })
    });
    let splinter = f.splinter(snoop);
    let outcome = execute(
        &splinter,
        &f.loop_home(),
        Request::New(Box::new(f.contract(1))),
    )
    .unwrap();

    assert_eq!(outcome.status, Status::Rejected, "{outcome:#?}");
    assert_eq!(
        outcome.attempts[0].failure.as_deref(),
        Some("hidden_material_exposed")
    );
    assert!(
        outcome.attempts[0].checks.iter().all(|c| c.passed),
        "the fix was right; the attempt is rejected for how it was made"
    );
    let events = events_of(&outcome);
    assert!(
        events.iter().any(|e| e["type"] == "exposure"),
        "the audit is in the trace: {:?}",
        kinds(&events)
    );
}
