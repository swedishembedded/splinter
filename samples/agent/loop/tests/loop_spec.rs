// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements self-improving coding agents whose every
// step is auditable, for its clients. If your team needs expertise in agent
// evaluation or locally operated coding agents, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: the loop turns a task into an audited, reproducible outcome.
//!
//! A scripted model stands in for a real one: these specs establish the
//! plumbing (isolation, limits, judging, feedback, checkpoints, cancel,
//! resume, the trace), not coding ability, which is measured with a real
//! model. The repository is a disposable fixture with a seeded bug; the
//! acceptance script lives outside it. The worker is the shared one (a
//! fresh copy of the repository's snapshot with sven's tools) and the
//! judgement is the shared workspace grading.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use common::*;
use serde_json::{json, Value};
use splinter_agent_loop::outcome::Status;
use splinter_agent_loop::run::{apply, execute, Request};
use splinter_agent_loop::trace::read_events;
use splinter_sdk::runs;

#[test]
fn a_model_that_fixes_the_seeded_bug_is_accepted_and_the_target_repository_is_untouched() {
    let f = Fixture::new();
    let fixer = model(Duration::ZERO, |req| {
        Ok(if tool_results(req) == 0 {
            write("calc.py", FIXED)
        } else {
            text("changed - to +")
        })
    });
    let splinter = f.splinter(fixer);
    let outcome = execute(
        &splinter,
        &f.loop_home(),
        Request::New(Box::new(f.contract(2))),
    )
    .unwrap();

    assert_eq!(outcome.status, Status::Accepted, "{outcome:#?}");
    assert_eq!(outcome.assistance, "unaided");
    assert_eq!(outcome.changed_files.len(), 1);
    assert_eq!(outcome.changed_files[0].path, "calc.py");
    assert!(outcome.changed_files[0].before.is_some() && outcome.changed_files[0].after.is_some());
    assert!(outcome.validation.iter().all(|c| c.passed));
    assert!(
        is_clean(&f.repo),
        "the repository named by the task is not edited"
    );
    assert_eq!(
        std::fs::read_to_string(f.repo.join("calc.py")).unwrap(),
        "def add(a, b):\n    return a - b\n"
    );
    assert!(outcome.artifacts.patch.as_ref().unwrap().exists());
    assert!(
        !outcome.artifacts.trajectories.is_empty(),
        "sven's own trajectory is kept"
    );

    let events = events_of(&outcome);
    let k = kinds(&events);
    for needed in [
        "run_started",
        "model_selected",
        "baseline_validation",
        "attempt_started",
        "tool_request",
        "tool_result",
        "file_mutation",
        "candidate",
        "validation",
        "attempt_finished",
        "run_finished",
    ] {
        assert!(k.contains(&needed), "{needed} missing from {k:?}");
    }
    let baseline = events
        .iter()
        .find(|e| e["type"] == "baseline_validation")
        .unwrap();
    assert_eq!(
        baseline["data"]["passed"], false,
        "the seeded failure is red before the work"
    );
    let ids: Vec<u64> = events.iter().map(|e| e["id"].as_u64().unwrap()).collect();
    assert!(
        ids.windows(2).all(|w| w[0] < w[1]),
        "event ids only grow: {ids:?}"
    );
    let result = events.iter().find(|e| e["type"] == "tool_result").unwrap();
    let request = events.iter().find(|e| e["type"] == "tool_request").unwrap();
    assert_eq!(
        result["parent"], request["id"],
        "a result is a child of its request"
    );
}

#[test]
fn a_failed_attempt_is_restarted_from_the_snapshot_and_the_retry_is_shown_the_failure_but_not_a_patch(
) {
    let f = Fixture::new();
    // Attempt 1 breaks things differently and gives up; attempt 2 fixes it.
    let seen_feedback = Arc::new(AtomicBool::new(false));
    let flag = seen_feedback.clone();
    let m = model(Duration::ZERO, move |req| {
        let prompt = format!("{:?}", req.messages);
        let retry = prompt.contains("previous attempt failed");
        flag.fetch_or(retry, Ordering::SeqCst);
        Ok(match (retry, tool_results(req)) {
            (false, 0) => write("calc.py", "def add(a, b):\n    return 0\n"),
            (true, 0) => call(
                "r",
                "shell",
                json!({"description": "read", "shell_command": "cat calc.py"}),
            ),
            (true, 1) => write("calc.py", FIXED),
            _ => text("done"),
        })
    });
    let splinter = f.splinter(m.clone());
    let outcome = execute(
        &splinter,
        &f.loop_home(),
        Request::New(Box::new(f.shown(3))),
    )
    .unwrap();

    assert_eq!(outcome.status, Status::Accepted);
    assert_eq!(outcome.attempts.len(), 2, "{outcome:#?}");
    assert_eq!(
        outcome.attempts[0].failure.as_deref(),
        Some("acceptance_failed: unit")
    );
    let read_back = events_of(&outcome)
        .iter()
        .filter(|e| e["type"] == "tool_result" && e["attempt"] == 2)
        .map(|e| e["data"]["output"].to_string())
        .collect::<String>();
    assert!(
        read_back.contains("a - b") && !read_back.contains("return 0"),
        "attempt 2 starts from the snapshot, not from attempt 1's edit: {read_back}"
    );
    assert!(
        seen_feedback.load(Ordering::SeqCst),
        "attempt 2 was told what failed"
    );
    let requests = m.requests.lock().unwrap();
    let retry_prompt = requests
        .iter()
        .map(|r| format!("{:?}", r.messages))
        .find(|p| p.contains("previous attempt failed"))
        .unwrap();
    assert!(
        retry_prompt.contains("AssertionError") || retry_prompt.contains("Traceback"),
        "the failing output is shown"
    );
    assert!(
        !retry_prompt.contains("return 0"),
        "the failed patch is not handed back"
    );
    let events = events_of(&outcome);
    let started: Vec<&Value> = events
        .iter()
        .filter(|e| e["type"] == "attempt_started")
        .collect();
    assert_eq!(started[1]["attempt"], 2);
    assert_eq!(started[1]["data"]["has_feedback"], true);
}

#[test]
fn a_worker_that_stops_with_the_checks_red_is_told_what_failed_in_the_same_attempt() {
    let f = Fixture::new();
    // The first send announces a step and stops, as small models do; only
    // the follow-up in the same conversation, which the shown check's output
    // earns, brings the fix.
    let m = model(Duration::ZERO, |req| {
        let told = format!("{:?}", req.messages).contains("still fail");
        Ok(match (told, tool_results(req)) {
            (false, _) => text("Let me fix the bug now."),
            (true, 0) => write("calc.py", FIXED),
            _ => text("fixed"),
        })
    });
    let splinter = f.splinter(m);
    let mut contract = f.shown(1);
    contract.limits.follow_ups = 2;
    let outcome = execute(&splinter, &f.loop_home(), Request::New(Box::new(contract))).unwrap();

    assert_eq!(outcome.status, Status::Accepted, "{outcome:#?}");
    assert_eq!(
        outcome.attempts.len(),
        1,
        "one attempt, with a follow-up inside it"
    );
    let events = events_of(&outcome);
    let follow = events.iter().find(|e| e["type"] == "follow_up").unwrap();
    assert_eq!(follow["data"]["round"], 1);
    assert!(follow["data"]["message"]
        .as_str()
        .unwrap()
        .contains("still fail"));
    let started = events
        .iter()
        .find(|e| e["type"] == "attempt_started")
        .unwrap();
    assert_eq!(started["data"]["follow_ups"], 2);
}

#[test]
fn touching_a_protected_file_rejects_the_attempt_even_when_the_checks_pass() {
    let f = Fixture::new();
    let cheat = model(Duration::ZERO, |req| {
        Ok(match tool_results(req) {
            0 => write("calc.py", FIXED),
            1 => write("tests/test_calc.py", "assert True\n"),
            _ => text("done"),
        })
    });
    let splinter = f.splinter(cheat);
    let outcome = execute(
        &splinter,
        &f.loop_home(),
        Request::New(Box::new(f.contract(1))),
    )
    .unwrap();

    assert_eq!(outcome.status, Status::Rejected);
    let failure = outcome.attempts[0].failure.clone().unwrap();
    assert!(failure.starts_with("protected_path_modified"), "{failure}");
    assert!(outcome
        .unresolved
        .iter()
        .any(|u| u.contains("protected_path_modified")));
}

#[test]
fn a_run_that_asks_for_more_tool_calls_than_its_cap_is_stopped_at_the_cap() {
    let f = Fixture::new();
    let looping = model(Duration::ZERO, |_| {
        Ok(call("c", "find_file", json!({"pattern": "*.nothing"})))
    });
    let splinter = f.splinter(looping);
    let mut contract = f.contract(1);
    contract.limits.max_tool_calls = 3;
    let outcome = execute(&splinter, &f.loop_home(), Request::New(Box::new(contract))).unwrap();

    assert_eq!(outcome.status, Status::Rejected);
    assert_eq!(
        outcome.attempts[0].failure.as_deref(),
        Some("tool_call_limit")
    );
    assert!(outcome.attempts[0].stopped_by_tool_cap);
}

#[test]
fn a_worker_that_keeps_getting_the_same_answer_to_the_same_call_is_stopped_as_going_nowhere() {
    let f = Fixture::new();
    let looping = model(Duration::ZERO, |_| {
        Ok(call("c", "find_file", json!({"pattern": "*.nothing"})))
    });
    let splinter = f.splinter(looping);
    let mut contract = f.contract(1);
    contract.limits.max_tool_calls = 40;
    contract.limits.max_repeated_results = 3;
    let outcome = execute(&splinter, &f.loop_home(), Request::New(Box::new(contract))).unwrap();

    assert_eq!(outcome.status, Status::Rejected);
    let attempt = &outcome.attempts[0];
    assert!(
        attempt
            .failure
            .as_deref()
            .is_some_and(|f| f.starts_with("stagnation")),
        "{:?}",
        attempt.failure
    );
    assert!(attempt.stopped_by_repetition && !attempt.stopped_by_tool_cap);
    assert!(
        attempt.tool_calls < 10,
        "stopped long before the cap: {}",
        attempt.tool_calls
    );
}

#[test]
fn a_run_that_uses_up_its_total_time_stops_with_a_recorded_limit_and_starts_no_new_attempt() {
    let f = Fixture::new();
    // Every model call takes two seconds against a one-second budget.
    let slow = model(Duration::from_secs(2), |_| Ok(text("thinking")));
    let splinter = f.splinter(slow);
    let mut contract = f.contract(3);
    contract.limits.attempt_secs = 1;
    contract.limits.total_secs = 1;
    let outcome = execute(&splinter, &f.loop_home(), Request::New(Box::new(contract))).unwrap();

    assert_eq!(outcome.status, Status::LimitExhausted, "{outcome:#?}");
    assert_eq!(
        outcome.attempts.len(),
        1,
        "no attempt starts after the budget is gone"
    );
    assert!(
        outcome.termination.contains("time limit"),
        "{}",
        outcome.termination
    );
    let k = kinds(&events_of(&outcome))
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    assert!(k.contains(&"limit_reached".to_string()), "{k:?}");
}

#[test]
fn a_provider_that_keeps_failing_ends_the_run_with_a_recorded_error_after_bounded_retries() {
    let f = Fixture::new();
    let down = model(Duration::ZERO, |_| Err(anyhow::anyhow!("connection reset")));
    let splinter = f.splinter(down);
    let mut contract = f.contract(3);
    contract.limits.provider_retries = 1;
    let outcome = execute(&splinter, &f.loop_home(), Request::New(Box::new(contract))).unwrap();

    assert_eq!(outcome.status, Status::Error, "{outcome:#?}");
    assert_eq!(outcome.attempts.len(), 1, "no retry storm across attempts");
    let k = kinds(&events_of(&outcome))
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    assert!(
        k.contains(&"provider_retry".to_string()) && k.contains(&"provider_error".to_string()),
        "{k:?}"
    );
}

#[test]
fn a_cancel_from_another_process_stops_the_run_with_a_checkpoint_and_a_resume_finishes_it() {
    let f = Fixture::new();
    let healed = Arc::new(AtomicBool::new(false));
    let flag = healed.clone();
    // Slow while "sick": each model call takes long enough to be cancelled in.
    let m = model(Duration::from_millis(400), move |req| {
        if flag.load(Ordering::SeqCst) {
            Ok(if tool_results(req) == 0 {
                write("calc.py", FIXED)
            } else {
                text("done")
            })
        } else {
            Ok(call("c", "find_file", json!({"pattern": "*.nothing"})))
        }
    });
    let splinter = Arc::new(f.splinter(m));
    let home = f.loop_home();
    let contract = f.contract(3);

    let canceller = {
        let splinter = splinter.clone();
        std::thread::spawn(move || {
            let ctx = splinter.context();
            for _ in 0..200 {
                if let Ok(list) = runs::list(&ctx) {
                    if let Some(run) = list
                        .runs
                        .iter()
                        .find(|r| format!("{:?}", r.status) == "Running")
                    {
                        std::thread::sleep(Duration::from_millis(600));
                        runs::cancel(&ctx, &run.id).unwrap();
                        return;
                    }
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            panic!("no running run to cancel");
        })
    };
    let cancelled = execute(&splinter, &home, Request::New(Box::new(contract))).unwrap();
    canceller.join().unwrap();
    assert_eq!(cancelled.status, Status::Cancelled, "{cancelled:#?}");
    let dir = cancelled.artifacts.run_dir.clone();
    assert!(dir.join("checkpoint.json").exists(), "a checkpoint is left");

    healed.store(true, Ordering::SeqCst);
    let resumed = execute(&splinter, &home, Request::Resume(cancelled.run.clone())).unwrap();
    assert_eq!(resumed.status, Status::Accepted, "{resumed:#?}");
    assert_eq!(
        resumed.run, cancelled.run,
        "the resumed run is the same run"
    );
    let events = read_events(&dir).unwrap();
    let k = kinds(&events);
    assert!(
        k.contains(&"run_resumed") && k.contains(&"reconcile"),
        "{k:?}"
    );
    let ids: Vec<u64> = events.iter().map(|e| e["id"].as_u64().unwrap()).collect();
    assert!(
        ids.windows(2).all(|w| w[0] < w[1]),
        "ids still only grow across the restart"
    );
}

#[test]
fn the_patch_of_an_accepted_run_applies_to_the_repository_at_its_baseline_and_only_there() {
    let f = Fixture::new();
    let fixer = model(Duration::ZERO, |req| {
        Ok(if tool_results(req) == 0 {
            write("calc.py", FIXED)
        } else {
            text("ok")
        })
    });
    let splinter = f.splinter(fixer);
    let home = f.loop_home();
    let outcome = execute(&splinter, &home, Request::New(Box::new(f.contract(1)))).unwrap();
    assert_eq!(outcome.status, Status::Accepted, "{outcome:#?}");

    apply(&splinter, &home, &outcome.run).unwrap();
    assert_eq!(
        std::fs::read_to_string(f.repo.join("calc.py")).unwrap(),
        FIXED
    );

    // The repository now differs from the baseline: nothing more is applied.
    let again = apply(&splinter, &home, &outcome.run).unwrap_err();
    assert!(format!("{again:#}").contains("uncommitted"), "{again:#}");
}

#[test]
fn the_patch_that_is_applied_is_the_candidate_exactly_even_where_its_text_looks_like_a_secret() {
    let f = Fixture::new();
    // The trace redacts what looks like a credential; the patch is code to
    // be applied and must reach the repository as it was judged.
    let fixed = "def add(a, b):\n    token = 5\n    return a + b\n";
    let fixer = model(Duration::ZERO, move |req| {
        Ok(if tool_results(req) == 0 {
            write("calc.py", fixed)
        } else {
            text("ok")
        })
    });
    let splinter = f.splinter(fixer);
    let home = f.loop_home();
    let outcome = execute(&splinter, &home, Request::New(Box::new(f.contract(1)))).unwrap();
    assert_eq!(outcome.status, Status::Accepted, "{outcome:#?}");

    apply(&splinter, &home, &outcome.run).unwrap();
    assert_eq!(
        std::fs::read_to_string(f.repo.join("calc.py")).unwrap(),
        fixed
    );
}

#[test]
fn a_hint_makes_the_run_assisted_and_remote_models_are_refused_without_the_opt_in() {
    let f = Fixture::new();
    let fixer = model(Duration::ZERO, |req| {
        Ok(if tool_results(req) == 0 {
            write("calc.py", FIXED)
        } else {
            text("ok")
        })
    });
    let splinter = f.splinter(fixer);
    let mut hinted = f.contract(1);
    hinted.hints.push("look at add()".into());
    let outcome = execute(&splinter, &f.loop_home(), Request::New(Box::new(hinted))).unwrap();
    assert_eq!(outcome.assistance, "assisted");

    let mut remote = f.contract(1);
    remote.model = "remote:openrouter/some/model".into();
    let err = execute(&splinter, &f.loop_home(), Request::New(Box::new(remote))).unwrap_err();
    assert!(
        format!("{err:#}").to_lowercase().contains("remote"),
        "{err:#}"
    );
}
