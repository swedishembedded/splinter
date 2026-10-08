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
//! acceptance script lives outside it.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{json, Value};
use splinter_agent_loop::acceptance::Check;
use splinter_agent_loop::contract::{Contract, Limits, CONTRACT_SCHEMA};
use splinter_agent_loop::outcome::{Outcome, Status};
use splinter_agent_loop::repo::{git, head_rev, is_clean};
use splinter_agent_loop::run::{execute, Request};
use splinter_agent_loop::store::LoopHome;
use splinter_agent_loop::trace::read_events;
use splinter_sdk::agent::solve::Model;
use splinter_sdk::config::Config;
use splinter_sdk::store::StateRoot;
use splinter_sdk::vocabulary::model_ref::ModelRef;
use splinter_sdk::{runs, Splinter};
use sven_sdk::model::{
    CompletionRequest, MessageContent, ModelProvider, ResponseEvent, ResponseStream,
};

// The loop's own dependency on sven is through the SDK; the tests reach the
// model trait the same way.
use splinter_sdk::agent::sven as sven_sdk;

const MODEL: &str = "local:/scripted";

/// What a scripted model answers a request with.
type Script = Box<dyn Fn(&CompletionRequest) -> anyhow::Result<Vec<ResponseEvent>> + Send + Sync>;

/// A model that answers from a script over the request it is sent.
struct Scripted {
    script: Script,
    pause: Duration,
    requests: Mutex<Vec<CompletionRequest>>,
}

#[async_trait::async_trait]
impl ModelProvider for Scripted {
    fn name(&self) -> &str {
        "scripted"
    }
    fn model_name(&self) -> &str {
        "fixture-1"
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        tokio::time::sleep(self.pause).await;
        let events = (self.script)(&req);
        self.requests.lock().unwrap().push(req);
        Ok(Box::pin(futures::stream::iter(
            events?.into_iter().map(Ok).collect::<Vec<_>>(),
        )))
    }
}

fn text(reply: &str) -> Vec<ResponseEvent> {
    vec![ResponseEvent::TextDelta(reply.into()), ResponseEvent::Done]
}

fn call(id: &str, name: &str, arguments: Value) -> Vec<ResponseEvent> {
    vec![
        ResponseEvent::ToolCall {
            index: 0,
            id: id.into(),
            name: name.into(),
            arguments: arguments.to_string(),
        },
        ResponseEvent::Done,
    ]
}

fn tool_results(req: &CompletionRequest) -> usize {
    req.messages
        .iter()
        .filter(|m| matches!(m.content, MessageContent::ToolResult { .. }))
        .count()
}

fn write(path: &str, content: &str) -> Vec<ResponseEvent> {
    call(
        "w",
        "write_file",
        json!({"path": path, "text": content, "append": false}),
    )
}

const FIXED: &str = "def add(a, b):\n    return a + b\n";

/// The disposable world of one spec.
struct Fixture {
    home: tempfile::TempDir,
    repo: PathBuf,
    _outside: tempfile::TempDir,
    check: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let home = tempfile::tempdir().unwrap();
        let repo = home.path().join("target-repo");
        std::fs::create_dir_all(repo.join("tests")).unwrap();
        // Like most repositories, this one ignores sven's directory and bytecode.
        std::fs::write(repo.join(".gitignore"), ".sven/\n__pycache__/\n").unwrap();
        std::fs::write(repo.join("calc.py"), "def add(a, b):\n    return a - b\n").unwrap();
        std::fs::write(
            repo.join("tests/test_calc.py"),
            "from calc import add\nassert add(2, 3) == 5\n",
        )
        .unwrap();
        git(&repo, &["init", "-q", "-b", "main"]).unwrap();
        git(&repo, &["config", "user.email", "t@example.com"]).unwrap();
        git(&repo, &["config", "user.name", "t"]).unwrap();
        git(&repo, &["add", "-A"]).unwrap();
        git(&repo, &["commit", "-q", "-m", "seeded bug"]).unwrap();
        let outside = tempfile::tempdir().unwrap();
        let check = outside.path().join("check.sh");
        std::fs::write(&check, "PYTHONPATH=. python3 tests/test_calc.py\n").unwrap();
        Self {
            home,
            repo,
            _outside: outside,
            check,
        }
    }

    fn contract(&self, max_attempts: u32) -> Contract {
        Contract {
            schema: CONTRACT_SCHEMA,
            task: "add() returns the difference; it must return the sum".into(),
            baseline_rev: head_rev(&self.repo).unwrap(),
            repository: self.repo.clone(),
            setup: Vec::new(),
            acceptance: vec![Check {
                name: "unit".into(),
                command: format!("sh {}", self.check.display()),
                timeout_secs: 30,
                visible: false,
            }],
            protected: vec!["tests/".into()],
            limits: Limits {
                attempt_secs: 60,
                total_secs: 120,
                max_output_tokens: 10_000,
                max_tool_calls: 10,
                max_repeated_results: 0,
                max_attempts,
                provider_retries: 0,
                follow_ups: 0,
                max_cost_usd: None,
            },
            model: MODEL.into(),
            allow_api_models: false,
            system_prompt_digest: splinter_sdk::vocabulary::digest::Digest::sha256_of(
                splinter_agent_loop::run::system_prompt(&self.loop_home())
                    .unwrap()
                    .0
                    .as_bytes(),
            )
            .to_string(),
            system_prompt_source: "built-in".into(),
            definitions_digest: String::new(),
            hints: Vec::new(),
        }
    }

    fn splinter(&self, model: Arc<Scripted>) -> Splinter {
        let state = StateRoot::under_home(self.home.path());
        let config = Config {
            state_root: state,
            model_store: self.home.path().join("models"),
            policy_base: self.home.path().join("models/policy"),
            policy_context_tokens: None,
            openrouter_api_key: None,
            brain_api_key: None,
            allow_remote: false,
            command_env: Default::default(),
            working_dir: self.home.path().to_path_buf(),
            brain_binary: None,
            front_door_model: None,
            assistant_model: None,
            judge_model: None,
            bf16_base: false,
            default_budget: None,
            remote_concurrency: 1,
            min_calibration_controls: 1,
            thinking: false,
            served_temperature: None,
        };
        let splinter = Splinter::builder(config).build().unwrap();
        splinter.runtime().add_model(
            MODEL.parse::<ModelRef>().unwrap(),
            Model::new(model, "scripted/fixture-1"),
        );
        splinter
    }

    fn loop_home(&self) -> LoopHome {
        LoopHome::under(self.home.path())
    }
}

fn model(
    pause: Duration,
    script: impl Fn(&CompletionRequest) -> anyhow::Result<Vec<ResponseEvent>> + Send + Sync + 'static,
) -> Arc<Scripted> {
    Arc::new(Scripted {
        script: Box::new(script),
        pause,
        requests: Mutex::new(Vec::new()),
    })
}

fn kinds(events: &[Value]) -> Vec<&str> {
    events.iter().map(|e| e["type"].as_str().unwrap()).collect()
}

fn events_of(outcome: &Outcome) -> Vec<Value> {
    read_events(outcome.artifacts.events.parent().unwrap()).unwrap()
}

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
        is_clean(&f.repo).unwrap(),
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
fn a_failed_attempt_is_reset_and_the_retry_is_shown_the_failure_but_not_a_patch() {
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

    assert_eq!(outcome.status, Status::Accepted);
    assert_eq!(outcome.attempts.len(), 2);
    assert_eq!(
        outcome.attempts[0].failure.as_deref(),
        Some("acceptance_failed: unit")
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
    // the loop's follow-up in the same conversation brings the fix.
    let m = model(Duration::ZERO, |req| {
        let told = format!("{:?}", req.messages).contains("Failure: acceptance_failed");
        Ok(match (told, tool_results(req)) {
            (false, _) => text("Let me fix the bug now."),
            (true, 0) => write("calc.py", FIXED),
            _ => text("fixed"),
        })
    });
    let splinter = f.splinter(m);
    let mut contract = f.contract(1);
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
        .contains("Failure: acceptance_failed"));
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
fn setup_commands_run_once_in_the_fresh_checkout_and_a_failing_one_stops_the_run() {
    let f = Fixture::new();
    let fixer = model(Duration::ZERO, |req| {
        Ok(if tool_results(req) == 0 {
            write("calc.py", FIXED)
        } else {
            text("ok")
        })
    });
    let splinter = f.splinter(fixer);
    let mut contract = f.contract(1);
    contract.setup = vec!["test -f calc.py".into()];
    let outcome = execute(&splinter, &f.loop_home(), Request::New(Box::new(contract))).unwrap();
    let events = events_of(&outcome);
    let setups: Vec<&Value> = events.iter().filter(|e| e["type"] == "setup").collect();
    assert_eq!(setups.len(), 1);
    assert_eq!(setups[0]["data"]["passed"], true);

    let mut broken = f.contract(1);
    broken.setup = vec!["exit 7".into()];
    let err = execute(&splinter, &f.loop_home(), Request::New(Box::new(broken))).unwrap_err();
    assert!(format!("{err:#}").contains("setup command"), "{err:#}");
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

#[test]
fn training_records_come_only_from_accepted_unaided_runs_and_only_the_accepted_attempt() {
    let f = Fixture::new();
    // Attempt 1 fails, attempt 2 is accepted; a second run is assisted; a
    // third is rejected.
    let flag = Arc::new(AtomicBool::new(false));
    let seen = flag.clone();
    let m = model(Duration::ZERO, move |req| {
        let retry = format!("{:?}", req.messages).contains("previous attempt failed");
        seen.fetch_or(retry, Ordering::SeqCst);
        Ok(match (retry, tool_results(req)) {
            (false, 0) => write("calc.py", "def add(a, b):\n    return 0\n"),
            (true, 0) => write("calc.py", FIXED),
            (_, _) => text("done"),
        })
    });
    let splinter = f.splinter(m);
    let home = f.loop_home();
    let good = execute(&splinter, &home, Request::New(Box::new(f.contract(3)))).unwrap();
    assert_eq!(good.status, Status::Accepted);
    let mut hinted = f.contract(3);
    hinted.hints.push("look at add()".into());
    let assisted = execute(&splinter, &home, Request::New(Box::new(hinted))).unwrap();
    let mut doomed = f.contract(1);
    doomed.limits.max_tool_calls = 0;
    let rejected = execute(&splinter, &home, Request::New(Box::new(doomed))).unwrap();
    assert_ne!(rejected.status, Status::Accepted);

    let out = f.home.path().join("data").join("sft.jsonl");
    let runs = vec![good.run.clone(), assisted.run.clone(), rejected.run.clone()];
    let manifest = splinter_agent_loop::dataset::export(&home, &runs, &out).unwrap();

    assert_eq!(manifest.records, 1, "{manifest:#?}");
    let why = |run: &str| {
        manifest
            .runs
            .iter()
            .find(|c| c.run == run)
            .unwrap()
            .excluded
            .clone()
            .unwrap_or_default()
    };
    assert!(
        why(&assisted.run).contains("assisted"),
        "{}",
        why(&assisted.run)
    );
    assert!(
        why(&rejected.run).contains("not accepted"),
        "{}",
        why(&rejected.run)
    );
    let record: Value = serde_json::from_str(
        std::fs::read_to_string(&out)
            .unwrap()
            .lines()
            .next()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        record["metadata"]["attempt"], 2,
        "the accepted attempt, not the failed one"
    );
    let messages = record["messages"].as_array().unwrap();
    let roles: Vec<&str> = messages
        .iter()
        .map(|m| m["role"].as_str().unwrap())
        .collect();
    assert_eq!(roles, ["system", "user", "assistant", "tool", "assistant"]);
    // Only the model's own turns are trained on; call ids are unique and answered.
    let flags: Vec<bool> = messages
        .iter()
        .map(|m| m["train"].as_bool().unwrap())
        .collect();
    assert_eq!(flags, [false, false, true, false, true]);
    for m in messages.iter().filter(|m| m["role"] == "assistant") {
        let content = m["content"].as_str().unwrap();
        assert_eq!(content, content.trim(), "{m}");
    }
    let call = &messages[2]["tool_calls"][0];
    assert_eq!(messages[3]["tool_call_id"], call["id"]);
    assert!(
        call["function"]["arguments"]
            .as_str()
            .unwrap()
            .contains("calc.py"),
        "arguments are a JSON string"
    );
    assert!(
        !messages[1]["content"]
            .as_str()
            .unwrap()
            .contains("return 0"),
        "nothing of the failed attempt"
    );
    assert!(
        !record["tools"].as_array().unwrap().is_empty(),
        "the tool definitions the model was offered"
    );
}

#[test]
fn the_patch_form_holds_the_task_and_the_accepted_patch_as_one_exchange() {
    let f = Fixture::new();
    let m = model(Duration::ZERO, move |req| {
        Ok(match tool_results(req) {
            0 => write("calc.py", FIXED),
            _ => text("done"),
        })
    });
    let splinter = f.splinter(m);
    let home = f.loop_home();
    let good = execute(&splinter, &home, Request::New(Box::new(f.contract(2)))).unwrap();
    assert_eq!(good.status, Status::Accepted);

    let out = f.home.path().join("data").join("patch.jsonl");
    let manifest = splinter_agent_loop::dataset::export_in(
        &home,
        std::slice::from_ref(&good.run),
        &out,
        splinter_agent_loop::dataset::Form::Patch,
    )
    .unwrap();
    assert_eq!(manifest.records, 1, "{manifest:#?}");
    let text = std::fs::read_to_string(&out).unwrap();
    let record: Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
    let messages = record["messages"].as_array().unwrap();
    let roles: Vec<&str> = messages
        .iter()
        .map(|m| m["role"].as_str().unwrap())
        .collect();
    assert_eq!(roles, ["system", "user", "assistant"]);
    assert!(messages[1]["content"].as_str().unwrap().contains("Task:"));
    assert!(messages[2]["content"]
        .as_str()
        .unwrap()
        .contains("diff --git"));
    assert_eq!(messages[2]["train"], true);
    assert_eq!(record["metadata"]["form"], "patch");
}
