// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements online reinforcement-learning runs whose
// every group can be followed while they run, for its clients. If your team
// needs expertise in observable asynchronous policy optimisation, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: `groups.jsonl` logs each group's transitions to sent, applied,
//! committed, filtered, stale and rolled back as they are made durable,
//! each once, through kills and resumes, with a line a crash cut short
//! dropped and the transitions it lost recovered from the ledger.

use std::collections::BTreeMap;
use std::io::Write;

use serde_json::Value;
use splinter_factory::online::driver::{run, RunLedger, Seams};
use splinter_factory::online::group_log::{GroupLog, LoggedLedger};
use splinter_factory::online::seams::{GroupLedger, RoundRobin};
use splinter_factory::online::Settings;
use splinter_store::factory::runstate::RunState;

use super::doubles::{Clock, Executor, ScriptedLearner};
use super::{learner_plan, script, settings, tasks};

fn lines(path: &std::path::Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

#[test]
fn every_transition_is_logged_once_through_kills_and_a_torn_line() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("groups.jsonl");
    let clock = Clock::default();
    let mut learner = ScriptedLearner::new(clock.clone(), learner_plan());
    let mut selector = RoundRobin::new(tasks(3), 8).unwrap();
    let settings = Settings {
        max_staleness: 100,
        ..settings()
    };
    for round in 0..3 {
        let state = RunState::open(dir.path()).unwrap();
        let now = clock.now() / 1000;
        let owner = format!("controller-{round}");
        let fence = if round == 0 {
            state.acquire(&owner, 30, now).unwrap()
        } else {
            state
                .take_over(&owner, &format!("controller-{}", round - 1), 30, now)
                .unwrap()
        };
        let mut ledger = RunLedger::new(&state, fence, 30, now);
        let entries = ledger.load().unwrap();
        let log = GroupLog::open(&log_path, &entries, clock.now()).unwrap();
        let mut logged = LoggedLedger::new(&mut ledger, log);
        let mut executor = Executor::new(clock.clone(), learner.served.clone(), script(1));
        match round {
            0 => executor.die_after = Some(7),
            1 => learner.die_after_accepts = Some(learner.accepts + 1),
            _ => learner.die_after_accepts = None,
        }
        let result = run(
            settings.clone(),
            Seams {
                executor: &mut executor,
                learner: &mut learner,
                selector: &mut selector,
                ledger: &mut logged,
                clock: &clock,
                cancel: &|| false,
            },
        );
        assert_eq!(result.is_ok(), round == 2, "round {round}");
        // A crash cut the last line short.
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&log_path)
            .unwrap();
        file.write_all(b"{\"group\":\"online:9\",\"sta").unwrap();
    }
    // Opening the log again drops the torn line and writes nothing new.
    let state = RunState::open(dir.path()).unwrap();
    let entries = state.group_entries().unwrap();
    drop(GroupLog::open(&log_path, &entries, clock.now()).unwrap());

    let logged = lines(&log_path);
    let mut seen: BTreeMap<(String, String), usize> = BTreeMap::new();
    for line in &logged {
        let pair = (
            line["group"].as_str().unwrap().to_string(),
            line["state"].as_str().unwrap().to_string(),
        );
        *seen.entry(pair).or_default() += 1;
        assert_eq!(line["epoch"], 0, "{line}");
        assert!(
            line["task"].as_str().unwrap().starts_with("task-"),
            "{line}"
        );
        assert!(
            line["at_ms"].is_u64() && line["learner_version"].is_u64(),
            "{line}"
        );
        assert_eq!(line["rollouts"].as_array().unwrap().len(), 2, "{line}");
    }
    assert!(
        seen.values().all(|n| *n == 1),
        "each transition once: {seen:?}"
    );
    let committed = seen.keys().filter(|(_, s)| s == "committed").count();
    assert_eq!(committed, 8, "every group's commit is logged: {seen:?}");
    for (group, state) in seen.keys() {
        assert!(
            ["sent", "applied", "committed"].contains(&state.as_str()),
            "{group} {state}"
        );
    }
    let last_commit = logged
        .iter()
        .filter(|l| l["state"] == "committed")
        .map(|l| l["learner_version"].as_u64().unwrap())
        .max()
        .unwrap();
    assert!(last_commit >= 1, "the learner's version is named");
}
