// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-data projection for agents that
// work in a source tree, for its clients. If your team needs expertise in
// turning verified agent trajectories into fine-tuning data, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: what a solver's workspace run teaches is what it should be taught.
//!
//! * A record opens with the system prompt the solve ran under, as its
//!   trajectory's solve profile records it, and the dataset records that
//!   prompt, so a model trained on it is asked under the same one. A run
//!   under a prompt this build cannot render is left out, runs under two
//!   different prompts do not make one dataset, and an explicit prompt
//!   renders every record again.
//! * The host's login name and the harness's scratch paths are not in a
//!   record: they say nothing about the task and a model would memorise them.
//! * An action that repeats a call whose result the agent already had is
//!   context and not an action to learn: the tail of a loop teaches the loop.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use common::*;
use serde_json::{json, Value};
use splinter_core::annotation::{Outcome, Strength};
use splinter_core::digest::Digest;
use splinter_core::experience::{Environment, Experience, Task};
use splinter_core::prompt::{CODING_AGENT_PROMPT, CODING_AGENT_PROMPT_ID, SYSTEM_PROMPT};
use splinter_data::{Corpus, Exclusion, Projection, Record, RecordBody, SftStep, View, ViewError};

const INSTRUCTION: &str = "Make the failing test pass.";

fn workspace_task() -> Task {
    Task::new(
        "workspace-change",
        Vec::new(),
        Environment::new(Environment::WORKSPACE, json!({"snapshot": "tree-1"})),
        INSTRUCTION,
        Vec::new(),
    )
    .unwrap()
}

/// A passed workspace run of `steps` whose agent record is `extra`.
fn run(steps: Vec<atif::TraceStep>, extra: Option<Value>, at: u32) -> Experience {
    let mut t = trajectory(steps);
    t.agent.extra = extra;
    experience(
        &workspace_task(),
        t,
        Some("PATCH"),
        &format!("2026-10-13T01:00:{at:02}.000Z"),
    )
}

fn short_run() -> Vec<atif::TraceStep> {
    vec![
        user(1, INSTRUCTION),
        call(2, "c1", "shell", json!({"cmd": "ls"}), "a.py"),
        say(3, "Done."),
    ]
}

fn under(prompt: &str, id: Option<&str>) -> Option<Value> {
    Some(json!({
        "solve_profile": { "system_prompt": { "id": id, "digest": Digest::of(prompt.as_bytes()) } }
    }))
}

fn project(experiences: Vec<Experience>) -> Result<Projection, ViewError> {
    let mut corpus = Corpus::new();
    for exp in experiences {
        let notes = vec![verdict(&exp, Outcome::Pass, Strength::Executable)];
        corpus.insert(exp, notes).unwrap();
    }
    SftStep::new(Strength::Executable).project(&corpus)
}

/// The projection as a dataset build leaves it: the prompt its records were
/// written under settled, as the manifest records it.
fn dataset_of(experiences: Vec<Experience>) -> Result<Projection, ViewError> {
    project(experiences)?.settled_system_prompt()
}

fn messages(record: &Record) -> &[splinter_core::chat::WireMessage] {
    match &record.body {
        RecordBody::Chat { messages, .. } => messages,
        other => panic!("a chat record, got {other:?}"),
    }
}

#[test]
fn a_run_is_rendered_under_the_prompt_its_solve_recorded_and_the_dataset_says_so() {
    let coding = under(CODING_AGENT_PROMPT, Some(CODING_AGENT_PROMPT_ID));
    let projection = dataset_of(vec![run(short_run(), coding, 1)]).unwrap();
    assert_eq!(projection.records.len(), 2);
    for record in &projection.records {
        assert_eq!(messages(record)[0].role, "system");
        assert_eq!(messages(record)[0].content, CODING_AGENT_PROMPT);
    }
    assert_eq!(
        projection.system_prompt.as_deref(),
        Some(CODING_AGENT_PROMPT),
        "the manifest names what the records were written under"
    );

    // A run under the chat prompt, or one that recorded none, stays default.
    for extra in [under(SYSTEM_PROMPT, None), None] {
        let projection = dataset_of(vec![run(short_run(), extra, 1)]).unwrap();
        assert_eq!(messages(&projection.records[0])[0].content, SYSTEM_PROMPT);
        assert_eq!(projection.system_prompt, None);
    }
}

#[test]
fn a_prompt_that_cannot_be_rendered_leaves_the_run_out_and_two_prompts_are_refused() {
    let custom = under("You are a pirate.", None);
    let projection = project(vec![run(short_run(), custom, 1)]).unwrap();
    assert!(projection.records.is_empty());
    assert_eq!(projection.count(Exclusion::Unrepresentable), 1);

    let coding = under(CODING_AGENT_PROMPT, Some(CODING_AGENT_PROMPT_ID));
    let mixed = dataset_of(vec![run(short_run(), coding, 1), run(short_run(), None, 2)]);
    let reason = mixed.unwrap_err().to_string();
    assert!(reason.contains("system prompt"), "{reason}");
}

#[test]
fn an_explicit_prompt_renders_every_record_again() {
    let coding = under(CODING_AGENT_PROMPT, Some(CODING_AGENT_PROMPT_ID));
    let projection = project(vec![run(short_run(), coding, 1), run(short_run(), None, 2)])
        .unwrap()
        .with_system_prompt(CODING_AGENT_PROMPT)
        .settled_system_prompt()
        .unwrap();
    assert!(projection
        .records
        .iter()
        .all(|r| messages(r)[0].content == CODING_AGENT_PROMPT));
    assert_eq!(
        projection.system_prompt.as_deref(),
        Some(CODING_AGENT_PROMPT)
    );
}

#[test]
fn the_host_login_and_the_harness_scratch_paths_are_not_in_a_record() {
    let listing = "total 24\ndrwxrwxr-x 5 marsch marsch   87 Oct 20 08:02 .\n\
                   -rw-r--r-- 1 root root 4365 Jan  1  1970 README.md\n";
    let steps = vec![
        user(1, INSTRUCTION),
        call(2, "c1", "shell", json!({"cmd": "ls -la"}), listing),
        call(
            3,
            "c2",
            "shell",
            json!({"cmd": "cd \x2ftmp/.tmpd9JLMJ/tree && python3 x.py"}),
            "Traceback (most recent call last):\n  File \"\x2ftmp/.tmpd9JLMJ/tree/ttlcache/x.py\", line 3\n\
             \x2ftmp/.tmpd9JLMJ/tree",
        ),
        call(
            4,
            "c3",
            "shell",
            json!({"cmd": "pwd"}),
            "/work/mine/tree/pkg\n/home/marsch/code",
        ),
        say(5, "Done."),
    ];
    let extra = Some(json!({ "workspace_tree": "/work/mine/tree" }));
    let projection = project(vec![run(steps, extra, 1)]).unwrap();
    let last = messages(projection.records.last().unwrap());
    let text: String = last
        .iter()
        .map(|m| format!("{}\n{}\n", m.content, json!(m.tool_calls)))
        .collect();
    for host in ["marsch", ".tmpd9JLMJ", "/work/mine"] {
        assert!(!text.contains(host), "{host} in {text}");
    }
    assert!(
        text.contains("drwxrwxr-x 5 user user   87 Oct 20 08:02 ."),
        "{text}"
    );
    assert!(
        text.contains("-rw-r--r-- 1 root root 4365"),
        "root is not a login"
    );
    assert!(text.contains("File \"ttlcache/x.py\", line 3"), "{text}");
    assert!(text.contains("cd . && python3 x.py"), "{text}");
}

#[test]
fn the_tail_of_a_loop_is_context_and_not_an_action_to_learn() {
    let rerun =
        |step: u64, id: &str| call(step, id, "shell", json!({"cmd": "python t.py"}), "boom");
    let steps = vec![
        user(1, INSTRUCTION),
        call(2, "c1", "edit", json!({"path": "a.py"}), "ok"),
        rerun(3, "c2"),
        call(4, "c3", "edit", json!({"path": "a.py"}), "ok again"),
        // The same call, now with another result, is progress.
        call(5, "c4", "shell", json!({"cmd": "python t.py"}), "fixed"),
        rerun(6, "c5"),
        rerun(7, "c6"),
        rerun(8, "c7"),
        say(9, "Done."),
    ];
    let projection = project(vec![run(steps, None, 1)]).unwrap();
    // c1 c2 c3 c4 and the answer; c5 repeats `python t.py -> boom` (c2) and
    // c6, c7 repeat c5.
    assert_eq!(projection.records.len(), 5);
    assert_eq!(projection.count(Exclusion::RepeatedAction), 3);
    let last = messages(projection.records.last().unwrap());
    let reruns = last
        .iter()
        .filter(|m| {
            m.tool_calls
                .iter()
                .any(|c| c.function.arguments.contains("t.py"))
        })
        .count();
    assert_eq!(reruns, 5, "the repeats stay in the context the answer saw");
    assert_eq!(last.iter().filter(|m| m.train).count(), 1);
}
