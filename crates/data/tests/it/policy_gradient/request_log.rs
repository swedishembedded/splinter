// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training data that holds an agent's model to
// exactly the requests it was sent, for its clients. If your team needs
// expertise in turning agent request logs into faithful training records, you
// can procure our services by sending an email to info@swedishembedded.com.

//! Spec: a single agent's request log that does not read as one conversation
//! is still what its model was sent, and is trained as sent; a context the
//! model was never sent is never supervised.
//!
//! The shapes are the ones an agent runtime writes (seen in real request
//! logs): an answer the agent stopped and dropped from its history, with a
//! note in its place; arguments the model wrote as text that is not JSON,
//! kept as written in the history the next request carries; a last request the run
//! ended before it was answered, recorded as a step that said nothing; and a
//! context rewritten between two requests (compaction, an edited note). Each
//! action's record is the request it answered with the answer supervised: one
//! conversation where the requests still extend one another, else each
//! request on its own. An answer that is no agent step (dropped, or the notes
//! of a compaction) is no action; an answer a record cannot carry is left out
//! and counted. A rollout none of whose answers can be paired with the agent
//! step it made keeps its group's baseline as one `baseline_only` record and
//! is counted `requests_unpaired`; its trajectory is never rendered instead.

use super::*;

use splinter_core::chat::WireMessage;
use splinter_core::harness::{ModelCall, ModelResponse, ModelToolCall, RecordedCalls, CALLS_KEY};
use splinter_data::{native_record, SftStep, Unprojectable};

const SYSTEM: &str = "You are the agent under test.";
const ASK: &str = "Fix the file.";
const NOTE: &str = "Your previous tool call was stopped. Make a shorter call.";

fn shell() -> Value {
    json!({"name": "shell", "description": "Run a command.", "parameters": {"type": "object"}})
}

fn sent(role: &str, content: &str) -> Value {
    json!({"role": role, "content": content})
}

/// The model's tool-call turn as the next request carries it, and its result.
fn called(id: &str, arguments: &str, result: &str) -> Vec<Value> {
    vec![
        sent("assistant", ""),
        json!({"role": "assistant", "content": {"tool_call_id": id, "function": {
            "name": "shell", "arguments": arguments}}}),
        json!({"role": "tool", "content": {"tool_call_id": id, "content": result}}),
    ]
}

fn calls(id: &str, arguments: &str) -> ModelResponse {
    ModelResponse {
        text: String::new(),
        reasoning: None,
        tool_calls: vec![ModelToolCall {
            id: id.into(),
            name: "shell".into(),
            arguments: arguments.into(),
        }],
    }
}

fn says(text: &str) -> ModelResponse {
    ModelResponse {
        text: text.into(),
        reasoning: None,
        tool_calls: Vec::new(),
    }
}

fn request(seq: u64, messages: Vec<Value>, response: ModelResponse) -> ModelCall {
    ModelCall {
        seq,
        messages,
        tools: vec![shell()],
        response,
    }
}

/// The agent's steps: the instruction, one call of `shell` as `c0`, the
/// final answer.
fn steps(arguments: Value) -> Vec<TraceStep> {
    vec![
        user(1, ASK),
        call(2, "c0", "shell", arguments, "a.txt"),
        say(3, "Done."),
    ]
}

/// A pass and a fail of one task (a group with a spread of rewards), each a
/// single agent's trajectory `steps` with its request log `log` recorded as
/// the factory files it.
fn attempts(steps: &[TraceStep], log: &[ModelCall]) -> Corpus {
    let mut corpus = Corpus::new();
    for trial in [Trial::pass("t1"), Trial::fail("t2", &[])] {
        let mut trajectory = trajectory(steps.to_vec());
        trajectory.extra = Some(json!({ CALLS_KEY: RecordedCalls::record("log", log).to_value() }));
        let at = format!("2026-11-26T00:00:00.{:03}Z", corpus.entries().len());
        let experience = experience(&bench_task("1"), trajectory, Some("Done."), &at);
        let note = graded(&experience, &trial, None);
        corpus.insert(experience, vec![note]).unwrap();
    }
    corpus
}

fn messages(record: &splinter_data::Record) -> &[WireMessage] {
    match &record.body {
        RecordBody::Chat { messages, .. } => messages,
        other => panic!("a chat record, got {other:?}"),
    }
}

/// Each record's context (its messages before the supervised turn, as
/// role and text) and its supervised turn's tool calls' arguments, or its
/// text when it calls nothing.
fn read(projection: &Projection) -> Vec<(Vec<(String, String)>, String)> {
    projection
        .records
        .iter()
        .map(|record| {
            let messages = messages(record);
            let at = messages.iter().position(|m| m.train).unwrap();
            assert_eq!(at, messages.len() - 1, "only the last turn is supervised");
            let context = messages[..at]
                .iter()
                .map(|m| (m.role.clone(), m.content.clone()))
                .collect();
            let answer = &messages[at];
            let supervised = if answer.tool_calls.is_empty() {
                answer.content.clone()
            } else {
                answer
                    .tool_calls
                    .iter()
                    .map(|c| c.function.arguments.clone())
                    .collect::<Vec<_>>()
                    .join(";")
            };
            (context, supervised)
        })
        .collect()
}

fn pair(role: &str, text: &str) -> (String, String) {
    (role.to_owned(), text.to_owned())
}

#[test]
fn an_answer_the_agent_stopped_and_dropped_is_no_action_and_the_rest_is_read_as_sent() {
    // The first answer's arguments ran on until the agent stopped them; it
    // ran nothing, kept no step and is not in any later request, which
    // carries a note instead.
    let opening = vec![sent("system", SYSTEM), sent("user", ASK)];
    let mut noted = opening.clone();
    noted.push(sent("user", NOTE));
    let mut later = noted.clone();
    later.extend(called("c0", "{\"command\": \"ls\"}", "a.txt"));
    let log = vec![
        request(1, opening, calls("x0", "{\"command\": \"ls ls ls ls")),
        request(2, noted, calls("c0", "{\"command\": \"ls\"}")),
        request(3, later, says("Done.")),
    ];
    let steps = steps(json!({"command": "ls"}));
    let projection = view().project(&attempts(&steps, &log)).unwrap();

    assert!(projection.excluded.is_empty(), "{:?}", projection.excluded);
    let got = read(&projection);
    assert_eq!(got.len(), 4, "two actions of each attempt");
    let asked = [
        pair("system", SYSTEM),
        pair("user", ASK),
        pair("user", NOTE),
    ];
    for (context, _) in &got {
        assert_eq!(context[..3], asked[..], "the context as it was sent");
    }
    assert_eq!(got[0].1, "{\"command\": \"ls\"}");
    assert_eq!(got[1].1, "Done.");
    assert!(outcomes(&projection).iter().all(|o| !o.baseline_only));

    // The conversation is still one: packed, each attempt is one record.
    let packed = view()
        .packed(true)
        .project(&attempts(&steps, &log))
        .unwrap();
    assert_eq!(packed.records.len(), 2);
    assert!(packed.excluded.is_empty(), "{:?}", packed.excluded);

    // The native record of the conversation supervises the same two actions.
    let native = native_record(&trajectory(steps), &log).unwrap();
    assert_eq!(native.actions, 2);
}

#[test]
fn arguments_that_are_not_json_are_left_out_and_counted_and_the_rest_keeps_the_context_sent() {
    // The model escaped a backslash JSON does not know; the next request
    // carries the call as the model wrote it.
    let opening = vec![sent("system", SYSTEM), sent("user", ASK)];
    let written = "{\"command\": \"grep \\s a.txt\"}";
    let mut later = opening.clone();
    later.extend(called("c0", written, "a.txt"));
    let log = vec![
        request(1, opening, calls("c0", written)),
        request(2, later, says("Done.")),
    ];
    let steps = steps(json!({"command": "grep \\s a.txt"}));
    let projection = view().project(&attempts(&steps, &log)).unwrap();

    assert_eq!(
        projection.excluded.get(&Exclusion::UnparsedArguments),
        Some(&2)
    );
    let got = read(&projection);
    assert_eq!(got.len(), 2, "the final answer of each attempt");
    for (context, answer) in &got {
        assert_eq!(answer, "Done.");
        assert_eq!(context[..2], [pair("system", SYSTEM), pair("user", ASK)]);
    }
    let history = projection.records[0].clone().body;
    let RecordBody::Chat { messages, .. } = history else {
        panic!("chat")
    };
    let sent_call = messages
        .iter()
        .find(|m| !m.tool_calls.is_empty())
        .expect("the call is context");
    assert!(!sent_call.train);
    assert_eq!(sent_call.tool_calls[0].function.arguments, written);
    assert!(outcomes(&projection).iter().all(|o| !o.baseline_only));

    let native = native_record(&trajectory(steps), &log).unwrap();
    assert_eq!(
        native.actions, 1,
        "the call is context, the answer supervised"
    );
}

#[test]
fn a_last_request_the_run_ended_before_it_was_answered_leaves_the_others_as_sent() {
    // The runtime filed the unanswered request as a step of its own: no text,
    // no call, the reasoning it had streamed.
    let opening = vec![sent("system", SYSTEM), sent("user", ASK)];
    let mut later = opening.clone();
    later.extend(called("c0", "{\"command\":\"ls\"}", "a.txt"));
    let log = vec![
        request(1, opening, calls("c0", "{\"command\": \"ls\"}")),
        request(2, later, says("Done.")),
    ];
    let mut steps = steps(json!({"command": "ls"}));
    let mut timed_out = say(4, "");
    timed_out.reasoning_content = Some("checking".into());
    steps.push(timed_out);
    let projection = view().project(&attempts(&steps, &log)).unwrap();

    assert!(projection.excluded.is_empty(), "{:?}", projection.excluded);
    let got = read(&projection);
    assert_eq!(got.len(), 4);
    for (context, _) in &got {
        assert_eq!(context[0], pair("system", SYSTEM), "the prompt as sent");
    }
}

/// A run whose context was compacted: after the call, the model is asked for
/// notes (an answer that is no agent step), and the next request holds the
/// instruction with its time line updated and the notes in place of the
/// history.
fn compacted() -> (Vec<TraceStep>, Vec<ModelCall>) {
    let first = vec![
        sent("system", SYSTEM),
        sent("user", &format!("{ASK}\n\nTime: 0 of 8 min")),
    ];
    let mut asked_notes = first.clone();
    asked_notes.extend(called("c0", "{\"command\":\"ls\"}", "a.txt"));
    asked_notes.push(sent("user", "Write notes to continue from."));
    let after = vec![
        sent("system", SYSTEM),
        sent("user", &format!("{ASK}\n\nTime: 1 of 8 min")),
        sent("user", "Notes: listed a.txt."),
    ];
    let log = vec![
        request(1, first, calls("c0", "{\"command\": \"ls\"}")),
        request(2, asked_notes, says("Notes: listed a.txt.")),
        request(3, after, says("Done.")),
    ];
    let steps = vec![
        user(1, ASK),
        call(2, "c0", "shell", json!({"command": "ls"}), "a.txt"),
        TraceStep::new(
            3,
            StepOrigin::System,
            "context_compaction: 900 -> 100 tokens",
        ),
        say(4, "Done."),
    ];
    (steps, log)
}

#[test]
fn a_context_rewritten_between_requests_is_read_request_by_request_as_sent() {
    let (steps, log) = compacted();
    let projection = view().project(&attempts(&steps, &log)).unwrap();

    assert!(projection.excluded.is_empty(), "{:?}", projection.excluded);
    let got = read(&projection);
    assert_eq!(got.len(), 4, "the call and the answer of each attempt");
    let first = vec![
        pair("system", SYSTEM),
        pair("user", &format!("{ASK}\n\nTime: 0 of 8 min")),
    ];
    let after = vec![
        pair("system", SYSTEM),
        pair("user", &format!("{ASK}\n\nTime: 1 of 8 min")),
        pair("user", "Notes: listed a.txt."),
    ];
    for pair in got.chunks(2) {
        assert_eq!(pair[0], (first.clone(), "{\"command\": \"ls\"}".to_owned()));
        assert_eq!(pair[1], (after.clone(), "Done.".to_owned()));
    }
    assert!(outcomes(&projection).iter().all(|o| !o.baseline_only));

    // No one context holds both actions: packed, each is a record of its own.
    let packed = view()
        .packed(true)
        .project(&attempts(&steps, &log))
        .unwrap();
    assert_eq!(packed.records.len(), 4);
    assert!(packed
        .records
        .iter()
        .all(|r| messages(r).iter().filter(|m| m.train).count() == 1));

    // Imitation reads the same contexts.
    let sft = SftStep::new(Strength::Executable)
        .project(&attempts(&steps, &log))
        .unwrap();
    assert_eq!(read(&sft), got[..2]);

    // One record of the whole conversation cannot hold what the model saw.
    assert!(matches!(
        native_record(&trajectory(steps), &log),
        Err(Unprojectable::RequestsDiverge { .. })
    ));
}

#[test]
fn a_step_no_request_answered_is_left_out_and_counted_and_with_none_only_the_baseline_stays() {
    // The call step is not the log's (another id): it has no context its
    // model was sent, so it is no record, from the trajectory or otherwise;
    // the final answer still is, as it was sent.
    let (mut steps, log) = compacted();
    steps[1] = call(2, "c9", "shell", json!({"command": "ls"}), "a.txt");
    let projection = view().project(&attempts(&steps, &log)).unwrap();
    assert_eq!(
        projection.excluded.get(&Exclusion::RequestsUnpaired),
        Some(&2)
    );
    let got = read(&projection);
    assert_eq!(got.len(), 2, "the final answer of each attempt");
    assert!(got.iter().all(|(_, answer)| answer == "Done."));

    // No step pairs with a request: each attempt keeps only its place in
    // its group's baseline, and is counted.
    steps[3] = say(4, "Finished.");
    let projection = view().project(&attempts(&steps, &log)).unwrap();
    assert_eq!(
        projection.excluded.get(&Exclusion::RequestsUnpaired),
        Some(&2)
    );
    let outcomes = outcomes(&projection);
    assert_eq!(outcomes.len(), 2);
    assert!(outcomes.iter().all(|o| o.baseline_only));
    for record in &projection.records {
        assert!(messages(record).iter().all(|m| !m.train));
    }
    let written = projection.groups.iter().flat_map(|g| &g.trials);
    assert!(
        written.clone().all(|t| t.baseline_only),
        "the report says so"
    );
}

#[test]
fn calls_sent_back_in_another_order_are_read_request_by_request_as_sampled() {
    // The model asked for two calls; the agent ran them and sent them back
    // in the order they finished, which is how its trajectory holds them.
    let opening = vec![sent("system", SYSTEM), sent("user", ASK)];
    let mut later = opening.clone();
    later.push(sent("assistant", ""));
    for id in ["c1", "c0"] {
        later.push(
            json!({"role": "assistant", "content": {"tool_call_id": id, "function": {
            "name": "shell", "arguments": format!("{{\"command\":\"ls {id}\"}}")}}}),
        );
    }
    for id in ["c1", "c0"] {
        later.push(json!({"role": "tool", "content": {"tool_call_id": id, "content": id}}));
    }
    let mut both = calls("c0", "{\"command\":\"ls c0\"}");
    both.tool_calls.push(ModelToolCall {
        id: "c1".into(),
        name: "shell".into(),
        arguments: "{\"command\":\"ls c1\"}".into(),
    });
    let log = vec![request(1, opening, both), request(2, later, says("Done."))];
    let mut step = call(2, "c1", "shell", json!({"command": "ls c1"}), "c1");
    let mut other = call(2, "c0", "shell", json!({"command": "ls c0"}), "c0");
    step.tool_calls
        .as_mut()
        .unwrap()
        .append(other.tool_calls.as_mut().unwrap());
    step.observation
        .as_mut()
        .unwrap()
        .results
        .append(&mut other.observation.as_mut().unwrap().results);
    let steps = vec![user(1, ASK), step, say(3, "Done.")];
    let projection = view().project(&attempts(&steps, &log)).unwrap();

    assert!(projection.excluded.is_empty(), "{:?}", projection.excluded);
    let got = read(&projection);
    assert_eq!(got.len(), 4);
    assert_eq!(
        got[0].1, "{\"command\":\"ls c0\"};{\"command\":\"ls c1\"}",
        "as sampled"
    );
    let sent_back: Vec<&str> = got[1]
        .0
        .iter()
        .filter(|(role, _)| role == "tool")
        .map(|(_, text)| text.as_str())
        .collect();
    assert_eq!(sent_back, ["c1", "c0"], "the context as it was sent");
}
