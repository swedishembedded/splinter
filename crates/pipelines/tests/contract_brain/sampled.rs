// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for proving that the training records one system writes are the records
// another system's trainer reads, for its clients. If your team needs expertise in training-data contracts between
// agent runtimes and trainers then you can procure our services by sending an email to
// info@swedishembedded.com.

//! A group of two attempts at one bench task whose agent steps recorded exactly what the policy sampled: the ids it
//! was given, the ids it generated and one log-probability per generated id. The ids are the toy tokenizer's, worked
//! out here from the texts of the turns, so brain can check them against its own rendering of the same turns.

#![allow(clippy::unwrap_used)]

use atif::{StepMetrics, StepOrigin, TraceStep};
use serde_json::{json, Value};
use splinter_core::annotation::{Annotation, AnnotationBody, Outcome, Producer, Strength};
use splinter_core::experience::{Environment, Task};
use splinter_core::harness::{ModelCall, ModelResponse, ModelToolCall, RecordedCalls};
use splinter_core::prompt::SYSTEM_PROMPT;
use splinter_data::Corpus;

use crate::bench::{answer, answer_prompt, turn};
use crate::common::{call, experience, trajectory, user};

const INSTRUCTION: &str = "Design a bracket.";
const ARGUMENTS: &str = "{\"command\":\"ls\"}";
const RESULT: &str = "part.py";
const FINAL: &str = "Done.";

/// What the policy sampled for the two actions of every attempt, and the log-probabilities it reported.
pub struct Sampled {
    pub call: Vec<i64>,
    pub answer: Vec<i64>,
}

impl Sampled {
    /// The ids of the two sampled actions: the call's name and arguments, and the final answer, each closed by the
    /// end-of-turn marker.
    pub fn new() -> Self {
        Self {
            call: answer(&format!("shell{ARGUMENTS}")),
            answer: answer(FINAL),
        }
    }
}

fn task() -> Task {
    Task::new(
        "bench/cad",
        Vec::new(),
        Environment::new(
            Environment::BENCH,
            json!({"task": {"id": "cad/bracket", "version": "1"}}),
        ),
        INSTRUCTION,
        Vec::new(),
    )
    .unwrap()
}

fn context() -> Vec<i64> {
    let mut ids = turn("system", SYSTEM_PROMPT);
    ids.extend(turn("user", INSTRUCTION));
    ids
}

fn metrics(prompt: Vec<i64>, completion: &[i64], tokenizer: &str) -> StepMetrics {
    // A figure per sampled id, each at most zero: the log-probabilities under the policy's own distribution.
    let logprobs = (0..completion.len())
        .map(|i| -0.125 * (i + 1) as f64)
        .collect();
    StepMetrics {
        prompt_token_ids: Some(prompt),
        completion_token_ids: Some(completion.to_vec()),
        logprobs: Some(logprobs),
        extra: Some(json!({"sven": {"logprob_kind": "policy", "tokenizer_digest": tokenizer}})),
        ..StepMetrics::default()
    }
}

/// The two requests the harness recorded: the opening, answered by the call, and the one after its result.
fn requests() -> Vec<ModelCall> {
    let opening = vec![
        json!({"role": "system", "content": SYSTEM_PROMPT}),
        json!({"role": "user", "content": INSTRUCTION}),
    ];
    let mut later = opening.clone();
    later.extend([
        json!({"role": "assistant", "content": "", "reasoning": "look"}),
        json!({"role": "assistant", "content": {"tool_call_id": "c0", "function": {
            "name": "shell", "arguments": ARGUMENTS}}}),
        json!({"role": "tool", "content": {"tool_call_id": "c0", "content": RESULT}}),
    ]);
    let shell =
        json!({"name": "shell", "description": "Run a command.", "parameters": {"type": "object"}});
    let response = |text: &str, calls: Vec<ModelToolCall>| ModelResponse {
        text: text.into(),
        reasoning: (!calls.is_empty()).then(|| "look".into()),
        tool_calls: calls,
    };
    vec![
        ModelCall {
            seq: 1,
            messages: opening,
            tools: vec![shell.clone()],
            response: response(
                "",
                vec![ModelToolCall {
                    id: "c0".into(),
                    name: "shell".into(),
                    arguments: ARGUMENTS.into(),
                }],
            ),
        },
        ModelCall {
            seq: 2,
            messages: later,
            tools: vec![shell],
            response: response(FINAL, Vec::new()),
        },
    ]
}

fn graded(
    experience: &splinter_core::experience::Experience,
    outcome: Outcome,
    trial: &str,
) -> Annotation {
    let metrics = if outcome == Outcome::Pass {
        json!({"iou": 1.0})
    } else {
        json!({"iou": 0.0})
    };
    Annotation {
        experience: experience.id().unwrap(),
        producer: Producer {
            name: "bench".into(),
            version: "v".into(),
        },
        body: AnnotationBody::Verdict {
            outcome,
            strength: Strength::Executable,
            evidence: json!({"trial": trial, "task": {"id": "cad/bracket", "version": "1"},
                "model": "m1", "metrics": metrics}),
        },
    }
}

/// A pass and a fail of the bench task, so the group has a spread of rewards.
pub fn group(tokenizer: &str) -> Corpus {
    let sampled = Sampled::new();
    let mut corpus = Corpus::new();
    for (index, (name, outcome)) in [("t1", Outcome::Pass), ("t2", Outcome::Fail)]
        .into_iter()
        .enumerate()
    {
        let mut call_step = call(3, "c0", "shell", json!({"command": "ls"}), RESULT);
        call_step.reasoning_content = Some("look".into());
        call_step.metrics = Some(metrics(
            [context(), answer_prompt()].concat(),
            &sampled.call,
            tokenizer,
        ));
        let mut answer_step = TraceStep::new(4, StepOrigin::Agent, FINAL);
        let after_call = [
            context(),
            turn("assistant", &format!("shell{ARGUMENTS}")),
            turn("tool", RESULT),
            answer_prompt(),
        ]
        .concat();
        answer_step.metrics = Some(metrics(after_call, &sampled.answer, tokenizer));
        let mut trajectory = trajectory(vec![
            TraceStep::new(1, StepOrigin::System, SYSTEM_PROMPT),
            user(2, INSTRUCTION),
            call_step,
            answer_step,
        ]);
        trajectory.agent.tool_definitions = Some(vec![json!({"type": "function", "function": {
            "name": "shell", "description": "Run a command.", "parameters": {"type": "object"}}})]);
        trajectory.extra = Some(json!({
            "splinter_harness": {"conversation": "designer", "role": "designer",
                "role_source": "export", "shown_images": false},
            "splinter_harness_calls": RecordedCalls::record("abc", &requests()).to_value(),
        }));
        let at = format!("2026-11-01T00:00:00.{index:03}Z");
        let experience = experience(&task(), trajectory, Some(FINAL), &at);
        let note = graded(&experience, outcome, name);
        corpus.insert(experience, vec![note]).unwrap();
    }
    corpus
}

/// A record line as a JSON value.
pub fn parse(line: &str) -> Value {
    serde_json::from_str(line).unwrap()
}
