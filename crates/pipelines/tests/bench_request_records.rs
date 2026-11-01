// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training data for multi-role agent systems,
// where each model role learns from exactly the requests it was sent, for its
// clients. If your team needs expertise in fine-tuning the roles of an agent
// harness, including those that read images, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec: a harness trial's conversations keep the requests their models were
//! sent, pictures included, and a dataset of the roles that read pictures is
//! built from them when the build is told the base reads images: each record
//! is a request as sent, with the model's answer supervised, and the write
//! is checked by brain's own parser. Without it the conversations whose model
//! was shown pictures stay counted out.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::fs;
use std::path::{Path, PathBuf};

use common::bench::{context, sft_request, NO_OPTIONS};
use serde_json::{json, Value};
use splinter_data::Exclusion;
use splinter_pipelines::bench::artifact::sha256_hex;
use splinter_pipelines::bench::ingest_bench_trials;
use splinter_pipelines::datasets::build;

const PNG: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAoAAAAHgCAIAAAAAAAAAAAAAAAAA";

/// Where the pictures of a conversation are.
#[derive(Clone, Copy, PartialEq)]
enum Pictures {
    None,
    /// In the user turn, which the conversation's trajectory does not hold.
    InTheTask,
    /// In the result of the first tool call.
    InAToolResult,
}

struct Conv {
    name: &'static str,
    role: &'static str,
    pictures: Pictures,
}

const CONVS: [Conv; 3] = [
    Conv {
        name: "survey",
        role: "visual",
        pictures: Pictures::InTheTask,
    },
    Conv {
        name: "specify",
        role: "specifier",
        pictures: Pictures::None,
    },
    Conv {
        name: "review",
        role: "reviewer",
        pictures: Pictures::InAToolResult,
    },
];

fn system(c: &Conv) -> String {
    format!("You are the {} of the trial.", c.name)
}

fn instruction(c: &Conv) -> String {
    format!("Phase: {}.", c.name)
}

fn trajectory(c: &Conv) -> String {
    let mut steps = vec![json!({"step_id": 1, "source": "system", "message": system(c)})];
    if c.pictures != Pictures::InTheTask {
        steps.push(json!({"step_id": 2, "source": "user", "message": instruction(c)}));
    }
    steps.push(
        json!({"step_id": 3, "source": "agent", "model_name": "m", "message": "\n\n\n",
        "reasoning_content": "Look first.",
        "tool_calls": [{"tool_call_id": "c0", "function_name": "shell",
                        "arguments": {"command": "ls"}}],
        "observation": {"results": [{"source_call_id": "c0", "content": "part.py"}]}}),
    );
    steps.push(json!({"step_id": 4, "source": "agent", "model_name": "m", "message": "Done."}));
    json!({
        "schema_version": "ATIF-v1.7", "trajectory_id": format!("run/{}", c.name),
        "agent": {"name": "sven", "version": "4.0.0", "model_name": "main",
                  "tool_definitions": [{"type": "function", "function": {
                      "name": "shell", "description": "Run a command.",
                      "parameters": {"type": "object"}}}]},
        "extra": {"harness": {"conversation": c.name, "role": c.role}},
        "steps": steps,
    })
    .to_string()
}

/// The two requests of a conversation as sven sent them, each with the
/// response it got.
fn requests(c: &Conv, seq: &mut u64) -> String {
    let user = match c.pictures {
        Pictures::InTheTask => json!([
            {"type": "text", "text": instruction(c)},
            {"type": "text", "text": "Attached image: sheet.png (1x1, image/png)"},
            {"type": "image", "image_url": PNG}]),
        _ => json!(instruction(c)),
    };
    let result = match c.pictures {
        Pictures::InAToolResult => json!([
            {"type": "text", "text": "Attached image: top.png"},
            {"type": "image", "image_url": PNG}]),
        _ => json!("part.py"),
    };
    let opening = vec![
        json!({"role": "system", "content": system(c)}),
        json!({"role": "user", "content": user}),
    ];
    let mut later = opening.clone();
    later.extend([
        json!({"role": "assistant", "content": "\n\n\n", "reasoning": "Look first."}),
        json!({"role": "assistant", "content": {"tool_call_id": "c0", "function": {
            "name": "shell", "arguments": "{\"command\":\"ls\"}"}}}),
        json!({"role": "tool", "content": {"tool_call_id": "c0", "content": result}}),
    ]);
    let tools = json!([{"name": "shell", "description": "Run a command.",
                        "parameters": {"type": "object"}}]);
    let responses = [
        json!({"text": "\n\n\n", "reasoning": "Look first.", "tool_calls": [
            {"index": 0, "id": "c0", "name": "shell", "arguments": "{\"command\": \"ls\"}"}]}),
        json!({"text": "\n\nDone.", "reasoning": "", "tool_calls": []}),
    ];
    let mut log = String::new();
    for (messages, response) in [opening, later].into_iter().zip(responses) {
        *seq += 1;
        let request = json!({"kind": "request", "seq": *seq, "provider": "p", "model": c.role,
            "request": {"messages": messages, "tools": tools}});
        log.push_str(&format!(
            "{request}\n{}\n",
            json!({"kind": "response", "seq": *seq, "response": response})
        ));
    }
    log
}

fn export(dir: &Path) -> PathBuf {
    let trial = dir.join("t1");
    fs::create_dir_all(trial.join("conversations")).unwrap();
    let mut seq = 0;
    let mut files: Vec<(String, String)> = vec![
        (
            "trajectory.atif.json".into(),
            json!({"schema_version": "ATIF-v1.7", "agent": {"name": "sven", "version": "4.0.0"},
                   "extra": {"harness": {"conversations": CONVS.iter().map(|c| c.name).collect::<Vec<_>>()}},
                   "steps": []})
            .to_string(),
        ),
        (
            "model-requests.jsonl".into(),
            CONVS.iter().map(|c| requests(c, &mut seq)).collect(),
        ),
        ("verification.json".into(), "{\"ok\": true}".into()),
        ("submission.patch".into(), String::new()),
    ];
    for (n, c) in CONVS.iter().enumerate() {
        files.push((
            format!("conversations/{n:04}-{}.json", c.name),
            trajectory(c),
        ));
    }
    for (file, text) in &files {
        fs::write(trial.join(file), text).unwrap();
    }
    let manifest = json!({
        "schema": "bench.trial-artifact/1", "trial": "t1",
        "task": {"id": "org/cad", "version": "1", "digest": "d1", "set": "train"},
        "agent": {"name": "sven", "version": "4.0.0"},
        "verifier": {"digest": "v1"},
        "grade": {"reward": 1.0, "verdict": "pass"},
        "files": files.iter().map(|(n, t)| (n.clone(), json!(sha256_hex(t.as_bytes()))))
            .collect::<serde_json::Map<String, Value>>(),
    });
    let text = serde_json::to_string_pretty(&manifest).unwrap();
    fs::write(trial.join("manifest.json"), &text).unwrap();
    let row = json!({"trial": "t1", "task": "org/cad", "reward": 1.0, "verdict": "pass",
                     "set": "train", "manifest_sha256": sha256_hex(text.as_bytes()), "path": "t1"});
    fs::write(dir.join("index.jsonl"), format!("{row}\n")).unwrap();
    dir.to_path_buf()
}

fn lines(path: &Path) -> Vec<Value> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

#[test]
fn the_pictures_a_model_was_shown_are_built_into_the_records_of_a_base_that_reads_them() {
    let (scratch, ctx) = context("request-records");
    let done = ingest_bench_trials(&ctx, &[export(&scratch.0.join("export"))], NO_OPTIONS).unwrap();
    assert!(done.refused.is_empty(), "{:?}", done.refused);
    let set = done.train_set.unwrap();

    // A picture a tool returned is a picture the model was shown, too.
    let mut request = sft_request(set);
    let text_only = build(&ctx, &request).unwrap();
    assert_eq!(text_only.records, 2, "only the specifier's two actions");
    assert_eq!(text_only.excluded.get(&Exclusion::ShownImages), Some(&2));

    request.images = true;
    let built = build(&ctx, &request).unwrap();
    assert_eq!(built.records, 6, "two actions of each of three roles");
    assert!(!built.excluded.contains_key(&Exclusion::ShownImages));
    let records = lines(&built.path);
    let images = |record: &Value| -> usize {
        record["messages"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|m| m["content"].as_array())
            .flatten()
            .filter(|part| part["type"] == "image_url")
            .count()
    };
    let by_system = |name: &str| -> Vec<&Value> {
        records
            .iter()
            .filter(|r| r["messages"][0]["content"] == format!("You are the {name} of the trial."))
            .collect()
    };
    // The survey's instruction carries the sheet, in both of its records.
    let survey = by_system("survey");
    assert_eq!(survey.len(), 2);
    assert!(survey.iter().all(|r| images(r) == 1));
    assert_eq!(
        survey[0]["messages"][1]["content"][0]["text"],
        "Phase: survey.Attached image: sheet.png (1x1, image/png)"
    );
    // The review's picture arrives with its first tool result: only the
    // second record has it.
    let review = by_system("review");
    assert_eq!((images(review[0]), images(review[1])), (0, 1));
    assert_eq!(
        by_system("specify")
            .iter()
            .map(|r| images(r))
            .sum::<usize>(),
        0
    );
    // Whatever the pictures, only the answer is supervised, and it holds none.
    for record in &records {
        let supervised: Vec<&Value> = record["messages"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|m| m["train"] == true)
            .collect();
        assert_eq!(supervised.len(), 1, "{record}");
        assert!(supervised[0]["content"].is_string());
    }
    // The first action is the call, with the reasoning the model had.
    let call = &survey[0]["messages"]
        .as_array()
        .unwrap()
        .last()
        .unwrap()
        .clone();
    assert_eq!(call["reasoning_content"], "Look first.");
    assert_eq!(
        call["tool_calls"][0]["function"]["arguments"],
        "{\"command\":\"ls\"}"
    );
}
