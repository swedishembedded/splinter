// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements on-policy learning from privileged
// demonstrations, for its clients. If your team needs expertise in turning a
// guided solution into training data the student could have produced itself,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: a repair of a bench-compiled dataset's task. The run is the
//! examiner's export of an adopted Harbor job and a report that names the
//! dataset as its material; the task's environment gives the agent its tools
//! as an MCP server (no built-in tool), which the examiner's session announces.
//!
//! * The fork's engine runs under the image's Sven configuration the session
//!   announces: the student is offered the task's tool and nothing else, and
//!   its server runs in the session (`session-exec --stdin`), where the
//!   recorded calls are replayed. The session runs one command at a time, so
//!   no server may stay connected past its call: the grade must get its turn.
//! * `path_guided` forks a guided pass of the task there, and the examiner's
//!   grade decides.

#![allow(clippy::unwrap_used)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::json;
use splinter_agent::solve::Model;
use splinter_core::annotation::{Annotation, AnnotationBody, Outcome, Producer, Strength};
use splinter_core::model_ref::ModelRef;
use splinter_orchestrator::Context;
use splinter_pipelines::bench::artifact::sha256_hex;
use splinter_pipelines::bench::practice::BenchCheckout;
use splinter_pipelines::bench::repair::{repair, InterventionKind, RepairOutcome};
use splinter_pipelines::bench::{ingest_bench_trials, IngestOptions};
use sven_sdk::model::ModelProvider;

use crate::bench_repair::{request, steps, teacher_ref, Step};
use crate::common::{self, Scratch};

const INSTRUCTION: &str = "Write 42 to answer.json.";
const GUIDANCE: &str = "The answer is 42, not 7.";

/// A stdio MCP server with one tool, `write`, that writes its text to a file.
const SERVER: &str = r#"
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue
    method = request["method"]
    if method == "initialize":
        result = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": "files", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": [{"name": "write", "description": "Write a file.", "inputSchema": {"type": "object",
                  "properties": {"path": {"type": "string"}, "text": {"type": "string"}}, "required": ["path", "text"]}}]}
    elif method == "tools/call":
        arguments = request["params"]["arguments"]
        open(arguments["path"], "w").write(arguments["text"])
        result = {"content": [{"type": "text", "text": "wrote " + arguments["path"]}]}
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
"#;

fn student_ref() -> ModelRef {
    "remote:scripted/student".parse().unwrap()
}

fn call(id: u64, call: &str, text: &str) -> atif::TraceStep {
    let mut step = atif::TraceStep::new(id, atif::StepOrigin::Agent, "");
    step.tool_calls = Some(vec![atif::ToolInvocation::new(call, "files-write")
        .with_arguments(json!({"path": "answer.json", "text": text}))]);
    step.observation = Some(atif::StepObservation::single(
        atif::ObservationEntry::for_call(call, "wrote answer.json"),
    ));
    step
}

/// The examiner: a compiled task's session whose environment gives the agent
/// the `files` server; a grade passes an answer of 42.
fn examiner(root: &Path, server: &Path) -> BenchCheckout {
    let dir = root.join("bench");
    fs::create_dir_all(&dir).unwrap();
    let servers =
        json!([{"name": "files", "transport": "stdio", "command": "python3", "args": [server]}]);
    // The image's own Sven configuration: the server's tool is the agent's only one.
    let sven_config = json!(format!(
        "tools:\n  only: [\"files-write\"]\nmcp_servers:\n  files:\n    transport:\n      type: stdio\n      command: python3\n      args: [\"{}\"]\n",
        server.display()
    ));
    let script = format!(
        r#"#!/bin/sh
set -eu
echo "$@" >> "{log}"
case "$2" in
session)
  ws="$7"; mkdir -p "$ws"
  printf '%s\n' '{{"schema":"bench.session/1","socket":"/nowhere/session.sock","workspace":"'"$ws"'","sandbox_workspace":"/workspace/repo","service":false,"mcp_servers":{servers},"sven_config":{sven_config}}}'
  while read -r line; do
    case "$line" in
    *grade*)
      # One command or grade at a time, as the examiner's session runs them:
      # a command still running (a server nobody ended) fails the grade.
      if ! flock -w 5 "{lock}" true; then echo '{{"error":"infrastructure: the session is still busy"}}'; continue; fi
      if [ "$(cat "$ws/answer.json" 2>/dev/null)" = "42" ]; then v=pass; r=1.0; else v=fail; r=0.0; fi
      echo "{{\"schema\":\"bench.grade/1\",\"task\":\"compiled/sums\",\"digest\":\"d1\",\"verdict\":\"$v\",\"reward\":$r,\"metrics\":{{}},\"reason\":null}}" ;;
    esac
  done ;;
session-exec)
  shift 2; [ "$1" = "--stdin" ] && shift; shift 2
  exec flock "{lock}" "$@" ;;
esac
"#,
        log = root.join("bench.log").display(),
        lock = root.join("session.lock").display(),
    );
    let path = dir.join("bench");
    fs::write(&path, script).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
    BenchCheckout::new(dir)
}

/// The run: the examiner's export of the failed trial (an adopted Harbor job's
/// trial of the dataset's `train/sums`) and a report naming the dataset.
fn compiled_run(ctx: &Context, root: &Path, trajectory: &atif::Trajectory) -> PathBuf {
    let dataset = root.join("dataset");
    fs::create_dir_all(dataset.join("train/sums")).unwrap();
    fs::write(
        dataset.join("bench-compile.json"),
        json!({"schema": "bench.compile/1", "tasks": []}).to_string(),
    )
    .unwrap();
    let run = root.join("run");
    let export = run.join("export");
    let trial = "sums__abc";
    let dir = export.join(trial);
    fs::create_dir_all(&dir).unwrap();
    let request = json!({"kind": "request", "seq": 1, "request": {"messages": [
        {"role": "system", "content": "You are the agent."},
        {"role": "user", "content": INSTRUCTION}], "tools": []}});
    let files: Vec<(&str, Vec<u8>)> = vec![
        (
            "trajectory.atif.json",
            serde_json::to_vec(trajectory).unwrap(),
        ),
        ("model-requests.jsonl", format!("{request}\n").into_bytes()),
        (
            "verification.json",
            json!({"verdict": "fail", "reason": "the answer is not 42"})
                .to_string()
                .into_bytes(),
        ),
    ];
    for (name, bytes) in &files {
        fs::write(dir.join(name), bytes).unwrap();
    }
    let score = json!({"schema": "bench.score/1", "status": "graded", "counted": true, "score": 0.0,
        "pass": false, "identity": {"task": "compiled/sums", "digest": "d1", "collection": "compiled"}});
    let manifest = json!({
        "schema": "bench.trial-artifact/1", "trial": trial,
        "task": {"id": "compiled/sums", "version": "d1", "digest": "d1", "set": "train", "path": "train/sums"},
        "snapshot": null, "agent": {"name": "sven", "version": "5"},
        "models": {"main": {"model": "student"}}, "verifier": {"digest": "v1", "scoring": "all"},
        "grade": {"reward": 0.0, "verdict": "fail", "metrics": {}}, "score": score,
        "files": files.iter().map(|(n, b)| ((*n).to_string(), json!(sha256_hex(b))))
            .collect::<serde_json::Map<String, serde_json::Value>>(),
    });
    let text = serde_json::to_string_pretty(&manifest).unwrap();
    fs::write(dir.join("manifest.json"), &text).unwrap();
    let row = json!({"trial": trial, "task": "compiled/sums", "reward": 0.0, "verdict": "fail",
        "set": "train", "manifest_sha256": sha256_hex(text.as_bytes()), "path": trial});
    fs::write(export.join("index.jsonl"), format!("{row}\n")).unwrap();
    let ingested =
        ingest_bench_trials(ctx, std::slice::from_ref(&export), IngestOptions::default()).unwrap();
    assert!(ingested.refused.is_empty(), "{:?}", ingested.refused);
    fs::write(
        run.join("report.json"),
        json!({"provenance": {"dir": dataset}}).to_string(),
    )
    .unwrap();
    run
}

#[test]
fn a_compiled_tasks_guided_path_is_resampled_under_its_environments_tools() {
    let scratch = Scratch::new("compiled-repair");
    let server = scratch.0.join("files.py");
    fs::write(&server, SERVER).unwrap();
    let student = steps(
        "pathstudent",
        vec![Step::Call(
            "files-write",
            json!({"path": "answer.json", "text": "42"}),
        )],
    );
    let ctx = Context::new(common::config(&scratch), true)
        .unwrap()
        .with_model(
            student_ref(),
            Model::new(
                Arc::clone(&student) as Arc<dyn ModelProvider>,
                "scripted/student",
            ),
        )
        .with_model(
            teacher_ref(),
            Model::new(
                steps("teacher", vec![]) as Arc<dyn ModelProvider>,
                "scripted/teacher",
            ),
        );
    let mut failed = atif::Trajectory::new("ATIF-v1.7", atif::AgentProfile::new("sven", "5"));
    failed.steps = vec![
        atif::TraceStep::new(1, atif::StepOrigin::User, INSTRUCTION),
        call(2, "s-1", "7"),
        atif::TraceStep::new(3, atif::StepOrigin::Agent, "Wrote 7."),
    ];
    let runs = vec![compiled_run(&ctx, &scratch.0, &failed)];
    let checkout = examiner(&scratch.0, &server);

    // The guided pass: the same call, the person's turn, the right write.
    let store = ctx.experiences();
    let attempt = splinter_pipelines::bench::split::trials(&ctx).unwrap()[0]
        .experience
        .clone();
    let mut path = store.get(&attempt).unwrap();
    let mut typed = atif::TraceStep::new(4, atif::StepOrigin::User, GUIDANCE);
    typed.extra = Some(json!({"sven": {"origin": "human"}}));
    let mut steps_kept: Vec<atif::TraceStep> = path.trajectory.steps.clone();
    steps_kept.extend([
        typed,
        call(0, "g-1", "42"),
        atif::TraceStep::new(0, atif::StepOrigin::Agent, "Done."),
    ]);
    for (index, step) in steps_kept.iter_mut().enumerate() {
        step.step_id = index as u64 + 1;
    }
    path.trajectory.steps = steps_kept;
    let id = store.put(&path).unwrap();
    store
        .annotate(&Annotation {
            experience: id,
            producer: Producer {
                name: "verifier".into(),
                version: "1".into(),
            },
            body: AnnotationBody::Verdict {
                outcome: Outcome::Pass,
                strength: Strength::Executable,
                evidence: json!({}),
            },
        })
        .unwrap();

    let teacher = teacher_ref();
    let student_model = student_ref();
    let only = [InterventionKind::PathGuided];
    let mut asked = request(&runs, &checkout, &teacher);
    asked.student = Some(&student_model);
    asked.interventions = &only;
    let report = repair(&ctx, &asked).unwrap();
    let [record] = report.repairs.as_slice() else {
        panic!("one repair: {:?} {:?}", report.repairs, report.skipped)
    };
    assert_eq!(
        record.outcome,
        RepairOutcome::Accepted,
        "{:?}",
        record.reason
    );

    // The student was offered the task's tool and nothing else, and never the
    // person's words.
    let shown = student.seen.lock().unwrap().join("\n");
    assert!(!shown.contains(GUIDANCE), "{shown}");
    assert!(
        shown.contains("offered tools: files-write\n")
            || shown.ends_with("offered tools: files-write"),
        "{shown}"
    );
    let log = fs::read_to_string(scratch.0.join("bench.log")).unwrap();
    assert!(
        log.contains("session-exec --stdin /nowhere/session.sock -- python3"),
        "{log}"
    );
    assert!(
        log.contains("session /") && log.contains("--task sums"),
        "{log}"
    );
}
