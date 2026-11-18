// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the loop from a failed agent attempt to a
// verified corrected one, for its clients. If your team needs expertise in
// turning agent failures into training data you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec: a failed practice attempt is diagnosed, forked at its last good
//! step, continued by a teacher through sven's fork and kept only when the
//! task's own verifier passes the result.
//!
//! * The student's attempt is a real sven run (compact profile) on a scripted
//!   model: it reads the input, writes a wrong answer and stops. Its export is
//!   admitted through the intake, so the student is an experience.
//! * The diagnosis places the decisive error at the answer's write; the fork
//!   keeps the steps before it, and the teacher is shown exactly those.
//! * The examiner is a scripted stand-in: `task workspace` lays out the input,
//!   `task grade` passes an answer of 42 and fails any other.
//! * An accepted repair carries its lineage and its relations; a rejected one
//!   is counted and never becomes experience; a second run changes nothing.

#![allow(clippy::unwrap_used)]

use crate::common;

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use common::Scratch;
use serde_json::json;
use splinter_agent::solve::Model;
use splinter_agent::CancelToken;
use splinter_core::annotation::{AnnotationBody, RelationKind};
use splinter_core::model_ref::ModelRef;
use splinter_core::outcome::FailureClass;
use splinter_orchestrator::Context;
use splinter_pipelines::bench::artifact::sha256_hex;
use splinter_pipelines::bench::practice::BenchCheckout;
use splinter_pipelines::bench::repair::{
    repair, route, RepairOutcome, RepairRequest, Route, DEFAULT_REPAIR_DEADLINE,
};
use splinter_pipelines::bench::{ingest_bench_trials, IngestOptions};
use sven_sdk::config::{compact_profile, Config, RunProfile};
use sven_sdk::model::{
    CompletionRequest, MessageContent, ModelProvider, ResponseEvent, ResponseStream,
};
use sven_sdk::{ApprovalPolicy, Engine, Toolset};

/// A model that answers the n-th request of a conversation with the n-th
/// step of its script, counted by the tool results it has been shown, and
/// keeps what it was shown.
struct Steps {
    name: &'static str,
    script: Vec<Step>,
    seen: Mutex<Vec<String>>,
}

#[derive(Clone)]
enum Step {
    Call(&'static str, serde_json::Value),
    Say(&'static str),
}

#[async_trait::async_trait]
impl ModelProvider for Steps {
    fn name(&self) -> &str {
        "scripted"
    }
    fn model_name(&self) -> &str {
        self.name
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        // Its own calls answered so far: a continuation is shown the
        // student's calls too, which are not its script's.
        let answered = req
            .messages
            .iter()
            .filter(|m| {
                matches!(&m.content, MessageContent::ToolResult { tool_call_id, .. }
                if tool_call_id.starts_with(self.name))
            })
            .count();
        let shown: Vec<String> = req
            .messages
            .iter()
            .map(|m| format!("{:?}", m.content))
            .collect();
        self.seen.lock().unwrap().push(shown.join("\n"));
        let events = match self
            .script
            .get(answered)
            .cloned()
            .unwrap_or(Step::Say("done"))
        {
            Step::Call(tool, arguments) => vec![
                Ok(ResponseEvent::ToolCall {
                    index: 0,
                    id: format!("{}-{answered}", self.name),
                    name: tool.into(),
                    arguments: arguments.to_string(),
                }),
                Ok(ResponseEvent::Done),
            ],
            Step::Say(text) => vec![
                Ok(ResponseEvent::TextDelta(text.into())),
                Ok(ResponseEvent::Done),
            ],
        };
        Ok(Box::pin(futures::stream::iter(events)))
    }
}

fn steps(name: &'static str, script: Vec<Step>) -> Arc<Steps> {
    Arc::new(Steps {
        name,
        script,
        seen: Mutex::new(Vec::new()),
    })
}

const INPUT: &str = "alpha,6\nbeta,7\n";
const INSTRUCTION: &str = "Add the numbers in input.csv and write the sum to answer.json.";

/// The student's real sven run, compact profile, in a tree of the input.
async fn student_trajectory(tree: &Path) -> atif::Trajectory {
    fs::write(tree.join("input.csv"), INPUT).unwrap();
    let student = steps(
        "student",
        vec![
            Step::Call("read_file", json!({"path": "input.csv"})),
            Step::Call("write_file", json!({"path": "answer.json", "text": "7"})),
        ],
    );
    let mut config = Config::default();
    config.agent.profile = RunProfile::Compact;
    let mut agent = Engine::builder()
        .config(config)
        .model_provider(student as Arc<dyn ModelProvider>)
        .toolset(Toolset::coding())
        .project_root(tree)
        .approvals(ApprovalPolicy::Auto)
        .build()
        .unwrap()
        .agent("agent");
    agent.send(INSTRUCTION).await.unwrap();
    agent.trajectory()
}

/// A practice run directory: its staged material, its export of the one
/// failed trial, and the report naming the trial's experience.
fn practice_run(ctx: &Context, root: &Path, trajectory: &atif::Trajectory) -> PathBuf {
    let run = root.join("run");
    let staged = run.join("staged").join("sums");
    fs::create_dir_all(&staged).unwrap();
    fs::write(
        staged.join("practice.json"),
        json!({"schema": "bench.practice/2"}).to_string(),
    )
    .unwrap();
    let export = run.join("export");
    let trial = "sums-1__abc";
    let dir = export.join(trial);
    fs::create_dir_all(&dir).unwrap();
    let request = json!({"kind": "request", "seq": 1, "request": {"messages": [
        {"role": "system", "content": compact_profile::SYSTEM_PROMPT},
        {"role": "user", "content": INSTRUCTION}], "tools": []}});
    let files: Vec<(&str, Vec<u8>)> = vec![
        (
            "trajectory.atif.json",
            serde_json::to_vec(trajectory).unwrap(),
        ),
        ("model-requests.jsonl", format!("{request}\n").into_bytes()),
        (
            "verification.json",
            json!({"verdict": "fail", "reason": "functional_correctness 0.0000; short: sum=0.00"})
                .to_string()
                .into_bytes(),
        ),
        (
            "worktree.patch",
            b"diff --git a/answer.json b/answer.json\n".to_vec(),
        ),
    ];
    for (name, bytes) in &files {
        fs::write(dir.join(name), bytes).unwrap();
    }
    let score = json!({"schema": "bench.score/1", "status": "graded", "counted": true, "score": 0.0,
        "pass": false, "identity": {"task": "practice/sums-1", "digest": "d1", "collection": "sums"}});
    let manifest = json!({
        "schema": "bench.trial-artifact/1", "trial": trial,
        "task": {"id": "practice/sums-1", "version": "1", "digest": "d1", "set": "train"},
        "snapshot": "abc", "agent": {"name": "sven", "version": "5"},
        "models": {"main": {"model": "student"}}, "verifier": {"digest": "v1", "scoring": "functional"},
        "grade": {"reward": 0.0, "verdict": "fail", "metrics": {}}, "score": score,
        "files": files.iter().map(|(n, b)| ((*n).to_string(), json!(sha256_hex(b))))
            .collect::<serde_json::Map<String, serde_json::Value>>(),
    });
    let text = serde_json::to_string_pretty(&manifest).unwrap();
    fs::write(dir.join("manifest.json"), &text).unwrap();
    let row = json!({"trial": trial, "task": "practice/sums-1", "reward": 0.0, "verdict": "fail",
        "set": "train", "manifest_sha256": sha256_hex(text.as_bytes()), "path": trial});
    fs::write(export.join("index.jsonl"), format!("{row}\n")).unwrap();
    let ingested =
        ingest_bench_trials(ctx, std::slice::from_ref(&export), IngestOptions::default()).unwrap();
    assert!(ingested.refused.is_empty(), "{:?}", ingested.refused);
    let experience = splinter_pipelines::bench::split::trials(ctx).unwrap()[0]
        .experience
        .clone();
    fs::write(
        run.join("report.json"),
        json!({"trials": [{"trial": trial, "task": "sums-1", "experiences": [experience]}]})
            .to_string(),
    )
    .unwrap();
    run
}

/// The examiner: a workspace holds the input; a grade passes an answer of 42.
fn examiner(root: &Path) -> BenchCheckout {
    let dir = root.join("bench");
    fs::create_dir_all(&dir).unwrap();
    let script = format!(
        r#"#!/bin/sh
set -eu
echo "$@" >> "{log}"
case "$2" in
workspace) mkdir -p "$7"; printf '{input}' > "$7/input.csv"; echo '{{"schema":"bench.workspace/1"}}' ;;
grade)
  if [ "$(cat "$7/answer.json" 2>/dev/null)" = "42" ]; then v=pass; r=1.0; else v=fail; r=0.0; fi
  echo "{{\"schema\":\"bench.grade/1\",\"task\":\"practice/sums-1\",\"digest\":\"d1\",\"verdict\":\"$v\",\"reward\":$r,\"metrics\":{{}},\"reason\":\"sum\"}}" ;;
esac
"#,
        log = root.join("bench.log").display(),
        input = INPUT.replace('\n', "\\n"),
    );
    let path = dir.join("bench");
    fs::write(&path, script).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
    BenchCheckout::new(dir)
}

fn teacher_ref() -> ModelRef {
    "remote:scripted/teacher".parse().unwrap()
}

fn request<'a>(
    runs: &'a [PathBuf],
    checkout: &'a BenchCheckout,
    teacher: &'a ModelRef,
) -> RepairRequest<'a> {
    RepairRequest {
        runs,
        checkout,
        teacher,
        max: 8,
        jobs: 2,
        deadline: DEFAULT_REPAIR_DEADLINE,
        classes: &[],
        cancel: CancelToken::new(),
    }
}

fn setup(
    test: &str,
    answer: &'static str,
) -> (Scratch, Context, Arc<Steps>, Vec<PathBuf>, BenchCheckout) {
    let scratch = Scratch::new(test);
    let teacher = steps(
        "teacher",
        vec![Step::Call(
            "write_file",
            json!({"path": "answer.json", "text": answer}),
        )],
    );
    let ctx = Context::new(common::config(&scratch), true)
        .unwrap()
        .with_model(
            teacher_ref(),
            Model::new(
                Arc::clone(&teacher) as Arc<dyn ModelProvider>,
                "scripted/teacher",
            ),
        );
    let tree = scratch.0.join("student-tree");
    fs::create_dir_all(&tree).unwrap();
    let trajectory = ctx.block_on(student_trajectory(&tree));
    let run = practice_run(&ctx, &scratch.0, &trajectory);
    let checkout = examiner(&scratch.0);
    (scratch, ctx, teacher, vec![run], checkout)
}

#[test]
fn a_verified_repair_forks_at_the_last_good_step_and_records_its_lineage() {
    let (scratch, ctx, teacher, runs, checkout) = setup("repair-accepted", "42");
    let teacher_ref = teacher_ref();
    let report = repair(&ctx, &request(&runs, &checkout, &teacher_ref)).unwrap();
    let [record] = report.repairs.as_slice() else {
        panic!("one repair: {:?}", report.repairs)
    };
    assert_eq!(record.diagnosis.class, FailureClass::WrongContent);
    assert_eq!(
        record.outcome,
        RepairOutcome::Accepted,
        "{:?}",
        record.reason
    );
    // The answer's write is the decisive step; the read before it is kept.
    assert_eq!(
        record.keep_through,
        record.diagnosis.decisive_step.unwrap() - 1
    );
    let shown = teacher.seen.lock().unwrap().join("\n");
    assert!(
        shown.contains("input.csv"),
        "the teacher sees the kept read"
    );
    assert!(
        !shown.contains("\\\"7\\\"") && !shown.contains("\"text\":\"7\""),
        "never the student's wrong write: {shown}"
    );

    let id = record.experience.clone().unwrap();
    let store = ctx.experiences();
    let repaired = store.get(&id).unwrap();
    let student = store.get(&record.student).unwrap();
    let prefix = record.prefix_steps.unwrap();
    for (mine, theirs) in repaired
        .trajectory
        .steps
        .iter()
        .zip(&student.trajectory.steps)
        .take(prefix)
    {
        assert_eq!(mine.message.as_text(), theirs.message.as_text());
        assert_eq!(
            mine.is_copied_context,
            Some(true),
            "the shared prefix is never supervised"
        );
    }
    assert!(repaired.trajectory.steps[prefix..]
        .iter()
        .all(|s| s.is_copied_context.is_none()));
    let lineage = &repaired.trajectory.extra.as_ref().unwrap()["correction"];
    assert_eq!(lineage["student"], json!(record.student));
    assert_eq!(lineage["keep_through"], json!(record.keep_through));
    assert_eq!(lineage["teacher"], json!("scripted/teacher"));
    assert_eq!(lineage["grade"]["verdict"], json!("pass"));
    let relations: Vec<RelationKind> = store
        .annotations(&id)
        .unwrap()
        .annotations
        .iter()
        .filter_map(|a| match &a.body {
            AnnotationBody::Relation { kind, other } if *other == record.student => Some(*kind),
            _ => None,
        })
        .collect();
    assert_eq!(relations, [RelationKind::RevisionOf]);
    assert!(report.repair_set.is_some());

    // The same attempt and teacher again: nothing is run or stored.
    let log = fs::read_to_string(scratch.0.join("bench.log")).unwrap();
    let again = repair(&ctx, &request(&runs, &checkout, &teacher_ref)).unwrap();
    assert_eq!(again.by_class["wrong_content"].held, 1);
    assert!(again.repair_set.is_none());
    assert_eq!(
        fs::read_to_string(scratch.0.join("bench.log")).unwrap(),
        log,
        "the examiner was not asked again"
    );
}

#[test]
fn an_unverified_repair_is_rejected_and_never_becomes_experience() {
    let (_scratch, ctx, _teacher, runs, checkout) = setup("repair-rejected", "41");
    let teacher_ref = teacher_ref();
    let before = ctx.experiences().list().unwrap().len();
    let report = repair(&ctx, &request(&runs, &checkout, &teacher_ref)).unwrap();
    assert_eq!(report.repairs[0].outcome, RepairOutcome::Rejected);
    assert_eq!(report.repairs[0].grade.as_ref().unwrap().verdict, "fail");
    assert_eq!(report.by_class["wrong_content"].rejected, 1);
    assert!(report.repair_set.is_none());
    assert_eq!(
        ctx.experiences().list().unwrap().len(),
        before,
        "nothing was stored"
    );
}

#[test]
fn an_infrastructure_or_task_failure_is_never_repaired() {
    for class in FailureClass::ALL.into_iter().filter(|c| !c.is_model()) {
        let diagnosis =
            splinter_core::diagnosis::DiagnosticResult::new(class, "r", Some(4), Vec::new());
        assert!(matches!(route(&diagnosis), Route::Skip(_)), "{class:?}");
    }
}
