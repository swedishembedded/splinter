// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the loop from a failed agent attempt to a
// verified corrected one, for its clients. If your team needs expertise in
// turning agent failures into training data you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec: the interventions behind the one `Intervention` interface (retry,
//! fork correction, knowledge-assisted correction, decomposition), routed by
//! the failure's class with a budget, each verified by the task's own
//! verifier in the examiner's session.
//!
//! The student's failed attempt, its practice run and the scripted examiner
//! are the fork correction's fixtures (`bench_repair`): an answer of 42
//! passes. The teacher writes 41 (a direct correction the verifier rejects);
//! the student's own model is a scripted policy that acts on the subgoal
//! turns it is given.
//!
//! * A decomposition yields a pass where the direct repair fails; its
//!   subgoals, proposer and per-subgoal outcomes are in the record and the
//!   lineage, the guided trajectory supervises only the student's own steps,
//!   and a rerun changes nothing.
//! * An unverified decomposition is rejected and never becomes experience.
//! * A retry samples the student's model under a seed of its own and the
//!   practice run's sampling.
//! * A knowledge-assisted correction records what it retrieved, from which
//!   snapshot; the teacher sees the passage, the kept trajectory does not;
//!   sealed material is never read, named or found in a directory.

#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde_json::json;
use splinter_agent::solve::Model;
use splinter_agent::CancelToken;
use splinter_core::model_ref::ModelRef;
use splinter_core::outcome::FailureClass;
use splinter_orchestrator::Context;
use splinter_pipelines::bench::practice::BenchCheckout;
use splinter_pipelines::bench::repair::{
    repair, repair_with, Intervention, InterventionKind, KnowledgeBase, Proposer, RepairOutcome,
    RepairReport, Retry, Routing, SubgoalState, DEFAULT_REPAIR_DEADLINE,
};
use sven_sdk::model::{
    CompletionRequest, MessageContent, ModelProvider, ResponseEvent, ResponseStream, Role,
};

use crate::bench_repair::{
    examiner, practice_run, request, steps, student_trajectory, teacher_ref, Step, Steps,
};
use crate::common::bench::sft_request;
use crate::common::{self, Scratch};
use splinter_pipelines::datasets::build;

/// The student's own model: on a subgoal turn ("Step k of n") it takes the
/// k-th action, at the start of a continuation the last, and after the
/// result of its own call it says it is done. It keeps the sampling seed of
/// every request.
struct Policy {
    actions: Vec<(&'static str, serde_json::Value)>,
    seeds: Mutex<Vec<Option<u64>>>,
    calls: Mutex<usize>,
}

#[async_trait::async_trait]
impl ModelProvider for Policy {
    fn name(&self) -> &str {
        "scripted"
    }
    fn model_name(&self) -> &str {
        "policy"
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        self.seeds.lock().unwrap().push(req.sampling.seed);
        let last = req.messages.last().unwrap();
        let action = match &last.content {
            MessageContent::ToolResult { tool_call_id, .. }
                if tool_call_id.starts_with("policy") =>
            {
                None
            }
            MessageContent::Text(text) if last.role == Role::User => {
                let step = text
                    .split("Step ")
                    .nth(1)
                    .and_then(|rest| rest.split(' ').next())
                    .and_then(|k| k.parse::<usize>().ok())
                    .unwrap_or(1);
                self.actions.get(step - 1)
            }
            _ => self.actions.last(),
        };
        let events = match action {
            Some((tool, arguments)) => {
                let mut n = self.calls.lock().unwrap();
                *n += 1;
                vec![
                    Ok(ResponseEvent::ToolCall {
                        index: 0,
                        id: format!("policy-{n}"),
                        name: (*tool).into(),
                        arguments: arguments.to_string(),
                    }),
                    Ok(ResponseEvent::Done),
                ]
            }
            None => vec![
                Ok(ResponseEvent::TextDelta("done".into())),
                Ok(ResponseEvent::Done),
            ],
        };
        Ok(Box::pin(futures::stream::iter(events)))
    }
}

/// The teacher, which also proposes a decomposition when asked for one.
struct Planning {
    teacher: Arc<Steps>,
    plan: String,
    asked: Mutex<usize>,
}

#[async_trait::async_trait]
impl ModelProvider for Planning {
    fn name(&self) -> &str {
        "scripted"
    }
    fn model_name(&self) -> &str {
        "teacher"
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        if format!("{:?}", req.messages).contains("You plan how an agent finishes") {
            *self.asked.lock().unwrap() += 1;
            return Ok(Box::pin(futures::stream::iter(vec![
                Ok(ResponseEvent::TextDelta(self.plan.clone())),
                Ok(ResponseEvent::Done),
            ])));
        }
        self.teacher.complete(req).await
    }
}

fn student_ref() -> ModelRef {
    "remote:scripted/policy".parse().unwrap()
}

const PLAN: &str = r#"{"subgoals": [
  {"goal": "Write the numbers you read to work.txt.", "check": {"kind": "file_exists", "path": "work.txt"}},
  {"goal": "Write the sum to answer.json.", "check": {"kind": "file_contains", "path": "answer.json", "text": "42"}}
]}"#;

struct Fixture {
    scratch: Scratch,
    ctx: Context,
    teacher: Arc<Steps>,
    planner: Arc<Planning>,
    policy: Arc<Policy>,
    runs: Vec<PathBuf>,
    checkout: BenchCheckout,
}

/// The student's failed attempt; a teacher that writes 41 and proposes
/// [`PLAN`]; a policy whose subgoal actions write `work.txt` and then
/// `answer`.
fn fixture(test: &str, answer: &'static str) -> Fixture {
    let scratch = Scratch::new(test);
    let teacher = steps(
        "teacher",
        vec![Step::Call(
            "write_file",
            json!({"path": "answer.json", "text": "41"}),
        )],
    );
    let planner = Arc::new(Planning {
        teacher: Arc::clone(&teacher),
        plan: PLAN.into(),
        asked: Mutex::new(0),
    });
    let policy = Arc::new(Policy {
        actions: vec![
            ("write_file", json!({"path": "work.txt", "text": "6 7"})),
            ("write_file", json!({"path": "answer.json", "text": answer})),
        ],
        seeds: Mutex::new(Vec::new()),
        calls: Mutex::new(0),
    });
    let ctx = Context::new(common::config(&scratch), true)
        .unwrap()
        .with_model(
            teacher_ref(),
            Model::new(
                Arc::clone(&planner) as Arc<dyn ModelProvider>,
                "scripted/teacher",
            ),
        )
        .with_model(
            student_ref(),
            Model::new(
                Arc::clone(&policy) as Arc<dyn ModelProvider>,
                "scripted/policy",
            ),
        );
    let tree = scratch.0.join("student-tree");
    fs::create_dir_all(&tree).unwrap();
    let trajectory = ctx.block_on(student_trajectory(&tree));
    let run = practice_run(&ctx, &scratch.0, &trajectory);
    let checkout = examiner(&scratch.0);
    Fixture {
        scratch,
        ctx,
        teacher,
        planner,
        policy,
        runs: vec![run],
        checkout,
    }
}

fn run_with(f: &Fixture, knowledge: &[PathBuf]) -> RepairReport {
    let teacher = teacher_ref();
    let student = student_ref();
    let mut req = request(&f.runs, &f.checkout, &teacher);
    req.student = Some(&student);
    req.proposer = Proposer::Teacher;
    req.knowledge = knowledge;
    repair(&f.ctx, &req).unwrap()
}

#[test]
fn a_decomposition_passes_where_the_direct_repair_fails_and_records_its_subgoals() {
    let f = fixture("decompose-accepted", "42");
    let report = run_with(&f, &[]);
    // The class's chain: the teacher's correction first, then the
    // decomposition (no knowledge was given).
    let outcomes: BTreeMap<&str, RepairOutcome> = report
        .repairs
        .iter()
        .map(|r| (r.intervention.as_str(), r.outcome))
        .collect();
    assert_eq!(
        outcomes,
        BTreeMap::from([
            ("decompose", RepairOutcome::Accepted),
            ("fork_correction", RepairOutcome::Rejected),
        ]),
        "{:?}",
        report.repairs.iter().map(|r| &r.reason).collect::<Vec<_>>()
    );
    assert_eq!(report.by_intervention["fork_correction"].rejected, 1);
    assert_eq!(report.by_intervention["decompose"].accepted, 1);
    assert_eq!(*f.planner.asked.lock().unwrap(), 1);

    let record = report
        .repairs
        .iter()
        .find(|r| r.intervention == "decompose")
        .unwrap();
    assert_eq!(record.teacher, "scripted/policy", "the student worked it");
    let decomposition = record.notes.decomposition.as_ref().unwrap();
    assert_eq!(decomposition.proposer, "scripted/teacher");
    assert_eq!(decomposition.proposed_by, Proposer::Teacher);
    assert_eq!(decomposition.subgoals.len(), 2);
    let states: Vec<SubgoalState> = decomposition.outcomes.iter().map(|o| o.state).collect();
    assert_eq!(states, [SubgoalState::Reached, SubgoalState::Reached]);

    // The guided trajectory: the student's kept prefix, then the subgoal
    // turns as guidance and the student's own steps, supervised.
    let repaired = f
        .ctx
        .experiences()
        .get(record.experience.as_ref().unwrap())
        .unwrap();
    let prefix = record.prefix_steps.unwrap();
    let suffix = &repaired.trajectory.steps[prefix..];
    let role = |s: &atif::TraceStep| {
        s.extra.as_ref().unwrap()["correction"]["role"]
            .as_str()
            .unwrap()
            .to_string()
    };
    let guidance: Vec<&atif::TraceStep> = suffix
        .iter()
        .filter(|s| s.source == atif::StepOrigin::User)
        .collect();
    assert_eq!(guidance.len(), 2);
    assert!(guidance.iter().all(|s| role(s) == "guidance"));
    assert!(guidance[1]
        .message
        .as_text()
        .unwrap()
        .contains("`answer.json` contains `42`"));
    let own: Vec<&atif::TraceStep> = suffix
        .iter()
        .filter(|s| s.source == atif::StepOrigin::Agent)
        .collect();
    assert!(!own.is_empty());
    assert!(own
        .iter()
        .all(|s| role(s) == "policy" && s.is_copied_context.is_none()));
    let lineage = &repaired.trajectory.extra.as_ref().unwrap()["correction"];
    assert_eq!(lineage["intervention"], json!("decompose"));
    assert_eq!(
        lineage["decomposition"]["proposer"],
        json!("scripted/teacher")
    );
    assert_eq!(
        lineage["decomposition"]["outcomes"][1]["state"],
        json!("reached")
    );
    assert!(repaired.provenance.teacher, "guided by the teacher's plan");

    // As training records: the student's own steps are supervised, under the
    // subgoal turns; the kept prefix and the guidance never are.
    let built = build(&f.ctx, &sft_request(report.repair_set.clone().unwrap())).unwrap();
    let text = fs::read_to_string(&built.path).unwrap();
    let records: Vec<serde_json::Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert!(!records.is_empty(), "{:?}", built);
    let mut trained = Vec::new();
    for record in &records {
        for message in record["messages"].as_array().unwrap() {
            if message["train"] == true {
                assert_eq!(message["role"], json!("assistant"), "{message}");
                trained.push(message.to_string());
            }
        }
    }
    assert_eq!(trained.len(), 3, "{trained:?}");
    assert!(trained.iter().any(|m| m.contains("work.txt")));
    assert!(
        trained.iter().all(|m| !m.contains("input.csv")),
        "the kept read"
    );

    // A rerun holds every graded attempt: nothing is asked again.
    let log = fs::read_to_string(f.scratch.0.join("bench.log")).unwrap();
    let again = run_with(&f, &[]);
    assert!(again.repair_set.is_none());
    assert_eq!(again.by_class["wrong_content"].held, 1);
    assert_eq!(*f.planner.asked.lock().unwrap(), 1);
    assert_eq!(
        fs::read_to_string(f.scratch.0.join("bench.log")).unwrap(),
        log
    );
}

#[test]
fn an_unverified_decomposition_is_rejected_and_never_becomes_experience() {
    let f = fixture("decompose-rejected", "41");
    let before = f.ctx.experiences().list().unwrap().len();
    let report = run_with(&f, &[]);
    assert!(report
        .repairs
        .iter()
        .all(|r| r.outcome == RepairOutcome::Rejected));
    let record = report
        .repairs
        .iter()
        .find(|r| r.intervention == "decompose")
        .unwrap();
    let states: Vec<SubgoalState> = record
        .notes
        .decomposition
        .as_ref()
        .unwrap()
        .outcomes
        .iter()
        .map(|o| o.state)
        .collect();
    assert_eq!(states, [SubgoalState::Reached, SubgoalState::Missed]);
    assert_eq!(record.grade.as_ref().unwrap().verdict, "fail");
    assert!(report.repair_set.is_none());
    assert_eq!(f.ctx.experiences().list().unwrap().len(), before);
}

#[test]
fn a_retry_samples_the_students_model_under_a_seed_of_its_own() {
    let f = fixture("retry", "42");
    let report_file = f.runs[0].join("report.json");
    let mut report: serde_json::Value =
        serde_json::from_slice(&fs::read(&report_file).unwrap()).unwrap();
    report["sampling"] = json!({"temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": 7});
    fs::write(&report_file, report.to_string()).unwrap();
    let policy_model = f.ctx.model(&student_ref()).unwrap();
    let retry = Retry {
        checkout: &f.checkout,
        student: policy_model,
        deadline: DEFAULT_REPAIR_DEADLINE,
        cancel: CancelToken::new(),
    };
    let interventions: BTreeMap<InterventionKind, &dyn Intervention> =
        BTreeMap::from([(InterventionKind::Retry, &retry as &dyn Intervention)]);
    let routing = Routing::new(
        BTreeMap::from([(FailureClass::WrongContent, vec![InterventionKind::Retry])]),
        2,
    )
    .unwrap();
    let teacher = teacher_ref();
    let mut req = request(&f.runs, &f.checkout, &teacher);
    req.samples = 2;
    let report = repair_with(&f.ctx, &req, &routing, &interventions).unwrap();
    let [record] = report.repairs.as_slice() else {
        panic!(
            "the first sample passes and ends the chain: {:?}",
            report.repairs
        )
    };
    assert_eq!(
        record.outcome,
        RepairOutcome::Accepted,
        "{:?}",
        record.reason
    );
    assert_eq!(record.intervention, "retry");
    let sampling = record.notes.sampling.as_ref().unwrap();
    assert_eq!(sampling["from"], json!("practice_run"));
    assert_eq!(sampling["temperature"], json!(1.0));
    let seed = sampling["seed"].as_u64().unwrap();
    assert_ne!(seed, 7, "not the practice run's seed");
    assert!(f
        .policy
        .seeds
        .lock()
        .unwrap()
        .iter()
        .all(|s| *s == Some(seed)));
    let repaired = f
        .ctx
        .experiences()
        .get(record.experience.as_ref().unwrap())
        .unwrap();
    assert!(!repaired.provenance.teacher, "the policy's own success");
    assert!(f.teacher.seen.lock().unwrap().is_empty(), "no teacher");
}

#[test]
fn a_knowledge_assisted_correction_records_its_retrieval_and_never_reads_sealed_material() {
    let f = fixture("knowledge", "42");
    let docs = f.scratch.0.join("docs");
    fs::create_dir_all(docs.join("tests")).unwrap();
    fs::write(
        docs.join("sums.md"),
        "# Sums\n\nTo add the numbers in input.csv, add every value of its second column.\n",
    )
    .unwrap();
    fs::write(
        docs.join("tests").join("truth.json"),
        "{\"answer\": \"the hidden numbers sum to 42\"}",
    )
    .unwrap();
    // The teacher's plain correction is rejected; the chain goes on to the
    // knowledge-assisted one, which is rejected too (the teacher writes 41
    // whatever it reads): what is asserted is what it was shown.
    let report = run_with(&f, std::slice::from_ref(&docs));
    let record = report
        .repairs
        .iter()
        .find(|r| r.intervention == "knowledge_correction")
        .unwrap_or_else(|| panic!("{:?}", report.repairs));
    let fork = report
        .repairs
        .iter()
        .find(|r| r.intervention == "fork_correction")
        .unwrap();
    assert_ne!(record.key, fork.key, "its own repair of the same failure");
    assert_eq!(record.teacher, fork.teacher, "by the same teacher");
    let used = record.notes.knowledge.as_ref().unwrap();
    assert_eq!(used.sources.len(), 1);
    assert!(used.sources[0].path.ends_with("sums.md"));
    assert!(used.withheld.iter().any(|p| p.ends_with("tests")));
    assert!(used.query.contains("input.csv"));
    assert!(!used.passages.is_empty());
    assert!(used.passages.iter().all(|p| p.source == used.sources[0].id));
    assert!(used.snapshot.starts_with("blake3:"), "{}", used.snapshot);

    let shown = f.teacher.seen.lock().unwrap().join("\n");
    assert!(
        shown.contains("second column"),
        "the teacher read the passage"
    );
    assert!(!shown.contains("hidden numbers"), "never the sealed truth");

    // Naming sealed material is refused outright.
    for sealed in [docs.join("tests"), docs.join("tests").join("truth.json")] {
        assert!(KnowledgeBase::load(&f.ctx, &[sealed]).is_err());
    }
}
