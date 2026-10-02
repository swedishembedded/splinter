// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: a judge model grades an answer closed-book through sven, refuses
//! to grade its own work, is calibrated against labelled answers, and when
//! calibrated abstains where its measured precision is too low. Run beside
//! stronger verifiers, every verdict is annotated and the store's decision
//! rule lets the strongest decide.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use splinter_agent::judge::JudgeVerifier;
use splinter_agent::solve::Model;
use splinter_lab::verifiers::calibration::{calibrate, CalibratedJudge};
use splinter_lab::verifiers::executable::{ExecutableCheck, ExecutableVerifier, Expectation};
use splinter_lab::verifiers::{annotation, verify_and_annotate, Strongest, Verifier, VerifyError};
use splinter_record::annotation::{decide, AnnotationBody, Outcome, Strength};
use splinter_record::clock::FixedClock;
use splinter_record::experience::{
    Environment, Experience, Privileged, PrivilegedKind, Provenance, Task,
};
use splinter_record::experiences::ExperienceStore;
use splinter_record::StateRoot;
use splinter_sandbox::{
    Limits, ProcessSandbox, ResolvedEnvironment, RuntimeEnvironment, RuntimeRegistry,
};
use sven_sdk::atif::{AgentProfile, Trajectory};
use sven_sdk::model::{
    CompletionRequest, MessageContent, ModelProvider, ResponseEvent, ResponseStream,
};

const JUDGE: &str = "scripted/judge-1";
const SOLVER: &str = "scripted/solver-1";

/// A judge whose reply is a function of the last user message, keeping
/// every prompt it was sent.
struct ScriptedJudge {
    reply: fn(&str) -> &'static str,
    prompts: Mutex<Vec<String>>,
}

impl ScriptedJudge {
    fn new(reply: fn(&str) -> &'static str) -> Arc<Self> {
        Arc::new(Self {
            reply,
            prompts: Mutex::new(Vec::new()),
        })
    }
}

#[async_trait::async_trait]
impl ModelProvider for ScriptedJudge {
    fn name(&self) -> &str {
        "scripted"
    }
    fn model_name(&self) -> &str {
        "judge-1"
    }
    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let prompt = req
            .messages
            .iter()
            .rev()
            .find_map(|m| match &m.content {
                MessageContent::Text(t) => Some(t.clone()),
                _ => None,
            })
            .unwrap_or_default();
        let reply = (self.reply)(&prompt);
        self.prompts.lock().unwrap().push(prompt);
        Ok(Box::pin(futures::stream::iter(vec![
            Ok(ResponseEvent::TextDelta(reply.into())),
            Ok(ResponseEvent::Done),
        ])))
    }
}

/// PASS when the graded answer says 5, ABSTAIN when it is a question,
/// FAIL otherwise.
fn grades_five(prompt: &str) -> &'static str {
    let answer = prompt.rsplit("ANSWER:").next().unwrap_or_default();
    if answer.contains('5') {
        "PASS\nit matches the reference"
    } else if answer.contains('?') {
        "ABSTAIN\ncannot tell"
    } else {
        "FAIL\nit does not match the reference"
    }
}

fn always_fails(_: &str) -> &'static str {
    "FAIL\nI disagree"
}

fn sum_task(environment: Environment, privileged: Vec<Privileged>) -> Task {
    Task::new("sum", vec![], environment, "What is 2 + 3?", privileged).unwrap()
}

fn reference() -> Privileged {
    Privileged {
        kind: PrivilegedKind::Reference,
        content: "REFERENCE-5".into(),
        span: None,
    }
}

fn experience(task: &Task, output: &str, solver: &str) -> Experience {
    let profile = AgentProfile {
        name: "t".into(),
        version: "1".into(),
        model_name: None,
        tool_definitions: None,
        extra: None,
    };
    Experience::new(
        task.clone(),
        Trajectory::new("ATIF-v1.7", profile),
        Some(output.into()),
        Provenance::new(solver, &FixedClock::new("2026-09-30T00:00:00.000Z")),
    )
    .unwrap()
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Runtime::new().unwrap()
}

fn judge(model: Arc<ScriptedJudge>, rt: &tokio::runtime::Runtime) -> JudgeVerifier {
    JudgeVerifier::new(
        Model::new(model, JUDGE),
        rt.handle().clone(),
        Duration::from_secs(60),
    )
}

fn outcome(verifier: &dyn Verifier, task: &Task, exp: &Experience) -> (Outcome, serde_json::Value) {
    match annotation(verifier, task, exp).unwrap().body {
        AnnotationBody::Verdict {
            outcome, evidence, ..
        } => (outcome, evidence),
        other => panic!("not a verdict: {other:?}"),
    }
}

#[test]
fn the_judge_grades_closed_book_from_the_output_and_the_reference() {
    let rt = runtime();
    let model = ScriptedJudge::new(grades_five);
    let judge = judge(model.clone(), &rt);
    assert_eq!(judge.strength(), Strength::Judged);
    let task = sum_task(Environment::closed_book(), vec![reference()]);

    let (verdict, evidence) = outcome(&judge, &task, &experience(&task, "5", SOLVER));
    assert_eq!(verdict, Outcome::Pass, "{evidence}");
    assert_eq!(evidence["judge"], JUDGE);
    assert_eq!(evidence["reason"], "it matches the reference");
    let (verdict, _) = outcome(&judge, &task, &experience(&task, "4", SOLVER));
    assert_eq!(verdict, Outcome::Fail);

    let prompts = model.prompts.lock().unwrap();
    assert!(
        prompts[0].contains("REFERENCE-5"),
        "the judge sees the reference"
    );
    assert!(prompts[0].contains("What is 2 + 3?"), "and the task");
}

#[test]
fn the_judge_called_inside_an_async_task_refuses_instead_of_panicking() {
    let rt = runtime();
    let judge = judge(ScriptedJudge::new(grades_five), &rt);
    let task = sum_task(Environment::closed_book(), vec![reference()]);
    let exp = experience(&task, "5", SOLVER);
    let result = rt.block_on(async { judge.verify(&task, &exp) });
    assert!(
        matches!(result, Err(VerifyError::InsideAsync { .. })),
        "{result:?}"
    );
}

#[test]
fn the_judge_refuses_to_grade_its_own_model() {
    let rt = runtime();
    let judge = judge(ScriptedJudge::new(grades_five), &rt);
    let task = sum_task(Environment::closed_book(), vec![reference()]);
    let own = experience(&task, "5", JUDGE);
    assert!(matches!(
        annotation(&judge, &task, &own),
        Err(VerifyError::SelfJudging { .. })
    ));
}

#[test]
fn calibration_measures_the_judge_and_the_calibrated_judge_abstains_below_threshold() {
    let rt = runtime();
    let judge = judge(ScriptedJudge::new(grades_five), &rt);
    let task = sum_task(Environment::closed_book(), vec![reference()]);
    // The judge answers pass, pass, pass, fail, abstain.
    let labelled: Vec<(Task, Experience, Outcome)> = [
        ("5", Outcome::Pass),
        ("five: 5", Outcome::Pass),
        ("15", Outcome::Fail),
        ("4", Outcome::Fail),
        ("?", Outcome::Pass),
    ]
    .into_iter()
    .map(|(answer, label)| (task.clone(), experience(&task, answer, SOLVER), label))
    .collect();
    let calibration = calibrate(&judge, &labelled).unwrap();
    assert_eq!(calibration.n, 5);
    assert_eq!(calibration.precision_pass, Some(2.0 / 3.0));
    assert_eq!(calibration.precision_fail, Some(1.0));
    assert_eq!(calibration.abstain_rate, Some(0.2));

    let strict = CalibratedJudge::new(judge, calibration.clone(), 0.9).unwrap();
    let (verdict, evidence) = outcome(&strict, &task, &experience(&task, "5", SOLVER));
    assert_eq!(verdict, Outcome::Abstain, "pass precision is below 0.9");
    assert_eq!(evidence["calibration"], calibration.id.as_str());
    let (verdict, _) = outcome(&strict, &task, &experience(&task, "4", SOLVER));
    assert_eq!(verdict, Outcome::Fail, "fail precision clears it");

    let judge = self::judge(ScriptedJudge::new(grades_five), &rt);
    let lenient = CalibratedJudge::new(judge, calibration, 0.5).unwrap();
    let (verdict, _) = outcome(&lenient, &task, &experience(&task, "5", SOLVER));
    assert_eq!(verdict, Outcome::Pass);
}

struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn an_executable_pass_outranks_a_judged_fail_through_the_store() {
    let scratch =
        Scratch(std::env::temp_dir().join(format!("splinter-judge-{}", std::process::id())));
    let _ = std::fs::remove_dir_all(&scratch.0);
    std::fs::create_dir_all(scratch.0.join("sandbox")).unwrap();
    let sandbox = ProcessSandbox::new(
        scratch.0.join("sandbox"),
        BTreeMap::new(),
        &[],
        Limits::default(),
    );
    let env =
        RuntimeEnvironment::new(&RuntimeRegistry::builtin(), "python3", Arc::new(sandbox)).unwrap();
    let check = ExecutableCheck {
        code: "print(answer())\n".into(),
        stdin: None,
        expect: Expectation::exit_code(0),
        environment: None,
    };
    let task = sum_task(
        ResolvedEnvironment::Runtime(env.clone()).record().unwrap(),
        vec![check.as_check().unwrap()],
    );
    let exp = experience(&task, "def answer():\n    return 2 + 3\n", SOLVER);
    let store = ExperienceStore::open(&StateRoot::new(scratch.0.join("state")));
    let id = store.put(&exp).unwrap();

    let rt = runtime();
    let verifiers = Strongest::new(vec![
        Box::new(ExecutableVerifier::new(vec![env])),
        Box::new(judge(ScriptedJudge::new(always_fails), &rt)),
    ]);
    let report = verify_and_annotate(&store, &verifiers, &task, &exp).unwrap();
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    assert_eq!(report.annotations.len(), 2);

    let log = store.annotations(&id).unwrap();
    assert_eq!(log.annotations.len(), 2);
    let decision = decide(&log.annotations).unwrap();
    assert!(decision.passed);
    assert_eq!(decision.strength, Strength::Executable);
}
