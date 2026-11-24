// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model evaluations whose every verdict is
// produced by code that names itself, for its clients. If your team needs
// expertise in rigorous answer verification, you can procure our services by
// sending an email to info@swedishembedded.com.

//! `answer.collect`: a closed-book task (one `task.generate` admitted from a
//! real document) answered in process, through the one solve, and graded by
//! the verifiers its kind names: the same verdicts, under the same producer,
//! as wherever an answer is graded. A kind graded against its evidence is
//! graded against the document the task was generated from, which the task's
//! item carries.

use std::sync::{Arc, Mutex};

use serde_json::Value;
use splinter_agent::solve::Model;
use splinter_core::factory::execution::ModelBinding;
use splinter_core::factory::items::{ItemIndex, ITEMS_FILE};
use splinter_factory::execute::Models;
use splinter_factory::resolve::resolve;
use splinter_factory::run::{Factory, Services};
use sven_sdk::model::{CompletionRequest, ModelProvider, ResponseEvent, ResponseStream};

use crate::fakes::{FakeBackend, Outcome};
use crate::repo::{connections, Repo};

/// A model that gives the answer it is set to.
struct Answering(Mutex<String>);

#[async_trait::async_trait]
impl ModelProvider for Answering {
    fn name(&self) -> &str {
        "scripted"
    }
    fn model_name(&self) -> &str {
        "student-1"
    }
    async fn complete(&self, _: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let reply = self.0.lock().unwrap().clone();
        Ok(Box::pin(futures::stream::iter(vec![
            Ok(ResponseEvent::TextDelta(reply)),
            Ok(ResponseEvent::Done),
        ])))
    }
}

struct Student(Arc<Answering>);

impl Models for Student {
    fn model(&self, _: &ModelBinding) -> Result<Model, String> {
        Ok(Model::new(self.0.clone(), "scripted/student-1"))
    }
}

/// A teacher that proposes `tasks` and a student that answers `answer`,
/// told apart by the model each role is bound to.
struct Classroom {
    teacher: Arc<Answering>,
    student: Arc<Answering>,
}

impl Models for Classroom {
    fn model(&self, binding: &ModelBinding) -> Result<Model, String> {
        let provider = if binding.model == "org/teacher" {
            &self.teacher
        } else {
            &self.student
        };
        Ok(Model::new(provider.clone(), "scripted/model-1"))
    }
}

const RECIPE: &str = "schema: splinter.recipe/v2
name: recall
inputs:
  tasks: { kind: files, path: closed }
models:
  student: \"hosted:org/student\"
stages:
  - id: answered
    uses: answer.collect
    foreach: $inputs.tasks
    inputs: { task: $item }
    with: { model: student }
";

/// The stated verdict of the one answer to the fixture task, as `answer`.
fn verdict(answer: &str) -> String {
    let repo = Repo::new();
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/it/fixtures/closed-book/recall.json");
    repo.write(
        "closed/recall/task.json",
        &std::fs::read_to_string(fixture).unwrap(),
    );
    let resolved = resolve(&repo.root, RECIPE, connections()).unwrap();
    let backend = FakeBackend::with(Outcome::Pass);
    let student = Student(Arc::new(Answering(Mutex::new(answer.into()))));
    let services = Services {
        backend: &backend,
        trainer: None,
        models: Some(&student),
        variables: None,
        online: None,
    };
    let dir = tempfile::tempdir().unwrap();
    Factory::open(&resolved, dir.path(), services, "test")
        .unwrap()
        .run()
        .unwrap();
    let verdicts: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("answered/recall/verdicts-0.json")).unwrap(),
    )
    .unwrap();
    assert!(dir
        .path()
        .join("answered/recall/experience-0.json")
        .is_file());
    let stated = verdicts
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["producer"]["name"] == "splinter-lab/reference-stated:lenient")
        .expect("the kind's stated verdict");
    stated["body"]["outcome"].as_str().unwrap().to_owned()
}

#[test]
fn a_closed_book_answer_is_graded_by_its_kinds_verifiers() {
    assert_eq!(
        verdict("A fork correction continues it instead of starting the task afresh."),
        "pass"
    );
    assert_eq!(verdict("It deletes the attempt and moves on."), "fail");
}

const LETTER: &str = "# To a young man

## Business

I have received your favour of the tenth and enclose the bill of lading for the hogsheads of tobacco shipped on the brig Eliza, which should reach Havre within the month.

## Study

I advise you to fix a habit of study every morning before you do anything else. Never let a day pass without reading something of history or ethics, and always write down what you have read, for the memory fails what the pen has not fixed.
";

const ADVICE: &str = "I advise you to fix a habit of study every morning before you do anything else. Never let a day pass without reading something of history or ethics, and always write down what you have read, for the memory fails what the pen has not fixed.";

const ADVISED: &str = "schema: splinter.recipe/v2
name: advised
inputs:
  letters: { kind: files, path: letters }
models:
  teacher: \"hosted:org/teacher\"
  student: \"hosted:org/student\"
stages:
  - id: generated
    uses: task.generate
    inputs: { documents: $inputs.letters }
    with: { model: teacher, kinds: [advise] }
  - id: answered
    uses: answer.collect
    foreach: $stages.generated.tasks
    inputs: { task: $item }
    with: { model: student }
";

/// The recipe `kinds` of tasks are generated from the letters by, and
/// answered.
fn classroom(kinds: &str) -> String {
    ADVISED.replace("[advise]", kinds)
}

/// The verdict `producer` gave the one answer `answer` to the one task the
/// teacher's `proposal` makes of `document`, by the recipe `recipe`.
fn graded(recipe: &str, document: &str, proposal: &Value, answer: &str, producer: &str) -> String {
    let repo = Repo::new();
    repo.write("letters/letter.md", document);
    let resolved = resolve(&repo.root, recipe, connections()).unwrap();
    let backend = FakeBackend::with(Outcome::Pass);
    let classroom = Classroom {
        teacher: Arc::new(Answering(Mutex::new(proposal.to_string()))),
        student: Arc::new(Answering(Mutex::new(answer.into()))),
    };
    let services = Services {
        backend: &backend,
        trainer: None,
        models: Some(&classroom),
        variables: None,
        online: None,
    };
    let dir = tempfile::tempdir().unwrap();
    Factory::open(&resolved, dir.path(), services, "test")
        .unwrap()
        .run()
        .unwrap();
    let answered = dir.path().join("answered");
    let item = std::fs::read_dir(&answered)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.join("verdicts-0.json").is_file())
        .unwrap_or_else(|| panic!("no task was answered in {}", answered.display()));
    let verdicts: Value =
        serde_json::from_str(&std::fs::read_to_string(item.join("verdicts-0.json")).unwrap())
            .unwrap();
    let verdict = verdicts
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["producer"]["name"] == producer)
        .unwrap_or_else(|| panic!("no verdict of {producer} in {verdicts:#}"));
    verdict["body"]["outcome"].as_str().unwrap().to_owned()
}

/// The quotation verdict of the one answer `answer` to the advice task
/// generated from the letter.
fn quoted(answer: &str) -> String {
    let proposal = serde_json::json!({ "tasks": [{
        "instruction": "I am nineteen and have finished my first year at college, but I find my mornings slip away in idleness; what would you advise me to do about my habits?",
        "reference": ADVICE,
        "evidence": [{ "section": 2, "quote": ADVICE }],
    }]});
    graded(ADVISED, LETTER, &proposal, answer, "splinter-lab/quotation")
}

#[test]
fn advice_is_graded_against_the_letter_it_was_generated_from() {
    // A quotation is held to the letter: one it holds passes, one it does
    // not hold fails, however much of the advice the answer gives.
    assert_eq!(quoted(&format!("As I wrote to you: \"{ADVICE}\"")), "pass");
    assert_eq!(
        quoted(&format!(
            "{ADVICE} As I wrote: \"sleep late every morning and read nothing at all of history\""
        )),
        "fail"
    );
}

const CLAMP_SPEC: &str = "# Numeric helpers

## clamp

clamp(x, lo, hi) returns lo when x is below lo, hi when x is above hi, and x otherwise.
";

const CLAMP: &str = "def clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
";

/// The verdict of the code checks on the one answer `answer` to a task to
/// fix a faulty clamp.
fn checked(answer: &str) -> String {
    let proposal = serde_json::json!({ "tasks": [{
        "instruction": "This Python clamp(x, lo, hi) returns x when x is above hi. Fix it.",
        "subject": "clamp",
        "reference": CLAMP,
        "material": "def clamp(x, lo, hi):\n    if x < lo:\n        return lo\n    return x\n",
        "evidence": [{ "section": 1, "quote": null }],
        "checks": [{ "code": "assert clamp(7, 0, 3) == 3\nassert clamp(-5, 0, 3) == 0\n", "exit_code": 0 }],
    }]});
    graded(
        &classroom("[debug]"),
        CLAMP_SPEC,
        &proposal,
        answer,
        "splinter-lab/executable",
    )
}

#[test]
fn fixed_code_is_graded_by_running_its_checks_in_its_runtime() {
    assert_eq!(checked(CLAMP), "pass");
    assert_eq!(
        checked("def clamp(x, lo, hi):\n    return max(x, lo)\n"),
        "fail"
    );
}

/// The in-process answers are experiences like any attempt's: projected by
/// the same views, a passed answer's task gives a retrieval record of its
/// question and the passage of the letter it rests on.
#[test]
fn answered_tasks_give_a_retrieval_dataset_of_the_passages_they_rest_on() {
    // seed: a dataset stage states its seed; one record has no split to draw.
    let recipe = ADVISED.to_owned()
        + "  - id: retrieval
    uses: dataset.project
    inputs: { answers: [$stages.answered.answers] }
    with: { objective: retrieval, target: student, seed: 13 }
";
    let repo = Repo::new();
    repo.write("letters/letter.md", LETTER);
    let question = "I am nineteen and have finished my first year at college, but I find my mornings slip away in idleness; what would you advise me to do about my habits?";
    let proposal = serde_json::json!({ "tasks": [{
        "instruction": question,
        "reference": ADVICE,
        "evidence": [{ "section": 2, "quote": ADVICE }],
    }]});
    let resolved = resolve(&repo.root, &recipe, connections()).unwrap();
    let backend = FakeBackend::with(Outcome::Pass);
    let classroom = Classroom {
        teacher: Arc::new(Answering(Mutex::new(proposal.to_string()))),
        student: Arc::new(Answering(Mutex::new(format!(
            "As I wrote to you: \"{ADVICE}\""
        )))),
    };
    let services = Services {
        backend: &backend,
        trainer: None,
        models: Some(&classroom),
        variables: None,
        online: None,
    };
    let dir = tempfile::tempdir().unwrap();
    Factory::open(&resolved, dir.path(), services, "test")
        .unwrap()
        .run()
        .unwrap();
    let text = std::fs::read_to_string(dir.path().join("retrieval/train.jsonl")).unwrap();
    let records: Vec<Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(records.len(), 1, "{text}");
    let record = records[0].to_string();
    assert!(record.contains("mornings slip away"), "{record}");
    assert!(
        record.contains("fix a habit of study every morning"),
        "{record}"
    );
}

/// Learning from documents: a passed answer to a generated task is an SFT
/// record of the question and the answer, supervised on the answer alone.
#[test]
fn passed_answers_give_an_sft_dataset_of_the_question_and_the_answer() {
    // seed: a dataset stage states its seed; one record has no split to draw.
    let recipe = ADVISED.to_owned()
        + "  - id: sft
    uses: dataset.project
    inputs: { answers: [$stages.answered.answers] }
    with: { target: student, seed: 13 }
";
    let repo = Repo::new();
    repo.write("letters/letter.md", LETTER);
    let question = "I am nineteen and have finished my first year at college, but I find my mornings slip away in idleness; what would you advise me to do about my habits?";
    let proposal = serde_json::json!({ "tasks": [{
        "instruction": question,
        "reference": ADVICE,
        "evidence": [{ "section": 2, "quote": ADVICE }],
    }]});
    let answer = format!("As I wrote to you: \"{ADVICE}\"");
    let resolved = resolve(&repo.root, &recipe, connections()).unwrap();
    let backend = FakeBackend::with(Outcome::Pass);
    let classroom = Classroom {
        teacher: Arc::new(Answering(Mutex::new(proposal.to_string()))),
        student: Arc::new(Answering(Mutex::new(answer.clone()))),
    };
    let services = Services {
        backend: &backend,
        trainer: None,
        models: Some(&classroom),
        variables: None,
        online: None,
    };
    let dir = tempfile::tempdir().unwrap();
    Factory::open(&resolved, dir.path(), services, "test")
        .unwrap()
        .run()
        .unwrap();
    let text = std::fs::read_to_string(dir.path().join("sft/train.jsonl")).unwrap();
    let records: Vec<Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(records.len(), 1, "{text}");
    let messages = records[0]["messages"].as_array().unwrap();
    let supervised: Vec<&str> = messages
        .iter()
        .filter(|m| m["train"] == true)
        .map(|m| m["content"].as_str().unwrap())
        .collect();
    assert_eq!(supervised, [answer.as_str()]);
    assert!(messages.iter().any(|m| m["content"]
        .as_str()
        .unwrap()
        .contains("mornings slip away")));
}

/// Scenario A: a document to a trained SFT candidate. Tasks generated from
/// the letter and admitted by code, answered in process and graded by their
/// kind's verifiers, the passed answers an SFT dataset, a candidate trained
/// on it by the trainer's SFT job. No experience database takes part: the
/// factory reads its corpus from the run's own products.
#[test]
fn scenario_a_a_document_becomes_a_trained_sft_candidate() {
    // seed: a dataset and a training stage state their seeds; nothing here draws.
    let recipe = ADVISED.to_owned()
        + "  - id: sft
    uses: dataset.project
    inputs: { answers: [$stages.answered.answers] }
    with: { target: student, seed: 13 }
  - id: lora
    uses: model.train
    inputs: { dataset: $stages.sft.dataset }
    with: { base: student, method: lora, params: { rank: 8 }, seed: 1 }
";
    let recipe = recipe.replace(
        "student: \"hosted:org/student\"",
        "student: \"brain:org/student\"",
    );
    let repo = Repo::new();
    repo.write("letters/letter.md", LETTER);
    let proposal = serde_json::json!({ "tasks": [{
        "instruction": "I am nineteen and have finished my first year at college, but I find my mornings slip away in idleness; what would you advise me to do about my habits?",
        "reference": ADVICE,
        "evidence": [{ "section": 2, "quote": ADVICE }],
    }]});
    let resolved = resolve(&repo.root, &recipe, connections()).unwrap();
    let backend = FakeBackend::with(Outcome::Pass);
    let classroom = Classroom {
        teacher: Arc::new(Answering(Mutex::new(proposal.to_string()))),
        student: Arc::new(Answering(Mutex::new(format!(
            "As I wrote to you: \"{ADVICE}\""
        )))),
    };
    let trainer = crate::fakes::FakeTrainer::default();
    let services = Services {
        backend: &backend,
        trainer: Some(&trainer),
        models: Some(&classroom),
        variables: None,
        online: None,
    };
    let dir = tempfile::tempdir().unwrap();
    Factory::open(&resolved, dir.path(), services, "test")
        .unwrap()
        .run()
        .unwrap();
    assert_eq!(*trainer.objectives.lock().unwrap(), ["sft"]);
    assert!(dir.path().join("lora").is_dir());
    assert!(
        backend.executed().is_empty(),
        "nothing ran outside the process"
    );
}

/// What the answered stage made of the one task the teacher's `proposal`
/// makes of `document` under `kinds`: the attempts it kept and the reasons
/// it rejected tasks for.
fn answered(kinds: &str, proposal: &Value) -> (usize, Vec<String>) {
    let repo = Repo::new();
    repo.write("letters/letter.md", LETTER);
    let resolved = resolve(&repo.root, &classroom(kinds), connections()).unwrap();
    let backend = FakeBackend::with(Outcome::Pass);
    let classroom = Classroom {
        teacher: Arc::new(Answering(Mutex::new(proposal.to_string()))),
        student: Arc::new(Answering(Mutex::new("Any answer at all.".into()))),
    };
    let services = Services {
        backend: &backend,
        trainer: None,
        models: Some(&classroom),
        variables: None,
        online: None,
    };
    let dir = tempfile::tempdir().unwrap();
    Factory::open(&resolved, dir.path(), services, "test")
        .unwrap()
        .run()
        .unwrap();
    let index: ItemIndex = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("answered").join(ITEMS_FILE)).unwrap(),
    )
    .unwrap();
    (
        index.items.len(),
        index.rejected.into_iter().map(|r| r.reason).collect(),
    )
}

#[test]
fn a_task_only_a_judge_grades_is_refused_where_no_judge_is_given() {
    // A counterexample is graded by a judge alone; answer.collect is given
    // none. Answered anyway, every attempt would be kept with no verdict, as
    // if it had been graded: it is refused, naming what it needs.
    let proposal = serde_json::json!({ "tasks": [{
        "instruction": "Someone claims that the brig Eliza carried no tobacco to Havre that month. Give a counterexample.",
        "subject": "brig Eliza",
        "reference": "The hogsheads of tobacco shipped on the brig Eliza, which should reach Havre within the month.",
        "evidence": [{ "section": 1, "quote": "the bill of lading for the hogsheads of tobacco shipped on the brig Eliza, which should reach Havre within the month" }],
    }]});
    let (kept, rejected) = answered("[counterexample]", &proposal);
    assert_eq!(kept, 0, "an attempt was kept with nothing to grade it");
    assert_eq!(rejected.len(), 1, "{rejected:?}");
    assert!(rejected[0].contains("judge"), "{rejected:?}");
}
