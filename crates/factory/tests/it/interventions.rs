// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the loop from a failed agent attempt to a
// verified corrected one, for its clients. If your team needs expertise in
// turning agent failures into training data you can procure our services by
// sending an email to info@swedishembedded.com.

//! A briefed correction: the tasks the student left unsolved
//! (`attempts.failures`), a teacher's brief for each (an `agent` stage), and
//! the student again on those tasks with its brief beside the instruction
//! (`hint: brief`). What it then solves is learned under the task's own
//! instruction; a trial that repeats its brief is excluded, as a hinted
//! trial that quotes its solution is. Knowledge-backed correction and
//! decomposition are this, with the teacher asked for passages or a plan.

use std::path::Path;
use std::sync::{Arc, Mutex};

use splinter_agent::solve::Model;
use splinter_core::factory::execution::ModelBinding;
use splinter_core::factory::items::{ItemIndex, ITEMS_FILE};
use splinter_factory::execute::Models;
use splinter_factory::resolve::{resolve, Resolved};
use splinter_factory::run::dataset::dataset_of;
use splinter_factory::run::{Factory, RunError, Services};
use sven_sdk::model::{CompletionRequest, ModelProvider, ResponseEvent, ResponseStream};

use crate::fakes::{FakeBackend, FakeTrainer, Outcome};
use crate::repo::{connections, Repo};

/// A teacher that briefs each task by what its request is about; the brief
/// for the second task says a line the fixture trial's own reply says.
struct Teacher;

#[async_trait::async_trait]
impl ModelProvider for Teacher {
    fn name(&self) -> &str {
        "scripted"
    }
    fn model_name(&self) -> &str {
        "teacher-1"
    }
    async fn complete(&self, request: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let text = serde_json::to_string(&request.messages).unwrap_or_default();
        let brief = if text.contains("train/002-second") {
            "Use the A command, which\nenters Insert mode at the end of the current line"
        } else {
            "Remember to save the file with :wq before you finish the job"
        };
        let reply = serde_json::json!({ "brief": brief }).to_string();
        Ok(Box::pin(futures::stream::iter(vec![
            Ok(ResponseEvent::TextDelta(reply)),
            Ok(ResponseEvent::Done),
        ])))
    }
}

struct Teachers;

impl Models for Teachers {
    fn model(&self, _: &ModelBinding) -> Result<Model, String> {
        Ok(Model::new(Arc::new(Teacher), "scripted/teacher-1"))
    }
}

// seed: a recipe must state every seed; the fake backend's outcomes are scripted, so none draws.
const RECIPE: &str = "schema: splinter.recipe/v2
name: briefed
inputs:
  course: { kind: files, path: collections/vim.yaml }
models:
  student: \"brain:Qwen/Qwen3-4B\"
  teacher: \"hosted:deepseek/deepseek-v4.1-flash\"
stages:
  - { id: tasks, uses: tasks.pin, inputs: { collection: $inputs.course }, with: { split: train } }
  - id: local
    uses: attempt.collect
    inputs: { tasks: $stages.tasks.tasks }
    with: { agent: sven, model: student, environment: environments/vim.yaml, attempts: 2, seed: 11 }
  - id: failures
    uses: attempts.failures
    inputs: { tasks: $stages.tasks.tasks, attempts: [$stages.local.attempts] }
  - id: briefs
    uses: agent
    foreach: $stages.failures.failures
    inputs: { input: $item }
    with: { instruction: prompts/brief.md, output_schema: schemas/brief.json, model: teacher }
  - id: corrected
    uses: attempt.collect
    inputs: { tasks: $stages.tasks.tasks, unsolved_by: [$stages.local.attempts],
              briefs: $stages.briefs.output }
    with: { agent: sven, model: student, hint: brief, environment: environments/vim.yaml,
            attempts: 2, seed: 12 }
  - id: sft
    uses: dataset.project
    inputs: { attempts: [$stages.local.attempts, $stages.corrected.attempts] }
    with: { target: student, seed: 13 }
";

fn repo() -> Repo {
    let repo = Repo::new();
    repo.write(
        "prompts/brief.md",
        "Brief the student on the task it failed.",
    );
    repo.write(
        "schemas/brief.json",
        r#"{"type": "object", "properties": {"brief": {"type": "string"}},
            "required": ["brief"], "additionalProperties": false}"#,
    );
    repo
}

fn run(resolved: &Resolved, dir: &Path, backend: &FakeBackend) -> Result<(), RunError> {
    let trainer = FakeTrainer::default();
    let services = Services {
        backend,
        trainer: Some(&trainer),
        models: Some(&Teachers),
    };
    let mut factory = Factory::open(resolved, dir, services, "test")?;
    let ran = factory.run().map(|_| ());
    drop(factory);
    ran
}

#[test]
fn a_briefed_correction_learns_the_task_without_its_brief_and_never_a_trial_that_repeats_it() {
    let repo = repo();
    let original =
        std::fs::read_to_string(repo.root.join("tasks/vim/train/001-first/instruction.md"))
            .unwrap();
    let backend = FakeBackend {
        echo_instruction: true,
        instructions: Mutex::new(Vec::new()),
        ..FakeBackend::with(Outcome::Fail)
    };
    for task in ["train/001-first", "train/002-second"] {
        backend.script(
            task,
            &[Outcome::Fail, Outcome::Fail, Outcome::Pass, Outcome::Pass],
        );
    }
    let dir = tempfile::tempdir().unwrap();
    run(
        &resolve(&repo.root, RECIPE, connections()).unwrap(),
        dir.path(),
        &backend,
    )
    .unwrap();

    // Every task the student left unsolved, with its instruction and attempt.
    let failures: ItemIndex = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("failures").join(ITEMS_FILE)).unwrap(),
    )
    .unwrap();
    let ids: Vec<&str> = failures.items.iter().map(|i| i.id.as_str()).collect();
    assert_eq!(ids, ["train--001-first", "train--002-second"]);
    let attempt =
        std::fs::read_to_string(dir.path().join("failures/train--001-first/attempt.md")).unwrap();
    assert!(attempt.starts_with("# The failed attempt"), "{attempt}");
    assert_eq!(
        std::fs::read_to_string(dir.path().join("failures/train--001-first/instruction.md"))
            .unwrap(),
        original
    );

    // The corrected attempts were shown their briefs beside the instruction.
    let instructions = backend.instructions.lock().unwrap().clone();
    let briefed: Vec<&String> = instructions
        .iter()
        .filter(|i| i.contains("<briefing>"))
        .collect();
    assert_eq!(briefed.len(), 4, "{instructions:?}");
    assert!(briefed
        .iter()
        .any(|i| i.starts_with(&original) && i.contains("save the file with :wq")));

    // Learned under the task's own instruction; the trials that repeat their
    // brief are not learned.
    let dataset = dataset_of(&dir.path().join("sft")).unwrap();
    assert_eq!(
        dataset.exclusions.get("hint_repeated"),
        Some(&2),
        "{:?}",
        dataset.exclusions
    );
    let train = std::fs::read_to_string(dir.path().join("sft/train.jsonl")).unwrap();
    assert_eq!(train.lines().count(), 2, "both successes of the first task");
    assert!(
        !train.contains("save the file with :wq"),
        "no brief in a record"
    );
}

#[test]
fn briefs_are_shown_only_by_a_pass_that_says_so() {
    let repo = repo();
    let refused = |recipe: String| {
        resolve(&repo.root, &recipe, connections())
            .unwrap_err()
            .0
            .join("\n")
    };
    let unbound = refused(RECIPE.replace(",\n              briefs: $stages.briefs.output }", " }"));
    assert!(unbound.contains("bind `briefs`"), "{unbound}");
    let unshown = refused(RECIPE.replace("hint: brief, ", ""));
    assert!(
        unshown.contains("only a pass with hint: brief"),
        "{unshown}"
    );
}

#[test]
fn a_hint_is_trusted_only_where_its_solution_passes_the_tasks_verifier() {
    let repo = repo();
    repo.write(
        "tasks/vim/train/001-first/solution/solve.sh",
        "#!/bin/sh\nprintf 'Hello world' > main.txt\n",
    );
    // seed: a recipe must state every seed; the fake backend's outcomes are scripted, so none draws.
    let recipe = "schema: splinter.recipe/v2
name: regraded
inputs:
  course: { kind: files, path: collections/vim.yaml }
models:
  student: \"brain:Qwen/Qwen3-4B\"
stages:
  - { id: tasks, uses: tasks.pin, inputs: { collection: $inputs.course }, with: { split: train } }
  - id: local
    uses: attempt.collect
    inputs: { tasks: $stages.tasks.tasks }
    with: { agent: sven, model: student, environment: environments/vim.yaml, attempts: 2, seed: 11 }
  - id: oracle
    uses: attempt.collect
    inputs: { tasks: $stages.tasks.tasks, unsolved_by: [$stages.local.attempts] }
    with: { agent: oracle, environment: environments/vim.yaml, seed: 12 }
  - id: hinted
    uses: attempt.collect
    inputs: { tasks: $stages.tasks.tasks, unsolved_by: [$stages.local.attempts],
              solved_by: [$stages.oracle.attempts] }
    with: { agent: sven, model: student, hint: solution, environment: environments/vim.yaml,
            attempts: 2, seed: 13 }
";
    let backend = FakeBackend::with(Outcome::Fail);
    // The student fails both; the reference solution of the second fails its
    // own verifier, so only the first is worth a hint.
    backend.script(
        "train/001-first",
        &[
            Outcome::Fail,
            Outcome::Fail,
            Outcome::Pass,
            Outcome::Pass,
            Outcome::Pass,
        ],
    );
    backend.script(
        "train/002-second",
        &[Outcome::Fail, Outcome::Fail, Outcome::Fail],
    );
    let dir = tempfile::tempdir().unwrap();
    run(
        &resolve(&repo.root, recipe, connections()).unwrap(),
        dir.path(),
        &backend,
    )
    .unwrap();
    let hinted: Vec<String> = backend
        .executed()
        .iter()
        .filter(|s| s.slot.pass == "hinted")
        .map(|s| s.task.id.clone())
        .collect();
    assert_eq!(hinted, ["train/001-first", "train/001-first"]);
}
