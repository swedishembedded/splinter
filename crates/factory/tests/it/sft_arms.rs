// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled fine-tuning experiments whose
// arms differ only in what they supervise, for its clients. If your team needs
// expertise in measuring what an agent learns from its data, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The options of an SFT `dataset.project` stage that make the arms of a
//! controlled experiment from one set of attempts: which actions are
//! supervised, how many trials of a task are kept, a negative control, and
//! the composition every SFT dataset reports.

use serde_json::Value;
use splinter_factory::run::fork;

use crate::factory_run::{resolved, run};
use crate::fakes::{FakeBackend, FakeTrainer, Outcome, FIRST_CALL};
use crate::repo::Repo;

// seed: a recipe states every seed; the fake backend's outcomes are scripted, so none draws.
const RECIPE: &str = "schema: splinter.recipe/v2
name: arms
inputs:
  course: { kind: files, path: collections/vim.yaml }
models:
  student: \"brain:Qwen/Qwen3-4B\"
stages:
  - { id: tasks, uses: tasks.pin, inputs: { collection: $inputs.course }, with: { split: train } }
  - id: local
    uses: attempt.collect
    inputs: { tasks: $stages.tasks.tasks }
    with: { agent: sven, model: student, environment: environments/vim.yaml, attempts: 1, seed: 11 }
  - id: sft
    uses: dataset.project
    inputs: { attempts: [$stages.local.attempts] }
    with: { target: student, seed: 13 }
";

/// `RECIPE` with a second SFT stage `arm` given `with`.
// seed: the arm states the same seed as `sft`, so two arms differ only in `with`.
fn with_arm(with: &str) -> String {
    format!(
        "{RECIPE}  - id: arm\n    uses: dataset.project\n    inputs: {{ attempts: [$stages.local.attempts] }}\n    \
         with: {{ target: student, seed: 13, {with} }}\n"
    )
}

fn lines(path: &std::path::Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

/// Each record's task and, in order, whether each assistant message is supervised.
fn supervision(path: &std::path::Path) -> Vec<(String, Vec<bool>)> {
    let mut out: Vec<(String, Vec<bool>)> = lines(path)
        .iter()
        .map(|r| {
            let task = r["metadata"]["group"].as_str().unwrap().to_owned();
            let trained = r["messages"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|m| m["role"] == "assistant")
                .map(|m| m["train"].as_bool().unwrap())
                .collect();
            (task, trained)
        })
        .collect();
    out.sort();
    out
}

#[test]
fn productive_supervision_keeps_an_action_answered_only_by_errors_as_context() {
    let repo = Repo::new();
    let backend = FakeBackend::with(Outcome::Pass);
    backend.script("train/001-first", &[Outcome::PassAfterError]);
    let trainer = FakeTrainer::default();
    let dir = tempfile::tempdir().unwrap();
    run(
        &resolved(&repo, &with_arm("supervise: productive")),
        dir.path(),
        &backend,
        &trainer,
    )
    .unwrap();

    // Every action supervised by default, the failed one included.
    assert_eq!(
        supervision(&dir.path().join("sft/train.jsonl")),
        [
            ("train/001-first".to_owned(), vec![true, true, true]),
            ("train/002-second".to_owned(), vec![true, true, true]),
        ]
    );
    // The action whose call failed is context; the recovery and the final
    // answer are supervised; a trial with no error is unchanged.
    assert_eq!(
        supervision(&dir.path().join("arm/train.jsonl")),
        [
            ("train/001-first".to_owned(), vec![false, true, true]),
            ("train/002-second".to_owned(), vec![true, true, true]),
        ]
    );
    let masked = lines(&dir.path().join("arm/train.jsonl"))
        .into_iter()
        .find(|r| r["metadata"]["group"] == "train/001-first")
        .unwrap();
    let context = masked["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["role"] == "assistant")
        .unwrap();
    assert!(
        context["tool_calls"][0]["id"]
            .as_str()
            .unwrap()
            .ends_with(FIRST_CALL),
        "{context}"
    );
    let unchanged = |stage: &str| {
        lines(&dir.path().join(stage).join("train.jsonl"))
            .into_iter()
            .find(|r| r["metadata"]["group"] == "train/002-second")
            .unwrap()
    };
    assert_eq!(unchanged("sft"), unchanged("arm"));
    let manifest: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("arm/dataset.json")).unwrap(),
    )
    .unwrap();
    assert!(manifest.get("control").is_none());
    let report = std::fs::read_to_string(dir.path().join("arm/report.md")).unwrap();
    assert!(
        report.contains("1 action whose every call failed kept as context"),
        "{report}"
    );
}

#[test]
fn naming_the_default_supervision_changes_no_stage_identity() {
    let repo = Repo::new();
    let backend = FakeBackend::with(Outcome::Pass);
    let trainer = FakeTrainer::default();
    let first = tempfile::tempdir().unwrap();
    run(&resolved(&repo, RECIPE), first.path(), &backend, &trainer).unwrap();
    let second = tempfile::tempdir().unwrap();
    fork(first.path(), second.path()).unwrap();
    let named = RECIPE.replace("seed: 13 }", "seed: 13, supervise: all }");
    let computed: Vec<String> = run(&resolved(&repo, &named), second.path(), &backend, &trainer)
        .unwrap()
        .into_iter()
        .filter(|(_, computed)| *computed)
        .map(|(key, _)| key)
        .collect();
    assert!(computed.is_empty(), "{computed:?}");
}

#[test]
fn productive_supervision_needs_the_requests_the_model_was_sent() {
    let repo = Repo::new();
    let recipe = with_arm("supervise: productive, projection: portable-sft");
    let refused =
        splinter_factory::resolve::resolve(&repo.root, &recipe, crate::repo::connections())
            .err()
            .map(|e| format!("{e:?}"))
            .unwrap_or_default();
    assert!(refused.contains("supervise: productive"), "{refused}");
    let recipe = with_arm("objective: preference, supervise: productive");
    let refused =
        splinter_factory::resolve::resolve(&repo.root, &recipe, crate::repo::connections())
            .err()
            .map(|e| format!("{e:?}"))
            .unwrap_or_default();
    assert!(refused.contains("supervise"), "{refused}");
}

/// `RECIPE` with two attempts per task.
fn twice(recipe: &str) -> String {
    recipe.replace("attempts: 1, seed: 11", "attempts: 2, seed: 11")
}

#[test]
fn a_per_task_limit_keeps_the_trials_with_the_fewest_actions_and_lists_the_rest() {
    let repo = Repo::new();
    let backend = FakeBackend::with(Outcome::Pass);
    backend.script("train/001-first", &[Outcome::LongPass, Outcome::Pass]);
    let trainer = FakeTrainer::default();
    let dir = tempfile::tempdir().unwrap();
    run(
        &resolved(&repo, &twice(&with_arm("per_task: 1"))),
        dir.path(),
        &backend,
        &trainer,
    )
    .unwrap();
    // Without the limit every verified trial is a record.
    assert_eq!(lines(&dir.path().join("sft/train.jsonl")).len(), 4);
    let kept = supervision(&dir.path().join("arm/train.jsonl"));
    assert_eq!(
        kept,
        [
            ("train/001-first".to_owned(), vec![true, true, true]),
            ("train/002-second".to_owned(), vec![true, true, true]),
        ],
        "one trial per task, the shorter one of the first"
    );
    let excluded = lines(&dir.path().join("arm/excluded.jsonl"));
    assert_eq!(excluded.len(), 2, "{excluded:#?}");
    assert!(excluded.iter().all(|e| e["reason"] == "per_task_limit"));
    let long = excluded
        .iter()
        .find(|e| e["task"] == "train/001-first")
        .unwrap();
    assert!(
        long["detail"].as_str().unwrap().contains("7 actions"),
        "{long}"
    );
    let manifest: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("arm/dataset.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(manifest["exclusions"]["per_task_limit"], 2);
    let steps: Vec<&str> = manifest["waterfall"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["step"].as_str().unwrap())
        .collect();
    assert_eq!(
        steps,
        [
            "raw_trials",
            "verified_successes",
            "projected",
            "per_task",
            "accepted"
        ]
    );
}

#[test]
fn a_random_per_task_draw_is_the_seeds() {
    let repo = Repo::new();
    let backend = FakeBackend::with(Outcome::Pass);
    backend.script("train/001-first", &[Outcome::LongPass, Outcome::Pass]);
    let trainer = FakeTrainer::default();
    let kept = |seed: u64| {
        let dir = tempfile::tempdir().unwrap();
        let recipe = twice(&with_arm("per_task: 1, prefer: random"))
            .replace("seed: 13, per_task", &format!("seed: {seed}, per_task"));
        run(&resolved(&repo, &recipe), dir.path(), &backend, &trainer).unwrap();
        lines(&dir.path().join("arm/accepted.jsonl"))
            .into_iter()
            .map(|a| (a["task"].to_string(), a["slot"].to_string()))
            .collect::<Vec<_>>()
    };
    let draws: Vec<_> = (0..12).map(kept).collect();
    assert!(draws.iter().all(|d| d.len() == 2), "{draws:?}");
    assert_eq!(kept(3), draws[3], "the same seed draws the same trials");
    assert!(
        draws.iter().any(|d| d != &draws[0]),
        "another seed draws others: {draws:?}"
    );
}

#[test]
fn a_preference_without_a_per_task_limit_is_refused() {
    let repo = Repo::new();
    for with in [
        "prefer: random",
        "per_task: 0",
        "objective: feedback, per_task: 1",
    ] {
        let refused = splinter_factory::resolve::resolve(
            &repo.root,
            &with_arm(with),
            crate::repo::connections(),
        )
        .err()
        .map(|e| format!("{e:?}"))
        .unwrap_or_default();
        assert!(refused.contains("per_task"), "{with}: {refused}");
    }
}

/// `repo` with two more training tasks, four in all.
fn four_tasks(repo: &Repo) {
    for id in ["train/003-third", "train/004-fourth"] {
        repo.write(
            &format!("tasks/vim/{id}/task.toml"),
            &format!("[task]\nname = \"org/{id}\"\n[metadata]\nbench_family = \"course\"\n"),
        );
        repo.write(
            &format!("tasks/vim/{id}/instruction.md"),
            &format!("Do {id}.\n"),
        );
        repo.write(&format!("tasks/vim/{id}/tests/test.sh"), "exit 0\n");
    }
}

/// The first user message of each record, by its trial's pass and slot.
fn requests(path: &std::path::Path) -> std::collections::BTreeMap<String, (String, Value)> {
    lines(path)
        .into_iter()
        .map(|r| {
            let trial = &r["metadata"]["trial"];
            let key = format!("{}#{}", trial["pass"], trial["slot"]);
            let task = r["metadata"]["group"].as_str().unwrap().to_owned();
            (key, (task, r))
        })
        .collect()
}

fn first_user(record: &Value) -> String {
    record["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["role"] == "user")
        .unwrap()["content"]
        .as_str()
        .unwrap()
        .to_owned()
}

#[test]
fn a_shuffled_request_control_gives_each_record_another_tasks_request_and_nothing_else() {
    let repo = Repo::new();
    four_tasks(&repo);
    let backend = FakeBackend {
        echo_instruction: true,
        ..FakeBackend::with(Outcome::Pass)
    };
    let trainer = FakeTrainer::default();
    let dir = tempfile::tempdir().unwrap();
    run(
        &resolved(&repo, &twice(&with_arm("control: shuffled-requests"))),
        dir.path(),
        &backend,
        &trainer,
    )
    .unwrap();
    let own = requests(&dir.path().join("sft/train.jsonl"));
    let shuffled = requests(&dir.path().join("arm/train.jsonl"));
    assert_eq!(own.len(), 8);
    assert_eq!(
        own.keys().collect::<Vec<_>>(),
        shuffled.keys().collect::<Vec<_>>()
    );
    // Each task's own request: its instruction as the agent was sent it.
    let mut request_of: std::collections::BTreeMap<String, String> = Default::default();
    for (task, record) in own.values() {
        let request = first_user(record);
        assert!(request.starts_with(&format!("Do {task}.")), "{request}");
        assert_eq!(
            request_of.entry(task.clone()).or_insert(request.clone()),
            &request
        );
    }
    let mut given: std::collections::BTreeMap<String, String> = Default::default();
    for (key, (task, record)) in &shuffled {
        let (own_task, own_record) = &own[key];
        assert_eq!(task, own_task, "a record keeps its own task");
        let request = first_user(record);
        assert_ne!(request, request_of[task], "{key} kept its own request");
        let donor = request_of
            .iter()
            .find(|(_, r)| **r == request)
            .map(|(t, _)| t.clone())
            .expect("another admitted task's request");
        // Every record of a task is given the same other task's request.
        assert_eq!(given.entry(task.clone()).or_insert(donor.clone()), &donor);
        // Everything but the request is the record's own.
        let mut restored = record.clone();
        for m in restored["messages"].as_array_mut().unwrap() {
            if m["role"] == "user" {
                m["content"] = Value::String(request_of[task].clone());
                break;
            }
        }
        assert_eq!(&restored, own_record);
    }
    assert_eq!(given.len(), 4, "every task is given another's request");
    let supervision = lines(&dir.path().join("arm/supervision.jsonl"));
    assert!(supervision
        .iter()
        .all(|l| given[l["task"].as_str().unwrap()] == l["request_of"].as_str().unwrap()));
    let manifest: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("arm/dataset.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(manifest["control"], "shuffled-requests");
    let plain: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("sft/dataset.json")).unwrap(),
    )
    .unwrap();
    assert!(plain.get("control").is_none());
}

#[test]
fn a_shuffled_request_control_needs_two_tasks() {
    let repo = Repo::new();
    let backend = FakeBackend::with(Outcome::Pass);
    backend.script("train/002-second", &[Outcome::Fail]);
    let trainer = FakeTrainer::default();
    let dir = tempfile::tempdir().unwrap();
    let failed = run(
        &resolved(&repo, &with_arm("control: shuffled-requests")),
        dir.path(),
        &backend,
        &trainer,
    )
    .err()
    .map(|e| e.to_string())
    .unwrap_or_default();
    assert!(failed.contains("two tasks"), "{failed}");
    let refused = splinter_factory::resolve::resolve(
        &repo.root,
        &with_arm("objective: preference, control: shuffled-requests"),
        crate::repo::connections(),
    )
    .err()
    .map(|e| format!("{e:?}"))
    .unwrap_or_default();
    assert!(refused.contains("control"), "{refused}");
}

fn composition(dir: &std::path::Path, stage: &str) -> Value {
    let manifest: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join(stage).join("dataset.json")).unwrap(),
    )
    .unwrap();
    let file = manifest["composition"]["path"].as_str().unwrap();
    serde_json::from_str(&std::fs::read_to_string(dir.join(stage).join(file)).unwrap()).unwrap()
}

/// The characters of a message the loss reads: its text, its reasoning and
/// each call's name and arguments.
fn characters(message: &Value) -> u64 {
    let text = |v: &Value| v.as_str().map_or(0, |s| s.chars().count() as u64);
    text(&message["content"])
        + text(&message["reasoning_content"])
        + message["tool_calls"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|c| text(&c["function"]["name"]) + text(&c["function"]["arguments"]))
            .sum::<u64>()
}

#[test]
fn every_sft_dataset_reports_its_composition() {
    let repo = Repo::new();
    let backend = FakeBackend::with(Outcome::Pass);
    backend.script("train/001-first", &[Outcome::PassAfterError, Outcome::Pass]);
    let trainer = FakeTrainer::default();
    let dir = tempfile::tempdir().unwrap();
    run(
        &resolved(&repo, &twice(&with_arm("supervise: productive"))),
        dir.path(),
        &backend,
        &trainer,
    )
    .unwrap();
    let all = composition(dir.path(), "sft");
    let productive = composition(dir.path(), "arm");
    for c in [&all, &productive] {
        assert_eq!(c["records"], 4, "{c:#}");
        assert_eq!(c["train_records"], 4);
        assert_eq!(c["validation_records"], 0);
        assert_eq!(c["trials"], 4);
        assert_eq!(c["tasks"], 2);
    }
    // The four trials take the same actions; leaving one failed action out
    // of the loss makes that trial's supervised sequence another one.
    assert_eq!(all["action_sequences"], 1);
    assert_eq!(productive["action_sequences"], 2);
    assert_eq!(all["supervised_messages"], 12);
    assert_eq!(all["masked_messages"], 0);
    assert_eq!(all["failed_actions_masked"], 0);
    assert_eq!(all["supervise"], "all");
    assert_eq!(productive["supervised_messages"], 11);
    assert_eq!(productive["masked_messages"], 1);
    assert_eq!(productive["failed_actions_masked"], 1);
    assert_eq!(productive["supervise"], "productive");
    assert!(all["control"].is_null());
    // The supervised characters are those of the supervised messages, and
    // the masked action's are no longer among them.
    let supervised: u64 = lines(&dir.path().join("sft/train.jsonl"))
        .iter()
        .flat_map(|r| r["messages"].as_array().unwrap().clone())
        .filter(|m| m["train"] == true)
        .map(|m| characters(&m))
        .sum();
    assert_eq!(all["supervised_characters"], supervised);
    let masked: u64 = lines(&dir.path().join("arm/train.jsonl"))
        .iter()
        .flat_map(|r| r["messages"].as_array().unwrap().clone())
        .filter(|m| m["role"] == "assistant" && m["train"] == false)
        .map(|m| characters(&m))
        .sum();
    assert!(masked > 0);
    assert_eq!(
        productive["supervised_characters"].as_u64().unwrap() + masked,
        supervised
    );
    let report = std::fs::read_to_string(dir.path().join("arm/report.md")).unwrap();
    assert!(report.contains("## Composition"), "{report}");
}

/// `repo` with `n` more training tasks.
fn more_tasks(repo: &Repo, n: usize) {
    for i in 0..n {
        let id = format!("train/1{i:02}-more");
        repo.write(
            &format!("tasks/vim/{id}/task.toml"),
            &format!("[task]\nname = \"org/{id}\"\n[metadata]\nbench_family = \"course\"\n"),
        );
        repo.write(
            &format!("tasks/vim/{id}/instruction.md"),
            &format!("Do {id}.\n"),
        );
        repo.write(&format!("tasks/vim/{id}/tests/test.sh"), "exit 0\n");
    }
}

/// Two pins of the training tasks: the kept and the held side of one
/// holdout of `percent` drawn by `seed`.
// seed: the attempt stages state a seed; the fake backend draws nothing from it.
fn sides(percent: u32, seed: u64) -> String {
    format!(
        "schema: splinter.recipe/v2
name: sides
inputs:
  course: {{ kind: files, path: collections/vim.yaml }}
models:
  student: \"brain:Qwen/Qwen3-4B\"
stages:
  - id: practice
    uses: tasks.pin
    inputs: {{ collection: $inputs.course }}
    with: {{ split: train, holdout: {{ percent: {percent}, seed: {seed}, side: kept }} }}
  - id: exam
    uses: tasks.pin
    inputs: {{ collection: $inputs.course }}
    with: {{ split: train, holdout: {{ percent: {percent}, seed: {seed}, side: held }} }}
  - id: local
    uses: attempt.collect
    inputs: {{ tasks: $stages.practice.tasks }}
    with: {{ agent: sven, model: student, environment: environments/vim.yaml, seed: 11 }}
  - id: exam-attempts
    uses: attempt.collect
    inputs: {{ tasks: $stages.exam.tasks }}
    with: {{ agent: sven, model: student, environment: environments/vim.yaml, seed: 11 }}
"
    )
}

fn pinned(resolved: &splinter_factory::resolve::Resolved, stage: &str) -> Vec<String> {
    resolved.collections[stage]
        .collection
        .tasks
        .iter()
        .map(|t| t.id.clone())
        .collect()
}

#[test]
fn a_pins_holdout_splits_the_tasks_by_id_into_complementary_sides() {
    let repo = Repo::new();
    more_tasks(&repo, 30);
    let all: std::collections::BTreeSet<String> = resolved(&repo, RECIPE).collections["tasks"]
        .collection
        .tasks
        .iter()
        .map(|t| t.id.clone())
        .collect();
    assert_eq!(all.len(), 32);
    // seed: any seed splits 32 tasks at 25% into two non-empty sides but for a negligible chance; one is enough.
    let split = resolved(&repo, &sides(25, 7));
    let (kept, held) = (pinned(&split, "practice"), pinned(&split, "exam"));
    assert!(!kept.is_empty() && !held.is_empty(), "{kept:?} {held:?}");
    assert!(
        kept.iter().all(|t| !held.contains(t)),
        "no task on both sides"
    );
    let union: std::collections::BTreeSet<String> = kept.iter().chain(&held).cloned().collect();
    assert_eq!(union, all, "every task on one side");
    // Each task's side is its own draw: adding tasks moves none of the others.
    more_tasks(&repo, 40);
    let grown = resolved(&repo, &sides(25, 7));
    let regrown = pinned(&grown, "exam");
    assert!(
        held.iter().all(|t| regrown.contains(t)),
        "{held:?} {regrown:?}"
    );
    assert!(pinned(&grown, "practice").iter().all(|t| !held.contains(t)));
    // Another seed draws another side.
    let other = resolved(&repo, &sides(25, 8));
    assert_ne!(pinned(&other, "exam"), regrown);
    // The run pins what the plan read.
    let backend = FakeBackend::with(Outcome::Pass);
    let dir = tempfile::tempdir().unwrap();
    run(&grown, dir.path(), &backend, &FakeTrainer::default()).unwrap();
    let ran: std::collections::BTreeSet<String> = backend
        .executed()
        .into_iter()
        .filter(|s| s.slot.pass == "exam-attempts")
        .map(|s| s.task.id)
        .collect();
    assert_eq!(ran, regrown.into_iter().collect());
}

#[test]
fn a_holdout_that_keeps_every_task_or_none_is_refused() {
    let repo = Repo::new();
    for percent in [0, 100] {
        let refused = splinter_factory::resolve::resolve(
            &repo.root,
            &sides(percent, 7),
            crate::repo::connections(),
        )
        .err()
        .map(|e| format!("{e:?}"))
        .unwrap_or_default();
        assert!(refused.contains("holdout"), "{percent}: {refused}");
    }
}
