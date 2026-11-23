// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training data built from what agents
// actually did, every record traceable to the attempt it came from, for its
// clients. If your team needs expertise in learning from agent experience,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Typed dataset projections: one set of attempts gives an SFT dataset (the
//! verified successes), a feedback dataset (each verified attempt, desirable
//! when it passed), a preference dataset (a pass and a fail of one task,
//! from where they part) and a policy-gradient dataset (every graded attempt
//! with its reward and its group), each by splinter-data's own view, and the
//! attempts are made once.

use serde_json::Value;
use splinter_factory::resolve::resolve;
use splinter_factory::run::{Factory, Services};

use crate::fakes::{FakeBackend, Outcome};
use crate::repo::{connections, Repo};

// seed: a recipe states every seed; the fake backend's outcomes are scripted, so none draws.
const RECIPE: &str = "schema: splinter.recipe/v2
name: projected
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
  - id: sft
    uses: dataset.project
    inputs: { attempts: [$stages.local.attempts] }
    with: { target: student, seed: 13 }
  - id: preferred
    uses: dataset.project
    inputs: { attempts: [$stages.local.attempts] }
    with: { objective: preference, target: student, seed: 13 }
  - id: judged
    uses: dataset.project
    inputs: { attempts: [$stages.local.attempts] }
    with: { objective: feedback, target: student, seed: 13 }
  - id: rewarded
    uses: dataset.project
    inputs: { attempts: [$stages.local.attempts] }
    with: { objective: policy-gradient, target: student, seed: 13 }
";

fn lines(path: &std::path::Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

#[test]
fn one_set_of_attempts_gives_sft_preference_and_policy_gradient_datasets() {
    let repo = Repo::new();
    let resolved = resolve(&repo.root, RECIPE, connections()).unwrap();
    let backend = FakeBackend {
        vary_failures: true,
        ..FakeBackend::with(Outcome::Fail)
    };
    for task in ["train/001-first", "train/002-second"] {
        backend.script(task, &[Outcome::Pass, Outcome::Fail]);
    }
    let services = Services {
        backend: &backend,
        trainer: None,
        models: None,
        variables: None,
    };
    let dir = tempfile::tempdir().unwrap();
    Factory::open(&resolved, dir.path(), services, "test")
        .unwrap()
        .run()
        .unwrap();
    assert_eq!(backend.executed().len(), 4, "the attempts are made once");

    let objective = |stage: &str| -> Value {
        serde_json::from_str::<Value>(
            &std::fs::read_to_string(dir.path().join(stage).join("dataset.json")).unwrap(),
        )
        .unwrap()["objective"]
            .clone()
    };
    assert_eq!(lines(&dir.path().join("sft/train.jsonl")).len(), 2);
    assert_eq!(objective("sft"), "sft");

    // A pass over a fail of each task, the shared prompt never trained on.
    let pairs = lines(&dir.path().join("preferred/train.jsonl"));
    assert_eq!(pairs.len(), 2, "{pairs:#?}");
    assert_eq!(objective("preferred"), "preference");
    for pair in &pairs {
        let (chosen, rejected) = (pair["chosen"].to_string(), pair["rejected"].to_string());
        assert!(chosen.contains("Done."), "{chosen}");
        assert!(rejected.contains("I could not finish."), "{rejected}");
        assert!(!pair["prompt"].to_string().contains("could not finish"));
    }

    // Each verified attempt, desirable when it passed.
    let judged = lines(&dir.path().join("judged/train.jsonl"));
    assert_eq!(objective("judged"), "feedback");
    let mut desirable: Vec<bool> = judged
        .iter()
        .map(|r| r["desirable"].as_bool().unwrap())
        .collect();
    desirable.sort();
    assert_eq!(desirable, [false, false, true, true], "{judged:#?}");

    // Every action of every graded attempt, with the attempt's reward and
    // its group: four attempts, two per task, half of them passed.
    let rewarded = lines(&dir.path().join("rewarded/train.jsonl"));
    assert_eq!(objective("rewarded"), "policy-gradient");
    let attempts: std::collections::BTreeMap<String, f64> = rewarded
        .iter()
        .map(|r| {
            (
                r["rollout"].as_str().unwrap().to_owned(),
                r["reward"].as_f64().unwrap(),
            )
        })
        .collect();
    assert_eq!(attempts.len(), 4, "{attempts:?}");
    let mut rewards: Vec<f64> = attempts.values().copied().collect();
    rewards.sort_by(f64::total_cmp);
    assert_eq!(rewards, [0.0, 0.0, 1.0, 1.0]);
    let groups: std::collections::BTreeSet<String> = rewarded
        .iter()
        .map(|r| r["group"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(groups.len(), 2, "a group per task");
}

/// Each dataset is trained by its objective's regime: Brain trains chat,
/// preference, judged-completion and rollout records, each by its own job.
#[test]
fn each_dataset_is_trained_by_its_objectives_regime() {
    let repo = Repo::new();
    // seed: a training stage states its seed; the fake trainer draws nothing.
    let trained: String = ["sft", "preferred", "judged", "rewarded"]
        .iter()
        .map(|d| {
            format!(
                "  - id: lora-{d}\n    uses: model.train\n    inputs: {{ dataset: $stages.{d}.dataset }}\n    with: {{ base: student, method: lora, params: {{ rank: 8 }}, seed: 1 }}\n"
            )
        })
        .collect();
    let resolved = resolve(&repo.root, &(RECIPE.to_owned() + &trained), connections()).unwrap();
    let backend = FakeBackend {
        vary_failures: true,
        ..FakeBackend::with(Outcome::Fail)
    };
    for task in ["train/001-first", "train/002-second"] {
        backend.script(task, &[Outcome::Pass, Outcome::Fail]);
    }
    let trainer = crate::fakes::FakeTrainer::default();
    let services = Services {
        backend: &backend,
        trainer: Some(&trainer),
        models: None,
        variables: None,
    };
    let dir = tempfile::tempdir().unwrap();
    Factory::open(&resolved, dir.path(), services, "test")
        .unwrap()
        .run()
        .unwrap();
    assert_eq!(
        *trainer.objectives.lock().unwrap(),
        ["sft", "preference", "feedback", "policy-gradient"]
    );
}

#[test]
fn a_dataset_is_projected_for_one_objective_and_trained_by_its_trainer() {
    let repo = Repo::new();
    let refused = |stages: &str| {
        let recipe = RECIPE.to_owned() + stages;
        resolve(&repo.root, &recipe, connections())
            .err()
            .map(|e| format!("{e:?}"))
            .unwrap_or_default()
    };
    // seed: a dataset and a training stage state their seeds; the refusal is at plan time.
    assert!(refused(
        "  - id: retrieved\n    uses: dataset.project\n    inputs: { attempts: [$stages.local.attempts] }\n    with: { objective: retrieval, target: student, seed: 13 }\n  - id: lora\n    uses: model.train\n    inputs: { dataset: $stages.retrieved.dataset }\n    with: { base: student, method: lora, params: { rank: 8 }, seed: 1 }\n"
    )
    .contains("a retrieval dataset trains an embedder"));
    assert!(refused(
        "  - id: odd\n    uses: dataset.project\n    inputs: { attempts: [$stages.local.attempts] }\n    with: { objective: policy-gradient, preference: { pairs_per_task: 2 }, target: student, seed: 13 }\n"
    )
    .contains("gives preference options to another objective"));
}

// seed: a recipe states every seed; the fake backend's outcomes are scripted, so none draws.
const LEARNABLE: &str = "  - id: rlvr
    uses: dataset.rlvr
    inputs: { tasks: $stages.tasks.tasks, attempts: [$stages.local.attempts] }
  - { id: online, uses: tasks.pin, inputs: { collection: $stages.rlvr.tasks }, with: { split: train } }
";

/// RLVR: the tasks a policy-gradient trainer can learn from are those the
/// attempts show a spread on (passed sometimes, failed sometimes), kept as
/// an executable collection with their own verifiers for an online trainer
/// to roll out again.
#[test]
fn the_tasks_attempts_solve_only_sometimes_are_an_rlvr_collection() {
    let repo = Repo::new();
    let resolved = resolve(&repo.root, &(RECIPE.to_owned() + LEARNABLE), connections()).unwrap();
    let backend = FakeBackend {
        vary_failures: true,
        ..FakeBackend::with(Outcome::Fail)
    };
    backend.script("train/001-first", &[Outcome::Pass, Outcome::Fail]);
    backend.script("train/002-second", &[Outcome::Pass, Outcome::Pass]);
    let services = Services {
        backend: &backend,
        trainer: None,
        models: None,
        variables: None,
    };
    let dir = tempfile::tempdir().unwrap();
    Factory::open(&resolved, dir.path(), services, "test")
        .unwrap()
        .run()
        .unwrap();
    // SFT, preference, policy-gradient and RLVR products of one set of
    // attempts, made once.
    assert_eq!(backend.executed().len(), 4);
    let online = splinter_factory::run::tasks::tasks_of(&dir.path().join("online")).unwrap();
    let ids: Vec<&str> = online.tasks.iter().map(|t| t.id.as_str()).collect();
    assert_eq!(ids, ["train/001-first"]);
    assert!(dir
        .path()
        .join("online/train/001-first/tests/test.sh")
        .is_file());
    let report: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.path().join("rlvr/rlvr.json")).unwrap())
            .unwrap();
    assert_eq!(report["train/002-second"]["kept"], false, "{report:#}");
    assert_eq!(report["train/001-first"]["pass_rate"], 0.5, "{report:#}");
}
