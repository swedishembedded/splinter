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
  - id: distilled
    uses: dataset.project
    inputs: { attempts: [$stages.local.attempts] }
    with: { objective: distill, target: student, seed: 13 }
  - id: rewarded
    uses: dataset.project
    inputs: { attempts: [$stages.local.attempts] }
    with: { objective: policy-gradient, target: student, seed: 13 }
";

/// The reference repository, each training task with its reference
/// solution: what a distilled record's teacher is shown.
fn repo() -> Repo {
    let repo = Repo::new();
    for task in ["train/001-first", "train/002-second"] {
        repo.write(
            &format!("tasks/vim/{task}/solution/solve.sh"),
            "echo solved\n",
        );
    }
    repo
}

fn lines(path: &std::path::Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

#[test]
fn one_set_of_attempts_gives_sft_preference_and_policy_gradient_datasets() {
    let repo = repo();
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
        online: None,
        build: splinter_core::build::BuildInfo::this_executable(),
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

    // Every attempt, its conversation as the student saw it, and beside it
    // what only a teacher is shown: the task's reference solution.
    let distilled = lines(&dir.path().join("distilled/train.jsonl"));
    assert_eq!(objective("distilled"), "distill");
    assert!(!distilled.is_empty());
    for record in &distilled {
        let teacher = record["teacher_context"].to_string();
        assert!(teacher.contains("echo solved"), "{record:#}");
        assert!(!record["messages"].to_string().contains("echo solved"));
    }

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
    let repo = repo();
    // seed: a training stage states its seed; the fake trainer draws nothing.
    let trained: String = ["sft", "preferred", "judged", "rewarded", "distilled"]
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
        online: None,
        build: splinter_core::build::BuildInfo::this_executable(),
    };
    let dir = tempfile::tempdir().unwrap();
    Factory::open(&resolved, dir.path(), services, "test")
        .unwrap()
        .run()
        .unwrap();
    assert_eq!(
        *trainer.objectives.lock().unwrap(),
        [
            "sft",
            "preference",
            "feedback",
            "policy-gradient",
            "distill"
        ]
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
    assert!(refused(
        "  - id: odd\n    uses: dataset.project\n    inputs: { attempts: [$stages.local.attempts] }\n    with: { objective: sft, reward: { penalties_percent: { loop: 50 } }, target: student, seed: 13 }\n"
    )
    .contains("reward options to another objective"));
    assert!(refused(
        "  - id: odd\n    uses: dataset.project\n    inputs: { attempts: [$stages.local.attempts] }\n    with: { objective: policy-gradient, reward: { penalties_percent: { dawdling: 50 } }, target: student, seed: 13 }\n"
    )
    .contains("not a behaviour metric"));
}

/// A pair is trained as two rows, so `max_block` bounds the longer of its continuations under the prompt:
/// a pair too long for the trainer's block is left out instead of sizing every row to it.
#[test]
fn a_preference_dataset_leaves_out_the_pairs_longer_than_the_block() {
    let run = |max_block: u64| {
        let repo = repo();
        // seed: a dataset states its seed; the fake backend's outcomes are scripted, so none draws.
        let bounded = RECIPE.replace(
            "objective: preference, target: student,",
            &format!("objective: preference, target: student, max_block: {max_block},"),
        );
        let resolved = resolve(&repo.root, &bounded, connections()).unwrap();
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
            online: None,
            build: splinter_core::build::BuildInfo::this_executable(),
        };
        let dir = tempfile::tempdir().unwrap();
        let outcome = Factory::open(&resolved, dir.path(), services, "test")
            .unwrap()
            .run();
        let pairs = dir.path().join("preferred/train.jsonl");
        (outcome.map(|_| lines(&pairs).len()), dir)
    };
    assert_eq!(run(1_000_000).0.unwrap(), 2);
    let (refused, _) = run(1);
    assert!(refused.is_err(), "no pair fits one token");
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
    let repo = repo();
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
        online: None,
        build: splinter_core::build::BuildInfo::this_executable(),
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

/// The projection recipe with an earlier text dataset replayed beside the
/// SFT dataset by a `model.train` stage `with` these settings.
fn replay_recipe(repo: &Repo, with: &str) -> String {
    // seed: a dataset stage states its seed; the text dataset's split is not what is tested.
    let recipe = RECIPE.to_owned()
        + "  - id: earlier
    uses: dataset.text
    inputs: { documents: $inputs.notes }
    with: { target: student, seed: 2 }
  - id: lora
    uses: model.train
    inputs: { dataset: $stages.sft.dataset, replay: [$stages.earlier.dataset] }
    with: "
        + with
        + "\n";
    repo.write(
        "notes/known.md",
        "# Known\n\nWhat the model already knew, rehearsed.\n",
    );
    recipe.replace(
        "  course: { kind: files, path: collections/vim.yaml }\n",
        "  course: { kind: files, path: collections/vim.yaml }\n  notes: { kind: files, path: notes }\n",
    )
}

/// Run `recipe` with every training task passing once of two, under `trainer`, in `dir`.
fn run_trained(
    repo: &Repo,
    recipe: &str,
    trainer: &crate::fakes::FakeTrainer,
    dir: &std::path::Path,
) {
    let resolved = resolve(&repo.root, recipe, connections()).unwrap();
    let backend = FakeBackend {
        vary_failures: true,
        ..FakeBackend::with(Outcome::Fail)
    };
    for task in ["train/001-first", "train/002-second"] {
        backend.script(task, &[Outcome::Pass, Outcome::Fail]);
    }
    let services = Services {
        backend: &backend,
        trainer: Some(trainer),
        models: None,
        variables: None,
        online: None,
        build: splinter_core::build::BuildInfo::this_executable(),
    };
    Factory::open(&resolved, dir, services, "test")
        .unwrap()
        .run()
        .unwrap();
}

/// The training records of a committed dataset view.
fn train_records(view: &std::path::Path) -> u64 {
    let dataset = splinter_factory::run::dataset::dataset_of(view).unwrap();
    lines(&view.join(dataset.train.path)).len() as u64
}

/// An epoch budget: the trainer is asked for `ceil(epochs * draws / records a step)`
/// steps, the draws an epoch being the dataset over its share of the draws when a
/// replay share is set and the union of dataset and replay when not; the budget is
/// recorded in the report, the training record and the candidate's identity.
#[test]
fn an_epoch_budget_is_resolved_over_the_records_drawn() {
    let repo = repo();
    // seed: a training stage states its seed; the fake trainer draws nothing.
    let shared = replay_recipe(
        &repo,
        "{ base: student, method: lora, params: { rank: 8, epochs: 3, grad_accum: 2 }, replay_share: 0.25, seed: 1 }",
    );
    let trainer = crate::fakes::FakeTrainer::default();
    let dir = tempfile::tempdir().unwrap();
    run_trained(&repo, &shared, &trainer, dir.path());
    let (dataset, replay) = (
        train_records(&dir.path().join("sft")),
        train_records(&dir.path().join("earlier")),
    );
    assert!(dataset > 0 && replay > 0, "{dataset} {replay}");
    let expected = (3.0 * dataset as f64 / 0.75 / 2.0).ceil() as u64;
    let trained = trainer.trained.lock().unwrap().clone();
    assert_eq!(trained.len(), 1);
    assert_eq!(
        trained[0]["steps"].as_u64(),
        Some(expected),
        "{}",
        trained[0]
    );
    assert!(trained[0].get("epochs").is_none(), "{}", trained[0]);
    let report: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("lora/report.json")).unwrap(),
    )
    .unwrap();
    let record: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("lora/training.json")).unwrap(),
    )
    .unwrap();
    for budget in [&report["budget"], &record["budget"]] {
        assert_eq!(budget["steps"].as_u64(), Some(expected), "{budget}");
        assert_eq!(budget["epochs"], 3.0, "{budget}");
        assert_eq!(
            budget["dataset_records"].as_u64(),
            Some(dataset),
            "{budget}"
        );
        assert_eq!(budget["replay_records"].as_u64(), Some(replay), "{budget}");
        assert_eq!(budget["records_per_step"], 2, "{budget}");
    }

    // The plain union: the same epochs over more draws is more steps, and another candidate.
    let union = shared.replace(", replay_share: 0.25", "");
    let trainer = crate::fakes::FakeTrainer::default();
    let other = tempfile::tempdir().unwrap();
    run_trained(&repo, &union, &trainer, other.path());
    let expected_union = (3.0 * (dataset + replay) as f64 / 2.0).ceil() as u64;
    assert_eq!(
        trainer.trained.lock().unwrap()[0]["steps"].as_u64(),
        Some(expected_union)
    );
    let identity = |root: &std::path::Path| -> Value {
        let candidate: Value = serde_json::from_str(
            &std::fs::read_to_string(root.join("lora/candidate.json")).unwrap(),
        )
        .unwrap();
        candidate["params"].clone()
    };
    assert_ne!(identity(dir.path()), identity(other.path()));

    // A budget is stated once: in epochs or in steps.
    let both = shared.replace("epochs: 3,", "epochs: 3, steps: 40,");
    let refused = resolve(&repo.root, &both, connections())
        .err()
        .map(|e| format!("{e:?}"))
        .unwrap_or_default();
    assert!(refused.contains("epochs and steps"), "{refused}");
}

/// Replay: a candidate trained on new records rehearses an earlier
/// dataset's beside them, at a stated share of the draws, so what it
/// learned before is kept.
#[test]
fn an_earlier_dataset_is_replayed_beside_the_new_records() {
    let repo = repo();
    // seed: a training stage states its seed; the fake trainer draws nothing.
    let recipe = replay_recipe(
        &repo,
        "{ base: student, method: lora, params: { rank: 8 }, replay_share: 0.25, seed: 1 }",
    );
    let trainer = crate::fakes::FakeTrainer::default();
    let dir = tempfile::tempdir().unwrap();
    run_trained(&repo, &recipe, &trainer, dir.path());
    assert_eq!(*trainer.replayed.lock().unwrap(), [(1, Some(0.25))]);

    // Replay rehearses chat records: a preference dataset trains without it.
    let refused = resolve(
        &repo.root,
        &recipe.replace(
            "dataset: $stages.sft.dataset, replay",
            "dataset: $stages.preferred.dataset, replay",
        ),
        connections(),
    )
    .err()
    .map(|e| format!("{e:?}"))
    .unwrap_or_default();
    assert!(refused.contains("replay"), "{refused}");
}

/// Opt-in: `sessions: merged` writes each attempt's session as one record
/// (every action supervised, its reward and group once) and says in
/// `dataset.json` how each session was written; a recipe that does not name
/// it keeps one record per action, and only a policy-gradient dataset takes
/// it. These attempts recorded no sampled tokens, so each session is packed
/// as text and counted as untraced.
#[test]
fn a_policy_gradient_dataset_merges_sessions_only_when_asked() {
    let repo = repo();
    let merged = RECIPE.to_owned()
        + "  - id: merged\n    uses: dataset.project\n    inputs: { attempts: [$stages.local.attempts] }\n    with: { objective: policy-gradient, sessions: merged, target: student, seed: 13 }\n";
    let resolved = resolve(&repo.root, &merged, connections()).unwrap();
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
        online: None,
        build: splinter_core::build::BuildInfo::this_executable(),
    };
    let dir = tempfile::tempdir().unwrap();
    Factory::open(&resolved, dir.path(), services, "test")
        .unwrap()
        .run()
        .unwrap();
    let per_action = lines(&dir.path().join("rewarded/train.jsonl"));
    let sessions = lines(&dir.path().join("merged/train.jsonl"));
    assert_eq!(sessions.len(), 4, "one record per attempt: {sessions:#?}");
    assert!(per_action.len() > sessions.len());
    let supervised = |records: &[Value]| -> usize {
        records
            .iter()
            .flat_map(|r| r["messages"].as_array().unwrap())
            .filter(|m| m["train"] == true)
            .count()
    };
    assert_eq!(supervised(&sessions), supervised(&per_action));
    let product = |stage: &str| -> Value {
        serde_json::from_str(
            &std::fs::read_to_string(dir.path().join(stage).join("dataset.json")).unwrap(),
        )
        .unwrap()
    };
    assert_eq!(product("merged")["sessions"]["untraced"], 4);
    assert_eq!(product("merged")["sessions"]["merged"], 0);
    assert!(product("rewarded").get("sessions").is_none());

    let refused = resolve(
        &repo.root,
        &(RECIPE.to_owned()
            + "  - id: odd\n    uses: dataset.project\n    inputs: { attempts: [$stages.local.attempts] }\n    with: { objective: sft, sessions: merged, target: student, seed: 13 }\n"),
        connections(),
    )
    .err()
    .map(|e| format!("{e:?}"))
    .unwrap_or_default();
    assert!(
        refused.contains("sessions: merged is a policy-gradient dataset's"),
        "{refused}"
    );
}
