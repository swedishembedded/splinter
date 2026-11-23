// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! A recipe's `imports`: the trials of Harbor jobs run elsewhere, read as a
//! pass of the run. They are graded by the one classifier, a trial of a task
//! the run does not select is refused with its reason, later passes and the
//! dataset read them like any pass, every record says which pass, agent and
//! model it came from (and which Harbor job an imported one), and a changed
//! job is a changed input.

use serde_json::Value;
use splinter_factory::resolve::resolve;
use splinter_factory::run::dataset::dataset_of;
use splinter_factory::run::rollout::trials_of;

use crate::factory_run::{resolved, run};
use crate::fakes::{harbor_trial, unfinished_harbor_trial, FakeBackend, FakeTrainer, Outcome};
use crate::repo::{connections, Repo};

// seed: a recipe must state every seed; the fake backend draws nothing from them.
const RECIPE: &str = r#"
schema: splinter.recipe/v2
name: vim-additions
inputs:
  course: { kind: files, path: collections/vim.yaml }
  earlier-job: { kind: files, path: jobs/earlier }
models:
  student: "brain:Qwen/Qwen3-4B"
  teacher: "hosted:deepseek/deepseek-v4.1-flash"
stages:
  - id: tasks
    uses: tasks.pin
    inputs: { collection: $inputs.course }
    with: { split: train }
  - id: earlier
    uses: attempt.import
    inputs: { tasks: $stages.tasks.tasks, jobs: [$inputs.earlier-job] }
    with: { model_label: deepseek/deepseek-v4.1-flash }
  - id: remote
    uses: attempt.collect
    inputs: { tasks: $stages.tasks.tasks, unsolved_by: [$stages.earlier.attempts] }
    with:
      agent: sven
      model: teacher
      environment: environments/vim.yaml
      environment_variables: { SVEN_KEY: "env:SVEN_KEY" }
      seed: 12
  - id: sft
    uses: dataset.project
    inputs: { attempts: [$stages.earlier.attempts, $stages.remote.attempts] }
    with: { target: student, seed: 13 }
"#;

fn records(view: &std::path::Path) -> Vec<Value> {
    let dataset = dataset_of(view).unwrap();
    [dataset.train, dataset.validation]
        .iter()
        .flat_map(|file| {
            std::fs::read_to_string(view.join(&file.path))
                .unwrap()
                .lines()
                .map(|l| serde_json::from_str::<Value>(l).unwrap())
                .collect::<Vec<_>>()
        })
        .collect()
}

#[test]
fn imported_harbor_trials_are_a_pass_that_datasets_and_later_passes_read_and_records_name_their_source(
) {
    let repo = Repo::new();
    let job = repo.root.join("jobs/earlier");
    harbor_trial(&job, "first__a", "org/train/001-first", 1.0);
    harbor_trial(&job, "first__b", "org/train/001-first", 1.0);
    harbor_trial(&job, "second__a", "org/train/002-second", 0.0);
    // Examination material and a trial Harbor never finished are refused.
    harbor_trial(&job, "held__a", "org/eval/001-held", 1.0);
    unfinished_harbor_trial(&job, "second__b");
    let recipe = resolved(&repo, RECIPE);
    let backend = FakeBackend::with(Outcome::Pass);
    let trainer = FakeTrainer::default();
    let dir = tempfile::tempdir().unwrap();

    let first = run(&recipe, dir.path(), &backend, &trainer).unwrap();
    assert!(first.iter().all(|(_, computed)| *computed), "{first:?}");

    // Nothing of an import is executed; its trials are graded as recorded.
    assert!(backend.executed().iter().all(|s| s.slot.pass != "earlier"));
    let imported = trials_of(&dir.path().join("earlier")).unwrap();
    let summary: Vec<(String, u32, Option<bool>)> = imported
        .iter()
        .map(|t| (t.task.clone(), t.slot.slot, t.grade.passed))
        .collect();
    assert_eq!(
        summary,
        [
            ("train/001-first".to_owned(), 0, Some(true)),
            ("train/001-first".to_owned(), 1, Some(true)),
            ("train/002-second".to_owned(), 0, Some(false)),
        ]
    );
    for trial in &imported {
        assert_eq!(trial.agent.name, "sven");
        assert_eq!(trial.model.as_deref(), Some("deepseek/deepseek-v4.1-flash"));
        let from = trial.imported.as_ref().expect("an imported trial says so");
        assert_eq!(from.harbor_job, "jobs/earlier");
    }
    let refused = std::fs::read_to_string(dir.path().join("earlier/refused.jsonl")).unwrap();
    let refused: Vec<Value> = refused
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let trials: Vec<&str> = refused
        .iter()
        .map(|r| r["trial"].as_str().unwrap())
        .collect();
    assert_eq!(trials, ["held__a", "second__b"]);
    assert!(refused
        .iter()
        .all(|r| r["harbor_job"] == "jobs/earlier" && !r["reason"].as_str().unwrap().is_empty()));

    // A later pass attempts only what the import left unsolved.
    let remote: Vec<String> = backend
        .executed()
        .into_iter()
        .filter(|s| s.slot.pass == "remote")
        .map(|s| s.task.id)
        .collect();
    assert_eq!(remote, ["train/002-second"]);

    // Every record names the pass, agent and model it came from; an imported
    // one also its Harbor job and trial.
    let records = records(&dir.path().join("sft"));
    assert_eq!(records.len(), 3);
    let mut sources: Vec<(String, String, Option<String>)> = records
        .iter()
        .map(|r| {
            let trial = &r["metadata"]["trial"];
            assert_eq!(trial["agent"], "sven");
            (
                trial["pass"].as_str().unwrap().to_owned(),
                trial["model"].as_str().unwrap().to_owned(),
                trial["imported"]["trial"].as_str().map(str::to_owned),
            )
        })
        .collect();
    sources.sort();
    assert_eq!(
        sources,
        [
            (
                "earlier".to_owned(),
                "deepseek/deepseek-v4.1-flash".to_owned(),
                Some("first__a".to_owned())
            ),
            (
                "earlier".to_owned(),
                "deepseek/deepseek-v4.1-flash".to_owned(),
                Some("first__b".to_owned())
            ),
            (
                "remote".to_owned(),
                "hosted:deepseek/deepseek-v4.1-flash".to_owned(),
                None
            ),
        ]
    );

    // A second run reuses everything; a changed job is a changed input.
    let second = run(&recipe, dir.path(), &backend, &trainer).unwrap();
    assert!(second.iter().all(|(_, computed)| !*computed), "{second:?}");
    harbor_trial(&job, "second__c", "org/train/002-second", 1.0);
    let changed = resolved(&repo, RECIPE);
    let refused = run(&changed, dir.path(), &backend, &trainer).unwrap_err();
    assert!(refused.to_string().contains("jobs/earlier"), "{refused}");
}

#[test]
fn an_import_is_checked_like_every_other_input() {
    let repo = Repo::new();
    let problems = |recipe: &str| match resolve(&repo.root, recipe, connections()) {
        Ok(_) => Vec::new(),
        Err(e) => e.0,
    };
    // The job must exist; ids must be among its tasks.
    let missing = problems(RECIPE);
    assert!(
        missing.iter().any(|p| p.contains("jobs/earlier")),
        "{missing:?}"
    );
    std::fs::create_dir_all(repo.root.join("jobs/earlier")).unwrap();
    assert_eq!(problems(RECIPE), Vec::<String>::new());
    let unknown = problems(&RECIPE.replace(
        "with: { model_label: deepseek/deepseek-v4.1-flash }",
        "with: { tasks: [train/404] }",
    ));
    assert!(
        unknown.iter().any(|p| p.contains("train/404")),
        "{unknown:?}"
    );

    // An import runs nothing: an agent or a seed is not one of its
    // parameters, and it imports at least one job.
    let agent = problems(&RECIPE.replace(
        "with: { model_label: deepseek/deepseek-v4.1-flash }",
        "with: { agent: sven }",
    ));
    assert!(agent.iter().any(|p| p.contains("agent")), "{agent:?}");
    let none = problems(&RECIPE.replace("jobs: [$inputs.earlier-job]", "jobs: []"));
    assert!(!none.is_empty(), "an import of no job is refused");
}
