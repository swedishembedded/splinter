// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the loop from a failed agent attempt to a
// verified corrected one, for its clients. If your team needs expertise in
// turning agent failures into training data you can procure our services by
// sending an email to info@swedishembedded.com.

//! A hinted pass: the model works each task from a copy whose instruction
//! carries the task's reference solution; a verified success becomes a
//! record whose user turn is the task's own instruction, so the model
//! learns to reach it unaided, and a trial whose own words repeat the
//! solution is excluded rather than taught.

use splinter_factory::resolve::resolve;
use splinter_factory::run::dataset::dataset_of;

use crate::factory_run::{resolved, run};
use crate::fakes::{FakeBackend, FakeTrainer, Outcome};
use crate::repo::{connections, Repo, RECIPE, TRAINING};

/// The recipe up to its dataset, with the second pass hinted.
fn hinted() -> String {
    RECIPE.split(TRAINING).next().unwrap().replace(
        "      environment_variables: { SVEN_KEY: \"env:SVEN_KEY\" }\n",
        "      environment_variables: { SVEN_KEY: \"env:SVEN_KEY\" }\n      hint: solution\n",
    )
}

#[test]
fn a_hinted_pass_teaches_the_task_without_its_hint_and_never_a_trial_that_repeats_it() {
    let repo = Repo::new();
    repo.write(
        "tasks/vim/train/001-first/solution/solve.sh",
        "#!/bin/sh\nprintf 'Hello world' > main.txt\n",
    );
    // A line the fixture trial's own reply says, word for word.
    repo.write(
        "tasks/vim/train/002-second/solution/solve.sh",
        "#!/bin/sh\n# enters Insert mode at the end of the current line\n",
    );
    let original =
        std::fs::read_to_string(repo.root.join("tasks/vim/train/001-first/instruction.md"))
            .unwrap();
    let backend = FakeBackend {
        echo_instruction: true,
        ..FakeBackend::with(Outcome::Fail)
    };
    backend.script(
        "train/001-first",
        &[Outcome::Fail, Outcome::Fail, Outcome::Pass, Outcome::Pass],
    );
    backend.script(
        "train/002-second",
        &[Outcome::Fail, Outcome::Fail, Outcome::Pass, Outcome::Pass],
    );
    let trainer = FakeTrainer::default();
    let dir = tempfile::tempdir().unwrap();
    run(&resolved(&repo, &hinted()), dir.path(), &backend, &trainer).unwrap();

    // The hinted pass worked from copies whose instruction carries the
    // solution; the task repository's own instruction is untouched.
    let instructions = backend.instructions.lock().unwrap().clone();
    let given: Vec<&String> = instructions
        .iter()
        .filter(|i| i.contains("printf 'Hello world'"))
        .collect();
    assert!(
        !given.is_empty() && given.iter().all(|i| i.starts_with(&original)),
        "{instructions:?}"
    );
    assert_eq!(
        std::fs::read_to_string(repo.root.join("tasks/vim/train/001-first/instruction.md"))
            .unwrap(),
        original
    );

    let dataset = dataset_of(&dir.path().join("sft")).unwrap();
    assert_eq!(
        dataset.exclusions.get("hint_repeated"),
        Some(&2),
        "{:?}",
        dataset.exclusions
    );
    let train = std::fs::read_to_string(dir.path().join("sft/train.jsonl")).unwrap();
    assert_eq!(train.lines().count(), 2, "both successes of the first task");
    assert!(train.contains(
        &serde_json::to_string(original.trim_end())
            .unwrap()
            .trim_matches('"')
            .to_owned()
    ));
    assert!(
        !train.contains("Hello world' > main.txt"),
        "the hint is in no record"
    );
    // Each record says what its target was produced with that it does not
    // show: the solution the hinted pass was given.
    let supervision = std::fs::read_to_string(dir.path().join("sft/supervision.jsonl")).unwrap();
    assert!(
        supervision.contains("\"withheld\":[\"hint:solution\"]"),
        "{supervision}"
    );
    for line in train.lines() {
        let record: serde_json::Value = serde_json::from_str(line).unwrap();
        assert_eq!(
            record["metadata"]["trial"]["withheld"],
            serde_json::json!(["hint:solution"]),
            "{record:#}"
        );
    }
}

#[test]
fn a_hinted_pass_is_never_played_and_needs_every_tasks_solution() {
    let repo = Repo::new();
    let played = hinted().replace(
        "      hint: solution\n",
        "      hint: solution\n      interactive: true\n",
    );
    let problems = resolve(&repo.root, &played, connections())
        .expect_err("refused")
        .0;
    assert!(
        problems
            .iter()
            .any(|p| p.contains("hint") && p.contains("interactive")),
        "{problems:?}"
    );

    let backend = FakeBackend::with(Outcome::Fail);
    let dir = tempfile::tempdir().unwrap();
    let error = run(
        &resolved(&repo, &hinted()),
        dir.path(),
        &backend,
        &FakeTrainer::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("solution/solve.sh"), "{error}");
}
