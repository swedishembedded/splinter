// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements automated improvement of agent harnesses
// that adopts a change only on verified evidence, for its clients. If your
// team needs expertise in measuring whether a harness change really helped,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! `harness.search`: candidates drawn from a mutation space of a baseline
//! harness, each run on the tasks as an arm of its own and compared with the
//! baseline's arm slot by slot; the product is the harness that won (the
//! baseline when none verifiably beat it) with every candidate's evidence.

use serde_json::Value;

use crate::factory_run::{resolved, run};
use crate::fakes::{FakeBackend, FakeTrainer, Outcome};
use crate::repo::Repo;

const PLAIN: &str = "{{ inputs.task }}";
const MENDING: &str = "MEND {{ inputs.task }}";

fn harness(prompt: &str) -> String {
    format!(
        "api_version: \"2\"\nname: solver\ninputs: {{ task: string }}\nagent: {{ profile: coding }}\n\
         execution:\n  limits: {{ model_turns: 6, tool_calls: 6 }}\nworkflow:\n  - name: work\n    task: \"{prompt}\"\n"
    )
}

// seed: the search draws its candidates from its seed; the fake backend draws nothing.
const RECIPE: &str = "schema: splinter.recipe/v2
name: searched
inputs:
  course: { kind: files, path: collections/vim.yaml }
models:
  student: \"brain:Qwen/Qwen3-4B\"
stages:
  - { id: tasks, uses: tasks.pin, inputs: { collection: $inputs.course }, with: { split: train } }
  - id: search
    uses: harness.search
    inputs: { tasks: $stages.tasks.tasks }
    with: { agent: sven, model: student, environment: environments/vim.yaml, harness: harness/baseline.yaml,
            space: harness/space.yaml, candidates: 2, attempts: 8, seed: 9 }
";

fn searched(mending: Outcome) -> (tempfile::TempDir, FakeBackend) {
    let repo = Repo::new();
    repo.write("harness/baseline.yaml", &harness(PLAIN));
    repo.write(
        "harness/space.yaml",
        &format!(
            "knobs:\n  - path: [workflow, '0', task]\n    choices: [\"{PLAIN}\", \"{MENDING}\"]\n"
        ),
    );
    let backend = FakeBackend {
        harness_says: vec![("MEND".into(), mending)],
        ..FakeBackend::with(Outcome::Fail)
    };
    let dir = tempfile::tempdir().unwrap();
    run(
        &resolved(&repo, RECIPE),
        dir.path(),
        &backend,
        &FakeTrainer::default(),
    )
    .unwrap();
    (dir, backend)
}

fn search(dir: &std::path::Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(dir.join("search/search.json")).unwrap()).unwrap()
}

#[test]
fn a_candidate_that_verifiably_beats_the_baseline_is_the_harness_found() {
    let (dir, backend) = searched(Outcome::Pass);
    let found = std::fs::read_to_string(dir.path().join("search/harness.yaml")).unwrap();
    assert!(found.contains("MEND"), "{found}");
    let search = search(dir.path());
    assert_eq!(search["winner"]["decision"], "improved", "{search:#}");
    // The baseline arm and each candidate's ran once, the tasks' attempts
    // paired slot by slot.
    let candidates = search["candidates"].as_array().unwrap();
    assert_eq!(candidates.len(), 2, "{search:#}");
    let arms = 1 + candidates
        .iter()
        .map(|c| c["harness"].as_str().unwrap())
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    assert_eq!(backend.executed().len(), arms * 2 * 8);
}

#[test]
fn without_a_verified_gain_the_baseline_stays_the_harness() {
    let (dir, _) = searched(Outcome::Fail);
    let found = std::fs::read_to_string(dir.path().join("search/harness.yaml")).unwrap();
    assert_eq!(found, harness(PLAIN));
    assert_eq!(search(dir.path())["winner"], Value::Null);
}

/// Picking the best of N candidates each tested at the search's alpha would
/// find a "winner" in most searches of N useless candidates: each is tested
/// at alpha / N (Bonferroni, valid however the candidates depend on each
/// other, as a hill climb's do), so the chance that any useless candidate
/// wins stays within alpha.
#[test]
fn each_candidate_is_tested_at_the_searchs_alpha_split_over_the_candidates() {
    let (dir, _) = searched(Outcome::Pass);
    let search = search(dir.path());
    assert_eq!(search["alpha"], 0.05, "{search:#}");
    assert_eq!(search["alpha_per_candidate"], 0.025, "{search:#}");
    for candidate in search["candidates"].as_array().unwrap() {
        if candidate.get("statistics").is_some() {
            assert_eq!(candidate["statistics"]["alpha"], 0.025, "{candidate:#}");
        }
    }
}
