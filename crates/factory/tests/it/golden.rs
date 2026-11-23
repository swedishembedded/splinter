// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The factory's observable products for the reference recipe, pinned before
//! the recipe language is generalised: whatever the recipe looks like, the
//! same tasks, agents, trials and seeds must give the same dataset records,
//! exclusions, evaluation decisions and release. Stage identities are not
//! pinned here; they may change with the recipe language.

use serde_json::{json, Value};
use splinter_factory::run::dataset::dataset_of;
use splinter_factory::run::eval::result_of;

use crate::factory_run::{decisive, resolved, run};
use crate::fakes::{FakeBackend, FakeTrainer, Outcome};
use crate::repo::{Repo, RECIPE};

/// What a run of the reference recipe gave, as a user would read it.
pub(crate) fn observed(dir: &std::path::Path) -> Value {
    let dataset = dataset_of(&dir.join("datasets/sft")).unwrap();
    let train = std::fs::read_to_string(dir.join("datasets/sft/train.jsonl")).unwrap();
    let mut evals = serde_json::Map::new();
    for candidate in ["lora-r8", "lora-r16"] {
        let result = result_of(&dir.join(format!("eval/{candidate}"))).unwrap();
        evals.insert(
            candidate.into(),
            json!({
                "decision": result.decision,
                "baseline": result.baseline.as_ref().map(|a| (a.trials, a.counted, a.passed)),
                "candidate": result.candidate.as_ref().map(|a| (a.trials, a.counted, a.passed)),
                "statistics": result.statistics,
            }),
        );
    }
    let release: Value =
        serde_json::from_slice(&std::fs::read(dir.join("release/release.json")).unwrap()).unwrap();
    json!({
        "records": splinter_core::digest::Digest::of(train.as_bytes()),
        "waterfall": dataset.waterfall,
        "exclusions": dataset.exclusions,
        "evaluations": evals,
        "release": { "promote": release["promote"], "promoted": release["promoted"] },
    })
}

/// The reference run: the recipe every factory test starts from, on the
/// scripted backend and trainer.
pub(crate) fn reference(dir: &std::path::Path) -> Value {
    let repo = Repo::new();
    decisive(&repo);
    let backend = FakeBackend {
        candidate: Some(Outcome::Pass),
        ..FakeBackend::with(Outcome::Fail)
    };
    backend.script("train/001-first", &[Outcome::Pass, Outcome::Pass]);
    run(
        &resolved(&repo, RECIPE),
        dir,
        &backend,
        &FakeTrainer::default(),
    )
    .unwrap();
    observed(dir)
}

fn golden_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/it/fixtures/golden-products.json")
}

#[test]
fn the_reference_recipe_gives_the_pinned_products() {
    let dir = tempfile::tempdir().unwrap();
    let found = reference(dir.path());
    let pinned: Value = serde_json::from_slice(&std::fs::read(golden_path()).unwrap()).unwrap();
    assert_eq!(
        found, pinned,
        "the reference products changed; bless an intended change with \
         `cargo test -p splinter-factory --test it golden::bless -- --ignored`"
    );
}

#[test]
#[ignore = "rewrites the golden record; run only to bless an intended change"]
fn bless() {
    let dir = tempfile::tempdir().unwrap();
    let found = reference(dir.path());
    std::fs::write(
        golden_path(),
        serde_json::to_string_pretty(&found).unwrap() + "\n",
    )
    .unwrap();
}
