// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model release processes that promote a
// candidate only when it improves, generalises and forgets nothing, for its
// clients. If your team needs expertise in continual learning you can trust,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Promotion: a candidate replaces the champion only when it improves on
//! what it was trained for (`target`), carries that to tasks it was not
//! trained on (`transfer`) and keeps what it could already do
//! (`retention`). Each evaluation states its purpose; a promotion requires
//! all three unless its recipe narrows that in plain sight.

use serde_json::Value;
use splinter_factory::resolve::resolve;

use crate::factory_run::{resolved, run};
use crate::fakes::{FakeBackend, FakeTrainer, Outcome};
use crate::repo::{connections, Repo, EVALUATION, RECIPE};

// seed: a plan states its seed; the fake backend draws nothing from it.
const PLAN: &str = "schema: splinter.evalplan/v1\nenvironment: environments/vim.yaml\n\
                    attempts: 8\nseed: 5\ndecision: { alpha: 0.05 }\n";

fn repo() -> Repo {
    let repo = Repo::new();
    for (id, family) in [
        ("transfer/001-other", "other"),
        ("retention/001-known", "known"),
    ] {
        repo.write(
            &format!("tasks/vim/{id}/task.toml"),
            &format!("[task]\nname = \"org/{id}\"\n[metadata]\nbench_family = \"{family}\"\n"),
        );
        repo.write(
            &format!("tasks/vim/{id}/instruction.md"),
            &format!("Do {id}.\n"),
        );
        repo.write(&format!("tasks/vim/{id}/tests/test.sh"), "exit 0\n");
    }
    repo.write("eval/held.yaml", PLAN);
    repo
}

/// The recipe with one candidate evaluated for `purposes`, and a promotion
/// with `with`.
fn recipe(purposes: &[&str], with: &str) -> String {
    let head = &RECIPE[..RECIPE.find(EVALUATION).unwrap()];
    let mut out = head.replace(
        "  - id: lora-r16\n    uses: model.train\n    inputs: { dataset: $stages.sft.dataset }\n    with: { base: student, method: lora, params: { rank: 16 }, seed: 1 }\n",
        "",
    );
    let mut evaluations = Vec::new();
    for purpose in purposes {
        let split = match *purpose {
            "target" => "eval",
            other => other,
        };
        out.push_str(&format!(
            "  - {{ id: {purpose}-tasks, uses: tasks.pin, inputs: {{ collection: $inputs.course }}, with: {{ split: {split} }} }}\n\
             \x20 - id: eval-{purpose}\n    uses: evaluation\n    inputs: {{ candidate: $stages.lora-r8.model, tasks: $stages.{purpose}-tasks.tasks }}\n    \
             with: {{ plan: eval/held.yaml, agent: sven, baseline: student, purpose: {purpose} }}\n"
        ));
        evaluations.push(format!("$stages.eval-{purpose}.evaluation"));
    }
    out.push_str(&format!(
        "  - id: release\n    uses: promotion\n    inputs: {{ evaluations: [{}] }}\n    with: {with}\n",
        evaluations.join(", ")
    ));
    out
}

/// The release of a run where the candidate passes everything except the
/// tasks in `fails`, and the base fails everything except those.
fn released(repo: &Repo, recipe: &str, fails: &[&str]) -> Value {
    let mut backend = FakeBackend {
        candidate: Some(Outcome::Pass),
        ..FakeBackend::with(Outcome::Fail)
    };
    // The student's own attempts succeed, so there is a candidate to train.
    backend.script("train/001-first", &[Outcome::Pass, Outcome::Pass]);
    for task in fails {
        backend.candidate_on.insert((*task).into(), Outcome::Fail);
        backend.script(task, &[Outcome::Pass; 8]);
    }
    let trainer = FakeTrainer::default();
    let dir = tempfile::tempdir().unwrap();
    run(&resolved(repo, recipe), dir.path(), &backend, &trainer).unwrap();
    serde_json::from_slice(&std::fs::read(dir.path().join("release/release.json")).unwrap())
        .unwrap()
}

#[test]
fn a_candidate_is_promoted_only_when_it_improves_transfers_and_retains() {
    let repo = repo();
    let all = recipe(&["target", "transfer", "retention"], "{ promote: true }");
    let release = released(&repo, &all, &[]);
    assert_eq!(release["promoted"], "lora-r8", "{release:#}");

    // It forgot what the base could do: not promoted, and the release says
    // which requirement it failed.
    let release = released(&repo, &all, &["retention/001-known"]);
    assert_eq!(release["promoted"], Value::Null, "{release:#}");
    let gates = &release["candidates"][0]["gates"];
    assert_eq!(gates["retention"], "regressed", "{release:#}");
    assert_eq!(gates["target"], "improved", "{release:#}");
}

#[test]
fn a_promotion_without_transfer_and_retention_evaluations_is_refused_unless_narrowed() {
    let repo = repo();
    let refused = resolve(
        &repo.root,
        &recipe(&["target"], "{ promote: true }"),
        connections(),
    )
    .err()
    .map(|e| format!("{e:?}"))
    .unwrap_or_default();
    assert!(refused.contains("no transfer evaluation"), "{refused}");
    assert!(refused.contains("no retention evaluation"), "{refused}");

    let narrowed = recipe(&["target"], "{ promote: true, requires: [target] }");
    let release = released(&repo, &narrowed, &[]);
    assert_eq!(release["promoted"], "lora-r8", "{release:#}");
}
