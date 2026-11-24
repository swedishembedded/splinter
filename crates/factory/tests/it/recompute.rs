// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements incremental experiment controllers that
// redo exactly the work a change invalidates, for its clients. If your team
// needs expertise in dependency-aware machine-learning pipelines, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec, by enumeration (plan 9C): for the reference recipe (pin, two
//! attempt passes, a dataset, two adapters, their evaluations, a release),
//! each input changed alone (a training task's instruction or verifier, the
//! held-out task's verifier, the environment, the agent's configuration, the
//! teacher, the student, the evaluation plan, a recipe parameter of one
//! stage, a file nothing reads) recomputes exactly what dependency-aware
//! incremental execution must:
//!
//! * a stage is computed again exactly when an input it reads itself
//!   changed, or a stage it consumes made another product than before;
//!   every other stage is reused;
//! * so a change reaches only the stages downstream of what read it, and a
//!   change nothing reads recomputes nothing.
//!
//! The stages a change reaches directly, and the edges of the graph, are
//! written here from the recipe and the repository's files, not read from the
//! planner.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use splinter_core::digest::Digest;
use splinter_factory::run::{fork, Factory, Services};

use crate::factory_run::{decisive, resolved};
use crate::fakes::{FakeBackend, FakeTrainer, Outcome};
use crate::repo::{Repo, RECIPE};

/// Which stage consumes which, from the recipe's `inputs`.
const EDGES: [(&str, &str); 11] = [
    ("tasks", "local"),
    ("tasks", "remote"),
    ("local", "remote"),
    ("local", "sft"),
    ("remote", "sft"),
    ("sft", "lora-r8"),
    ("sft", "lora-r16"),
    ("lora-r8", "eval-lora-r8"),
    ("lora-r16", "eval-lora-r16"),
    ("eval-lora-r8", "release"),
    ("eval-lora-r16", "release"),
];

/// One change: what it does to the repository or the recipe, and the
/// stages that read what it changes.
struct Change {
    name: &'static str,
    apply: fn(&Repo) -> String,
    reads: &'static [&'static str],
    /// The stages it must recompute, where the products it leads to are
    /// known (a change whose product effect the fakes decide is checked by
    /// the rule alone).
    exactly: Option<&'static [&'static str]>,
}

const EVALUATIONS: [&str; 2] = ["eval-lora-r8", "eval-lora-r16"];

fn changes() -> Vec<Change> {
    vec![
        Change {
            name: "a training task's instruction",
            apply: |r| {
                r.write("tasks/vim/train/001-first/instruction.md", "Do it twice.\n");
                RECIPE.into()
            },
            reads: &["tasks"],
            exactly: None,
        },
        Change {
            name: "a training task's verifier",
            apply: |r| {
                r.write("tasks/vim/train/002-second/tests/test.sh", "exit 1\n");
                RECIPE.into()
            },
            reads: &["tasks"],
            exactly: None,
        },
        Change {
            name: "the held-out task's verifier",
            apply: |r| {
                r.write("tasks/vim/eval/001-held/tests/test.sh", "test -f done\n");
                RECIPE.into()
            },
            reads: &EVALUATIONS,
            exactly: Some(&["eval-lora-r8", "eval-lora-r16", "release"]),
        },
        Change {
            name: "the environment",
            apply: |r| {
                r.write(
                    "environments/vim.yaml",
                    "schema: splinter.environment/v1\nsandbox: { provider: proot, resources: { timeout: 20m } }\nnetwork: { mode: open }\n",
                );
                RECIPE.into()
            },
            reads: &["local", "remote"],
            exactly: None,
        },
        Change {
            name: "the evaluation's own environment",
            apply: |r| {
                r.write(
                    "environments/eval.yaml",
                    "schema: splinter.environment/v1\nsandbox: { provider: proot, resources: { timeout: 30m } }\nnetwork: { mode: open }\n",
                );
                RECIPE.into()
            },
            reads: &EVALUATIONS,
            exactly: Some(&["eval-lora-r8", "eval-lora-r16", "release"]),
        },
        Change {
            name: "the local agent's configuration",
            apply: |r| {
                r.write("presets/sven.yaml", "agent: { max_turns: 40 }\n");
                RECIPE.into()
            },
            reads: &["local"],
            exactly: None,
        },
        Change {
            name: "the teacher model",
            apply: |_| RECIPE.replace("deepseek-v4.1-flash", "deepseek-v4.2"),
            reads: &["remote"],
            exactly: None,
        },
        Change {
            name: "the student model",
            apply: |_| RECIPE.replace("Qwen/Qwen3-4B", "Qwen/Qwen3-8B"),
            reads: &[
                "local",
                "sft",
                "lora-r8",
                "lora-r16",
                "eval-lora-r8",
                "eval-lora-r16",
            ],
            exactly: None,
        },
        Change {
            name: "the evaluation plan",
            apply: |r| {
                // seed: a plan must state its seed; the fake backend draws nothing from it.
                r.write(
                    "eval/dev.yaml",
                    "schema: splinter.evalplan/v1\ncollection: collections/vim.yaml\nsplit: eval\n\
                     environment: environments/eval.yaml\nattempts: 8\nseed: 5\ndecision: { alpha: 0.01 }\n",
                );
                RECIPE.into()
            },
            reads: &EVALUATIONS,
            exactly: Some(&["eval-lora-r8", "eval-lora-r16", "release"]),
        },
        Change {
            name: "one adapter's rank",
            apply: |_| RECIPE.replace("rank: 16", "rank: 32"),
            reads: &["lora-r16"],
            exactly: Some(&["lora-r16", "eval-lora-r16", "release"]),
        },
        Change {
            name: "one evaluation's comparisons",
            apply: |_| {
                RECIPE.replacen(
                    "candidate: $stages.lora-r8.model }\n    with: { plan: eval/dev.yaml, agent: sven, baseline: student, comparisons: 2 }",
                    "candidate: $stages.lora-r8.model }\n    with: { plan: eval/dev.yaml, agent: sven, baseline: student, comparisons: 3 }",
                    1,
                )
            },
            reads: &["eval-lora-r8"],
            exactly: Some(&["eval-lora-r8", "release"]),
        },
        Change {
            name: "a file nothing reads",
            apply: |r| {
                r.write("NOTES.md", "Nothing reads this.\n");
                RECIPE.into()
            },
            reads: &[],
            exactly: Some(&[]),
        },
    ]
}

/// The reference repository, its evaluation plan decisive and under an
/// environment of its own, so a change to either reaches only the
/// evaluations.
fn repository() -> Repo {
    let repo = Repo::new();
    decisive(&repo);
    repo.write(
        "environments/eval.yaml",
        "schema: splinter.environment/v1\nsandbox: { provider: proot, resources: { timeout: 10m } }\nnetwork: { mode: open }\n",
    );
    // seed: a plan must state its seed; the fake backend draws nothing from it.
    repo.write(
        "eval/dev.yaml",
        "schema: splinter.evalplan/v1\ncollection: collections/vim.yaml\nsplit: eval\n\
         environment: environments/eval.yaml\nattempts: 8\nseed: 5\ndecision: { alpha: 0.05 }\n",
    );
    repo
}

/// Each stage of a run of `recipe` in `dir`: its product and whether this
/// run computed it.
fn run(repo: &Repo, recipe: &str, dir: &Path) -> BTreeMap<String, (Digest, bool)> {
    let backend = FakeBackend {
        candidate: Some(Outcome::Pass),
        ..FakeBackend::with(Outcome::Pass)
    };
    let trainer = FakeTrainer::default();
    let services = Services {
        backend: &backend,
        trainer: Some(&trainer),
        models: None,
        variables: None,
        online: None,
    };
    let resolved = resolved(repo, recipe);
    let mut factory = Factory::open(&resolved, dir, services, "test-controller").unwrap();
    factory
        .run()
        .unwrap()
        .stages
        .into_iter()
        .map(|s| (s.key, (s.product, s.computed)))
        .collect()
}

/// The stages the rule says must be computed again: those `change` reaches
/// directly, and those consuming a stage whose product differs from
/// `before`'s.
fn must_recompute(
    change: &Change,
    before: &BTreeMap<String, (Digest, bool)>,
    after: &BTreeMap<String, (Digest, bool)>,
) -> BTreeSet<String> {
    after
        .keys()
        .filter(|stage| {
            change.reads.contains(&stage.as_str())
                || EDGES
                    .iter()
                    .filter(|(_, to)| to == stage)
                    .any(|(from, _)| before[*from].0 != after[*from].0)
        })
        .cloned()
        .collect()
}

#[test]
fn each_input_changed_alone_recomputes_exactly_what_depends_on_it() {
    let mut wrong = Vec::new();
    for change in changes() {
        let repo = repository();
        let dir = tempfile::tempdir().unwrap();
        let before = run(&repo, RECIPE, dir.path());
        assert!(before.values().all(|(_, computed)| *computed));
        let recipe = (change.apply)(&repo);
        // A run is locked to its inputs: the changed one is a fork of it,
        // holding the products it may reuse.
        let changed = tempfile::tempdir().unwrap();
        fork(dir.path(), changed.path()).unwrap();
        let after = run(&repo, &recipe, changed.path());
        assert_eq!(
            after.keys().collect::<Vec<_>>(),
            before.keys().collect::<Vec<_>>(),
            "{}: the same stages",
            change.name
        );
        let computed: BTreeSet<String> = after
            .iter()
            .filter(|(_, (_, c))| *c)
            .map(|(k, _)| k.clone())
            .collect();
        let rule = must_recompute(&change, &before, &after);
        if computed != rule {
            wrong.push(format!(
                "{}: computed {computed:?}, the rule says {rule:?}",
                change.name
            ));
        }
        if let Some(exactly) = change.exactly {
            let exactly: BTreeSet<String> = exactly.iter().map(|s| (*s).to_string()).collect();
            if computed != exactly {
                wrong.push(format!(
                    "{}: computed {computed:?}, expected exactly {exactly:?}",
                    change.name
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
