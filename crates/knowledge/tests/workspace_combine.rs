// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements executable evaluation of code changes
// against a project's own test suite, for its clients. If your team needs
// expertise in test-driven benchmarks for code-editing agents, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: admitted single-fault tasks of one snapshot are combined into
//! harder tasks that hold several independent faults.
//!
//! * a combined task takes faults in different files; its tree is the clean
//!   snapshot with every fault applied and none of the hidden tests, its
//!   check the union of the members' commands over the union of their hidden
//!   files, its reference change the union of theirs, and its request every
//!   member's symptoms and the file of none;
//! * its family is the sorted member families joined, and the members are
//!   recorded;
//! * it is admitted only after the combined check fails, the reference makes
//!   it pass on every run, and no single member's reference makes it pass;
//! * a candidate that is not admitted is counted under a reason, and so is a
//!   single task that cannot be combined at all;
//! * the same seed and tasks give the same combined tasks whatever order the
//!   tasks are named in;
//! * a combined task is exported as any workspace task is.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use std::sync::Arc;

use common::{git, write, Repo, AT, CHANGED_LINES};
use splinter_core::annotation::Outcome;
use splinter_core::clock::FixedClock;
use splinter_core::experience::{
    Environment, Experience, Privileged, PrivilegedKind, Provenance, Task,
};
use splinter_core::rng::Seed;
use splinter_eval::verifiers::workspace::{grade, WorkspaceCheck, WorkspaceVerifier};
use splinter_eval::verifiers::Verifier;
use splinter_knowledge::tasks::workspace::{
    Combination, CombinePlan, CombineReason, Combined, Held, RepositoryProfile, WorkspaceError,
    WorkspaceGenerator, WorkspacePlan,
};
use splinter_sandbox::{Limits, Patch, ProcessSandbox};

/// A third module and its test, so that three faults can lie in three files.
const ROUNDING: &str = "def to_cents(amount):\n    cents = amount * 100\n    return int(cents)\n";
const TEST_ROUNDING: &str = "import unittest\n\nfrom shop.rounding import to_cents\n\n\nclass RoundingTest(unittest.TestCase):\n    def test_whole_amounts_become_cents(self):\n        self.assertEqual(to_cents(3), 300)\n";
const THIRD_MODULE: [(&str, &str); 2] = [
    ("shop/rounding.py", ROUNDING),
    ("tests/test_rounding.py", TEST_ROUNDING),
];

struct Rig {
    repo: Repo,
    generator: WorkspaceGenerator,
}

impl Rig {
    fn new(name: &str, extra: &[(&str, &str)]) -> Self {
        let repo = Repo::new(name, extra);
        let calls = repo.scratch.0.join("calls");
        fs::create_dir_all(&calls).unwrap();
        let sandbox = ProcessSandbox::new(&calls, [], &[], Limits::default());
        let generator = WorkspaceGenerator::new(
            Arc::new(sandbox),
            repo.host.clone(),
            repo.store.clone(),
            repo.scratch.0.join("work"),
        )
        .with_plan(WorkspacePlan {
            mutants_per_file: 6,
            ..WorkspacePlan::default()
        })
        .unwrap();
        Self { repo, generator }
    }

    /// The single-fault tasks admitted for the repository's `HEAD`.
    fn singles(&self) -> Vec<Task> {
        let tree = self.repo.capture("HEAD");
        let laid = self.repo.scratch.0.join("laid");
        let _ = fs::remove_dir_all(&laid);
        let _ = fs::remove_dir_all(self.repo.scratch.0.join("laid.git"));
        self.generator
            .environment(&tree.snapshot)
            .materialize(&tree.archive_path, &laid)
            .unwrap();
        let profile = RepositoryProfile::detect(&laid).unwrap();
        let qualified = self.generator.qualify(&profile, &tree.snapshot).unwrap();
        let generation = self
            .generator
            .from_mutations(
                &tree.snapshot,
                &qualified,
                &Held::none(),
                &FixedClock::new(AT),
            )
            .unwrap();
        generation.admitted.into_iter().map(|t| t.task).collect()
    }

    fn combine(&self, singles: &[Task], faults: usize, per_family: usize) -> Combination {
        self.try_combine(singles, faults, per_family).unwrap()
    }

    fn try_combine(
        &self,
        singles: &[Task],
        faults: usize,
        per_family: usize,
    ) -> Result<Combination, WorkspaceError> {
        let plan = CombinePlan {
            faults,
            per_family,
            max_tasks: 16,
            // seed: the specs that compare two runs need the seed to be the same; the rest do not depend on it
            seed: Seed::new(11),
        };
        self.generator.combine(singles, &plan, &FixedClock::new(AT))
    }

    /// Whether the combined task's check passes on the tree the solver is
    /// given, with the hidden files laid over it.
    fn passes_unchanged(&self, combined: &Combined) -> bool {
        let env = self.generator.environment(&combined.task.snapshot);
        let check = check_of(&combined.task.task);
        let nothing = Patch {
            diff: Vec::new(),
            files: Vec::new(),
        };
        grade(
            &env,
            &self.repo.store,
            &self.repo.scratch.0.join("grade"),
            &check,
            &nothing,
        )
        .unwrap()
        .passed()
    }

    /// The verdict the workspace verifier gives `patch` on `task`.
    fn verdict(&self, combined: &Combined, patch: &str) -> Outcome {
        let env = self.generator.environment(&combined.task.snapshot);
        let verifier = WorkspaceVerifier::new(
            vec![env],
            Arc::new(self.repo.store.clone()),
            self.repo.scratch.0.join("verify"),
        );
        let experience = Experience::answered_without_a_run(
            combined.task.task.clone(),
            patch,
            Provenance::new("test", &FixedClock::new(AT)),
        )
        .unwrap();
        verifier
            .verify(&combined.task.task, &experience)
            .unwrap()
            .outcome
    }
}

fn check_of(task: &Task) -> WorkspaceCheck {
    WorkspaceCheck::of_task(task).unwrap().unwrap().1
}

fn reference_of(task: &Task) -> String {
    task.privileged
        .iter()
        .find(|p| p.kind == PrivilegedKind::Reference)
        .unwrap()
        .content
        .clone()
}

/// The symptom lines of a request.
fn symptoms(task: &Task) -> BTreeSet<String> {
    task.instruction
        .lines()
        .filter(|line| line.starts_with("- ") && !line.contains(" more checks "))
        .map(str::to_string)
        .collect()
}

fn family_of(task: &Task) -> &str {
    &task.concepts[0]
}

/// One single task of each family in `families`.
fn one_of_each<'a>(singles: &'a [Task], families: &[&str]) -> Vec<&'a Task> {
    families
        .iter()
        .map(|family| {
            singles
                .iter()
                .find(|t| family_of(t) == *family)
                .unwrap_or_else(|| panic!("no single task of {family}"))
        })
        .collect()
}

const PRICING: &str = "shop/pricing.py";
const STATUS: &str = "shop/status.py";

#[test]
fn a_combined_task_holds_every_fault_every_symptom_and_every_hidden_file_and_names_its_members() {
    let rig = Rig::new("combine-admit", &[]);
    let singles = rig.singles();
    let combination = rig.combine(&singles, 2, 1);
    assert_eq!(combination.admitted.len(), 1, "{:?}", combination.refused);
    let combined = &combination.admitted[0];
    let task = &combined.task.task;

    // Its family is the sorted member families joined; the members are the
    // single tasks it was made of, one of each file.
    assert_eq!(family_of(task), "shop/pricing.py + shop/status.py");
    assert_eq!(combined.task.family, family_of(task));
    assert_eq!(task.concepts.len(), 1);
    assert_eq!(combined.members.len(), 2);
    let members: Vec<&Task> = combined
        .members
        .iter()
        .map(|id| singles.iter().find(|t| &t.task.id == id).unwrap())
        .collect();
    let member_families: BTreeSet<&str> = members.iter().map(|t| family_of(t)).collect();
    assert_eq!(member_families, BTreeSet::from([PRICING, STATUS]));

    // Its check is the union of the members' commands and hidden files.
    let check = check_of(task);
    let checks: Vec<WorkspaceCheck> = members.iter().map(|t| check_of(t)).collect();
    assert_eq!(
        check.fail_to_pass.len(),
        checks.iter().map(|c| c.fail_to_pass.len()).sum::<usize>()
    );
    for member in &checks {
        for call in member.fail_to_pass.iter().chain(&member.pass_to_pass) {
            assert!(
                check.fail_to_pass.contains(call) || check.pass_to_pass.contains(call),
                "{call:?} is not in the combined check"
            );
        }
        for hidden in &member.hidden_paths {
            assert!(
                check.hidden_paths.contains(hidden),
                "{hidden} is not hidden"
            );
        }
    }
    assert!(check.hidden.is_some());

    // Its reference change is the union of the members'.
    let reference = reference_of(task);
    for member in &members {
        assert!(reference.contains(reference_of(member).trim_end()));
    }

    // Its request states every symptom, that there are several independent
    // faults, and names no file and no changed line.
    let stated = symptoms(task);
    for member in &members {
        assert!(symptoms(member).is_subset(&stated), "{}", task.instruction);
    }
    assert!(
        task.instruction.contains("independent"),
        "{}",
        task.instruction
    );
    assert!(!task.instruction.contains("shop/"), "{}", task.instruction);
    for line in CHANGED_LINES {
        assert!(
            !task.instruction.contains(line),
            "{line:?} in {}",
            task.instruction
        );
    }

    // Its tree holds both faults and none of the hidden tests.
    let solver = rig.repo.scratch.0.join("solver");
    rig.repo.solver_tree(&combined.task, &solver);
    assert_ne!(read(&solver, PRICING), common::PRICING);
    assert_ne!(read(&solver, STATUS), common::STATUS);
    for hidden in &check.hidden_paths {
        assert!(
            !solver.join(hidden).exists(),
            "{hidden} is in the solver's tree"
        );
    }

    // The reference makes the check pass; neither member's alone does.
    assert_eq!(rig.verdict(combined, &reference), Outcome::Pass);
    for member in &members {
        assert_ne!(
            rig.verdict(combined, &reference_of(member)),
            Outcome::Pass,
            "one fault fixed is not the task solved"
        );
    }
    // The tree as it is fails the check.
    assert!(!rig.passes_unchanged(combined));
}

fn read(root: &Path, file: &str) -> String {
    fs::read_to_string(root.join(file)).unwrap()
}

#[test]
fn a_third_file_makes_three_faults_and_each_file_takes_part_in_at_most_the_asked_number_of_tasks() {
    let rig = Rig::new("combine-three", &THIRD_MODULE);
    let singles = rig.singles();
    let combination = rig.combine(&singles, 3, 2);
    assert!(
        !combination.admitted.is_empty(),
        "{:?}",
        combination.refused
    );
    let mut taking_part: BTreeMap<String, usize> = BTreeMap::new();
    for combined in &combination.admitted {
        assert_eq!(combined.members.len(), 3);
        assert_eq!(
            family_of(&combined.task.task),
            "shop/pricing.py + shop/rounding.py + shop/status.py"
        );
        for family in splinter_core::family::members(&combined.task.family) {
            *taking_part.entry(family.to_string()).or_default() += 1;
        }
    }
    assert!(taking_part.values().all(|n| *n <= 2), "{taking_part:?}");
    // Every file is covered.
    assert_eq!(taking_part.len(), 3);
}

#[test]
fn the_same_seed_and_tasks_give_the_same_combined_tasks_whatever_order_they_are_named_in() {
    let rig = Rig::new("combine-seeded", &THIRD_MODULE);
    let singles = rig.singles();
    let ids = |c: &Combination| -> Vec<_> {
        c.admitted
            .iter()
            .map(|t| t.task.task.task.id.clone())
            .collect()
    };
    let first = rig.combine(&singles, 2, 2);
    let mut reversed = singles.clone();
    reversed.reverse();
    let again = rig.combine(&reversed, 2, 2);
    assert!(!first.admitted.is_empty());
    assert_eq!(ids(&first), ids(&again));
}

#[test]
fn a_task_that_cannot_be_combined_is_counted_by_reason_and_the_rest_still_are() {
    let rig = Rig::new("combine-unusable", &[]);
    let mut singles = rig.singles();
    let closed_book = Task::new(
        "recall",
        Vec::new(),
        Environment::closed_book(),
        "What is two plus two?",
        Vec::new(),
    )
    .unwrap();
    let no_reference = {
        let kept: Vec<Privileged> = singles[0]
            .privileged
            .iter()
            .filter(|p| p.kind != PrivilegedKind::Reference)
            .cloned()
            .collect();
        Task::new(
            singles[0].task.kind.clone(),
            Vec::new(),
            singles[0].environment.clone(),
            singles[0].instruction.clone(),
            kept,
        )
        .unwrap()
        .with_concepts([family_of(&singles[0]).to_string()])
        .unwrap()
    };
    singles.push(closed_book);
    singles.push(no_reference);

    let combination = rig.combine(&singles, 2, 1);
    assert_eq!(combination.admitted.len(), 1, "{:?}", combination.refused);
    assert_eq!(
        combination.unusable.get(&CombineReason::NotAWorkspaceTask),
        Some(&1),
        "{:?}",
        combination.unusable
    );
    assert_eq!(
        combination.unusable.get(&CombineReason::NoReference),
        Some(&1),
        "{:?}",
        combination.unusable
    );
}

#[test]
fn fewer_families_than_faults_is_refused_by_name_not_reported_as_nothing_admitted() {
    let rig = Rig::new("combine-few", &[]);
    let singles = rig.singles();
    let pricing: Vec<Task> = singles
        .into_iter()
        .filter(|t| family_of(t) == PRICING)
        .collect();
    let error = rig.try_combine(&pricing, 2, 1).unwrap_err().to_string();
    assert!(error.contains("families"), "{error}");
    let error = rig.try_combine(&pricing, 1, 1).unwrap_err().to_string();
    assert!(error.contains("faults"), "{error}");
}

#[test]
fn faults_of_two_snapshots_are_not_combined_and_the_refusal_is_counted() {
    let rig = Rig::new("combine-snapshots", &[]);
    let first = rig.singles();
    // A later commit changes a file the tests do not cover, so the tasks of
    // the new snapshot are the old ones over a different baseline.
    write(
        &rig.repo.repo.join("shop/status.py"),
        &format!("{}\n# later\n", common::STATUS),
    );
    git(&rig.repo.repo, &["add", "-A"]);
    git(&rig.repo.repo, &["commit", "-q", "-m", "later"]);
    let second = rig.singles();
    let mixed: Vec<Task> = one_of_each(&first, &[PRICING])
        .into_iter()
        .chain(one_of_each(&second, &[STATUS]))
        .cloned()
        .collect();

    let combination = rig.combine(&mixed, 2, 1);
    assert!(combination.admitted.is_empty());
    assert!(
        combination
            .refused
            .get(&CombineReason::DifferentSnapshots)
            .is_some_and(|n| *n >= 1),
        "{:?}",
        combination.refused
    );
}

#[test]
fn faults_whose_references_change_the_same_file_are_not_combined() {
    let rig = Rig::new("combine-overlap", &[]);
    let singles = rig.singles();
    let status = one_of_each(&singles, &[STATUS])[0];
    // The same file under another family's name: two references that change
    // one file do not make two independent faults.
    let renamed = Task::new(
        status.task.kind.clone(),
        Vec::new(),
        status.environment.clone(),
        status.instruction.clone(),
        status.privileged.clone(),
    )
    .unwrap()
    .with_concepts(["shop/renamed.py".to_string()])
    .unwrap();
    let pair = vec![status.clone(), renamed];

    let combination = rig.combine(&pair, 2, 1);
    assert!(combination.admitted.is_empty());
    assert_eq!(
        combination.refused.get(&CombineReason::SharedFiles),
        Some(&1),
        "{:?}",
        combination.refused
    );
}

#[test]
fn a_combined_task_is_exported_as_a_workspace_task_with_every_hidden_file_and_command() {
    let rig = Rig::new("combine-harbor", &[]);
    let singles = rig.singles();
    let combination = rig.combine(&singles, 2, 1);
    let combined = &combination.admitted[0];
    let task = &combined.task.task;
    let check = check_of(task);
    let out = rig.repo.scratch.0.join("exported");

    rig.generator
        .export_harbor(task, &out, "ubuntu:24.04")
        .unwrap();

    for file in [
        "instruction.md",
        "task.toml",
        "environment/Dockerfile",
        "tests/test.sh",
        "solution/solve.sh",
        "solution/reference.patch",
    ] {
        assert!(out.join(file).is_file(), "{file} is missing");
    }
    // The workspace holds both faults and no hidden test; the overlay holds
    // every hidden file.
    assert_ne!(
        read(&out.join("environment/workspace"), PRICING),
        common::PRICING
    );
    assert_ne!(
        read(&out.join("environment/workspace"), STATUS),
        common::STATUS
    );
    for hidden in &check.hidden_paths {
        assert!(out.join("tests/overlay").join(hidden).is_file(), "{hidden}");
        assert!(
            !out.join("environment/workspace").join(hidden).exists(),
            "{hidden}"
        );
    }
    let patch = read(&out, "solution/reference.patch");
    assert!(patch.contains(PRICING) && patch.contains(STATUS), "{patch}");
    // The metadata carries the composite family and every command.
    let config: toml::Table = read(&out, "task.toml").parse().unwrap();
    let meta = &config["metadata"]["splinter"];
    assert_eq!(meta["family"].as_str(), Some(family_of(task)));
    assert_eq!(
        meta["fail_to_pass"].as_array().unwrap().len(),
        check.fail_to_pass.len()
    );
    let verifier = read(&out, "tests/test.sh");
    for call in check.fail_to_pass.iter().chain(&check.pass_to_pass) {
        assert!(verifier.contains(&call.argv.join("' '")), "{verifier}");
    }
}
