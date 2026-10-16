// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements executable evaluation of code changes
// against a project's own test suite, for its clients. If your team needs
// expertise in test-driven benchmarks for code-editing agents, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: a git revision is captured as a snapshot; a repository's tests are
//! qualified twice before tasks are made from it; a mutant becomes a task
//! only when the project's own tests catch it, reproducibly, and the reverse
//! patch makes the check pass; the hidden tests are not in the solver's
//! tree; the request names symptoms and never the change; a second run over
//! the same snapshot adds nothing; an authored fixture is held to the same
//! bar.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::collections::BTreeMap;
use std::fs;
use std::ops::Deref;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use common::{git, write, Repo, Scratch, AT, CHANGED_LINES};
use splinter_core::annotation::Outcome;
use splinter_core::clock::FixedClock;
use splinter_core::digest::Digest;
use splinter_core::experience::{Experience, Provenance, Task};
use splinter_core::kinds::WORKSPACE_CHANGE;
use splinter_eval::verifiers::workspace::{WorkspaceCheck, WorkspaceVerifier};
use splinter_eval::verifiers::Verifier;
use splinter_knowledge::capture::CapturedTree;
use splinter_knowledge::tasks::workspace::{
    Fixture, Held, Qualified, Reason, RepositoryProfile, WorkspaceError, WorkspaceGeneration,
    WorkspaceGenerator, WorkspacePlan, WorkspaceTask, MIN_HIDDEN_KILL_RATE,
};
use splinter_knowledge::tasks::{Catalogue, SolverEnvironment, VerifierKind};
use splinter_sandbox::{CommandCall, Limits, ProcessSandbox};

/// A project and the generator that makes tasks from its snapshots.
struct Rig {
    repo: Repo,
    generator: WorkspaceGenerator,
}

impl Deref for Rig {
    type Target = Repo;

    fn deref(&self) -> &Repo {
        &self.repo
    }
}

impl Rig {
    fn new(name: &str, extra: &[(&str, &str)]) -> Self {
        Self::with_mutants(name, extra, 6)
    }

    /// A rig that considers `mutants_per_file` mutants of each file.
    fn with_mutants(name: &str, extra: &[(&str, &str)], mutants_per_file: usize) -> Self {
        Self::over(Repo::new(name, extra), mutants_per_file, &[])
    }

    /// A rig over the Rust crate [`RUST_CRATE`]; its sandbox reaches the
    /// toolchain through the caller's `PATH`, `CARGO_HOME` and `RUSTUP_HOME`.
    fn rust(name: &str) -> Self {
        Self::over(
            Repo::of_files(name, &RUST_CRATE),
            6,
            &["PATH", "CARGO_HOME", "RUSTUP_HOME"],
        )
    }

    fn over(repo: Repo, mutants_per_file: usize, pass_through: &[&str]) -> Self {
        let calls = repo.scratch.0.join("calls");
        fs::create_dir_all(&calls).unwrap();
        let sandbox =
            ProcessSandbox::new(&calls, std::env::vars(), pass_through, Limits::default());
        let generator = WorkspaceGenerator::new(
            Arc::new(sandbox),
            repo.host.clone(),
            repo.store.clone(),
            repo.scratch.0.join("work"),
        )
        .with_plan(WorkspacePlan {
            mutants_per_file,
            ..WorkspacePlan::default()
        })
        .unwrap();
        Self { repo, generator }
    }

    /// The profile the rules give for the project, with `flaky` named in
    /// its quarantine, qualified on `tree`.
    fn qualified(&self, tree: &CapturedTree, flaky: &[&str]) -> Result<Qualified, WorkspaceError> {
        let laid = self.scratch.0.join("laid");
        let _ = fs::remove_dir_all(&laid);
        let _ = fs::remove_dir_all(self.scratch.0.join("laid.git"));
        self.generator
            .environment(&tree.snapshot)
            .materialize(&tree.archive_path, &laid)
            .unwrap();
        let mut profile = RepositoryProfile::detect(&laid).unwrap();
        profile.flaky = flaky.iter().map(|id| (*id).to_string()).collect();
        self.generator.qualify(&profile, &tree.snapshot)
    }

    fn generate(&self, tree: &CapturedTree, held: &Held) -> WorkspaceGeneration {
        let qualified = self.qualified(tree, &[]).unwrap();
        self.generator
            .from_mutations(&tree.snapshot, &qualified, held, &FixedClock::new(AT))
            .unwrap()
    }
}

/// A crate with no dependencies whose tests are integration tests, so a
/// fault in `src/lib.rs` is caught by tests that are not in the file.
const RUST_CRATE: [(&str, &str); 5] = [
    (
        "Cargo.toml",
        "[package]\nname = \"shop\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    ),
    (
        "Cargo.lock",
        "version = 3\n\n[[package]]\nname = \"shop\"\nversion = \"0.1.0\"\n",
    ),
    (
        "src/lib.rs",
        "pub fn total(prices: &[u32]) -> u32 {\n    let mut sum = 0;\n    for price in prices {\n        sum += price;\n    }\n    sum\n}\n\npub fn discount(price: u32, percent: u32) -> Result<u32, String> {\n    if percent > 100 {\n        return Err(\"percent over 100\".to_string());\n    }\n    Ok(price - price * percent / 100)\n}\n",
    ),
    (
        "tests/total.rs",
        "use shop::total;\n\n#[test]\nfn empty_basket_costs_nothing() {\n    assert_eq!(total(&[]), 0);\n}\n\n#[test]\nfn prices_are_added_up() {\n    assert_eq!(total(&[2, 3]), 5);\n}\n",
    ),
    (
        "tests/discount.rs",
        "use shop::discount;\n\n#[test]\nfn ten_percent_off() {\n    assert_eq!(discount(200, 10), Ok(180));\n}\n\n#[test]\nfn over_hundred_percent_is_refused() {\n    assert!(discount(10, 101).is_err());\n}\n",
    ),
];

/// The ids the runner reports for [`RUST_CRATE`]'s tests: the test binary's
/// source file, then the test's name.
const RUST_TESTS: [&str; 4] = [
    "tests/discount.rs::over_hundred_percent_is_refused",
    "tests/discount.rs::ten_percent_off",
    "tests/total.rs::empty_basket_costs_nothing",
    "tests/total.rs::prices_are_added_up",
];

fn check_of(task: &Task) -> WorkspaceCheck {
    let item = task
        .privileged
        .iter()
        .find(|p| format!("{:?}", p.kind).contains("workspace-check"))
        .unwrap();
    serde_json::from_str(&item.content).unwrap()
}

#[test]
fn the_profile_is_detected_by_rule_and_the_baseline_is_qualified_twice() {
    let rig = Rig::new("profile", &[]);
    let tree = rig.capture("HEAD");
    let qualified = rig.qualified(&tree, &[]).unwrap();
    let profile = &qualified.profile;
    assert_eq!(profile.test_command, ["python3", "-m", "unittest"]);
    assert_eq!(
        profile.files_under_test,
        ["shop/pricing.py", "shop/status.py"]
    );
    assert!(profile.baseline_secs.is_some());
    assert_eq!(qualified.tests.len(), 5);

    // A cargo project is recognised too.
    let cargo = rig.scratch.0.join("cargo");
    write(&cargo.join("Cargo.toml"), "[package]\nname = \"x\"\n");
    write(&cargo.join("src/lib.rs"), "pub fn f() {}\n");
    let detected = RepositoryProfile::detect(&cargo).unwrap();
    assert_eq!(detected.test_command, ["cargo", "test", "--offline"]);
    assert_eq!(detected.files_under_test, ["src/lib.rs"]);
}

#[test]
fn a_red_baseline_is_refused_with_a_count() {
    let red = "import unittest\n\n\nclass Red(unittest.TestCase):\n    def test_broken(self):\n        self.fail('no')\n";
    let rig = Rig::new("red", &[("tests/test_red.py", red)]);
    let tree = rig.capture("HEAD");
    match rig.qualified(&tree, &[]) {
        Err(WorkspaceError::BaselineRefused { reason, count, .. }) => {
            assert_eq!(reason, Reason::RedBaseline);
            assert_eq!(count, 1);
        }
        other => panic!("expected a red baseline, got {other:?}"),
    }
}

/// A test that passes on the first run and fails on the second, whatever
/// the tree: its state is a counter file outside it.
fn flaky_test(counter: &Path) -> String {
    format!(
        "import os\nimport unittest\n\nCOUNTER = {counter:?}\n\n\nclass Flaky(unittest.TestCase):\n    \
         def test_sometimes_fails(self):\n        n = int(open(COUNTER).read()) if os.path.exists(COUNTER) else 0\n        \
         open(COUNTER, 'w').write(str(n + 1))\n        self.assertEqual(n % 2, 0)\n"
    )
}

#[test]
fn a_flaky_test_refuses_the_baseline_and_once_quarantined_is_never_a_target() {
    let scratch = Scratch::new("flaky-state");
    let counter = scratch.0.join("count");
    let text = flaky_test(&counter);
    let rig = Rig::new("flaky", &[("tests/test_flaky.py", &text)]);
    let tree = rig.capture("HEAD");
    match rig.qualified(&tree, &[]) {
        Err(WorkspaceError::BaselineRefused { reason, count, .. }) => {
            assert_eq!(reason, Reason::FlakyBaseline);
            assert_eq!(count, 1);
        }
        other => panic!("expected a flaky baseline, got {other:?}"),
    }

    let flaky = "tests.test_flaky.Flaky.test_sometimes_fails";
    let qualified = rig.qualified(&tree, &[flaky]).unwrap();
    assert!(!qualified.tests.contains(flaky));
    let generation = rig
        .generator
        .from_mutations(
            &tree.snapshot,
            &qualified,
            &Held::none(),
            &FixedClock::new(AT),
        )
        .unwrap();
    assert!(!generation.admitted.is_empty());
    for admitted in &generation.admitted {
        let check = check_of(&admitted.task);
        let text = serde_json::to_string(&check).unwrap();
        assert!(!text.contains("test_flaky"), "{text}");
        assert!(!check.hidden_paths.iter().any(|p| p.contains("flaky")));
    }
}

/// The reverse patch makes the check pass under the workspace verifier.
fn reverse_patch_passes(rig: &Rig, admitted: &WorkspaceTask) -> bool {
    let env = rig.generator.environment(&admitted.snapshot);
    let reference = admitted
        .task
        .privileged
        .iter()
        .find(|p| format!("{:?}", p.kind).contains("Reference"))
        .unwrap();
    let verifier = WorkspaceVerifier::new(
        vec![env],
        Arc::new(rig.store.clone()),
        rig.scratch.0.join("verify"),
    );
    let experience = Experience::answered_without_a_run(
        admitted.task.clone(),
        &reference.content,
        Provenance::new("test", &FixedClock::new(AT)),
    )
    .unwrap();
    verifier
        .verify(&admitted.task, &experience)
        .unwrap()
        .outcome
        == Outcome::Pass
}

/// Runs the task's failing commands on the tree the solver is given, with
/// the hidden files laid over it, as a verifier would an empty change.
fn baseline_fails(rig: &Rig, admitted: &WorkspaceTask) -> bool {
    let check = check_of(&admitted.task);
    let env = rig.generator.environment(&admitted.snapshot);
    let tree = rig
        .scratch
        .0
        .join(format!("baseline-{}", &admitted.key.hex()[..12]));
    let _ = fs::remove_dir_all(&tree);
    env.materialize(
        &rig.generator.archive_of(&admitted.snapshot).unwrap(),
        &tree,
    )
    .unwrap();
    if let Some(hidden) = &check.hidden {
        let status = Command::new("tar")
            .args(["--extract", "--file"])
            .arg(rig.store.path(hidden).unwrap())
            .arg("--directory")
            .arg(&tree)
            .status()
            .unwrap();
        assert!(status.success());
    }
    check.fail_to_pass.iter().all(|call| {
        let mut argv = vec!["env".to_string(), "PYTHONDONTWRITEBYTECODE=1".to_string()];
        argv.extend(call.argv.clone());
        let result = env.run(&tree, &CommandCall { argv, stdin: None }).unwrap();
        result.exit_code != Some(0)
    })
}

#[test]
fn every_admitted_task_fails_on_the_baseline_passes_with_the_reverse_patch_and_hides_its_tests() {
    let rig = Rig::new("admit", &[]);
    let tree = rig.capture("HEAD");
    let generation = rig.generate(&tree, &Held::none());
    assert!(
        generation.admitted.len() >= 2,
        "admitted {}: refused {:?}",
        generation.admitted.len(),
        generation.refused
    );
    for admitted in &generation.admitted {
        let task = &admitted.task;
        assert_eq!(task.task.kind, WORKSPACE_CHANGE);
        assert_eq!(task.environment.kind, "workspace");
        assert_eq!(task.concepts, std::slice::from_ref(&admitted.family));
        assert!(admitted.family.starts_with("shop/"), "{}", admitted.family);

        assert!(baseline_fails(&rig, admitted), "{}", task.instruction);
        assert!(reverse_patch_passes(&rig, admitted), "{}", task.instruction);

        // The tree the solver is given holds no hidden path.
        let check = check_of(task);
        assert!(!check.hidden_paths.is_empty());
        let laid = rig
            .scratch
            .0
            .join(format!("solver-{}", &admitted.key.hex()[..12]));
        let _ = fs::remove_dir_all(&laid);
        rig.generator
            .environment(&admitted.snapshot)
            .materialize(
                &rig.generator.archive_of(&admitted.snapshot).unwrap(),
                &laid,
            )
            .unwrap();
        for hidden in &check.hidden_paths {
            assert!(
                !laid.join(hidden).exists(),
                "{hidden} is in the solver's tree"
            );
        }
        // ... and the hidden tests are protected from edits, as are the
        // tests that stay.
        assert!(check
            .protected
            .iter()
            .all(|p| !check.hidden_paths.contains(p)));

        // The measurement is recorded on the task: the hidden tests caught at
        // least the floor of the faults seeded near this one, or there was
        // none to measure (never a zero standing for "not measured").
        if let Some(rate) = check.hidden_strength.and_then(|s| s.kill_rate()) {
            assert!(
                rate >= MIN_HIDDEN_KILL_RATE,
                "{rate} for {}",
                task.instruction
            );
        }

        // The request states a symptom and none of the changed lines.
        let instruction = &task.instruction;
        for line in CHANGED_LINES {
            assert!(!instruction.contains(line), "{line:?} in {instruction}");
        }
        assert!(!instruction.contains("Traceback"));
        assert!(instruction.contains("python3 -m unittest"));
    }
    // A killed mutant of `total` is described by its tests' names.
    assert!(generation.admitted.iter().any(|t| {
        t.family == "shop/pricing.py"
            && (t.task.instruction.contains("prices are added up")
                || t.task.instruction.contains("empty basket costs nothing")
                || t.task.instruction.contains("ten percent off")
                || t.task
                    .instruction
                    .contains("over hundred percent is refused"))
    }));
}

#[test]
fn a_mutant_no_test_can_tell_from_the_original_is_refused_with_a_reason() {
    let rig = Rig::new("equivalent", &[]);
    let tree = rig.capture("HEAD");
    let generation = rig.generate(&tree, &Held::none());
    // `verbose` in status.py is never read: changing it changes nothing.
    assert!(
        generation.count(Reason::EquivalentMutant) >= 1,
        "refused {:?}",
        generation.refused
    );
    assert!(generation
        .admitted
        .iter()
        .all(|t| !t.task.instruction.contains("verbose")));
}

#[test]
fn a_second_run_over_the_same_snapshot_adds_no_task() {
    let rig = Rig::new("again", &[]);
    let tree = rig.capture("HEAD");
    let first = rig.generate(&tree, &Held::none());
    assert!(!first.admitted.is_empty(), "refused {:?}", first.refused);

    // Generation is deterministic: the same tasks, by address.
    let repeated = rig.generate(&tree, &Held::none());
    let ids = |g: &WorkspaceGeneration| -> Vec<Digest> {
        g.admitted.iter().map(|t| t.task.task.id.clone()).collect()
    };
    assert_eq!(ids(&first), ids(&repeated));

    // Told which keys are held, it admits none of them again.
    let held: Held = first.admitted.iter().collect();
    assert_eq!(held.len(), first.admitted.len(), "keys are distinct");
    let second = rig.generate(&tree, &held);
    assert!(second.admitted.is_empty(), "{:?}", ids(&second));
    assert_eq!(second.count(Reason::AlreadyHeld), held.len());
}

/// A rendered fixture: a repository with one seeded bug and an acceptance
/// script kept outside it.
fn rendered(dir: &Path, request: &str) -> String {
    let repo = dir.join("repo");
    write(&repo.join("pkg/__init__.py"), "from .core import *\n");
    write(
        &repo.join("pkg/core.py"),
        "def double(x):\n    return x + x + 1\n",
    );
    write(
        &repo.join("tests/test_core.py"),
        "import unittest\n\nfrom pkg.core import double\n\n\nclass Core(unittest.TestCase):\n    def test_basic(self):\n        self.assertIsInstance(double(2), int)\n",
    );
    git(&repo, &["init", "-q", "-b", "main"]);
    git(&repo, &["add", "-A"]);
    git(&repo, &["commit", "-q", "-m", "pkg"]);
    write(
        &repo.join("pkg/core.py"),
        "def double(x):\n    return x + x\n",
    );
    let reference = git(&repo, &["diff"]);
    git(&repo, &["checkout", "-q", "--", "."]);
    write(
        &dir.join("accept.py"),
        "import sys\n\nsys.path.insert(0, \".\")\nfrom pkg.core import double\n\nsys.exit(0 if double(2) == 4 else 1)\n",
    );
    write(&dir.join("task.txt"), request);
    reference
}

#[test]
fn an_authored_fixture_is_admitted_only_when_it_fails_as_described() {
    let rig = Rig::new("fixture", &[]);
    let dir = rig.scratch.0.join("rendered");
    let reference = rendered(
        &dir,
        "The function pkg.double does not double its argument. Find the bug and fix it.\n",
    );
    let fixture = Fixture::rendered(&dir, "double")
        .unwrap()
        .with_reference(reference);
    let clock = FixedClock::new(AT);

    let admitted = rig
        .generator
        .import_fixture(&fixture, &Held::none(), &clock)
        .unwrap();
    assert_eq!(admitted.admitted.len(), 1, "{:?}", admitted.refused);
    let task = &admitted.admitted[0];
    assert_eq!(task.family, "double");
    assert!(baseline_fails(&rig, task));
    assert!(reverse_patch_passes(&rig, task));
    assert!(!check_of(&task.task).hidden_paths.is_empty());

    // The same fixture again is held, not admitted twice.
    let held: Held = admitted.admitted.iter().collect();
    let again = rig
        .generator
        .import_fixture(&fixture, &held, &clock)
        .unwrap();
    assert!(again.admitted.is_empty());
    assert_eq!(again.count(Reason::AlreadyHeld), 1);

    // A check that already passes is not a task.
    let mut passing = fixture.clone();
    passing.fail_to_pass = vec![CommandCall {
        argv: ["python3", "-c", "pass"].map(String::from).to_vec(),
        stdin: None,
    }];
    let refused = rig
        .generator
        .import_fixture(&passing, &Held::none(), &clock)
        .unwrap();
    assert!(refused.admitted.is_empty());
    assert_eq!(refused.count(Reason::NotFailingAsDescribed), 1);

    // A request that quotes the hidden script is not self-contained.
    let mut quoting = fixture.clone();
    quoting.instruction = "Make this pass: sys.exit(0 if double(2) == 4 else 1) and sys.path.insert(0, \".\") then from pkg.core import double".into();
    let refused = rig
        .generator
        .import_fixture(&quoting, &Held::none(), &clock)
        .unwrap();
    assert_eq!(refused.count(Reason::NotSelfContained), 1);
}

#[test]
fn a_task_family_layout_is_read_for_its_request_its_writable_paths_and_its_hidden_world() {
    let rig = Rig::new("family", &[]);
    let dir = rig.scratch.0.join("family");
    write(
        &dir.join("family.toml"),
        "id = \"retries\"\nrequest = \"\"\"\nEnable 3 retry attempts for the upload service.\n\"\"\"\n\n[workspace]\ntemplate = \"workspace\"\nwritable = [\"config/**\"]\n",
    );
    write(&dir.join("workspace/config/app.json"), "{\"retries\": 0}\n");
    write(
        &dir.join("workspace/src/upload.py"),
        "RETRIES_KEY = 'retries'\n",
    );
    write(
        &dir.join("world/check.py"),
        "import json, sys\n\nsys.exit(0 if json.load(open('config/app.json'))['retries'] == 3 else 1)\n",
    );
    let check = |argv: &[&str]| CommandCall {
        argv: argv.iter().map(|a| (*a).to_string()).collect(),
        stdin: None,
    };
    let fixture =
        Fixture::family_layout(&dir, vec![check(&["python3", "world/check.py"])], vec![]).unwrap();
    assert_eq!(fixture.family, "retries");
    assert_eq!(
        fixture.instruction,
        "Enable 3 retry attempts for the upload service."
    );
    assert_eq!(fixture.protected, ["src/upload.py"]);
    assert_eq!(
        fixture.hidden.keys().collect::<Vec<_>>(),
        ["world/check.py"]
    );
    let generation = rig
        .generator
        .import_fixture(&fixture, &Held::none(), &FixedClock::new(AT))
        .unwrap();
    assert_eq!(generation.admitted.len(), 1, "{:?}", generation.refused);
    assert!(baseline_fails(&rig, &generation.admitted[0]));
}

#[test]
fn the_catalogue_names_the_workspace_kind_and_never_asks_a_model_for_it() {
    let catalogue = Catalogue::builtin();
    let kind = catalogue.get(WORKSPACE_CHANGE).unwrap();
    assert_eq!(kind.verifiers, [VerifierKind::Workspace]);
    assert_eq!(kind.environment, SolverEnvironment::Workspace);
    assert!(!kind.model_proposable());
    assert!(kind.validate().is_ok());
    assert!(!kind.needs_judge());
}

/// Two functions on neighbouring lines, each looked at by its own test file.
/// A fault in one is caught by its file alone, so the hidden file (the one
/// that catches it) would let a plausible wrong patch of the other function
/// through: every fault seeded beside it is caught by the visible file only.
const LEVEL: &str = "\
def double(x):
    return x * 2
def bump(x):
    return x + 1
";

const TEST_DOUBLE: &str = "\
import unittest

from shop.level import double


class DoubleTest(unittest.TestCase):
    def test_doubling(self):
        self.assertEqual(double(3), 6)
";

const TEST_BUMP: &str = "\
import unittest

from shop.level import bump


class BumpTest(unittest.TestCase):
    def test_bumping(self):
        self.assertEqual(bump(3), 4)
";

#[test]
fn a_task_whose_hidden_tests_miss_the_faults_seeded_beside_it_is_refused_with_a_reason() {
    let rig = Rig::with_mutants(
        "weak",
        &[
            ("shop/level.py", LEVEL),
            ("tests/test_double.py", TEST_DOUBLE),
            ("tests/test_bump.py", TEST_BUMP),
        ],
        16,
    );
    let tree = rig.capture("HEAD");
    let generation = rig.generate(&tree, &Held::none());
    assert!(
        generation.count(Reason::WeakHiddenTests) >= 1,
        "refused {:?}",
        generation.refused
    );
    assert!(
        generation
            .admitted
            .iter()
            .all(|t| t.family != "shop/level.py"),
        "a weak task was admitted"
    );
    // The project whose one test file covers all its faults still yields tasks.
    assert!(
        generation
            .admitted
            .iter()
            .any(|t| t.family == "shop/pricing.py"),
        "refused {:?}",
        generation.refused
    );
}

const SPIN: &str = "def count(n):\n    i = 0\n    while i < n:\n        i += 1\n    return i\n";
const TEST_SPIN: &str = "import unittest\n\nfrom shop.spin import count\n\n\nclass Spin(unittest.TestCase):\n    def test_count(self):\n        self.assertEqual(count(3), 3)\n";

#[test]
fn a_mutant_that_loops_is_stopped_at_the_time_budget_and_not_at_the_sandbox_limit() {
    // The sandbox lets a command run two minutes; the budget is two seconds.
    // The rest of the run (a few dozen mutants) takes about a minute, so a
    // looping mutant that ran to the sandbox's limit would show.
    let mut rig = Rig::with_mutants(
        "spin",
        &[("shop/spin.py", SPIN), ("tests/test_spin.py", TEST_SPIN)],
        16,
    );
    let calls = rig.scratch.0.join("slow-calls");
    fs::create_dir_all(&calls).unwrap();
    let limits = Limits {
        wall_time_ms: 120_000,
        ..Limits::default()
    };
    rig.generator = WorkspaceGenerator::new(
        Arc::new(ProcessSandbox::new(&calls, BTreeMap::new(), &[], limits)),
        rig.host.clone(),
        rig.store.clone(),
        rig.scratch.0.join("slow-work"),
    )
    .with_plan(WorkspacePlan {
        mutants_per_file: 16,
        min_time_budget: std::time::Duration::from_secs(2),
        ..WorkspacePlan::default()
    })
    .unwrap();
    let tree = rig.capture("HEAD");
    let began = std::time::Instant::now();
    let generation = rig.generate(&tree, &Held::none());
    assert!(
        generation.count(Reason::OverTimeBudget) >= 1,
        "refused {:?}",
        generation.refused
    );
    assert!(
        began.elapsed() < std::time::Duration::from_secs(110),
        "took {:?}",
        began.elapsed()
    );
}

#[test]
fn a_cargo_crate_is_read_per_test_and_a_seeded_fault_is_caught_by_exactly_the_named_tests() {
    let rig = Rig::rust("cargo-tasks");
    let tree = rig.capture("HEAD");
    let qualified = rig.qualified(&tree, &[]).unwrap();
    assert_eq!(
        qualified.profile.test_command,
        ["cargo", "test", "--offline"]
    );
    assert_eq!(qualified.tests.iter().collect::<Vec<_>>(), RUST_TESTS);

    let generation = rig.generate(&tree, &Held::none());
    assert!(
        !generation.admitted.is_empty(),
        "refused {:?}",
        generation.refused
    );
    for admitted in &generation.admitted {
        let check = check_of(&admitted.task);
        let call = &check.fail_to_pass[0];
        let at = call.argv.iter().position(|a| a == "--exact").unwrap();
        let named: Vec<&str> = call.argv[at + 1..].iter().map(String::as_str).collect();
        assert!(!named.is_empty());
        for name in &named {
            // The runner is told the test's name alone; the symptom the
            // request gives is that name in words.
            assert!(RUST_TESTS
                .iter()
                .any(|id| id.ends_with(&format!("::{name}"))));
            assert!(
                admitted.task.instruction.contains(&name.replace('_', " ")),
                "{name} in {}",
                admitted.task.instruction
            );
        }
        assert!(baseline_fails(&rig, admitted));
        assert!(reverse_patch_passes(&rig, admitted));
    }
}
