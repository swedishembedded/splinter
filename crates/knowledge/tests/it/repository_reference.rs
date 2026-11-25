// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements repository training tasks whose faults,
// symptoms and fixes are checked against the projects' own test runners, for
// its clients. If your team needs expertise in building bug-fixing
// environments for coding agents, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Spec (4B, `repository-mutants` against SWE-smith-style construction): each
//! instance's project is run by the standard library's own runner
//! (`python3 -m unittest`, never the plugin's runner or verifier) as the
//! original project and as the mutant the solver is given with its hidden
//! tests laid back. As SWE-smith admits a task, the tests that fail on the
//! mutant and pass on the original (FAIL_TO_PASS) are not empty and are
//! exactly the ones the symptoms name; every other test passes on both
//! (PASS_TO_PASS); and the mutant changes nothing but the files the reference
//! patch restores. The plugin's verifier agrees: the workspace as given earns
//! nothing, the reference patch and the original files earn the full reward.
//!
//! And `repository-bugs` against its own catalogue: every one of its sixteen
//! faults, injected alone, shows as its report says by an independent reading
//! of the report evaluated in Python on the mutant library and not on the
//! original; the hidden suite, run by `unittest`, passes on the original and
//! fails on the mutant only in the faulty module's tests; and the verifier
//! rewards the reference fix, never the mutant.

#![allow(clippy::unwrap_used, clippy::expect_used)] // a panic is the failure report

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use splinter_core::verifiability::Split;
use splinter_knowledge::factory::artifact::Instance;
use splinter_knowledge::factory::folder::Folder;
use splinter_knowledge::factory::host::Host;
use splinter_knowledge::factory::plugin::{InstanceRequest, Plugin};
use splinter_knowledge::factory::plugins::repository_bugs::RepositoryBugs;
use splinter_knowledge::factory::plugins::repository_mutants::RepositoryMutants;

fn tasks_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tasks")
}

/// Each test's outcome by its unittest id (`module.Class.test`), as
/// `python3 -m unittest discover -v` reports it run in `dir`.
fn outcomes(dir: &Path) -> BTreeMap<String, bool> {
    discovered(dir, ".")
}

/// The outcomes of the tests discovered under `start`, run in `dir` (which
/// is on the import path, as `-m` puts the working directory there).
fn discovered(dir: &Path, start: &str) -> BTreeMap<String, bool> {
    let out = Command::new("python3")
        .args([
            "-B", "-m", "unittest", "discover", "-v", "-s", start, "-t", start,
        ])
        .env_remove("PYTHONPATH")
        .current_dir(dir)
        .output()
        .unwrap();
    let report = String::from_utf8_lossy(&out.stderr);
    let mut outcomes = BTreeMap::new();
    for line in report.lines() {
        // `test_name (module.Class.test_name) ... ok`; a docstring, when a
        // test has one, takes the next line and the verdict follows it.
        let Some((head, verdict)) = line.rsplit_once(" ... ") else {
            continue;
        };
        let Some(id) = head
            .split_once(" (")
            .and_then(|(_, rest)| rest.strip_suffix(')'))
        else {
            continue;
        };
        outcomes.insert(id.to_string(), verdict.trim() == "ok");
    }
    assert!(
        !outcomes.is_empty(),
        "no test ran in {}:\n{report}",
        dir.display()
    );
    outcomes
}

/// The test names the instruction's symptoms give.
fn symptom_names(instance: &Instance) -> BTreeSet<String> {
    let (_, symptoms) = instance.instruction().split_once("Symptoms:\n").unwrap();
    symptoms
        .lines()
        .take_while(|l| l.starts_with("- `"))
        .map(|l| l[3..].split('`').next().unwrap().to_string())
        .collect()
}

fn repository_of(instance: &Instance) -> String {
    instance
        .instruction()
        .split('`')
        .nth(3)
        .unwrap()
        .to_string()
}

/// The project the solver is given, its hidden tests laid back over it.
fn mutant_project(instance: &Instance, dir: &Path) {
    let mut project = instance.workspace();
    for (path, entry) in instance.folder().under("tests/overlay") {
        project
            .put(
                &path["tests/overlay/".len()..],
                entry.bytes.clone(),
                entry.executable,
            )
            .unwrap();
    }
    project.write_to(dir).unwrap();
}

fn reward_of(
    instance: &Instance,
    patch: impl FnOnce(&splinter_knowledge::factory::host::Session),
) -> Option<f64> {
    let session = Host::default().open(instance).unwrap();
    patch(&session);
    session.verify().unwrap().reward()
}

#[test]
fn each_instance_is_a_swe_smith_task_by_the_projects_own_runner() {
    let plugin = RepositoryMutants::new(tasks_root(), Host::default());
    let mut repositories = BTreeSet::new();
    for (split, seed, bugs) in [
        (Split::Train, 3, 1),
        (Split::Train, 8, 2),
        (Split::Train, 21, 3),
        (Split::Dev, 5, 1),
        (Split::Dev, 13, 2),
    ] {
        // seed: any, several of each split and fault count
        let instance = plugin
            .generate(&InstanceRequest {
                split,
                seed,
                difficulty: BTreeMap::from([("bugs".to_string(), bugs), ("detail".to_string(), 3)]),
            })
            .unwrap();
        let repository = repository_of(&instance);
        repositories.insert(repository.clone());
        let original_dir = tasks_root().join(&repository);
        let original = outcomes(&original_dir);
        let mutant_dir = tempfile::tempdir().unwrap();
        mutant_project(&instance, mutant_dir.path());
        let mutant = outcomes(mutant_dir.path());

        let context = format!("{repository} (seed {seed}, {bugs} bugs)");
        assert_eq!(
            original.keys().collect::<Vec<_>>(),
            mutant.keys().collect::<Vec<_>>(),
            "{context}: the same tests run on both"
        );
        assert!(
            original.values().all(|&ok| ok),
            "{context}: the original passes"
        );
        let fail_to_pass: BTreeSet<String> = mutant
            .iter()
            .filter(|(_, &ok)| !ok)
            .map(|(id, _)| id.rsplit('.').next().unwrap().to_string())
            .collect();
        assert!(!fail_to_pass.is_empty(), "{context}: a mutant no test sees");
        assert_eq!(
            fail_to_pass,
            symptom_names(&instance),
            "{context}: the symptoms are FAIL_TO_PASS"
        );

        // The mutant differs from the original only in files the reference
        // patch restores, as many as there are faults.
        let original_tree = Folder::read_from(&original_dir).unwrap();
        let mutant_tree = Folder::read_from(mutant_dir.path()).unwrap();
        let changed: Vec<&str> = original_tree
            .paths()
            .filter(|p| original_tree.text(p) != mutant_tree.text(p))
            .collect();
        assert_eq!(changed.len(), bugs as usize, "{context}: {changed:?}");
        let solve = instance.folder().text("solution/solve.sh").unwrap();
        for path in &changed {
            assert!(
                solve.contains(&format!("/{path}\"")),
                "{context}: {path} is not restored"
            );
        }

        assert_eq!(
            reward_of(&instance, |_| ()),
            Some(0.0),
            "{context}: as given"
        );
        assert_eq!(
            reward_of(&instance, |s| {
                assert!(s.solve().unwrap().succeeded());
            }),
            Some(1.0),
            "{context}: the reference patch"
        );
        assert_eq!(
            reward_of(&instance, |s| {
                for path in &changed {
                    s.write(path, original_tree.text(path).unwrap().as_bytes())
                        .unwrap();
                }
            }),
            Some(1.0),
            "{context}: the original files"
        );
    }
    assert!(repositories.len() >= 3, "{repositories:?}");
}

/// Each fault's report, by the call it starts with, read independently: the
/// module it is in, and a Python expression over the library that is true
/// where the fault is.
const SHOWN: [(&str, &str, &str); 16] = [
    ("clamp(9, 0, 5)", "numeric", "clamp(9, 0, 5) == 0"),
    ("mean([1, 2])", "numeric", "mean([1, 2]) == 1"),
    (
        "median([4, 1, 3, 2])",
        "numeric",
        "median([4, 1, 3, 2]) == 3",
    ),
    ("percent(1, 3)", "numeric", "percent(1, 3) == 33.0"),
    (
        "slugify(\"Hello, World!\")",
        "strings",
        "slugify('Hello, World!').endswith('-')",
    ),
    (
        "truncate(\"abcdefghij\", 8)",
        "strings",
        "len(truncate('abcdefghij', 8)) > 8",
    ),
    (
        "truncate(\"abc\", 3)",
        "strings",
        "truncate('abc', 3) == '...'",
    ),
    (
        "is_palindrome(\"Racecar\")",
        "strings",
        "is_palindrome('Racecar') is False",
    ),
    (
        "title_case(\"hELLO\")",
        "strings",
        "title_case('hELLO') == 'HELLO'",
    ),
    (
        "overlaps((0, 1), (1, 2))",
        "ranges",
        "overlaps((0, 1), (1, 2)) is True",
    ),
    (
        "merge_intervals([(1, 2), (2, 3)])",
        "ranges",
        "len(merge_intervals([(1, 2), (2, 3)])) == 2",
    ),
    (
        "merge_intervals([(1, 10), (2, 3)])",
        "ranges",
        "[tuple(i) for i in merge_intervals([(1, 10), (2, 3)])] == [(1, 3)]",
    ),
    (
        "total_length([(0, 4), (2, 6)])",
        "ranges",
        "total_length([(0, 4), (2, 6)]) == 8",
    ),
    (
        "count_words(\"The the\")",
        "words",
        "len(count_words('The the')) == 2",
    ),
    (
        "top_words(\"b a\", 1)",
        "words",
        "list(top_words('b a', 1)) == ['b']",
    ),
    (
        "longest_word(\"ab cd\")",
        "words",
        "longest_word('ab cd') == 'cd'",
    ),
];

/// Whether `predicate` holds over the library in `dir` (its four modules'
/// names together).
fn holds(dir: &Path, predicate: &str) -> bool {
    let program = format!(
        "import textkit.numeric as n, textkit.strings as s, textkit.ranges as r, textkit.words as w\n\
         names = {{**vars(n), **vars(s), **vars(r), **vars(w)}}\n\
         print(bool(eval({predicate:?}, names)))\n"
    );
    let out = Command::new("python3")
        .args(["-B", "-c", &program])
        .env_remove("PYTHONPATH")
        .current_dir(dir)
        .output()
        .unwrap();
    let said = String::from_utf8_lossy(&out.stdout).trim().to_string();
    assert!(
        said == "True" || said == "False",
        "{predicate}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    said == "True"
}

/// The instance's files under `prefix`, the prefix taken off.
fn part(instance: &Instance, prefix: &str) -> Folder {
    let mut part = Folder::new();
    for (path, entry) in instance.folder().under(prefix) {
        part.put(
            &path[prefix.len() + 1..],
            entry.bytes.clone(),
            entry.executable,
        )
        .unwrap();
    }
    part
}

#[test]
fn every_catalogued_fault_shows_as_reported_and_only_the_fix_is_rewarded() {
    let plugin = RepositoryBugs::new();
    let mut seen = BTreeSet::new();
    for seed in 0..400 {
        // seed: any; drawn until every fault has been injected alone
        if seen.len() == SHOWN.len() {
            break;
        }
        let instance = plugin
            .generate(&InstanceRequest {
                split: Split::Train,
                seed,
                difficulty: BTreeMap::from([("bugs".to_string(), 1), ("detail".to_string(), 3)]),
            })
            .unwrap();
        let (call, module, predicate) = SHOWN
            .iter()
            .find(|(call, _, _)| instance.instruction().contains(&format!("- `{call}`")))
            .unwrap_or_else(|| {
                panic!("a report not in the catalogue:\n{}", instance.instruction())
            });
        if !seen.insert(*call) {
            continue;
        }
        let mut given = part(&instance, "environment");
        for (path, entry) in part(&instance, "tests/suite").entries() {
            given
                .put(&format!("hidden/{path}"), entry.bytes.clone(), false)
                .unwrap();
        }
        let mut fixed = given.clone();
        for (path, entry) in part(&instance, "solution").entries() {
            if path.starts_with("textkit/") {
                fixed.put(path, entry.bytes.clone(), false).unwrap();
            }
        }
        let mutant = tempfile::tempdir().unwrap();
        given.write_to(mutant.path()).unwrap();
        let original = tempfile::tempdir().unwrap();
        fixed.write_to(original.path()).unwrap();

        assert!(
            holds(mutant.path(), predicate),
            "{call}: the fault does not show"
        );
        assert!(
            !holds(original.path(), predicate),
            "{call}: the original shows it too"
        );
        let before = discovered(original.path(), "hidden");
        assert!(
            before.values().all(|&ok| ok),
            "{call}: the original fails its suite"
        );
        let after = discovered(mutant.path(), "hidden");
        assert_eq!(
            before.keys().collect::<Vec<_>>(),
            after.keys().collect::<Vec<_>>()
        );
        let failing: Vec<&String> = after
            .iter()
            .filter(|(_, &ok)| !ok)
            .map(|(id, _)| id)
            .collect();
        assert!(!failing.is_empty(), "{call}: the suite does not catch it");
        assert!(
            failing
                .iter()
                .all(|id| id.starts_with(&format!("test_{module}."))),
            "{call}: {failing:?}"
        );

        assert_eq!(reward_of(&instance, |_| ()), Some(0.0), "{call}: as given");
        assert_eq!(
            reward_of(&instance, |s| assert!(s.solve().unwrap().succeeded())),
            Some(1.0),
            "{call}: the reference fix"
        );
    }
    assert_eq!(
        seen.len(),
        SHOWN.len(),
        "faults never injected alone: {seen:?}"
    );
}
