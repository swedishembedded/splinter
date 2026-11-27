// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements bug-fixing training environments with
// trustworthy verifiers, for its clients. If your team needs expertise in
// repository tasks for agent reinforcement learning, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Repository software tasks on real projects: faults injected by mutation
//! into projects that have their own test suites.
//!
//! Each instance draws one repository, from the train repositories for the
//! train split and from the dev repositories for the dev split, and the two
//! sets never share a repository. The generator mutates the repository's
//! source ([`splinter_eval::verifiers::mutation::mutants`]) and keeps a
//! mutant only when the project's own suite, run on it, still runs every test
//! and fails a few of them: a mutant that breaks the build or that no test
//! sees is dropped, so no instance is equivalent or unsolvable by
//! construction. The test files holding the failing tests are taken out of
//! the workspace and kept as a hidden overlay, so the solver works from the
//! symptoms (test names, and with more detail the assertion messages) and not
//! from the tests.
//!
//! The verifier is the one of the embedded-library plugin, widened to a whole
//! project: it copies the workspace, lays the hidden tests over the copy, runs
//! the entire suite in a child process that reads a nonce before importing
//! anything, requires the pinned number of tests to have run and all to pass,
//! and requires a canary test to fail.
//!
//! Knobs: `bugs` (faults injected, in different files) and `detail` (3 test
//! names with their assertion messages, 2 names, 1 only the test files).
//!
//! The repositories are the original projects under `tasks/`, which no
//! model has seen; a claim about generalisation to other code still needs a
//! private or newer repository, and the fix of any instance is the project as
//! it was, so a model that memorised a project solves its instances.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use splinter_core::rng::SplitMix64;
use splinter_core::verifiability::{Split, VerifierClass};
use splinter_eval::verifiers::mutation::mutants;

use crate::factory::artifact::{
    environment_id, EnvMeta, Instance, Parts, SolverTools, SCHEMA_VERSION,
};
use crate::factory::attack::{Attack, AttackKind, Edit};
use crate::factory::folder::Folder;
use crate::factory::host::Host;
use crate::factory::plugin::{
    fault_novelty_text, resolve_difficulty, InstanceRequest, Knob, Plugin, PluginError, Reference,
};

use super::REPOSITORY_MUTANTS;

/// The plugin's version. Any change to the mutation choice, the validation,
/// the symptoms, the verifier or the instruction text is a new version.
pub const VERSION: &str = "2";

/// The repositories trained on, and the ones kept for dev. They never share
/// one: a dev instance is a project the policy never practised on.
pub const TRAIN_REPOSITORIES: [&str; 5] =
    ["iniparse", "ttlcache", "exprcalc", "mdrender", "graphkit"];
/// The repositories dev instances are drawn from.
pub const DEV_REPOSITORIES: [&str; 3] = ["linediff", "caldate", "tokenbucket"];

const KNOBS: &[Knob] = &[
    Knob { name: "bugs", min: 1, max: 3, default: 1, doc: "how many faults are injected, each in another file" },
    Knob { name: "detail", min: 1, max: 3, default: 3, doc: "how much the report says: 3 failing tests with their messages, 2 their names, 1 only their files" },
];

/// Candidate mutants asked of the mutation operators per source file.
const MUTANTS_PER_FILE: usize = 40;
/// Suite runs one generation may spend looking for valid mutants.
const MAX_RUNS: usize = 40;
/// The most tests a valid mutant may fail: a local fault, not a broken module.
const MAX_FAILING: usize = 6;
/// The most test files that may be hidden.
const MAX_HIDDEN_FILES: usize = 3;

const RUNNER: &str = include_str!("repository_mutants/runner.py");
const VERIFY: &str = include_str!("repository_mutants/verify.py");

/// A test that failed on a mutant.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Failure {
    id: String,
    name: String,
    message: String,
}

/// What running a project's suite found.
#[derive(Debug, PartialEq, Eq)]
struct SuiteRun {
    ran: u32,
    failures: Vec<Failure>,
    broken: bool,
}

/// One result of two runs of the same suite on the same tree, or a broken
/// one when they differ: a generator never decides on a result that varies.
fn agreed(first: SuiteRun, second: SuiteRun) -> SuiteRun {
    if first == second {
        first
    } else {
        SuiteRun {
            ran: 0,
            failures: Vec::new(),
            broken: true,
        }
    }
}

fn parse_suite(stderr: &str) -> SuiteRun {
    let ran = stderr
        .lines()
        .find_map(|l| l.strip_prefix("Ran "))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|n| n.parse().ok())
        .unwrap_or(0);
    let mut failures = Vec::new();
    let mut broken = false;
    for block in stderr
        .split("======================================================================\n")
        .skip(1)
    {
        let mut lines = block.lines();
        let Some(header) = lines.next() else { continue };
        let Some(rest) = header
            .strip_prefix("FAIL: ")
            .or_else(|| header.strip_prefix("ERROR: "))
        else {
            continue;
        };
        if rest.contains("_FailedTest") || rest.contains("unittest.loader") {
            broken = true;
            continue;
        }
        let name = rest.split(" (").next().unwrap_or(rest).to_string();
        let id = rest
            .split_once(" (")
            .and_then(|(_, tail)| tail.split(')').next())
            .unwrap_or("")
            .to_string();
        let body = block
            .split_once(
                "----------------------------------------------------------------------\nRan ",
            )
            .map_or(block, |(before, _)| before);
        let message = body
            .lines()
            .skip(1)
            .filter(|l| !l.trim().is_empty() && !l.starts_with("---"))
            .last()
            .unwrap_or("")
            .trim()
            .to_string();
        failures.push(Failure { id, name, message });
    }
    SuiteRun {
        ran,
        failures,
        broken,
    }
}

/// The repository repositories' catalogue entries the plugin needs.
struct Catalog {
    files_under_test: Vec<String>,
    summary: String,
}

fn catalog_of(root: &Path, family: &str) -> Result<Catalog, PluginError> {
    let path = root.join("catalog.toml");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| PluginError::Generation(format!("{}: {e}", path.display())))?;
    let parsed: toml::Table = toml::from_str(&text)
        .map_err(|e| PluginError::Generation(format!("{}: {e}", path.display())))?;
    parsed
        .get("family")
        .and_then(toml::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(toml::Value::as_table)
        .find(|t| t.get("id").and_then(toml::Value::as_str) == Some(family))
        .map(|t| Catalog {
            files_under_test: t
                .get("files_under_test")
                .and_then(toml::Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|v| v.as_str().map(ToString::to_string))
                .collect(),
            summary: t
                .get("summary")
                .and_then(toml::Value::as_str)
                .unwrap_or("")
                .to_string(),
        })
        .ok_or_else(|| PluginError::Generation(format!("{family} is not in {}", path.display())))
}

/// The repository bug-injection plugin.
pub struct RepositoryMutants {
    root: PathBuf,
    host: Host,
}

impl RepositoryMutants {
    /// A plugin over the repositories under `root` (the `tasks/` directory:
    /// a `catalog.toml` and one directory per repository).
    #[must_use]
    pub fn new(root: impl Into<PathBuf>, host: Host) -> Self {
        Self {
            root: root.into(),
            host,
        }
    }

    fn read_repository(&self, family: &str) -> Result<Folder, PluginError> {
        let folder = Folder::read_from(&self.root.join(family))?;
        let keep: Vec<String> = folder
            .paths()
            .filter(|p| !p.contains("__pycache__") && !p.ends_with(".pyc"))
            .map(ToString::to_string)
            .collect();
        let mut kept = Folder::new();
        for path in keep {
            if let Some(entry) = folder.get(&path) {
                kept.put(&path, entry.bytes.clone(), entry.executable)?;
            }
        }
        Ok(kept)
    }

    /// The suite run twice: a result the two runs do not agree on is a flaky
    /// one, and an instance built on it would not be reproducible.
    fn run_suite(&self, tree: &Folder) -> Result<SuiteRun, PluginError> {
        Ok(agreed(
            self.run_suite_once(tree)?,
            self.run_suite_once(tree)?,
        ))
    }

    fn run_suite_once(&self, tree: &Folder) -> Result<SuiteRun, PluginError> {
        let scratch = tempfile::tempdir().map_err(|e| PluginError::Generation(e.to_string()))?;
        let dir = scratch.path().join("tree");
        tree.write_to(&dir)?;
        let ran = self
            .host
            .exec(
                &dir,
                ["python3", "-B", "-m", "unittest"]
                    .map(String::from)
                    .to_vec(),
                BTreeMap::new(),
            )
            .map_err(|e| PluginError::Generation(e.to_string()))?;
        Ok(parse_suite(&ran.stderr))
    }
}

/// The test file a failing test's id names: the longest module prefix that is
/// a file of the repository.
fn test_file(tree: &Folder, id: &str) -> Option<String> {
    let parts: Vec<&str> = id.split('.').collect();
    (1..=parts.len())
        .rev()
        .map(|n| format!("{}.py", parts[..n].join("/")))
        .find(|candidate| tree.get(candidate).is_some())
}

impl Plugin for RepositoryMutants {
    fn id(&self) -> &'static str {
        REPOSITORY_MUTANTS
    }

    fn version(&self) -> &'static str {
        VERSION
    }

    fn verifier_class(&self) -> VerifierClass {
        VerifierClass::Objective
    }

    fn knobs(&self) -> &'static [Knob] {
        KNOBS
    }

    fn shares_answers(&self) -> bool {
        true
    }

    fn generate(&self, request: &InstanceRequest) -> Result<Instance, PluginError> {
        let difficulty = resolve_difficulty(self.id(), KNOBS, &request.difficulty)?;
        let (bugs, detail) = (difficulty["bugs"] as usize, difficulty["detail"]);
        let mut rng = SplitMix64::new(request.seed);
        let repositories: &[&str] = match request.split {
            Split::Train => &TRAIN_REPOSITORIES,
            Split::Dev => &DEV_REPOSITORIES,
        };
        let family = repositories[rng.below(repositories.len())];
        let original = self.read_repository(family)?;
        let catalog = catalog_of(&self.root, family)?;
        let baseline = self.run_suite(&original)?;
        if baseline.ran == 0 || baseline.broken || !baseline.failures.is_empty() {
            return Err(PluginError::Generation(format!(
                "{family}: the suite does not pass as it is ({} tests, {} failing)",
                baseline.ran,
                baseline.failures.len()
            )));
        }

        // Valid single mutants, each in another file, until `bugs` are held.
        let mut chosen: Vec<(String, String)> = Vec::new();
        for _ in 0..MAX_RUNS {
            if chosen.len() == bugs {
                break;
            }
            let Some(file) = rng.pick(&catalog.files_under_test) else {
                break;
            };
            if chosen.iter().any(|(path, _)| path == file) {
                continue;
            }
            let Some(source) = original.text(file) else {
                continue;
            };
            let candidates = mutants(source, MUTANTS_PER_FILE);
            let Some(mutant) = rng.pick(&candidates) else {
                continue;
            };
            let mut tree = original.clone();
            tree.put_text(file, &mutant.text)?;
            let run = self.run_suite(&tree)?;
            if !run.broken
                && run.ran == baseline.ran
                && (1..=MAX_FAILING).contains(&run.failures.len())
            {
                chosen.push((file.clone(), mutant.text.clone()));
            }
        }
        if chosen.len() < bugs {
            return Err(PluginError::Generation(format!(
                "{family}: only {} valid mutants found in {MAX_RUNS} runs, {bugs} asked",
                chosen.len()
            )));
        }
        let mut mutated = original.clone();
        for (file, text) in &chosen {
            mutated.put_text(file, text)?;
        }
        let combined = self.run_suite(&mutated)?;
        if combined.broken || combined.ran != baseline.ran || combined.failures.is_empty() {
            return Err(PluginError::Generation(format!(
                "{family}: the combined faults do not fail the suite cleanly"
            )));
        }

        // The failing tests' files leave the workspace and become the overlay.
        let mut hidden: Vec<String> = combined
            .failures
            .iter()
            .filter_map(|f| test_file(&original, &f.id))
            .collect();
        hidden.sort();
        hidden.dedup();
        if hidden.is_empty() || hidden.len() > MAX_HIDDEN_FILES {
            return Err(PluginError::Generation(format!(
                "{family}: the failing tests lie in {} files",
                hidden.len()
            )));
        }
        let workspace = mutated.without(&hidden.iter().map(String::as_str).collect::<Vec<_>>());
        let mut tests = Folder::new();
        for path in &hidden {
            if let Some(entry) = original.get(path) {
                tests.put(&format!("overlay/{path}"), entry.bytes.clone(), false)?;
            }
        }
        tests.put_script(
            "test.sh",
            "#!/bin/sh\n# Lays the hidden tests over a copy of the workspace and runs the whole suite; see verify.py.\npython3 -I \"${TESTS_DIR:-/tests}/verify.py\"\n",
        )?;
        tests.put_text("verify.py", VERIFY)?;
        tests.put_text("runner.py", RUNNER)?;
        tests.put_text(
            "expected.json",
            &format!("{{\"total\": {}}}\n", baseline.ran),
        )?;

        let mut solution = Folder::new();
        let mut solve = String::from(
            "#!/bin/sh\n# The reference: the original source of each faulty file.\nset -e\n",
        );
        for (n, (file, _)) in chosen.iter().enumerate() {
            solution.put_text(
                &format!("files/{n}"),
                original.text(file).unwrap_or_default(),
            )?;
            solve.push_str(&format!(
                "cp \"${{SOLUTION_DIR:-/solution}}/files/{n}\" \"${{APP_DIR:-/app}}/{file}\"\n"
            ));
        }
        solution.put_script("solve.sh", &solve)?;
        let changed: String = chosen.iter().map(|(file, _)| format!("{file}\n")).collect();
        solution.put_text("changed.txt", &changed)?;

        let mut symptoms = String::new();
        for failure in &combined.failures {
            let line = match detail {
                3 => format!("- `{}` fails: {}\n", failure.name, failure.message),
                2 => format!("- `{}` fails\n", failure.name),
                _ => String::new(),
            };
            symptoms.push_str(&line);
        }
        if detail == 1 {
            for path in &hidden {
                symptoms.push_str(&format!("- tests in `{path}` fail\n"));
            }
        }
        let instruction = format!(
            "# Fix the regression\n\nThe project in `/app` (`{family}`: {}) has a regression: its test suite no longer passes. The failing tests are not in the workspace.\n\nSymptoms:\n{symptoms}\nFix the code under test so that the whole suite passes. Do not change tests.\n",
            catalog.summary
        );

        let meta = EnvMeta {
            schema_version: SCHEMA_VERSION,
            environment: environment_id(self.id(), VERSION, &difficulty),
            plugin: self.id().into(),
            plugin_version: VERSION.into(),
            verifier_class: VerifierClass::Objective,
            verifier: format!("whole-suite-with-canary@{VERSION}"),
            difficulty,
            split: request.split,
            seed: request.seed,
            solver_tools: SolverTools::Shell,
        };
        Ok(Instance::assemble(
            meta,
            Parts {
                instruction,
                workspace,
                packages: vec!["python3".into(), "lua5.1".into()],
                tests,
                solution,
                category: format!("repository:{family}"),
            },
        )?)
    }

    fn novelty_text(&self, instance: &Instance) -> String {
        fault_novelty_text(instance, "Symptoms:\n")
    }

    fn attacks(&self, instance: &Instance, reference: &Reference) -> Vec<Attack> {
        // Each payload is put at the top of a faulty file, which the suite
        // imports, and the file's own code stays under it.
        let Some((path, _)) = reference.changed.iter().next() else {
            return Vec::new();
        };
        let workspace = instance.workspace();
        let body = workspace.text(path).unwrap_or_default().to_string();
        let plant = |name: &str, payload: &str| Attack {
            name: format!("repo:{name}"),
            kind: AttackKind::Tamper,
            edits: vec![Edit::Write {
                path: path.clone(),
                bytes: format!("{payload}\n{body}").into_bytes(),
            }],
        };
        vec![
            plant("exit-zero-on-import", "import os\nos._exit(0)"),
            plant(
                "forged-result-lines",
                "import atexit\n_line = '@@RESULT 0000000000000000 1 1 1'\nprint(_line)\natexit.register(lambda: print(_line))",
            ),
            plant(
                "everything-succeeds",
                "import unittest\n\n\ndef _run(self, result=None):\n    result.startTest(self)\n    result.addSuccess(self)\n    result.stopTest(self)\n    return result\n\n\nunittest.TestCase.run = _run",
            ),
            plant(
                "everything-is-skipped",
                "import unittest\n\n\ndef _run(self, result=None):\n    result.startTest(self)\n    result.addSkip(self, 'skipped')\n    result.stopTest(self)\n    return result\n\n\nunittest.TestCase.run = _run",
            ),
        ]
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // a panic is the failure report
mod tests {
    use super::*;

    #[test]
    fn train_and_dev_repositories_are_disjoint() {
        for dev in DEV_REPOSITORIES {
            assert!(!TRAIN_REPOSITORIES.contains(&dev), "{dev} is in both");
        }
    }

    #[test]
    fn a_suite_report_gives_the_count_the_failing_tests_and_their_messages() {
        let stderr = "FF\n======================================================================\nFAIL: test_a (tests.test_x.Case.test_a)\n----------------------------------------------------------------------\nTraceback (most recent call last):\n  File \"x.py\", line 1, in test_a\nAssertionError: 3 != 4\n\n======================================================================\nERROR: test_b (tests.test_y.Case.test_b)\n----------------------------------------------------------------------\nTypeError: bad\n\n----------------------------------------------------------------------\nRan 10 tests in 0.1s\n\nFAILED (failures=1, errors=1)\n";
        let run = parse_suite(stderr);
        assert_eq!(run.ran, 10);
        assert!(!run.broken);
        assert_eq!(run.failures.len(), 2);
        assert_eq!(run.failures[0].name, "test_a");
        assert_eq!(run.failures[0].id, "tests.test_x.Case.test_a");
        assert_eq!(run.failures[0].message, "AssertionError: 3 != 4");
        assert_eq!(run.failures[1].message, "TypeError: bad");
    }

    #[test]
    fn two_runs_that_disagree_are_a_broken_suite_and_two_that_agree_are_the_result() {
        let report = |failing: &str| {
            parse_suite(&format!("F\n======================================================================\nFAIL: {failing} (tests.test_x.Case.{failing})\n----------------------------------------------------------------------\nAssertionError: no\n\n----------------------------------------------------------------------\nRan 10 tests in 0.1s\n\nFAILED (failures=1)\n"))
        };
        let same = agreed(report("test_a"), report("test_a"));
        assert!(!same.broken);
        assert_eq!(same.failures.len(), 1);
        let flaky = agreed(report("test_a"), report("test_b"));
        assert!(flaky.broken);
        assert!(flaky.failures.is_empty());
    }

    #[test]
    fn a_module_that_cannot_be_imported_is_a_broken_build_not_a_failing_test() {
        let stderr = "E\n======================================================================\nERROR: tests.test_x (unittest.loader._FailedTest.tests.test_x)\n----------------------------------------------------------------------\nImportError: no\n\n----------------------------------------------------------------------\nRan 1 test in 0.0s\n\nFAILED (errors=1)\n";
        let run = parse_suite(stderr);
        assert!(run.broken);
        assert!(run.failures.is_empty());
    }

    #[test]
    fn a_failing_test_names_the_longest_module_prefix_that_is_a_file() {
        let mut tree = Folder::new();
        tree.put_text("tests/test_x.py", "").unwrap();
        assert_eq!(
            test_file(&tree, "tests.test_x.Case.test_a").as_deref(),
            Some("tests/test_x.py")
        );
        assert_eq!(test_file(&tree, "tests.test_z.Case.test_a"), None);
    }
}
