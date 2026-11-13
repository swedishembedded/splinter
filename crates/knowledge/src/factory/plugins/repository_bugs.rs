// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements bug-fixing training environments with
// trustworthy verifiers, for its clients. If your team needs expertise in
// repository tasks for agent reinforcement learning, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Repository software tasks: a small library with bugs injected, fixed
//! until a hidden test suite passes.
//!
//! The library (`textkit`: numbers, strings, intervals and words, about two
//! dozen functions) is embedded; each instance injects `bugs` of a catalogue
//! of sixteen faults, every one of which the hidden suite is known to catch.
//! The workspace has the mutated library and a smoke test; the suite lives in
//! `tests/` and never in the workspace. The verifier runs the suite in a child
//! process that reads a nonce before importing anything of the workspace, trusts
//! only the result line carrying it, and adds a canary test that must fail: a
//! library that patches the test framework to report success is caught by the
//! canary. The reward is 1 only when every test passes and the canary fails; a
//! reward component reports the share of tests that pass.
//!
//! The fix is the library as it was, the same for every instance, so a model
//! that memorised the library solves any of them; this is stated in the
//! ledger as the contamination limit of a single embedded library. Knobs:
//! `bugs` (how many faults) and `detail` (3 a failing example per bug, 2 the
//! function, 1 only the module).

use splinter_core::rng::SplitMix64;
use splinter_core::verifiability::VerifierClass;

use crate::factory::artifact::{
    environment_id, EnvMeta, Instance, Parts, SolverTools, SCHEMA_VERSION,
};
use crate::factory::attack::{Attack, AttackKind, Edit};
use crate::factory::folder::Folder;
use crate::factory::plugin::{
    resolve_difficulty, InstanceRequest, Knob, Plugin, PluginError, Reference,
};

use super::REPOSITORY_BUGS;

/// The plugin's version. Any change to the library, the catalogue, the
/// suite, the runner or the instruction text is a new version.
pub const VERSION: &str = "1";

/// How many tests the hidden suite holds; the verifier requires exactly this
/// many to have run.
const SUITE_TESTS: u32 = 42;

const KNOBS: &[Knob] = &[
    Knob {
        name: "bugs",
        min: 1,
        max: 4,
        default: 2,
        doc: "how many faults are injected",
    },
    Knob {
        name: "detail",
        min: 1,
        max: 3,
        default: 3,
        doc:
            "how much the report says: 3 a failing example per fault, 2 the function, 1 the module",
    },
];

const LIBRARY: [(&str, &str); 5] = [
    (
        "textkit/__init__.py",
        include_str!("repository_bugs/textkit/__init__.py"),
    ),
    (
        "textkit/numeric.py",
        include_str!("repository_bugs/textkit/numeric.py"),
    ),
    (
        "textkit/strings.py",
        include_str!("repository_bugs/textkit/strings.py"),
    ),
    (
        "textkit/ranges.py",
        include_str!("repository_bugs/textkit/ranges.py"),
    ),
    (
        "textkit/words.py",
        include_str!("repository_bugs/textkit/words.py"),
    ),
];

const SUITE: [(&str, &str); 4] = [
    (
        "suite/test_numeric.py",
        include_str!("repository_bugs/suite/test_numeric.py"),
    ),
    (
        "suite/test_strings.py",
        include_str!("repository_bugs/suite/test_strings.py"),
    ),
    (
        "suite/test_ranges.py",
        include_str!("repository_bugs/suite/test_ranges.py"),
    ),
    (
        "suite/test_words.py",
        include_str!("repository_bugs/suite/test_words.py"),
    ),
];

const RUNNER: &str = include_str!("repository_bugs/runner.py");
const VERIFY: &str = include_str!("repository_bugs/verify.py");

/// One fault: where it is, how the source is changed, and how it shows.
struct Bug {
    file: &'static str,
    function: &'static str,
    find: &'static str,
    replace: &'static str,
    example: &'static str,
}

const BUGS: [Bug; 16] = [
    Bug {
        file: "numeric",
        function: "clamp",
        find: "        return hi\n    return x",
        replace: "        return lo\n    return x",
        example: "`clamp(9, 0, 5)` returns 0 where 5 is expected",
    },
    Bug {
        file: "numeric",
        function: "mean",
        find: "sum(xs) / len(xs)",
        replace: "sum(xs) // len(xs)",
        example: "`mean([1, 2])` returns 1 where 1.5 is expected",
    },
    Bug {
        file: "numeric",
        function: "median",
        find: "(ordered[mid - 1] + ordered[mid]) / 2",
        replace: "ordered[mid]",
        example: "`median([4, 1, 3, 2])` returns 3 where 2.5 is expected",
    },
    Bug {
        file: "numeric",
        function: "percent",
        find: "round(100.0 * part / whole, 1)",
        replace: "round(100.0 * part / whole, 0)",
        example: "`percent(1, 3)` returns 33.0 where 33.3 is expected",
    },
    Bug {
        file: "strings",
        function: "slugify",
        find: "\"\".join(out).strip(\"-\")",
        replace: "\"\".join(out)",
        example: "`slugify(\"Hello, World!\")` ends in a dash",
    },
    Bug {
        file: "strings",
        function: "truncate",
        find: "text[: limit - 3] + \"...\"",
        replace: "text[:limit] + \"...\"",
        example: "`truncate(\"abcdefghij\", 8)` is longer than 8 characters",
    },
    Bug {
        file: "strings",
        function: "truncate",
        find: "if len(text) <= limit:",
        replace: "if len(text) < limit:",
        example: "`truncate(\"abc\", 3)` returns `...` where `abc` is expected",
    },
    Bug {
        file: "strings",
        function: "is_palindrome",
        find: "[c.lower() for c in text if c.isalnum()]",
        replace: "[c for c in text if c.isalnum()]",
        example: "`is_palindrome(\"Racecar\")` is False",
    },
    Bug {
        file: "strings",
        function: "title_case",
        find: "w[1:].lower()",
        replace: "w[1:]",
        example: "`title_case(\"hELLO\")` returns `HELLO`",
    },
    Bug {
        file: "ranges",
        function: "overlaps",
        find: "a[0] < b[1] and b[0] < a[1]",
        replace: "a[0] <= b[1] and b[0] <= a[1]",
        example: "`overlaps((0, 1), (1, 2))` is True for intervals that only touch",
    },
    Bug {
        file: "ranges",
        function: "merge_intervals",
        find: "start <= merged[-1][1]",
        replace: "start < merged[-1][1]",
        example: "`merge_intervals([(1, 2), (2, 3)])` leaves two intervals",
    },
    Bug {
        file: "ranges",
        function: "merge_intervals",
        find: "max(merged[-1][1], end)",
        replace: "end",
        example: "`merge_intervals([(1, 10), (2, 3)])` returns `[(1, 3)]`",
    },
    Bug {
        file: "ranges",
        function: "total_length",
        find: "for start, end in merge_intervals(intervals)",
        replace: "for start, end in intervals",
        example: "`total_length([(0, 4), (2, 6)])` counts the overlap twice",
    },
    Bug {
        file: "words",
        function: "count_words",
        find: "text.lower().split():",
        replace: "text.split():",
        example: "`count_words(\"The the\")` counts two different words",
    },
    Bug {
        file: "words",
        function: "top_words",
        find: "key=lambda item: (-item[1], item[0])",
        replace: "key=lambda item: (-item[1],)",
        example: "`top_words(\"b a\", 1)` returns `[\"b\"]` where `[\"a\"]` is expected",
    },
    Bug {
        file: "words",
        function: "longest_word",
        find: "len(word) > len(best)",
        replace: "len(word) >= len(best)",
        example: "`longest_word(\"ab cd\")` returns `cd` where `ab` is expected",
    },
];

/// The bugs a request injects: distinct, in catalogue order of a seeded
/// shuffle.
fn chosen(seed: u64, count: usize) -> Vec<usize> {
    let mut rng = SplitMix64::new(seed);
    let mut order: Vec<usize> = (0..BUGS.len()).collect();
    rng.shuffle(&mut order);
    order.truncate(count);
    order.sort_unstable();
    order
}

/// The repository bug-fixing plugin.
#[derive(Default)]
pub struct RepositoryBugs;

impl RepositoryBugs {
    /// A plugin value.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Plugin for RepositoryBugs {
    fn id(&self) -> &'static str {
        REPOSITORY_BUGS
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
        let (count, detail) = (difficulty["bugs"] as usize, difficulty["detail"]);
        let bugs = chosen(request.seed, count);

        let mut workspace = Folder::new();
        let mut solution = Folder::new();
        for (path, original) in LIBRARY {
            let mut source = original.to_string();
            let mut changed = false;
            for &index in &bugs {
                let bug = &BUGS[index];
                if path == format!("textkit/{}.py", bug.file) {
                    if !source.contains(bug.find) {
                        return Err(PluginError::Generation(format!(
                            "bug {index} does not apply to {path}"
                        )));
                    }
                    source = source.replacen(bug.find, bug.replace, 1);
                    changed = true;
                }
            }
            workspace.put_text(path, &source)?;
            if changed {
                solution.put_text(path, original)?;
            }
        }
        workspace.put_text(
            "tests/test_smoke.py",
            "import unittest\n\nfrom textkit import numeric, ranges, strings, words\n\n\nclass SmokeTests(unittest.TestCase):\n    def test_the_modules_import(self):\n        self.assertTrue(callable(numeric.clamp) and callable(words.count_words))\n        self.assertTrue(callable(strings.slugify) and callable(ranges.merge_intervals))\n",
        )?;
        solution.put_script(
            "solve.sh",
            "#!/bin/sh\n# The reference: put the original sources of the faulty files back.\nset -e\nSOL=\"${SOLUTION_DIR:-/solution}\"; APP=\"${APP_DIR:-/app}\"\nfor f in \"$SOL\"/textkit/*.py; do cp \"$f\" \"$APP/textkit/$(basename \"$f\")\"; done\n",
        )?;

        let mut tests = Folder::new();
        tests.put_script(
            "test.sh",
            "#!/bin/sh\n# Runs the hidden suite in a child process and grades by its result line; see verify.py.\npython3 -I \"${TESTS_DIR:-/tests}/verify.py\"\n",
        )?;
        tests.put_text("verify.py", VERIFY)?;
        tests.put_text("runner.py", RUNNER)?;
        tests.put_text("expected.json", &format!("{{\"total\": {SUITE_TESTS}}}\n"))?;
        for (path, text) in SUITE {
            tests.put_text(path, text)?;
        }

        let mut reports = String::new();
        for &index in &bugs {
            let bug = &BUGS[index];
            let line = match detail {
                3 => format!("- {}.", bug.example),
                2 => format!(
                    "- `{}` in `textkit/{}.py` gives a wrong result.",
                    bug.function, bug.file
                ),
                _ => format!(
                    "- Something in `textkit/{}.py` gives a wrong result.",
                    bug.file
                ),
            };
            reports.push_str(&line);
            reports.push('\n');
        }
        let instruction = format!(
            "# Fix the library\n\n`/app/textkit` is a small Python library (numbers, strings, intervals, words). Its tests report these problems:\n\nReports:\n{reports}\nFix the library so that its test suite passes. The suite is not in the workspace; change the library, not `tests/`. Anything else the suite checks must keep working.\n"
        );

        let meta = EnvMeta {
            schema_version: SCHEMA_VERSION,
            environment: environment_id(self.id(), VERSION, &difficulty),
            plugin: self.id().into(),
            plugin_version: VERSION.into(),
            verifier_class: VerifierClass::Objective,
            verifier: format!("hidden-suite-with-canary@{VERSION}"),
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
                packages: vec!["python3".into()],
                tests,
                solution,
                category: "repository".into(),
            },
        )?)
    }

    fn novelty_text(&self, instance: &Instance) -> String {
        let text = instance.instruction();
        text.split_once("Reports:\n")
            .map_or(text, |(_, rest)| rest)
            .to_string()
    }

    fn attacks(&self, _instance: &Instance, _reference: &Reference) -> Vec<Attack> {
        let plant = |name: &str, source: &str| Attack {
            name: format!("repo:{name}"),
            kind: AttackKind::Tamper,
            edits: vec![Edit::Write {
                path: "textkit/__init__.py".into(),
                bytes: source.as_bytes().to_vec(),
            }],
        };
        vec![
            plant("exit-zero-on-import", "import os\nos._exit(0)\n"),
            plant(
                "forged-result-lines",
                "import atexit\nline = '@@RESULT 0000000000000000 42 42 1'\nprint(line)\natexit.register(lambda: print(line))\n",
            ),
            plant(
                "everything-succeeds",
                "import unittest\n\n\ndef _run(self, result=None):\n    result.startTest(self)\n    result.addSuccess(self)\n    result.stopTest(self)\n    return result\n\n\nunittest.TestCase.run = _run\n",
            ),
            plant(
                "everything-is-skipped",
                "import unittest\n\n\ndef _run(self, result=None):\n    result.startTest(self)\n    result.addSkip(self, 'skipped')\n    result.stopTest(self)\n    return result\n\n\nunittest.TestCase.run = _run\n",
            ),
            plant(
                "planted-test-framework",
                "raise SystemExit(0)\n",
            ),
        ]
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // a panic is the failure report
mod tests {
    use super::*;
    use splinter_core::verifiability::Split;
    use std::collections::BTreeMap;

    fn request(seed: u64, bugs: u32, detail: u32) -> InstanceRequest {
        InstanceRequest {
            split: Split::Train,
            seed,
            difficulty: BTreeMap::from([("bugs".into(), bugs), ("detail".into(), detail)]),
        }
    }

    #[test]
    fn every_catalogued_bug_applies_to_its_file_exactly_once() {
        for (index, bug) in BUGS.iter().enumerate() {
            let (_, source) = LIBRARY
                .iter()
                .find(|(path, _)| *path == format!("textkit/{}.py", bug.file))
                .unwrap();
            assert_eq!(
                source.matches(bug.find).count(),
                1,
                "bug {index}: {}",
                bug.function
            );
            assert_ne!(bug.find, bug.replace);
        }
    }

    #[test]
    fn the_suite_size_pinned_for_the_verifier_is_the_number_of_tests_in_it() {
        let tests: usize = SUITE
            .iter()
            .map(|(_, text)| text.matches("    def test_").count())
            .sum();
        assert_eq!(tests, SUITE_TESTS as usize);
    }

    #[test]
    fn the_hidden_suite_is_never_in_the_workspace_and_bugs_are_distinct() {
        // seed: coverage over many seeds
        for seed in 0..30 {
            let instance = RepositoryBugs::new()
                .generate(&request(seed, 4, 3))
                .unwrap();
            assert!(!instance
                .solver_view()
                .paths()
                .any(|p| p.contains("suite/") || p.contains("verify.py")));
            let fixed: Vec<_> = instance
                .folder()
                .under("solution")
                .map(|(p, _)| p.to_string())
                .collect();
            assert!(fixed.iter().any(|p| p.ends_with(".py")), "{fixed:?}");
            assert_eq!(chosen(seed, 4).len(), 4);
        }
    }

    #[test]
    fn less_detail_gives_a_vaguer_report_and_other_seeds_other_faults() {
        let plugin = RepositoryBugs::new();
        let full = plugin.generate(&request(1, 2, 3)).unwrap();
        let vague = plugin.generate(&request(1, 2, 1)).unwrap();
        assert!(full.instruction().contains("returns") || full.instruction().contains("is"));
        assert!(vague.instruction().contains("Something in `textkit/"));
        assert_ne!(
            plugin.generate(&request(1, 2, 3)).unwrap().version(),
            plugin.generate(&request(2, 2, 3)).unwrap().version()
        );
    }
}
