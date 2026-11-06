// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements comprehension drills made by code from a
// codebase's own functions and tests, for its clients. If your team needs
// expertise in teaching a model what your code does, not only how to repair
// it, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: comprehension drills are made by code from a repository's clean
//! snapshot.
//!
//! * a drill shows a program that calls one function on inputs taken from the
//!   repository's own tests, and its reference is what the interpreter prints
//!   when that exact program runs against the snapshot (a working tree that
//!   was changed since the commit does not matter);
//! * a candidate is admitted only when it runs, prints something informative
//!   and the same twice, and a function no test calls with literals gets none;
//! * a drill is tagged with its file (its family) and `file::function`, and
//!   no assertion, expected value, test name or test text reaches it;
//! * the same seed makes the same drills;
//! * Lua modules are drilled by the same rules, a table result printed by a
//!   serialiser the program carries;
//! * the formal verifier grades an answer to a drill, and a drill is
//!   exported as a Harbor task that rewards the reference.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Arc;

use common::{git, write, Repo, AT};
use splinter_core::annotation::Outcome;
use splinter_core::clock::FixedClock;
use splinter_core::experience::{Experience, PrivilegedKind, Provenance, Task};
use splinter_core::rng::Seed;
use splinter_core::selfcontained::check_self_contained;
use splinter_eval::verifiers::normalise::Normalisation;
use splinter_eval::verifiers::{formal::ExactMatchVerifier, Verifier};
use splinter_knowledge::capture::{DirtyPolicy, TreeRequest};
use splinter_knowledge::tasks::workspace::{
    touched_lines, Drill, DrillGeneration, DrillOrigin, DrillPlan, DrillReason, WorkspaceError,
    WorkspaceGenerator,
};
use splinter_sandbox::{Limits, ProcessSandbox};

const PRICING: &str = "\
import time


def total(prices):
    result = 0
    for price in prices:
        result += price
    return result


def discount(price, percent):
    if percent > 100:
        raise ValueError(\"percent over 100\")
    return price - price * percent // 100


def label(code, width=6):
    return code.rjust(width, \"0\")


def stamp():
    return time.time()


def tags(items):
    return {item.upper() for item in items}


def log(message):
    return None


def untested(value):
    return value * 2
";

const TEST_PRICING: &str = "\
import unittest

from shop.pricing import discount, label, log, stamp, tags, total


class SENTINEL_CLASS(unittest.TestCase):
    def test_sentinel_name_of_a_hidden_test(self):
        \"\"\"SENTINEL_DOCSTRING.\"\"\"
        self.assertEqual(total([2, 3]), 91827364)
        self.assertEqual(total([10, 20, 30]), 60)
        self.assertEqual(total([]), 0)

    def test_discounts(self):
        self.assertEqual(discount(200, 10), 180)
        self.assertEqual(discount(50, 50), 25)
        with self.assertRaises(ValueError):
            discount(10, 101)

    def test_labels(self):
        self.assertEqual(label(\"7\"), \"SENTINEL_EXPECTED\")
        self.assertEqual(label(\"42\", 4), \"0042\")

    def test_others(self):
        self.assertTrue(stamp() > 0)
        self.assertEqual(tags([\"a\", \"b\", \"c\"]), {\"A\", \"B\", \"C\"})
        self.assertIsNone(log(\"hello\"))
";

const RATIO_LUA: &str = "\
local M = {}

function M.half(n)
  return n / 2
end

function M.pair(a, b)
  return { first = a, second = b }
end

function M.fail(text)
  error(\"bad \" .. text, 0)
end

function M.unreached(n)
  return n + 1
end

return M
";

const TEST_RATIO: &str = "\
from tests.host import LuaCase


class Ratio(LuaCase):
    MODULE = \"ratio\"

    def test_half(self):
        self.check(\"run(M.half, 9)\", \"4.5\")
        self.check(\"run(M.half, 3.5)\", \"1.75\")

    def test_pair(self):
        self.check(\"run(M.pair, 'x', 2)\", \"{first='x', second=2}\")

    def test_fail(self):
        self.check(\"run(M.fail, 'input')\", \"error: bad input\")
";

/// The helper a Lua test class runs its bodies through: it binds the module
/// the class names to `M`, as a host helper does.
const HOST: &str = "\
import subprocess
import unittest


class LuaCase(unittest.TestCase):
    MODULE = \"\"

    def check(self, body, expected):
        script = f'local M = require(\"{self.MODULE}\")\\n{body}'
        self.assertEqual(subprocess.run([\"lua\", \"-e\", script]).returncode, 0)
";

const FILES: [(&str, &str); 8] = [
    (
        "pyproject.toml",
        "[project]\nname = \"shop\"\nversion = \"1\"\n",
    ),
    ("shop/__init__.py", ""),
    ("shop/pricing.py", PRICING),
    ("tests/__init__.py", ""),
    ("tests/test_pricing.py", TEST_PRICING),
    ("lua/ratio.lua", RATIO_LUA),
    ("tests/test_ratio.py", TEST_RATIO),
    ("tests/host.py", HOST),
];

struct Rig {
    repo: Repo,
    generator: WorkspaceGenerator,
}

fn rig(name: &str) -> Rig {
    let repo = Repo::of_files(name, &FILES);
    let calls = repo.scratch.0.join("calls");
    fs::create_dir_all(&calls).unwrap();
    let sandbox = ProcessSandbox::new(&calls, [], &[], Limits::default());
    let generator = WorkspaceGenerator::new(
        Arc::new(sandbox),
        repo.host.clone(),
        repo.store.clone(),
        repo.scratch.0.join("work"),
    );
    Rig { repo, generator }
}

impl Rig {
    fn drills(&self, plan: &DrillPlan) -> DrillGeneration {
        // What is archived is the commit: a changed working tree is allowed
        // here to show that it is not what the drills run against.
        let request = TreeRequest {
            dirty: DirtyPolicy::Allow,
            ..self.repo.request("HEAD")
        };
        let tree = self.repo.try_capture(&request).unwrap();
        self.generator.drills(&tree.snapshot, plan).unwrap()
    }

    fn plan(per_function: usize, seed: u64) -> DrillPlan {
        DrillPlan::new(per_function, Seed::new(seed))
    }
}

fn origin(drill: &Drill) -> DrillOrigin {
    DrillOrigin::of_task(&drill.task).unwrap()
}

fn reference(task: &Task) -> &str {
    task.privileged
        .iter()
        .find(|p| p.kind == PrivilegedKind::Reference)
        .map(|p| p.content.as_str())
        .unwrap()
}

/// The program a drill's instruction shows, from its fence.
fn shown_program(task: &Task) -> String {
    let start = task.instruction.find("```").unwrap();
    let body = &task.instruction[start..];
    let body = body.split_once('\n').unwrap().1;
    body.rsplit_once("```").unwrap().0.to_string()
}

/// What `program` prints when run by `interpreter` in `dir`.
fn print_of(interpreter: &str, program: &str, dir: &std::path::Path) -> String {
    let mut child = Command::new(interpreter)
        .arg("-")
        .current_dir(dir)
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(program.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "{program}");
    String::from_utf8(out.stdout)
        .unwrap()
        .trim_end()
        .to_string()
}

fn functions(generation: &DrillGeneration) -> BTreeSet<String> {
    generation
        .admitted
        .iter()
        .map(|d| origin(d).function_concept())
        .collect()
}

#[test]
fn a_drill_is_a_program_over_test_inputs_whose_reference_is_what_it_prints_on_the_clean_tree() {
    let rig = rig("drills-reference");
    // A working tree changed since the commit is not the snapshot: the
    // drills run against what was committed.
    write(
        &rig.repo.repo.join("shop/pricing.py"),
        &PRICING.replace("result += price", "result += price + 1000"),
    );
    // Enough per function to draw every candidate there is.
    let made = rig.drills(&Rig::plan(30, 3));
    assert!(!made.admitted.is_empty(), "{made:#?}");

    let committed = rig.repo.scratch.0.join("committed");
    git(
        &rig.repo.repo,
        &[
            "worktree",
            "add",
            "-q",
            "--detach",
            committed.to_str().unwrap(),
            "HEAD",
        ],
    );
    let mut calls = BTreeSet::new();
    for drill in &made.admitted {
        let program = shown_program(&drill.task);
        let interpreter = match origin(drill).language {
            splinter_knowledge::tasks::workspace::DrillLanguage::Python => "python3",
            splinter_knowledge::tasks::workspace::DrillLanguage::Lua => "lua",
        };
        assert_eq!(
            print_of(interpreter, &program, &committed),
            reference(&drill.task),
            "{program}"
        );
        calls.insert(program);
    }
    // The inputs are the tests' literals: a call a test makes is among the
    // drills, an error is something a program prints, and what it prints is
    // not the broken working tree's output. (Of the calls that raise the same
    // error only one is kept: a function's drills differ in their answers.)
    let raises: Vec<_> = made
        .admitted
        .iter()
        .filter(|d| d.function == "discount" && reference(&d.task).starts_with("raised"))
        .collect();
    assert_eq!(raises.len(), 1);
    assert_eq!(
        reference(&raises[0].task),
        "raised ValueError: percent over 100"
    );
    let total = made
        .admitted
        .iter()
        .find(|d| shown_program(&d.task).contains("total([10, 20, 30])"))
        .unwrap();
    assert_eq!(reference(&total.task), "60");
}

#[test]
fn candidates_that_do_not_run_cleanly_twice_or_say_nothing_are_refused_and_counted() {
    let rig = rig("drills-admission");
    let made = rig.drills(&Rig::plan(6, 3));
    let admitted = functions(&made);
    for refused in [
        "shop/pricing.py::stamp",
        "shop/pricing.py::tags",
        "shop/pricing.py::log",
    ] {
        assert!(!admitted.contains(refused), "{refused}: {admitted:?}");
    }
    for admitted_function in [
        "shop/pricing.py::total",
        "shop/pricing.py::discount",
        "shop/pricing.py::label",
        "lua/ratio.lua::half",
        "lua/ratio.lua::pair",
        "lua/ratio.lua::fail",
    ] {
        assert!(admitted.contains(admitted_function), "{admitted_function}");
    }
    let count = |reason| made.refused.get(&reason).copied().unwrap_or(0);
    // The clock differs between runs; a set of strings is ordered by a hash
    // that differs between processes; `None` has nothing to predict; a
    // function no test calls has no input.
    assert!(
        count(DrillReason::Nondeterministic) >= 2,
        "{:?}",
        made.refused
    );
    assert!(count(DrillReason::Uninformative) >= 1, "{:?}", made.refused);
    assert!(count(DrillReason::NoTestInputs) >= 2, "{:?}", made.refused);
    assert!(admitted
        .iter()
        .all(|f| !f.ends_with("::untested") && !f.ends_with("::unreached")));
}

#[test]
fn a_drill_is_tagged_with_its_file_and_function_and_takes_nothing_from_a_hidden_test() {
    let rig = rig("drills-tags");
    let made = rig.drills(&Rig::plan(6, 3));
    for drill in &made.admitted {
        let o = origin(drill);
        assert_eq!(drill.task.concepts, [o.file.clone(), o.function_concept()]);
        assert_eq!(drill.family, o.file);
        let language = serde_json::to_value(o.language).unwrap();
        assert_eq!(
            drill.task.task.kind,
            format!(
                "predict-{}",
                if language == "python" {
                    "python"
                } else {
                    "lua"
                }
            )
        );
        assert_eq!(drill.task.environment.kind, "closed-book");
        drill.task.validate().unwrap();

        // Neither a test's name, docstring, class or expected value, nor
        // any line of a test, is in the task.
        let everything = serde_json::to_string(&drill.task).unwrap();
        for leaked in [
            "SENTINEL",
            "91827364",
            "test_",
            "assertEqual",
            "unittest",
            "LuaCase",
        ] {
            assert!(!everything.contains(leaked), "{leaked} in {everything}");
        }

        // The student's turn stands on its own once the privileged
        // records are dropped.
        let dropped: Vec<_> = drill.task.privileged.iter().collect();
        check_self_contained(&drill.task.instruction, &dropped).unwrap();
    }
    // `SENTINEL_EXPECTED` is only ever the expected value of a test: the
    // output of `label("7")` is what the code prints.
    let label = made
        .admitted
        .iter()
        .find(|d| shown_program(&d.task).contains("label('7')"))
        .unwrap();
    assert_eq!(reference(&label.task), "'000007'");
}

#[test]
fn the_same_seed_makes_the_same_drills_and_another_seed_draws_other_inputs() {
    let rig = rig("drills-seed");
    // seed: the second run must draw the very drills the first did
    let first = rig.drills(&Rig::plan(1, 11));
    let again = rig.drills(&Rig::plan(1, 11));
    let ids = |g: &DrillGeneration| -> Vec<_> {
        g.admitted.iter().map(|d| d.task.task.id.clone()).collect()
    };
    assert_eq!(ids(&first), ids(&again));
    // seed: a handful of seeds, to show that the draw depends on the seed
    let other: BTreeSet<Vec<_>> = (12..20)
        .map(|s| ids(&rig.drills(&Rig::plan(1, s))))
        .collect();
    assert!(
        other.len() > 1 || other.iter().next() != Some(&ids(&first)),
        "the seed decides nothing"
    );
    // At most the drills asked for are made of each function.
    let mut per: BTreeMap<String, usize> = BTreeMap::new();
    for d in &first.admitted {
        *per.entry(d.function.clone()).or_default() += 1;
    }
    assert!(per.values().all(|n| *n == 1), "{per:?}");
}

#[test]
fn lua_modules_are_drilled_and_a_table_result_is_printed_by_the_programs_own_serialiser() {
    let rig = rig("drills-lua");
    let mut plan = Rig::plan(6, 3);
    plan.files = vec!["lua/ratio.lua".into()];
    let made = rig.drills(&plan);
    assert!(made
        .admitted
        .iter()
        .all(|d| d.task.task.kind == "predict-lua"));
    let pair = made
        .admitted
        .iter()
        .find(|d| d.function == "pair")
        .expect("a table result is drilled");
    let program = shown_program(&pair.task);
    assert!(program.contains("local function show"), "{program}");
    assert_eq!(
        reference(&pair.task),
        "true\t{[\"first\"]=\"x\", [\"second\"]=2}"
    );
    let half = made
        .admitted
        .iter()
        .find(|d| {
            shown_program(&d.task).contains("lib.half")
                || shown_program(&d.task).contains("ratio.half, 9")
        })
        .unwrap();
    assert_eq!(reference(&half.task), "true\t4.5");
    let failing = made.admitted.iter().find(|d| d.function == "fail").unwrap();
    assert_eq!(reference(&failing.task), "false\tbad input");
    assert!(origin(&made.admitted[0]).interpreter.contains("Lua 5.1"));
}

#[test]
fn only_the_named_files_and_functions_are_drilled_and_a_stranger_is_refused() {
    let rig = rig("drills-scope");
    let mut plan = Rig::plan(2, 3);
    plan.files = vec!["shop/pricing.py".into()];
    plan.targets = Some(BTreeMap::from([(
        "shop/pricing.py".to_string(),
        BTreeSet::from(["total".to_string()]),
    )]));
    let made = rig.drills(&plan);
    assert_eq!(
        functions(&made),
        BTreeSet::from(["shop/pricing.py::total".to_string()])
    );

    let tree = rig.repo.capture("HEAD");
    for named in ["tests/test_pricing.py", "nowhere.py"] {
        let mut plan = Rig::plan(1, 3);
        plan.files = vec![named.into()];
        let refused = rig.generator.drills(&tree.snapshot, &plan).unwrap_err();
        assert!(
            matches!(refused, WorkspaceError::Parameter { name: "files", .. }),
            "{refused}"
        );
    }
}

#[test]
fn the_functions_a_reference_change_touches_are_found_in_the_revision_it_was_made_for() {
    let rig = rig("drills-enclosing");
    let tree = rig.repo.capture("HEAD");
    let patch = "\
--- a/shop/pricing.py
+++ b/shop/pricing.py
@@ -5,4 +5,4 @@ def total(prices):
     result = 0
     for price in prices:
-        result += price + 1
+        result += price
     return result
";
    let touched = touched_lines(patch);
    let within = rig
        .generator
        .enclosing_functions(&tree.snapshot, &touched)
        .unwrap();
    assert_eq!(
        within,
        BTreeMap::from([(
            "shop/pricing.py".to_string(),
            BTreeSet::from(["total".to_string()])
        )])
    );
    // Lines that are not those of the revision say so instead of guessing.
    let other = patch.replace("    for price in prices:", "    for cost in prices:");
    let refused = rig
        .generator
        .enclosing_functions(&tree.snapshot, &touched_lines(&other))
        .unwrap_err();
    assert!(
        refused.to_string().contains("drill the revision"),
        "{refused}"
    );
}

#[test]
fn the_formal_verifier_grades_the_answer_to_a_drill_by_its_fenced_output() {
    let rig = rig("drills-verifier");
    let made = rig.drills(&Rig::plan(1, 3));
    let drill = made
        .admitted
        .iter()
        .find(|d| d.function == "total")
        .unwrap();
    let verifier = ExactMatchVerifier::new(
        splinter_core::annotation::Producer {
            name: "formal".into(),
            version: "1".into(),
        },
        Normalisation::LENIENT,
    );
    let grade = |answer: &str| {
        let experience = Experience::answered_without_a_run(
            drill.task.clone(),
            answer,
            Provenance::new("scripted", &FixedClock::new(AT)),
        )
        .unwrap();
        verifier.verify(&drill.task, &experience).unwrap().outcome
    };
    let right = reference(&drill.task);
    assert_eq!(grade(&format!("```\n{right}\n```")), Outcome::Pass);
    assert_eq!(grade("```\n123456789\n```"), Outcome::Fail);
}

#[test]
fn a_drill_exports_as_a_harbor_task_that_rewards_exactly_the_reference() {
    let rig = rig("drills-harbor");
    let made = rig.drills(&Rig::plan(1, 3));
    let drill = made
        .admitted
        .iter()
        .find(|d| d.function == "total")
        .unwrap();
    let out = rig.repo.scratch.0.join("export");
    rig.generator
        .export_drill_harbor(&drill.task, &out.join("task"), "python:3.12-slim")
        .unwrap();
    let instruction = fs::read_to_string(out.join("task/instruction.md")).unwrap();
    assert!(instruction.contains(&drill.task.instruction));
    assert!(instruction.contains("answer.txt"));
    // The reference is where only the verifier looks, never in the
    // instruction or the environment.
    let expected = fs::read_to_string(out.join("task/tests/expected_output.txt")).unwrap();
    assert_eq!(expected.trim_end(), reference(&drill.task));
    assert!(!fs::read_to_string(out.join("task/environment/Dockerfile"))
        .unwrap()
        .contains(reference(&drill.task)));
    assert!(fs::read_to_string(out.join("task/task.toml"))
        .unwrap()
        .contains("splinter.drill.harbor.v1"));

    // The verifier script rewards the reference and nothing else.
    let script = out.join("task/tests/test.sh");
    let sandboxed = rig.repo.scratch.0.join("harbor-run");
    for (answer, reward) in [
        (format!("  {}\n\n", reference(&drill.task)), "1"),
        ("999".to_string(), "0"),
    ] {
        let root = sandboxed.join(reward);
        fs::create_dir_all(&root).unwrap();
        write(&root.join("app/answer.txt"), &answer);
        let text = fs::read_to_string(&script)
            .unwrap()
            .replace("/logs/verifier", root.join("logs").to_str().unwrap())
            .replace("/app", root.join("app").to_str().unwrap())
            .replace(
                "/tests/expected_output.txt",
                out.join("task/tests/expected_output.txt").to_str().unwrap(),
            );
        write(&root.join("test.sh"), &text);
        let status = Command::new("sh")
            .arg(root.join("test.sh"))
            .status()
            .unwrap();
        assert!(status.success());
        assert_eq!(
            fs::read_to_string(root.join("logs/reward.txt"))
                .unwrap()
                .trim(),
            reward
        );
    }
    // A task that is not a drill is not exported as one.
    let mut not_a_drill = drill.task.clone();
    not_a_drill
        .privileged
        .retain(|p| p.kind == PrivilegedKind::Reference);
    let refused = rig
        .generator
        .export_drill_harbor(&not_a_drill, &out.join("other"), "python:3.12-slim")
        .unwrap_err();
    assert!(
        refused.to_string().contains("not a comprehension drill"),
        "{refused}"
    );
}
