// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: the tree an agent works in holds the commit's parent and nothing
//! of the future; the plan and the records never hold the answer.

#![allow(clippy::unwrap_used)]

use splinter_repo_replay::fixture;
use splinter_repo_replay::git::{mirror, single_commit_repo, Git};
use splinter_repo_replay::mask::mask_diff;
use splinter_repo_replay::plan::{validate, PlanCheck, PlanRejection};
use splinter_repo_replay::serving_prompt::SERVING_PROMPT;
use splinter_repo_replay::style::PlanStyle;

#[test]
fn the_attempt_tree_is_a_single_commit_without_the_future() {
    let work = tempfile::tempdir().unwrap();
    let source = work.path().join("source");
    std::fs::create_dir(&source).unwrap();
    let ids = fixture::build(&source).unwrap();
    let git = mirror(&source, &work.path().join("mirror.git")).unwrap();
    let tree = work.path().join("tree");
    std::fs::create_dir(&tree).unwrap();
    git.export_tree(&ids[1], &tree).unwrap();
    single_commit_repo(&tree).unwrap();

    let attempt = Git::at(&tree);
    let commits = attempt.text(&["rev-list", "--all"]).unwrap();
    assert_eq!(
        commits.lines().count(),
        1,
        "one commit and no other ref: {commits}"
    );
    assert_eq!(attempt.text(&["remote"]).unwrap().trim(), "", "no remote");
    for later in &ids[2..] {
        assert!(
            attempt.bytes(&["cat-file", "-e", later]).is_err(),
            "{later} must not exist in the tree"
        );
    }
    assert!(
        tree.join("shapes.py").is_file(),
        "the parent's files are there"
    );
    assert!(!std::fs::read_to_string(tree.join("calc.py"))
        .unwrap()
        .contains("compute_total_with_tax"));
}

#[test]
fn a_plan_must_not_carry_the_answer() {
    let commit = "0123456789abcdef0123456789abcdef01234567";
    let lines = vec![
        "    result = subtotal + subtotal * rate - discount + rounding_adjustment".to_string(),
        "def compute_total_with_tax(subtotal, rate):".to_string(),
    ];
    let check = PlanCheck {
        diff_lines: &lines,
        commit,
        min_line_tokens: 6,
    };
    let good = "Please make total() include the tax rate in the price it returns, because customers see prices without tax today and that surprises them at checkout.";
    validate(good, PlanStyle::Detailed, &check).unwrap();

    let class = |plan: &str, style| validate(plan, style, &check).unwrap_err().class();
    let copied = format!(
        "{good} Use this: result = subtotal + subtotal * rate - discount + rounding_adjustment"
    );
    assert_eq!(class(&copied, PlanStyle::Detailed), "plan_verbatim_diff");
    let marker = format!("{good}\n@@ -1,3 +1,4 @@");
    assert_eq!(class(&marker, PlanStyle::Detailed), "plan_diff_marker");
    let sha = format!("{good} See 0123456.");
    assert_eq!(class(&sha, PlanStyle::Detailed), "plan_diff_marker");
    let refers = format!("{good} Do what the diff does.");
    assert_eq!(class(&refers, PlanStyle::Detailed), "plan_refers_to_change");
    assert_eq!(class("Fix it.", PlanStyle::Detailed), "plan_length");
    // Names in backticks are what a good request is made of.
    let named = "Change `total` so that it includes `rate` in what it returns, as `compute_total_with_tax` does elsewhere in the project.";
    validate(named, PlanStyle::Terse, &check).unwrap();
    // A fenced block is code, and too much of it is the answer.
    let fenced = "Make total include tax:\n```python\ndef total(price, quantity, rate):\n    subtotal = price * quantity\n    return subtotal + subtotal * rate\n```\nThat is all.";
    assert!(matches!(
        validate(fenced, PlanStyle::Terse, &check),
        Err(PlanRejection::CodeDensity { .. })
    ));
    // A long span quoted from a changed line is the answer quoted.
    let quoted = "Make total return `result = subtotal + subtotal * rate - discount + rounding_adjustment` for every caller of the function, please do so now.";
    assert_eq!(class(quoted, PlanStyle::Terse), "plan_verbatim_diff");
}

#[test]
fn the_serving_prompt_is_the_one_a_solver_runs_under() {
    assert_eq!(
        SERVING_PROMPT,
        splinter_sdk::vocabulary::prompt::CODING_AGENT_PROMPT
    );
}

#[test]
fn the_diff_the_agent_sees_has_no_offsets_and_no_unchanged_lines() {
    let diff = "diff --git a/calc.py b/calc.py\nindex 1111111..2222222 100644\n--- a/calc.py\n+++ b/calc.py\n@@ -5,7 +5,7 @@ def total(price):\n def add(a, b):\n     return a + b\n \n-def old_name(x):\n+def new_name(x):\n     return x\n \n";
    let masked = mask_diff(diff, &|_| None).unwrap();
    let masked = masked.as_str();
    assert!(!masked.contains("@@ -"), "{masked}");
    assert!(
        !masked.contains("def total") && !masked.contains("return a + b"),
        "{masked}"
    );
    assert!(!masked.contains("index "), "{masked}");
    for kept in [
        "diff --git a/calc.py b/calc.py",
        "--- a/calc.py",
        "+++ b/calc.py",
        "@@ @@",
        "-def old_name(x):",
        "+def new_name(x):",
        " xxx",
    ] {
        assert!(masked.contains(kept), "{kept} missing from {masked}");
    }
}

const CALC: &str = "def add(a, b):\n    return a + b\n\n\ndef total(price):\n    return price\n\n\ndef other(x):\n    return x\n";

/// An insertion between the two functions: no removed line, three unchanged lines each side.
const INSERTION: &str = "diff --git a/calc.py b/calc.py\n--- a/calc.py\n+++ b/calc.py\n@@ -2,6 +2,10 @@\n     return a + b\n \n \n+def sub(a, b):\n+    return a - b\n+\n+\n def total(price):\n     return price\n \n";

#[test]
fn an_insertion_keeps_the_unchanged_lines_that_make_its_place_unique() {
    let parent = |path: &str| (path == "calc.py").then(|| CALC.to_string());
    let masked = mask_diff(INSERTION, &parent).unwrap();
    let text = masked.as_str();
    // Blank lines alone occur everywhere; with the line that follows them the place is one.
    assert!(text.contains(" def total(price):"), "{text}");
    assert!(text.contains("+def sub(a, b):"), "{text}");
    assert!(!text.contains("@@ -"), "no offsets: {text}");
    // The shown old-side lines occur once in the file.
    let shown: Vec<&str> = text
        .lines()
        .filter(|l| l.starts_with(' ') && *l != " xxx")
        .map(|l| &l[1..])
        .collect();
    let file: Vec<&str> = CALC.lines().collect();
    let count = file
        .windows(shown.len())
        .filter(|w| **w == shown[..])
        .count();
    assert_eq!(count, 1, "{shown:?}");
}

#[test]
fn a_hunk_whose_place_cannot_be_determined_is_refused() {
    // Both functions are the same text: an insertion between them has no unique place.
    let twin = "def f():\n    pass\n\n\ndef f():\n    pass\n\n\n";
    let diff = "diff --git a/t.py b/t.py\n--- a/t.py\n+++ b/t.py\n@@ -1,3 +1,4 @@\n     pass\n \n+# note\n \n";
    let parent = |_: &str| Some(twin.to_string());
    let refused = mask_diff(diff, &parent).unwrap_err();
    assert_eq!(refused.path, "t.py");
}
