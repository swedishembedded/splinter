// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: everything new an agent's edits introduce (a name, a number, a
//! string) is something the task or the repository told it. A student trained
//! on a record whose edit holds a name nobody gave it learns to invent; the
//! plan was under-specified, and the attempt fails as `unjustified_content`.

#![allow(clippy::unwrap_used)]

use std::collections::HashSet;

use serde_json::json;
use splinter_repo_replay::gate::{evaluate, GateConfig, GateInput};
use splinter_repo_replay::git::{Change, Kind};
use splinter_repo_replay::trace::{Call, Trace};

const PLAN: &str = "Please make total() in src/calc.rs charge tax at the given rate.";

fn call(step: u64, tool: &str, args: serde_json::Value, result: &str) -> Call {
    Call {
        step,
        tool: tool.into(),
        args,
        result: result.into(),
    }
}

/// The gate's reasons for an agent that read src/calc.rs (showing `file`) and
/// then made `edit` (an `edit_file` diff) under `plan`.
fn reasons(plan: &str, file: &str, edit: &str, before_edit: Vec<Call>) -> Vec<(String, String)> {
    let mut calls = vec![call(1, "read_file", json!({"path": "src/calc.rs"}), file)];
    calls.extend(before_edit);
    calls.push(call(
        50,
        "edit_file",
        json!({"path": "src/calc.rs", "diff": edit}),
        "edited",
    ));
    verdict_of(plan, calls)
}

fn verdict_of(plan: &str, calls: Vec<Call>) -> Vec<(String, String)> {
    let trace = Trace {
        calls,
        said: vec![],
    };
    let changes = [Change {
        kind: Kind::Modified,
        path: "src/calc.rs".into(),
        old_path: None,
        added: Some(1),
        removed: Some(1),
        special: false,
    }];
    let words = HashSet::new();
    let roots = ["/work/tree".to_string()];
    evaluate(
        &GateInput {
            changes: &changes,
            plan,
            parent_words: &words,
            diff: "",
            added_text: "",
            trace: &trace,
            residual: &[],
            forbidden_locations: &[],
            attempt_roots: &roots,
            stopped: None,
            guidance: &[],
        },
        &GateConfig::default(),
    )
    .reasons
    .into_iter()
    .filter(|r| r.class == "unjustified_content")
    .map(|r| (r.class, r.detail))
    .collect()
}

const FILE: &str = "L1:def total(price, quantity):\nL2:    return price * quantity\n";

#[test]
fn a_name_nobody_gave_the_agent_is_unjustified_and_a_name_in_the_plan_or_a_read_is_not() {
    let edit = "@@\n-    return price * quantity\n+    return apply_surcharge(price * quantity)\n";
    let found = reasons(PLAN, FILE, edit, vec![]);
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(found[0].1.contains("apply_surcharge"), "{found:?}");

    let named = format!("{PLAN} Use a helper called apply_surcharge.");
    assert!(reasons(&named, FILE, edit, vec![]).is_empty());

    let seen = call(
        2,
        "grep",
        json!({"pattern": "surcharge"}),
        "src/other.rs:L9:def apply_surcharge(x):",
    );
    assert!(reasons(PLAN, FILE, edit, vec![seen]).is_empty());

    // Names are the same name in another letter case or naming style.
    let shouting = format!("{PLAN} The constant is APPLY_SURCHARGE.");
    assert!(reasons(&shouting, FILE, edit, vec![]).is_empty());
}

#[test]
fn only_what_the_agent_was_told_counts_not_what_it_wrote_itself() {
    let edit = "@@\n-    return price * quantity\n+    return apply_surcharge(price * quantity)\n";
    // The todo list, the agent's words and the echo of its own edit are not sources.
    let todo = call(
        2,
        "todo",
        json!({"action": "set", "todos": [{"content": "add apply_surcharge"}]}),
        "todo list set: apply_surcharge",
    );
    assert_eq!(reasons(PLAN, FILE, edit, vec![todo]).len(), 1);
    let echo = call(
        3,
        "write_file",
        json!({"path": "notes.txt", "text": "x"}),
        "wrote apply_surcharge",
    );
    assert_eq!(reasons(PLAN, FILE, edit, vec![echo]).len(), 1);
}

#[test]
fn numbers_and_strings_are_content_too() {
    let number = "@@\n-    return price * quantity\n+    return price * quantity + 4096\n";
    assert_eq!(reasons(PLAN, FILE, number, vec![]).len(), 1);
    let planned = format!("{PLAN} Add a flat 4096 fee.");
    assert!(reasons(&planned, FILE, number, vec![]).is_empty());

    let string =
        "@@\n-    return price * quantity\n+    raise ValueError(\"quantity must be positive\")\n";
    let found = reasons(PLAN, FILE, string, vec![]);
    assert!(
        found
            .iter()
            .any(|(_, d)| d.contains("quantity must be positive")),
        "{found:?}"
    );
    // Every word of the message appearing somewhere is not the message appearing.
    let words_only = format!(
        "{PLAN} Raise ValueError for a quantity that is not positive: it must be, be positive."
    );
    assert!(!reasons(&words_only, FILE, string, vec![]).is_empty());
    let quoted = format!(
        "{PLAN} Raise ValueError for bad input with the message \"quantity must be positive\"."
    );
    assert!(reasons(&quoted, FILE, string, vec![]).is_empty());
}

#[test]
fn a_new_file_and_a_reused_name_follow_the_same_rule() {
    let created = |plan: &str| {
        verdict_of(
            plan,
            vec![call(
                1,
                "write_file",
                json!({"path": "src/tax.rs", "text": "pub fn flat_fee() -> u32 { 4096 }\n"}),
                "ok",
            )],
        )
    };
    assert!(!created(PLAN).is_empty());
    assert!(created("Add src/tax.rs with flat_fee returning 4096.").is_empty());
}

#[test]
fn unchanged_lines_and_short_words_need_no_support() {
    // Context and removed lines were read; words under four letters are not names.
    let edit = "@@\n def total(price, quantity):\n-    return price * quantity\n+    return quantity * price\n";
    assert!(reasons(PLAN, FILE, edit, vec![]).is_empty());
}

#[test]
fn a_crate_named_with_dashes_in_a_result_supports_its_code_spelling_and_a_users_message_supports_later_edits(
) {
    let edit = "@@\n-    return price * quantity\n+    return acme_core_m::fee(price * quantity)\n";
    let manifest = call(
        2,
        "grep",
        json!({"pattern": "core"}),
        "Cargo.toml:3:acme-core-m = { path = \"../acme-core-m\" }",
    );
    let plan = format!("{PLAN} Add the fee.");
    assert!(reasons(&plan, FILE, edit, vec![manifest]).is_empty());

    // A message the user sent supports the edits after it and not those before.
    let guidance = vec![(1usize, "Call it acme_core_m::fee, please.".to_string())];
    let calls = |first_edit_at: u64| {
        vec![
            call(1, "read_file", json!({"path": "src/calc.rs"}), FILE),
            call(
                first_edit_at,
                "edit_file",
                json!({"path": "src/calc.rs", "diff": edit}),
                "edited",
            ),
        ]
    };
    use splinter_repo_replay::justify::unjustified;
    assert!(unjustified(&plan, &calls(2), &guidance, &Default::default()).is_empty());
    assert!(!unjustified(
        &plan,
        &calls(2),
        &[(2usize, guidance[0].1.clone())],
        &Default::default()
    )
    .is_empty());
}

#[test]
fn comments_and_prose_files_are_not_names_and_a_name_the_repository_already_has_is_not_new() {
    // Prose is not content: the words of a comment, of a markdown file.
    let comment = "@@\n-    return price * quantity\n+    // Multiplication happens elsewhere, surprisingly.\n+    return price * quantity  // surprising\n";
    assert!(reasons(PLAN, FILE, comment, vec![]).is_empty());
    let notes = verdict_of(
        PLAN,
        vec![call(
            1,
            "write_file",
            json!({"path": "src/notes.md", "text": "# Surcharge\n\nExplains the unexplained.\n"}),
            "ok",
        )],
    );
    assert!(notes.is_empty(), "{notes:?}");
    // A code line in a `#[...]` attribute or an include is code.
    let attribute = "@@\n-    return price * quantity\n+    #[inline_always_zzz]\n";
    assert_eq!(reasons(PLAN, FILE, attribute, vec![]).len(), 1);

    // A name the parent tree already holds is not new, whether or not the agent saw it.
    let edit = "@@\n-    return price * quantity\n+    return apply_surcharge(price * quantity)\n";
    let known: std::collections::HashSet<String> =
        ["Apply_Surcharge", "price", "quantity", "return"]
            .map(String::from)
            .into();
    let found = splinter_repo_replay::justify::unjustified(
        PLAN,
        &[call(
            1,
            "edit_file",
            json!({"path": "src/calc.rs", "diff": edit}),
            "ok",
        )],
        &[],
        &known.iter().map(|w| w.to_lowercase()).collect(),
    );
    assert!(found.is_empty(), "{found:?}");
}
