// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec of the rationalisation filters, on small trajectories written here:
//! the hint section round-trips out of an instruction, and each filter
//! passes what it must and rejects what it must, with the reason.

#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;

use serde_json::{json, Value};

use super::*;

const INSTRUCTION: &str = "# Paste above\n\nCopy line two and paste it above line one of \
`main.txt`.\n\n**Starter - `main.txt`**\n\n```text\ntop line\nbottom line\n```\n";

const HINT: &str = "#!/usr/bin/env bash\n# Reference solution: one call of the vim tool, as the \
tool makes it.\nvim -Nu NONE -i NONE -n -es -Z -S script\nargs main.txt\nnormal! jyyggP\nwqall\n";

const SYSTEM: &str =
    "You are an agent. Run `vim -Nu NONE -i NONE -n -es -Z -S <script>` via the tool.";

/// A user turn, then one agent step per `(reasoning, commands, reply)`.
fn trajectory(user: &str, steps: &[(&str, &[&str], &str)]) -> atif::Trajectory {
    let mut all = vec![json!({"step_id": 1, "source": "user", "message": user})];
    for (n, (reasoning, commands, reply)) in steps.iter().enumerate() {
        let id = format!("c{n}");
        let mut step = json!({
            "step_id": n + 2, "source": "agent", "message": "", "reasoning_content": reasoning,
        });
        if !commands.is_empty() {
            step["tool_calls"] = json!([{"tool_call_id": id, "function_name": "vim-vim",
                "arguments": {"commands": commands}}]);
            step["observation"] = json!({"results": [{"source_call_id": id, "content": reply}]});
        }
        all.push(step);
    }
    serde_json::from_value(json!({
        "schema_version": "ATIF-v1.7",
        "agent": {"name": "sven", "version": "1"},
        "steps": all,
    }))
    .unwrap()
}

fn task_files(expected: Option<&[(&str, &str)]>) -> TaskFiles {
    let map = |pairs: &[(&str, &str)]| -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(p, c)| ((*p).to_string(), (*c).to_string()))
            .collect()
    };
    TaskFiles {
        starter: map(&[
            ("main.txt", "top line\nbottom line\n"),
            ("keep.txt", "same\n"),
        ]),
        expected: expected.map(map),
    }
}

fn target() -> TaskFiles {
    task_files(Some(&[
        ("main.txt", "bottom line\ntop line\nbottom line\n"),
        ("keep.txt", "same\n"),
    ]))
}

const READ: &[&str] = &["args main.txt", "%print"];
const READ_REPLY: &str = "exit status: 0\noutput:\ntop line\nbottom line";
const EDIT: &[&str] = &["args main.txt", "normal! jyyggP", "wqall"];
const EDIT_REPLY: &str = "exit status: 0\nmessages:\n\"main.txt\" 3L, 33B written";

fn terms() -> Vec<String> {
    LEAK_TERMS.iter().map(|t| (*t).to_string()).collect()
}

fn leak(trajectory: &atif::Trajectory) -> Verdict {
    let hint = hinted_instruction(INSTRUCTION, HINT);
    let section = &hint[hint.find(HINT_BEGIN).unwrap()..];
    let terms = terms();
    no_leak(
        trajectory,
        &LeakCheck {
            terms: &terms,
            hint: section,
            instruction: INSTRUCTION,
            shown: SYSTEM,
            files: &target(),
        },
    )
}

#[test]
fn the_hint_section_is_stripped_back_to_the_instruction_exactly() {
    for instruction in [INSTRUCTION, INSTRUCTION.trim_end(), "one line"] {
        let staged = hinted_instruction(instruction, HINT);
        assert!(staged.starts_with(instruction.trim_end()));
        assert!(staged.contains(HINT_BEGIN) && staged.contains(HINT_END));
        assert!(staged.contains(HINT_PREAMBLE) && staged.contains("normal! jyyggP"));
        assert_eq!(
            strip_hint(&staged).unwrap().trim_end(),
            instruction.trim_end()
        );
    }
    // A hint holding a code fence is fenced by a longer one.
    let fenced = hinted_instruction("x", "```\ncode\n```");
    assert!(fenced.contains("````\n```\ncode\n```\n````"), "{fenced}");
    // No section, two sections, or an end before the begin are refused.
    assert!(strip_hint(INSTRUCTION).is_err());
    let twice = format!(
        "{}{}",
        hinted_instruction("a", "b"),
        hinted_instruction("", "c")
    );
    assert!(strip_hint(&twice).is_err());
    assert!(strip_hint(&format!("x\n{HINT_END}\ny\n{HINT_BEGIN}\n")).is_err());
    // The notes section carries the notes and the instruction.
    let notes = notes_instruction(INSTRUCTION, "Step 1: read the file.");
    assert!(notes.starts_with(INSTRUCTION.trim_end()) && notes.contains(NOTES_BEGIN));
    assert!(notes.contains("Step 1: read the file.") && !notes.contains(HINT_BEGIN));
}

#[test]
fn only_a_reward_of_one_is_verified() {
    assert!(verified(Some(1.0)).passed);
    for reward in [Some(0.0), Some(0.5), None] {
        let verdict = verified(reward);
        assert!(!verdict.passed && verdict.reason.is_some(), "{reward:?}");
    }
}

/// An instruction that names the file but does not show it: the model has seen nothing of it.
const UNSHOWN: &str = "# Paste above\n\nCopy line two and paste it above line one of `main.txt`.\n";

/// A file the instruction itself shows (a vim task prints the starting files) is observed before any tool call:
/// editing it in the first call is not editing blind, and the base model's own unhinted passes do exactly that
/// (155 of 221, knowledge 200). A file the instruction does not show must still be read first.
#[test]
fn a_file_the_instruction_shows_counts_as_observed() {
    let files = target();
    let blind = trajectory(INSTRUCTION, &[("edit", EDIT, EDIT_REPLY)]);
    assert_eq!(
        read_before_write(&blind, &files, INSTRUCTION),
        Verdict::pass()
    );
    assert!(!read_before_write(&blind, &files, UNSHOWN).passed);
    let partly = "Copy line two of `main.txt`:\n\n```text\ntop line\n```\n";
    assert!(
        !read_before_write(&blind, &files, partly).passed,
        "one of its lines unshown is not shown"
    );
}

#[test]
fn the_first_call_must_read_every_changed_file_before_writing_it() {
    let files = target();
    assert_eq!(files.changed(), Some(vec!["main.txt"]));
    let good = trajectory(
        INSTRUCTION,
        &[("read it", READ, READ_REPLY), ("edit", EDIT, EDIT_REPLY)],
    );
    assert_eq!(read_before_write(&good, &files, UNSHOWN), Verdict::pass());
    // A first call that edits blind.
    let blind = trajectory(INSTRUCTION, &[("edit", EDIT, EDIT_REPLY)]);
    let verdict = read_before_write(&blind, &files, UNSHOWN);
    assert!(!verdict.passed, "{verdict:?}");
    // A first call that prints the file and writes it in the same call.
    let both: Vec<&str> = READ.iter().chain(EDIT).copied().collect();
    let same = trajectory(
        INSTRUCTION,
        &[("all", &both, &format!("{READ_REPLY}\n{EDIT_REPLY}"))],
    );
    let verdict = read_before_write(&same, &files, UNSHOWN);
    assert!(!verdict.passed && verdict.reason.unwrap().contains("writes"));
    // A first call that reads only part of the file.
    let part = trajectory(
        INSTRUCTION,
        &[
            ("peek", &["args main.txt", "1print"], "output:\ntop line"),
            ("edit", EDIT, EDIT_REPLY),
        ],
    );
    assert!(!read_before_write(&part, &files, UNSHOWN).passed);
    // No tool call at all, and a task without target files.
    assert!(
        !read_before_write(
            &trajectory(INSTRUCTION, &[("think", &[], "")]),
            &files,
            UNSHOWN
        )
        .passed
    );
    let undecidable = read_before_write(&good, &files_without_target(), UNSHOWN);
    assert!(!undecidable.passed && undecidable.reason.unwrap().contains("target"));
    // A task that only creates a file has nothing to read first.
    let create = task_files(Some(&[
        ("main.txt", "top line\nbottom line\n"),
        ("keep.txt", "same\n"),
        ("new.txt", "hello\n"),
    ]));
    assert_eq!(create.changed(), Some(vec!["new.txt"]));
    let write_new = trajectory(
        INSTRUCTION,
        &[(
            "make it",
            &["edit new.txt", "wq"],
            "\"new.txt\" [New] 1L, 6B written",
        )],
    );
    assert_eq!(
        read_before_write(&write_new, &create, UNSHOWN),
        Verdict::pass()
    );
    // A removed file is a changed file too.
    let removed = task_files(Some(&[("main.txt", "top line\nbottom line\n")]));
    assert_eq!(removed.changed(), Some(vec!["keep.txt"]));
}

fn files_without_target() -> TaskFiles {
    task_files(None)
}

#[test]
fn a_mention_of_a_leak_term_in_the_models_own_text_is_a_leak() {
    let clean = trajectory(
        INSTRUCTION,
        &[
            ("I print main.txt to see it.", READ, READ_REPLY),
            (
                "Line two goes above line one; yank it and put it above.",
                EDIT,
                EDIT_REPLY,
            ),
        ],
    );
    assert_eq!(leak(&clean), Verdict::pass());
    for said in [
        "The expected result has three lines.",
        "As given, I run normal.",
        "Following the Solution, I yank.",
        "solve.sh does it in one call.",
        "The reference says jyyggP.",
        "The hint shows the commands.",
    ] {
        let leaky = trajectory(INSTRUCTION, &[(said, READ, READ_REPLY)]);
        let verdict = leak(&leaky);
        assert!(!verdict.passed, "{said}");
        assert!(verdict.reason.unwrap().contains("step 2"), "{said}");
    }
    // The same rule counts the terms an attempt's own text mentions.
    let said = trajectory(
        INSTRUCTION,
        &[("The Expected result, as given.", READ, READ_REPLY)],
    );
    assert_eq!(
        mentioned_terms(&said, &terms()),
        vec!["as given", "expected"]
    );
    assert!(mentioned_terms(&clean, &terms()).is_empty());
    // A word that merely contains a term later in it is no mention.
    assert!(
        leak(&trajectory(
            INSTRUCTION,
            &[("An unexpected reply.", READ, READ_REPLY)]
        ))
        .passed
    );
    // Tool arguments and tool replies are not the model's text.
    let in_tools = trajectory(
        INSTRUCTION,
        &[(
            "I print the file.",
            &["args main.txt", "%s/solution/x/ge", "%print"],
            "output:\nexpected reference hint",
        )],
    );
    assert!(leak(&in_tools).passed);
}

#[test]
fn repeating_the_hints_own_wording_is_a_leak_unless_the_model_was_shown_or_sent_it() {
    // A comment line of the hint, repeated.
    let repeated = trajectory(
        INSTRUCTION,
        &[(
            "This is one call of the vim tool, as the tool makes it.",
            READ,
            READ_REPLY,
        )],
    );
    let verdict = leak(&repeated);
    assert!(!verdict.passed && verdict.reason.unwrap().contains("wording"));
    // The preamble the hint section opens with is the hint's wording too.
    let preamble = trajectory(
        INSTRUCTION,
        &[(
            "A worked answer to this task follows, so I copy it.",
            READ,
            READ_REPLY,
        )],
    );
    assert!(!leak(&preamble).passed);
    // The wrapper line is also in the system prompt the model was shown.
    let shown = trajectory(
        INSTRUCTION,
        &[(
            "The tool runs vim -Nu NONE -i NONE -n -es -Z -S on my script.",
            READ,
            READ_REPLY,
        )],
    );
    assert!(leak(&shown).passed);
}

#[test]
fn quoting_file_content_nothing_showed_yet_is_a_leak() {
    let extra = task_files(Some(&[(
        "main.txt",
        "bottom line\ntop line\nbottom line\nunseen trailer line\n",
    )]));
    let terms = terms();
    let check = |t: &atif::Trajectory| {
        no_leak(
            t,
            &LeakCheck {
                terms: &terms,
                hint: "",
                instruction: INSTRUCTION,
                shown: SYSTEM,
                files: &extra,
            },
        )
    };
    // The target's last line is in neither the instruction nor a reply.
    let early = trajectory(
        INSTRUCTION,
        &[("It must end with unseen trailer line.", READ, READ_REPLY)],
    );
    let verdict = check(&early);
    assert!(!verdict.passed && verdict.reason.unwrap().contains("unseen trailer line"));
    // Once a reply showed it, quoting it is observation.
    let seen = trajectory(
        INSTRUCTION,
        &[
            (
                "Print it.",
                READ,
                "output:\ntop line\nbottom line\nunseen trailer line",
            ),
            ("So it ends with unseen trailer line.", EDIT, EDIT_REPLY),
        ],
    );
    assert!(check(&seen).passed);
    // The instruction shows the starting lines.
    let from_task = trajectory(
        INSTRUCTION,
        &[("Line two is bottom line.", READ, READ_REPLY)],
    );
    assert!(check(&from_task).passed);
}

#[test]
fn the_kept_trajectory_holds_the_instruction_and_no_part_of_the_hint() {
    let staged = hinted_instruction(INSTRUCTION, HINT);
    let hinted = trajectory(
        &staged,
        &[("Read.", READ, READ_REPLY), ("Edit.", EDIT, EDIT_REPLY)],
    );
    assert!(hint_residue(&hinted).is_some());
    let kept = kept_trajectory(&hinted, INSTRUCTION).unwrap();
    assert_eq!(kept.steps[0].message.as_text(), Some(INSTRUCTION));
    assert_eq!(kept.steps.len(), hinted.steps.len());
    assert_eq!(hint_residue(&kept), None);
    let text = serde_json::to_string(&kept).unwrap();
    assert!(!text.contains(HINT_PREAMBLE) && !text.contains("Reference solution"));
    // Another task's instruction under the hint is refused.
    assert!(kept_trajectory(&hinted, "another task").is_err());
    // A hint echoed anywhere else is refused.
    let mut echoed = hinted.clone();
    echoed.steps[1].message = atif::MessageBody::text(staged.clone());
    assert!(kept_trajectory(&echoed, INSTRUCTION).is_err());
}

#[test]
fn the_reasoning_shown_to_a_sufficiency_run_is_the_models_text_alone() {
    let t = trajectory(
        INSTRUCTION,
        &[
            ("Read the file first.", READ, READ_REPLY),
            ("Yank line two.", EDIT, EDIT_REPLY),
        ],
    );
    let notes = reasoning_of(&t);
    assert!(notes.contains("Read the file first.") && notes.contains("Yank line two."));
    assert!(notes.find("Read the file").unwrap() < notes.find("Yank line").unwrap());
    // No tool argument and no tool reply.
    assert!(!notes.contains("normal! jyyggP") && !notes.contains("written"));
    let _: Value = json!(notes);
}
