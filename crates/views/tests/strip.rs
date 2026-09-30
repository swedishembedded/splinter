// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: what a student sees of the teacher's material.
//!
//! By default nothing: the student's turn is the instruction alone. A view
//! keeps privileged items in the student's turn only when told to - by
//! kind, or for a fixed fraction of experiences chosen by their id - and it
//! drops privileged context only from an instruction that stands on its
//! own; one that does not is excluded and counted.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use common::*;
use splinter_store::annotation::{Outcome, Strength};
use splinter_store::experience::{Experience, Privileged, PrivilegedKind};
use splinter_views::{
    check_self_contained, Corpus, Exclusion, Fraction, NotSelfContained, RecordBody, SftFinal,
    Strip, View, ViewError, MIN_QUOTED_CHARS,
};

const PASSAGE: &str = "TEACHER-ONLY passage: the store keeps every experience under its digest.";
const HINT: &str = "TEACHER-ONLY hint: think about content addressing.";
const INSTRUCTION: &str = "How does the experience store name what it keeps?";

fn solve(instruction: &str, at: &str) -> Experience {
    let task = task(
        "recall",
        instruction,
        vec![
            privileged(PrivilegedKind::Passage, PASSAGE),
            privileged(PrivilegedKind::Hint, HINT),
            privileged(PrivilegedKind::Reference, "by digest"),
        ],
    );
    answered(&task, "By its digest.", at)
}

fn corpus(experiences: &[Experience]) -> Corpus {
    let mut corpus = Corpus::new();
    for exp in experiences {
        corpus
            .insert(
                exp.clone(),
                vec![verdict(exp, Outcome::Pass, Strength::Formal)],
            )
            .unwrap();
    }
    corpus
}

/// The student's turn of every record `strip` yields from `corpus`.
fn student_turns(corpus: &Corpus, strip: Strip) -> Vec<String> {
    SftFinal::new(Strength::Formal)
        .with_strip(strip)
        .project(corpus)
        .unwrap()
        .records
        .iter()
        .map(|r| match &r.body {
            RecordBody::Chat { messages } => messages[0].content.clone(),
            other => panic!("a chat record, got {other:?}"),
        })
        .collect()
}

#[test]
fn the_student_sees_only_what_the_strip_policy_keeps() {
    let corpus = corpus(&[solve(INSTRUCTION, "2026-09-30T01:00:00.000Z")]);
    assert_eq!(student_turns(&corpus, Strip::default()), [INSTRUCTION]);
    assert_eq!(Strip::default(), Strip::All);

    let [kept] = &student_turns(&corpus, Strip::Keep(vec![PrivilegedKind::Hint]))[..] else {
        panic!("one record")
    };
    assert!(kept.contains(HINT) && kept.ends_with(INSTRUCTION), "{kept}");
    assert!(
        !kept.contains(PASSAGE) && !kept.contains("by digest"),
        "{kept}"
    );
}

#[test]
fn mix_keeps_a_fraction_chosen_by_experience_id_alone() {
    let experiences: Vec<Experience> = (0..40)
        .map(|i| solve(INSTRUCTION, &format!("2026-09-30T01:00:{i:02}.000Z")))
        .collect();
    let mix = |fraction: f64| Strip::Mix {
        keep_fraction: Fraction::new(fraction).unwrap(),
        seed: 7,
    };
    let keeps = |corpus: &Corpus, fraction: f64| -> Vec<bool> {
        student_turns(corpus, mix(fraction))
            .iter()
            .map(|turn| turn.contains(PASSAGE))
            .collect()
    };
    let forward = corpus(&experiences);
    assert!(keeps(&forward, 0.0).iter().all(|k| !k));
    assert!(keeps(&forward, 1.0).iter().all(|k| *k));

    let half = keeps(&forward, 0.5);
    assert!(
        half.iter().any(|k| *k) && half.iter().any(|k| !k),
        "{half:?}"
    );
    assert_eq!(keeps(&forward, 0.5), half, "the same every time");
    let mut reversed: Vec<Experience> = experiences.clone();
    reversed.reverse();
    let mut backward = keeps(&corpus(&reversed), 0.5);
    backward.reverse();
    assert_eq!(
        backward, half,
        "decided by the experience, not its position"
    );
    for turn in student_turns(&forward, mix(1.0)) {
        assert!(!turn.contains("by digest"), "mix never keeps the reference");
    }

    assert!(matches!(
        Fraction::new(1.5),
        Err(ViewError::Parameter { .. })
    ));
    assert!(Fraction::new(f64::NAN).is_err());
}

#[test]
fn an_instruction_that_needs_the_dropped_context_is_excluded_and_counted() {
    let passage = [privileged(PrivilegedKind::Passage, PASSAGE)];
    let dropped: Vec<&Privileged> = passage.iter().collect();
    assert_eq!(check_self_contained(INSTRUCTION, &dropped), Ok(()));
    assert!(matches!(
        check_self_contained("Summarise the passage   ABOVE in one line.", &dropped),
        Err(NotSelfContained::Refers { .. })
    ));
    let quoted = format!("Explain: \"{}\"", &PASSAGE[22..22 + MIN_QUOTED_CHARS]);
    assert!(matches!(
        check_self_contained(&quoted, &dropped),
        Err(NotSelfContained::Quotes { .. })
    ));
    let short = format!("Explain: \"{}\"", &PASSAGE[22..22 + MIN_QUOTED_CHARS - 1]);
    assert_eq!(check_self_contained(&short, &dropped), Ok(()));
    assert_eq!(
        check_self_contained("Summarise the passage above.", &[]),
        Ok(()),
        "nothing dropped, nothing missing"
    );

    let dependent = solve(
        "According to the passage above, how are experiences named?",
        "2026-09-30T02:00:00.000Z",
    );
    let corpus = corpus(&[solve(INSTRUCTION, "2026-09-30T01:00:00.000Z"), dependent]);
    let projection = SftFinal::new(Strength::Formal).project(&corpus).unwrap();
    assert_eq!(projection.records.len(), 1);
    assert_eq!(projection.count(Exclusion::NotSelfContained), 1);

    // Keeping the context it refers to makes it whole again.
    let kept = SftFinal::new(Strength::Formal)
        .with_strip(Strip::Keep(vec![
            PrivilegedKind::Passage,
            PrivilegedKind::Hint,
        ]))
        .project(&corpus)
        .unwrap();
    assert_eq!(kept.records.len(), 2);
    assert_eq!(kept.count(Exclusion::NotSelfContained), 0);
}

/// Executable checks and generated tests grade the answer, like the
/// reference: they are never context, so an instruction that shares text
/// with one (a function signature, say) or names shown code needs no
/// justification for dropping them.
#[test]
fn dropping_checks_and_tests_needs_no_justification() {
    use splinter_lab::verifiers::executable::{CHECK_KIND, OUTPUT_CHECK_KIND};
    use splinter_lab::verifiers::mutation::TEST_KIND;

    let signature = "def clamp(value, low, high): return the value clamped";
    let grading: Vec<Privileged> = [CHECK_KIND, TEST_KIND, OUTPUT_CHECK_KIND]
        .iter()
        .map(|kind| privileged(PrivilegedKind::Other((*kind).into()), signature))
        .collect();
    let dropped: Vec<&Privileged> = grading.iter().collect();
    let instruction = format!("Fix the code above so that it passes: {signature}");
    assert_eq!(check_self_contained(&instruction, &dropped), Ok(()));

    let hint = [privileged(PrivilegedKind::Hint, signature)];
    let dropped: Vec<&Privileged> = hint.iter().collect();
    assert!(
        check_self_contained(&instruction, &dropped).is_err(),
        "a dropped hint is context"
    );
}
