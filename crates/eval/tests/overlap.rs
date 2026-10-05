// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements held-out measurement that cannot leak, for
// its clients. If your team needs expertise in evaluating a model on
// documents it has not seen when its sources overlap, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: texts that print the same passage - two editions of one letter -
//! fall in one group, so a held-out set can hold out the whole group and no
//! near-copy of a held-out text is trained on. Texts that share only a short
//! phrase stay apart. A group is named by the index of its first text, so the
//! answer does not depend on how the texts are ordered after the first.

use splinter_eval::overlap::overlap_groups;

/// `n` distinct words, so two passages built from different seeds share none.
fn passage(seed: &str, n: usize) -> String {
    (0..n)
        .map(|i| format!("{seed}w{i}"))
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn two_editions_of_one_letter_are_one_group_whatever_surrounds_them() {
    let letter = passage("a", 120);
    let texts = [
        format!("Dear Sir, {letter} Yours."),
        passage("b", 90),
        format!("{} {letter} {}", passage("c", 15), passage("d", 20)),
    ];
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    assert_eq!(overlap_groups(&refs), [0, 1, 0]);
}

#[test]
fn a_short_shared_phrase_does_not_merge_two_texts() {
    let phrase = passage("shared", 12);
    let texts = [
        format!("{} {phrase}", passage("x", 100)),
        format!("{phrase} {}", passage("y", 100)),
    ];
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    assert_eq!(overlap_groups(&refs), [0, 1]);
}

#[test]
fn grouping_is_transitive_and_stable_under_reordering() {
    let (a, b) = (passage("p", 120), passage("q", 120));
    // The first and third share `a`, the third and fourth share `b`.
    let texts = [
        format!("{a} {}", passage("m", 10)),
        passage("n", 80),
        format!("{a} {b}"),
        format!("{} {b}", passage("o", 10)),
    ];
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    assert_eq!(overlap_groups(&refs), [0, 1, 0, 0]);
    let reversed: Vec<&str> = refs.iter().rev().copied().collect();
    assert_eq!(overlap_groups(&reversed), [0, 0, 2, 0]);
}

/// A resolution reused inside an article, a letter copied into another: the
/// shared passage may sit anywhere in either text, far past its opening.
#[test]
fn a_passage_reused_deep_inside_two_longer_texts_makes_them_one_group() {
    let reused = passage("reused", 40);
    let texts = [
        format!("{} {reused} {}", passage("own1", 900), passage("own2", 300)),
        passage("other", 500),
        format!("{} {reused}", passage("own3", 1200)),
    ];
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    assert_eq!(overlap_groups(&refs), [0, 1, 0]);
}

/// A formula of the period's letters is held by many texts and joins none
/// of them; a text that is one print of another is still its group.
#[test]
fn boilerplate_held_by_many_texts_joins_none_of_them() {
    let formula = passage("formula", 40);
    let mut texts: Vec<String> = (0..12)
        .map(|i| {
            format!(
                "{} {formula} {}",
                passage(&format!("a{i}"), 150),
                passage(&format!("b{i}"), 150)
            )
        })
        .collect();
    let letter = passage("letter", 120);
    texts.push(format!("Dear Sir, {letter}"));
    texts.push(format!("Sir, {letter} Yours."));
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let groups = overlap_groups(&refs);
    assert_eq!(groups[..12], (0..12).collect::<Vec<_>>()[..]);
    assert_eq!((groups[12], groups[13]), (12, 12));
}

#[test]
fn nothing_overlaps_in_nothing() {
    assert!(overlap_groups(&[]).is_empty());
    assert_eq!(overlap_groups(&["", "short text"]), [0, 1]);
}
