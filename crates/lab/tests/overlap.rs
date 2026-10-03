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

use splinter_lab::overlap::overlap_groups;

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

#[test]
fn nothing_overlaps_in_nothing() {
    assert!(overlap_groups(&[]).is_empty());
    assert_eq!(overlap_groups(&["", "short text"]), [0, 1]);
}
