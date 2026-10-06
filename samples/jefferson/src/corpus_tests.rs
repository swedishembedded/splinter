// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The corpus's specs.

use std::collections::BTreeSet;

use super::*;

const WASHINGTON: &str = "\
Preface text.

TO JAMES MADISON.

                                   Paris, March 15, 1789.

Dear Sir,--I wrote you last on the 12th of January; since which I have
received yours of October the 17th, December the 8th and 12th. That of
October the 17th came to hand only February the 23d. How it happened to
be four months on the way I cannot tell.

TO JOHN JAY.

                                   Paris, May 4, 1789.

Sir,

The Assembly of the Notables has dissolved itself and the nation looks
to the States General for a constitution founded on the consent of the
governed, and I find the temper of the people favourable to it.
";

const RANDOLPH: &str = "\
TO James Madison.

Paris, March 15, 1789.

Dear Sir,--I wrote you last on the 12th of January; since which I have
received yours of October the 17th, December the 8th, and 12th. That of
October the 17th came to hand only February the 23rd. How it happened to
be four months on the way, I cannot tell.
";

#[test]
fn letters_are_parsed_with_recipient_place_year_and_body() {
    let letters = parse_letters("washington-v3", WASHINGTON);
    assert_eq!(letters.len(), 2);
    assert_eq!(letters[0].recipient, "James Madison");
    assert_eq!(letters[0].place, "Paris");
    assert_eq!(letters[0].year, 1789);
    assert!(letters[0].body.starts_with("Dear Sir,--I wrote you last"));
    assert!(
        !letters[0].body.contains("TO JOHN JAY"),
        "a letter ends at the next"
    );
    assert_eq!(letters[1].recipient, "John Jay");
    assert_eq!(letters[1].id, "washington-v3-1");
}

#[test]
fn a_letter_ends_where_the_volume_turns_to_something_that_is_not_a_letter() {
    let text = "\
TO JOHN JAY.

Paris, May 4, 1789.

Sir, the Assembly of the Notables has dissolved itself and the nation looks
to the States General for a constitution.

BOOK III.

OFFICIAL PAPERS.

Reports and opinions that run on for pages and are no part of any letter.

INDEX TO VOL. VII.

Abbreviations, 12.
";
    let letters = parse_letters("washington-v7", text);
    assert_eq!(letters.len(), 1);
    assert!(letters[0].body.contains("States General"));
    assert!(
        !letters[0].body.contains("OFFICIAL PAPERS"),
        "{:?}",
        letters[0].body
    );
    assert!(!letters[0].body.contains("Reports and opinions"));
    assert!(!letters[0].body.contains("Abbreviations"));
}

#[test]
fn a_stretch_too_long_to_be_a_letter_is_not_one() {
    let filler = "the people are the only safe depositories of their own liberty. ".repeat(2000);
    let text = format!(
        "TO JOHN JAY.\n\nParis, May 4, 1789.\n\n{filler}\n\nTO JAMES MADISON.\n\nParis, May 5, 1789.\n\nDear Sir, a short note.\n"
    );
    let letters = parse_letters("w", &text);
    assert_eq!(
        letters.len(),
        1,
        "the work glued under a heading is no letter"
    );
    assert_eq!(letters[0].recipient, "James Madison");
}

#[test]
fn windows_line_endings_parse_like_unix_ones() {
    let unix = parse_letters("w", WASHINGTON);
    let dos = parse_letters("w", &WASHINGTON.replace('\n', "\r\n"));
    assert_eq!(dos, unix);
}

#[test]
fn a_heading_without_a_date_line_is_not_a_letter() {
    let text = "TO THE READER.\n\nThis is a preface and has no date.\n\nTO JOHN JAY.\n\nParis, May 4, 1789.\n\nSir, hello there friend of mine.\n";
    let letters = parse_letters("x", text);
    assert_eq!(letters.len(), 1);
    assert_eq!(letters[0].recipient, "John Jay");
}

#[test]
fn a_salutation_is_not_part_of_what_the_letter_says() {
    assert_eq!(
        without_salutation("Dear Sir,--I wrote you last on the 12th of January and more."),
        "I wrote you last on the 12th of January and more."
    );
    assert_eq!(
        without_salutation("Sir,\n\nThe Assembly dissolved itself today."),
        "The Assembly dissolved itself today."
    );
}

#[test]
fn two_printings_of_one_letter_are_one_family() {
    let mut letters = parse_letters("washington-v3", WASHINGTON);
    letters.extend(parse_letters("randolph-v1", RANDOLPH));
    let family = families(&letters);
    assert_eq!(family[0], family[2], "the Madison letter in both editions");
    assert_ne!(family[0], family[1], "the Jay letter is another letter");
    assert_eq!(family[2], 0, "a family is named by its first letter");
}

/// The materials Splinter is pointed at hold one printing of every
/// training letter and nothing of any exam family, in any edition.
#[test]
fn the_materials_hold_one_file_per_training_family_and_none_of_the_exam() {
    let mut letters = parse_letters("washington-v3", WASHINGTON);
    letters.extend(parse_letters("randolph-v1", RANDOLPH));
    let family = families(&letters);
    for seed in 0..40 {
        let files = materials(&letters, &family, seed);
        let trained: std::collections::HashSet<usize> = (0..letters.len())
            .filter(|&n| family[n] == n && !is_exam_family(&letters[n].id, seed, 20))
            .collect();
        assert_eq!(files.len(), trained.len(), "seed {seed}");
        for (name, text) in &files {
            let owner = letters
                .iter()
                .position(|l| name.starts_with(&l.id))
                .unwrap();
            assert!(trained.contains(&owner), "{name} is of an exam family");
            assert!(text.starts_with("To ") && text.contains(&letters[owner].body));
        }
    }
}

const NOTES: &str = "The Mississippi is the largest river that waters the western country and \
    carries to the sea the produce of many states, and the Ohio joins it from the east with a \
    volume of water that no other tributary can match in all that wide and fertile land.";

#[test]
fn a_letter_that_a_work_prints_is_not_a_letter_of_the_corpus() {
    let mut letters = parse_letters("washington-v3", WASHINGTON);
    let quoting = format!(
        "TO JAMES MONROE.\n\nParis, June 1, 1789.\n\nDear Sir, {NOTES} and so I remain yours.\n"
    );
    letters.extend(parse_letters("randolph-v1", &quoting));
    assert_eq!(letters.len(), 3);
    let kept = without_works_prints(letters, &[NOTES]);
    let recipients: Vec<&str> = kept.iter().map(|l| l.recipient.as_str()).collect();
    assert_eq!(recipients, ["James Madison", "John Jay"]);
}

/// Nothing the materials hold shares a passage with any exam letter, in
/// any edition, however the files are chosen.
#[test]
fn the_materials_share_no_passage_with_an_exam_family() {
    let mut letters = parse_letters("washington-v3", WASHINGTON);
    letters.extend(parse_letters("randolph-v1", RANDOLPH));
    let family = families(&letters);
    for seed in 0..40 {
        let files = materials(&letters, &family, seed);
        let texts: Vec<&str> = files.iter().map(|(_, text)| text.as_str()).collect();
        let leaks = exam_leaks(&letters, &family, seed, &texts);
        assert!(
            leaks.iter().all(|leak| leak.merged_materials.is_empty()),
            "seed {seed}: {leaks:?}"
        );
    }
}

#[test]
fn a_file_the_overlap_rule_calls_one_text_with_an_exam_letter_is_held_back() {
    let mut letters = parse_letters("washington-v3", WASHINGTON);
    letters.extend(parse_letters("randolph-v1", RANDOLPH));
    // Every letter its own family: the second printing of the Madison
    // letter is not known to be the first's.
    let family: Vec<usize> = (0..letters.len()).collect();
    let is_exam = |n: usize, seed: u64| is_exam_family(&letters[n].id, seed, 20);
    let seed = (0..500)
        .find(|&s| is_exam(0, s) && !is_exam(2, s) && !is_exam(1, s))
        .unwrap();
    let before = materials(&letters, &family, seed);
    assert!(before
        .iter()
        .any(|(name, _)| name.starts_with("randolph-v1-0")));
    let cleared = materials_clear_of_exam(&letters, &family, &[], seed).unwrap();
    assert_eq!(cleared.held_back, ["randolph-v1-0.txt"]);
    assert!(cleared
        .letters
        .iter()
        .all(|(name, _)| !name.starts_with("randolph-v1-0")));
    assert!(cleared.leaks.iter().all(|l| l.merged_materials.is_empty()));
}

#[test]
fn a_file_that_prints_an_exam_letter_is_a_leak_the_check_reports() {
    let letters = parse_letters("washington-v3", WASHINGTON);
    let family = families(&letters);
    let is_exam = |n: usize, seed: u64| family[n] == n && is_exam_family(&letters[n].id, seed, 20);
    // A seed whose exam holds the first letter's family.
    let seed = (0..200).find(|&s| is_exam(0, s)).unwrap();
    let printing = letters[0].body.clone();
    let leaks = exam_leaks(&letters, &family, seed, &[printing.as_str()]);
    assert!(
        leaks.iter().any(|leak| leak.family == letters[0].id
            && leak.run_share > 0.9
            && leak.merged_materials == [0]),
        "{leaks:?}"
    );
}

#[test]
fn the_exam_pool_is_the_families_the_materials_leave_out() {
    let mut letters = parse_letters("washington-v3", WASHINGTON);
    letters.extend(parse_letters("randolph-v1", RANDOLPH));
    let family = families(&letters);
    for seed in 0..40 {
        let trained: BTreeSet<String> = materials(&letters, &family, seed)
            .into_iter()
            .map(|(name, _)| name)
            .collect();
        let pool: BTreeSet<String> = exam_pool(&letters, &family, seed)
            .into_iter()
            .map(|(name, _)| name)
            .collect();
        assert!(trained.is_disjoint(&pool), "seed {seed}");
        assert_eq!(
            trained.len() + pool.len(),
            2,
            "every family is one or the other"
        );
    }
}

#[test]
fn the_exam_split_depends_only_on_the_family_and_the_seed() {
    let exam = (0..1000)
        .filter(|n| is_exam_family(&format!("family-{n}"), 7, 20))
        .count();
    assert!((150..250).contains(&exam), "about a fifth, got {exam}");
    for n in 0..50 {
        let key = format!("family-{n}");
        assert_eq!(is_exam_family(&key, 7, 20), is_exam_family(&key, 7, 20));
    }
    let moved = (0..200)
        .filter(|n| {
            let key = format!("family-{n}");
            is_exam_family(&key, 7, 20) != is_exam_family(&key, 8, 20)
        })
        .count();
    assert!(moved > 0, "another seed is another split");
}

#[test]
fn gutenberg_boilerplate_is_stripped() {
    let text = "header\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\nthe book\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\nlicense";
    assert_eq!(strip_gutenberg(text).trim(), "the book");
    assert_eq!(strip_gutenberg("plain"), "plain");
}
