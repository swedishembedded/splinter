// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements provenance-tracked training corpora for
// language models for its clients. If your team needs expertise in building
// leakage-free datasets from public-domain sources, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Jefferson's letters, parsed out of the editions that print them, grouped
//! into families (the same letter as two editions print it) and split into
//! training and exam by family.
//!
//! The editions overlap heavily: a letter in the Washington edition is often
//! in the Randolph edition and the Library edition too, with different line
//! breaks and small editorial changes. A split by file, or by position, would
//! train on one printing of a letter and examine on another. The unit of the
//! split is therefore the family, found from the words themselves.

use std::collections::{HashMap, HashSet};

use regex::Regex;

/// One letter as an edition prints it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Letter {
    /// `<edition>-<n>`: stable for a given corpus.
    pub id: String,
    /// The edition's short name, such as `washington-v3`.
    pub edition: String,
    /// The person written to, title-cased: `James Madison`.
    pub recipient: String,
    /// The place written from, title-cased: `Paris`.
    pub place: String,
    /// The year written.
    pub year: u16,
    /// The letter's text after its date line: paragraphs separated by a blank
    /// line, lines within a paragraph joined by a space.
    pub body: String,
}

/// The text between a Project Gutenberg file's start and end markers; the
/// whole text when it carries none.
#[must_use]
pub fn strip_gutenberg(text: &str) -> &str {
    let start = text
        .find("*** START OF")
        .and_then(|at| text[at..].find('\n').map(|nl| at + nl + 1))
        .unwrap_or(0);
    let end = text.find("*** END OF").unwrap_or(text.len());
    if start < end {
        &text[start..end]
    } else {
        text
    }
}

/// Every letter in `text` that has a `TO <NAME>.` heading followed by a date
/// line naming a year of the founding era. A heading with no such date line
/// is not a letter's heading and stays inside the previous letter's body.
///
/// # Panics
/// Never: the patterns are constants.
#[must_use]
pub fn parse_letters(edition: &str, text: &str) -> Vec<Letter> {
    // Gutenberg ships Windows line endings; the heading pattern is line-anchored.
    let text = &text.replace("\r\n", "\n");
    #[allow(clippy::expect_used)]
    let heading = Regex::new(r"(?m)^(?:TO|To) ([A-Z][A-Za-z.,' -]{2,70})\.[ \t]*$")
        .expect("a constant pattern");
    #[allow(clippy::expect_used)]
    let year = Regex::new(r"\b(1[78]\d\d)\b").expect("a constant pattern");

    let mut headings = Vec::new();
    for caps in heading.captures_iter(text) {
        let whole = caps.get(0).map_or(0..0, |m| m.range());
        let name = caps.get(1).map_or("", |m| m.as_str());
        let after = &text[whole.end..];
        // The date line is the first non-blank line, within a few lines.
        let date_line = after
            .split('\n')
            .skip(1)
            .take(4)
            .map(str::trim)
            .find(|line| !line.is_empty());
        let Some(date_line) = date_line else { continue };
        let Some(found) = year.find(date_line) else {
            continue;
        };
        let Ok(parsed_year) = found.as_str().parse::<u16>() else {
            continue;
        };
        let line_at = after.find(date_line).map_or(0, |at| whole.end + at);
        headings.push(Heading {
            start: whole.start,
            body_from: line_at + date_line.len(),
            recipient: title_case(name),
            place: place_of(date_line),
            year: parsed_year,
        });
    }

    let mut letters = Vec::with_capacity(headings.len());
    for (n, h) in headings.iter().enumerate() {
        let end = headings.get(n + 1).map_or(text.len(), |next| next.start);
        let body = tidy_body(&text[h.body_from.min(end)..end]);
        letters.push(Letter {
            id: format!("{edition}-{n}"),
            edition: edition.to_string(),
            recipient: h.recipient.clone(),
            place: h.place.clone(),
            year: h.year,
            body,
        });
    }
    letters
}

struct Heading {
    start: usize,
    body_from: usize,
    recipient: String,
    place: String,
    year: u16,
}

/// `JAMES MADISON` or `James Madison` as `James Madison`; small joining words
/// stay lower case.
fn title_case(name: &str) -> String {
    name.split_whitespace()
        .map(|word| {
            let lower = word.to_lowercase();
            if matches!(lower.as_str(), "of" | "and" | "the") {
                lower
            } else {
                let mut chars = lower.chars();
                chars
                    .next()
                    .map(|c| c.to_uppercase().collect::<String>() + chars.as_str())
                    .unwrap_or_default()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// The place in `Paris, March 15, 1789.`: the text before the first comma.
fn place_of(date_line: &str) -> String {
    title_case(date_line.split(',').next().unwrap_or("").trim())
}

/// A letter's text with its lines joined into paragraphs and its page-number
/// debris dropped.
fn tidy_body(raw: &str) -> String {
    let mut paragraphs: Vec<String> = Vec::new();
    for block in raw.split("\n\n") {
        let joined = block.split_whitespace().collect::<Vec<_>>().join(" ");
        if joined.is_empty() || joined.chars().all(|c| c.is_ascii_digit() || c == '.') {
            continue;
        }
        paragraphs.push(joined);
    }
    paragraphs.join("\n\n")
}

/// `body` without its leading salutation (`Dear Sir,--` or a short opening
/// line of its own), so an excerpt starts at what the letter says.
#[must_use]
pub fn without_salutation(body: &str) -> String {
    let first_end = body.find("\n\n").unwrap_or(body.len());
    let first = &body[..first_end];
    if first.split_whitespace().count() <= 5 {
        return body[first_end..].trim_start().to_string();
    }
    if let Some(at) = first.find(",--").or_else(|| first.find(",\u{2014}")) {
        if at <= 40 {
            let skip = at + if first[at..].starts_with(",--") { 3 } else { 4 };
            return body[skip..].trim_start().to_string();
        }
    }
    body.to_string()
}

/// Lower-case alphanumeric words of `text`: what two printings of a letter
/// agree on once spacing, punctuation and capitals are set aside.
#[must_use]
pub fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// For each letter, the index of the first letter of its family: letters that
/// share enough sampled eight-word runs are one letter printed twice.
#[must_use]
pub fn families(letters: &[Letter]) -> Vec<usize> {
    const RUN: usize = 8;
    const SAMPLE_EVERY: u64 = 8;
    const WINDOW: usize = 700;
    const SHARED_TO_MERGE: usize = 3;
    const MAX_HOLDERS: usize = 6;

    let mut holders: HashMap<u64, Vec<usize>> = HashMap::new();
    for (n, letter) in letters.iter().enumerate() {
        let w = words(&without_salutation(&letter.body));
        let limit = w.len().saturating_sub(RUN).min(WINDOW);
        let mut seen = HashSet::new();
        for at in 0..limit {
            let h = hash_run(&w[at..at + RUN]);
            if h.is_multiple_of(SAMPLE_EVERY) && seen.insert(h) {
                holders.entry(h).or_default().push(n);
            }
        }
    }

    let mut shared: HashMap<(usize, usize), usize> = HashMap::new();
    for group in holders.values().filter(|g| g.len() <= MAX_HOLDERS) {
        for (i, &a) in group.iter().enumerate() {
            for &b in &group[i + 1..] {
                *shared.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
    }

    let mut parent: Vec<usize> = (0..letters.len()).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for (&(a, b), &count) in &shared {
        if count >= SHARED_TO_MERGE {
            let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
            if ra != rb {
                parent[ra.max(rb)] = ra.min(rb);
            }
        }
    }
    (0..letters.len()).map(|n| find(&mut parent, n)).collect()
}

fn hash_run(run: &[String]) -> u64 {
    let digest = blake3::hash(run.join(" ").as_bytes());
    let mut first = [0u8; 8];
    first.copy_from_slice(&digest.as_bytes()[..8]);
    u64::from_le_bytes(first)
}

/// Whether the family named `key` belongs to the exam: a seeded hash of the
/// key, so the answer never depends on file order or on which editions are
/// present, and `percent` of families land in the exam.
#[must_use]
pub fn is_exam_family(key: &str, seed: u64, percent: u64) -> bool {
    let mut input = seed.to_le_bytes().to_vec();
    input.extend_from_slice(key.as_bytes());
    let digest = blake3::hash(&input);
    let mut first = [0u8; 8];
    first.copy_from_slice(&digest.as_bytes()[..8]);
    u64::from_le_bytes(first) % 100 < percent
}

/// The files Splinter is pointed at: for each family of letters that is not
/// in the exam, the one printing the family is named by, as `(file name,
/// text)`. A letter of an exam family is in no file in any edition, so what
/// Splinter learns from leaves the exam unseen.
#[must_use]
pub fn materials(letters: &[Letter], family: &[usize], seed: u64) -> Vec<(String, String)> {
    letters
        .iter()
        .enumerate()
        .filter(|(n, letter)| family[*n] == *n && !is_exam_family(&letter.id, seed, 20))
        .map(|(_, letter)| {
            let place = if letter.place.is_empty() {
                String::new()
            } else {
                format!("{}, ", letter.place)
            };
            (
                format!("{}.txt", letter.id),
                format!(
                    "To {}\n{place}{}\n\n{}\n",
                    letter.recipient, letter.year, letter.body
                ),
            )
        })
        .collect()
}

/// The clean editions under `resources/thomas-jefferson` that print letters
/// with a `TO <NAME>.` heading: `(file stem, edition short name)`.
fn letter_editions() -> Vec<(String, String)> {
    let mut editions = Vec::new();
    for n in 1..=9 {
        editions.push((
            format!("writings-washington-ed-v{n}"),
            format!("washington-v{n}"),
        ));
    }
    for n in 1..=4 {
        editions.push((
            format!("memoir-correspondence-miscellanies-v{n}"),
            format!("randolph-v{n}"),
        ));
    }
    editions.push((
        "writings-library-ed-v6".to_string(),
        "library-v6".to_string(),
    ));
    editions
}

/// Every letter the clean editions in `resources` print, in a fixed order.
///
/// # Errors
/// An edition file that exists but cannot be read.
pub fn load_letters(resources: &std::path::Path) -> anyhow::Result<Vec<Letter>> {
    let mut letters = Vec::new();
    for (stem, edition) in letter_editions() {
        let path = resources
            .join("thomas-jefferson")
            .join(format!("{stem}.txt"));
        if !path.is_file() {
            continue;
        }
        let text = std::fs::read_to_string(&path)
            .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
        letters.extend(parse_letters(&edition, strip_gutenberg(&text)));
    }
    Ok(letters)
}

#[cfg(test)]
mod tests {
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
}
