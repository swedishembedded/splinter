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

use std::collections::HashSet;

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

/// The most words a letter has: a stretch longer than this under a letter's
/// heading is a work or an appendix that the heading's volume prints after
/// its last letter, and is no letter.
pub const MAX_LETTER_WORDS: usize = 15_000;

/// Every letter in `text` that has a `TO <NAME>.` heading followed by a date
/// line naming a year of the founding era. A heading with no such date line
/// is not a letter's heading and stays inside the previous letter's body.
/// A letter ends at the next letter's heading or where its volume turns to
/// something that is not a letter (a book, a part, an appendix, an index),
/// whichever comes first; one that would still run past
/// [`MAX_LETTER_WORDS`] is dropped.
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

    #[allow(clippy::expect_used)]
    let section_break = Regex::new(
        r"(?m)^(?:BOOK|PART|APPENDIX|INDEX|END OF VOL|CONTENTS|OFFICIAL PAPERS|MISCELLANEOUS|NOTES ON|QUERY)\b[^\n]{0,80}$",
    )
    .expect("a constant pattern");
    let mut letters = Vec::with_capacity(headings.len());
    for (n, h) in headings.iter().enumerate() {
        let next = headings.get(n + 1).map_or(text.len(), |next| next.start);
        let from = h.body_from.min(next);
        let end = section_break
            .find(&text[from..next])
            .map_or(next, |found| from + found.start());
        let body = tidy_body(&text[from..end]);
        if body.split_whitespace().count() > MAX_LETTER_WORDS {
            continue;
        }
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
pub use splinter_sdk::measure::verifiers::quotation::words;

/// For each letter, the index of the first letter of its family: letters
/// that print the same text, found by Splinter's own overlap rule over what
/// each says after its salutation.
#[must_use]
pub fn families(letters: &[Letter]) -> Vec<usize> {
    let bodies: Vec<String> = letters
        .iter()
        .map(|letter| without_salutation(&letter.body))
        .collect();
    let texts: Vec<&str> = bodies.iter().map(String::as_str).collect();
    splinter_sdk::measure::overlap::overlap_groups(&texts)
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

/// `letters` without those whose text one of `works` prints, in whole or in
/// part (the overlap rule Splinter itself holds a split to). The works are
/// documents of their own in what Splinter learns from, so a letter they
/// print is one text there twice: an exam on it would examine what was
/// trained on, and training on it as a letter would repeat the work. Such a
/// letter is neither trained on as a letter nor examined.
#[must_use]
pub fn without_works_prints(letters: Vec<Letter>, works: &[&str]) -> Vec<Letter> {
    let bodies: Vec<String> = letters
        .iter()
        .map(|letter| without_salutation(&letter.body))
        .collect();
    let texts: Vec<&str> = bodies
        .iter()
        .map(String::as_str)
        .chain(works.iter().copied())
        .collect();
    let group = splinter_sdk::measure::overlap::overlap_groups(&texts);
    let printed: HashSet<usize> = (letters.len()..texts.len()).map(|n| group[n]).collect();
    letters
        .into_iter()
        .enumerate()
        .filter(|(n, _)| !printed.contains(&group[*n]))
        .map(|(_, letter)| letter)
        .collect()
}

/// How far the materials hold an exam family's text.
#[derive(Clone, Debug, PartialEq)]
pub struct ExamLeak {
    /// The family, by its first letter's id.
    pub family: String,
    /// Letters of the family, over all editions.
    pub letters: usize,
    /// Distinct runs of [`splinter_sdk::measure::overlap::RUN`] words the
    /// family's letters share with the materials.
    pub shared_runs: usize,
    /// The share of the family's distinct runs that is.
    pub run_share: f64,
    /// The materials (by position in the list given) Splinter's own overlap
    /// rule puts in one group with the family: the ones it would call one
    /// text with an exam letter. Runs the
    /// period's letters share as formulas do not count towards that.
    pub merged_materials: Vec<usize>,
}

/// For each exam family, how many of its runs of words the `materials` hold:
/// no material in one group with any exam family, by the overlap rule
/// Splinter itself holds its split to, when nothing Splinter learns from
/// prints any exam letter, in any edition. The runs shared are reported too,
/// formulas of the period's letters included, which the rule does not count.
#[must_use]
pub fn exam_leaks(
    letters: &[Letter],
    family: &[usize],
    seed: u64,
    materials: &[&str],
) -> Vec<ExamLeak> {
    use splinter_sdk::measure::overlap::RUN;
    let runs = |text: &str| -> HashSet<u64> {
        let w = words(text);
        w.windows(RUN)
            .map(|run| {
                let hash = blake3::hash(run.join(" ").as_bytes());
                let mut first = [0u8; 8];
                first.copy_from_slice(&hash.as_bytes()[..8]);
                u64::from_le_bytes(first)
            })
            .collect()
    };
    let held: HashSet<u64> = materials.iter().flat_map(|text| runs(text)).collect();
    let exam_roots: Vec<usize> = (0..letters.len())
        .filter(|&n| family[n] == n && is_exam_family(&letters[n].id, seed, 20))
        .collect();
    let exam_letters: Vec<usize> = (0..letters.len())
        .filter(|&m| exam_roots.contains(&family[m]))
        .collect();
    let bodies: Vec<String> = exam_letters
        .iter()
        .map(|&m| without_salutation(&letters[m].body))
        .collect();
    let texts: Vec<&str> = bodies
        .iter()
        .map(String::as_str)
        .chain(materials.iter().copied())
        .collect();
    let group = splinter_sdk::measure::overlap::overlap_groups(&texts);
    exam_roots
        .iter()
        .map(|&root| {
            let members: Vec<usize> = (0..exam_letters.len())
                .filter(|&k| family[exam_letters[k]] == root)
                .collect();
            let own: HashSet<u64> = members
                .iter()
                .flat_map(|&k| runs(&letters[exam_letters[k]].body))
                .collect();
            let shared = own.intersection(&held).count();
            let groups: HashSet<usize> = members.iter().map(|&k| group[k]).collect();
            ExamLeak {
                family: letters[root].id.clone(),
                letters: members.len(),
                shared_runs: shared,
                run_share: if own.is_empty() {
                    0.0
                } else {
                    shared as f64 / own.len() as f64
                },
                merged_materials: (exam_letters.len()..texts.len())
                    .filter(|&k| groups.contains(&group[k]))
                    .map(|k| k - exam_letters.len())
                    .collect(),
            }
        })
        .collect()
}

/// The files Splinter is pointed at, and which of the letters' files were
/// held back.
#[derive(Debug)]
pub struct Materials {
    /// `(file name, text)` of every letter file written.
    pub letters: Vec<(String, String)>,
    /// Letter files left out because Splinter's overlap rule would call one
    /// text with an exam letter, by file name.
    pub held_back: Vec<String>,
    /// How far the files written hold each exam family: none is one text
    /// with any of them.
    pub leaks: Vec<ExamLeak>,
}

/// [`materials`] cleared of everything that overlaps an exam letter: a file
/// that the overlap rule, run over the exam letters and the files together,
/// groups with one is held back, and the check is repeated until none is.
/// Two printings are grouped by the same rule that groups them for the
/// split, but not over the same set of texts, so a formula the period's
/// letters share can make a file one text with an exam letter that the
/// families did not.
///
/// # Errors
/// A work (which is never held back) that is one text with an exam letter.
pub fn materials_clear_of_exam(
    letters: &[Letter],
    family: &[usize],
    works: &[&str],
    seed: u64,
) -> anyhow::Result<Materials> {
    let mut files = materials(letters, family, seed);
    let mut held_back = Vec::new();
    loop {
        let texts: Vec<&str> = files
            .iter()
            .map(|(_, text)| text.as_str())
            .chain(works.iter().copied())
            .collect();
        let leaks = exam_leaks(letters, family, seed, &texts);
        let merged: HashSet<usize> = leaks
            .iter()
            .flat_map(|leak| leak.merged_materials.iter().copied())
            .collect();
        if merged.is_empty() {
            return Ok(Materials {
                letters: files,
                held_back,
                leaks,
            });
        }
        if let Some(work) = merged.iter().find(|&&k| k >= files.len()) {
            anyhow::bail!(
                "work {} is one text with an exam letter: the exam would be on what is trained on",
                work - files.len()
            );
        }
        let mut kept = Vec::with_capacity(files.len());
        for (k, file) in files.into_iter().enumerate() {
            if merged.contains(&k) {
                held_back.push(file.0);
            } else {
                kept.push(file);
            }
        }
        files = kept;
    }
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

/// Jefferson's own works that are not letters, as files under `resources`.
/// The Life and Morals of Jesus of Nazareth is not among them: it is the
/// Gospels cut and arranged, with a modern editor's introduction, and what it
/// would teach is not Jefferson's prose.
pub const OWN_WORKS: [&str; 4] = [
    "notes-on-the-state-of-virginia-1853",
    "a-summary-view-of-the-rights-of-british-america-1774",
    "manual-of-parliamentary-practice-1820",
    "declaration-of-independence",
];

/// Each of [`OWN_WORKS`] present under `resources`, as `(file stem, text
/// without its Gutenberg wrapper)`.
///
/// # Errors
/// A work file that exists but cannot be read.
pub fn own_works(resources: &std::path::Path) -> anyhow::Result<Vec<(String, String)>> {
    let mut works = Vec::new();
    for stem in OWN_WORKS {
        let path = resources
            .join("thomas-jefferson")
            .join(format!("{stem}.txt"));
        if !path.is_file() {
            continue;
        }
        let text = std::fs::read_to_string(&path)
            .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
        works.push((stem.to_string(), strip_gutenberg(&text).to_string()));
    }
    Ok(works)
}

/// Every letter the clean editions in `resources` print that no work of
/// [`OWN_WORKS`] prints too ([`without_works_prints`]), in a fixed order.
///
/// # Errors
/// An edition or work file that exists but cannot be read.
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
    let works = own_works(resources)?;
    let texts: Vec<&str> = works.iter().map(|(_, text)| text.as_str()).collect();
    Ok(without_works_prints(letters, &texts))
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
        let filler =
            "the people are the only safe depositories of their own liberty. ".repeat(2000);
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
        let is_exam =
            |n: usize, seed: u64| family[n] == n && is_exam_family(&letters[n].id, seed, 20);
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
