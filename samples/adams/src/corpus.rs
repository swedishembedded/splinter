// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements provenance-tracked training corpora for
// language models for its clients. If your team needs expertise in turning
// scanned editions into evidence a model can be trained on and audited
// against, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The documents of Cushing's edition, parsed out of its OCR text.
//!
//! An entry is a heading, the editor's bracketed note saying where the text
//! comes from, usually a dateline, and the text. The OCR keeps the running
//! heads of every page, splits words across lines and doubles spaces, so the
//! parser works on normalised lines. A source note it cannot turn into an
//! entry is reported as skipped, with the reason: nothing is dropped silently.

use crate::document::Date;
use regex::Regex;
use std::sync::LazyLock;

/// One document as the edition prints it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    /// `<edition>-<n>`, in order of appearance.
    pub id: String,
    pub edition: String,
    /// The heading as printed, upper case, its lines joined by a space.
    pub heading: String,
    /// Who it was written to, title-cased, when the heading is `TO <NAME>`.
    pub recipient: Option<String>,
    /// The editor's bracketed source note, whitespace normalised.
    pub note: String,
    pub date: Date,
    /// Paragraphs separated by a blank line; words split by a hyphen at a
    /// line end are rejoined.
    pub body: String,
}

/// A source note that did not become an entry, and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Skipped {
    pub line: usize,
    pub note: String,
    pub reason: &'static str,
}

#[derive(Debug, Default)]
pub struct Parsed {
    pub entries: Vec<Entry>,
    pub skipped: Vec<Skipped>,
}

/// Page furniture the OCR keeps: `1765] SAMUEL ADAMS. 27`, `28 THE WRITINGS OF
/// [1765`, the same split over two lines, and the bare signature. None of it
/// is the text, and none of it is part of a heading.
static RUNNING_HEAD: LazyLock<Regex> = LazyLock::new(|| {
    #[allow(clippy::expect_used)]
    Regex::new(r"^(?:\d+ )?(?:THE WRITINGS OF(?: \[?[\di]\d{3})?|\[[\di]\d{3}|[\di]\d{3}\] SAMUEL ADAMS\.?(?: \d+)?|SAMUEL ADAMS\.?(?: \d+)?)$|^\d{1,4}$")
        .expect("a constant pattern")
});

/// The year a running head announces, `i765` being the OCR's `1765`.
static PAGE_YEAR: LazyLock<Regex> = LazyLock::new(|| {
    #[allow(clippy::expect_used)]
    Regex::new(r"(?:^([\di]\d{3})\] SAMUEL ADAMS|THE WRITINGS OF \[([\di]\d{3})$|^\[([\di]\d{3})$)")
        .expect("a constant pattern")
});

static YEAR: LazyLock<Regex> = LazyLock::new(|| {
    #[allow(clippy::expect_used)]
    Regex::new(r"\b([1i][78]\d\d)\b").expect("a constant pattern")
});

/// The longest a dateline can be: a place, a date and little else.
const MAX_DATELINE_CHARS: usize = 70;
/// How far below a note a dateline is looked for, and how far a note runs.
const DATELINE_REACH: usize = 3;
const NOTE_REACH: usize = 4;
/// Blank lines tolerated between a heading and its note, and between the
/// lines of a wrapped heading.
const GAP_BEFORE_NOTE: usize = 3;
const GAP_WITHIN_HEADING: usize = 1;

/// Month names and the abbreviations of the period, as the OCR prints them.
const MONTHS: [(&str, u8); 28] = [
    ("january", 1),
    ("jan", 1),
    ("february", 2),
    ("feb", 2),
    ("march", 3),
    ("mar", 3),
    ("april", 4),
    ("apr", 4),
    ("may", 5),
    ("june", 6),
    ("july", 7),
    ("august", 8),
    ("september", 9),
    ("october", 10),
    ("november", 11),
    ("december", 12),
    ("nov", 11),
    ("novr", 11),
    ("decr", 12),
    ("dec", 12),
    ("octr", 10),
    ("oct", 10),
    ("septr", 9),
    ("sept", 9),
    ("sep", 9),
    ("augt", 8),
    ("aug", 8),
    ("feby", 2),
];

/// Parse one volume of the edition.
pub fn parse_cushing(edition: &str, text: &str) -> Parsed {
    let lines: Vec<String> = text
        .lines()
        .map(|l| l.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect();
    let page_years = page_years(&lines);
    let mut parsed = Parsed::default();
    let mut found = Vec::new();

    let mut at = 0;
    while at < lines.len() {
        if !is_note_start(&lines[at]) {
            at += 1;
            continue;
        }
        let end = (at..lines.len().min(at + NOTE_REACH))
            .find(|&k| lines[k].contains(']'))
            .unwrap_or(at);
        let note = lines[at..=end].join(" ");
        let (heading, heading_start) = heading_above(&lines, at);
        if heading.is_empty() {
            parsed.skipped.push(Skipped {
                line: at + 1,
                note,
                reason: "no heading above the source note",
            });
        } else {
            found.push(Found {
                heading,
                heading_start,
                note,
                note_end: end,
                page_year: page_years[at],
            });
        }
        at = end + 1;
    }

    add_letters_without_notes(&lines, &page_years, &mut found);
    found.sort_by_key(|f| f.heading_start);

    for (n, f) in found.iter().enumerate() {
        let boundary = found
            .get(n + 1)
            .map_or(lines.len(), |next| next.heading_start);
        let (dateline, body_from) = dateline_after(&lines, f.note_end, boundary);
        let date = dateline
            .as_deref()
            .and_then(date_of)
            .or_else(|| {
                year_in(&f.heading).map(|year| Date {
                    year,
                    month: None,
                    day: None,
                })
            })
            .or_else(|| printed_date(&f.note))
            .or_else(|| {
                f.page_year.map(|year| Date {
                    year,
                    month: None,
                    day: None,
                })
            });
        let Some(date) = date else {
            parsed.skipped.push(Skipped {
                line: f.note_end + 1,
                note: f.note.clone(),
                reason: "no year in the dateline, the heading or the page's running head",
            });
            continue;
        };
        parsed.entries.push(Entry {
            id: format!("{edition}-{}", parsed.entries.len()),
            edition: edition.to_string(),
            recipient: recipient_of(&f.heading),
            heading: f.heading.clone(),
            note: f.note.clone(),
            date,
            body: body_of(&lines[body_from.min(boundary)..boundary]),
        });
    }
    parsed
}

/// Letters the editor gave no source note: a `TO <NAME>` heading line that has
/// a dateline below it, outside any heading already found.
fn add_letters_without_notes(lines: &[String], page_years: &[Option<u16>], found: &mut Vec<Found>) {
    let taken: Vec<(usize, usize)> = found
        .iter()
        .map(|f| (f.heading_start, f.note_end))
        .collect();
    for at in 0..lines.len() {
        let line = lines[at].as_str();
        if !line.starts_with("TO ")
            || !is_heading_line(line)
            || taken.iter().any(|&(from, to)| (from..=to).contains(&at))
        {
            continue;
        }
        let (dateline, _) = dateline_after(lines, at, lines.len());
        if dateline.is_some() {
            found.push(Found {
                heading: line.to_string(),
                heading_start: at,
                note: String::new(),
                note_end: at,
                page_year: page_years[at],
            });
        }
    }
}

struct Found {
    heading: String,
    heading_start: usize,
    note: String,
    note_end: usize,
    page_year: Option<u16>,
}

fn is_note_start(line: &str) -> bool {
    line.strip_prefix('[')
        .and_then(|rest| rest.chars().next())
        .is_some_and(char::is_alphabetic)
}

/// The year of the latest running head at or before each line.
fn page_years(lines: &[String]) -> Vec<Option<u16>> {
    let mut year = None;
    lines
        .iter()
        .map(|line| {
            if let Some(found) = PAGE_YEAR
                .captures(line)
                .and_then(|c| c.iter().skip(1).flatten().next())
            {
                year = parse_year(found.as_str()).or(year);
            }
            year
        })
        .collect()
}

fn parse_year(text: &str) -> Option<u16> {
    let fixed = text.replacen('i', "1", 1);
    fixed
        .parse::<u16>()
        .ok()
        .filter(|y| (1700..=1810).contains(y))
}

fn year_in(text: &str) -> Option<u16> {
    YEAR.captures(text).and_then(|c| parse_year(&c[1]))
}

fn is_furniture(line: &str) -> bool {
    RUNNING_HEAD.is_match(line)
}

/// Lowercase letters a heading may carry for every ten letters, plus one: the
/// OCR leaves a stray one in `TO G w.` and a footnote mark as `l`.
const STRAY_LOWERCASE_PER_TEN: usize = 1;
const MIN_HEADING_LETTERS: usize = 3;

/// A heading is upper case, give or take OCR noise, with at least three letters.
fn is_heading_line(line: &str) -> bool {
    let letters = line.chars().filter(|c| c.is_alphabetic()).count();
    let lowercase = line.chars().filter(|c| c.is_lowercase()).count();
    line.len() <= 110
        && !is_furniture(line)
        && !line.starts_with('[')
        && letters >= MIN_HEADING_LETTERS
        && lowercase <= 1 + letters / 10 * STRAY_LOWERCASE_PER_TEN
}

/// A line with too few letters to be words: a rule, a mark, a stray digit.
fn is_noise(line: &str) -> bool {
    line.chars().filter(|c| c.is_alphabetic()).count() < MIN_HEADING_LETTERS
}

/// The upper-case lines directly above a note, joined, and the line the
/// heading starts on. Page furniture between them is stepped over.
fn heading_above(lines: &[String], note_at: usize) -> (String, usize) {
    let mut parts: Vec<&str> = Vec::new();
    let mut start = note_at;
    let mut gap = 0;
    let mut k = note_at;
    while k > 0 {
        k -= 1;
        let line = lines[k].as_str();
        if is_noise(line) || is_furniture(line) {
            gap += 1;
            if gap
                > if parts.is_empty() {
                    GAP_BEFORE_NOTE
                } else {
                    GAP_WITHIN_HEADING
                }
            {
                break;
            }
        } else if is_heading_line(line) {
            parts.push(line);
            start = k;
            gap = 0;
        } else {
            break;
        }
    }
    parts.reverse();
    (parts.join(" "), start)
}

/// The dateline just below a note, if there is one, and the line the text
/// starts on.
fn dateline_after(lines: &[String], note_end: usize, boundary: usize) -> (Option<String>, usize) {
    let first = (note_end + 1..boundary.min(note_end + 1 + DATELINE_REACH))
        .find(|&k| !lines[k].is_empty() && !is_furniture(&lines[k]));
    match first {
        Some(k) if lines[k].len() <= MAX_DATELINE_CHARS && year_in(&lines[k]).is_some() => {
            (Some(lines[k].clone()), k + 1)
        }
        _ => (None, note_end + 1),
    }
}

fn date_of(dateline: &str) -> Option<Date> {
    let year = year_in(dateline)?;
    let tokens: Vec<String> = dateline
        .split_whitespace()
        .map(|t| {
            t.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .collect();
    let month_at = tokens
        .iter()
        .position(|t| MONTHS.iter().any(|(name, _)| name == t));
    let month = month_at.and_then(|at| {
        MONTHS
            .iter()
            .find(|(name, _)| *name == tokens[at])
            .map(|&(_, m)| m)
    });
    let day = month_at
        .and_then(|at| tokens.get(at + 1))
        .and_then(|t| t.parse::<u8>().ok())
        .filter(|d| (1..=31).contains(d));
    Some(Date { year, month, day })
}

/// A printed source's note dates the piece (`[Boston Gazette, January 9,
/// 1769.]`); a manuscript's does not, its years being those of publication.
fn printed_date(note: &str) -> Option<Date> {
    if note.to_lowercase().starts_with("[ms") {
        return None;
    }
    date_of(note)
}

fn recipient_of(heading: &str) -> Option<String> {
    let name = heading
        .strip_prefix("TO ")?
        .trim_end_matches(|c: char| c.is_ascii_digit() || c == '.' || c == ' ');
    (!name.is_empty()).then(|| title_case(name))
}

fn title_case(upper: &str) -> String {
    upper
        .split(' ')
        .map(|word| {
            let mut chars = word.chars();
            chars.next().map_or_else(String::new, |first| {
                first
                    .to_uppercase()
                    .chain(chars.flat_map(char::to_lowercase))
                    .collect()
            })
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Paragraphs of the entry's lines with the page furniture removed, words
/// split by a hyphen rejoined, and a paragraph a page break cut in two mended.
fn body_of(lines: &[String]) -> String {
    let mut paragraphs: Vec<String> = Vec::new();
    let mut current = String::new();
    let flush = |current: &mut String, paragraphs: &mut Vec<String>| {
        if !current.is_empty() {
            paragraphs.push(std::mem::take(current));
        }
    };
    for line in lines {
        if is_furniture(line) {
            continue;
        }
        if line.is_empty() {
            flush(&mut current, &mut paragraphs);
        } else if current.ends_with('-') && line.chars().next().is_some_and(char::is_lowercase) {
            current.pop();
            current.push_str(line);
        } else {
            if !current.is_empty() {
                current.push(' ');
            }
            current.push_str(line);
        }
    }
    flush(&mut current, &mut paragraphs);
    let mut mended: Vec<String> = Vec::new();
    for paragraph in paragraphs {
        let cut = mended.last().is_some_and(|last| {
            !last.ends_with(['.', '?', '!', ':', ';', '"', ')', ']'])
                && paragraph.chars().next().is_some_and(char::is_lowercase)
        });
        match mended.last_mut() {
            Some(last) if cut => {
                last.push(' ');
                last.push_str(&paragraph);
            }
            _ => mended.push(paragraph),
        }
    }
    mended.join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A letter as the OCR prints it, running heads and a split word inside.
    const DARTMOUTH: &str = "\
TO    REVEREND    G W


[MS.,  Collections  of  the  Earl  of  Dartmouth.]

BOSTON  NoV  n  1765

REVD  SIR

Our  good  Friend  Mr  Jonathan  Mason  has  com
municated  to  us  a  Letter.


1765]  SAMUEL  ADAMS.  27

We  are  the  Descend-
ents  of  Ancestors  remarkeable  for  their  Zeal.

28  THE  WRITINGS  OF  [1765

Another  Paragraph  here.

SAMUEL  ADAMS.


TO    JOHN    SMITH.


[MS.,  Samuel  Adams  Papers,  Lenox  Library.]

BOSTON,  April  2,  1766.

Dear  Sir,  the  second  letter.
";

    fn parse(text: &str) -> Parsed {
        parse_cushing("cushing-1", text)
    }

    #[test]
    fn a_letter_becomes_an_entry_with_its_heading_note_date_and_text() {
        let parsed = parse(DARTMOUTH);
        let e = &parsed.entries[0];
        assert_eq!(e.id, "cushing-1-0");
        assert_eq!(e.heading, "TO REVEREND G W");
        assert_eq!(e.recipient.as_deref(), Some("Reverend G W"));
        assert_eq!(e.note, "[MS., Collections of the Earl of Dartmouth.]");
        assert_eq!(
            e.date,
            Date {
                year: 1765,
                month: Some(11),
                day: None
            }
        );
        assert!(e.body.starts_with("REVD SIR"), "{}", e.body);
    }

    #[test]
    fn a_hyphen_at_a_line_end_is_rejoined_and_paragraphs_are_kept() {
        let body = &parse(DARTMOUTH).entries[0].body;
        assert!(body.contains("Descendents of Ancestors"), "{body}");
        assert_eq!(body.matches("\n\n").count(), 3, "{body}");
    }

    #[test]
    fn the_pages_running_heads_and_the_signature_are_not_part_of_the_text() {
        let body = &parse(DARTMOUTH).entries[0].body;
        for stray in ["WRITINGS", "SAMUEL ADAMS", "1765]", "[1765"] {
            assert!(!body.contains(stray), "{stray} in {body}");
        }
    }

    #[test]
    fn entries_split_at_the_next_heading_and_keep_their_own_text() {
        let parsed = parse(DARTMOUTH);
        assert_eq!(parsed.entries.len(), 2);
        let second = &parsed.entries[1];
        assert_eq!(second.id, "cushing-1-1");
        assert_eq!(second.recipient.as_deref(), Some("John Smith"));
        assert_eq!(
            second.date,
            Date {
                year: 1766,
                month: Some(4),
                day: Some(2)
            }
        );
        assert_eq!(second.body, "Dear Sir, the second letter.");
        assert!(!parsed.entries[0].body.contains("second letter"));
    }

    #[test]
    fn an_entry_without_a_dateline_takes_the_year_of_the_page_it_is_on() {
        let text = "1770]  SAMUEL  ADAMS.  101\n\nTHE  BOSTON  MASSACRE.\n\n[Boston  Gazette,  March  12.]\n\nThe  text  of  the  piece.\n";
        let e = &parse(text).entries[0];
        assert_eq!(e.date.year, 1770);
        assert_eq!(e.recipient, None);
        assert_eq!(e.body, "The text of the piece.");
    }

    #[test]
    fn a_year_in_a_wrapped_heading_dates_the_entry_and_the_heading_is_joined() {
        let text = "INSTRUCTIONS    OF  THE    TOWN    OF  BOSTON    TO  ITS    REPRE\nSENTATIVES  IN  THE  GENERAL  COURT.     MAY,  1764.\n\n[MS.,  Boston  Public  Library.]\n\nGentlemen,  you  are  chosen.\n";
        let e = &parse(text).entries[0];
        assert_eq!(e.heading, "INSTRUCTIONS OF THE TOWN OF BOSTON TO ITS REPRE SENTATIVES IN THE GENERAL COURT. MAY, 1764.");
        assert_eq!(e.date.year, 1764);
    }

    #[test]
    fn a_source_note_with_no_date_anywhere_is_reported_not_dropped() {
        let text = "TO  NOBODY.\n\n[MS.,  Somewhere.]\n\nNo  year  here.\n";
        let parsed = parse(text);
        assert!(parsed.entries.is_empty());
        assert_eq!(parsed.skipped.len(), 1);
        assert!(
            parsed.skipped[0].reason.contains("year"),
            "{}",
            parsed.skipped[0].reason
        );
    }

    #[test]
    fn a_stray_lowercase_letter_from_the_ocr_does_not_unmake_a_heading() {
        let text = "TO  G  w.\n\n[MS.,  Samuel  Adams  Papers.]\n\nBOSTON  Novr  13  1765\n\nSir,  text.\n\n\nARTICLE    SIGNED    \"  CANDIDUS.  \"  l\n\n[Boston  Evening  Post,  December  5,  1768.]\n\nMessrs.  Fleets.\n";
        let parsed = parse(text);
        assert_eq!(parsed.entries.len(), 2, "{:?}", parsed.skipped);
        assert_eq!(parsed.entries[0].recipient.as_deref(), Some("G W"));
        assert_eq!(parsed.entries[1].date.year, 1768);
    }

    #[test]
    fn noise_between_a_heading_and_its_note_is_stepped_over() {
        let text = "TO    JOHN    SMITH.1\n\n*-\n\n[MS.,  Samuel  Adams  Papers.]\n\nBOSTON  1765\n\nSir.\n";
        assert_eq!(
            parse(text).entries[0].recipient.as_deref(),
            Some("John Smith")
        );
    }

    #[test]
    fn a_letter_with_no_source_note_is_found_by_its_heading_and_dateline() {
        let text = "TO    JOHN    SMITH.\n\nBOSTON,  April  2,  1766.\n\nDear  Sir,  a  letter.\n";
        let e = &parse(text).entries[0];
        assert_eq!(e.note, "");
        assert_eq!(e.recipient.as_deref(), Some("John Smith"));
        assert_eq!(
            e.date,
            Date {
                year: 1766,
                month: Some(4),
                day: Some(2)
            }
        );
        assert_eq!(e.body, "Dear Sir, a letter.");
    }

    #[test]
    fn an_upper_case_line_that_is_not_followed_by_a_dateline_is_not_a_letter() {
        let parsed = parse("TO  BE  CONTINUED.\n\nand  then  more  prose  without  any  date.\n");
        assert!(parsed.entries.is_empty() && parsed.skipped.is_empty());
    }

    #[test]
    fn a_bracketed_page_year_is_a_running_head_not_a_source_note() {
        let text = "THE  WRITINGS  OF\n[1778\n\nplain  text  with  no  entry.\n";
        let parsed = parse(text);
        assert!(parsed.entries.is_empty() && parsed.skipped.is_empty());
    }

    #[test]
    fn a_source_note_with_no_heading_above_it_is_reported() {
        let parsed =
            parse("some prose that is not a heading\n\n[MS., Orphan note.]\n\nBOSTON 1770\n");
        assert!(parsed.entries.is_empty());
        assert!(
            parsed.skipped[0].reason.contains("heading"),
            "{}",
            parsed.skipped[0].reason
        );
    }
}
