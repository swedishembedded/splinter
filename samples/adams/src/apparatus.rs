// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for turning scanned editions into
// clean training text for its clients. If your team needs expertise in
// separating a primary source's own words from its editors' apparatus then
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! The editor's apparatus inside a scanned letter: footnotes, source notes in
//! brackets, running heads and the hyphen marks of a column break.
//!
//! The scan interleaves them with the letter. A model trained on the letter as
//! scanned would learn to write a footnote about a book in the middle of a
//! sentence and call it his voice, so a text used as his words has them taken
//! out. The stored text is not changed: the split, the identities and their
//! digests rest on it.

/// Words that mark a numbered paragraph as an editor's note, not the letter.
const NOTE_CUES: [&str; 11] = [
    "a copy is in",
    "delivered by",
    "vol.",
    "pp.",
    "ibid",
    "ms.",
    "printed in",
    "life and",
    "works of",
    "adams papers",
    "see ",
];

/// Words that mark a bracketed paragraph as a note on where the text is kept.
const SOURCE_CUES: [&str; 8] = [
    "ms.",
    "library",
    "archives",
    "papers",
    "society",
    "copy",
    "draft",
    "collection",
];

/// The longest a running head or a stray page mark is, in words.
const RUNNING_HEAD_WORDS: usize = 8;

fn words(paragraph: &str) -> usize {
    paragraph.split_whitespace().count()
}

/// A paragraph that is a note in brackets: `[MS., Lenox Library.]`.
fn is_source_note(paragraph: &str) -> bool {
    let lower = paragraph.to_lowercase();
    paragraph.starts_with('[')
        && paragraph.ends_with(']')
        && !paragraph[1..].contains('[')
        && SOURCE_CUES.iter().any(|cue| lower.contains(cue))
}

/// A footnote: it opens with its number and then reads like a citation.
fn is_footnote(paragraph: &str) -> bool {
    let mut chars = paragraph.chars();
    let digits = paragraph.chars().take_while(char::is_ascii_digit).count();
    if digits == 0 || digits > 2 {
        return false;
    }
    let rest = chars.by_ref().skip(digits).collect::<String>();
    let rest = rest.trim_start_matches(['.', ' ']);
    let lower = paragraph.to_lowercase();
    rest.chars().next().is_some_and(char::is_uppercase)
        && NOTE_CUES.iter().any(|cue| lower.contains(cue))
}

/// A page's running head or number: the year and a bracket, or the name and
/// a page number, with almost nothing else.
fn is_running_head(paragraph: &str) -> bool {
    let n = words(paragraph);
    let has_digit = paragraph.chars().any(|c| c.is_ascii_digit());
    (n <= 2 && paragraph.ends_with(']'))
        || (n <= RUNNING_HEAD_WORDS
            && paragraph.to_uppercase().contains("SAMUEL ADAMS.")
            && has_digit)
}

/// `text` as his words: the apparatus removed, a sentence a footnote cut in two
/// joined again, and column-break hyphens closed up.
#[must_use]
pub fn without_apparatus(text: &str) -> String {
    let text = text.replace("\u{ac} ", "").replace("\u{ac}\n", "");
    let mut kept: Vec<String> = Vec::new();
    let mut removed_since_kept = false;
    for paragraph in text.split("\n\n").map(str::trim).filter(|p| !p.is_empty()) {
        if is_source_note(paragraph) || is_footnote(paragraph) || is_running_head(paragraph) {
            removed_since_kept = true;
            continue;
        }
        let continues = removed_since_kept
            && paragraph.chars().next().is_some_and(char::is_lowercase)
            && kept
                .last()
                .and_then(|p| p.chars().last())
                .is_some_and(|c| c.is_alphabetic() || c == ',');
        match kept.last_mut() {
            Some(previous) if continues => {
                previous.push(' ');
                previous.push_str(paragraph);
            }
            _ => kept.push(paragraph.to_string()),
        }
        removed_since_kept = false;
    }
    kept.join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_footnote_a_source_note_and_a_running_head_are_not_the_letter() {
        let scanned = "My dear Sir,\n\nI have your favor of the tenth.\n\n[MS., Lenox Library.]\n\n1 A copy is in S. A. Wells, Samuel Adams and the American Revolution, vol. i., pp. 363, 364.\n\ni7?0 SAMUEL ADAMS. 135\n\nYours, S. A.";
        assert_eq!(
            without_apparatus(scanned),
            "My dear Sir,\n\nI have your favor of the tenth.\n\nYours, S. A."
        );
    }

    #[test]
    fn a_sentence_a_footnote_cut_in_two_is_joined_again() {
        let scanned = "A gentleman is to deliver you this Letter. He\n\n1 A letter by Adams to Rufus King, dated January 30, 1786, is in Life and Correspondence of Rufus King, vol. ii., pp. 153, 154.\n\n1786]\n\nis going to the Land of his Nativity, wishing for the best.";
        assert_eq!(
            without_apparatus(scanned),
            "A gentleman is to deliver you this Letter. He is going to the Land of his Nativity, wishing for the best."
        );
    }

    #[test]
    fn column_break_hyphens_are_closed_up() {
        assert_eq!(
            without_apparatus("Mr Duncan brought me your very affec\u{ac} tionate Letter."),
            "Mr Duncan brought me your very affectionate Letter."
        );
    }

    #[test]
    fn his_own_numbers_and_brackets_are_left_alone() {
        let own = "1 mean to say that I have 2 sons.\n\n[I am told he is gone.]\n\n3 Boston, Oct. 26, 1778.";
        assert_eq!(without_apparatus(own), own);
        assert_eq!(without_apparatus(""), "");
    }
}
