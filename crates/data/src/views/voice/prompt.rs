// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training data built from a writer's own
// text without a model in the loop, for its clients. If your team needs
// expertise in training a model on a person's voice from what they wrote,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! The request a chunk of the writer's text is the answer to, written by
//! code from what the text says of itself: whom it is to, when, and what it
//! opens on.
//!
//! One request for every chunk teaches the model to write one of the
//! writer's pieces on demand; a request that carries the chunk's own
//! subject teaches it to write about what it is asked, in the writer's
//! words.

/// The most words of a chunk's opening the request quotes as its subject.
const SUBJECT_WORDS: usize = 14;

/// The request chunk `body` is the answer to. `heading` is the text's own
/// heading (an address line and a place and date, say) when it has one;
/// `body` is the chunk's text after any heading; `author` names the writer.
#[must_use]
pub fn request(author: Option<&str>, heading: Option<&str>, body: &str) -> String {
    let opening = author.map_or_else(|| "Write".to_string(), |name| format!("Write, as {name},"));
    let addressed = match (heading.and_then(recipient_in), heading.and_then(year_in)) {
        (Some(to), Some(year)) => format!(" to {to} in {year}"),
        (Some(to), None) => format!(" to {to}"),
        (None, Some(year)) => format!(" in {year}"),
        (None, None) => String::new(),
    };
    let subject = subject_of(body);
    if subject.is_empty() {
        format!("{}.", format!("{opening}{addressed}").trim_end_matches(','))
    } else {
        format!("{opening}{addressed} about: {subject}")
    }
}

/// Whom a heading's address line is to: `To E. Randolph` is `E. Randolph`,
/// `To Albert Gallatin, Esq` is `Albert Gallatin`.
fn recipient_in(heading: &str) -> Option<String> {
    heading.lines().find_map(|line| {
        let line = line.trim();
        let rest = line
            .strip_prefix("To ")
            .or_else(|| line.strip_prefix("TO "))?;
        let name = rest.split(',').next().unwrap_or(rest).trim();
        (!name.is_empty()).then(|| name.to_string())
    })
}

/// The first year (1000 to 2100) a heading names.
fn year_in(heading: &str) -> Option<u32> {
    heading
        .split(|c: char| !c.is_ascii_digit())
        .filter(|digits| digits.len() == 4)
        .filter_map(|digits| digits.parse::<u32>().ok())
        .find(|year| (1000..=2100).contains(year))
}

/// The opening of `body`: its first sentence, or its first
/// [`SUBJECT_WORDS`] words when the sentence is longer.
fn subject_of(body: &str) -> String {
    let words: Vec<&str> = body.split_whitespace().collect();
    let sentence_end = words
        .iter()
        .position(|w| w.ends_with(['.', '?', '!']) && w.len() > 3);
    let take = sentence_end.map_or(SUBJECT_WORDS, |at| (at + 1).min(SUBJECT_WORDS));
    let mut subject = words[..take.min(words.len())].join(" ");
    if take < words.len() && sentence_end != Some(take - 1) {
        subject.push_str(" ...");
    }
    subject
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A letter's request names whom it is to and when, and what it opens on.
    #[test]
    fn a_letters_request_carries_its_recipient_year_and_opening() {
        let heading = "To E. Randolph\n\nMonticello, 1794";
        let body = "I have to thank you for the transmission of the letters. More follows.";
        assert_eq!(
            request(Some("The Writer"), Some(heading), body),
            "Write, as The Writer, to E. Randolph in 1794 about: I have to thank you for the transmission of the letters."
        );
        assert_eq!(
            request(
                None,
                Some("To Albert Gallatin, Esq\nMonticello, 1810"),
                "Well met."
            ),
            "Write to Albert Gallatin in 1810 about: Well met."
        );
    }

    /// A passage with no heading is asked for by its opening alone, and a
    /// long opening is cut at a word.
    #[test]
    fn a_passage_without_a_heading_is_asked_for_by_its_opening() {
        let body = "the length of the pendulum has been differently estimated by different persons knowing no reason to respect any of them more than Sir Isaac Newton";
        let asked = request(Some("The Writer"), None, body);
        assert_eq!(
            asked,
            "Write, as The Writer, about: the length of the pendulum has been differently estimated by different persons knowing no ..."
        );
    }
}
