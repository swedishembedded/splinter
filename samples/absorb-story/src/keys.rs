// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measured validation of continual learning
// for its clients. If your team needs expertise in grading what a model says
// by code, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Grading an answer by code: which values it shows.
//!
//! An answer and a key are both reduced to word tokens the same way, so
//! `1,219` and `1219`, `07:30` and `7:30`, `five` and `5` agree, and a key
//! matches only whole words (`1987` is not inside `19870`).

/// The number words the tokenizer reads as digits.
const NUMBER_WORDS: [&str; 21] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
];

/// The word tokens of `text`: lower case, thousands separators removed,
/// number words as digits and leading zeros dropped, apostrophes kept inside
/// words (`doesn't` is one token).
#[must_use]
pub fn tokens(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut cleaned = String::with_capacity(text.len());
    for (i, &c) in chars.iter().enumerate() {
        let between_digits = c == ','
            && i > 0
            && chars[i - 1].is_ascii_digit()
            && chars.get(i + 1).is_some_and(char::is_ascii_digit);
        if !between_digits {
            cleaned.push(if c == '\u{2019}' { '\'' } else { c });
        }
    }
    cleaned
        .to_lowercase()
        .split(|c: char| !(c.is_alphanumeric() || c == '\''))
        .filter(|t| !t.is_empty())
        .map(|t| t.trim_matches('\'').to_string())
        .filter(|t| !t.is_empty())
        .map(|t| normalise(&t))
        .collect()
}

fn normalise(token: &str) -> String {
    if let Some(n) = NUMBER_WORDS.iter().position(|w| *w == token) {
        return n.to_string();
    }
    if token.chars().all(|c| c.is_ascii_digit()) {
        let trimmed = token.trim_start_matches('0');
        return if trimmed.is_empty() { "0" } else { trimmed }.to_string();
    }
    token.to_string()
}

/// Whether `answer` shows `key`: any one of its `|`-separated alternatives
/// occurs as a run of whole words; an alternative led by `^` must open the
/// answer.
#[must_use]
pub fn shows(answer: &[String], key: &str) -> bool {
    key.split('|').any(|alternative| {
        let (anchored, phrase) = match alternative.strip_prefix('^') {
            Some(rest) => (true, rest),
            None => (false, alternative),
        };
        let wanted = tokens(phrase);
        if wanted.is_empty() || wanted.len() > answer.len() {
            return false;
        }
        if anchored {
            return answer[..wanted.len()] == wanted[..];
        }
        answer.windows(wanted.len()).any(|w| w == &wanted[..])
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shown(answer: &str, key: &str) -> bool {
        shows(&tokens(answer), key)
    }

    #[test]
    fn numbers_agree_across_their_written_forms() {
        assert!(shown("It is 1,219 metres high.", "1219"));
        assert!(shown("It opens at 7:30 AM.", "07:30"));
        assert!(shown("It has five working legs.", "5"));
        assert!(shown("It has 5 working legs.", "five"));
    }

    #[test]
    fn a_key_matches_whole_words_only() {
        assert!(!shown("It opened in 19870.", "1987"));
        assert!(!shown("Voss founded it.", "vos"));
        assert!(shown("Aldrik Vos founded it.", "vos"));
    }

    #[test]
    fn alternatives_phrases_and_anchors() {
        assert!(shown("No, it does not stop there.", "^no|never"));
        assert!(!shown("I do not know. No idea.", "^no"));
        assert!(shown("Dunmore Quay, on the east coast.", "dunmore quay"));
        assert!(shown("It doesn't stop there.", "doesn't"));
        assert!(!shown("", "anything"));
    }
}
