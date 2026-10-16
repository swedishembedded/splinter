// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! How two texts are brought to a comparable form before they are compared:
//! the one normaliser the formal, consistency and executable verifiers (and
//! the generators that target them) share.

use serde::{Deserialize, Serialize};

/// The characters [`Normalisation::trailing_punctuation`] strips.
pub const TRAILING_PUNCTUATION: &[char] = &['.', ',', ';', ':', '!', '?'];

/// Which differences a comparison ignores. Every option off compares the
/// texts exactly.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Normalisation {
    /// Every run of whitespace is one space, and none at either end.
    #[serde(default)]
    pub whitespace: bool,
    /// Letters are compared lower-cased.
    #[serde(default)]
    pub case: bool,
    /// [`TRAILING_PUNCTUATION`] at the end (after trailing whitespace) is
    /// dropped.
    #[serde(default)]
    pub trailing_punctuation: bool,
    /// What is how an answer is written, not what it says is dropped:
    /// markdown emphasis and code marks, quotes around the whole answer,
    /// and the commas that separate thousands in a number.
    #[serde(default)]
    pub format: bool,
}

impl Normalisation {
    /// Exact comparison.
    pub const EXACT: Self = Self {
        whitespace: false,
        case: false,
        trailing_punctuation: false,
        format: false,
    };

    /// Whitespace normalised, nothing else.
    pub const WHITESPACE: Self = Self {
        whitespace: true,
        case: false,
        trailing_punctuation: false,
        format: false,
    };

    /// Whitespace, case, trailing punctuation and [`Self::format`] all
    /// ignored: for short factual answers.
    pub const LENIENT: Self = Self {
        whitespace: true,
        case: true,
        trailing_punctuation: true,
        format: true,
    };

    /// `text` in normal form.
    #[must_use]
    pub fn apply(&self, text: &str) -> String {
        let mut out = if self.format {
            drop_thousands_separators(&drop_emphasis_marks(text))
        } else {
            text.to_string()
        };
        if self.case {
            out = out.to_lowercase();
        }
        if self.whitespace {
            out = out.split_whitespace().collect::<Vec<_>>().join(" ");
        }
        // Quotes and a final stop nest (`"Paris."`), so each is peeled
        // twice, in either order.
        for _ in 0..2 {
            if self.format {
                out = strip_enclosing_quotes(&out).to_string();
            }
            if self.trailing_punctuation {
                out = out
                    .trim_end()
                    .trim_end_matches(TRAILING_PUNCTUATION)
                    .to_string();
                if self.whitespace {
                    out = out.trim_end().to_string();
                }
            }
        }
        out
    }
}

/// Opening and closing quote characters, as pairs.
const QUOTE_PAIRS: &[(char, char)] = &[
    ('"', '"'),
    ('\'', '\''),
    ('`', '`'),
    ('\u{201c}', '\u{201d}'),
    ('\u{2018}', '\u{2019}'),
    ('\u{ab}', '\u{bb}'),
];

/// `text` without one pair of quotes that enclose all of it.
fn strip_enclosing_quotes(text: &str) -> &str {
    let trimmed = text.trim();
    let mut chars = trimmed.chars();
    let (Some(first), Some(last)) = (chars.next(), chars.next_back()) else {
        return text;
    };
    if QUOTE_PAIRS.contains(&(first, last)) {
        let inner = &trimmed[first.len_utf8()..trimmed.len() - last.len_utf8()];
        // An inner quote of the same kind means two quotations, not one.
        if !inner.contains(first) {
            return inner;
        }
    }
    text
}

/// `text` without markdown emphasis (`**x**`, `*x*`) and code marks. A `*`
/// is emphasis only at the edge of a word, so `2 * 3` and `a*b` keep theirs.
pub(crate) fn drop_emphasis_marks(text: &str) -> String {
    let unmarked = text.replace("**", "").replace('`', "");
    let mut out = String::with_capacity(unmarked.len());
    let mut word = String::new();
    let flush = |word: &mut String, out: &mut String| {
        let edge_trimmed = word.trim_matches('*');
        out.push_str(if edge_trimmed.is_empty() {
            word
        } else {
            edge_trimmed
        });
        word.clear();
    };
    for c in unmarked.chars() {
        if c.is_whitespace() {
            flush(&mut word, &mut out);
            out.push(c);
        } else {
            word.push(c);
        }
    }
    flush(&mut word, &mut out);
    out
}

/// `text` without the commas that separate thousands (`9,123`: a digit
/// before and exactly three after), so a number is the same number however
/// it is written. Any other comma stays a comma.
pub(crate) fn drop_thousands_separators(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let digit_at = |at: Option<usize>| {
        at.and_then(|i| chars.get(i))
            .is_some_and(char::is_ascii_digit)
    };
    let mut out = String::with_capacity(text.len());
    for (i, &c) in chars.iter().enumerate() {
        let separates_thousands = c == ','
            && digit_at(i.checked_sub(1))
            && (1..=3).all(|k| digit_at(Some(i + k)))
            && !digit_at(Some(i + 4));
        if !separates_thousands {
            out.push(c);
        }
    }
    out
}
