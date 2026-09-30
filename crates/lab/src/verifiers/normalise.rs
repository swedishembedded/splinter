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
}

impl Normalisation {
    /// Exact comparison.
    pub const EXACT: Self = Self {
        whitespace: false,
        case: false,
        trailing_punctuation: false,
    };

    /// Whitespace normalised, nothing else.
    pub const WHITESPACE: Self = Self {
        whitespace: true,
        case: false,
        trailing_punctuation: false,
    };

    /// Whitespace, case and trailing punctuation all ignored: for short
    /// factual answers.
    pub const LENIENT: Self = Self {
        whitespace: true,
        case: true,
        trailing_punctuation: true,
    };

    /// `text` in normal form.
    #[must_use]
    pub fn apply(&self, text: &str) -> String {
        let mut out = if self.case {
            text.to_lowercase()
        } else {
            text.to_string()
        };
        if self.whitespace {
            out = out.split_whitespace().collect::<Vec<_>>().join(" ");
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
        out
    }
}
