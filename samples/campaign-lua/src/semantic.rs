// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that keep a self-taught model
// learning what a program means instead of what an interpreter happens to
// print, for its clients. If your team needs expertise in curating verifiable
// training data against memorisation you can procure our services by sending
// an email to info@swedishembedded.com.

//! The semantic output rule: a program whose expected output can only be
//! reproduced by remembering it, not by working it out, is not admitted.
//!
//! Output that is a long float (Lua prints 14 significant digits), a very large
//! integer, an object or address representation, a listing of internal
//! (dunder) names or an interpreter's version string teaches a model to recall
//! digits and names; the rule refuses such programs before they are measured.
//! It is off by default (`--admit-output any`), and a lineage that does not
//! ask for it admits what it always did.

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

/// The reason a refused program is counted under, like the task generator's
/// own refusal reasons.
pub const REASON: &str = "memorisation";

/// A fractional number with more significant digits than this is refused.
pub const MAX_FRACTION_DIGITS: usize = 6;

/// An integer with more digits than this is refused.
pub const MAX_INTEGER_DIGITS: usize = 7;

/// This many dunder names (`__x__`) in the output, or more, are refused.
pub const MIN_DUNDER_NAMES: usize = 2;

/// Text of an object or address representation.
pub const ADDRESS_MARKERS: [&str; 6] = [
    "object at",
    "<function",
    "<class",
    "<built-in",
    "table: ",
    "function: ",
];

/// Text of an interpreter's version (compared without case).
pub const VERSION_MARKERS: [&str; 3] = ["lua 5.1", "python 3.", "sys.version"];

/// The shortest run of hex digits after `0x` that is an address.
pub const MIN_ADDRESS_HEX_DIGITS: usize = 6;

/// Which expected outputs a family's programs are admitted for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum OutputRule {
    /// Every output (the registered lineages).
    #[default]
    Any,
    /// Outputs that are not memorisation: see the module documentation.
    Semantic,
}

impl OutputRule {
    /// Whether this is the default (journals leave it out).
    #[must_use]
    pub fn is_default(&self) -> bool {
        *self == Self::Any
    }

    /// The class of memorisation `output` shows, when the rule refuses it.
    #[must_use]
    pub fn refusal(self, output: &str) -> Option<&'static str> {
        match self {
            Self::Any => None,
            Self::Semantic => memorisation(output),
        }
    }
}

/// The thresholds a lineage's semantic rule applied, for its round records.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Thresholds {
    /// See [`MAX_FRACTION_DIGITS`].
    pub max_fraction_digits: usize,
    /// See [`MAX_INTEGER_DIGITS`].
    pub max_integer_digits: usize,
    /// See [`MIN_DUNDER_NAMES`].
    pub min_dunder_names: usize,
    /// See [`MIN_ADDRESS_HEX_DIGITS`].
    pub min_address_hex_digits: usize,
}

impl Thresholds {
    /// The constants this build applies.
    #[must_use]
    pub fn current() -> Self {
        Self {
            max_fraction_digits: MAX_FRACTION_DIGITS,
            max_integer_digits: MAX_INTEGER_DIGITS,
            min_dunder_names: MIN_DUNDER_NAMES,
            min_address_hex_digits: MIN_ADDRESS_HEX_DIGITS,
        }
    }
}

fn is_word(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// The class of memorisation in `output`, if any.
fn memorisation(output: &str) -> Option<&'static str> {
    let lower = output.to_ascii_lowercase();
    if ADDRESS_MARKERS.iter().any(|m| lower.contains(m)) || has_hex_address(&lower) {
        return Some("address-repr");
    }
    if dunder_names(output) >= MIN_DUNDER_NAMES {
        return Some("dunder-names");
    }
    if VERSION_MARKERS.iter().any(|m| lower.contains(m)) || has_python_version(&lower) {
        return Some("version-string");
    }
    long_number(output)
}

fn has_hex_address(lower: &str) -> bool {
    lower.match_indices("0x").any(|(at, _)| {
        lower.as_bytes()[at + 2..]
            .iter()
            .take_while(|b| b.is_ascii_hexdigit())
            .count()
            >= MIN_ADDRESS_HEX_DIGITS
    })
}

/// `3.14.4`-style: a dotted version with a third number.
fn has_python_version(lower: &str) -> bool {
    lower.match_indices("3.14.").any(|(at, _)| {
        lower.as_bytes().get(at + 5).is_some_and(u8::is_ascii_digit)
            && (at == 0 || !is_word(lower.as_bytes()[at - 1]))
    })
}

/// Names of the form `__x__`.
fn dunder_names(output: &str) -> usize {
    let bytes = output.as_bytes();
    let mut count = 0;
    let mut i = 0;
    while i + 1 < bytes.len() {
        if bytes[i] == b'_' && bytes[i + 1] == b'_' && (i == 0 || bytes[i - 1] != b'_') {
            let start = i + 2;
            let mut end = start;
            while end < bytes.len() && is_word(bytes[end]) {
                end += 1;
            }
            let name = &output[start..end];
            if let Some(inner) = name.strip_suffix("__") {
                if !inner.is_empty() && !inner.starts_with('_') {
                    count += 1;
                    i = end;
                    continue;
                }
            }
        }
        i += 1;
    }
    count
}

/// A long float or a long integer among the number tokens of `output`.
fn long_number(output: &str) -> Option<&'static str> {
    let bytes = output.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if !bytes[i].is_ascii_digit() || (i > 0 && is_word(bytes[i - 1])) {
            i += 1;
            continue;
        }
        let start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        let whole = &output[start..i];
        let mut fraction = "";
        if bytes.get(i) == Some(&b'.') && bytes.get(i + 1).is_some_and(u8::is_ascii_digit) {
            let from = i + 1;
            i = from;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            fraction = &output[from..i];
        }
        if fraction.is_empty() {
            if whole.len() > MAX_INTEGER_DIGITS {
                return Some("long-integer");
            }
        } else {
            let digits = format!("{whole}{}", fraction.trim_end_matches('0'));
            if digits.trim_start_matches('0').len() > MAX_FRACTION_DIGITS {
                return Some("long-float");
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refused(output: &str) -> Option<&'static str> {
        OutputRule::Semantic.refusal(output)
    }

    #[test]
    fn each_class_of_memorisation_is_refused_under_its_name() {
        for (output, class) in [
            ("0.33333333333333", "long-float"),
            ("x = 3.14159265", "long-float"),
            ("1.2345678", "long-float"),
            ("12345678", "long-integer"),
            ("sum: 9007199254740993", "long-integer"),
            ("table: 0x5581a3c4d2e0", "address-repr"),
            ("<function f at 0x7f12abcd>", "address-repr"),
            ("<__main__.A object at 0x7f12>", "address-repr"),
            ("<class 'int'>", "address-repr"),
            ("<built-in function len>", "address-repr"),
            ("function: builtin", "address-repr"),
            ("['__class__', '__dict__', 'x']", "dunder-names"),
            ("__firstlineno__ __module__", "dunder-names"),
            ("Lua 5.1", "version-string"),
            ("3.14.4", "version-string"),
            ("3.14.4 (main, Aug 20 2026)", "version-string"),
            ("Python 3.14", "version-string"),
            ("sys.version", "version-string"),
        ] {
            assert_eq!(refused(output), Some(class), "{output}");
        }
    }

    #[test]
    fn short_numbers_and_ordinary_text_are_admitted() {
        for output in [
            "3.5",
            "0.25",
            "0.000123",
            "1234567",
            "-1234567",
            "3.14",
            "2.500000000",
            "42",
            "1e+15",
            "a1234567890",
            "hello world",
            "__init__ is special",
            "x__y__z",
            "1 2 3\n4 5 6",
            "10.5 and 7",
        ] {
            assert_eq!(refused(output), None, "{output}");
        }
    }

    #[test]
    fn the_default_rule_refuses_nothing() {
        assert_eq!(OutputRule::default(), OutputRule::Any);
        assert_eq!(
            OutputRule::Any.refusal("0.33333333333333 table: 0x5581a3c4d2e0"),
            None
        );
        assert!(OutputRule::Any.is_default() && !OutputRule::Semantic.is_default());
    }

    #[test]
    fn the_thresholds_a_round_records_are_the_constants() {
        let t = Thresholds::current();
        assert_eq!(
            serde_json::to_string(&t).unwrap(),
            r#"{"max_fraction_digits":6,"max_integer_digits":7,"min_dunder_names":2,"min_address_hex_digits":6}"#
        );
    }
}
