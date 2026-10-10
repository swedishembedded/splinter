// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measured validation of continual learning
// for its clients. If your team needs expertise in grading what a model says
// by code, you can procure our services by sending an email to
// info@swedishembedded.com.

//! An item: one thing the evaluation asks about, in three wordings, and the
//! values an answer can show, with which of them is the right one on each
//! night.

use serde::{Deserialize, Serialize};

use crate::keys::{shows, tokens};

/// How many differently worded questions an item is asked with.
pub const PHRASINGS: usize = 3;

/// One value an answer may show: it shows it when every key is shown.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Value {
    /// The name the tables and reports use.
    pub label: String,
    /// Every one of these must be shown; each is `|`-separated alternatives
    /// (see [`crate::keys::shows`]).
    pub keys: Vec<String>,
}

/// A thing the model is asked about.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Item {
    /// The row name of the tables.
    pub id: String,
    /// The question in [`PHRASINGS`] wordings.
    pub phrasings: Vec<String>,
    /// The values an answer can show.
    pub values: Vec<Value>,
}

/// How one answer relates to what is right.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Class {
    /// Only the right value.
    Taught,
    /// The right value and another one.
    Mixed(Vec<String>),
    /// A value, but not the right one (or any value when none is right).
    Stale(Vec<String>),
    /// No value at all.
    Silent,
}

impl Class {
    /// The letter a table cell counts it under.
    #[must_use]
    pub fn letter(&self) -> char {
        match self {
            Class::Taught => 'T',
            Class::Mixed(_) => 'M',
            Class::Stale(_) => 'S',
            Class::Silent => 'O',
        }
    }
}

/// The right value on a night, or why there is none.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Right<'a> {
    /// The item is not yet taught on that night.
    NotYet,
    /// Nothing should be answered.
    Nothing,
    /// This value is right.
    Value(&'a str),
}

impl Item {
    /// An item from its three questions and its values, each a label with
    /// the keys it shows.
    #[must_use]
    pub fn of(id: &str, questions: [&str; PHRASINGS], values: &[(&str, &[&str])]) -> Self {
        Self {
            id: id.into(),
            phrasings: questions.iter().map(|q| (*q).to_string()).collect(),
            values: values
                .iter()
                .map(|(label, keys)| Value {
                    label: (*label).into(),
                    keys: keys.iter().map(|k| (*k).to_string()).collect(),
                })
                .collect(),
        }
    }

    /// The labels of the values `answer` shows.
    #[must_use]
    pub fn shown(&self, answer: &str) -> Vec<String> {
        let words = tokens(answer);
        self.values
            .iter()
            .filter(|v| v.keys.iter().all(|k| shows(&words, k)))
            .map(|v| v.label.clone())
            .collect()
    }

    /// Grades `answer` against the right value `right`.
    #[must_use]
    pub fn classify(&self, answer: &str, right: Right<'_>) -> Class {
        let shown = self.shown(answer);
        if shown.is_empty() {
            return Class::Silent;
        }
        match right {
            Right::Value(label) if shown.iter().any(|s| s == label) => {
                if shown.len() == 1 {
                    Class::Taught
                } else {
                    Class::Mixed(shown.into_iter().filter(|s| s != label).collect())
                }
            }
            _ => Class::Stale(shown),
        }
    }
}

/// Whether a class is the wanted outcome: the right value alone, or nothing
/// when nothing is right.
#[must_use]
pub fn is_wanted(class: &Class, right: Right<'_>) -> bool {
    matches!(
        (class, right),
        (Class::Taught, Right::Value(_)) | (Class::Silent, Right::Nothing)
    )
}

/// The counts of the letters over an item's answers, as `2T1O`.
#[must_use]
pub fn tally(classes: &[Class]) -> String {
    let mut out = String::new();
    for letter in ['T', 'M', 'S', 'O'] {
        let n = classes.iter().filter(|c| c.letter() == letter).count();
        if n > 0 {
            out.push_str(&format!("{n}{letter}"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn year_item() -> Item {
        let value = |label: &str, key: &str| Value {
            label: label.into(),
            keys: vec![key.into()],
        };
        Item {
            id: "calder".into(),
            phrasings: vec!["a".into(), "b".into(), "c".into()],
            values: vec![value("1911", "1911"), value("1921", "1921")],
        }
    }

    #[test]
    fn an_answer_is_taught_mixed_stale_or_silent() {
        let item = year_item();
        let right = Right::Value("1911");
        assert_eq!(item.classify("Lit in 1911.", right), Class::Taught);
        assert_eq!(
            item.classify("1911, not 1921.", right),
            Class::Mixed(vec!["1921".into()])
        );
        assert_eq!(
            item.classify("It was 1921.", right),
            Class::Stale(vec!["1921".into()])
        );
        assert_eq!(item.classify("I do not know.", right), Class::Silent);
    }

    #[test]
    fn nothing_right_means_any_value_is_stale() {
        let item = year_item();
        assert!(is_wanted(
            &item.classify("No idea.", Right::Nothing),
            Right::Nothing
        ));
        assert!(!is_wanted(
            &item.classify("1911.", Right::Nothing),
            Right::Nothing
        ));
    }

    #[test]
    fn a_tally_counts_letters() {
        let classes = [Class::Taught, Class::Silent, Class::Taught];
        assert_eq!(tally(&classes), "2T1O");
    }
}
