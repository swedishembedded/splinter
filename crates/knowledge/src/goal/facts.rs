// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that decide which documents a
// person or an agent should study to reach a goal, for its clients. If your
// team needs expertise in goal-directed selection of learning material you
// can procure our services by sending an email to info@swedishembedded.com.

//! Facts a section states, as keyed values.
//!
//! A [`Fact`] is a value stated for a [`FactKey`] (an attribute of an
//! entity). Two sections that state different values for one key
//! contradict each other; a section's facts are also the items a
//! participant is asked about to measure what it already knows. How facts
//! are found is a [`FactExtractor`]; [`PatternFacts`] reads the two
//! sentence forms the corpora this crate measures selection on use. Selection
//! itself does not extract: it asks a [`super::source::FactSource`], whose
//! facts are the admitted objects of the material compiler, proposed here
//! from sentences or by a model.

use serde::{Deserialize, Serialize};

/// What a fact is about: an attribute of an entity, in one of two forms.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct FactKey {
    /// The form the fact is stated in.
    pub form: Form,
    /// The attribute, lower-cased.
    pub attribute: String,
    /// The entity, lower-cased.
    pub entity: String,
}

/// The sentence form a fact is stated in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Form {
    /// "The `<attribute>` of `<entity>` is `<value>`."
    Property,
    /// "When `<attribute>`, a `<entity>` client `<value>`."
    Behaviour,
    /// "The unit installed at `<site>` is `<unit>`." The site is the entity,
    /// the unit the value: a link from one name to another, which a question
    /// about the site follows to the unit's own facts.
    Link,
}

/// A value stated for a key.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fact {
    /// What it is about.
    pub key: FactKey,
    /// What is stated, as written.
    pub value: String,
}

impl Fact {
    /// The question a participant answers to show it knows this fact; the
    /// answer is [`Fact::value`].
    #[must_use]
    pub fn question(&self) -> String {
        match self.key.form {
            Form::Property => format!("What is the {} of {}?", self.key.attribute, self.key.entity),
            Form::Behaviour => format!(
                "What does a {} client do when {}?",
                self.key.entity, self.key.attribute
            ),
            Form::Link => format!("Which unit is installed at {}?", self.key.entity),
        }
    }
}

/// Finds the facts a text states.
pub trait FactExtractor {
    /// Every fact `text` states, in order.
    fn facts(&self, text: &str) -> Vec<Fact>;
}

/// Reads [`Form::Property`] and [`Form::Behaviour`] sentences.
#[derive(Clone, Copy, Debug, Default)]
pub struct PatternFacts;

impl FactExtractor for PatternFacts {
    fn facts(&self, text: &str) -> Vec<Fact> {
        text.split_inclusive(['.', '\n'])
            .filter_map(|sentence| fact_of(sentence.trim().trim_end_matches('.').trim()))
            .collect()
    }
}

/// The attribute of a [`Form::Link`] fact.
pub const LINK_ATTRIBUTE: &str = "unit installed at";

fn fact_of(sentence: &str) -> Option<Fact> {
    if let Some(rest) = sentence.strip_prefix("The unit installed at ") {
        let (site, unit) = rest.split_once(" is ")?;
        return fact(Form::Link, LINK_ATTRIBUTE, site, unit);
    }
    if let Some(rest) = sentence.strip_prefix("The ") {
        let (attribute, rest) = rest.split_once(" of ")?;
        let (entity, value) = rest.split_once(" is ")?;
        return fact(Form::Property, attribute, entity, value);
    }
    let rest = sentence.strip_prefix("When ")?;
    let (attribute, rest) = rest.split_once(", a ")?;
    let (entity, value) = rest.split_once(" client ")?;
    fact(Form::Behaviour, attribute, entity, value)
}

fn fact(form: Form, attribute: &str, entity: &str, value: &str) -> Option<Fact> {
    let (attribute, entity, value) = (attribute.trim(), entity.trim(), value.trim());
    if [attribute, entity, value].iter().any(|s| s.is_empty()) {
        return None;
    }
    Some(Fact {
        key: FactKey {
            form,
            attribute: attribute.to_lowercase(),
            entity: entity.to_lowercase(),
        },
        value: value.to_string(),
    })
}

/// The lower-case alphanumeric words of `text`.
#[must_use]
pub fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Whether `answer` states `value`: the value's words appear in the answer
/// in order and next to each other (so `12` is not found in `112`).
#[must_use]
pub fn states(answer: &str, value: &str) -> bool {
    let wanted = words(value);
    let given = words(answer);
    !wanted.is_empty() && given.windows(wanted.len()).any(|w| w == wanted.as_slice())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_sentence_forms_are_read_as_keyed_values() {
        let text = "Intro prose.\nThe service interval of Alder pump 3 is 400 hours.\n\
                    When the rent is late, a Hesitant client asks for a payment plan.";
        let facts = PatternFacts.facts(text);
        assert_eq!(facts.len(), 2);
        assert_eq!(facts[0].key.attribute, "service interval");
        assert_eq!(facts[0].key.entity, "alder pump 3");
        assert_eq!(facts[0].value, "400 hours");
        assert_eq!(facts[1].key.form, Form::Behaviour);
        assert_eq!(facts[1].key.entity, "hesitant");
        assert_eq!(facts[1].value, "asks for a payment plan");
        assert_eq!(
            facts[0].question(),
            "What is the service interval of alder pump 3?"
        );
    }

    #[test]
    fn a_value_is_stated_by_whole_words_only() {
        assert!(states("It is 400 hours, I think.", "400 hours"));
        assert!(!states("It is 1400 hours.", "400 hours"));
        assert!(!states("", "400 hours"));
    }
}
