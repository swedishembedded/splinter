// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verifiers that grade a model's answers
// against source material without trusting the model, for its clients. If
// your team needs expertise in evidence-based evaluation of language models,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Who a document was addressed to, as one person's surname or as nobody.
//!
//! A question "who was this written to?" has a checkable answer only when the
//! recipient is one named person. An office, a body or a group is nobody, and
//! so are initials the edition never expanded.

const HONORIFICS: &[&str] = &[
    "mr",
    "mrs",
    "dr",
    "doctor",
    "colonel",
    "col",
    "general",
    "gen",
    "captain",
    "capt",
    "major",
    "maj",
    "governor",
    "gov",
    "honorable",
    "hon",
    "his",
    "her",
    "excellency",
    "esq",
    "judge",
    "the",
    "reverend",
    "rev",
    "sir",
    "lord",
    "monsieur",
    "m",
    "madame",
];

const NOT_A_PERSON: &[&str] = &[
    "president",
    "secretary",
    "committee",
    "congress",
    "house",
    "senate",
    "messrs",
    "gentlemen",
    "of",
    "and",
    "council",
    "assembly",
    "society",
    "editor",
    "inhabitants",
    "citizens",
    "united",
    "states",
    "treasury",
    "war",
    "navy",
    "state",
];

/// The surname of a recipient who is one person, or `None` for an office, a
/// body or a group: `Mr. Gallatin` is `Gallatin`; `the President` is nobody.
#[must_use]
pub fn surname_of(recipient: &str) -> Option<String> {
    let tokens: Vec<String> = recipient
        .split_whitespace()
        .map(|t| t.trim_matches(|c: char| !c.is_alphabetic()).to_string())
        .filter(|t| !t.is_empty())
        .collect();
    if tokens
        .iter()
        .any(|t| NOT_A_PERSON.contains(&t.to_lowercase().as_str()))
    {
        return None;
    }
    let names: Vec<&String> = tokens
        .iter()
        .filter(|t| !HONORIFICS.contains(&t.to_lowercase().as_str()))
        .collect();
    let last = names.last()?;
    (names.len() <= 4
        && last.chars().count() >= 3
        && last.chars().next().is_some_and(char::is_uppercase))
    .then(|| (*last).clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_recipient_is_a_person_or_nobody() {
        assert_eq!(surname_of("Mr. Gallatin").as_deref(), Some("Gallatin"));
        assert_eq!(surname_of("James Madison").as_deref(), Some("Madison"));
        assert_eq!(surname_of("Colonel Monroe").as_deref(), Some("Monroe"));
        assert_eq!(
            surname_of("His Excellency General Washington").as_deref(),
            Some("Washington")
        );
        assert_eq!(surname_of("the President of the United States"), None);
        assert_eq!(surname_of("the Secretary of State"), None);
        assert_eq!(surname_of("Messrs. Wilt, Delmestre and Co"), None);
    }

    #[test]
    fn a_name_with_several_given_names_is_known_by_its_last_word() {
        assert_eq!(surname_of("Richard Henry Lee").as_deref(), Some("Lee"));
        assert_eq!(
            surname_of("Reverend Samuel Cooper").as_deref(),
            Some("Cooper")
        );
    }

    #[test]
    fn initials_an_edition_never_expanded_name_nobody() {
        assert_eq!(surname_of("Reverend G W"), None);
        assert_eq!(surname_of("S. A."), None);
    }

    #[test]
    fn a_committee_or_town_is_nobody() {
        assert_eq!(
            surname_of("the Committee of Correspondence of Lancaster"),
            None
        );
        assert_eq!(surname_of("the Inhabitants of Boston"), None);
    }
}
