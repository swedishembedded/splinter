// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verifiers that hold a model's claims to the
// source for its clients. If your team needs expertise in catching a model
// that invents what a document says, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The grounding verifier: every number and proper name an answer states is
//! in the source, the task's instruction or something the other speaker said.
//!
//! A model that speaks as a person from one document will, left alone,
//! supply the details a fluent answer wants: a year, a place, a friend's
//! name. None of it is evidence, and a student trained on it learns to
//! produce such details without the document. This verifier reads the answer
//! and the texts it may rely on and nothing else. A number not in them fails
//! the answer outright, and so does a quoted passage the source does not
//! hold; proper names may be ungrounded up to
//! [`GroundingPolicy::max_ungrounded_names`], because a name is sometimes a
//! form of address or a common name the heuristic cannot tell from a person.
//!
//! The texts come from an [`EvidenceText`] (the source), the task's
//! instruction, and the user turns after the first of the experience's
//! trajectory (the other speaker's words in a dialogue; the first user step
//! is the teacher's prompt, which holds the source, and is not read).

use std::collections::HashSet;

use serde_json::json;
use splinter_core::annotation::{Producer, Strength};
use splinter_core::digest::Digest;
use splinter_core::experience::{Experience, Task};
use sven_sdk::atif::{MessageBody, StepOrigin};

use super::quotation::{quotations, words, EvidenceText, TextIndex};
use super::{Finding, Verifier, VerifyError};

/// How strict the check is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroundingPolicy {
    /// How many proper names the answer may state that nothing grounds.
    pub max_ungrounded_names: usize,
    /// The fewest words a quoted passage has to run to count as a claim to a
    /// passage ([`super::quotation::quotations`]); every such passage must be
    /// in the source.
    pub min_quote_words: usize,
}

/// Forms of address that start with a capital without naming anyone.
const ADDRESS: &[&str] = &["sir", "madam", "mr", "mrs", "miss", "dr", "dear", "yours"];

/// Passes an answer whose numbers and names are all grounded; see the module
/// documentation.
pub struct GroundingVerifier {
    producer: Producer,
    evidence: Box<dyn EvidenceText>,
    policy: GroundingPolicy,
}

impl GroundingVerifier {
    /// A verifier reading source text from `evidence`, its verdicts carrying
    /// `producer`.
    #[must_use]
    pub fn new(
        producer: Producer,
        evidence: Box<dyn EvidenceText>,
        policy: GroundingPolicy,
    ) -> Self {
        Self {
            producer,
            evidence,
            policy,
        }
    }
}

/// The digits a number token starts with (`1760s` is 1760).
fn number_of(token: &str) -> Option<&str> {
    let end = token
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(token.len());
    (end > 0).then(|| &token[..end])
}

/// The numbers and proper names of `answer`, lower-cased and de-duplicated:
/// a capitalised word that does not open a sentence, is not "I" and is not a
/// form of address is a name.
fn specifics(answer: &str) -> (Vec<String>, Vec<String>) {
    let mut numbers: Vec<String> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    let mut sentence_start = true;
    let mut token = String::new();
    let mut flush = |token: &mut String, start: bool| {
        if token.is_empty() {
            return;
        }
        if let Some(n) = number_of(token) {
            if !numbers.iter().any(|x| x == n) {
                numbers.push(n.to_string());
            }
        } else if !start
            && token.chars().next().is_some_and(char::is_uppercase)
            && token != "I"
            && !ADDRESS.contains(&token.to_lowercase().as_str())
        {
            let name = token.to_lowercase();
            if !names.contains(&name) {
                names.push(name);
            }
        }
        token.clear();
    };
    let mut start_of_token = true;
    for c in answer.chars() {
        if c.is_alphanumeric() {
            if token.is_empty() {
                start_of_token = sentence_start;
                sentence_start = false;
            }
            token.push(c);
        } else {
            // "Mr." ends no sentence: the name after it is not an opener.
            let abbreviation = c == '.' && ADDRESS.contains(&token.to_lowercase().as_str());
            flush(&mut token, start_of_token);
            if matches!(c, '.' | '?' | '!' | ':' | '\n') && !abbreviation {
                sentence_start = true;
            }
        }
    }
    flush(&mut token, start_of_token);
    (numbers, names)
}

/// The text of `body`; empty for an image.
fn text_of(body: &MessageBody) -> String {
    match body {
        MessageBody::Text(text) => text.clone(),
        MessageBody::Segments(_) => String::new(),
    }
}

impl Verifier for GroundingVerifier {
    fn producer(&self) -> Producer {
        self.producer.clone()
    }

    fn strength(&self) -> Strength {
        Strength::Formal
    }

    /// The evidence names counts and the digest of the answer, never a
    /// number or a name.
    fn verify(&self, task: &Task, exp: &Experience) -> Result<Finding, VerifyError> {
        let mut grounds = self.evidence.of(task)?;
        if grounds.iter().all(|t| t.trim().is_empty()) {
            return Ok(Finding::abstain(
                "the task's source text is not available",
                json!({}),
            ));
        }
        let Some(answer) = exp.final_output.as_deref() else {
            return Ok(Finding::decided(false, json!({ "answer": false })));
        };
        grounds.push(task.instruction.clone());
        grounds.extend(
            exp.trajectory
                .steps
                .iter()
                .filter(|s| s.source == StepOrigin::User)
                .skip(1)
                .map(|s| text_of(&s.message)),
        );
        let mut known_words: HashSet<String> = HashSet::new();
        let mut known_numbers: HashSet<String> = HashSet::new();
        for text in &grounds {
            for word in words(text) {
                if let Some(n) = number_of(&word) {
                    known_numbers.insert(n.to_string());
                }
                known_words.insert(word);
            }
        }
        let (numbers, names) = specifics(answer);
        let loose_numbers = numbers
            .iter()
            .filter(|n| !known_numbers.contains(*n))
            .count();
        let loose_names = names.iter().filter(|n| !known_words.contains(*n)).count();
        let index = TextIndex::new(grounds.iter().map(String::as_str));
        let invented_quotes = quotations(answer, self.policy.min_quote_words)
            .iter()
            .filter(|q| !index.contains(q))
            .count();
        let passed = loose_numbers == 0
            && invented_quotes == 0
            && loose_names <= self.policy.max_ungrounded_names;
        Ok(Finding::decided(
            passed,
            json!({
                "comparison": "numbers and names against the source, the instruction and the other speaker's turns",
                "numbers": numbers.len(),
                "numbers_not_in_source": loose_numbers,
                "names": names.len(),
                "names_not_in_source": loose_names,
                "quotations_not_in_source": invented_quotes,
                "max_ungrounded_names": self.policy.max_ungrounded_names,
                "answer": Digest::of(answer.as_bytes()),
            }),
        ))
    }
}
