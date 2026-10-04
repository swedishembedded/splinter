// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verifiers that keep a model speaking as a
// person and not about a document, for its clients. If your team needs
// expertise in training personas on a person's writing, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The speech verifier: a reply is in the speaker's own voice, not about the
//! material the speaker was shown.
//!
//! A teacher shown a person's writing answers as that person. The student it
//! teaches is never shown the writing, so a reply that says "according to the
//! material", "the passage says" or "the speaker" refers to a document the
//! student cannot see, and teaches it to pretend to one. This verifier fails
//! such a reply and says nothing of any other: its strength is a
//! [`Strength::Constraint`], so a pass establishes nothing and only a verifier
//! that can say a reply is right can pass it.
//!
//! It reads the final output and looks for a short list of phrasings by their
//! words, so a speaker who mentions their own letters, or the text of a
//! constitution, is not mistaken for one who is reading a source.

use serde_json::json;
use splinter_core::annotation::{Producer, Strength};
use splinter_core::experience::{Experience, Task};

use super::{Finding, Verifier, VerifyError};

/// The nouns that name a document the speaker was shown.
const DOCUMENTS: [&str; 9] = [
    "material", "passage", "excerpt", "document", "text", "record", "records", "section",
    "sections",
];

/// What a document does, said by someone reading it.
const READS: [&str; 15] = [
    "says",
    "states",
    "mentions",
    "indicates",
    "describes",
    "provides",
    "gives",
    "notes",
    "suggests",
    "provided",
    "given",
    "does",
    "doesn't",
    "do",
    "did",
];

/// The speech verifier; see the module documentation.
pub struct SpeechVerifier {
    producer: Producer,
}

impl SpeechVerifier {
    /// A speech verifier whose verdicts name `producer`.
    #[must_use]
    pub fn new(producer: Producer) -> Self {
        Self { producer }
    }
}

/// The lower-case words of `text`, apostrophes kept inside a word.
fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !(c.is_alphanumeric() || c == '\''))
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// How many phrasings of `text` refer to a document the speaker is reading.
fn document_references(text: &str) -> usize {
    let w = words(text);
    let is = |n: usize, set: &[&str]| w.get(n).is_some_and(|word| set.contains(&word.as_str()));
    let eq = |n: usize, to: &str| w.get(n).is_some_and(|word| word == to);
    let mut found = 0;
    for n in 0..w.len() {
        // "according to the <document>"
        let according =
            eq(n, "according") && eq(n + 1, "to") && eq(n + 2, "the") && is(n + 3, &DOCUMENTS);
        // "the provided <document>", "the given <document>"
        let provided =
            eq(n, "the") && (eq(n + 1, "provided") || eq(n + 1, "given")) && is(n + 2, &DOCUMENTS);
        // "the <document> says" and its kind, but "the text of ..." is no one reading
        let reads = eq(n, "the") && is(n + 1, &DOCUMENTS) && is(n + 2, &READS);
        // "the speaker", said of the writer
        let speaker = eq(n, "the") && eq(n + 1, "speaker");
        if according || provided || reads || speaker {
            found += 1;
        }
    }
    found
}

impl Verifier for SpeechVerifier {
    fn producer(&self) -> Producer {
        self.producer.clone()
    }

    fn strength(&self) -> Strength {
        Strength::Constraint
    }

    /// The evidence counts the phrasings found, never quotes them.
    fn verify(&self, _task: &Task, exp: &Experience) -> Result<Finding, VerifyError> {
        let Some(reply) = exp.final_output.as_deref() else {
            return Ok(Finding::abstain("there is no reply to read", json!({})));
        };
        let found = document_references(reply);
        Ok(Finding::decided(
            found == 0,
            json!({ "document_references": found }),
        ))
    }
}
