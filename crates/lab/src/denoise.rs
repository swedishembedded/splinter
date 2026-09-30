// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verifiers that grade agent answers against
// references the agent never saw, for its clients. If your team needs
// expertise in verifier design or learning from verified experience, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The denoise task family's verifier.
//!
//! A denoise task shows the student a corrupted passage and asks for the
//! original; the original travels with the experience as its one
//! [`PrivilegedKind::Reference`]. [`FormalVerifier`] passes an answer iff it
//! equals that reference once whitespace is normalised (every run of
//! whitespace one space, none at either end). The comparison is exact
//! otherwise - case and punctuation count - so the verdict is formal, not
//! judged.

use splinter_store::annotation::{Annotation, AnnotationBody, Outcome, Producer, Strength};
use splinter_store::experience::{Digest, Experience, ExperienceError, PrivilegedKind};

/// The task kind denoise tasks carry, shared by the generator and this
/// verifier.
pub const KIND: &str = "denoise";

/// The producer name the verifier's annotations carry.
pub const PRODUCER: &str = "splinter-lab/denoise-formal";

/// The verifier's version: bumped whenever what it accepts changes, so its
/// old verdicts stay distinguishable from new ones.
pub const VERSION: &str = "1";

/// Why the verifier could not judge an experience.
#[derive(Debug, thiserror::Error)]
pub enum VerifyError {
    /// The experience is not a denoise task.
    #[error("the denoise verifier cannot judge a {kind:?} task")]
    WrongKind {
        /// The experience's task kind.
        kind: String,
    },
    /// The experience does not carry exactly one reference passage.
    #[error("a denoise experience carries exactly one reference; this one carries {found}")]
    Reference {
        /// How many it carries.
        found: usize,
    },
    /// The experience's id cannot be computed.
    #[error(transparent)]
    Experience(#[from] ExperienceError),
}

/// Grades a denoise experience's final output against its reference.
#[derive(Clone, Copy, Debug, Default)]
pub struct FormalVerifier;

impl FormalVerifier {
    /// The verifier.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// The verdict on `experience`: pass iff its final output equals the
    /// reference after whitespace normalisation, fail otherwise (no output
    /// included). The evidence names the comparison and the digests of the
    /// two normalised texts, never the reference itself.
    pub fn verify(&self, experience: &Experience) -> Result<Annotation, VerifyError> {
        if experience.task.kind != KIND {
            return Err(VerifyError::WrongKind {
                kind: experience.task.kind.clone(),
            });
        }
        let references: Vec<&str> = experience
            .privileged
            .iter()
            .filter(|p| p.kind == PrivilegedKind::Reference)
            .map(|p| p.content.as_str())
            .collect();
        let [reference] = references[..] else {
            return Err(VerifyError::Reference {
                found: references.len(),
            });
        };
        let reference = normalise(reference);
        let output = experience.final_output.as_deref().map(normalise);
        let passed = output.as_deref() == Some(reference.as_str());
        Ok(Annotation {
            experience: experience.id()?,
            producer: Producer {
                name: PRODUCER.into(),
                version: VERSION.into(),
            },
            body: AnnotationBody::Verdict {
                outcome: if passed { Outcome::Pass } else { Outcome::Fail },
                strength: Strength::Formal,
                evidence: serde_json::json!({
                    "comparison": "exact after whitespace normalisation",
                    "reference": Digest::of(reference.as_bytes()),
                    "output": output.map(|o| Digest::of(o.as_bytes())),
                }),
            },
        })
    }
}

/// Every run of whitespace one space, none at either end.
fn normalise(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}
