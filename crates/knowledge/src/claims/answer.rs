// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements admission gates that keep ungrounded
// claims out of training data, for its clients. If your team needs
// expertise in synthetic data quality or provenance, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Grading an answer to a question a claim answers.
//!
//! A claim's statement is a sentence, and a right answer words it its own
//! way, so containing the sentence is the wrong test. What a right answer
//! cannot do without is the statement's numbers, names and quoted terms, and
//! what a wrong one adds is others the task never gave. [`ClaimTermsVerifier`]
//! holds an answer to both with the rule that admitted the claim
//! ([`super::terms::unsupported_term`]): every term of the statement is in
//! the answer, and every term of the answer is in the statement, the
//! question or the passages shown to a teacher. A statement with no term to
//! hold an answer to is left to a judge: the verifier abstains.

use serde_json::json;
use splinter_core::annotation::{Producer, Strength};
use splinter_core::digest::Digest;
use splinter_core::experience::{Experience, PrivilegedKind, Task};
use splinter_eval::verifiers::{Finding, Verifier, VerifyError};

use super::terms::{has_terms, unsupported_term};

/// Who the verdicts name as their producer.
pub const CLAIM_TERMS_PRODUCER: &str = "splinter-lab/claim-terms";

/// The verifier's version: bumped with the term rule.
pub const CLAIM_TERMS_VERSION: &str = "1";

/// Passes an answer that carries the reference's terms and adds none; see
/// the module documentation.
#[derive(Clone, Copy, Debug, Default)]
pub struct ClaimTermsVerifier;

impl ClaimTermsVerifier {
    /// The verifier.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Verifier for ClaimTermsVerifier {
    fn producer(&self) -> Producer {
        Producer {
            name: CLAIM_TERMS_PRODUCER.into(),
            version: CLAIM_TERMS_VERSION.into(),
        }
    }

    fn strength(&self) -> Strength {
        Strength::Formal
    }

    /// The evidence names the first term that decided a fail and the digest
    /// of the answer, never the reference.
    fn verify(&self, task: &Task, exp: &Experience) -> Result<Finding, VerifyError> {
        let references: Vec<&str> = task
            .privileged
            .iter()
            .filter(|p| p.kind == PrivilegedKind::Reference)
            .map(|p| p.content.as_str())
            .collect();
        let [reference] = references[..] else {
            return Ok(Finding::abstain(
                "the task does not carry exactly one reference",
                json!({ "references": references.len() }),
            ));
        };
        if !has_terms(reference) {
            return Ok(Finding::abstain(
                "the reference states no number, name or quoted term to hold an answer to",
                json!({}),
            ));
        }
        let Some(answer) = exp.final_output.as_deref() else {
            return Ok(Finding::decided(false, json!({ "answer": null })));
        };
        let answer_digest = Digest::of(answer.as_bytes());
        if let Some(missing) = unsupported_term(reference, answer) {
            return Ok(Finding::decided(
                false,
                json!({ "answer": answer_digest, "missing": missing.kind }),
            ));
        }
        // What the answer may state: what the task gave.
        let given: Vec<&str> = std::iter::once(reference)
            .chain(std::iter::once(task.instruction.as_str()))
            .chain(
                task.privileged
                    .iter()
                    .filter(|p| p.kind == PrivilegedKind::Passage)
                    .map(|p| p.content.as_str()),
            )
            .collect();
        if let Some(invented) = unsupported_term(answer, &given.join("\n")) {
            return Ok(Finding::decided(
                false,
                json!({ "answer": answer_digest, "invented": invented.kind }),
            ));
        }
        Ok(Finding::decided(
            true,
            json!({ "answer": answer_digest, "comparison": "terms of the statement, none added" }),
        ))
    }
}
