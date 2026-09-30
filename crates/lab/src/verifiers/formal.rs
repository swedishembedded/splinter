// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verifiers that grade agent answers against
// references the agent never saw, for its clients. If your team needs
// expertise in verifier design, you can procure our services by sending an
// email to info@swedishembedded.com.

//! The formal verifier: the answer against the task's one reference, under
//! a configured [`Normalisation`].

use serde_json::json;
use splinter_store::annotation::{Producer, Strength};
use splinter_store::digest::Digest;
use splinter_store::experience::{Experience, Task};

use super::normalise::Normalisation;
use super::{single_reference, Finding, Verifier, VerifyError};

/// Passes an answer iff it equals the task's one
/// [`Reference`](splinter_store::experience::PrivilegedKind::Reference)
/// once both are normalised; fails a different answer or none. Abstains on
/// a task without exactly one reference, or of a kind it is not for.
#[derive(Clone, Debug)]
pub struct ExactMatchVerifier {
    producer: Producer,
    normalisation: Normalisation,
    kind: Option<String>,
}

impl ExactMatchVerifier {
    /// A verifier for tasks of every kind, its verdicts carrying `producer`.
    #[must_use]
    pub fn new(producer: Producer, normalisation: Normalisation) -> Self {
        Self {
            producer,
            normalisation,
            kind: None,
        }
    }

    /// The same verifier, applying only to tasks of `kind`.
    #[must_use]
    pub fn for_kind(mut self, kind: &str) -> Self {
        self.kind = Some(kind.to_string());
        self
    }
}

impl Verifier for ExactMatchVerifier {
    fn producer(&self) -> Producer {
        self.producer.clone()
    }

    fn strength(&self) -> Strength {
        Strength::Formal
    }

    /// The evidence names the normalisation and the digests of the two
    /// normalised texts, never the reference itself.
    fn verify(&self, task: &Task, exp: &Experience) -> Result<Finding, VerifyError> {
        if let Some(kind) = &self.kind {
            if task.task.kind != *kind {
                return Ok(Finding::abstain(
                    "the verifier is for another task kind",
                    json!({ "for": kind, "task": task.task.kind }),
                ));
            }
        }
        let reference = match single_reference(task) {
            Ok(reference) => self.normalisation.apply(reference),
            Err(abstain) => return Ok(abstain),
        };
        let output = exp
            .final_output
            .as_deref()
            .map(|o| self.normalisation.apply(o));
        let passed = output.as_deref() == Some(reference.as_str());
        Ok(Finding::decided(
            passed,
            json!({
                "comparison": "exact after normalisation",
                "normalisation": self.normalisation,
                "reference": Digest::of(reference.as_bytes()),
                "output": output.map(|o| Digest::of(o.as_bytes())),
            }),
        ))
    }
}
