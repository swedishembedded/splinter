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
//! [`Reference`](splinter_record::experience::PrivilegedKind::Reference).
//! [`FormalVerifier`] passes an answer iff it equals that reference once
//! whitespace is normalised (every run of whitespace one space, none at
//! either end). The comparison is exact otherwise - case and punctuation
//! count - so the verdict is formal, not judged. It is the
//! [`ExactMatchVerifier`] with that normalisation, for denoise tasks only.

use splinter_record::annotation::{Producer, Strength};
use splinter_record::experience::{Experience, Task};

use crate::verifiers::formal::ExactMatchVerifier;
use crate::verifiers::normalise::Normalisation;
use crate::verifiers::{Finding, Verifier, VerifyError};

/// The task kind denoise tasks carry, shared by the generator and this
/// verifier.
pub const KIND: &str = "denoise";

/// The producer name the verifier's annotations carry.
pub const PRODUCER: &str = "splinter-lab/denoise-formal";

/// The verifier's version: bumped whenever what it accepts changes, so its
/// old verdicts stay distinguishable from new ones.
pub const VERSION: &str = "1";

/// Grades a denoise experience's final output against its reference.
#[derive(Clone, Debug)]
pub struct FormalVerifier(ExactMatchVerifier);

impl Default for FormalVerifier {
    fn default() -> Self {
        Self::new()
    }
}

impl FormalVerifier {
    /// The verifier.
    #[must_use]
    pub fn new() -> Self {
        let producer = Producer {
            name: PRODUCER.into(),
            version: VERSION.into(),
        };
        Self(ExactMatchVerifier::new(producer, Normalisation::WHITESPACE).for_kind(KIND))
    }
}

impl Verifier for FormalVerifier {
    fn producer(&self) -> Producer {
        self.0.producer()
    }

    fn strength(&self) -> Strength {
        self.0.strength()
    }

    /// Pass iff the final output equals the reference after whitespace
    /// normalisation, fail otherwise (no output included); abstains on a
    /// task of another kind or without exactly one reference.
    fn verify(&self, task: &Task, exp: &Experience) -> Result<Finding, VerifyError> {
        self.0.verify(task, exp)
    }
}
