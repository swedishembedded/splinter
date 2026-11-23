// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model evaluations whose every verdict is
// produced by code that names itself, for its clients. If your team needs
// expertise in rigorous answer verification, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The verifiers of a task kind that need nothing of a run's environment.
//!
//! A task kind lists the verifiers its answers are held to
//! ([`VerifierKind`]). Some of them read only the task and the answer: an
//! exact match under a normalisation, a stated reference, the claim gate's
//! terms, a speech's form, a final number, a line count. [`plain_verifier`]
//! makes those, the same wherever an answer is graded, by the producer names
//! and versions every verdict they write carries. The others need what a
//! run has (a runtime, the source store, a judge model, a workspace) and are
//! made by whoever has it.

use splinter_core::annotation::Producer;
use splinter_eval::verifiers::form::{FinalNumberVerifier, LineCountVerifier};
use splinter_eval::verifiers::formal::{ExactMatchVerifier, StatedReferenceVerifier};
use splinter_eval::verifiers::normalise::Normalisation;
use splinter_eval::verifiers::speech::SpeechVerifier;
use splinter_eval::verifiers::Verifier;

use super::VerifierKind;
use crate::claims::answer::ClaimTermsVerifier;

/// The producer of the formal verifier's verdicts on generated tasks.
pub const FORMAL_PRODUCER: &str = "splinter-lab/exact-match:lenient";

/// The formal verifier's version: bumped with its normalisation.
pub const FORMAL_VERSION: &str = "1";

/// The producer of the stated-reference verifier's verdicts.
pub const STATED_PRODUCER: &str = "splinter-lab/reference-stated:lenient";

/// The stated-reference verifier's version: bumped with its normalisation
/// or its bound on the answer's length.
pub const STATED_VERSION: &str = "1";

/// Who the speech verdicts name as their producer.
pub const SPEECH_PRODUCER: &str = "splinter-lab/speech";

/// The version of the speech check; a changed check carries a new one.
pub const SPEECH_VERSION: &str = "1";

/// The producer of the final-number verifier's verdicts.
pub const FINAL_NUMBER_PRODUCER: &str = "splinter-lab/final-number";
/// Its version.
pub const FINAL_NUMBER_VERSION: &str = "1";

/// The producer of the line-count verifier's verdicts.
pub const LINE_COUNT_PRODUCER: &str = "splinter-lab/line-count";
/// Its version.
pub const LINE_COUNT_VERSION: &str = "1";

fn producer(name: &str, version: &str) -> Producer {
    Producer {
        name: name.into(),
        version: version.into(),
    }
}

/// The verifier `kind` names when it reads only the task and the answer;
/// `None` for a kind that needs a run's environment.
#[must_use]
pub fn plain_verifier(kind: VerifierKind) -> Option<Box<dyn Verifier>> {
    Some(match kind {
        VerifierKind::Formal => Box::new(ExactMatchVerifier::new(
            producer(FORMAL_PRODUCER, FORMAL_VERSION),
            Normalisation::LENIENT,
        )),
        VerifierKind::Stated => Box::new(StatedReferenceVerifier::new(
            producer(STATED_PRODUCER, STATED_VERSION),
            Normalisation::LENIENT,
        )),
        VerifierKind::Terms => Box::new(ClaimTermsVerifier::new()),
        VerifierKind::Speech => Box::new(SpeechVerifier::new(producer(
            SPEECH_PRODUCER,
            SPEECH_VERSION,
        ))),
        VerifierKind::FinalNumber => Box::new(FinalNumberVerifier::new(producer(
            FINAL_NUMBER_PRODUCER,
            FINAL_NUMBER_VERSION,
        ))),
        VerifierKind::LineCount => Box::new(LineCountVerifier::new(producer(
            LINE_COUNT_PRODUCER,
            LINE_COUNT_VERSION,
        ))),
        _ => return None,
    })
}
