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
//! made from what they need: [`evidence_verifier`] from the source store
//! the task's evidence is in, [`runtime_verifier`] from the runtimes its
//! code runs in. A judge and a workspace are a run's own.

use splinter_core::annotation::Producer;
use splinter_eval::verifiers::executable::ExecutableVerifier;
use splinter_eval::verifiers::form::{FinalNumberVerifier, LineCountVerifier};
use splinter_eval::verifiers::formal::{ExactMatchVerifier, StatedReferenceVerifier};
use splinter_eval::verifiers::grounding::{GroundingPolicy, GroundingVerifier};
use splinter_eval::verifiers::mutation::{MutationPolicy, MutationValidatedVerifier};
use splinter_eval::verifiers::normalise::Normalisation;
use splinter_eval::verifiers::quotation::{QuotationPolicy, QuotationVerifier, StoredEvidence};
use splinter_eval::verifiers::speech::SpeechVerifier;
use splinter_eval::verifiers::Verifier;
use splinter_sandbox::RuntimeEnvironment;
use splinter_store::sources::SourceStore;

use super::VerifierKind;
use crate::claims::answer::ClaimTermsVerifier;

/// The producer of the formal verifier's verdicts on generated tasks.
pub const FORMAL_PRODUCER: &str = "splinter-lab/exact-match:lenient";

/// The formal verifier's version: bumped with its normalisation.
pub const FORMAL_VERSION: &str = "1";

/// The producer of the stated-reference verifier's verdicts.
pub const STATED_PRODUCER: &str = "splinter-lab/reference-stated:lenient";

/// The stated-reference verifier's version: bumped with its normalisation,
/// its bound on the answer's length or what it takes for a hedge.
pub const STATED_VERSION: &str = "2";

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

/// The producer of the quotation verifier's verdicts.
pub const QUOTATION_PRODUCER: &str = "splinter-lab/quotation";

/// The quotation verifier's version: bumped with its policy or the way it
/// reads a quotation.
pub const QUOTATION_VERSION: &str = "1";

/// The fewest words a quoted passage runs to count as a claim to a passage.
pub const QUOTATION_MIN_WORDS: usize = 8;

/// The share of the reference advice an answer must reproduce.
pub const QUOTATION_MIN_RECALL: f64 = 0.5;

/// Who the grounding verdicts name as their producer.
pub const GROUNDING_PRODUCER: &str = "splinter-lab/grounding";

/// The version of the grounding check; a changed check carries a new one.
pub const GROUNDING_VERSION: &str = "1";

/// The proper names a dialogue may state that nothing in its source grounds:
/// the check cannot tell a form of address or a common name from a person,
/// and one stray name is not the invented specifics it exists to refuse.
pub const GROUNDING_MAX_NAMES: usize = 1;

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

/// The grounding check over the sources in `store`.
#[must_use]
pub fn grounding_verifier(store: SourceStore) -> GroundingVerifier {
    GroundingVerifier::new(
        producer(GROUNDING_PRODUCER, GROUNDING_VERSION),
        Box::new(StoredEvidence::new(store)),
        GroundingPolicy {
            max_ungrounded_names: GROUNDING_MAX_NAMES,
            min_quote_words: QUOTATION_MIN_WORDS,
        },
    )
}

/// The verifier `kind` names when it reads the task's evidence in its
/// source store: a quotation, or grounding; `None` otherwise.
#[must_use]
pub fn evidence_verifier(kind: VerifierKind, store: &SourceStore) -> Option<Box<dyn Verifier>> {
    Some(match kind {
        VerifierKind::Quotation => Box::new(QuotationVerifier::new(
            producer(QUOTATION_PRODUCER, QUOTATION_VERSION),
            Box::new(StoredEvidence::new(store.clone())),
            QuotationPolicy {
                min_words: QUOTATION_MIN_WORDS,
                min_reference_recall: QUOTATION_MIN_RECALL,
            },
        )),
        VerifierKind::Grounding => Box::new(grounding_verifier(store.clone())),
        _ => return None,
    })
}

/// The verifier `kind` names when it runs the answer's code in
/// `runtimes`: its checks, or its generated tests; `None` otherwise.
#[must_use]
pub fn runtime_verifier(
    kind: VerifierKind,
    runtimes: Vec<RuntimeEnvironment>,
) -> Option<Box<dyn Verifier>> {
    Some(match kind {
        VerifierKind::Executable => Box::new(ExecutableVerifier::new(runtimes)),
        VerifierKind::MutationValidated => Box::new(MutationValidatedVerifier::new(
            runtimes,
            MutationPolicy::default(),
        )),
        _ => return None,
    })
}
