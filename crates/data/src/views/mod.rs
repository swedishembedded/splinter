// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The views, one module each; the crate documentation lists what each
//! projects.

mod attempts;
mod cpt;
mod critic;
mod decision;
mod denoise;
mod feedback;
mod outcome;
mod policy_gradient;
mod preference;
mod privileged;
mod rehearsal;
mod rendering;
mod retrieval;
mod rewards;
mod sft_final;
mod sft_step;
mod transition;
mod verifier;
mod voice;

pub use cpt::Cpt;
pub use critic::Critic;
pub use decision::DecisionView;
pub use denoise::DenoiseView;
pub use feedback::{Feedback, FeedbackBalance};
pub use outcome::OutcomeView;
pub use policy_gradient::{Distill, PolicyGradient};
pub use preference::{PairingPolicy, Preference, PAIRING_RULE};
pub use rehearsal::Rehearsal;
pub use retrieval::Retrieval;
pub use rewards::signals;
pub use rewards::{
    AdvantageFloor, AdvantageKind, AdvantageNorm, GroupKind, GroupReport, Overlong, PartialCredit,
    RewardShape, TrialReport,
};
pub use sft_final::SftFinal;
pub use sft_step::{Control, SftStep};
pub use transition::{Transition, DEFAULT_PER_TRANSITION_CAP};
pub use verifier::VerifierView;
pub use voice::description;
pub use voice::{chars_as_tokens, DescriptionOf, Sectioner, Voice, DEFAULT_MAX_FAMILY_SHARE};

use splinter_store::experiences::StoreError;

use crate::{Exclusion, ViewError};

/// Source content read from the store, as text: an exclusion when the
/// store does not hold it or it is not non-empty UTF-8, an error for any
/// other failure (corruption is never a reason to skip quietly).
fn source_text(read: Result<Vec<u8>, StoreError>) -> Result<Result<String, Exclusion>, ViewError> {
    let bytes = match read {
        Ok(bytes) => bytes,
        Err(
            StoreError::UnknownSource(_)
            | StoreError::UnknownPart { .. }
            | StoreError::UnknownBlob(_),
        ) => return Ok(Err(Exclusion::SourceMissing)),
        Err(e) => return Err(e.into()),
    };
    Ok(match String::from_utf8(bytes) {
        Ok(text) if text.is_empty() => Err(Exclusion::Empty),
        Ok(text) => Ok(text),
        Err(_) => Err(Exclusion::NotText),
    })
}
