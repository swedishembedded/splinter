// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The views, one module each; the crate documentation lists what each
//! projects.

mod cpt;
mod critic;
mod decision;
mod denoise;
mod outcome;
mod preference;
mod rehearsal;
mod retrieval;
mod sft_final;
mod sft_step;
mod verifier;
mod voice;

pub use cpt::Cpt;
pub use critic::Critic;
pub use decision::DecisionView;
pub use denoise::DenoiseView;
pub use outcome::OutcomeView;
pub use preference::Preference;
pub use rehearsal::Rehearsal;
pub use retrieval::Retrieval;
pub use sft_final::SftFinal;
pub use sft_step::SftStep;
pub use verifier::VerifierView;
pub use voice::{chars_as_tokens, Sectioner, Voice};

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
