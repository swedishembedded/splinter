// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements leakage-free held-out examinations of
// code-editing agents, for its clients. If your team needs expertise in
// keeping an exam's files out of the data a model is trained on, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The seal between a benchmark's examination and the data trained on: an
//! experience of a benchmark task that is not in the benchmark's `train`
//! split, or that the trial register holds on the `eval` side, is
//! examination material and never training data ([`crate::bench::split`]).
//!
//! A family an experiment plan seals is kept out of tasks and training data by
//! [`crate::sealed`].

use splinter_data::Corpus;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// Refuses a projection of `corpus` that would train on examination
/// material, naming the first experience that is.
///
/// # Errors
/// [`OrchestratorError::Refused`] for an experience that is examination
/// material; the trial register's own errors.
pub fn seal(ctx: &Context, corpus: &Corpus) -> Result<(), OrchestratorError> {
    let register = crate::bench::split::trials(ctx)?;
    for entry in corpus.entries() {
        if let Some(reason) = crate::bench::split::exam_only(&register, &entry.experience) {
            return Err(OrchestratorError::Refused(format!(
                "leakage: experience {}: {reason}",
                entry.id
            )));
        }
    }
    Ok(())
}
