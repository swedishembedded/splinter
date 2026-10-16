// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the measurement of fine-tuned language
// models for its clients. If your team needs expertise in sizing and
// analysing a paired evaluation with honest statistics, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The report stage: read the study's manifests and outcomes and conclude.

use super::analysis::{across_authors, analyse_author, Report, LEVEL, RESAMPLES};
use super::collect::collect;
use super::layout::Study;
use super::protocol::Multiplicity;

/// The report of the study in `study`. `bootstrap_seed` seeds every interval;
/// when absent a fresh one is drawn and printed, and either way the report
/// records it.
///
/// # Errors
/// As [`collect`].
pub fn run(study: &Study, bootstrap_seed: Option<u64>) -> anyhow::Result<Report> {
    let bootstrap_seed = splinter_sdk::seed::resolve("bootstrap-seed", bootstrap_seed);
    let protocol = study.protocol();
    let collected = collect(study)?;
    let authors: Vec<_> = collected
        .authors
        .iter()
        .map(|input| analyse_author(protocol, input, bootstrap_seed))
        .collect();
    let Multiplicity::HolmOverComparisons { alpha } = protocol.multiplicity;
    Ok(Report {
        protocol_digest: study.digest().to_string(),
        bootstrap_seed,
        resamples: RESAMPLES,
        level: LEVEL,
        alpha,
        across: across_authors(protocol, &authors),
        authors,
        unmeasured: collected.unmeasured,
    })
}
