// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning agents whose held-out measurement
// cannot leak, for its clients. If your team needs expertise in evaluating a
// model on documents it has not seen when its sources overlap, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Naming, in a dataset, which records came from overlapping source text.
//!
//! Sources overlap: two editions print the same letter. A held-out split
//! that divides such records between training and scoring measures memory of
//! the print, not learning, so each record carries the group of its evidence
//! (`splinter_eval::overlap`), and the split keeps a group whole
//! (`splinter_data::holdout`).

use std::collections::{BTreeMap, BTreeSet};

use splinter_core::digest::Digest;
use splinter_core::experience::Experience;
use splinter_data::{Corpus, Projection};
use splinter_eval::overlap::overlap_groups;

use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// Sets each record of `projection` to the group of the source text its
/// experience is grounded in. A record grounded in several texts takes the
/// least of their groups; one grounded in none has no group.
pub(crate) fn assign_groups(
    ctx: &Context,
    corpus: &Corpus,
    projection: &mut Projection,
) -> Result<(), OrchestratorError> {
    let evidence = |e: &Experience| -> Vec<Digest> {
        e.evidence.iter().map(|span| span.source.clone()).collect()
    };
    let mut blobs: BTreeSet<Digest> = BTreeSet::new();
    for entry in corpus.entries() {
        blobs.extend(evidence(&entry.experience));
    }
    let blobs: Vec<Digest> = blobs.into_iter().collect();
    let sources = ctx.sources();
    let texts = blobs
        .iter()
        .map(|digest| {
            sources
                .read_blob(digest)
                .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let group_of: BTreeMap<&Digest, String> = overlap_groups(&refs)
        .into_iter()
        .zip(&blobs)
        .map(|(first, blob)| (blob, blobs[first].as_str().to_string()))
        .collect();
    let by_experience: BTreeMap<_, _> = corpus
        .entries()
        .iter()
        .map(|entry| (&entry.id, &entry.experience))
        .collect();
    for record in &mut projection.records {
        record.metadata.group = record
            .metadata
            .experiences
            .first()
            .and_then(|id| by_experience.get(id))
            .and_then(|experience| {
                evidence(experience)
                    .iter()
                    .filter_map(|blob| group_of.get(blob).cloned())
                    .min()
            });
    }
    Ok(())
}
