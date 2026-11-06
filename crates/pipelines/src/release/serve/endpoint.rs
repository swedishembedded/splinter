// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model releases proven to run on the
// plain serving stack before they ship, for its clients. If your team
// needs expertise in rolling fine-tuned models out to a shared inference
// service, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The serve check when the gate runs on brain endpoints.
//!
//! Every answer of the gate then came from the endpoint that serves the
//! candidate, found there by the digest of its adapter file; there is no
//! in-process run to compare a served one with, and no `brain` of the gate's
//! own to start (the cards belong to the endpoint's campaign). What remains to
//! prove is that each endpoint lists the candidate under the digest the
//! release will record, so the release is what was graded
//! ([`Agreement::Listed`]).

use std::path::PathBuf;

use splinter_eval::gate::{self, Agreement, Check, Serve};
use splinter_model::BrainEndpoints;

use super::ServeCheck;

/// The digest reported when an endpoint lists no adapter under the
/// candidate's.
const NOT_LISTED: &str = "not listed";

/// The serve check of `request` against `brain`'s endpoints.
pub(super) fn check(brain: &BrainEndpoints, request: &ServeCheck<'_>) -> Check<Serve> {
    if request.adapter.is_none() {
        return Check::unmeasured(
            "a consolidated checkpoint has no adapter to find on a brain endpoint: it is served \
             as a base the endpoint was started on",
        );
    }
    let mut lines = Vec::new();
    let mut served_digest = request.adapter_digest.to_string();
    for (index, url) in brain.urls().iter().enumerate() {
        let cards = match brain.list(index) {
            Ok(cards) => cards,
            Err(e) => return Check::unmeasured(format!("the serve check: {e}")),
        };
        match cards
            .iter()
            .find(|card| card.adapter_digest.as_deref() == Some(request.adapter_digest))
        {
            Some(card) => lines.push(format!("{url} lists {}", card.id)),
            None => {
                NOT_LISTED.clone_into(&mut served_digest);
                lines.push(format!("{url} lists no adapter with that digest"));
            }
        }
    }
    gate::serve(Serve {
        binary: PathBuf::from(brain.urls().join(",")),
        startup_line: lines.join("; "),
        served_digest,
        expected_digest: request.adapter_digest.to_string(),
        sampled: 0,
        agreed: 0,
        disagreed: Vec::new(),
        agreement: Agreement::Listed,
    })
}
