// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Which conversations with a counterpart may become evidence or training data.
//!
//! sven records every conversation with a counterpart (`sven_sdk::seat`'s
//! `ConversationRecord`) under a provenance: `simulated` (an agent played the
//! seat from a spec) or `live` (a person typed it). Live transcripts are
//! untrusted and may hold secrets or customer data, so they are never pooled
//! with simulated evidence and are admitted only once [`Scrubber::scrub_record`]
//! ran on them (the record says which detector version did) and no detector
//! still finds anything in it; the caller does not get to assert it. A
//! simulated transcript is usable only if its seat was measured clean: an
//! answer-key leak makes the run worthless as evidence.
//!
//! The record is read as JSON, by its documented fields, so this gate does not
//! move when sven's types do.
//!
//! Swedish Embedded AB implements data admission controls for learning
//! pipelines for its clients. If your team needs expertise in keeping
//! untrusted user data out of training sets then you can procure our services
//! by sending an email to info@swedishembedded.com.

use serde_json::Value;
use splinter_data::pii::{Scrubber, DETECTORS_VERSION};

/// The pool an admitted conversation belongs to. Pools are reported
/// separately, never summed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pool {
    /// Played by a simulated seat: reproducible, the source of bench numbers.
    Simulated,
    /// A person was on the other side: reported separately, scrubbed first.
    Live,
}

/// Why a conversation is not admitted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// The record has no recognisable provenance.
    UnknownProvenance,
    /// A live transcript that no scrub ran on, or one by older detectors.
    LiveNotScrubbed,
    /// A live transcript still holding something a detector finds.
    ResidualPii,
    /// A live transcript without the person's recorded consent.
    NoConsent,
    /// A simulated transcript with no seat measurement to vouch for it.
    Unmeasured,
    /// The simulated seat leaked the answer key.
    AnswerKeyLeak,
}

/// Admits a conversation record for training or evidence use.
///
/// A live record must carry the `scrub` entry [`Scrubber::scrub_record`]
/// writes, from the current detectors, and must be clean when checked again.
///
/// # Errors
///
/// A [`Refusal`] naming the first rule the record fails.
pub fn admit(record: &Value) -> Result<Pool, Refusal> {
    let provenance = &record["provenance"];
    match provenance["kind"].as_str() {
        Some("live") => {
            if provenance["consent_recorded"] != Value::Bool(true) {
                return Err(Refusal::NoConsent);
            }
            if record["scrub"]["detectors_version"] != DETECTORS_VERSION {
                return Err(Refusal::LiveNotScrubbed);
            }
            if Scrubber::new().record_has_residue(record) {
                return Err(Refusal::ResidualPii);
            }
            Ok(Pool::Live)
        }
        Some("simulated") => {
            let violations = record["checks"]["violations"]
                .as_array()
                .ok_or(Refusal::Unmeasured)?;
            let leaked = violations.iter().any(|v| {
                matches!(
                    v[1]["kind"].as_str(),
                    Some("answer_key_leak" | "answer_key_paraphrase")
                )
            });
            if leaked {
                return Err(Refusal::AnswerKeyLeak);
            }
            Ok(Pool::Simulated)
        }
        _ => Err(Refusal::UnknownProvenance),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn simulated(violations: Value) -> Value {
        json!({"provenance": {"kind": "simulated", "spec_id": "s", "spec_sha256": "x", "seed": 1},
               "checks": {"replies": 2, "clean_replies": 1, "violations": violations}})
    }

    fn live(consent: bool, text: &str) -> Value {
        json!({"provenance": {"kind": "live", "consent_recorded": consent, "retention_days": 30},
               "checks": null,
               "exchanges": [{"speaker": "seat", "text": text}]})
    }

    const SYNTHETIC: &str = "Mail me at anna@example.se or ring 070-123 45 67.";

    #[test]
    fn a_measured_clean_simulated_conversation_is_admitted_to_its_own_pool() {
        assert_eq!(admit(&simulated(json!([]))), Ok(Pool::Simulated));
        let role_break = json!([[0, {"kind": "role_break", "detail": "as an ai"}]]);
        assert_eq!(admit(&simulated(role_break)), Ok(Pool::Simulated));
    }

    #[test]
    fn a_leaking_or_unmeasured_simulated_conversation_is_refused() {
        let leak = json!([[1, {"kind": "answer_key_leak", "detail": "k"}]]);
        assert_eq!(admit(&simulated(leak)), Err(Refusal::AnswerKeyLeak));
        let paraphrase = json!([[0, {"kind": "answer_key_paraphrase", "detail": "k"}]]);
        assert_eq!(admit(&simulated(paraphrase)), Err(Refusal::AnswerKeyLeak));
        let unmeasured = json!({"provenance": {"kind": "simulated"}, "checks": null});
        assert_eq!(admit(&unmeasured), Err(Refusal::Unmeasured));
    }

    #[test]
    fn a_live_conversation_needs_consent_and_a_scrub_and_stays_in_its_own_pool() {
        let scrubber = Scrubber::new();
        let raw = live(true, SYNTHETIC);
        assert_eq!(admit(&live(false, "hi")), Err(Refusal::NoConsent));
        assert_eq!(admit(&raw), Err(Refusal::LiveNotScrubbed));
        let clean = scrubber.scrub_record(&raw);
        assert_eq!(admit(&clean), Ok(Pool::Live));
        assert_eq!(clean["scrub"]["counts"]["email"], 1);
    }

    #[test]
    fn a_scrub_marker_does_not_vouch_for_text_the_detectors_still_find() {
        let mut forged = live(true, SYNTHETIC);
        forged["scrub"] = json!({"detectors_version": DETECTORS_VERSION, "counts": {}});
        assert_eq!(admit(&forged), Err(Refusal::ResidualPii));
        let mut old = Scrubber::new().scrub_record(&live(true, SYNTHETIC));
        old["scrub"]["detectors_version"] = json!(0);
        assert_eq!(admit(&old), Err(Refusal::LiveNotScrubbed));
    }

    #[test]
    fn a_record_without_provenance_is_refused() {
        assert_eq!(admit(&json!({})), Err(Refusal::UnknownProvenance));
    }
}
