// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Which conversations with a counterpart may become evidence or training data.
//!
//! sven records every conversation with a counterpart (`sven_sdk::seat`'s
//! `ConversationRecord`) under a provenance: `simulated` (an agent played the
//! seat from a spec) or `live` (a person typed it). Live transcripts are
//! untrusted and may hold secrets or customer data, so they are never pooled
//! with simulated evidence and are used for training only once scrubbed. A
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
    /// A live transcript that has not been scrubbed.
    LiveNotScrubbed,
    /// A live transcript without the person's recorded consent.
    NoConsent,
    /// A simulated transcript with no seat measurement to vouch for it.
    Unmeasured,
    /// The simulated seat leaked the answer key.
    AnswerKeyLeak,
}

/// Admits a conversation record for training or evidence use.
///
/// `scrubbed` states that the caller has already scrubbed this live
/// transcript; it is ignored for a simulated one.
///
/// # Errors
///
/// A [`Refusal`] naming the first rule the record fails.
pub fn admit(record: &Value, scrubbed: bool) -> Result<Pool, Refusal> {
    let provenance = &record["provenance"];
    match provenance["kind"].as_str() {
        Some("live") => {
            if provenance["consent_recorded"] != Value::Bool(true) {
                return Err(Refusal::NoConsent);
            }
            if !scrubbed {
                return Err(Refusal::LiveNotScrubbed);
            }
            Ok(Pool::Live)
        }
        Some("simulated") => {
            let violations = record["checks"]["violations"]
                .as_array()
                .ok_or(Refusal::Unmeasured)?;
            let leaked = violations
                .iter()
                .any(|v| v[1]["kind"].as_str() == Some("answer_key_leak"));
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

    fn live(consent: bool) -> Value {
        json!({"provenance": {"kind": "live", "consent_recorded": consent, "retention_days": 30},
               "checks": null})
    }

    #[test]
    fn a_measured_clean_simulated_conversation_is_admitted_to_its_own_pool() {
        assert_eq!(admit(&simulated(json!([])), false), Ok(Pool::Simulated));
        let role_break = json!([[0, {"kind": "role_break", "detail": "as an ai"}]]);
        assert_eq!(admit(&simulated(role_break), false), Ok(Pool::Simulated));
    }

    #[test]
    fn a_leaking_or_unmeasured_simulated_conversation_is_refused() {
        let leak = json!([[1, {"kind": "answer_key_leak", "detail": "k"}]]);
        assert_eq!(admit(&simulated(leak), false), Err(Refusal::AnswerKeyLeak));
        let unmeasured = json!({"provenance": {"kind": "simulated"}, "checks": null});
        assert_eq!(admit(&unmeasured, false), Err(Refusal::Unmeasured));
    }

    #[test]
    fn a_live_conversation_needs_consent_and_a_scrub_and_stays_in_its_own_pool() {
        assert_eq!(admit(&live(false), true), Err(Refusal::NoConsent));
        assert_eq!(admit(&live(true), false), Err(Refusal::LiveNotScrubbed));
        assert_eq!(admit(&live(true), true), Ok(Pool::Live));
    }

    #[test]
    fn a_record_without_provenance_is_refused() {
        assert_eq!(admit(&json!({}), true), Err(Refusal::UnknownProvenance));
    }
}
