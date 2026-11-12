// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that make recordings of real
// people safe to learn from, for its clients. If your team needs expertise
// in scrubbing personal data out of conversations and documents before they
// become training material, you can procure our services by sending an
// email to info@swedishembedded.com.

//! What may enter a selection: documents, with live behaviour data only
//! after it was scrubbed.
//!
//! Material about how real people behave (transcripts of a live user, notes
//! on a client) may hold names, addresses and account numbers. It enters a
//! corpus only if its owner consented, the scrub of the current detectors
//! ran on it, and the detectors find nothing in it when run again: a marker
//! alone does not vouch for text. Material of any other origin (public
//! documents, simulated data) is not held to this.

use serde_json::{json, Value};
use splinter_data::pii::{Scrubber, DETECTORS_VERSION};
use splinter_knowledge::goal::corpus::Document;

/// Where a document came from.
#[derive(Clone, Debug, PartialEq)]
pub enum Origin {
    /// Public, licensed or simulated: no person is in it.
    Open,
    /// Recorded from a real person.
    Live {
        /// Whether the person agreed to its use as training material.
        consent: bool,
        /// The `scrub` entry [`Scrubber::scrub_record`] writes, when a scrub ran.
        scrub: Option<Value>,
    },
}

/// A document and where it came from.
#[derive(Clone, Debug, PartialEq)]
pub struct Intake {
    /// The document.
    pub document: Document,
    /// Its origin.
    pub origin: Origin,
}

/// Why a document did not enter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// A live document whose person did not consent.
    NoConsent,
    /// A live document no scrub ran on, or one by older detectors.
    NotScrubbed,
    /// A live document the detectors still find something in.
    Residue,
}

impl Intake {
    /// A live document, scrubbed now: its text with typed placeholders and
    /// the marker that records the detectors that ran.
    #[must_use]
    pub fn live_scrubbed(document: &Document, consent: bool, scrubber: &Scrubber) -> Self {
        let scrubbed = scrubber.scrub_record(&json!({"text": document.text}));
        Self {
            document: Document {
                title: document.id.clone(),
                text: scrubbed["text"].as_str().unwrap_or_default().to_string(),
                ..document.clone()
            },
            origin: Origin::Live {
                consent,
                scrub: Some(scrubbed["scrub"].clone()),
            },
        }
    }
}

/// The documents that may enter a selection, and the ones refused with
/// their reasons.
#[must_use]
pub fn admit(intakes: &[Intake]) -> (Vec<Document>, Vec<(String, Refusal)>) {
    let scrubber = Scrubber::new();
    let mut admitted = Vec::new();
    let mut refused = Vec::new();
    for intake in intakes {
        let verdict = match &intake.origin {
            Origin::Open => Ok(()),
            Origin::Live { consent: false, .. } => Err(Refusal::NoConsent),
            Origin::Live { scrub, .. } => match scrub {
                Some(s) if s["detectors_version"] == DETECTORS_VERSION => {
                    if scrubber.finds_any(&intake.document.text) {
                        Err(Refusal::Residue)
                    } else {
                        Ok(())
                    }
                }
                _ => Err(Refusal::NotScrubbed),
            },
        };
        match verdict {
            Ok(()) => admitted.push(intake.document.clone()),
            Err(reason) => refused.push((intake.document.id.clone(), reason)),
        }
    }
    (admitted, refused)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Synthetic personal data.
    const NOTE: &str = "When the rent is late, a hesitant client writes to anna.test@example.com and asks for a payment plan.\n";

    fn note() -> Document {
        Document::parse("note", NOTE)
    }

    #[test]
    fn live_behaviour_data_enters_only_scrubbed_and_with_consent() {
        let raw = Intake {
            document: note(),
            origin: Origin::Live {
                consent: true,
                scrub: None,
            },
        };
        let scrubbed = Intake::live_scrubbed(&note(), true, &Scrubber::new());
        let unconsented = Intake::live_scrubbed(&note(), false, &Scrubber::new());
        let open = Intake {
            document: Document::parse("public", "Public text.\n"),
            origin: Origin::Open,
        };
        let (admitted, refused) = admit(&[raw, scrubbed, unconsented, open]);
        assert_eq!(
            refused,
            vec![
                ("note".to_string(), Refusal::NotScrubbed),
                ("note".to_string(), Refusal::NoConsent)
            ]
        );
        let ids: Vec<&str> = admitted.iter().map(|d| d.id.as_str()).collect();
        assert_eq!(ids, ["note", "public"]);
        assert!(!admitted[0].text.contains("example.com"));
    }

    #[test]
    fn a_marker_does_not_vouch_for_text_the_detectors_still_find() {
        let forged = Intake {
            document: note(),
            origin: Origin::Live {
                consent: true,
                scrub: Some(json!({"detectors_version": DETECTORS_VERSION, "counts": {}})),
            },
        };
        let mut old = Intake::live_scrubbed(&note(), true, &Scrubber::new());
        old.origin = Origin::Live {
            consent: true,
            scrub: Some(json!({"detectors_version": 0})),
        };
        assert_eq!(admit(&[forged]).1[0].1, Refusal::Residue);
        assert_eq!(admit(&[old]).1[0].1, Refusal::NotScrubbed);
    }
}
