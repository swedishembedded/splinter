// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements speaking assistants assembled from parts that
// are checked to belong together, for its clients. If your team needs
// expertise in composing speech models without retraining them, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The parts of a speaking persona and the rule that they belong together.
//!
//! A persona that listens and speaks is a language model (the thinker), an
//! adapter that makes it the persona, a projector that carries a recogniser's
//! features into the thinker's input, the recogniser and synthesizer those
//! rely on, and a voice. Each is trained or chosen against particular other
//! parts: a projector fits one thinker, a persona adapter one base. A bundle
//! names every part by content address and what it was made for, and refuses
//! to exist when a part is attached to something it was not made for.

use serde::{Deserialize, Serialize};

use crate::digest::{canonical_json, Digest};
use crate::speech::SpeakerProfile;

/// One part: where it comes from and the content address of what was used.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Part {
    /// What it is called where it comes from (a model store reference, a file name).
    pub id: String,
    /// The content address of the bytes that were used.
    pub digest: Digest,
}

/// Why a bundle is not one.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum BundleError {
    /// The persona adapter was trained on a different base than the thinker.
    #[error("the persona adapter was trained on {adapter_base} but the thinker is {thinker}")]
    AdapterBase {
        /// The base the adapter names.
        adapter_base: Digest,
        /// The thinker's address.
        thinker: Digest,
    },
    /// The speech projector was trained against a different thinker.
    #[error("the speech projector was trained against {trained_on} but the thinker is {thinker}")]
    ProjectorThinker {
        /// The thinker the projector names.
        trained_on: Digest,
        /// The thinker's address.
        thinker: Digest,
    },
    /// The projector reads features of another recogniser than the one named.
    #[error(
        "the speech projector reads the features of {expected} but the recogniser is {recogniser}"
    )]
    ProjectorRecogniser {
        /// The recogniser the projector names.
        expected: Digest,
        /// The recogniser's address.
        recogniser: Digest,
    },
    /// The persona's name is blank.
    #[error("a persona has a name")]
    Unnamed,
}

/// A part made for another part: it names that other part's address.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MadeFor {
    /// The part itself.
    pub part: Part,
    /// The address of the part it was trained against.
    pub made_for: Digest,
}

/// A speaking persona: every part, checked.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpeechBundle {
    persona: String,
    thinker: Part,
    persona_adapter: Option<MadeFor>,
    projector: MadeFor,
    recogniser: Part,
    synthesizer: Part,
    voice: SpeakerProfile,
}

impl SpeechBundle {
    /// A bundle of these parts, if they belong together: the adapter was
    /// trained on the thinker, the projector against the thinker and on the
    /// recogniser's features (`projector.part` names the projector;
    /// `projector_recogniser` is the recogniser it reads).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        persona: impl Into<String>,
        thinker: Part,
        persona_adapter: Option<MadeFor>,
        projector: MadeFor,
        projector_recogniser: &Digest,
        recogniser: Part,
        synthesizer: Part,
        voice: SpeakerProfile,
    ) -> Result<Self, BundleError> {
        let persona = persona.into();
        if persona.trim().is_empty() {
            return Err(BundleError::Unnamed);
        }
        if let Some(adapter) = &persona_adapter {
            if adapter.made_for != thinker.digest {
                return Err(BundleError::AdapterBase {
                    adapter_base: adapter.made_for.clone(),
                    thinker: thinker.digest,
                });
            }
        }
        if projector.made_for != thinker.digest {
            return Err(BundleError::ProjectorThinker {
                trained_on: projector.made_for.clone(),
                thinker: thinker.digest,
            });
        }
        if *projector_recogniser != recogniser.digest {
            return Err(BundleError::ProjectorRecogniser {
                expected: projector_recogniser.clone(),
                recogniser: recogniser.digest,
            });
        }
        Ok(Self {
            persona,
            thinker,
            persona_adapter,
            projector,
            recogniser,
            synthesizer,
            voice,
        })
    }

    /// The person the thinker answers as.
    #[must_use]
    pub fn persona(&self) -> &str {
        &self.persona
    }

    /// The same bundle answering as another persona: a new adapter on the same
    /// thinker. The projector, recogniser, synthesizer and voice stay, because
    /// none of them was made for a persona.
    pub fn with_persona(
        &self,
        persona: impl Into<String>,
        adapter: MadeFor,
    ) -> Result<Self, BundleError> {
        Self::new(
            persona,
            self.thinker.clone(),
            Some(adapter),
            self.projector.clone(),
            &self.recogniser.digest,
            self.recogniser.clone(),
            self.synthesizer.clone(),
            self.voice.clone(),
        )
    }

    /// The voice the persona speaks in, with its portrayal.
    #[must_use]
    pub fn voice(&self) -> &SpeakerProfile {
        &self.voice
    }

    /// The content address of the bundle.
    pub fn digest(&self) -> Result<Digest, serde_json::Error> {
        Ok(Digest::of(&canonical_json(self)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speech::Portrayal;

    fn d(tag: &str) -> Digest {
        Digest::of(tag.as_bytes())
    }

    fn part(id: &str) -> Part {
        Part {
            id: id.into(),
            digest: d(id),
        }
    }

    fn made_for(id: &str, thinker: &str) -> MadeFor {
        MadeFor {
            part: part(id),
            made_for: d(thinker),
        }
    }

    fn bundle(
        adapter_base: &str,
        projector_thinker: &str,
        projector_reads: &str,
    ) -> Result<SpeechBundle, BundleError> {
        SpeechBundle::new(
            "Samuel Adams",
            part("thinker"),
            Some(made_for("adams-adapter", adapter_base)),
            made_for("projector", projector_thinker),
            &d(projector_reads),
            part("recogniser"),
            part("synthesizer"),
            SpeakerProfile::new(1, Portrayal::synthetic_theatrical()),
        )
    }

    #[test]
    fn parts_made_for_each_other_make_a_bundle() {
        let b = bundle("thinker", "thinker", "recogniser").unwrap();
        assert_eq!(b.persona(), "Samuel Adams");
        assert!(b.voice().portrayal().label().contains("no recording"));
    }

    #[test]
    fn a_part_attached_to_something_it_was_not_made_for_is_refused() {
        assert!(matches!(
            bundle("another-base", "thinker", "recogniser"),
            Err(BundleError::AdapterBase { .. })
        ));
        assert!(matches!(
            bundle("thinker", "another-thinker", "recogniser"),
            Err(BundleError::ProjectorThinker { .. })
        ));
        assert!(matches!(
            bundle("thinker", "thinker", "another-recogniser"),
            Err(BundleError::ProjectorRecogniser { .. })
        ));
    }

    #[test]
    fn another_persona_reuses_everything_but_its_adapter() {
        let adams = bundle("thinker", "thinker", "recogniser").unwrap();
        let jefferson = adams
            .with_persona("Thomas Jefferson", made_for("jefferson-adapter", "thinker"))
            .unwrap();
        assert_eq!(jefferson.persona(), "Thomas Jefferson");
        assert_eq!(
            jefferson.projector, adams.projector,
            "the projector was not made for a persona"
        );
        assert_ne!(jefferson.digest().unwrap(), adams.digest().unwrap());
        // An adapter of another base cannot be swapped in.
        assert!(adams
            .with_persona(
                "Thomas Jefferson",
                made_for("jefferson-adapter", "another-base")
            )
            .is_err());
    }

    #[test]
    fn a_bundle_has_a_name_and_a_stable_address() {
        let unnamed = SpeechBundle::new(
            "  ",
            part("t"),
            None,
            made_for("p", "t"),
            &d("r"),
            part("r"),
            part("s"),
            SpeakerProfile::new(1, Portrayal::synthetic_theatrical()),
        );
        assert_eq!(unnamed, Err(BundleError::Unnamed));
        let a = bundle("thinker", "thinker", "recogniser").unwrap();
        let back: SpeechBundle = serde_json::from_str(&serde_json::to_string(&a).unwrap()).unwrap();
        assert_eq!(back.digest().unwrap(), a.digest().unwrap());
    }
}
