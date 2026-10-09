// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements voice agents that learn how to speak from
// what their users tell them, for its clients. If your team needs expertise
// in turning spoken feedback into training data, you can procure our services
// by sending an email to info@swedishembedded.com.

//! What a user tells a speaking persona about how to speak, kept as a lesson.
//!
//! A user says "Talk as follows: Jeff-er-son" or "Pronounce Jefferson as
//! Jeff-er-son". The utterance is split into the directive and its example, and
//! the example becomes a [`Lesson`] with a stated [`Objective`]: what the
//! system is meant to learn from it, and by which mechanism. One conversation
//! can yield lessons for several objectives, because the same correction
//! serves a lexicon today, an in-context demonstration now, and a training
//! example later.
//!
//! A lesson names the recording it came from by digest and never holds audio,
//! and carries the [`Portrayal`] of the voice it applies to: a lesson never
//! teaches a synthetic voice to pass as a real person.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::digest::Digest;
use crate::speech::Portrayal;

/// What the user asked for, read from what they said.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Directive {
    /// "Talk as follows: <example>": say the example the way it is said here.
    Imitate {
        /// The words of the example.
        example: String,
    },
    /// "Pronounce <word> as <spelling>": say the word as the spelling reads.
    Respell {
        /// The word as it is written.
        word: String,
        /// How it is to be said, spelled out.
        spoken_as: String,
    },
}

/// Openers of an imitation directive, lowercase, longest first.
const IMITATE_OPENERS: [&str; 5] = [
    "talk as follows",
    "speak as follows",
    "say it like this",
    "say this like this",
    "say it as follows",
];

/// Read a [`Directive`] from a transcript, or `None` when the user was not
/// giving one. Case and the punctuation around the opener are ignored; the
/// example keeps its own punctuation.
#[must_use]
pub fn parse_directive(transcript: &str) -> Option<Directive> {
    let text = transcript.trim();
    let lower = text.to_lowercase();
    for opener in IMITATE_OPENERS {
        if let Some(rest) = lower.strip_prefix(opener) {
            let example = text[text.len() - rest.len()..]
                .trim_start_matches([':', ',', '.', ';', '-', ' '])
                .trim();
            return (!example.is_empty()).then(|| Directive::Imitate {
                example: example.to_string(),
            });
        }
    }
    for opener in ["pronounce ", "say "] {
        if let Some(rest) = lower.strip_prefix(opener) {
            let rest_original = &text[text.len() - rest.len()..];
            if let Some(at) = rest.find(" as ") {
                let word = rest_original[..at].trim().trim_matches(['"', '\'']);
                let spoken_as = rest_original[at + 4..]
                    .trim()
                    .trim_matches(['"', '\''])
                    .trim_end_matches(['.', '!']);
                if !word.is_empty()
                    && !spoken_as.is_empty()
                    && !word.contains(' ')
                    && !spoken_as.contains(' ')
                {
                    return Some(Directive::Respell {
                        word: word.to_string(),
                        spoken_as: spoken_as.to_string(),
                    });
                }
            }
        }
    }
    None
}

/// What a lesson is meant to teach, and so how it is used.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Objective {
    /// A word is said as spelled out: applied before synthesis, nothing trained.
    Lexicon,
    /// The recording is a reference the voice imitates when speaking, in
    /// context: nothing trained.
    InContext,
    /// Text and recording form a supervised example for the synthesizer.
    Supervised,
    /// The recording is preferred over the synthesizer's own take: a
    /// preference pair for reinforcement from feedback.
    Preference,
}

/// The material a lesson teaches from.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Exemplar {
    /// A word and the spelling it is to be said as.
    Respelling {
        /// The word as written.
        word: String,
        /// How it is to be said.
        spoken_as: String,
    },
    /// A recording of the example, by the digest of its audio.
    Recording {
        /// The audio's digest.
        audio: Digest,
        /// The words said in it.
        words: String,
    },
}

/// One thing a user taught, with where it came from and what it is for.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Lesson {
    /// What the user said, as heard: provenance, never rewritten.
    pub heard: String,
    /// What to learn from.
    pub exemplar: Exemplar,
    /// What it is for.
    pub objective: Objective,
    /// The voice it applies to.
    pub portrayal: Portrayal,
}

/// Why a directive could not become a lesson.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum LessonError {
    /// An imitation was asked for but the user's example was not recorded.
    #[error("the example to imitate was not recorded, so only its words are known")]
    NoRecording,
}

/// The lessons a [`Directive`] makes. A respelling is a lexicon entry; an
/// imitation needs the recording of the example and yields an in-context
/// reference, a supervised example and a preference pair's preferred side.
///
/// # Errors
/// An imitation without `recording` (the digest of the example's audio).
pub fn lessons_from(
    heard: &str,
    directive: &Directive,
    recording: Option<&Digest>,
    portrayal: &Portrayal,
) -> Result<Vec<Lesson>, LessonError> {
    let lesson = |exemplar: Exemplar, objective| Lesson {
        heard: heard.to_string(),
        exemplar,
        objective,
        portrayal: portrayal.clone(),
    };
    match directive {
        Directive::Respell { word, spoken_as } => Ok(vec![lesson(
            Exemplar::Respelling {
                word: word.clone(),
                spoken_as: spoken_as.clone(),
            },
            Objective::Lexicon,
        )]),
        Directive::Imitate { example } => {
            let audio = recording.ok_or(LessonError::NoRecording)?;
            let exemplar = || Exemplar::Recording {
                audio: audio.clone(),
                words: example.clone(),
            };
            Ok([
                Objective::InContext,
                Objective::Supervised,
                Objective::Preference,
            ]
            .into_iter()
            .map(|objective| lesson(exemplar(), objective))
            .collect())
        }
    }
}

/// How words are to be said, learned from respelling lessons.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Lexicon {
    entries: BTreeMap<String, String>,
}

impl Lexicon {
    /// A lexicon with nothing in it.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The lexicon the lexicon lessons among `lessons` make; a later lesson
    /// for the same word replaces an earlier one.
    #[must_use]
    pub fn from_lessons<'a>(lessons: impl IntoIterator<Item = &'a Lesson>) -> Self {
        let mut lexicon = Self::new();
        for lesson in lessons {
            if let (Objective::Lexicon, Exemplar::Respelling { word, spoken_as }) =
                (lesson.objective, &lesson.exemplar)
            {
                lexicon.teach(word, spoken_as);
            }
        }
        lexicon
    }

    /// Say `word` as `spoken_as` from now on.
    pub fn teach(&mut self, word: &str, spoken_as: &str) {
        self.entries
            .insert(word.to_lowercase(), spoken_as.to_string());
    }

    /// How many words are taught.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether nothing is taught.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// `text` with every taught word replaced by how it is said. Whole words
    /// only, whatever their case, with the punctuation around them kept.
    #[must_use]
    pub fn apply(&self, text: &str) -> String {
        if self.is_empty() {
            return text.to_string();
        }
        let mut out = String::with_capacity(text.len());
        let mut word = String::new();
        let flush = |word: &mut String, out: &mut String| {
            match self.entries.get(&word.to_lowercase()) {
                Some(said) if !word.is_empty() => out.push_str(said),
                _ => out.push_str(word),
            }
            word.clear();
        };
        for ch in text.chars() {
            if ch.is_alphanumeric() || ch == '\'' {
                word.push(ch);
            } else {
                flush(&mut word, &mut out);
                out.push(ch);
            }
        }
        flush(&mut word, &mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn portrayal() -> Portrayal {
        Portrayal::synthetic_theatrical()
    }

    fn audio() -> Digest {
        Digest::of(b"the user's recording")
    }

    #[test]
    fn an_imitation_directive_is_split_from_its_example() {
        assert_eq!(
            parse_directive("Talk as follows: Thomas Jef-fer-son, of Monticello."),
            Some(Directive::Imitate {
                example: "Thomas Jef-fer-son, of Monticello.".into()
            })
        );
        assert_eq!(
            parse_directive("  talk as follows - well, I never"),
            Some(Directive::Imitate {
                example: "well, I never".into()
            })
        );
    }

    #[test]
    fn a_respelling_directive_names_the_word_and_how_it_is_said() {
        assert_eq!(
            parse_directive("Pronounce Jefferson as Jeff-er-son."),
            Some(Directive::Respell {
                word: "Jefferson".into(),
                spoken_as: "Jeff-er-son".into()
            })
        );
        assert_eq!(
            parse_directive("say \"Adams\" as Add-ums"),
            Some(Directive::Respell {
                word: "Adams".into(),
                spoken_as: "Add-ums".into()
            })
        );
    }

    #[test]
    fn what_is_not_a_directive_is_not_read_as_one() {
        for said in [
            "What do you think of taxation?",
            "Talk as follows",
            "Please say it as you see fit.",
            "say it as you see fit.",
            "pronounce the new tax as a burden",
            "",
        ] {
            assert_eq!(parse_directive(said), None, "{said:?}");
        }
    }

    #[test]
    fn an_imitation_yields_a_lesson_for_every_use_of_the_recording_and_needs_the_recording() {
        let directive = parse_directive("Talk as follows: Good morning.").unwrap();
        let lessons = lessons_from(
            "Talk as follows: Good morning.",
            &directive,
            Some(&audio()),
            &portrayal(),
        )
        .unwrap();
        let objectives: Vec<_> = lessons.iter().map(|l| l.objective).collect();
        assert_eq!(
            objectives,
            [
                Objective::InContext,
                Objective::Supervised,
                Objective::Preference
            ]
        );
        assert!(lessons.iter().all(|l| l.portrayal == portrayal()));
        assert!(matches!(
            &lessons[0].exemplar,
            Exemplar::Recording { words, .. } if words == "Good morning."
        ));
        assert_eq!(
            lessons_from("x", &directive, None, &portrayal()),
            Err(LessonError::NoRecording)
        );
    }

    #[test]
    fn a_lexicon_says_taught_words_as_taught_and_leaves_the_rest_alone() {
        let directive = parse_directive("Pronounce Jefferson as Jeff-er-son").unwrap();
        let lessons = lessons_from("heard", &directive, None, &portrayal()).unwrap();
        let lexicon = Lexicon::from_lessons(&lessons);
        assert_eq!(lexicon.len(), 1);
        assert_eq!(
            lexicon.apply("Jefferson, and jefferson's friend Jeffersonian said \"Jefferson!\""),
            "Jeff-er-son, and jefferson's friend Jeffersonian said \"Jeff-er-son!\""
        );
        assert_eq!(Lexicon::new().apply("unchanged"), "unchanged");
    }

    #[test]
    fn a_later_lesson_for_a_word_replaces_the_earlier_one() {
        let mut lexicon = Lexicon::new();
        lexicon.teach("Adams", "Add-ums");
        lexicon.teach("adams", "Ad-ems");
        assert_eq!(lexicon.apply("Adams"), "Ad-ems");
        assert_eq!(lexicon.len(), 1);
    }

    #[test]
    fn a_lesson_round_trips_through_json_with_its_portrayal() {
        let directive = parse_directive("Talk as follows: Hello.").unwrap();
        let lessons = lessons_from("h", &directive, Some(&audio()), &portrayal()).unwrap();
        let json = serde_json::to_string(&lessons[0]).unwrap();
        assert!(json.contains("Synthetic theatrical portrayal"), "{json}");
        assert_eq!(serde_json::from_str::<Lesson>(&json).unwrap(), lessons[0]);
    }
}
