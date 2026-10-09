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

/// Whether a transcript confirms (`Some(true)`) or refuses (`Some(false)`)
/// what the agent just offered to learn, or says neither (`None`). Only a
/// short, plain answer counts: a long sentence that happens to contain "yes"
/// is not a confirmation, because a lesson is kept only when the user means it.
#[must_use]
pub fn parse_confirmation(transcript: &str) -> Option<bool> {
    let words = plain_words(transcript);
    let said = words
        .join(" ")
        .replace("that's", "that is")
        .replace("don't", "do not");
    const YES: [&str; 8] = [
        "yes",
        "yes please",
        "yeah",
        "yep",
        "keep it",
        "keep that",
        "that is right",
        "that is correct",
    ];
    const NO: [&str; 7] = [
        "no",
        "no thanks",
        "nope",
        "do not keep it",
        "do not keep that",
        "that is wrong",
        "that is not right",
    ];
    if YES.contains(&said.as_str()) {
        Some(true)
    } else if NO.contains(&said.as_str()) {
        Some(false)
    } else {
        None
    }
}

/// Whether the user asks to take back the last thing taught ("forget that",
/// "undo that", "never mind that").
#[must_use]
pub fn parse_retraction(transcript: &str) -> bool {
    matches!(
        plain_words(transcript).join(" ").as_str(),
        "forget that"
            | "forget it"
            | "undo that"
            | "undo it"
            | "never mind that"
            | "take that back"
    )
}

/// What a user says about the last thing the agent said aloud.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Feedback {
    /// It did not sound right: say it again.
    Reject,
    /// It sounded right.
    Approve,
}

/// Read feedback on the last utterance from a transcript, or `None`. Only a
/// plain remark counts, for the same reason a confirmation must be plain.
#[must_use]
pub fn parse_feedback(transcript: &str) -> Option<Feedback> {
    let said = plain_words(transcript)
        .join(" ")
        .replace("that's", "that is");
    const REJECT: [&str; 12] = [
        "that sounded wrong",
        "that sounded odd",
        "that sounded off",
        "that sounded bad",
        "that did not sound right",
        "that was wrong",
        "say that again",
        "try that again",
        "say it again",
        "that was not right",
        "that is not right",
        "that is wrong",
    ];
    const APPROVE: [&str; 6] = [
        "that sounded good",
        "that sounded right",
        "that was good",
        "that was perfect",
        "perfect",
        "that is right",
    ];
    if REJECT.contains(&said.as_str()) {
        Some(Feedback::Reject)
    } else if APPROVE.contains(&said.as_str()) {
        Some(Feedback::Approve)
    } else {
        None
    }
}

/// Which of two takes of the same words the user prefers, asked after the
/// agent said them again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Comparison {
    /// The second take (the new one) is better.
    NewBetter,
    /// The first take (the old one) was better.
    OldBetter,
}

/// Read an answer to "was that better?", or `None`.
#[must_use]
pub fn parse_comparison(transcript: &str) -> Option<Comparison> {
    let said = plain_words(transcript)
        .join(" ")
        .replace("that's", "that is");
    const NEW: [&str; 7] = [
        "yes",
        "better",
        "that is better",
        "that sounded better",
        "much better",
        "the second one",
        "yes that is better",
    ];
    const OLD: [&str; 7] = [
        "no",
        "worse",
        "that is worse",
        "the first one",
        "the first one was better",
        "go back",
        "no that is worse",
    ];
    if NEW.contains(&said.as_str()) {
        Some(Comparison::NewBetter)
    } else if OLD.contains(&said.as_str()) {
        Some(Comparison::OldBetter)
    } else {
        None
    }
}

/// `text` lower-cased as its words, punctuation dropped.
fn plain_words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect()
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
    /// Two takes of the same text, one preferred by the user.
    Preferred {
        /// The words both takes say.
        text: String,
        /// The take the user preferred, by the digest of its audio.
        chosen: Digest,
        /// The take the user turned down.
        rejected: Digest,
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
    fn only_a_plain_answer_confirms_or_refuses_a_lesson() {
        for yes in ["Yes.", "yes please", "Yeah!", "Keep it.", "That's right"] {
            assert_eq!(parse_confirmation(yes), Some(true), "{yes:?}");
        }
        for no in ["No.", "nope", "Do not keep that.", "That is wrong"] {
            assert_eq!(parse_confirmation(no), Some(false), "{no:?}");
        }
        for other in [
            "Yes, and also tell me about the tax",
            "I said no such thing",
            "What do you think of Jefferson?",
            "",
        ] {
            assert_eq!(parse_confirmation(other), None, "{other:?}");
        }
    }

    #[test]
    fn taking_back_the_last_lesson_is_a_plain_request() {
        for said in [
            "Forget that.",
            "undo that",
            "Never mind that!",
            "take that back",
        ] {
            assert!(parse_retraction(said), "{said:?}");
        }
        for said in ["Forget the tax", "do not forget that", "undo"] {
            assert!(!parse_retraction(said), "{said:?}");
        }
    }

    #[test]
    fn a_plain_remark_on_the_last_utterance_is_feedback_and_a_sentence_about_something_else_is_not()
    {
        for said in [
            "That sounded wrong.",
            "Say that again!",
            "That's not right",
            "try that again",
        ] {
            assert_eq!(parse_feedback(said), Some(Feedback::Reject), "{said:?}");
        }
        for said in ["That sounded good.", "Perfect.", "That's right"] {
            assert_eq!(parse_feedback(said), Some(Feedback::Approve), "{said:?}");
        }
        for said in [
            "Tell me why that sounded wrong to the delegates",
            "perfectly",
            "That was the wrong tax",
            "",
        ] {
            assert_eq!(parse_feedback(said), None, "{said:?}");
        }
    }

    #[test]
    fn an_answer_to_was_that_better_names_one_of_the_two_takes() {
        for said in ["Yes.", "Better!", "That's better", "the second one"] {
            assert_eq!(
                parse_comparison(said),
                Some(Comparison::NewBetter),
                "{said:?}"
            );
        }
        for said in ["No.", "Worse", "go back", "The first one was better."] {
            assert_eq!(
                parse_comparison(said),
                Some(Comparison::OldBetter),
                "{said:?}"
            );
        }
        for said in ["Maybe", "I liked the tax part", ""] {
            assert_eq!(parse_comparison(said), None, "{said:?}");
        }
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
