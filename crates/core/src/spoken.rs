// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements spoken-dialogue training data whose every
// recording is traceable to the text it renders and kept apart from the
// recordings it is tested on, for its clients. If your team needs expertise
// in speech datasets without leakage, you can procure our services by sending
// an email to info@swedishembedded.com.

//! Speech recorded from text: what a recording is a rendering of, and how a
//! set of them is split so a model is tested on voices and questions it never
//! trained on.
//!
//! A recording is another representation of a text record, never a record of
//! its own: it names what it represents and the digest of its audio, so a
//! better tokenizer or recogniser can be run over the same recordings, and the
//! text stays the one source of truth.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::digest::{canonical_json, Digest};
use crate::speech::Portrayal;

/// One recording of a text.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Spoken {
    /// This recording's name within its set.
    pub id: String,
    /// The text record this renders (an experience or a question's id).
    pub represents: String,
    /// The group splits keep whole: every recording of one question has the
    /// same group.
    pub group: String,
    /// The words spoken.
    pub text: String,
    /// The audio file, relative to the set's directory.
    pub file: String,
    /// The content address of the audio file's bytes.
    pub audio: Digest,
    /// The seed the voice was rendered from.
    pub seed: u64,
    /// Words wrong over words spoken when a recogniser that did not make the
    /// recording heard it back.
    pub word_error_rate: f32,
}

/// Why a set of recordings is not a set.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum SpokenSetError {
    /// Two recordings share a name.
    #[error("two recordings are named {0:?}")]
    DuplicateId(String),
    /// A recording names nothing it represents or no group.
    #[error("recording {0:?} names no text it represents, or no group")]
    Unattached(String),
    /// A recording's word error rate is not a number.
    #[error("recording {0:?} has a word error rate that is not a finite number")]
    BadRate(String),
}

/// A named set of recordings and the portrayal every voice in it declares.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpokenSet {
    name: String,
    portrayal: Portrayal,
    items: Vec<Spoken>,
}

impl SpokenSet {
    /// A set of `items`, each attached to a text and a group, uniquely named.
    pub fn new(
        name: impl Into<String>,
        portrayal: Portrayal,
        items: Vec<Spoken>,
    ) -> Result<Self, SpokenSetError> {
        let mut seen = BTreeSet::new();
        for item in &items {
            if item.represents.trim().is_empty() || item.group.trim().is_empty() {
                return Err(SpokenSetError::Unattached(item.id.clone()));
            }
            if !item.word_error_rate.is_finite() {
                return Err(SpokenSetError::BadRate(item.id.clone()));
            }
            if !seen.insert(item.id.as_str()) {
                return Err(SpokenSetError::DuplicateId(item.id.clone()));
            }
        }
        Ok(Self {
            name: name.into(),
            portrayal,
            items,
        })
    }

    /// The set's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// What every voice in the set declares itself to be.
    #[must_use]
    pub fn portrayal(&self) -> &Portrayal {
        &self.portrayal
    }

    /// The recordings, in the order they were given.
    #[must_use]
    pub fn items(&self) -> &[Spoken] {
        &self.items
    }

    /// The content address of the set: its name, portrayal and every
    /// recording with the address of its audio. Equal sets have equal
    /// addresses however they were written.
    pub fn digest(&self) -> Result<Digest, serde_json::Error> {
        Ok(Digest::of(&canonical_json(self)?))
    }

    /// Split into the recordings to train on and the recordings to test on.
    /// A test recording is in a held-out group AND spoken in a held-out voice;
    /// a training recording is in neither. A recording that is only half
    /// held out is in neither side, because either the model has heard its
    /// words or it has heard its voice.
    #[must_use]
    pub fn split(
        &self,
        held_out_groups: &BTreeSet<String>,
        held_out_seeds: &BTreeSet<u64>,
    ) -> Split<'_> {
        let mut split = Split {
            train: Vec::new(),
            test: Vec::new(),
            dropped: 0,
        };
        for item in &self.items {
            match (
                held_out_groups.contains(&item.group),
                held_out_seeds.contains(&item.seed),
            ) {
                (true, true) => split.test.push(item),
                (false, false) => split.train.push(item),
                _ => split.dropped += 1,
            }
        }
        split
    }
}

/// The two sides of [`SpokenSet::split`].
#[derive(Debug)]
pub struct Split<'a> {
    /// Recordings to train on.
    pub train: Vec<&'a Spoken>,
    /// Recordings to test on.
    pub test: Vec<&'a Spoken>,
    /// Recordings held out on one count only, kept out of both.
    pub dropped: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clip(id: &str, group: &str, seed: u64) -> Spoken {
        Spoken {
            id: id.into(),
            represents: format!("question-{group}"),
            group: group.into(),
            text: "What is liberty?".into(),
            file: format!("{id}.wav"),
            audio: Digest::of(id.as_bytes()),
            seed,
            word_error_rate: 0.0,
        }
    }

    fn set(items: Vec<Spoken>) -> SpokenSet {
        SpokenSet::new("sp1", Portrayal::synthetic_theatrical(), items).unwrap()
    }

    #[test]
    fn a_recording_must_say_what_it_represents_and_be_named_once() {
        let mut orphan = clip("a", "q1", 1);
        orphan.represents = " ".into();
        assert_eq!(
            SpokenSet::new("s", Portrayal::synthetic_theatrical(), vec![orphan]),
            Err(SpokenSetError::Unattached("a".into()))
        );
        let twice = vec![clip("a", "q1", 1), clip("a", "q2", 1)];
        assert_eq!(
            SpokenSet::new("s", Portrayal::synthetic_theatrical(), twice),
            Err(SpokenSetError::DuplicateId("a".into()))
        );
        let mut nan = clip("b", "q1", 1);
        nan.word_error_rate = f32::NAN;
        assert_eq!(
            SpokenSet::new("s", Portrayal::synthetic_theatrical(), vec![nan]),
            Err(SpokenSetError::BadRate("b".into()))
        );
    }

    #[test]
    fn a_split_tests_only_on_held_out_voices_speaking_held_out_questions() {
        let items = vec![
            clip("seen-seen", "q1", 1),
            clip("seen-voice-new", "q1", 9),
            clip("new-question-seen-voice", "q2", 1),
            clip("new-new", "q2", 9),
        ];
        let set = set(items);
        let groups: BTreeSet<String> = ["q2".to_string()].into();
        let seeds: BTreeSet<u64> = [9].into();
        let split = set.split(&groups, &seeds);

        let names = |side: &[&Spoken]| side.iter().map(|s| s.id.clone()).collect::<Vec<_>>();
        assert_eq!(names(&split.train), ["seen-seen"]);
        assert_eq!(names(&split.test), ["new-new"]);
        assert_eq!(
            split.dropped, 2,
            "half-held-out recordings are in neither side"
        );
        // No question and no voice on both sides.
        for t in &split.test {
            assert!(split
                .train
                .iter()
                .all(|r| r.group != t.group && r.seed != t.seed));
        }
    }

    #[test]
    fn a_digest_follows_the_content_and_not_the_construction() {
        let a = set(vec![clip("a", "q1", 1)]);
        assert_eq!(
            a.digest().unwrap(),
            set(vec![clip("a", "q1", 1)]).digest().unwrap()
        );
        let mut changed = clip("a", "q1", 1);
        changed.audio = Digest::of(b"another recording");
        assert_ne!(a.digest().unwrap(), set(vec![changed]).digest().unwrap());
        let json = serde_json::to_string(&a).unwrap();
        let back: SpokenSet = serde_json::from_str(&json).unwrap();
        assert_eq!(back.digest().unwrap(), a.digest().unwrap());
    }
}
