// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Recording questions in several voices into a [`SpokenSet`].
//!
//! Each question is spoken once per voice (seed) and kept only if a recogniser
//! that did not make it hears it back within the bound; the recordings are
//! files beside the set, each named by the content address of its bytes in the
//! set, so the set can be pinned and checked later.

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use splinter_sdk::model::speech::{Recognizer, Synthesizer};
use splinter_sdk::vocabulary::digest::Digest;
use splinter_sdk::vocabulary::speech::{Portrayal, SpeakerProfile};
use splinter_sdk::vocabulary::spoken::{Spoken, SpokenSet};

/// One line of a questions file.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Question {
    /// The question's name; also the text record the recordings represent.
    pub id: String,
    /// What splits keep whole; the id when the line names none.
    #[serde(default)]
    pub group: Option<String>,
    /// The words to speak.
    pub text: String,
}

/// The questions in a JSON-lines file: one object per line with `id`, `text`
/// and optionally `group`. Blank lines are skipped.
pub fn read_questions(jsonl: &str) -> Result<Vec<Question>> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for (n, line) in jsonl
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
    {
        let q: Question = serde_json::from_str(line)
            .with_context(|| format!("line {}: not a question", n + 1))?;
        if q.text.split_whitespace().next().is_none() {
            bail!("line {}: question {:?} has no words", n + 1, q.id);
        }
        if !seen.insert(q.id.clone()) {
            bail!("line {}: two questions are named {:?}", n + 1, q.id);
        }
        out.push(q);
    }
    Ok(out)
}

/// Every `every`-th distinct group name, in name order: the groups held out
/// for testing. `every` of zero holds nothing out.
pub fn held_out_group_names<'a>(
    groups: impl IntoIterator<Item = &'a str>,
    every: usize,
) -> BTreeSet<String> {
    let distinct: BTreeSet<&str> = groups.into_iter().collect();
    distinct
        .into_iter()
        .enumerate()
        .filter(|(i, _)| every > 0 && i % every == every - 1)
        .map(|(_, g)| g.to_string())
        .collect()
}

/// A recording that no recogniser heard back well enough.
#[derive(Debug, PartialEq, Serialize)]
pub struct Rejected {
    /// The question.
    pub id: String,
    /// The voice that was tried.
    pub seed: u64,
    /// What was heard.
    pub heard: String,
}

/// Record every question in every voice of `seeds` into `dir`, keeping a
/// recording only if it is heard back with a word error rate of at most
/// `max_wer`.
pub fn record_set(
    name: &str,
    questions: &[Question],
    seeds: &[u64],
    max_wer: f32,
    voices: (&dyn Synthesizer, &dyn Recognizer),
    dir: &Path,
) -> Result<(SpokenSet, Vec<Rejected>)> {
    let (synthesizer, recognizer) = voices;
    std::fs::create_dir_all(dir)?;
    let (mut kept, mut rejected) = (Vec::new(), Vec::new());
    for q in questions {
        for &seed in seeds {
            let speaker = SpeakerProfile::new(seed, Portrayal::synthetic_theatrical());
            let clip = synthesizer.speak(&q.text, &speaker)?;
            let heard = recognizer.transcribe(&clip)?.text;
            let rate = splinter_sdk::model::speech::corpus_word_error_rate([(
                q.text.as_str(),
                heard.as_str(),
            )])
            .unwrap_or(1.0);
            if rate > max_wer {
                rejected.push(Rejected {
                    id: q.id.clone(),
                    seed,
                    heard,
                });
                continue;
            }
            let id = format!("{}-v{seed}", q.id);
            let file = format!("{id}.wav");
            clip.save(dir.join(&file))
                .with_context(|| format!("writing {file}"))?;
            let audio = Digest::of(&std::fs::read(dir.join(&file))?);
            kept.push(Spoken {
                id,
                represents: q.id.clone(),
                group: q.group.clone().unwrap_or_else(|| q.id.clone()),
                text: q.text.clone(),
                file,
                audio,
                seed,
                word_error_rate: rate,
            });
        }
    }
    let set = SpokenSet::new(name, Portrayal::synthetic_theatrical(), kept)?;
    Ok((set, rejected))
}

#[cfg(test)]
mod tests {
    use super::*;
    use splinter_sdk::model::error::PolicyError;
    use splinter_sdk::model::speech::{Clip, Transcription};

    /// Speaks a sentence as one sample per byte; seed 13 mumbles.
    struct Voice;
    impl Synthesizer for Voice {
        fn speak(&self, text: &str, speaker: &SpeakerProfile) -> Result<Clip, PolicyError> {
            let spoken = if speaker.seed() == 13 { "" } else { text };
            Ok(Clip::new(
                spoken.bytes().map(|b| f32::from(b) / 255.0).collect(),
                16_000,
            ))
        }
    }
    struct Ear;
    impl Recognizer for Ear {
        fn transcribe(&self, clip: &Clip) -> Result<Transcription, PolicyError> {
            let bytes: Vec<u8> = clip
                .samples()
                .iter()
                .map(|s| (s * 255.0).round() as u8)
                .collect();
            Ok(Transcription {
                text: String::from_utf8_lossy(&bytes).into_owned(),
                truncated: false,
            })
        }
    }

    fn questions() -> Vec<Question> {
        read_questions("{\"id\":\"q1\",\"text\":\"What is liberty\"}\n\n{\"id\":\"q2\",\"group\":\"g\",\"text\":\"Who pays the tax\"}\n").unwrap()
    }

    #[test]
    fn a_questions_file_names_each_question_once_and_gives_it_words() {
        assert_eq!(questions().len(), 2);
        assert_eq!(questions()[0].group, None);
        assert!(
            read_questions("{\"id\":\"a\",\"text\":\"x\"}\n{\"id\":\"a\",\"text\":\"y\"}").is_err()
        );
        assert!(read_questions("{\"id\":\"a\",\"text\":\"  \"}").is_err());
        assert!(read_questions("not json").is_err());
    }

    #[test]
    fn held_out_groups_are_every_kth_distinct_name() {
        let names = ["b", "a", "c", "a", "d", "e", "f"];
        let held = held_out_group_names(names, 3);
        assert_eq!(
            held.iter().map(String::as_str).collect::<Vec<_>>(),
            ["c", "f"]
        );
        assert!(held_out_group_names(names, 0).is_empty());
    }

    #[test]
    fn every_voice_that_is_heard_back_is_recorded_and_each_file_matches_its_address() {
        let dir = tempfile::tempdir().unwrap();
        let (set, rejected) = record_set(
            "sp",
            &questions(),
            &[1, 2, 13],
            0.2,
            (&Voice, &Ear),
            dir.path(),
        )
        .unwrap();

        assert_eq!(
            set.items().len(),
            4,
            "two questions in two voices that are heard back"
        );
        assert_eq!(
            rejected.len(),
            2,
            "the mumbling voice is rejected for both questions"
        );
        assert!(rejected.iter().all(|r| r.seed == 13 && r.heard.is_empty()));
        for item in set.items() {
            let bytes = std::fs::read(dir.path().join(&item.file)).unwrap();
            assert_eq!(item.audio, Digest::of(&bytes), "{}", item.id);
        }
        let by_group: Vec<_> = set.items().iter().map(|i| i.group.as_str()).collect();
        assert_eq!(
            by_group,
            ["q1", "q1", "g", "g"],
            "a question is its own group unless it names one"
        );
    }
}
