// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The batch commands: a directory of recorded questions answered, by the
//! cascade (`turns`) or by a model that listens (`listen-turns`).

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use splinter_sdk::model::answer::Answerer;
use splinter_sdk::model::error::PolicyError;
use splinter_sdk::model::speech::{
    corpus_word_error_rate, take_turn, BrainRecognizer, BrainSynthesizer, Clip, ListenerOptions,
    Recognizer, Sentences,
};
use splinter_sdk::vocabulary::prompt::persona_prompt;
use splinter_sdk::vocabulary::speech::SpeakerProfile;

use crate::{
    pair_questions, read_sentences, ListenItem, ListenReport, Spread, TurnItem, TurnsReport,
};

/// The most words one piece of an answer has when it is spoken: a synthesizer
/// renders a bounded stretch of speech and stops.
pub const SPOKEN_PIECE_WORDS: usize = 30;

/// The longest answer a turn generates, in tokens: a spoken answer is short.
pub const ANSWER_TOKENS: u32 = 400;

/// The models a batch speaks and listens with.
pub struct Names<'a> {
    /// The recognition model.
    pub asr: &'a str,
    /// The synthesis model.
    pub tts: &'a str,
    /// The persona's voice.
    pub speaker: SpeakerProfile,
}

/// The persona as a text model: a base, maybe an adapter, and the system turn
/// that makes it answer as the person.
pub struct Persona {
    /// The person.
    pub name: String,
    system: String,
    answerer: Answerer,
    runtime: tokio::runtime::Runtime,
    base: PathBuf,
    adapter: Option<PathBuf>,
}

impl Persona {
    /// Load the model (with its adapter, if any) the persona answers with.
    pub fn load(
        name: &str,
        base: &std::path::Path,
        adapter: Option<&std::path::Path>,
    ) -> Result<Self> {
        Ok(Self {
            name: name.to_string(),
            system: persona_prompt(name),
            answerer: Answerer::load(base, adapter, None, name)?,
            runtime: tokio::runtime::Runtime::new()?,
            base: base.to_path_buf(),
            adapter: adapter.map(std::path::Path::to_path_buf),
        })
    }

    /// The same persona answering under another system turn.
    #[must_use]
    pub fn under(mut self, system: String) -> Self {
        self.system = system;
        self
    }

    /// The persona's answer to a question, as text.
    pub fn answer(&self, question: &str) -> Result<String, PolicyError> {
        self.answer_streaming(question, &mut |_| {})
    }

    /// [`Self::answer`], handing `on_text` each piece as it is written.
    pub fn answer_streaming(
        &self,
        question: &str,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<String, PolicyError> {
        self.runtime
            .block_on(
                self.answerer
                    .ask_streaming(&self.system, question, ANSWER_TOKENS, on_text),
            )
            .map(|reply| reply.text.trim().to_string())
            .map_err(|e| PolicyError::Generate {
                path: self.base.clone(),
                adapter: self.adapter.clone(),
                reason: e.to_string(),
            })
    }
}

pub fn read_clip(path: &std::path::Path) -> Result<Clip> {
    Clip::from_wav(&std::fs::read(path).with_context(|| format!("reading {}", path.display()))?)
        .with_context(|| format!("{} is not a WAV file", path.display()))
}

/// Where a batch of questions comes from and its answers go.
pub struct Batch<'a> {
    /// The directory of recordings.
    pub recordings: &'a std::path::Path,
    /// The questions they were made from, one per line.
    pub questions: Option<&'a std::path::Path>,
    /// Where each spoken answer is written.
    pub out_dir: &'a std::path::Path,
}

pub fn turns(
    names: &Names<'_>,
    batch: &Batch,
    persona: &Persona,
    keep: Option<&std::path::Path>,
) -> Result<()> {
    let mut recordings: Vec<PathBuf> = std::fs::read_dir(batch.recordings)
        .with_context(|| format!("reading {}", batch.recordings.display()))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "wav"))
        .collect();
    recordings.sort();
    if recordings.is_empty() {
        bail!("{} holds no WAV recordings", batch.recordings.display());
    }
    let asked = batch
        .questions
        .map(|path| std::fs::read_to_string(path).map(|t| read_sentences(&t)))
        .transpose()?;
    let paired = pair_questions(&recordings, asked.as_deref()).map_err(anyhow::Error::msg)?;
    std::fs::create_dir_all(batch.out_dir)?;

    let speaker = names.speaker.clone();
    let recognizer = BrainRecognizer::load(names.asr)?;
    let synthesizer = Sentences::new(BrainSynthesizer::load(names.tts)?, SPOKEN_PIECE_WORDS);
    let mut items = Vec::new();
    for (recording, question) in paired {
        let name = recording.display().to_string();
        let turn = take_turn(
            &recognizer,
            &mut |q: &str| persona.answer(q),
            &synthesizer,
            &speaker,
            &read_clip(&recording)?,
        )
        .with_context(|| format!("turn on {name}"))?;
        let stem = recording
            .file_stem()
            .map_or_else(String::new, |s| s.to_string_lossy().into_owned());
        turn.reply
            .save(batch.out_dir.join(format!("{stem}.reply.wav")))?;
        let answer_heard = recognizer.transcribe(&turn.reply)?.text;
        items.push(TurnItem {
            recording: name,
            asked: question,
            heard: turn.heard,
            answer: turn.answer,
            answer_heard,
            reply_seconds: turn.reply.seconds(),
            recognise_seconds: turn.timings.recognise.as_secs_f64(),
            respond_seconds: turn.timings.respond.as_secs_f64(),
            synthesise_seconds: turn.timings.synthesise.as_secs_f64(),
        });
    }

    let column = |f: fn(&TurnItem) -> f64| Spread::of(&items.iter().map(f).collect::<Vec<_>>());
    let report = TurnsReport {
        persona: &persona.name,
        speaker: &speaker,
        turns: items.len(),
        question_word_error_rate: corpus_word_error_rate(
            items
                .iter()
                .filter_map(|i| i.asked.as_deref().map(|a| (a, i.heard.as_str()))),
        ),
        answer_word_error_rate: corpus_word_error_rate(
            items
                .iter()
                .map(|i| (i.answer.as_str(), i.answer_heard.as_str())),
        ),
        recognise_seconds: column(|i| i.recognise_seconds),
        respond_seconds: column(|i| i.respond_seconds),
        synthesise_seconds: column(|i| i.synthesise_seconds),
        total_seconds: column(|i| i.recognise_seconds + i.respond_seconds + i.synthesise_seconds),
        items: &items,
    };
    emit(&report, keep)
}

/// Answer every recording of `batch` with a model that listens: no transcript
/// between the question and the answer.
pub fn listen_turns(
    names: &Names<'_>,
    batch: &Batch<'_>,
    persona: &str,
    opts: &ListenerOptions,
    keep: Option<&std::path::Path>,
) -> Result<()> {
    let listener = splinter_sdk::model::speech::BrainListener::load(opts)?;
    spoken_turns(names, batch, persona, &listener, keep)
}

/// Answer every recording of `batch` with a recognizer in front of the text
/// model, speaking each sentence as the model writes it.
pub fn streamed_turns(
    names: &Names<'_>,
    batch: &Batch<'_>,
    persona: &Persona,
    keep: Option<&std::path::Path>,
) -> Result<()> {
    let listener = splinter_sdk::model::speech::CascadeListener::new(
        BrainRecognizer::load(names.asr)?,
        |question: &str, on_text: &mut dyn FnMut(&str)| persona.answer_streaming(question, on_text),
    );
    spoken_turns(names, batch, &persona.name, &listener, keep)
}

/// The turns of `batch`, taken by `listener` and spoken sentence by sentence.
fn spoken_turns(
    names: &Names<'_>,
    batch: &Batch<'_>,
    persona: &str,
    listener: &dyn splinter_sdk::model::speech::Listener,
    keep: Option<&std::path::Path>,
) -> Result<()> {
    let mut recordings: Vec<PathBuf> = std::fs::read_dir(batch.recordings)
        .with_context(|| format!("reading {}", batch.recordings.display()))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "wav"))
        .collect();
    recordings.sort();
    if recordings.is_empty() {
        bail!("{} holds no WAV recordings", batch.recordings.display());
    }
    let asked = batch
        .questions
        .map(|path| std::fs::read_to_string(path).map(|t| read_sentences(&t)))
        .transpose()?;
    let paired = pair_questions(&recordings, asked.as_deref()).map_err(anyhow::Error::msg)?;
    std::fs::create_dir_all(batch.out_dir)?;

    let speaker = names.speaker.clone();
    let judge = BrainRecognizer::load(names.asr)?;
    let synthesizer = BrainSynthesizer::load(names.tts)?;
    let mut items = Vec::new();
    for (recording, question) in paired {
        let name = recording.display().to_string();
        let turn = splinter_sdk::model::speech::take_spoken_turn(
            listener,
            &synthesizer,
            &speaker,
            &read_clip(&recording)?,
            SPOKEN_PIECE_WORDS,
        )
        .with_context(|| format!("turn on {name}"))?;
        let stem = recording
            .file_stem()
            .map_or_else(String::new, |s| s.to_string_lossy().into_owned());
        turn.reply
            .save(batch.out_dir.join(format!("{stem}.reply.wav")))?;
        let answer_heard = judge.transcribe(&turn.reply)?.text;
        items.push(ListenItem {
            recording: name,
            asked: question,
            answer: turn.answer,
            answer_heard,
            reply_seconds: turn.reply.seconds(),
            first_audio_seconds: turn.timings.first_audio.as_secs_f64(),
            think_seconds: turn.timings.think.as_secs_f64(),
            synthesise_seconds: turn.timings.synthesise.as_secs_f64(),
            total_seconds: turn.timings.total.as_secs_f64(),
        });
    }
    let column = |f: fn(&ListenItem) -> f64| Spread::of(&items.iter().map(f).collect::<Vec<_>>());
    let report = ListenReport {
        persona,
        speaker: &speaker,
        turns: items.len(),
        answer_word_error_rate: corpus_word_error_rate(
            items
                .iter()
                .map(|i| (i.answer.as_str(), i.answer_heard.as_str())),
        ),
        first_audio_seconds: column(|i| i.first_audio_seconds),
        think_seconds: column(|i| i.think_seconds),
        synthesise_seconds: column(|i| i.synthesise_seconds),
        total_seconds: column(|i| i.total_seconds),
        items: &items,
    };
    emit(&report, keep)
}

/// Print `report` as JSON, and keep a copy where asked.
pub fn emit(report: &impl serde::Serialize, keep: Option<&std::path::Path>) -> Result<()> {
    let json = serde_json::to_string_pretty(report)?;
    if let Some(path) = keep {
        std::fs::write(path, &json).with_context(|| format!("writing {}", path.display()))?;
    }
    println!("{json}");
    Ok(())
}
