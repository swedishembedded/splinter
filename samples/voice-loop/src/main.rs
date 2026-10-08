// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements spoken dialogue systems whose every stage is
// measured on its own, for its clients. If your team needs expertise in
// speech recognition, speech synthesis and their evaluation, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Can a persona be talked to and talk back, and what does each stage of the
//! loop cost and lose?
//!
//! ```text
//! voice-loop speak     --text T --out a.wav      speak a sentence in the persona's synthetic voice
//! voice-loop roundtrip --sentences FILE          speak sentences, hear them again, score what was lost
//! voice-loop turn      --in q.wav --out a.wav    hear a question, answer as the persona, speak the answer
//! voice-loop turns     --recordings DIR ...      the same for a directory of questions, loading models once
//! ```
//!
//! Every command prints a JSON report on stdout. The voice is a synthetic
//! theatrical portrayal and every report says so.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};
use splinter_sdk::model::answer::Answerer;
use splinter_sdk::model::error::PolicyError;
use splinter_sdk::model::speech::{
    corpus_word_error_rate, round_trip, take_turn, BrainRecognizer, BrainSynthesizer, Clip,
    Recognizer, Synthesizer, DEFAULT_RECOGNIZER, DEFAULT_SYNTHESIZER,
};
use splinter_sdk::vocabulary::prompt::persona_prompt;
use splinter_sdk::vocabulary::speech::{Portrayal, SpeakerProfile};
use voice_loop::{
    pair_questions, read_sentences, RoundTripReport, Spread, TurnItem, TurnReport, TurnsReport,
};

/// The longest answer a turn generates, in tokens: a spoken answer is short.
const ANSWER_TOKENS: u32 = 400;

#[derive(Parser)]
#[command(about = "Talk to a persona and have it talk back, measuring each stage")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Args)]
struct Models {
    /// The recognition model, in brain's model store.
    #[arg(long, default_value = DEFAULT_RECOGNIZER)]
    asr: String,
    /// The synthesis model, in brain's model store.
    #[arg(long, default_value = DEFAULT_SYNTHESIZER)]
    tts: String,
    /// The seed the persona's synthetic voice is rendered from.
    #[arg(long, default_value_t = 1)]
    seed: u64,
}

impl Models {
    fn speaker(&self) -> SpeakerProfile {
        SpeakerProfile::new(self.seed, Portrayal::synthetic_theatrical())
    }
}

#[derive(Subcommand)]
enum Command {
    /// Speak a sentence in the persona's synthetic voice.
    Speak {
        #[command(flatten)]
        models: Models,
        /// What to say.
        #[arg(long)]
        text: String,
        /// Where to write the WAV file.
        #[arg(long)]
        out: PathBuf,
    },
    /// Speak sentences, hear them again, and score what was lost.
    Roundtrip {
        #[command(flatten)]
        models: Models,
        /// A text file with one sentence per line.
        #[arg(long)]
        sentences: PathBuf,
        /// Also keep the report here.
        #[arg(long)]
        report: Option<PathBuf>,
    },
    /// Hear a question, answer as the persona, and speak the answer.
    Turn {
        #[command(flatten)]
        models: Models,
        /// The listener's speech, a WAV file.
        #[arg(long = "in")]
        input: PathBuf,
        /// Where to write the spoken answer, a WAV file.
        #[arg(long)]
        out: PathBuf,
        /// The person the persona answers as.
        #[arg(long, default_value = "Samuel Adams")]
        persona: String,
        /// The text model's checkpoint directory.
        #[arg(long)]
        base: PathBuf,
        /// A persona adapter to attach to it.
        #[arg(long)]
        adapter: Option<PathBuf>,
        /// Also keep the report here.
        #[arg(long)]
        report: Option<PathBuf>,
    },
    /// Answer a directory of recorded questions, loading the models once.
    Turns {
        #[command(flatten)]
        models: Models,
        /// A directory of WAV recordings, taken in file-name order.
        #[arg(long)]
        recordings: PathBuf,
        /// The questions the recordings were made from, one per line in the
        /// same order, to score recognition against.
        #[arg(long)]
        questions: Option<PathBuf>,
        /// Where to write each spoken answer.
        #[arg(long)]
        out_dir: PathBuf,
        /// The person the persona answers as.
        #[arg(long, default_value = "Samuel Adams")]
        persona: String,
        /// The text model's checkpoint directory.
        #[arg(long)]
        base: PathBuf,
        /// A persona adapter to attach to it.
        #[arg(long)]
        adapter: Option<PathBuf>,
        /// Also keep the report here.
        #[arg(long)]
        report: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Speak { models, text, out } => speak(&models, &text, &out),
        Command::Roundtrip {
            models,
            sentences,
            report,
        } => roundtrip(&models, &sentences, report.as_deref()),
        Command::Turn {
            models,
            input,
            out,
            persona,
            base,
            adapter,
            report,
        } => turn(
            &models,
            &input,
            &out,
            &persona,
            &base,
            adapter.as_deref(),
            report.as_deref(),
        ),
        Command::Turns {
            models,
            recordings,
            questions,
            out_dir,
            persona,
            base,
            adapter,
            report,
        } => turns(
            &models,
            &Batch {
                recordings: &recordings,
                questions: questions.as_deref(),
                out_dir: &out_dir,
            },
            &Persona::load(&persona, &base, adapter.as_deref())?,
            report.as_deref(),
        ),
    }
}

fn speak(models: &Models, text: &str, out: &std::path::Path) -> Result<()> {
    let speaker = models.speaker();
    let clip = BrainSynthesizer::load(&models.tts)?.speak(text, &speaker)?;
    clip.save(out)
        .with_context(|| format!("writing {}", out.display()))?;
    println!(
        "{}",
        serde_json::json!({
            "wrote": out,
            "seconds": clip.seconds(),
            "speaker": speaker,
        })
    );
    Ok(())
}

fn roundtrip(
    models: &Models,
    file: &std::path::Path,
    keep: Option<&std::path::Path>,
) -> Result<()> {
    let text =
        std::fs::read_to_string(file).with_context(|| format!("reading {}", file.display()))?;
    let sentences = read_sentences(&text);
    if sentences.is_empty() {
        bail!("{} holds no sentences", file.display());
    }
    let speaker = models.speaker();
    let result = round_trip(
        &BrainSynthesizer::load(&models.tts)?,
        &BrainRecognizer::load(&models.asr)?,
        &speaker,
        &sentences,
    )?;
    emit(
        &RoundTripReport::new(&models.asr, &models.tts, &speaker, &result),
        keep,
    )
}

/// The persona as a text model: a base, maybe an adapter, and the system turn
/// that makes it answer as the person.
struct Persona {
    name: String,
    system: String,
    answerer: Answerer,
    runtime: tokio::runtime::Runtime,
    base: PathBuf,
    adapter: Option<PathBuf>,
}

impl Persona {
    fn load(name: &str, base: &std::path::Path, adapter: Option<&std::path::Path>) -> Result<Self> {
        Ok(Self {
            name: name.to_string(),
            system: persona_prompt(name),
            answerer: Answerer::load(base, adapter, None, name)?,
            runtime: tokio::runtime::Runtime::new()?,
            base: base.to_path_buf(),
            adapter: adapter.map(std::path::Path::to_path_buf),
        })
    }

    fn answer(&self, question: &str) -> Result<String, PolicyError> {
        self.runtime
            .block_on(self.answerer.ask(&self.system, question, ANSWER_TOKENS))
            .map(|reply| reply.text.trim().to_string())
            .map_err(|e| PolicyError::Generate {
                path: self.base.clone(),
                adapter: self.adapter.clone(),
                reason: e.to_string(),
            })
    }
}

fn read_clip(path: &std::path::Path) -> Result<Clip> {
    Clip::from_wav(&std::fs::read(path).with_context(|| format!("reading {}", path.display()))?)
        .with_context(|| format!("{} is not a WAV file", path.display()))
}

fn turn(
    models: &Models,
    input: &std::path::Path,
    out: &std::path::Path,
    persona: &str,
    base: &std::path::Path,
    adapter: Option<&std::path::Path>,
    keep: Option<&std::path::Path>,
) -> Result<()> {
    let heard = read_clip(input)?;
    let speaker = models.speaker();
    let persona = Persona::load(persona, base, adapter)?;
    let turn = take_turn(
        &BrainRecognizer::load(&models.asr)?,
        &mut |question: &str| persona.answer(question),
        &BrainSynthesizer::load(&models.tts)?,
        &speaker,
        &heard,
    )?;
    turn.reply
        .save(out)
        .with_context(|| format!("writing {}", out.display()))?;
    emit(&TurnReport::new(&persona.name, &speaker, &turn), keep)
}

/// Where a batch of questions comes from and its answers go.
struct Batch<'a> {
    recordings: &'a std::path::Path,
    questions: Option<&'a std::path::Path>,
    out_dir: &'a std::path::Path,
}

fn turns(
    models: &Models,
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

    let speaker = models.speaker();
    let recognizer = BrainRecognizer::load(&models.asr)?;
    let synthesizer = BrainSynthesizer::load(&models.tts)?;
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

/// Print `report` as JSON, and keep a copy where asked.
fn emit(report: &impl serde::Serialize, keep: Option<&std::path::Path>) -> Result<()> {
    let json = serde_json::to_string_pretty(report)?;
    if let Some(path) = keep {
        std::fs::write(path, &json).with_context(|| format!("writing {}", path.display()))?;
    }
    println!("{json}");
    Ok(())
}
