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
//! ```
//!
//! Every command prints a JSON report on stdout. The voice is a synthetic
//! theatrical portrayal and every report says so.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};
use splinter_sdk::model::answer::Answerer;
use splinter_sdk::model::speech::{
    round_trip, take_turn, BrainRecognizer, BrainSynthesizer, Clip, Synthesizer,
    DEFAULT_RECOGNIZER, DEFAULT_SYNTHESIZER,
};
use splinter_sdk::vocabulary::prompt::persona_prompt;
use splinter_sdk::vocabulary::speech::{Portrayal, SpeakerProfile};
use voice_loop::{read_sentences, RoundTripReport, TurnReport};

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

fn turn(
    models: &Models,
    input: &std::path::Path,
    out: &std::path::Path,
    persona: &str,
    base: &std::path::Path,
    adapter: Option<&std::path::Path>,
    keep: Option<&std::path::Path>,
) -> Result<()> {
    let heard = Clip::from_wav(
        &std::fs::read(input).with_context(|| format!("reading {}", input.display()))?,
    )
    .with_context(|| format!("{} is not a WAV file", input.display()))?;
    let speaker = models.speaker();
    let answerer = Answerer::load(base, adapter, None, persona)?;
    let system = persona_prompt(persona);
    let runtime = tokio::runtime::Runtime::new()?;
    let mut respond = |question: &str| {
        runtime
            .block_on(answerer.ask(&system, question, ANSWER_TOKENS))
            .map(|reply| reply.text.trim().to_string())
            .map_err(|e| splinter_sdk::model::error::PolicyError::Generate {
                path: base.to_path_buf(),
                adapter: adapter.map(std::path::Path::to_path_buf),
                reason: e.to_string(),
            })
    };
    let turn = take_turn(
        &BrainRecognizer::load(&models.asr)?,
        &mut respond,
        &BrainSynthesizer::load(&models.tts)?,
        &speaker,
        &heard,
    )?;
    turn.reply
        .save(out)
        .with_context(|| format!("writing {}", out.display()))?;
    emit(&TurnReport::new(persona, &speaker, &turn), keep)
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
