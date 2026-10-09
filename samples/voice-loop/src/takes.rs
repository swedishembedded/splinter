// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements evaluation of synthetic speech that does not
// trust a single recogniser, for its clients. If your team needs expertise in
// measuring how natural and how intelligible generated speech is, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Several takes of each sentence in one pinned voice, each heard by two
//! recognisers.
//!
//! The takes are the material of every judgement of delivery: a rating page
//! shows a person pairs of them, a reward ranks them, a preference is learned
//! from them. The voice is pinned to one rendering of a reference sentence so the
//! seeds vary delivery and not who is speaking.
//!
//! One recogniser is a poor judge. A one-shot transducer can drop the last words of
//! a clip that ends with the speech, return nothing, or say the end twice. Each take
//! is therefore heard by two recognisers, once as it is and once with a short tail
//! of silence, and a take is `agreed` only when both recognisers, given the tail,
//! hear it within the bound. The record keeps every transcript.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Args;
use serde::Serialize;
use splinter_sdk::model::speech::{
    corpus_word_error_rate, with_trailing_silence, BrainRecognizer, BrainSynthesizer, Clip,
    Recognizer, Synthesizer, DEFAULT_RECOGNIZER, DEFAULT_SYNTHESIZER,
};
use splinter_sdk::vocabulary::digest::Digest;
use splinter_sdk::vocabulary::speech::{Portrayal, SpeakerProfile};

use crate::batch::{emit, read_clip};
use crate::read_sentences;

/// The sentence whose rendering (seed 1) fixes the voice.
const REFERENCE_SENTENCE: &str = "The people must be taught to value their liberty.";

/// The decode window of a fixed-window recogniser: a take is a sentence, and a
/// window sized to it is several times faster than the default thirty seconds.
const WINDOW_SECONDS: f32 = 10.0;

/// Seconds of silence after a take before it is heard again.
const TAIL_SECONDS: f32 = 0.6;

#[derive(Args)]
pub struct TakesArgs {
    /// The synthesis model, in brain's model store.
    #[arg(long, default_value = DEFAULT_SYNTHESIZER)]
    tts: String,
    /// The first recogniser, in brain's model store.
    #[arg(long, default_value = DEFAULT_RECOGNIZER)]
    asr: String,
    /// The second recogniser.
    #[arg(long, default_value = "Qwen/Qwen3-ASR-1.7B")]
    asr_second: String,
    /// The sentences to speak, one per line.
    #[arg(long)]
    sentences: PathBuf,
    /// Where the takes and their manifest are written.
    #[arg(long)]
    out_dir: PathBuf,
    /// Takes per sentence.
    #[arg(long, default_value_t = 4)]
    per_sentence: usize,
    /// The seed the first take of a sentence is drawn with; take k of sentence i
    /// uses first_seed + 1000 * i + k.
    #[arg(long, default_value_t = 10_000)]
    first_seed: u64,
    /// The largest word error rate, under both recognisers, of an agreed take.
    #[arg(long, default_value_t = 0.2)]
    max_wer: f32,
    /// Also keep the summary here.
    #[arg(long)]
    report: Option<PathBuf>,
}

/// What one recogniser heard of a take.
#[derive(Serialize)]
struct Heard {
    /// As the take is.
    raw: String,
    raw_word_error_rate: Option<f32>,
    /// With a tail of silence.
    tailed: String,
    tailed_word_error_rate: Option<f32>,
}

/// One take.
#[derive(Serialize)]
struct Take {
    id: String,
    sentence: usize,
    text: String,
    seed: u64,
    wav: String,
    audio: Digest,
    seconds: f64,
    heard: BTreeMap<String, Heard>,
    /// Both recognisers, with the tail, hear it within the bound.
    agreed: bool,
}

/// What the run found.
#[derive(Serialize)]
struct Summary<'a> {
    portrayal: &'a Portrayal,
    takes: usize,
    agreed: usize,
    /// Per recogniser: transcripts that were empty as the take is, and with the tail.
    empty_raw: BTreeMap<String, usize>,
    empty_tailed: BTreeMap<String, usize>,
    /// Per recogniser: takes whose raw transcript differed from the text, and tailed.
    differing_raw: BTreeMap<String, usize>,
    differing_tailed: BTreeMap<String, usize>,
    /// Takes the two recognisers disagree on (one within the bound, one not).
    contested: usize,
}

fn wer(text: &str, heard: &str) -> Option<f32> {
    corpus_word_error_rate([(text, heard)])
}

fn hear(recogniser: &BrainRecognizer, text: &str, take: &Clip) -> Result<Heard> {
    let raw = recogniser.transcribe(take)?.text;
    let tailed = recogniser
        .transcribe(&with_trailing_silence(take, TAIL_SECONDS))?
        .text;
    Ok(Heard {
        raw_word_error_rate: wer(text, &raw),
        tailed_word_error_rate: wer(text, &tailed),
        raw,
        tailed,
    })
}

/// Add `by` to the count kept for `name`.
fn bump(counts: &mut BTreeMap<String, usize>, name: &str, by: usize) {
    *counts.entry(name.to_string()).or_insert(0) += by;
}

/// Render the takes, hear them and write the manifest.
pub fn run(args: &TakesArgs) -> Result<()> {
    let portrayal = Portrayal::synthetic_theatrical();
    let sentences = read_sentences(
        &std::fs::read_to_string(&args.sentences)
            .with_context(|| format!("reading {}", args.sentences.display()))?,
    );
    std::fs::create_dir_all(&args.out_dir)?;

    // The voice: one rendering of the reference sentence, fixed for the run.
    let free = BrainSynthesizer::load(&args.tts)?;
    let reference = args.out_dir.join("voice.wav");
    free.speak(
        REFERENCE_SENTENCE,
        &SpeakerProfile::new(1, portrayal.clone()),
    )?
    .save(&reference)
    .context("writing the reference voice")?;
    drop(free);
    let voice = BrainSynthesizer::load(&args.tts)?.pinned_to(&reference);

    let recognisers = [
        (
            args.asr.clone(),
            BrainRecognizer::load_windowed(&args.asr, WINDOW_SECONDS)?,
        ),
        (
            args.asr_second.clone(),
            BrainRecognizer::load_windowed(&args.asr_second, WINDOW_SECONDS)?,
        ),
    ];
    let mut manifest = std::fs::File::create(args.out_dir.join("manifest.jsonl"))?;
    let (mut takes, mut agreed, mut contested) = (0, 0, 0);
    let mut counts: [BTreeMap<String, usize>; 4] = Default::default();
    for (i, text) in sentences.iter().enumerate() {
        for k in 0..args.per_sentence {
            let seed = args.first_seed + 1000 * i as u64 + k as u64;
            let clip = voice.speak(text, &SpeakerProfile::new(seed, portrayal.clone()))?;
            let id = format!("take-{i:04}-{k}");
            let wav = format!("{id}.wav");
            let path = args.out_dir.join(&wav);
            clip.save(&path)?;
            let bytes = std::fs::read(&path)?;
            let mut heard = BTreeMap::new();
            for (name, recogniser) in &recognisers {
                // Heard at the recogniser's own rate, from the file as written.
                let heard_by = hear(recogniser, text, &read_clip(&path)?)?;
                let empty = |text: &str| usize::from(text.trim().is_empty());
                let differs = |heard: &str| usize::from(wer(text, heard).is_none_or(|w| w > 0.0));
                bump(&mut counts[0], name, empty(&heard_by.raw));
                bump(&mut counts[1], name, empty(&heard_by.tailed));
                bump(&mut counts[2], name, differs(&heard_by.raw));
                bump(&mut counts[3], name, differs(&heard_by.tailed));
                heard.insert(name.clone(), heard_by);
            }
            let within = |h: &Heard| h.tailed_word_error_rate.is_some_and(|w| w <= args.max_wer);
            let ok: Vec<bool> = heard.values().map(within).collect();
            let all = ok.iter().all(|b| *b);
            takes += 1;
            agreed += usize::from(all);
            contested += usize::from(ok.iter().any(|b| *b) && !all);
            let take = Take {
                id,
                sentence: i,
                text: text.clone(),
                seed,
                wav,
                audio: Digest::of(&bytes),
                seconds: clip.seconds(),
                heard,
                agreed: all,
            };
            writeln!(manifest, "{}", serde_json::to_string(&take)?)?;
        }
    }
    let [empty_raw, empty_tailed, differing_raw, differing_tailed] = counts;
    emit(
        &Summary {
            portrayal: &portrayal,
            takes,
            agreed,
            empty_raw,
            empty_tailed,
            differing_raw,
            differing_tailed,
            contested,
        },
        args.report.as_deref(),
    )
}
