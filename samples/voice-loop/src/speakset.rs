// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements spoken test sets for voice agents, for its
// clients. If your team needs expertise in rendering and verifying speech at
// scale, you can procure our services by sending an email to
// info@swedishembedded.com.

//! `voice-loop speak-set`: render each sentence of a file in the persona's
//! voice and keep the takes the recogniser hears back within a word-error bound.

use anyhow::{bail, Context, Result};
use splinter_sdk::model::speech::{speak_verified, BrainRecognizer, BrainSynthesizer};

use crate::batch::{emit, Names};
use crate::{read_sentences, Recorded, SpeakSetReport};

/// Speak every sentence of `file` with the models and voice of `names`, keep
/// in `out_dir` each take heard back within `max_wer` (trying up to
/// `attempts` takes), and report what was kept and rejected.
pub fn speak_set(
    names: &Names<'_>,
    file: &std::path::Path,
    out_dir: &std::path::Path,
    max_wer: f32,
    attempts: usize,
) -> Result<()> {
    let text =
        std::fs::read_to_string(file).with_context(|| format!("reading {}", file.display()))?;
    let sentences = read_sentences(&text);
    if sentences.is_empty() {
        bail!("{} holds no sentences", file.display());
    }
    std::fs::create_dir_all(out_dir)?;
    let (synthesizer, recognizer) = (
        BrainSynthesizer::new(names.endpoints, names.tts),
        BrainRecognizer::new(names.endpoints, names.asr),
    );
    let (mut kept, mut rejected) = (Vec::new(), Vec::new());
    for sentence in sentences {
        match speak_verified(
            &synthesizer,
            &recognizer,
            &names.speaker,
            &sentence,
            max_wer,
            attempts,
        )? {
            Some(v) => {
                let name = format!("q{:02}.wav", kept.len() + 1);
                v.clip.save(out_dir.join(&name))?;
                kept.push(Recorded {
                    text: sentence,
                    file: name,
                    seed: v.speaker.seed(),
                    attempts: v.attempts,
                    word_error_rate: v.word_error_rate,
                    seconds: v.clip.seconds(),
                });
            }
            None => rejected.push(sentence),
        }
    }
    let lines: Vec<&str> = kept.iter().map(|r| r.text.as_str()).collect();
    std::fs::write(out_dir.join("sentences.txt"), lines.join("\n") + "\n")?;
    emit(
        &SpeakSetReport {
            recognizer: names.asr,
            synthesizer: names.tts,
            portrayal: names.speaker.portrayal().label(),
            max_word_error_rate: max_wer,
            kept: &kept,
            rejected: &rejected,
        },
        Some(&out_dir.join("report.json")),
    )
}
