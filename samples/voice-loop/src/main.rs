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
//! voice-loop speak-set  --sentences FILE          record sentences, keeping those heard back well enough
//! voice-loop spoken-set --questions FILE         record questions in several voices into a pinned set
//! voice-loop spoken-set --questions FILE         record questions in several voices into a pinned set
//! voice-loop ingress-train --set DIR ...        teach a frozen language model to listen
//! voice-loop listen-turns --recordings DIR ...  answer recorded questions without a transcript
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
use splinter_sdk::model::speech::{
    round_trip, speak_verified, take_turn, BrainRecognizer, BrainSynthesizer, ListenerOptions,
    Sentences, Synthesizer, DEFAULT_RECOGNIZER, DEFAULT_SYNTHESIZER,
};
use splinter_sdk::vocabulary::prompt::persona_prompt;
use splinter_sdk::vocabulary::speech::{Portrayal, SpeakerProfile};
use voice_loop::batch::{
    emit, listen_turns, read_clip, turns, Batch, Names, Persona, SPOKEN_PIECE_WORDS,
};
use voice_loop::{read_sentences, Recorded, RoundTripReport, SpeakSetReport, TurnReport};

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
    fn names(&self) -> Names<'_> {
        Names {
            asr: &self.asr,
            tts: &self.tts,
            speaker: self.speaker(),
        }
    }

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
    /// Record sentences, keeping each only if it is heard back well enough.
    SpeakSet {
        #[command(flatten)]
        models: Models,
        /// A text file with one sentence per line.
        #[arg(long)]
        sentences: PathBuf,
        /// Where to write `q01.wav`, `q02.wav` ... and `sentences.txt`, the
        /// sentences that were kept, in order.
        #[arg(long)]
        out_dir: PathBuf,
        /// The word error rate above which a recording is not kept.
        #[arg(long, default_value_t = 0.2)]
        max_wer: f32,
        /// How many seeds to try for a sentence before rejecting it.
        #[arg(long, default_value_t = 6)]
        attempts: usize,
    },
    /// Record questions in several voices into a set that can be pinned and split.
    SpokenSet {
        #[command(flatten)]
        models: Models,
        /// A JSON-lines file of `{"id", "text", "group"?}`.
        #[arg(long)]
        questions: PathBuf,
        /// The voices training questions are recorded in: comma-separated seeds.
        #[arg(long, value_delimiter = ',', required = true)]
        voices: Vec<u64>,
        /// The held-out voices the held-out questions are recorded in.
        #[arg(long, value_delimiter = ',', required = true)]
        test_voices: Vec<u64>,
        /// Every this-many-th question group is held out.
        #[arg(long, default_value_t = 6)]
        test_every: usize,
        /// Record only this share of the questions, `INDEX/COUNT`, so several
        /// processes can record one set at once; `spoken-merge` joins them.
        #[arg(long, default_value = "0/1")]
        shard: String,
        /// The set's name.
        #[arg(long, default_value = "spoken")]
        name: String,
        /// Where to write the recordings and `set.json`.
        #[arg(long)]
        out_dir: PathBuf,
        /// The word error rate above which a recording is not kept.
        #[arg(long, default_value_t = 0.2)]
        max_wer: f32,
    },
    /// Name the parts of a speaking persona by content address, refusing parts
    /// attached to something they were not made for.
    Bundle {
        #[command(flatten)]
        models: Models,
        /// The person.
        #[arg(long)]
        persona: String,
        /// The language model's checkpoint directory.
        #[arg(long)]
        thinker: PathBuf,
        /// The persona adapter, if the persona has one.
        #[arg(long)]
        adapter: Option<PathBuf>,
        /// The checkpoint directory the adapter was trained on; the thinker when absent.
        #[arg(long)]
        adapter_base: Option<PathBuf>,
        /// The trained projector.
        #[arg(long)]
        projector: PathBuf,
        /// The checkpoint directory the projector was trained against; the thinker when absent.
        #[arg(long)]
        projector_thinker: Option<PathBuf>,
        /// The recogniser whose features the projector reads.
        #[arg(long)]
        features_dir: PathBuf,
        /// The synthesizer's checkpoint directory.
        #[arg(long)]
        synthesizer_dir: PathBuf,
        /// Where to write `bundle.json`.
        #[arg(long)]
        out: PathBuf,
    },
    /// The text path's loss on the chat records `ingress-train` kept of its
    /// held-out questions: what the model does with the words themselves.
    TextScore {
        /// The language model's checkpoint directory.
        #[arg(long)]
        base: PathBuf,
        /// A persona adapter folded into it.
        #[arg(long)]
        adapter: Option<PathBuf>,
        /// The `test-chat.jsonl` an `ingress-train` run wrote.
        #[arg(long)]
        records: PathBuf,
    },
    /// Join the shards `spoken-set --shard` wrote into one set.
    SpokenMerge {
        /// The directory the shards wrote `set-N.json` into.
        #[arg(long)]
        dir: PathBuf,
    },
    /// Teach a frozen language model to listen: train a projector on a spoken set.
    IngressTrain {
        #[command(flatten)]
        models: Models,
        /// The directory `spoken-set` wrote.
        #[arg(long)]
        set: PathBuf,
        /// A JSON-lines file of `{"id", "system", "answer"}`: the text path's answers.
        #[arg(long)]
        answers: PathBuf,
        /// The language model's checkpoint directory.
        #[arg(long)]
        base: PathBuf,
        /// A persona adapter held frozen on it.
        #[arg(long)]
        adapter: Option<PathBuf>,
        /// Where to write the projector, the held-out chat records and the report.
        #[arg(long)]
        out: PathBuf,
        /// The recogniser whose encoder gives the features.
        #[arg(long, default_value = "Qwen/Qwen3-ASR-1.7B")]
        features_model: String,
        /// Seconds every clip is padded to.
        #[arg(long, default_value_t = 8.0)]
        window: f32,
        /// Optimiser steps.
        #[arg(long, default_value_t = 300)]
        steps: u32,
        /// Examples per step.
        #[arg(long, default_value_t = 8)]
        batch: usize,
        /// Peak projector learning rate.
        #[arg(long, default_value_t = 1e-3)]
        lr: f32,
        /// Every this-many-th question group is held out.
        #[arg(long, default_value_t = 6)]
        test_every: usize,
        /// Held-out voices: comma-separated seeds.
        #[arg(long, value_delimiter = ',', required = true)]
        test_voices: Vec<u64>,
        /// Share of recordings that also train a transcription example.
        #[arg(long, default_value_t = 1.0)]
        transcribe_share: f32,
        /// Longest example, in tokens.
        #[arg(long, default_value_t = 480)]
        block: u32,
        /// Start from this projector; with `--steps 0`, only evaluate it.
        #[arg(long)]
        init_projector: Option<PathBuf>,
    },
    /// Answer a directory of recorded questions with a model that listens: no transcript in between.
    ListenTurns {
        #[command(flatten)]
        models: Models,
        /// The recogniser whose encoder gives the features the projector reads.
        #[arg(long, default_value = "Qwen/Qwen3-ASR-1.7B")]
        features_model: String,
        /// The trained projector.
        #[arg(long)]
        projector: PathBuf,
        /// A directory of WAV recordings, taken in file-name order.
        #[arg(long)]
        recordings: PathBuf,
        /// The questions the recordings were made from, one per line in order.
        #[arg(long)]
        questions: Option<PathBuf>,
        /// Where to write each spoken answer.
        #[arg(long)]
        out_dir: PathBuf,
        /// The person the persona answers as.
        #[arg(long, default_value = "Samuel Adams")]
        persona: String,
        /// The system turn the persona answers under; the persona prompt when absent.
        #[arg(long)]
        system: Option<String>,
        /// The text model's checkpoint directory.
        #[arg(long)]
        base: PathBuf,
        /// A persona adapter to attach to it.
        #[arg(long)]
        adapter: Option<PathBuf>,
        /// Seconds every clip is padded to; the projector was trained on this.
        #[arg(long, default_value_t = 8.0)]
        window: f32,
        /// Also keep the report here.
        #[arg(long)]
        report: Option<PathBuf>,
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
        /// The system turn the persona answers under; the persona prompt when absent.
        #[arg(long)]
        system: Option<String>,
        /// Also keep the report here.
        #[arg(long)]
        report: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Speak { models, text, out } => speak(&models, &text, &out),
        Command::SpeakSet {
            models,
            sentences,
            out_dir,
            max_wer,
            attempts,
        } => speak_set(&models, &sentences, &out_dir, max_wer, attempts),
        Command::SpokenMerge { dir } => spoken_merge(&dir),
        Command::TextScore {
            base,
            adapter,
            records,
        } => {
            let score =
                splinter_sdk::model::train::score_chat(&base, adapter.as_deref(), &records)?;
            println!(
                "{}",
                serde_json::json!({"loss": score.loss, "token_accuracy": score.token_accuracy, "positions": score.positions, "records": score.records, "skipped": score.skipped})
            );
            Ok(())
        }
        Command::Bundle {
            models,
            persona,
            thinker,
            adapter,
            adapter_base,
            projector,
            projector_thinker,
            features_dir,
            synthesizer_dir,
            out,
        } => {
            use splinter_sdk::vocabulary::speech_bundle::{MadeFor, SpeechBundle};
            let base_digest = |dir: &Option<PathBuf>| {
                voice_loop::bundle::digest_dir(dir.as_deref().unwrap_or(&thinker))
            };
            let recogniser = voice_loop::bundle::dir_part("qwen3-asr", &features_dir)?;
            let reads = recogniser.digest.clone();
            let bundle = SpeechBundle::new(
                persona,
                voice_loop::bundle::dir_part("thinker", &thinker)?,
                adapter
                    .as_ref()
                    .map(|path| {
                        Ok::<_, anyhow::Error>(MadeFor {
                            part: voice_loop::bundle::file_part("persona-adapter", path)?,
                            made_for: base_digest(&adapter_base)?,
                        })
                    })
                    .transpose()?,
                MadeFor {
                    part: voice_loop::bundle::file_part("speech-projector", &projector)?,
                    made_for: base_digest(&projector_thinker)?,
                },
                &reads,
                recogniser,
                voice_loop::bundle::dir_part("synthesizer", &synthesizer_dir)?,
                models.speaker(),
            )?;
            std::fs::write(&out, serde_json::to_string_pretty(&bundle)?)?;
            println!(
                "{}",
                serde_json::json!({"bundle": out, "digest": bundle.digest()?, "persona": bundle.persona(), "portrayal": bundle.voice().portrayal().label()})
            );
            Ok(())
        }
        Command::SpokenSet {
            models,
            questions,
            voices,
            test_voices,
            test_every,
            shard,
            name,
            out_dir,
            max_wer,
        } => spoken_set(
            &models,
            &questions,
            (&voices, &test_voices, test_every),
            &shard,
            &name,
            &out_dir,
            max_wer,
        ),
        Command::IngressTrain {
            models,
            set,
            answers,
            base,
            adapter,
            out,
            features_model,
            window,
            steps,
            batch,
            lr,
            test_every,
            test_voices,
            transcribe_share,
            block,
            init_projector,
        } => {
            let plan = voice_loop::ingress::Plan {
                set_dir: set,
                recognizer: features_model,
                base,
                adapter,
                window_seconds: window,
                steps,
                batch,
                lr,
                test_every,
                test_voices: test_voices.into_iter().collect(),
                transcribe_share,
                seed: models.seed,
                block,
                init_projector,
            };
            let answers = voice_loop::ingress::read_answers(&std::fs::read_to_string(&answers)?)?;
            let report = voice_loop::ingress::train(&plan, &answers, &out)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        Command::ListenTurns {
            models,
            features_model,
            projector,
            recordings,
            questions,
            out_dir,
            persona,
            system,
            base,
            adapter,
            window,
            report,
        } => listen_turns(
            &models.names(),
            &Batch {
                recordings: &recordings,
                questions: questions.as_deref(),
                out_dir: &out_dir,
            },
            &persona,
            &ListenerOptions {
                recognizer: features_model,
                projector,
                base,
                adapter,
                system: system.unwrap_or_else(|| persona_prompt(&persona)),
                window_seconds: window,
                max_new: 300,
            },
            report.as_deref(),
        ),
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
            system,
            report,
        } => turns(
            &models.names(),
            &Batch {
                recordings: &recordings,
                questions: questions.as_deref(),
                out_dir: &out_dir,
            },
            &{
                let loaded = Persona::load(&persona, &base, adapter.as_deref())?;
                match system {
                    Some(system) => loaded.under(system),
                    None => loaded,
                }
            },
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

fn speak_set(
    models: &Models,
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
    let speaker = models.speaker();
    let (synthesizer, recognizer) = (
        BrainSynthesizer::load(&models.tts)?,
        BrainRecognizer::load(&models.asr)?,
    );
    let (mut kept, mut rejected) = (Vec::new(), Vec::new());
    for sentence in sentences {
        match speak_verified(
            &synthesizer,
            &recognizer,
            &speaker,
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
            recognizer: &models.asr,
            synthesizer: &models.tts,
            portrayal: speaker.portrayal().label(),
            max_word_error_rate: max_wer,
            kept: &kept,
            rejected: &rejected,
        },
        Some(&out_dir.join("report.json")),
    )
}

fn spoken_set(
    models: &Models,
    file: &std::path::Path,
    split: (&[u64], &[u64], usize),
    shard: &str,
    name: &str,
    out_dir: &std::path::Path,
    max_wer: f32,
) -> Result<()> {
    let (voices, test_voices, test_every) = split;
    let (index, count) = parse_shard(shard)?;
    let text =
        std::fs::read_to_string(file).with_context(|| format!("reading {}", file.display()))?;
    let questions = voice_loop::spoken::read_questions(&text)?;
    if questions.is_empty() {
        bail!("{} holds no questions", file.display());
    }
    // Held-out questions are recorded only in held-out voices, and training
    // questions only in training voices: a recording held out on one count
    // alone would be in neither side of the split.
    let group_of =
        |q: &voice_loop::spoken::Question| q.group.clone().unwrap_or_else(|| q.id.clone());
    let held = voice_loop::spoken::held_out_group_names(
        questions
            .iter()
            .map(|q| q.group.as_deref().unwrap_or(&q.id)),
        test_every,
    );
    let (test_q, train_q): (Vec<_>, Vec<_>) = questions
        .into_iter()
        .enumerate()
        .filter(|(i, _)| i % count == index)
        .map(|(_, q)| q)
        .partition(|q| held.contains(&group_of(q)));
    let (synthesizer, recognizer) = (
        BrainSynthesizer::load(&models.tts)?,
        BrainRecognizer::load(&models.asr)?,
    );
    let (train_set, mut rejected) = voice_loop::spoken::record_set(
        name,
        &train_q,
        voices,
        max_wer,
        (&synthesizer, &recognizer),
        out_dir,
    )?;
    let (test_set, rejected_test) = voice_loop::spoken::record_set(
        name,
        &test_q,
        test_voices,
        max_wer,
        (&synthesizer, &recognizer),
        out_dir,
    )?;
    rejected.extend(rejected_test);
    let items: Vec<_> = train_set
        .items()
        .iter()
        .chain(test_set.items())
        .cloned()
        .collect();
    let set = splinter_sdk::vocabulary::spoken::SpokenSet::new(
        name,
        train_set.portrayal().clone(),
        items,
    )?;
    let suffix = if count == 1 {
        String::new()
    } else {
        format!("-{index}")
    };
    std::fs::write(
        out_dir.join(format!("set{suffix}.json")),
        serde_json::to_string_pretty(&set)?,
    )?;
    std::fs::write(
        out_dir.join(format!("rejected{suffix}.json")),
        serde_json::to_string_pretty(&rejected)?,
    )?;
    println!(
        "{}",
        serde_json::json!({
            "set": name,
            "digest": set.digest()?,
            "recordings": set.items().len(),
            "train_questions": train_q.len(),
            "test_questions": test_q.len(),
            "rejected": rejected.len(),
            "portrayal": set.portrayal().label(),
        })
    );
    Ok(())
}

/// `INDEX/COUNT` as two numbers, with `INDEX < COUNT`.
fn parse_shard(text: &str) -> Result<(usize, usize)> {
    let (i, n) = text
        .split_once('/')
        .with_context(|| format!("shard {text:?} is not INDEX/COUNT"))?;
    let (i, n): (usize, usize) = (i.parse()?, n.parse()?);
    if n == 0 || i >= n {
        bail!("shard {text:?}: the index must be below the count");
    }
    Ok((i, n))
}

fn spoken_merge(dir: &std::path::Path) -> Result<()> {
    let mut parts: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("set-") && n.ends_with(".json"))
        })
        .collect();
    parts.sort();
    if parts.is_empty() {
        bail!("{} holds no set-N.json shards", dir.display());
    }
    let mut items = Vec::new();
    let mut named: Option<(String, splinter_sdk::vocabulary::speech::Portrayal)> = None;
    for part in &parts {
        let shard: splinter_sdk::vocabulary::spoken::SpokenSet =
            serde_json::from_str(&std::fs::read_to_string(part)?)?;
        named.get_or_insert_with(|| (shard.name().to_string(), shard.portrayal().clone()));
        items.extend(shard.items().iter().cloned());
    }
    let (name, portrayal) = named.context("no shard")?;
    let set = splinter_sdk::vocabulary::spoken::SpokenSet::new(name, portrayal, items)?;
    std::fs::write(dir.join("set.json"), serde_json::to_string_pretty(&set)?)?;
    println!(
        "{}",
        serde_json::json!({"recordings": set.items().len(), "digest": set.digest()?, "shards": parts.len()})
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
    let heard = read_clip(input)?;
    let speaker = models.speaker();
    let persona = Persona::load(persona, base, adapter)?;
    let turn = take_turn(
        &BrainRecognizer::load(&models.asr)?,
        &mut |question: &str| persona.answer(question),
        &Sentences::new(BrainSynthesizer::load(&models.tts)?, SPOKEN_PIECE_WORDS),
        &speaker,
        &heard,
    )?;
    turn.reply
        .save(out)
        .with_context(|| format!("writing {}", out.display()))?;
    emit(&TurnReport::new(&persona.name, &speaker, &turn), keep)
}
