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
//! voice-loop roundtrip --sentences FILE          speak sentences, hear them again, score what was lost
//! voice-loop turn      --in q.wav --out a.wav    hear a question, answer as the persona, speak the answer
//! voice-loop turns     --recordings DIR ...      the same for a directory of questions
//! ```
//!
//! Every command prints a JSON report on stdout. The voice is a synthetic
//! theatrical portrayal and every report says so. Recognition and synthesis
//! are requests to a brain server (`--brain-endpoint`, or one started from
//! `--brain-bin`), which serves the models `--asr` and `--tts` name.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};
use splinter_sdk::model::speech::{
    round_trip, take_turn, BrainRecognizer, BrainSynthesizer, Recognizer, Sentences, Synthesizer,
    DEFAULT_RECOGNIZER, DEFAULT_SYNTHESIZER,
};
use splinter_sdk::model::BrainEndpoints;
use splinter_sdk::vocabulary::speech::SpeakerProfile;
use voice_loop::batch::{
    emit, read_clip, streamed_turns, turns, Batch, Names, Persona, SPOKEN_PIECE_WORDS,
};
use voice_loop::voice::VoiceSeed;
use voice_loop::{read_sentences, RoundTripReport, TurnReport};

#[derive(Parser)]
#[command(about = "Talk to a persona and have it talk back, measuring each stage")]
struct Cli {
    #[command(flatten)]
    device: splinter_sdk::device::DeviceFlags,
    #[command(subcommand)]
    command: Command,
}

#[derive(Args)]
struct Models {
    /// The recognition model, as the brain server lists it.
    #[arg(long, default_value = DEFAULT_RECOGNIZER)]
    asr: String,
    /// The synthesis model, as the brain server lists it.
    #[arg(long, default_value = DEFAULT_SYNTHESIZER)]
    tts: String,
}

impl Models {
    fn names<'a>(&'a self, endpoints: &'a BrainEndpoints, speaker: SpeakerProfile) -> Names<'a> {
        Names {
            endpoints,
            asr: &self.asr,
            tts: &self.tts,
            speaker,
        }
    }
}

#[derive(Subcommand)]
enum Command {
    /// Talk to a sven agent that answers as the persona: its words are spoken as they are written.
    Agent(voice_loop::agent::AgentArgs),
    /// Fit the naturalness judge to a listener's blind choices and report what it predicts.
    Judge(voice_loop::judge::JudgeArgs),
    /// Teach an agent facts, rules and procedures by what is said to it and test what it kept.
    TeachEval(voice_loop::teach_eval::TeachEvalArgs),
    /// Read utterances for what they teach, with the plain readers and with a model.
    TeachRead(voice_loop::teach_eval::TeachReadArgs),
    /// Teach the persona how to speak: hear a directive and keep what it teaches.
    Teach(voice_loop::teach::TeachArgs),
    /// Speak a sentence in the persona's synthetic voice.
    Speak {
        #[command(flatten)]
        models: Models,
        #[command(flatten)]
        voice: VoiceSeed,
        /// Lessons the persona has been taught; words they cover are said as taught.
        #[arg(long)]
        lessons: Option<PathBuf>,
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
        #[command(flatten)]
        voice: VoiceSeed,
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
    /// The text path's loss on chat records (such as the held-out questions an
    /// earlier `ingress-train` run kept): what the model does with the words
    /// themselves.
    TextScore {
        /// The language model's checkpoint directory.
        #[arg(long)]
        base: PathBuf,
        /// A persona adapter folded into it.
        #[arg(long)]
        adapter: Option<PathBuf>,
        /// A chat-records file (`test-chat.jsonl`).
        #[arg(long)]
        records: PathBuf,
    },
    /// Join the shards `spoken-set --shard` wrote into one set.
    SpokenMerge {
        /// The directory the shards wrote `set-N.json` into.
        #[arg(long)]
        dir: PathBuf,
    },
    /// Speak sentences, hear them again, and score what was lost.
    Roundtrip {
        #[command(flatten)]
        models: Models,
        #[command(flatten)]
        voice: VoiceSeed,
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
        #[command(flatten)]
        voice: VoiceSeed,
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
        #[command(flatten)]
        voice: VoiceSeed,
        /// Speak each sentence as the model writes it, rather than the whole answer at the end.
        #[arg(long)]
        stream: bool,
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
    let cli = Cli::parse();
    cli.device.apply()?;
    match cli.command {
        Command::Agent(args) => voice_loop::agent::run(&cli.device, &args),
        Command::TeachEval(args) => voice_loop::teach_eval::run(&cli.device, &args),
        Command::TeachRead(args) => voice_loop::teach_eval::read(&cli.device, &args),
        Command::Judge(args) => voice_loop::judge::run(&args),
        Command::Teach(args) => voice_loop::teach::teach(&cli.device, &args),
        Command::Speak {
            models,
            voice,
            lessons,
            text,
            out,
        } => voice_loop::teach::speak(
            cli.device.brain()?.endpoints(),
            &models.tts,
            &voice.speaker(&voice_loop::voice::manifest_beside(&out))?,
            lessons.as_deref(),
            &text,
            &out,
        ),
        Command::SpeakSet {
            models,
            voice,
            sentences,
            out_dir,
            max_wer,
            attempts,
        } => voice_loop::speakset::speak_set(
            &models.names(
                cli.device.brain()?.endpoints(),
                voice.speaker(&voice_loop::voice::manifest_in(&out_dir))?,
            ),
            &sentences,
            &out_dir,
            max_wer,
            attempts,
        ),
        Command::SpokenMerge { dir } => voice_loop::spoken::merge(&dir),
        Command::TextScore {
            base,
            adapter,
            records,
        } => {
            let brain = cli.device.brain()?;
            let weights = splinter_sdk::model::LocalWeights {
                base,
                adapter,
                context_tokens: None,
            };
            let score = splinter_sdk::model::served_train::score_chat(
                brain.endpoints(),
                &weights,
                &records,
                false,
            )?;
            println!(
                "{}",
                serde_json::json!({"loss": score.loss, "positions": score.positions, "records": score.records})
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
        } => {
            let brain = cli.device.brain()?;
            spoken_set(
                (
                    &BrainSynthesizer::new(brain.endpoints(), &models.tts),
                    &BrainRecognizer::new(brain.endpoints(), &models.asr),
                ),
                &questions,
                (&voices, &test_voices, test_every),
                &shard,
                &name,
                &out_dir,
                max_wer,
            )
        }
        Command::Roundtrip {
            models,
            voice,
            sentences,
            report,
        } => roundtrip(
            cli.device.brain()?.endpoints(),
            &models,
            voice.unrecorded_speaker()?,
            &sentences,
            report.as_deref(),
        ),
        Command::Turn {
            models,
            voice,
            input,
            out,
            persona,
            base,
            adapter,
            report,
        } => {
            let brain = cli.device.brain()?;
            turn(
                brain.endpoints(),
                &models,
                voice.speaker(&voice_loop::voice::manifest_beside(&out))?,
                (&input, &out),
                &Persona::load(brain.endpoints(), &persona, &base, adapter.as_deref())?,
                report.as_deref(),
            )
        }
        Command::Turns {
            models,
            voice,
            stream,
            recordings,
            questions,
            out_dir,
            persona,
            base,
            adapter,
            system,
            report,
        } => {
            let brain = cli.device.brain()?;
            let names = models.names(
                brain.endpoints(),
                voice.speaker(&voice_loop::voice::manifest_in(&out_dir))?,
            );
            let batch = Batch {
                recordings: &recordings,
                questions: questions.as_deref(),
                out_dir: &out_dir,
            };
            let persona = {
                let loaded = Persona::load(brain.endpoints(), &persona, &base, adapter.as_deref())?;
                match system {
                    Some(system) => loaded.under(system),
                    None => loaded,
                }
            };
            if stream {
                streamed_turns(&names, &batch, &persona, report.as_deref())
            } else {
                turns(&names, &batch, &persona, report.as_deref())
            }
        }
    }
}

fn spoken_set(
    voices_of: (&dyn Synthesizer, &dyn Recognizer),
    file: &std::path::Path,
    split: (&[u64], &[u64], usize),
    shard: &str,
    name: &str,
    out_dir: &std::path::Path,
    max_wer: f32,
) -> Result<()> {
    let (voices, test_voices, test_every) = split;
    let (index, count) = voice_loop::spoken::parse_shard(shard)?;
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
    let (train_set, mut rejected) =
        voice_loop::spoken::record_set(name, &train_q, voices, max_wer, voices_of, out_dir)?;
    let (test_set, rejected_test) =
        voice_loop::spoken::record_set(name, &test_q, test_voices, max_wer, voices_of, out_dir)?;
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

fn roundtrip(
    endpoints: &BrainEndpoints,
    models: &Models,
    speaker: SpeakerProfile,
    file: &std::path::Path,
    keep: Option<&std::path::Path>,
) -> Result<()> {
    let text =
        std::fs::read_to_string(file).with_context(|| format!("reading {}", file.display()))?;
    let sentences = read_sentences(&text);
    if sentences.is_empty() {
        bail!("{} holds no sentences", file.display());
    }
    let result = round_trip(
        &BrainSynthesizer::new(endpoints, &models.tts),
        &BrainRecognizer::new(endpoints, &models.asr),
        &speaker,
        &sentences,
    )?;
    emit(
        &RoundTripReport::new(&models.asr, &models.tts, &speaker, &result),
        keep,
    )
}

fn turn(
    endpoints: &BrainEndpoints,
    models: &Models,
    speaker: SpeakerProfile,
    (input, out): (&std::path::Path, &std::path::Path),
    persona: &Persona,
    keep: Option<&std::path::Path>,
) -> Result<()> {
    let heard = read_clip(input)?;
    let turn = take_turn(
        &BrainRecognizer::new(endpoints, &models.asr),
        &mut |question: &str| persona.answer(question),
        &Sentences::new(
            BrainSynthesizer::new(endpoints, &models.tts),
            SPOKEN_PIECE_WORDS,
        ),
        &speaker,
        &heard,
    )?;
    turn.reply
        .save(out)
        .with_context(|| format!("writing {}", out.display()))?;
    emit(&TurnReport::new(&persona.name, &speaker, &turn), keep)
}
