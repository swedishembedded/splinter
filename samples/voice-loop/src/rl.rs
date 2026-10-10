// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training of speech synthesizers from feedback
// on how they sound, for its clients. If your team needs expertise in
// reinforcement learning for voice models, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Train the synthesizer from scores of its own takes.
//!
//! Each iteration draws a group of takes of a few sentences, scores every take,
//! turns the scores into advantages within their group and moves the policy
//! toward the takes that scored above it. The held-out sentences are spoken before
//! training, during it and after, with fixed seeds, and what they score and how
//! many of their words survive being heard again is the report.
//!
//! The scoring here is a stand-in with a known answer: `--reward speed` pays for
//! fewer codec frames per word. It is a machinery gate, not a claim about naturalness: if
//! the loop cannot make a voice faster, it cannot make one better.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Args, ValueEnum};
use serde::Serialize;
use splinter_sdk::measure::speech_reward::{group_advantages, preference_pair};
use splinter_sdk::model::speech::{
    corpus_word_error_rate, with_trailing_silence, BrainRecognizer, BrainSynthesizer, Recognizer,
    SpeechPolicy, SpeechTake, Synthesizer, UpdateRule, DEFAULT_RECOGNIZER, DEFAULT_SYNTHESIZER,
};
use splinter_sdk::vocabulary::speech::{Portrayal, SpeakerProfile};

use crate::batch::emit;
use crate::read_sentences;

const REFERENCE_SENTENCE: &str = "The people must be taught to value their liberty.";

/// What a take is worth.
#[derive(Clone, Copy, ValueEnum)]
enum Reward {
    /// Fewer codec frames per word: a gate for the machinery.
    Speed,
}

/// How scores become updates.
#[derive(Clone, Copy, ValueEnum)]
enum Method {
    /// Group-relative policy gradient over every take of a group.
    Grpo,
    /// Supervised on the best-scoring take of each group (best-of-n).
    BestOfN,
    /// Direct preference on the best and worst take of each group.
    Dpo,
}

#[derive(Args)]
pub struct RlArgs {
    /// How scores become updates.
    #[arg(long, value_enum, default_value = "grpo")]
    method: Method,
    /// The synthesis model, in brain's model store.
    #[arg(long, default_value = DEFAULT_SYNTHESIZER)]
    tts: String,
    /// The recogniser that hears the held-out takes.
    #[arg(long, default_value = DEFAULT_RECOGNIZER)]
    asr: String,
    /// The sentences, one per line; the last `--held-out` are never trained on.
    #[arg(long)]
    sentences: PathBuf,
    /// Where the reference voice and the trained adapter are written.
    #[arg(long)]
    out_dir: PathBuf,
    /// What a take is worth.
    #[arg(long, value_enum, default_value = "speed")]
    reward: Reward,
    /// Updates to take.
    #[arg(long, default_value_t = 12)]
    iterations: usize,
    /// Sentences per update.
    #[arg(long, default_value_t = 4)]
    sentences_per_iteration: usize,
    /// Takes per sentence.
    #[arg(long, default_value_t = 6)]
    group: usize,
    /// Offsets the seeds of the takes sampled for training, so that a run can be
    /// repeated with different rollouts; the held-out takes are not changed.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Held-out sentences.
    #[arg(long, default_value_t = 8)]
    held_out: usize,
    /// Learning rate.
    #[arg(long, default_value_t = 1e-4)]
    learning_rate: f32,
    /// Weight of the KL term to the original voice.
    #[arg(long, default_value_t = 0.02)]
    kl: f32,
    /// Measure the held-out sentences every this many updates.
    #[arg(long, default_value_t = 4)]
    eval_every: usize,
    /// Also keep the report here.
    #[arg(long)]
    report: Option<PathBuf>,
}

/// What the held-out sentences came to at one point of training.
#[derive(Serialize)]
struct Eval {
    after_updates: usize,
    mean_frames_per_word: f32,
    mean_reward: f32,
    word_error_rate: Option<f32>,
}

#[derive(Serialize)]
struct Step {
    update: usize,
    mean_reward: f32,
    /// The mean log-probability of the sampled tokens; for DPO, the probability the
    /// policy gave the preference.
    mean_logprob: f32,
    mean_kl: f32,
}

#[derive(Serialize)]
struct Report<'a> {
    portrayal: &'a Portrayal,
    method: &'static str,
    iterations: usize,
    group: usize,
    train_sentences: usize,
    held_out_sentences: usize,
    steps: Vec<Step>,
    evals: Vec<Eval>,
}

fn words(text: &str) -> f32 {
    text.split_whitespace().count().max(1) as f32
}

fn score(reward: Reward, text: &str, take: &SpeechTake) -> f32 {
    match reward {
        Reward::Speed => -(take.frames() as f32) / words(text),
    }
}

/// The held-out sentences spoken under the current policy with fixed seeds.
fn evaluate(
    policy: &mut SpeechPolicy,
    judge: &BrainRecognizer,
    held_out: &[String],
    reward: Reward,
    after_updates: usize,
) -> Result<Eval> {
    const TAKES: u64 = 3;
    let (mut frames, mut total, mut n) = (0.0, 0.0, 0.0);
    let mut heard = Vec::new();
    for (i, text) in held_out.iter().enumerate() {
        for k in 0..TAKES {
            let take = policy.take(text, 900_000 + 100 * i as u64 + k)?;
            frames += take.frames() as f32 / words(text);
            total += score(reward, text, &take);
            n += 1.0;
            let said = judge
                .transcribe(&with_trailing_silence(take.audio(), 0.6))?
                .text;
            heard.push((text.clone(), said));
        }
    }
    Ok(Eval {
        after_updates,
        mean_frames_per_word: frames / n,
        mean_reward: total / n,
        word_error_rate: corpus_word_error_rate(
            heard.iter().map(|(a, b)| (a.as_str(), b.as_str())),
        ),
    })
}

/// Train, measuring as it goes.
pub fn run(args: &RlArgs) -> Result<()> {
    let portrayal = Portrayal::synthetic_theatrical();
    let all = read_sentences(
        &std::fs::read_to_string(&args.sentences)
            .with_context(|| format!("reading {}", args.sentences.display()))?,
    );
    if all.len() <= args.held_out + args.sentences_per_iteration {
        bail!(
            "{} sentences are too few for {} held out",
            all.len(),
            args.held_out
        );
    }
    let (train, held_out) = all.split_at(all.len() - args.held_out);
    std::fs::create_dir_all(&args.out_dir)?;

    // The voice every take is pinned to: one rendering of a reference sentence.
    let reference = args.out_dir.join("voice.wav");
    BrainSynthesizer::load(&args.tts)?
        .speak(
            REFERENCE_SENTENCE,
            &SpeakerProfile::new(1, portrayal.clone()),
        )?
        .save(&reference)
        .context("writing the reference voice")?;
    let judge = BrainRecognizer::load(&args.asr)?;
    let mut policy = SpeechPolicy::load(&args.tts, Some(&reference))?;

    let rule = UpdateRule {
        learning_rate: args.learning_rate,
        kl_weight: args.kl,
    };
    let mut evals = vec![evaluate(&mut policy, &judge, held_out, args.reward, 0)?];
    eprintln!(
        "before: {:.3} frames/word, reward {:.3}",
        evals[0].mean_frames_per_word, evals[0].mean_reward
    );
    let mut steps = Vec::new();
    for update in 1..=args.iterations {
        let (mut rewards, mut logprob, mut kl) = (0.0, 0.0, 0.0);
        let mut groups = 0.0;
        for slot in 0..args.sentences_per_iteration {
            let text = &train[((update - 1) * args.sentences_per_iteration + slot) % train.len()];
            let takes = (0..args.group)
                .map(|k| {
                    policy.take(
                        text,
                        1_000_000 * args.seed
                            + 10_000 * update as u64
                            + 100 * slot as u64
                            + k as u64,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let scores: Vec<f32> = takes.iter().map(|t| score(args.reward, text, t)).collect();
            rewards += scores.iter().sum::<f32>() / scores.len() as f32;
            let report = match args.method {
                Method::Grpo => {
                    let group: Vec<(&SpeechTake, f32)> =
                        takes.iter().zip(group_advantages(&scores)).collect();
                    Some(policy.update(&group, &rule)?)
                }
                Method::BestOfN => {
                    let best = (0..takes.len())
                        .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
                        .unwrap_or(0);
                    Some(policy.update(&[(&takes[best], 1.0)], &rule)?)
                }
                Method::Dpo => match preference_pair(&scores) {
                    Some((best, worst)) => {
                        let probability = policy.prefer(&takes[best], &takes[worst], 0.5, &rule)?;
                        logprob += probability;
                        None
                    }
                    None => None,
                },
            };
            if let Some(report) = report {
                logprob += report.mean_logprob;
                kl += report.mean_kl;
            }
            groups += 1.0;
        }
        steps.push(Step {
            update,
            mean_reward: rewards / groups,
            mean_logprob: logprob / groups,
            mean_kl: kl / groups,
        });
        eprintln!(
            "update {update}: train reward {:.3}, kl {:.4}",
            rewards / groups,
            kl / groups
        );
        if update % args.eval_every == 0 || update == args.iterations {
            let eval = evaluate(&mut policy, &judge, held_out, args.reward, update)?;
            eprintln!(
                "held out after {update}: {:.3} frames/word, reward {:.3}",
                eval.mean_frames_per_word, eval.mean_reward
            );
            evals.push(eval);
        }
    }
    policy.save(&args.out_dir.join("adapter.safetensors"))?;
    emit(
        &Report {
            portrayal: &portrayal,
            method: match args.method {
                Method::Grpo => "grpo",
                Method::BestOfN => "best-of-n",
                Method::Dpo => "dpo",
            },
            iterations: args.iterations,
            group: args.group,
            train_sentences: train.len(),
            held_out_sentences: held_out.len(),
            steps,
            evals,
        },
        args.report.as_deref(),
    )
}
