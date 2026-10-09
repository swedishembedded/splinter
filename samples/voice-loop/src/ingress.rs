// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Teaching a language model to listen to a recorded set of questions.
//!
//! The recordings of a [`SpokenSet`] are split into what the projector trains
//! on and what it is tested on (voices and questions it never heard). Each
//! recording becomes examples for a frozen persona model: answer the question
//! it hears, and (for a share of them) say what was said. The answers are the
//! text path's own, so the test is whether hearing a question changes what the
//! model says.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use splinter_sdk::model::speech::{
    padded_to, AudioFeatures, BrainRecognizer, Clip, Ingress, IngressExample, IngressOptions,
    StepSettings,
};
use splinter_sdk::vocabulary::spoken::{Spoken, SpokenSet};

/// The instruction that asks for a transcript instead of an answer.
pub const TRANSCRIBE: &str = "Write down exactly what was said.";

/// A question's answer by the text path, with the system turn it was given under.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Answer {
    /// The question's id (a recording's `represents`).
    pub id: String,
    /// The system turn the answer was written under.
    pub system: String,
    /// The answer.
    pub answer: String,
}

/// The answers in a JSON-lines file.
pub fn read_answers(jsonl: &str) -> Result<BTreeMap<String, Answer>> {
    let mut out = BTreeMap::new();
    for (n, line) in jsonl
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
    {
        let a: Answer =
            serde_json::from_str(line).with_context(|| format!("line {}: not an answer", n + 1))?;
        if a.answer.trim().is_empty() {
            bail!("line {}: the answer to {:?} is empty", n + 1, a.id);
        }
        out.insert(a.id.clone(), a);
    }
    Ok(out)
}

/// Every `every`-th distinct group of `set`, in name order, held out for testing.
pub fn held_out_groups(set: &SpokenSet, every: usize) -> BTreeSet<String> {
    crate::spoken::held_out_group_names(set.items().iter().map(|i| i.group.as_str()), every)
}

/// How a training run goes.
#[derive(Clone, Debug)]
pub struct Plan {
    /// Where the recordings and `set.json` are.
    pub set_dir: PathBuf,
    /// The recogniser whose encoder gives the features.
    pub recognizer: String,
    /// The language model's checkpoint directory.
    pub base: PathBuf,
    /// A persona adapter held frozen on it.
    pub adapter: Option<PathBuf>,
    /// The window every clip is padded to, in seconds.
    pub window_seconds: f32,
    /// Optimiser steps.
    pub steps: u32,
    /// Examples per step.
    pub batch: usize,
    /// Peak learning rate of the projector.
    pub lr: f32,
    /// Every this-many-th question group is held out.
    pub test_every: usize,
    /// Voices (seeds) held out.
    pub test_voices: BTreeSet<u64>,
    /// Share of recordings that also train a transcription example.
    pub transcribe_share: f32,
    /// Seed of the projector and of the order examples are drawn in.
    pub seed: u64,
    /// Longest example in tokens.
    pub block: u32,
    /// A projector to start from (and, with no steps, to evaluate).
    pub init_projector: Option<PathBuf>,
}

/// What a run measured.
#[derive(Debug, Serialize)]
pub struct Report {
    /// Recordings trained on.
    pub train_recordings: usize,
    /// Recordings tested on.
    pub test_recordings: usize,
    /// Recordings held out on one count only, in neither side.
    pub dropped: usize,
    /// Recordings longer than the window, skipped.
    pub too_long: usize,
    /// Mean test loss of the answers before training.
    pub test_loss_before: f32,
    /// Mean test loss after training.
    pub test_loss_after: f32,
    /// Test loss at each evaluation: `(step, loss)`.
    pub test_loss_curve: Vec<(u32, f32)>,
    /// Mean training loss of the last ten steps.
    pub train_loss_last: Option<f32>,
    /// Steps taken.
    pub steps: u32,
}

/// The features of every recording that fits the window, in order; the
/// recordings that do not are counted in the second value and left out.
fn features_of(
    recognizer: &BrainRecognizer,
    dir: &Path,
    items: &[&Spoken],
    window: f32,
) -> Result<(Vec<(Spoken, AudioFeatures)>, usize)> {
    let mut kept = Vec::new();
    let mut clips = Vec::new();
    for item in items {
        let bytes = std::fs::read(dir.join(&item.file))
            .with_context(|| format!("reading {}", item.file))?;
        let clip =
            Clip::from_wav(&bytes).with_context(|| format!("{} is not a WAV file", item.file))?;
        if let Ok(padded) = padded_to(&clip, window) {
            kept.push((*item).clone());
            clips.push(padded);
        }
    }
    let refs: Vec<&Clip> = clips.iter().collect();
    let features = recognizer.features_many(&refs)?;
    Ok((
        kept.into_iter().zip(features).collect(),
        items.len() - clips.len(),
    ))
}

/// Train the projector as `plan` says and write it, with `test-chat.jsonl`
/// (the text path's records of the held-out items) and `report.json`, to `out`.
pub fn train(plan: &Plan, answers: &BTreeMap<String, Answer>, out: &Path) -> Result<Report> {
    let set: SpokenSet = serde_json::from_str(
        &std::fs::read_to_string(plan.set_dir.join("set.json")).context("reading set.json")?,
    )?;
    let groups = held_out_groups(&set, plan.test_every);
    let split = set.split(&groups, &plan.test_voices);
    if (split.train.is_empty() && plan.steps > 0) || split.test.is_empty() {
        bail!(
            "the split leaves {} recordings to train on and {} to test on",
            split.train.len(),
            split.test.len()
        );
    }
    std::fs::create_dir_all(out)?;

    let recognizer = BrainRecognizer::load(&plan.recognizer)?;
    let (train_feats, long_train) = features_of(
        &recognizer,
        &plan.set_dir,
        &split.train,
        plan.window_seconds,
    )?;
    let (test_feats, long_test) =
        features_of(&recognizer, &plan.set_dir, &split.test, plan.window_seconds)?;
    let too_long = long_train + long_test;
    drop(recognizer);
    let rows = train_feats
        .first()
        .or(test_feats.first())
        .map(|(_, f)| f.rows)
        .context("no recording fits the window")?;

    let mut ingress = Ingress::load(&IngressOptions {
        base: plan.base.clone(),
        adapter: plan.adapter.clone(),
        rows,
        block: plan.block,
        seed: plan.seed,
    })?;
    if let Some(init) = &plan.init_projector {
        ingress.load_projector(init)?;
    }
    let answer_of = |item: &Spoken| {
        answers
            .get(&item.represents)
            .with_context(|| format!("no answer for question {:?}", item.represents))
    };
    let mut train_ex: Vec<IngressExample> = Vec::new();
    let mut rng = plan.seed ^ 0x5EED;
    let mut draw = move || {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (rng >> 33) as f32 / (1u64 << 31) as f32
    };
    for (item, f) in &train_feats {
        let a = answer_of(item)?;
        train_ex.push(ingress.example(f, &a.system, "", &a.answer)?);
        if draw() < plan.transcribe_share {
            train_ex.push(ingress.example(f, &a.system, TRANSCRIBE, &item.text)?);
        }
    }
    let mut test_ex = Vec::new();
    let mut chat = String::new();
    for (item, f) in &test_feats {
        let a = answer_of(item)?;
        test_ex.push(ingress.example(f, &a.system, "", &a.answer)?);
        chat.push_str(&serde_json::json!({"messages": [
            {"role": "system", "content": a.system, "train": false},
            {"role": "user", "content": item.text, "train": false},
            {"role": "assistant", "content": format!("<think>\n\n</think>\n\n{}", a.answer), "train": true}
        ]}).to_string());
        chat.push('\n');
    }
    std::fs::write(out.join("test-chat.jsonl"), chat)?;

    let mean = |ingress: &mut Ingress, ex: &[IngressExample]| {
        ex.iter().map(|e| ingress.loss(e)).sum::<f32>() / ex.len() as f32
    };
    let probe = &test_ex[..test_ex.len().min(60)];
    let test_loss_before = mean(&mut ingress, probe);
    let (mut curve, mut recent) = (Vec::new(), Vec::new());
    let mut queue: Vec<usize> = Vec::new();
    for step in 1..=plan.steps {
        let progress = step as f32 / plan.steps as f32;
        let warm = (step as f32 / (plan.steps as f32 * 0.05).max(1.0)).min(1.0);
        let lr =
            plan.lr * warm * (0.1 + 0.9 * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos()));
        let mut batch = Vec::new();
        while batch.len() < plan.batch {
            if queue.is_empty() {
                queue = (0..train_ex.len()).collect();
                for i in (1..queue.len()).rev() {
                    queue.swap(i, (draw() * (i as f32 + 1.0)) as usize % (i + 1));
                }
            }
            match queue.pop() {
                Some(i) => batch.push(&train_ex[i]),
                None => bail!("there are no training examples"),
            }
        }
        let loss = ingress.step(
            &batch,
            step,
            &StepSettings {
                projector_lr: lr,
                model_lr: 0.0,
                weight_decay: 0.0,
                grad_clip: 1.0,
            },
        );
        recent.push(loss);
        if step % 10 == 0 || step == plan.steps {
            eprintln!("step {step}/{} loss {loss:.3} lr {lr:.2e}", plan.steps);
        }
        if step % 50 == 0 && step != plan.steps {
            curve.push((step, mean(&mut ingress, probe)));
        }
    }
    let test_loss_after = mean(&mut ingress, &test_ex);
    curve.push((plan.steps, test_loss_after));
    ingress.save_projector(&out.join("projector.safetensors"))?;
    let last = &recent[recent.len().saturating_sub(10)..];
    let report = Report {
        train_recordings: train_feats.len(),
        test_recordings: test_feats.len(),
        dropped: split.dropped,
        too_long,
        test_loss_before,
        test_loss_after,
        test_loss_curve: curve,
        train_loss_last: (!last.is_empty()).then(|| last.iter().sum::<f32>() / last.len() as f32),
        steps: plan.steps,
    };
    std::fs::write(
        out.join("report.json"),
        serde_json::to_string_pretty(&report)?,
    )?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use splinter_sdk::vocabulary::digest::Digest;
    use splinter_sdk::vocabulary::speech::Portrayal;

    fn set_of_groups(n: usize) -> SpokenSet {
        let items = (0..n)
            .map(|i| Spoken {
                id: format!("r{i}"),
                represents: format!("q{i}"),
                group: format!("g{i:02}"),
                text: "x".into(),
                file: "x.wav".into(),
                audio: Digest::of(b"x"),
                seed: 1,
                word_error_rate: 0.0,
            })
            .collect();
        SpokenSet::new("s", Portrayal::synthetic_theatrical(), items).unwrap()
    }

    #[test]
    fn every_kth_group_by_name_is_held_out() {
        let held = held_out_groups(&set_of_groups(10), 4);
        assert_eq!(
            held.iter().map(String::as_str).collect::<Vec<_>>(),
            ["g03", "g07"]
        );
        assert!(held_out_groups(&set_of_groups(10), 0).is_empty());
    }

    #[test]
    fn answers_are_read_by_question_and_an_empty_one_is_refused() {
        let ok = read_answers("{\"id\":\"q1\",\"system\":\"s\",\"answer\":\"a\"}\n\n").unwrap();
        assert_eq!(ok["q1"].answer, "a");
        assert!(read_answers("{\"id\":\"q1\",\"system\":\"s\",\"answer\":\" \"}").is_err());
    }
}
