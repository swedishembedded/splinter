// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measurements of whether a fine-tuned model
// took on a writer's own diction without a judge, for its clients. If your
// team needs expertise in measuring a model's voice by the likelihood it gives
// a writer's held-out text, you can procure our services by sending an email
// to info@swedishembedded.com.

//! The voice metric: how likely each arm finds the writer's own held-out
//! text, needing no judge.
//!
//! The text is that of the reserved families, in the shape the policy is
//! trained and asked in (the voice view: one request, the writer's words as
//! the answer), scored by brain's own held-out scoring path as the mean
//! per-token loss over the answer. A model that took on the writer's diction
//! gives the writer's unseen text a lower loss than the base does, and than
//! the base told whose voice to take. Nothing is generated and no answer is
//! graded, so what a judge's blind spots or a long answer's weight do to the
//! other measures does not touch this one.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use splinter_core::digest::Digest;
use splinter_data::{Corpus, RecordBody, View, Voice};
use splinter_knowledge::sections::sections;
use splinter_model::train::score_chat;
use splinter_model::TokenCounter;

use crate::exam_set::ExamSet;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::{io, OrchestratorError};

/// The most records of the writer's text one arm is scored on.
pub const MAX_VOICE_RECORDS: usize = 300;

/// One arm of the voice metric: which weights, asked under which prompt.
pub struct VoiceArm<'a> {
    /// The arm's name.
    pub name: &'a str,
    /// The adapter folded into the base; `None` is the base alone.
    pub adapter: Option<&'a Path>,
    /// The system prompt the text is shown under; `None` is the default.
    pub system: Option<&'a str>,
}

/// One arm's score.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct VoiceScore {
    /// The arm.
    pub arm: String,
    /// Mean per-token cross-entropy over the writer's words; lower is
    /// likelier.
    pub loss: Option<f32>,
    /// `exp(loss)`.
    pub perplexity: Option<f64>,
    /// The share of positions whose most likely next token was the writer's.
    pub token_accuracy: Option<f64>,
    /// Positions scored.
    pub positions: usize,
    /// Records scored.
    pub records: usize,
}

/// The records of the exam's text, one JSON line each, without the parts in
/// `skip` (text a model was trained on). Written to `file`; the count of
/// records, at most [`MAX_VOICE_RECORDS`], evenly spread.
fn write_records(
    ctx: &Context,
    exam: &ExamSet,
    skip: &BTreeSet<Digest>,
    system: Option<&str>,
    file: &Path,
) -> Result<usize, OrchestratorError> {
    let counter = TokenCounter::for_model(&ctx.config().policy_base).ok();
    let tokens = |text: &str| {
        counter
            .as_ref()
            .map_or_else(|| splinter_data::chars_as_tokens(text), |c| c.count(text))
    };
    let sectioner = |text: &str, media_type: &str| {
        sections(text, media_type)
            .into_iter()
            .map(|section| section.range)
            .collect()
    };
    let sources = ctx.sources();
    let mut corpus = Corpus::new();
    for source in &exam.sources {
        corpus.add_source(source.clone());
    }
    let mut projection = Voice::new(&sources)
        .measured_by(&tokens)
        .sectioned_by(&sectioner)
        .project(&corpus)?;
    projection
        .records
        .retain(|r| !r.metadata.sources.iter().any(|s| skip.contains(s)));
    let projection = projection.thinned_to(MAX_VOICE_RECORDS);
    let projection = match system {
        Some(prompt) => projection.with_system_prompt(prompt),
        None => projection,
    };
    let mut text = String::new();
    for record in &projection.records {
        if let RecordBody::Chat { messages } = &record.body {
            let line = serde_json::json!({ "messages": messages, "tools": [] });
            text.push_str(&line.to_string());
            text.push('\n');
        }
    }
    std::fs::write(file, &text).map_err(io(file))?;
    Ok(projection.records.len())
}

/// Scores each of `arms` on the exam's text, leaving out the parts in
/// `skip`. The files are written under `scratch` and not kept.
///
/// # Errors
/// Refused when no text is left to score; scoring failures as they come.
pub fn score(
    ctx: &Context,
    exam: &ExamSet,
    skip: &BTreeSet<Digest>,
    arms: &[VoiceArm<'_>],
    scratch: &Path,
) -> Result<Vec<VoiceScore>, OrchestratorError> {
    std::fs::create_dir_all(scratch).map_err(io(scratch))?;
    let base = &ctx.config().policy_base;
    let mut scores = Vec::with_capacity(arms.len());
    for arm in arms {
        let file: PathBuf = scratch.join(format!("voice-{}.jsonl", arm.name));
        if write_records(ctx, exam, skip, arm.system, &file)? == 0 {
            return Err(OrchestratorError::Refused(
                "no text of the exam is left to score the voice on".into(),
            ));
        }
        let scored = score_chat(base, arm.adapter, &file)
            .map_err(|e| OrchestratorError::Refused(e.to_string()));
        std::fs::remove_file(&file).map_err(io(&file))?;
        let scored = scored?;
        scores.push(VoiceScore {
            arm: arm.name.into(),
            loss: scored.loss,
            perplexity: scored.loss.map(|l| f64::from(l).exp()),
            token_accuracy: scored.token_accuracy,
            positions: scored.positions,
            records: scored.records,
        });
    }
    Ok(scores)
}
