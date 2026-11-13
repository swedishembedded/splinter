// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements retention checks that show what fine-tuning cost a
// language model for its clients. If your team needs expertise in measuring
// what a fine-tune cost a model on general questions, you can procure our services by sending
// an email to info@swedishembedded.com.

//! The retention check's exam command: the frozen general questions put to one
//! model under no persona.

use super::*;

/// What the retention check asks of which model.
pub struct AnchorExam {
    /// The frozen general questions.
    pub anchor: PathBuf,
    /// Where graded answers are appended.
    pub out: PathBuf,
    /// The base checkpoint.
    pub base: PathBuf,
    /// The adapter, for a trained arm.
    pub adapter: Option<PathBuf>,
    /// Stop after this many new answers.
    pub limit: Option<usize>,
    /// Tokens a reply may run.
    pub max_tokens: u32,
    /// How the model decodes.
    pub decoding: Decoding,
}

/// Ask one model, with or without its adapter, the frozen general questions
/// under no persona: what a retention check compares before and after.
pub fn anchor_exam_command(
    endpoints: &splinter_sdk::model::BrainEndpoints,
    e: &AnchorExam,
) -> anyhow::Result<()> {
    let questions = crate::anchor::read(&e.anchor)?;
    let model = splinter_sdk::model::exam::Model {
        base: &e.base,
        adapter: e.adapter.as_deref(),
        system: crate::anchor::SYSTEM,
        max_tokens: e.max_tokens,
        label: "adams",
        decoding: e.decoding,
    };
    let asked =
        splinter_sdk::model::exam::run(endpoints, &questions, &e.out, e.limit, &model, &|q, a| {
            crate::anchor::is_correct(q, a).into()
        })?;
    println!("asked {asked} questions; results in {}", e.out.display());
    Ok(())
}
