// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A persona that listens: recogniser features, a trained projector and the
//! language model (with its persona adapter) answering the projected rows.

use std::path::PathBuf;

use brain::{CancelToken, ChatPipeline, SpeechIngress, SpeechProjector};

use super::{padded_to, BrainRecognizer, Clip, Listener};
use crate::error::PolicyError;

/// What to load.
#[derive(Clone, Debug)]
pub struct ListenerOptions {
    /// The recogniser whose encoder gives the features (Qwen3-ASR).
    pub recognizer: String,
    /// The trained projector.
    pub projector: PathBuf,
    /// The language model's checkpoint directory.
    pub base: PathBuf,
    /// The persona adapter attached to it.
    pub adapter: Option<PathBuf>,
    /// The system turn the persona answers under.
    pub system: String,
    /// The window every clip is padded to, in seconds.
    pub window_seconds: f32,
    /// The longest answer, in tokens.
    pub max_new: usize,
}

/// A language model that hears speech.
pub struct BrainListener {
    recognizer: BrainRecognizer,
    projector: SpeechProjector,
    chat: ChatPipeline,
    opts: ListenerOptions,
}

fn failed(reason: impl std::fmt::Display) -> PolicyError {
    PolicyError::Transcription {
        reason: reason.to_string(),
    }
}

impl BrainListener {
    /// Load the recogniser, the projector and the model with its adapter.
    pub fn load(opts: &ListenerOptions) -> Result<Self, PolicyError> {
        let recognizer = BrainRecognizer::load(&opts.recognizer)?;
        let projector = SpeechProjector::load(&opts.projector).map_err(failed)?;
        let base = opts
            .base
            .to_str()
            .ok_or_else(|| failed(format!("{} is not UTF-8", opts.base.display())))?;
        let mut chat = ChatPipeline::from_pretrained(base).map_err(|e| PolicyError::Load {
            path: opts.base.clone(),
            reason: e.to_string(),
        })?;
        if let Some(adapter) = &opts.adapter {
            let path = adapter
                .to_str()
                .ok_or_else(|| failed(format!("{} is not UTF-8", adapter.display())))?;
            chat.attach_adapter(path)
                .map_err(|e| PolicyError::Adapter {
                    path: opts.base.clone(),
                    adapter: Some(adapter.clone()),
                    reason: e.to_string(),
                })?;
        }
        Ok(Self {
            recognizer,
            projector,
            chat,
            opts: opts.clone(),
        })
    }
}

impl Listener for BrainListener {
    fn answer(&self, clip: &Clip, on_text: &mut dyn FnMut(&str)) -> Result<String, PolicyError> {
        let features = self
            .recognizer
            .features(&padded_to(clip, self.opts.window_seconds)?)?;
        let rows = self.projector.project(&features.embeds).map_err(failed)?;
        self.chat
            .generate_with_rows(
                &SpeechIngress::prefix(&self.opts.system),
                &rows,
                &SpeechIngress::suffix(""),
                self.opts.max_new,
                &CancelToken::default(),
                |piece| on_text(piece),
            )
            .map(|text| text.trim().to_string())
            .map_err(|e| PolicyError::Generate {
                path: self.opts.base.clone(),
                adapter: self.opts.adapter.clone(),
                reason: e.to_string(),
            })
    }
}
