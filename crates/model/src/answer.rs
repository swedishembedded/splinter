// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements repeatable evaluation of locally run
// language models for its clients. If your team needs expertise in measuring
// what a fine-tune changed, you can procure our services by sending an email
// to info@swedishembedded.com.

//! One local model answering one question at a time, greedily.
//!
//! An evaluation asks the same model twice: with no adapter, then with the one
//! training produced. Greedy decoding means a question has one answer for a
//! given set of weights, so a difference between the two is the weights'.

use std::path::Path;
use std::time::Instant;

use futures::StreamExt;
use sven_sdk::model::{
    CompletionRequest, Message, MessageContent, ModelProvider, ResponseEvent, Role,
};

use crate::error::PolicyError;
use crate::local::{LocalQwen, LocalWeights, Sampling, GREEDY_SAMPLING};
use crate::residency::Residency;

/// What a model said to one question.
#[derive(Clone, Debug, Default)]
pub struct Reply {
    /// Its reasoning, when it reasoned.
    pub thinking: String,
    /// The visible reply after its reasoning.
    pub text: String,
    /// Tokens it generated.
    pub tokens: u32,
    /// Whether it ran out of its token budget.
    pub truncated: bool,
    /// Wall-clock seconds the answer took.
    pub seconds: f64,
}

/// How a model decodes its answers.
///
/// Greedy is the default for measurement: a question then has one answer for
/// a given set of weights. A reasoning model that is only ever asked greedily
/// with its reasoning off is handicapped, so an exam can also let it think,
/// or sample, to give a baseline its due.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum Decoding {
    /// Argmax, reasoning off.
    #[default]
    Greedy,
    /// Argmax, with the model reasoning before it answers.
    Thinking,
    /// Sampled at this temperature, reasoning off.
    Sampled(f32),
    /// Sampled at this temperature, with the model reasoning first.
    SampledThinking(f32),
}

/// The default temperature of the sampled decodings.
const DEFAULT_SAMPLE_TEMPERATURE: f32 = 0.6;

impl Decoding {
    /// The sampling this decoding stands for.
    #[must_use]
    pub fn sampling(self) -> Sampling {
        let (temperature, thinking) = match self {
            Self::Greedy => (0.0, false),
            Self::Thinking => (0.0, true),
            Self::Sampled(t) => (t, false),
            Self::SampledThinking(t) => (t, true),
        };
        Sampling {
            temperature,
            thinking,
            ..GREEDY_SAMPLING
        }
    }
}

impl std::str::FromStr for Decoding {
    type Err = String;

    /// `greedy`, `thinking`, `sample[:T]` or `sample-thinking[:T]`.
    fn from_str(s: &str) -> Result<Self, String> {
        let (name, temperature) = match s.split_once(':') {
            Some((name, t)) => (
                name,
                Some(
                    t.parse::<f32>()
                        .ok()
                        .filter(|t| t.is_finite() && *t > 0.0)
                        .ok_or_else(|| format!("{t:?} is not a positive temperature"))?,
                ),
            ),
            None => (s, None),
        };
        let at = temperature.unwrap_or(DEFAULT_SAMPLE_TEMPERATURE);
        match (name, temperature) {
            ("greedy", None) => Ok(Self::Greedy),
            ("thinking", None) => Ok(Self::Thinking),
            ("sample", _) => Ok(Self::Sampled(at)),
            ("sample-thinking", _) => Ok(Self::SampledThinking(at)),
            _ => Err(format!(
                "unknown decoding {s:?}: use greedy, thinking, sample[:T] or sample-thinking[:T]"
            )),
        }
    }
}

/// A loaded model, with its adapter if it has one.
pub struct Answerer {
    model: LocalQwen,
    _residency: Residency,
}

impl Answerer {
    /// Loads the base at `base`, attaching `adapter` when given. `label` names
    /// the model in the provider seam's records.
    ///
    /// # Errors
    /// The checkpoint cannot be opened or the adapter does not fit it.
    pub fn load(
        base: &Path,
        adapter: Option<&Path>,
        context_tokens: u32,
        label: &str,
    ) -> Result<Self, PolicyError> {
        Self::load_with(base, adapter, context_tokens, label, Decoding::Greedy)
    }

    /// [`Self::load`], decoding as `decoding` says.
    ///
    /// # Errors
    /// The checkpoint cannot be opened or the adapter does not fit it.
    pub fn load_with(
        base: &Path,
        adapter: Option<&Path>,
        context_tokens: u32,
        label: &str,
        decoding: Decoding,
    ) -> Result<Self, PolicyError> {
        let residency = Residency::default();
        let weights = LocalWeights {
            base: base.to_path_buf(),
            adapter: adapter.map(Path::to_path_buf),
            context_tokens,
        };
        let model = LocalQwen::load(&residency, &weights, label)?.resampled(decoding.sampling());
        Ok(Self {
            model,
            _residency: residency,
        })
    }

    /// The model's reply to `user` under the system message `system`, within
    /// `max_tokens` generated tokens.
    ///
    /// # Errors
    /// The generation failed on the device.
    pub async fn ask(&self, system: &str, user: &str, max_tokens: u32) -> anyhow::Result<Reply> {
        let message = |role, text: &str| Message {
            role,
            content: MessageContent::Text(text.to_string()),
        };
        let request = CompletionRequest {
            messages: vec![message(Role::System, system), message(Role::User, user)],
            max_output_tokens_override: Some(max_tokens),
            ..CompletionRequest::default()
        };
        let started = Instant::now();
        let mut stream = self.model.complete(request).await?;
        let mut reply = Reply::default();
        while let Some(event) = stream.next().await {
            match event? {
                ResponseEvent::TextDelta(delta) => reply.text.push_str(&delta),
                ResponseEvent::ThinkingDelta(delta) => reply.thinking.push_str(&delta),
                ResponseEvent::MaxTokens => reply.truncated = true,
                ResponseEvent::Usage { output_tokens, .. } => reply.tokens = output_tokens,
                ResponseEvent::Error(e) => anyhow::bail!("generation failed: {e}"),
                _ => {}
            }
        }
        reply.seconds = started.elapsed().as_secs_f64();
        Ok(reply)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_decoding_is_read_from_its_name_and_says_whether_the_model_reasons_and_samples() {
        assert_eq!("greedy".parse(), Ok(Decoding::Greedy));
        assert_eq!("thinking".parse(), Ok(Decoding::Thinking));
        assert_eq!("sample".parse(), Ok(Decoding::Sampled(0.6)));
        assert_eq!("sample:0.3".parse(), Ok(Decoding::Sampled(0.3)));
        assert_eq!(
            "sample-thinking:0.7".parse(),
            Ok(Decoding::SampledThinking(0.7))
        );
        for bad in ["greedy:0.5", "sample:0", "sample:hot", "think", ""] {
            assert!(bad.parse::<Decoding>().is_err(), "{bad:?}");
        }
        assert_eq!(Decoding::Greedy.sampling(), GREEDY_SAMPLING);
        let thinking = Decoding::Thinking.sampling();
        assert!(thinking.thinking && thinking.temperature == 0.0);
        let sampled = Decoding::Sampled(0.6).sampling();
        assert!(!sampled.thinking && sampled.temperature == 0.6);
    }
}
