// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements repeatable evaluation of locally run
// language models for its clients. If your team needs expertise in measuring
// what a fine-tune changed, you can procure our services by sending an email
// to info@swedishembedded.com.

//! One served model answering one question at a time, greedily.
//!
//! An evaluation asks the same model twice: with no adapter, then with the one
//! training produced. Greedy decoding means a question has one answer for a
//! given set of weights, so a difference between the two is the weights'.

use std::time::Instant;

use futures::StreamExt;
use sven_sdk::model::{
    CompletionRequest, Message, MessageContent, ModelProvider, ResponseEvent, Role,
};

use crate::error::PolicyError;
use std::sync::Arc;

use crate::endpoint::BrainEndpoints;
use crate::local::{LocalWeights, Sampling, GREEDY_SAMPLING};
use crate::selection::{LoadedModel, ModelSelection};

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
///
/// A sampled decoding with a `seed` repeats: the same question on the same
/// weights gives the same reply. Without one, the provider draws a seed for
/// each request and the run that made the request records it.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum Decoding {
    /// Argmax, reasoning off.
    #[default]
    Greedy,
    /// Argmax, with the model reasoning before it answers.
    Thinking,
    /// Sampled at this temperature, reasoning off.
    Sampled {
        /// The softmax temperature.
        temperature: f32,
        /// The seed of the draws, if the caller fixes one.
        seed: Option<u64>,
    },
    /// Sampled at this temperature, with the model reasoning first.
    SampledThinking {
        /// The softmax temperature.
        temperature: f32,
        /// The seed of the draws, if the caller fixes one.
        seed: Option<u64>,
    },
}

/// The default temperature of the sampled decodings.
const DEFAULT_SAMPLE_TEMPERATURE: f32 = 0.6;

impl Decoding {
    /// The sampling this decoding stands for.
    #[must_use]
    pub fn sampling(self) -> Sampling {
        let (temperature, thinking, seed) = match self {
            Self::Greedy => (0.0, false, None),
            Self::Thinking => (0.0, true, None),
            Self::Sampled { temperature, seed } => (temperature, false, seed),
            Self::SampledThinking { temperature, seed } => (temperature, true, seed),
        };
        Sampling {
            temperature,
            thinking,
            seed,
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
            ("sample", _) => Ok(Self::Sampled {
                temperature: at,
                seed: None,
            }),
            ("sample-thinking", _) => Ok(Self::SampledThinking {
                temperature: at,
                seed: None,
            }),
            _ => Err(format!(
                "unknown decoding {s:?}: use greedy, thinking, sample[:T] or sample-thinking[:T]"
            )),
        }
    }
}

/// A loaded model, with its adapter if it has one.
pub struct Answerer {
    model: LoadedModel,
    provider: Arc<dyn ModelProvider>,
}

impl Answerer {
    /// The model `weights` names, requested from the brain `endpoints` that
    /// serve it, decoding as `decoding` says. `label` names the model in the
    /// provider seam's records.
    ///
    /// # Errors
    /// No endpoint serves the checkpoint (or the adapter, which is offered
    /// when the endpoints follow a manifest), or an endpoint is unreachable.
    pub fn request(
        endpoints: &BrainEndpoints,
        weights: &LocalWeights,
        decoding: Decoding,
    ) -> Result<Self, PolicyError> {
        let model = ModelSelection::Local(weights.clone()).load_served(endpoints)?;
        let provider = model
            .resampled(decoding.sampling())?
            .unwrap_or_else(|| model.provider());
        Ok(Self { model, provider })
    }

    /// The model as a sven model provider, for an agent to run on. It sends
    /// its requests to the same endpoints, so the answerer must outlive
    /// nothing: the provider holds what it needs.
    #[must_use]
    pub fn provider(&self) -> Arc<dyn ModelProvider> {
        Arc::clone(&self.provider)
    }

    /// [`Self::provider`], decoding as `decoding` says instead of as the
    /// answerer was requested.
    ///
    /// # Errors
    /// The endpoint the decoding's seed pins it to cannot be reached.
    pub fn provider_with(&self, decoding: Decoding) -> Result<Arc<dyn ModelProvider>, PolicyError> {
        Ok(self
            .model
            .resampled(decoding.sampling())?
            .unwrap_or_else(|| self.model.provider()))
    }

    /// The model's reply to `user` under the system message `system`, within
    /// `max_tokens` generated tokens.
    ///
    /// # Errors
    /// The generation failed on the device.
    pub async fn ask(&self, system: &str, user: &str, max_tokens: u32) -> anyhow::Result<Reply> {
        self.ask_streaming(system, user, max_tokens, &mut |_| {})
            .await
    }

    /// [`Self::ask`], decoding as `decoding` says instead of as the answerer
    /// was loaded: the weights are shared, so many draws under seeds of the
    /// caller's choosing load nothing again, and the same seed draws the same
    /// reply.
    ///
    /// # Errors
    /// The generation failed on the device.
    pub async fn ask_decoding(
        &self,
        decoding: Decoding,
        system: &str,
        user: &str,
        max_tokens: u32,
    ) -> anyhow::Result<Reply> {
        let model = self.provider_with(decoding).map_err(anyhow::Error::from)?;
        Self::complete(&*model, system, user, max_tokens, |_| {}).await
    }

    /// The model's reply to a conversation of `turns` (role, text) pairs, roles `system`, `user` or `assistant`, decoding
    /// as `decoding` says: how a later step of a multi-step conversation is asked in the context of the earlier ones.
    ///
    /// # Errors
    /// A role that is none of the three, or the generation failed on the device.
    pub async fn ask_turns(
        &self,
        decoding: Decoding,
        turns: &[(&str, &str)],
        max_tokens: u32,
    ) -> anyhow::Result<Reply> {
        let mut messages = Vec::with_capacity(turns.len());
        for (role, text) in turns {
            let role = match *role {
                "system" => Role::System,
                "user" => Role::User,
                "assistant" => Role::Assistant,
                other => anyhow::bail!("{other:?} is not a role: system, user or assistant"),
            };
            messages.push(Message {
                role,
                content: MessageContent::Text((*text).to_string()),
                reasoning: None,
            });
        }
        let model = self.provider_with(decoding).map_err(anyhow::Error::from)?;
        Self::complete_messages(&*model, messages, max_tokens, |_| {}).await
    }

    /// [`Self::ask`], handing `on_text` each piece of the reply as it is
    /// written.
    ///
    /// # Errors
    /// The generation failed on the device.
    pub async fn ask_streaming(
        &self,
        system: &str,
        user: &str,
        max_tokens: u32,
        on_text: &mut dyn FnMut(&str),
    ) -> anyhow::Result<Reply> {
        Self::complete(&*self.provider, system, user, max_tokens, |piece| {
            on_text(piece)
        })
        .await
    }

    /// Generic in the callback so that a caller handing a `Send` one gets a
    /// `Send` future (a search drives it from sven's `Send` proposer).
    async fn complete(
        model: &dyn ModelProvider,
        system: &str,
        user: &str,
        max_tokens: u32,
        on_text: impl FnMut(&str),
    ) -> anyhow::Result<Reply> {
        let message = |role, text: &str| Message {
            role,
            content: MessageContent::Text(text.to_string()),
            reasoning: None,
        };
        let messages = vec![message(Role::System, system), message(Role::User, user)];
        Self::complete_messages(model, messages, max_tokens, on_text).await
    }

    async fn complete_messages(
        model: &dyn ModelProvider,
        messages: Vec<Message>,
        max_tokens: u32,
        mut on_text: impl FnMut(&str),
    ) -> anyhow::Result<Reply> {
        let request = CompletionRequest {
            messages,
            max_output_tokens_override: Some(max_tokens),
            ..CompletionRequest::default()
        };
        let started = Instant::now();
        let mut stream = model.complete(request).await?;
        let mut reply = Reply::default();
        while let Some(event) = stream.next().await {
            match event? {
                ResponseEvent::TextDelta(delta) => {
                    on_text(&delta);
                    reply.text.push_str(&delta);
                }
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
        let sampled = |temperature| Decoding::Sampled {
            temperature,
            seed: None,
        };
        assert_eq!("sample".parse(), Ok(sampled(0.6)));
        assert_eq!("sample:0.3".parse(), Ok(sampled(0.3)));
        assert_eq!(
            "sample-thinking:0.7".parse(),
            Ok(Decoding::SampledThinking {
                temperature: 0.7,
                seed: None
            })
        );
        for bad in ["greedy:0.5", "sample:0", "sample:hot", "think", ""] {
            assert!(bad.parse::<Decoding>().is_err(), "{bad:?}");
        }
        assert_eq!(Decoding::Greedy.sampling(), GREEDY_SAMPLING);
        let thinking = Decoding::Thinking.sampling();
        assert!(thinking.thinking && thinking.temperature == 0.0);
        let sampling = sampled(0.6).sampling();
        assert!(!sampling.thinking && sampling.temperature == 0.6 && sampling.seed.is_none());
        let seed = crate::seed::draw().value();
        let seeded = Decoding::Sampled {
            temperature: 0.6,
            seed: Some(seed),
        };
        assert_eq!(seeded.sampling().seed, Some(seed));
    }
}
