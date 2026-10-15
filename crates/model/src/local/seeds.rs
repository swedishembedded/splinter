// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible training and evaluation
// pipelines for its clients. If your team needs expertise in seeding,
// reproducibility and experiment provenance for sampled decoding then you can
// procure our services by sending an email to info@swedishembedded.com.

//! The seeds a local model drew for the sampled requests that named none.
//!
//! sven's response events carry no seed, so a drawn seed cannot travel with
//! the completion it produced. The provider keeps them here, in the order
//! the requests were made, and the run that owns the provider drains them
//! into its record: passing a seed back as [`Sampling::seed`] replays that
//! request's decoding.

use std::sync::{Arc, Mutex, PoisonError};

use splinter_core::rng::Seed;

use super::request::Sampling;

/// The seeds drawn for requests that named none, shared by every handle on
/// one model.
#[derive(Clone, Debug, Default)]
pub(super) struct DrawnSeeds(Arc<Mutex<Vec<Seed>>>);

impl DrawnSeeds {
    /// `sampling` with its seed resolved for one request: a seed it names
    /// stays; an unnamed seed under a positive temperature is drawn and
    /// recorded. A greedy request is left unseeded: brain's sampler takes
    /// the argmax at temperature zero before it touches its random stream,
    /// so no seed can change the reply and none is invented.
    pub(super) fn resolve(&self, sampling: Sampling) -> Sampling {
        if sampling.seed.is_some() || sampling.temperature <= 0.0 {
            return sampling;
        }
        let seed = crate::seed::draw();
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(seed);
        Sampling {
            seed: Some(seed.value()),
            ..sampling
        }
    }

    /// Every seed drawn since the last call, oldest first.
    pub(super) fn take(&self) -> Vec<Seed> {
        std::mem::take(&mut *self.0.lock().unwrap_or_else(PoisonError::into_inner))
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Arc;

    use brain::chat::{ChatDelta, ChatUsage, FinishReason};
    use brain::{CancelToken, ChatRequest, ChatResponse};
    use futures::StreamExt;
    use sven_sdk::model::{
        CompletionRequest, Message, MessageContent, ModelProvider, ResponseEvent, Role,
    };

    use crate::error::PolicyError;
    use crate::local::{LocalQwen, LocalWeights, Sampling, AGENT_SAMPLING, GREEDY_SAMPLING};
    use crate::residency::{BaseLoader, Engine, Residency};

    /// A base whose reply is a function of the seed its request carries and
    /// nothing else, as a sampled decode is: `seed:<n>` for a seeded
    /// request, `argmax` for an unseeded one.
    struct SeedEcho;

    impl Engine for SeedEcho {
        fn attach_adapter(&mut self, _adapter: &Path) -> Result<(), String> {
            Ok(())
        }

        fn detach_adapter(&mut self) -> Result<(), String> {
            Ok(())
        }

        fn generate_stream(
            &self,
            request: &ChatRequest,
            _cancel: &CancelToken,
            on_delta: &mut dyn FnMut(ChatDelta),
        ) -> Result<ChatResponse, String> {
            // ChatRequest exposes no accessor for its sampling; its debug
            // form is the only view of the seed it carries.
            let shown = format!("{request:?}");
            let text = shown
                .split("seed: Some(")
                .nth(1)
                .and_then(|rest| rest.split(')').next())
                .map_or_else(|| "argmax".to_string(), |seed| format!("seed:{seed}"));
            on_delta(ChatDelta::Text(text.clone()));
            Ok(ChatResponse {
                text,
                reasoning: String::new(),
                tool_calls: Vec::new(),
                finish_reason: FinishReason::Stop,
                usage: ChatUsage::default(),
            })
        }
    }

    struct SeedEchoLoader;

    impl BaseLoader for SeedEchoLoader {
        fn load(&self, _: &Path, _: u32) -> Result<Box<dyn Engine>, PolicyError> {
            Ok(Box::new(SeedEcho))
        }
    }

    fn model(dir: &tempfile::TempDir, sampling: Sampling) -> LocalQwen {
        std::fs::write(dir.path().join("model.safetensors"), b"scripted").unwrap();
        let weights = LocalWeights {
            base: dir.path().to_path_buf(),
            adapter: None,
            context_tokens: Some(4096),
        };
        let residency = Residency::new(Arc::new(SeedEchoLoader));
        LocalQwen::load(&residency, &weights, "m")
            .unwrap()
            .resampled(sampling)
    }

    async fn reply(model: &LocalQwen) -> String {
        let request = CompletionRequest {
            messages: vec![Message {
                role: Role::User,
                content: MessageContent::Text("hello".into()),
            }],
            ..CompletionRequest::default()
        };
        let mut stream = model.complete(request).await.unwrap();
        let mut text = String::new();
        while let Some(event) = stream.next().await {
            match event.unwrap() {
                ResponseEvent::TextDelta(delta) => text.push_str(&delta),
                ResponseEvent::Error(e) => panic!("generation failed: {e}"),
                _ => {}
            }
        }
        text
    }

    /// Two sampled requests that name no seed are given different seeds,
    /// each reported in request order; a request replayed with a reported
    /// seed decodes to the completion that seed produced.
    #[tokio::test]
    async fn unseeded_sampled_requests_are_given_distinct_reported_seeds() {
        let dir = tempfile::tempdir().unwrap();
        let sampled = model(&dir, AGENT_SAMPLING);
        let first = reply(&sampled).await;
        let second = reply(&sampled).await;
        let drawn = sampled.take_drawn_seeds();
        assert_eq!(drawn.len(), 2, "one seed per request: {drawn:?}");
        assert_ne!(drawn[0], drawn[1]);
        assert_eq!(first, format!("seed:{}", drawn[0]));
        assert_eq!(second, format!("seed:{}", drawn[1]));
        assert!(sampled.take_drawn_seeds().is_empty(), "reported once");

        let replay = model(
            &dir,
            Sampling {
                seed: Some(drawn[0].value()),
                ..AGENT_SAMPLING
            },
        );
        assert_eq!(reply(&replay).await, first);
        assert!(
            replay.take_drawn_seeds().is_empty(),
            "a named seed is not a draw"
        );
    }

    /// A greedy request has no random draw to record: it carries no seed.
    #[tokio::test]
    async fn greedy_requests_are_not_seeded() {
        let dir = tempfile::tempdir().unwrap();
        let greedy = model(&dir, GREEDY_SAMPLING);
        assert_eq!(reply(&greedy).await, "argmax");
        assert!(greedy.take_drawn_seeds().is_empty());
    }
}
