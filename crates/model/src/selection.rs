// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Which model a stage runs on, and the one name every record gives it.
//!
//! Local-first: the model Splinter trains is served in-process from local
//! weights; a model reached over an API is an explicit alternative, used
//! for comparison and development. Every stage - an agent attempt, an
//! exploration, an evaluation, a one-shot question - selects its model the
//! same way and records it under the same identity, so a manifest's
//! `model` answers "which weights produced this" whichever stage wrote it.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use sven_sdk::model::ModelProvider;

use crate::error::PolicyError;
use crate::local::{LocalQwen, LocalWeights, Sampling};
use crate::residency::Residency;

/// How long dropping a [`LoadedModel`] waits for a local generation still
/// running to stop before the process may exit.
pub const QUIESCE_GRACE: Duration = Duration::from_secs(60);

/// A model reached over an API, named `provider/model` the way sven's
/// configuration names it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RemoteModel {
    /// `provider/model`, e.g. `openrouter/z-ai/glm-5.3-flash`.
    pub spec: String,
    /// Endpoint override written into sven's model configuration; `None`
    /// keeps the endpoint sven's configuration already has for the provider.
    pub base_url: Option<String>,
    /// API key written into sven's model configuration; `None` keeps the key
    /// sven's configuration already has.
    pub api_key: Option<String>,
    /// Let a model served on this machine reason before it answers; off
    /// asks its server for no reasoning block. A model of another provider
    /// is not affected.
    pub thinking: bool,
}

impl RemoteModel {
    /// Writes this model into sven's configuration: provider, model name,
    /// endpoint and key.
    pub fn apply_to(&self, config: &mut sven_sdk::config::Config) -> Result<(), PolicyError> {
        let (provider, name) = self
            .spec
            .split_once('/')
            .ok_or_else(|| PolicyError::Remote {
                spec: self.spec.clone(),
                reason: "a remote model is provider/model".into(),
            })?;
        config.model.provider = provider.to_string();
        config.model.name = name.to_string();
        if self.base_url.is_some() {
            config.model.base_url = self.base_url.clone();
        }
        if self.api_key.is_some() {
            config.model.api_key = self.api_key.clone();
        }
        // A model served on this machine answers like the in-process one:
        // without a reasoning block that would spend the caller's output
        // budget before any answer or tool call is written.
        if provider == LOCAL_SERVER_PROVIDER && !self.thinking {
            let mut options = config
                .model
                .driver_options
                .as_object()
                .cloned()
                .unwrap_or_default();
            options.insert(
                "chat_template_kwargs".into(),
                serde_json::json!({ "enable_thinking": false }),
            );
            config.model.driver_options = serde_json::Value::Object(options);
        }
        Ok(())
    }
}

/// The provider name under which sven reaches a `brain serve` on this machine.
const LOCAL_SERVER_PROVIDER: &str = "brain";

/// The model one stage runs on.
#[derive(Clone, Debug)]
pub enum ModelSelection {
    /// Served in-process from local weights, with an optional adapter.
    Local(LocalWeights),
    /// Reached over an API.
    Remote(RemoteModel),
}

impl ModelSelection {
    /// The identity records carry: `brain/<org>/<model>`, plus
    /// `+<adapter>` when an adapter rides along, for a local model; the
    /// `provider/model` spec for a remote one.
    #[must_use]
    pub fn identity(&self) -> String {
        match self {
            Self::Remote(remote) => remote.spec.clone(),
            Self::Local(weights) => {
                let base = format!("brain/{}", local_model_name(&weights.base));
                match weights.adapter.as_deref().and_then(Path::file_stem) {
                    Some(stem) => format!("{base}+{}", stem.to_string_lossy()),
                    None => base,
                }
            }
        }
    }

    /// The local weights, when this selection is local.
    #[must_use]
    pub fn local(&self) -> Option<&LocalWeights> {
        match self {
            Self::Local(weights) => Some(weights),
            Self::Remote(_) => None,
        }
    }

    /// This selection loaded: a provider for it, known by its
    /// [`Self::identity`]. Local weights go on `residency`'s copy of their
    /// base, loaded there if no other model holds it - the expensive step,
    /// so load once per command, not per request.
    pub fn load(&self, residency: &Residency) -> Result<LoadedModel, PolicyError> {
        let identity = self.identity();
        match self {
            Self::Remote(remote) => {
                let unreachable = |e: anyhow::Error| PolicyError::Remote {
                    spec: remote.spec.clone(),
                    reason: format!("{e:#}"),
                };
                let mut config = sven_sdk::config::load(None).map_err(unreachable)?;
                remote.apply_to(&mut config)?;
                let driver = sven_sdk::drivers::from_config(&config.model).map_err(unreachable)?;
                Ok(LoadedModel {
                    provider: Arc::from(driver),
                    identity,
                    local: None,
                })
            }
            Self::Local(weights) => {
                let local = Arc::new(LocalQwen::load(
                    residency,
                    weights,
                    &local_model_name(&weights.base),
                )?);
                Ok(LoadedModel {
                    provider: local.clone(),
                    identity,
                    local: Some(local),
                })
            }
        }
    }
}

/// A model served by a `brain serve` at `base_url` (its OpenAI-compatible
/// surface, ending in `/v1`), as `model` with the API key `api_key` it
/// printed at startup.
///
/// Built on sven's default configuration rather than the user's: what is
/// measured is the server, and no setting of the person running the check
/// (a temperature, a token limit) may change what it answers. `temperature`
/// replaces the default's when given: zero asks the server to decode
/// greedily.
pub fn served_model(
    base_url: &str,
    api_key: &str,
    model: &str,
    temperature: Option<f32>,
) -> Result<LoadedModel, PolicyError> {
    let spec = format!("openai/{model}");
    let config = served_config(&spec, base_url, api_key, temperature)?;
    let driver =
        sven_sdk::drivers::from_config(&config.model).map_err(|e| PolicyError::Remote {
            spec: spec.clone(),
            reason: format!("{e:#}"),
        })?;
    Ok(LoadedModel {
        provider: Arc::from(driver),
        identity: format!("{spec}@{base_url}"),
        local: None,
    })
}

/// The configuration [`served_model`] builds its driver from. Every
/// request asks the server not to open with a reasoning block
/// (`chat_template_kwargs.enable_thinking: false`), as an in-process
/// generation does ([`crate::local`]), so a served answer and an
/// in-process one are produced from the same prompt.
///
/// A typed call is not sent as a constrained `response_format`: the server
/// refuses a request field it cannot honor, and treats an explicit null as
/// absent. The schema stays in the prompt and the reply is parsed and, when it
/// is not the shape, sent back for correction.
fn served_config(
    spec: &str,
    base_url: &str,
    api_key: &str,
    temperature: Option<f32>,
) -> Result<sven_sdk::config::Config, PolicyError> {
    let remote = RemoteModel {
        spec: spec.to_string(),
        base_url: Some(base_url.to_string()),
        api_key: Some(api_key.to_string()),
        thinking: false,
    };
    let mut config = sven_sdk::config::Config::default();
    remote.apply_to(&mut config)?;
    if temperature.is_some() {
        config.model.temperature = temperature;
    }
    config.model.driver_options = serde_json::json!({
        "chat_template_kwargs": { "enable_thinking": false },
        "response_format": null
    });
    Ok(config)
}

/// A model loaded for the length of a command: the provider every engine
/// on it shares, and its identity. Dropping it stops a local generation
/// still running and waits up to [`QUIESCE_GRACE`] for the device to go
/// quiet, so a process never exits under a live device call.
pub struct LoadedModel {
    provider: Arc<dyn ModelProvider>,
    identity: String,
    local: Option<Arc<LocalQwen>>,
}

impl LoadedModel {
    /// The provider engines run the model through.
    #[must_use]
    pub fn provider(&self) -> Arc<dyn ModelProvider> {
        Arc::clone(&self.provider)
    }

    /// The identity records give the model ([`ModelSelection::identity`]).
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// Stops a local generation still running and waits up to
    /// [`QUIESCE_GRACE`] for it to end, as dropping the model does, while
    /// keeping the model: before its base is released for other work.
    /// Returns whether every generation has stopped.
    pub fn quiesce(&self) -> bool {
        self.local
            .as_ref()
            .is_none_or(|local| local.stop_generation(QUIESCE_GRACE))
    }

    /// A provider for the same loaded weights sampling as `sampling` says;
    /// `None` for a model reached over an API, whose sampling is its
    /// server's. The provider shares the weights: this model must outlive
    /// its use, as it must outlive [`Self::provider`]'s.
    #[must_use]
    pub fn resampled(&self, sampling: Sampling) -> Option<Arc<dyn ModelProvider>> {
        self.local
            .as_ref()
            .map(|local| Arc::new(local.resampled(sampling)) as Arc<dyn ModelProvider>)
    }
}

impl std::fmt::Debug for LoadedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadedModel")
            .field("identity", &self.identity)
            .finish_non_exhaustive()
    }
}

impl Drop for LoadedModel {
    fn drop(&mut self) {
        // Best effort by contract: a generation that outlives the grace is
        // cancelling and stops on its own.
        let _ = self.quiesce();
    }
}

/// The name a local model reports: its checkpoint directory as an
/// `org/model` pair (`.../models/Qwen/Qwen3-0.6B` is `Qwen/Qwen3-0.6B`),
/// which is how brain's model store names what it holds.
#[must_use]
pub fn local_model_name(base: &Path) -> String {
    let model = base
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let org = base
        .parent()
        .and_then(Path::file_name)
        .map(|n| n.to_string_lossy().into_owned());
    match org {
        Some(org) if !model.is_empty() => format!("{org}/{model}"),
        _ => model,
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_served_model_is_asked_to_answer_without_a_reasoning_block_and_greedily() {
        let config = served_config("openai/m", "http://127.0.0.1:1/v1", "k", Some(0.0)).unwrap();
        assert_eq!(
            config.model.driver_options["chat_template_kwargs"]["enable_thinking"],
            serde_json::json!(false)
        );
        assert_eq!(config.model.temperature, Some(0.0));
    }

    #[test]
    fn a_model_served_on_this_machine_is_asked_for_no_reasoning_block_and_another_provider_is_not()
    {
        let mut config = sven_sdk::config::Config::default();
        let remote = |spec: &str| RemoteModel {
            spec: spec.to_string(),
            base_url: None,
            api_key: None,
            thinking: false,
        };
        remote("brain/unsloth/Qwen3.8-27B-Q8_0")
            .apply_to(&mut config)
            .unwrap();
        assert_eq!(
            config.model.driver_options["chat_template_kwargs"]["enable_thinking"],
            serde_json::json!(false)
        );
        let mut other = sven_sdk::config::Config::default();
        remote("openrouter/z-ai/glm-5.3-flash")
            .apply_to(&mut other)
            .unwrap();
        assert!(other
            .model
            .driver_options
            .get("chat_template_kwargs")
            .is_none());
        // unless the configuration lets models reason
        let mut reasoning = sven_sdk::config::Config::default();
        RemoteModel {
            thinking: true,
            ..remote("brain/unsloth/Qwen3.8-27B-Q8_0")
        }
        .apply_to(&mut reasoning)
        .unwrap();
        assert!(reasoning
            .model
            .driver_options
            .get("chat_template_kwargs")
            .is_none());
    }

    #[test]
    fn a_served_model_is_never_sent_a_structured_output_request_its_server_would_refuse() {
        // brain's server refuses `response_format` rather than ignore it; an explicit
        // null is "absent" to it, and keeps a typed call on its prompt-described schema.
        let config = served_config("openai/m", "http://127.0.0.1:1/v1", "k", None).unwrap();
        let options = config.model.driver_options.as_object().unwrap();
        assert!(options
            .get("response_format")
            .is_some_and(serde_json::Value::is_null));
    }

    use super::*;
    use std::path::PathBuf;

    fn local(adapter: Option<&str>) -> ModelSelection {
        ModelSelection::Local(LocalWeights {
            base: PathBuf::from("models/Qwen/Qwen3-0.6B"),
            adapter: adapter.map(PathBuf::from),
            context_tokens: Some(4096),
        })
    }

    /// One identity per model, whichever stage records it: the store's
    /// org/model name, with the adapter's stem when one rides along.
    #[test]
    fn a_local_selection_is_named_by_its_checkpoint_and_adapter() {
        assert_eq!(local(None).identity(), "brain/Qwen/Qwen3-0.6B");
        assert_eq!(
            local(Some("train/t1/adapter.safetensors")).identity(),
            "brain/Qwen/Qwen3-0.6B+adapter"
        );
        let remote = ModelSelection::Remote(RemoteModel {
            spec: "openrouter/z-ai/glm-5.3-flash".into(),
            base_url: None,
            api_key: None,
            thinking: false,
        });
        assert_eq!(remote.identity(), "openrouter/z-ai/glm-5.3-flash");
    }

    #[test]
    fn a_remote_spec_without_a_provider_is_refused() {
        let mut config = sven_sdk::config::Config::default();
        let bare = RemoteModel {
            spec: "glm".into(),
            base_url: None,
            api_key: None,
            thinking: false,
        };
        assert!(bare.apply_to(&mut config).is_err());
        let named = RemoteModel {
            spec: "openrouter/z-ai/glm".into(),
            base_url: Some("http://example.invalid/v1".into()),
            api_key: None,
            thinking: false,
        };
        named.apply_to(&mut config).unwrap();
        assert_eq!(config.model.provider, "openrouter");
        assert_eq!(config.model.name, "z-ai/glm");
        assert_eq!(
            config.model.base_url.as_deref(),
            Some("http://example.invalid/v1")
        );
    }
}
