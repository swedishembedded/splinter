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
use crate::local::{LocalQwen, LocalWeights};

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
        Ok(())
    }
}

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
    /// [`Self::identity`]. Loading local weights is the expensive step -
    /// load once per command, not per request.
    pub fn load(&self) -> Result<LoadedModel, PolicyError> {
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
                let local = Arc::new(LocalQwen::load(weights, &local_model_name(&weights.base))?);
                Ok(LoadedModel {
                    provider: local.clone(),
                    identity,
                    local: Some(local),
                })
            }
        }
    }
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
        if let Some(local) = &self.local {
            // Best effort by contract: a generation that outlives the grace
            // is cancelling and stops on its own.
            let _ = local.stop_generation(QUIESCE_GRACE);
        }
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
    use super::*;
    use std::path::PathBuf;

    fn local(adapter: Option<&str>) -> ModelSelection {
        ModelSelection::Local(LocalWeights {
            base: PathBuf::from("models/Qwen/Qwen3-0.6B"),
            adapter: adapter.map(PathBuf::from),
            context_tokens: 4096,
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
        };
        assert!(bare.apply_to(&mut config).is_err());
        let named = RemoteModel {
            spec: "openrouter/z-ai/glm".into(),
            base_url: Some("http://example.invalid/v1".into()),
            api_key: None,
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
