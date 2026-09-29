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

use sven_sdk::model::ModelProvider;

use crate::local::{LocalQwen, LocalWeights};

/// A model reached over an API, named `provider/model` the way sven's
/// configuration names it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RemoteModel {
    /// `provider/model`, e.g. `openrouter/z-ai/glm-5.3-flash`.
    pub spec: String,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
}

impl RemoteModel {
    /// Writes this model into sven's configuration: provider, model name,
    /// endpoint and key.
    pub fn apply_to(&self, config: &mut sven_sdk::config::Config) -> anyhow::Result<()> {
        let (provider, name) = self.spec.split_once('/').ok_or_else(|| {
            anyhow::anyhow!("a remote model is provider/model, got {:?}", self.spec)
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

    /// A provider for this selection. Loading local weights is the expensive
    /// step - build one provider per stage, not per request.
    pub fn provider(&self) -> anyhow::Result<Box<dyn ModelProvider>> {
        match self {
            Self::Remote(remote) => {
                let mut config = sven_sdk::config::load(None)?;
                remote.apply_to(&mut config)?;
                sven_sdk::drivers::from_config(&config.model)
            }
            Self::Local(weights) => Ok(Box::new(LocalQwen::load(
                weights,
                &local_model_name(&weights.base),
            )?)),
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
