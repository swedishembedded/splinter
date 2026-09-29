// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The model a command asked for, checked and resolved to a selection.
//!
//! Local-first: every command runs on local weights, served in-process,
//! unless the caller names a remote model AND opts into models reached over
//! an API. A question command (`ask`, `eval-facts`) serves the promoted
//! adapter by default; a delegating command (`run`, `explore`) serves the
//! base, because a facts adapter's `{"answer": ...}` reply shape ends a
//! coding run answer-less.

use std::path::PathBuf;

use splinter_policy::{LocalWeights, ModelSelection, RemoteModel};

use crate::config::Config;

/// Inline context budget for the local model when a command names none.
pub const DEFAULT_CONTEXT_TOKENS: u32 = 16_384;

/// The model part of a command line, before defaults.
#[derive(Clone, Debug)]
pub struct ModelChoice {
    /// Local checkpoint directory; default [`Config::default_local_weights`].
    pub local_weights: Option<PathBuf>,
    /// LoRA adapter (or promotion pointer) served on the local model.
    pub adapter: Option<PathBuf>,
    /// Inline context budget of the local model.
    pub context_tokens: u32,
    /// `provider/model` of a model reached over an API.
    pub remote: Option<String>,
    /// Endpoint of that remote model.
    pub base_url: Option<String>,
    /// Key for that endpoint; defaults from the configuration.
    pub api_key: Option<String>,
    /// The explicit opt-in a remote model needs.
    pub allow_api_models: bool,
}

impl Default for ModelChoice {
    fn default() -> Self {
        Self {
            local_weights: None,
            adapter: None,
            context_tokens: DEFAULT_CONTEXT_TOKENS,
            remote: None,
            base_url: None,
            api_key: None,
            allow_api_models: false,
        }
    }
}

impl ModelChoice {
    /// Whether this choice names a model reached over an API.
    #[must_use]
    pub fn names_api_model(&self) -> bool {
        self.remote.is_some() || self.base_url.is_some()
    }

    /// Refuses the combinations that would be silently wrong: a remote model
    /// without the opt-in, an endpoint with no model to reach there, an
    /// adapter on a model that is not local.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.allow_api_models || !self.names_api_model(),
            "--model and --base-url reach a model over an API; that is refused unless \
             --allow-api-models is given (the default runs every model call locally)"
        );
        anyhow::ensure!(
            self.base_url.is_none() || self.remote.is_some(),
            "--base-url names the endpoint of a remote model; give --model provider/name with it"
        );
        anyhow::ensure!(
            self.adapter.is_none() || self.remote.is_none(),
            "--adapter applies to the local model; drop --model to run locally"
        );
        Ok(())
    }

    /// The local weights this choice serves: the named checkpoint (or the
    /// configured default) with the named adapter.
    #[must_use]
    pub fn local_weights(&self, config: &Config) -> LocalWeights {
        LocalWeights {
            base: self
                .local_weights
                .clone()
                .unwrap_or_else(|| config.default_local_weights.clone()),
            adapter: self.adapter.clone(),
            context_tokens: self.context_tokens,
        }
    }

    /// The local weights a QUESTION command serves: the promoted adapter by
    /// default, so querying what the pipeline learned needs no flag. With
    /// `force_base` the untouched base model - the adapter-vs-base contrast;
    /// combining it with an explicit adapter or a remote model is refused.
    pub fn query_weights(&self, config: &Config, force_base: bool) -> anyhow::Result<LocalWeights> {
        anyhow::ensure!(
            !(force_base && self.adapter.is_some()),
            "--base and --adapter are exclusive: one names the base model, the other an adapter"
        );
        anyhow::ensure!(
            !(force_base && self.remote.is_some()),
            "--base applies to the local model; drop --model to query locally"
        );
        let mut weights = self.local_weights(config);
        if weights.adapter.is_none() && !force_base {
            let pointer = config.state_root.adapter_pointer();
            if pointer.is_file() {
                weights.adapter = Some(pointer);
            }
        }
        Ok(weights)
    }

    /// The selection a command runs on: the remote model when one is named,
    /// else `local`.
    #[must_use]
    pub fn selection(&self, config: &Config, local: LocalWeights) -> ModelSelection {
        match &self.remote {
            Some(spec) => ModelSelection::Remote(RemoteModel {
                spec: spec.clone(),
                base_url: self.base_url.clone(),
                api_key: self.api_key(config, spec),
            }),
            None => ModelSelection::Local(local),
        }
    }

    /// The key follows the provider: OpenRouter's lives in its own
    /// configuration entry, every other endpoint uses the generic one. Unset
    /// is not filled in - a provider that needs a key fails naming it.
    fn api_key(&self, config: &Config, spec: &str) -> Option<String> {
        self.api_key.clone().or_else(|| {
            if spec.starts_with("openrouter/") {
                config.openrouter_api_key.clone()
            } else {
                config.brain_api_key.clone()
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use splinter_store::StateRoot;

    fn config(root: &str) -> Config {
        Config {
            state_root: StateRoot::new(
                std::env::temp_dir().join(format!("splinter-{root}-{}", std::process::id())),
            ),
            default_local_weights: PathBuf::from("models/Qwen/Qwen3-0.6B"),
            openrouter_api_key: Some("or-key".into()),
            brain_api_key: Some("brain-key".into()),
        }
    }

    fn remote(spec: &str) -> ModelChoice {
        ModelChoice {
            remote: Some(spec.into()),
            ..ModelChoice::default()
        }
    }

    /// Local models are the default and remote ones an explicit opt-in:
    /// naming a remote model or endpoint without it is refused before
    /// anything is sent anywhere.
    #[test]
    fn remote_models_need_the_explicit_opt_in() {
        let refused = remote("openrouter/z-ai/glm-5.3-flash")
            .validate()
            .unwrap_err();
        assert!(
            refused.to_string().contains("--allow-api-models"),
            "{refused}"
        );
        let endpoint_only = ModelChoice {
            base_url: Some("http://127.0.0.1:9/v1".into()),
            allow_api_models: true,
            ..ModelChoice::default()
        };
        assert!(
            endpoint_only.validate().is_err(),
            "an endpoint needs a model"
        );
        let allowed = ModelChoice {
            allow_api_models: true,
            ..remote("openrouter/z-ai/glm-5.3-flash")
        };
        allowed.validate().unwrap();
        assert!(matches!(
            allowed.selection(&config("m1"), allowed.local_weights(&config("m1"))),
            ModelSelection::Remote(_)
        ));
        let local_only = ModelChoice {
            allow_api_models: true,
            ..ModelChoice::default()
        };
        assert!(matches!(
            local_only.selection(&config("m2"), local_only.local_weights(&config("m2"))),
            ModelSelection::Local(_)
        ));
    }

    #[test]
    fn the_key_follows_the_provider() {
        let cfg = config("keys");
        let key_of =
            |spec: &str| match remote(spec).selection(&cfg, remote(spec).local_weights(&cfg)) {
                ModelSelection::Remote(r) => r.api_key,
                ModelSelection::Local(_) => unreachable!(),
            };
        assert_eq!(key_of("openrouter/z-ai/glm").as_deref(), Some("or-key"));
        assert_eq!(key_of("brain/qwen3").as_deref(), Some("brain-key"));
    }

    /// A question command serves the promoted adapter by default - querying
    /// what the pipeline learned needs no flag - while `--base` forces the
    /// untouched base model for the contrast, and the two never combine
    /// with an explicit `--adapter`.
    #[test]
    fn question_commands_serve_the_promoted_adapter_until_base_is_asked() {
        let cfg = config("query");
        let pointer = cfg.state_root.adapter_pointer();
        std::fs::create_dir_all(pointer.parent().unwrap()).unwrap();
        std::fs::write(&pointer, r#"{"adapter": "adapter.safetensors"}"#).unwrap();

        let plain = ModelChoice::default();
        assert_eq!(
            plain.query_weights(&cfg, false).unwrap().adapter,
            Some(pointer.clone())
        );
        assert_eq!(plain.query_weights(&cfg, true).unwrap().adapter, None);
        let explicit = ModelChoice {
            adapter: Some(PathBuf::from("mine.safetensors")),
            ..ModelChoice::default()
        };
        assert_eq!(
            explicit.query_weights(&cfg, false).unwrap().adapter,
            Some(PathBuf::from("mine.safetensors"))
        );
        assert!(explicit.query_weights(&cfg, true).is_err());
        assert!(remote("openrouter/z-ai/glm")
            .query_weights(&cfg, true)
            .is_err());
        // A delegating command keeps the base even when a pointer exists.
        assert_eq!(plain.local_weights(&cfg).adapter, None);
        let _ = std::fs::remove_dir_all(cfg.state_root.path());
    }
}
