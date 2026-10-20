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

use splinter_core::rng::Seed;
use sven_sdk::model::ModelProvider;

use crate::error::PolicyError;
use crate::local::{LocalQwen, LocalWeights, Sampling};
use crate::residency::Residency;

/// How long dropping a [`LoadedModel`] waits for a local generation still
/// running to stop before the process may exit.
pub const QUIESCE_GRACE: Duration = Duration::from_secs(60);

/// A model reached over an API, named `provider/model` the way sven's
/// configuration names it.
#[derive(Clone, Debug, Default, PartialEq)]
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
    /// The sampling temperature to ask for; `None` keeps the provider's.
    pub temperature: Option<f32>,
    /// Nucleus sampling to ask for (`top_p`); `None` leaves the server's.
    pub top_p: Option<f32>,
    /// The number of most likely tokens to sample among (`top_k`); `None`
    /// leaves the server's.
    pub top_k: Option<u32>,
    /// The seed every request carries, so a sampled answer is the one that
    /// seed draws; `None` leaves the server to draw its own and echo it.
    pub seed: Option<u64>,
}

impl RemoteModel {
    /// How this model is asked to sample, as a record an attempt keeps: what
    /// was asked of the server, not what it did (the seed it echoes is in the
    /// trajectory).
    #[must_use]
    pub fn sampling_record(&self) -> serde_json::Value {
        serde_json::json!({
            "model": self.spec,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "seed": self.seed,
            "thinking": self.thinking,
        })
    }
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
        if self.temperature.is_some() {
            config.model.temperature = self.temperature;
        }
        let mut options = config
            .model
            .driver_options
            .as_object()
            .cloned()
            .unwrap_or_default();
        // A model served on this machine answers like the in-process one:
        // without a reasoning block that would spend the caller's output
        // budget before any answer or tool call is written. When it may
        // reason, the request says so, so the answer does not depend on the
        // server's default.
        if provider == LOCAL_SERVER_PROVIDER {
            options.insert(
                "chat_template_kwargs".into(),
                serde_json::json!({ "enable_thinking": self.thinking }),
            );
        }
        // Not in the OpenAI schema, so they ride in the driver's options with
        // the seed; a server that does not know one refuses the request.
        if let Some(top_p) = self.top_p {
            options.insert("top_p".into(), serde_json::json!(top_p));
        }
        if let Some(top_k) = self.top_k {
            options.insert("top_k".into(), serde_json::json!(top_k));
        }
        if let Some(seed) = self.seed {
            // An unsigned 64-bit seed, the range brain draws and echoes its
            // own seeds from.
            options.insert("seed".into(), serde_json::json!(seed));
        }
        if !options.is_empty() {
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
                let driver = probed_provider(&config).map_err(unreachable)?;
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

/// The provider for `config`, told what context window its server really
/// has when the server says (a brain server and OpenRouter list it).
///
/// Without it sven cannot bound a request's reply by the room the prompt
/// leaves, and a long conversation with a generous output budget is refused
/// by the server ("prompt + max_new exceeds context capacity") or never
/// compacted in time. The probe is best effort and short; a server that does
/// not say leaves the provider as the configuration describes it. It runs on
/// a thread of its own, so it is safe to call from inside an async runtime.
fn probed_provider(config: &sven_sdk::config::Config) -> anyhow::Result<Box<dyn ModelProvider>> {
    let model = config.model.clone();
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()?
                    .block_on(sven_sdk::drivers::from_config_probed(&model))
            })
            .join()
            .map_err(|_| anyhow::anyhow!("the context probe panicked"))?
    })
}

/// How a served model is asked to sample: the settings a request carries.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ServedSampling {
    /// The sampling temperature; zero asks the server to decode greedily.
    pub temperature: Option<f32>,
    /// Nucleus sampling (`top_p`).
    pub top_p: Option<f32>,
    /// Sampling among the `top_k` most likely tokens.
    pub top_k: Option<u32>,
    /// The seed every request carries, so a sampled answer is the one that
    /// seed draws.
    pub seed: Option<u64>,
    /// Whether the model reasons before it answers.
    pub thinking: bool,
}

/// A model served by a `brain serve` at `base_url` (its OpenAI-compatible
/// surface, ending in `/v1`), as `model` with the API key `api_key` it
/// printed at startup.
///
/// Built on sven's default configuration rather than the user's: what is
/// measured is the server, and no setting of the person running the check
/// (a temperature, a token limit) may change what it answers. `temperature`
/// replaces the default's when given: zero asks the server to decode
/// greedily. `seed`, when given, is sent with every request, so a sampled
/// answer is the one that seed draws.
pub fn served_model(
    base_url: &str,
    api_key: &str,
    model: &str,
    temperature: Option<f32>,
    seed: Option<u64>,
) -> Result<LoadedModel, PolicyError> {
    served_model_sampled(
        base_url,
        api_key,
        model,
        &ServedSampling {
            temperature,
            seed,
            ..ServedSampling::default()
        },
    )
}

/// [`served_model`] asked to sample as `sampling` says (temperature, top-p,
/// top-k, seed and whether it reasons), every setting on every request.
pub fn served_model_sampled(
    base_url: &str,
    api_key: &str,
    model: &str,
    sampling: &ServedSampling,
) -> Result<LoadedModel, PolicyError> {
    let spec = format!("openai/{model}");
    let config = served_config(&spec, base_url, api_key, sampling)?;
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
/// (`chat_template_kwargs.enable_thinking: false`) unless `sampling` lets
/// the model reason, as an in-process generation does ([`crate::local`]), so
/// a served answer and an in-process one are produced from the same prompt.
///
/// A typed call is not sent as a constrained `response_format`: the server
/// refuses a request field it cannot honor, and treats an explicit null as
/// absent. The schema stays in the prompt and the reply is parsed and, when it
/// is not the shape, sent back for correction.
fn served_config(
    spec: &str,
    base_url: &str,
    api_key: &str,
    sampling: &ServedSampling,
) -> Result<sven_sdk::config::Config, PolicyError> {
    let remote = RemoteModel {
        spec: spec.to_string(),
        base_url: Some(base_url.to_string()),
        api_key: Some(api_key.to_string()),
        thinking: sampling.thinking,
        temperature: sampling.temperature,
        top_p: sampling.top_p,
        top_k: sampling.top_k,
        seed: sampling.seed,
    };
    let mut config = sven_sdk::config::Config::default();
    remote.apply_to(&mut config)?;
    let mut options = config
        .model
        .driver_options
        .as_object()
        .cloned()
        .unwrap_or_default();
    options.insert(
        "chat_template_kwargs".into(),
        serde_json::json!({ "enable_thinking": sampling.thinking }),
    );
    options.insert("response_format".into(), serde_json::Value::Null);
    config.model.driver_options = serde_json::Value::Object(options);
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

    /// Every seed this model drew for a sampled request that named none since
    /// the last call, in request order ([`LocalQwen::take_drawn_seeds`]);
    /// empty for a model reached over an API, whose server samples.
    #[must_use]
    pub fn take_drawn_seeds(&self) -> Vec<Seed> {
        self.local
            .as_ref()
            .map_or_else(Vec::new, |local| local.take_drawn_seeds())
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
        let config = served_config(
            "openai/m",
            "http://127.0.0.1:1/v1",
            "k",
            &ServedSampling {
                temperature: Some(0.0),
                ..ServedSampling::default()
            },
        )
        .unwrap();
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
            temperature: None,
            ..RemoteModel::default()
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
        // and then the request says so rather than leave it to the server
        assert_eq!(
            reasoning.model.driver_options["chat_template_kwargs"]["enable_thinking"],
            serde_json::json!(true)
        );
    }

    #[test]
    fn the_sampling_of_a_remote_model_rides_every_request_and_is_recorded() {
        let model = RemoteModel {
            spec: "brain/unsloth/Qwen3-8B".into(),
            temperature: Some(0.6),
            top_p: Some(0.95),
            top_k: Some(20),
            // seed: the request and the record must carry exactly this value
            seed: Some(16_022_413_115_216_019_911),
            thinking: true,
            ..RemoteModel::default()
        };
        let mut config = sven_sdk::config::Config::default();
        model.apply_to(&mut config).unwrap();
        assert_eq!(config.model.temperature, Some(0.6));
        let options = &config.model.driver_options;
        assert!((options["top_p"].as_f64().unwrap() - 0.95).abs() < 1e-6);
        assert_eq!(options["top_k"], serde_json::json!(20));
        assert_eq!(options["seed"].as_u64(), Some(16_022_413_115_216_019_911));
        assert_eq!(options["chat_template_kwargs"]["enable_thinking"], true);
        // What was asked is kept with the attempt.
        let record = model.sampling_record();
        assert_eq!(record["top_k"], 20);
        assert_eq!(record["seed"].as_u64(), Some(16_022_413_115_216_019_911));
        assert_eq!(record["thinking"], true);
        // A model not given any is asked for none: the server's own settings.
        let mut plain = sven_sdk::config::Config::default();
        RemoteModel {
            spec: "brain/m".into(),
            ..RemoteModel::default()
        }
        .apply_to(&mut plain)
        .unwrap();
        for key in ["top_p", "top_k", "seed"] {
            assert!(plain.model.driver_options.get(key).is_none(), "{key}");
        }
    }

    #[test]
    fn a_served_model_carries_every_sampling_setting_it_is_given() {
        let config = served_config(
            "openai/m",
            "http://127.0.0.1:1/v1",
            "k",
            &ServedSampling {
                temperature: Some(0.7),
                top_p: Some(0.8),
                top_k: Some(20),
                // seed: the request must carry exactly this value
                seed: Some(7),
                thinking: false,
            },
        )
        .unwrap();
        let options = &config.model.driver_options;
        assert_eq!(config.model.temperature, Some(0.7));
        assert_eq!(options["top_k"], serde_json::json!(20));
        assert!((options["top_p"].as_f64().unwrap() - 0.8).abs() < 1e-6);
        assert_eq!(options["seed"], serde_json::json!(7));
        assert_eq!(options["chat_template_kwargs"]["enable_thinking"], false);
    }

    #[test]
    fn a_remote_model_is_told_the_context_window_its_server_reports() {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for stream in listener.incoming().take(4) {
                let mut stream = stream.unwrap();
                let mut request = [0u8; 2048];
                let _ = stream.read(&mut request);
                let body = r#"{"object":"list","data":[{"id":"m","context_length":98304}]}"#;
                let _ = write!(
                    stream,
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
            }
        });
        let mut config = sven_sdk::config::Config::default();
        RemoteModel {
            spec: "brain/m".into(),
            base_url: Some(format!("http://127.0.0.1:{port}/v1")),
            api_key: Some("k".into()),
            thinking: false,
            temperature: None,
            ..RemoteModel::default()
        }
        .apply_to(&mut config)
        .unwrap();
        // from inside a runtime too: the probe has a thread of its own
        let provider = probed_provider(&config).unwrap();
        assert_eq!(provider.catalog_context_window(), Some(98_304));
    }

    #[test]
    fn the_configured_temperature_is_asked_of_a_remote_model_and_otherwise_left_alone() {
        let model = |temperature| RemoteModel {
            spec: "brain/unsloth/Qwen3.8-27B-Q8_0".into(),
            base_url: None,
            api_key: None,
            thinking: false,
            temperature,
            ..RemoteModel::default()
        };
        let mut asked = sven_sdk::config::Config::default();
        model(Some(0.7)).apply_to(&mut asked).unwrap();
        assert_eq!(asked.model.temperature, Some(0.7));
        let mut default = sven_sdk::config::Config::default();
        let before = default.model.temperature;
        model(None).apply_to(&mut default).unwrap();
        assert_eq!(default.model.temperature, before);
    }

    #[test]
    fn a_served_model_is_never_sent_a_structured_output_request_its_server_would_refuse() {
        // brain's server refuses `response_format` rather than ignore it; an explicit
        // null is "absent" to it, and keeps a typed call on its prompt-described schema.
        let config = served_config(
            "openai/m",
            "http://127.0.0.1:1/v1",
            "k",
            &ServedSampling::default(),
        )
        .unwrap();
        let options = config.model.driver_options.as_object().unwrap();
        assert!(options
            .get("response_format")
            .is_some_and(serde_json::Value::is_null));
        assert!(
            options.get("seed").is_none(),
            "no seed is sent unless given"
        );
    }

    #[test]
    fn a_seed_given_to_a_served_model_is_sent_with_every_request_as_the_same_number() {
        let seed = u64::MAX - 6;
        let config = served_config(
            "openai/m",
            "http://127.0.0.1:1/v1",
            "k",
            &ServedSampling {
                temperature: Some(0.7),
                seed: Some(seed),
                ..ServedSampling::default()
            },
        )
        .unwrap();
        assert_eq!(config.model.driver_options["seed"].as_u64(), Some(seed));
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
            temperature: None,
            ..RemoteModel::default()
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
            temperature: None,
            ..RemoteModel::default()
        };
        assert!(bare.apply_to(&mut config).is_err());
        let named = RemoteModel {
            spec: "openrouter/z-ai/glm".into(),
            base_url: Some("http://example.invalid/v1".into()),
            api_key: None,
            thinking: false,
            temperature: None,
            ..RemoteModel::default()
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
