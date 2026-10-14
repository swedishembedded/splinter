// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements local-first learning agents whose every
// network use is an explicit opt-in, for its clients. If your team needs
// expertise in model selection for agent systems, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Resolving a model reference ([`ModelRef`]) to the model it names under the
//! configuration.
//!
//! A checkpoint that is not a path is a name in brain's model store, and an
//! adapter that is a name (`+maren`) is the adapter of that name trained on it.
//! A
//! remote reference resolves only with the network opt-in: `--allow-remote`,
//! or [`Config::allow_remote`].

use std::path::{Path, PathBuf};

use splinter_core::model_ref::{ModelRef, RefError};
use splinter_model::{LocalWeights, ModelSelection, RemoteModel};

use crate::config::Config;

/// What a policy alias was resolved to: the base its release's generation
/// sits on (`None` is the configured base) and the adapter of the release
/// (`None` before any release, and for a consolidation, which is the base
/// itself).
#[derive(Clone, Copy, Debug, Default)]
pub struct PolicyWeights<'a> {
    /// The generation's base directory; `None` is `config.policy_base`.
    pub base: Option<&'a Path>,
    /// The release's adapter file.
    pub adapter: Option<&'a Path>,
}

/// The model `reference` names under `config`; a policy reference is the
/// configured base with `policy_adapter`, the adapter of the release its
/// alias was resolved to (`None` before any release). A remote reference is
/// refused unless `allow_remote` or the configuration opts in.
pub fn resolve(
    reference: &ModelRef,
    config: &Config,
    allow_remote: bool,
    policy_adapter: Option<&Path>,
) -> Result<ModelSelection, RefError> {
    resolve_on(
        reference,
        config,
        allow_remote,
        PolicyWeights {
            base: None,
            adapter: policy_adapter,
        },
    )
}

/// [`resolve`], with the policy on the base its release's generation sits on.
pub fn resolve_on(
    reference: &ModelRef,
    config: &Config,
    allow_remote: bool,
    policy: PolicyWeights<'_>,
) -> Result<ModelSelection, RefError> {
    match reference {
        ModelRef::Policy(_) => Ok(ModelSelection::Local(LocalWeights {
            base: policy
                .base
                .map_or_else(|| config.policy_base.clone(), Path::to_path_buf),
            adapter: policy.adapter.map(Path::to_path_buf),
            context_tokens: config.policy_context_tokens,
        })),
        ModelRef::Local {
            checkpoint,
            adapter,
            context_tokens,
        } => {
            let base = checkpoint_path(config, checkpoint);
            let adapter = adapter.as_deref().map(|a| adapter_path(&base, a));
            Ok(ModelSelection::Local(LocalWeights {
                base,
                adapter,
                context_tokens: *context_tokens,
            }))
        }
        ModelRef::Remote { provider, name } => {
            if !(allow_remote || config.allow_remote) {
                return Err(RefError::RemoteNotAllowed {
                    reference: reference.to_string(),
                });
            }
            let api_key = if provider == "openrouter" {
                config.openrouter_api_key.clone()
            } else {
                config.brain_api_key.clone()
            };
            Ok(ModelSelection::Remote(RemoteModel {
                spec: format!("{provider}/{name}"),
                base_url: None,
                api_key,
                thinking: config.thinking,
                temperature: config.served_temperature,
            }))
        }
    }
}

/// Where `checkpoint` is: a path as written, or a name in the model store.
#[must_use]
pub fn checkpoint_path(config: &Config, checkpoint: &str) -> PathBuf {
    let path = PathBuf::from(checkpoint);
    if path.is_absolute() || checkpoint.starts_with("./") || checkpoint.starts_with("../") {
        path
    } else {
        config.model_store.join(checkpoint)
    }
}

/// The adapter file `adapter` names on the checkpoint at `base`: a path as
/// written when it holds a directory, an extension or a leading dot, and
/// otherwise a name (`maren` or `maren:v2`) in the checkpoint's adapter store,
/// where brain keeps the adapters it trains.
fn adapter_path(base: &Path, adapter: &str) -> PathBuf {
    let is_name = !adapter.contains(['/', '\\'])
        && !adapter.starts_with('.')
        && !adapter.ends_with(".safetensors");
    if !is_name {
        return PathBuf::from(adapter);
    }
    let (name, tag) = adapter.split_once(':').unwrap_or((adapter, "latest"));
    named_adapter_path(base, name, tag)
}

/// The file of the adapter `name`, tagged `tag`, trained on the checkpoint at
/// `base`: where brain's model store keeps it.
#[must_use]
pub fn named_adapter_path(base: &Path, name: &str, tag: &str) -> PathBuf {
    base.join("adapters")
        .join("local")
        .join(name)
        .join(tag)
        .join("adapter.brain.safetensors")
}
