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
//! A checkpoint that is not a path is a name in brain's model store. A
//! remote reference resolves only with the network opt-in: `--allow-remote`,
//! or [`Config::allow_remote`].

use std::path::{Path, PathBuf};

use splinter_core::model_ref::{ModelRef, RefError};
use splinter_model::{LocalWeights, ModelSelection, RemoteModel};

use crate::config::Config;

/// The model `reference` names under `config`; a policy reference is the
/// base with `policy_adapter`, the adapter of the release its alias was
/// resolved to (`None` before any release). A remote reference is refused
/// unless `allow_remote` or the configuration opts in.
pub fn resolve(
    reference: &ModelRef,
    config: &Config,
    allow_remote: bool,
    policy_adapter: Option<&Path>,
) -> Result<ModelSelection, RefError> {
    match reference {
        ModelRef::Policy(_) => Ok(ModelSelection::Local(LocalWeights {
            base: config.policy_base.clone(),
            adapter: policy_adapter.map(Path::to_path_buf),
            context_tokens: config.policy_context_tokens,
        })),
        ModelRef::Local {
            checkpoint,
            adapter,
            context_tokens,
        } => Ok(ModelSelection::Local(LocalWeights {
            base: checkpoint_path(config, checkpoint),
            adapter: adapter.as_ref().map(PathBuf::from),
            context_tokens: *context_tokens,
        })),
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
            }))
        }
    }
}

/// Where `checkpoint` is: a path as written, or a name in the model store.
fn checkpoint_path(config: &Config, checkpoint: &str) -> PathBuf {
    let path = PathBuf::from(checkpoint);
    if path.is_absolute() || checkpoint.starts_with("./") || checkpoint.starts_with("../") {
        path
    } else {
        config.model_store.join(checkpoint)
    }
}
