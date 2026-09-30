// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements local-first learning agents whose every
// network use is an explicit opt-in, for its clients. If your team needs
// expertise in model selection for agent systems, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The one way a command names a model.
//!
//! | Reference | Model |
//! |---|---|
//! | `policy:default` | the configured base, plus the champion adapter once releases exist |
//! | `local:<checkpoint>[+<adapter>]` | a local checkpoint, optionally with a LoRA adapter file |
//! | `remote:<provider>/<name>` | a model reached over the network |
//!
//! A checkpoint is a path when it is absolute or starts with `./` or
//! `../`, and otherwise a name in brain's model store (`Qwen/Qwen3-0.6B`).
//! The first `+` separates the adapter, so a checkpoint path holding a `+`
//! cannot be named. A remote reference resolves only with the network
//! opt-in: `--allow-remote`, or [`Config::allow_remote`].

use std::path::PathBuf;
use std::str::FromStr;

use splinter_policy::{LocalWeights, ModelSelection, RemoteModel};

use crate::config::Config;

/// Inline context budget of a local model.
pub const DEFAULT_CONTEXT_TOKENS: u32 = 16_384;

/// The one policy alias there is.
const POLICY_DEFAULT: &str = "default";

/// The grammar, as refusals state it.
const GRAMMAR: &str =
    "a model reference is policy:default, local:<checkpoint>[+<adapter>] or remote:<provider>/<name>";

/// A model a command names; see the module documentation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ModelRef {
    /// `policy:<alias>`; `default` is the only alias.
    Policy(String),
    /// `local:<checkpoint>[+<adapter>]`.
    Local {
        /// The checkpoint, as written.
        checkpoint: String,
        /// The adapter file, as written.
        adapter: Option<String>,
    },
    /// `remote:<provider>/<name>`.
    Remote {
        /// The provider, as sven's configuration names it.
        provider: String,
        /// The model's name at that provider; may itself hold `/`.
        name: String,
    },
}

/// Why a model reference is refused.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum RefError {
    /// Nothing was given.
    #[error("the model reference is empty; {GRAMMAR}")]
    Empty,
    /// No `scheme:` prefix.
    #[error("{found:?} has no scheme; {GRAMMAR}")]
    NoScheme {
        /// The reference.
        found: String,
    },
    /// A scheme other than `policy`, `local` or `remote`.
    #[error("unknown scheme {scheme:?} in {found:?}; {GRAMMAR}")]
    UnknownScheme {
        /// The scheme given.
        scheme: String,
        /// The reference.
        found: String,
    },
    /// A policy alias other than `default`.
    #[error("unknown policy {alias:?}: `policy:default` is the only policy until releases exist")]
    UnknownPolicy {
        /// The alias given.
        alias: String,
    },
    /// `local:` with no checkpoint before the adapter.
    #[error("{found:?} names no checkpoint: local:<checkpoint>[+<adapter>]")]
    EmptyCheckpoint {
        /// The reference.
        found: String,
    },
    /// `local:<checkpoint>+` with nothing after the `+`.
    #[error("{found:?} names an empty adapter after `+`: local:<checkpoint>[+<adapter>]")]
    EmptyAdapter {
        /// The reference.
        found: String,
    },
    /// `remote:` not followed by `<provider>/<name>`.
    #[error("{found:?} is not remote:<provider>/<name> (both non-empty)")]
    RemoteShape {
        /// The reference.
        found: String,
    },
    /// A remote model without the network opt-in.
    #[error(
        "{reference} is reached over the network; that is refused unless --allow-remote is \
         given or SPLINTER_ALLOW_REMOTE=1 is set (every other model runs locally)"
    )]
    RemoteNotAllowed {
        /// The reference.
        reference: String,
    },
}

impl ModelRef {
    /// `policy:default`.
    #[must_use]
    pub fn policy_default() -> Self {
        Self::Policy(POLICY_DEFAULT.into())
    }

    /// Whether the model is reached over the network.
    #[must_use]
    pub fn is_remote(&self) -> bool {
        matches!(self, Self::Remote { .. })
    }

    /// The model this reference names under `config`. A remote reference
    /// is refused unless `allow_remote` or the configuration opts in.
    pub fn resolve(&self, config: &Config, allow_remote: bool) -> Result<ModelSelection, RefError> {
        match self {
            Self::Policy(_) => Ok(ModelSelection::Local(LocalWeights {
                base: config.policy_base.clone(),
                adapter: None,
                context_tokens: DEFAULT_CONTEXT_TOKENS,
            })),
            Self::Local {
                checkpoint,
                adapter,
            } => Ok(ModelSelection::Local(LocalWeights {
                base: checkpoint_path(config, checkpoint),
                adapter: adapter.as_ref().map(PathBuf::from),
                context_tokens: DEFAULT_CONTEXT_TOKENS,
            })),
            Self::Remote { provider, name } => {
                if !(allow_remote || config.allow_remote) {
                    return Err(RefError::RemoteNotAllowed {
                        reference: self.to_string(),
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

impl FromStr for ModelRef {
    type Err = RefError;

    fn from_str(text: &str) -> Result<Self, RefError> {
        if text.is_empty() {
            return Err(RefError::Empty);
        }
        let found = || text.to_string();
        let Some((scheme, rest)) = text.split_once(':') else {
            return Err(RefError::NoScheme { found: found() });
        };
        match scheme {
            "policy" if rest == POLICY_DEFAULT => Ok(Self::policy_default()),
            "policy" => Err(RefError::UnknownPolicy {
                alias: rest.to_string(),
            }),
            "local" => {
                let (checkpoint, adapter) = match rest.split_once('+') {
                    Some((checkpoint, adapter)) => (checkpoint, Some(adapter)),
                    None => (rest, None),
                };
                if checkpoint.is_empty() {
                    return Err(RefError::EmptyCheckpoint { found: found() });
                }
                if adapter.is_some_and(str::is_empty) {
                    return Err(RefError::EmptyAdapter { found: found() });
                }
                Ok(Self::Local {
                    checkpoint: checkpoint.to_string(),
                    adapter: adapter.map(str::to_string),
                })
            }
            "remote" => match rest.split_once('/') {
                Some((provider, name)) if !provider.is_empty() && !name.is_empty() => {
                    Ok(Self::Remote {
                        provider: provider.to_string(),
                        name: name.to_string(),
                    })
                }
                _ => Err(RefError::RemoteShape { found: found() }),
            },
            _ => Err(RefError::UnknownScheme {
                scheme: scheme.to_string(),
                found: found(),
            }),
        }
    }
}

impl std::fmt::Display for ModelRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Policy(alias) => write!(f, "policy:{alias}"),
            Self::Local {
                checkpoint,
                adapter: None,
            } => write!(f, "local:{checkpoint}"),
            Self::Local {
                checkpoint,
                adapter: Some(adapter),
            } => write!(f, "local:{checkpoint}+{adapter}"),
            Self::Remote { provider, name } => write!(f, "remote:{provider}/{name}"),
        }
    }
}

/// Serialized as the reference itself, `policy:default` and so on.
impl serde::Serialize for ModelRef {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}
