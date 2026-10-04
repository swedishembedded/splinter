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
//! | `policy:<alias>` | the configured base, plus the adapter of the release `<alias>` points at |
//! | `local:<checkpoint>[+<adapter>][@<tokens>]` | a local checkpoint, optionally with a LoRA adapter file and a context limit |
//! | `remote:<provider>/<name>` | a model reached over the network |
//!
//! A checkpoint is a path when it is absolute or starts with `./` or
//! `../`, and otherwise a name in the model store (`Qwen/Qwen3-0.6B`).
//! The first `+` separates the adapter, so a checkpoint path holding a `+`
//! cannot be named. A local model has the largest context its checkpoint
//! supports unless `@<tokens>` limits it, as fitting a card's memory
//! requires; a trailing `@` that is not followed by digits belongs to the
//! name. A remote reference resolves only with the network opt-in.

use std::str::FromStr;

/// The alias the policy commands use by default: the champion.
pub const POLICY_DEFAULT: &str = "default";

/// The grammar, as refusals state it.
const GRAMMAR: &str = "a model reference is policy:default (or policy:<alias>), \
     local:<checkpoint>[+<adapter>][@<tokens>] or remote:<provider>/<name>";

/// A model a command names; see the module documentation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ModelRef {
    /// `policy:<alias>`: the base, and the release the alias points at.
    Policy(String),
    /// `local:<checkpoint>[+<adapter>][@<tokens>]`.
    Local {
        /// The checkpoint, as written.
        checkpoint: String,
        /// The adapter file, as written.
        adapter: Option<String>,
        /// The context limit in tokens; `None` is the checkpoint's largest.
        context_tokens: Option<u32>,
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
    /// A policy alias that is not an alias name.
    #[error(
        "{alias:?} is not an alias name: a lowercase letter, then up to 63 lowercase letters, \
         digits, - or _"
    )]
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
    /// `local:<checkpoint>@0`: a context of no tokens.
    #[error("{found:?} limits the context to 0 tokens: local:<checkpoint>[+<adapter>][@<tokens>]")]
    ZeroContext {
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
}

/// Whether `name` is an alias name: a lowercase ASCII letter, then up to 63
/// lowercase letters, digits, `-` or `_` - safe as a file name, and never
/// a path.
#[must_use]
pub fn is_alias_name(name: &str) -> bool {
    let mut chars = name.chars();
    chars.next().is_some_and(|c| c.is_ascii_lowercase())
        && name.len() <= 64
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
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
            "policy" if is_alias_name(rest) => Ok(Self::Policy(rest.to_string())),
            "policy" => Err(RefError::UnknownPolicy {
                alias: rest.to_string(),
            }),
            "local" => {
                let (rest, context_tokens) = match rest.rsplit_once('@') {
                    Some((before, tokens))
                        if !tokens.is_empty() && tokens.bytes().all(|b| b.is_ascii_digit()) =>
                    {
                        match tokens.parse::<u32>() {
                            Ok(n) if n > 0 => (before, Some(n)),
                            _ => return Err(RefError::ZeroContext { found: found() }),
                        }
                    }
                    _ => (rest, None),
                };
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
                    context_tokens,
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
                adapter,
                context_tokens,
            } => {
                write!(f, "local:{checkpoint}")?;
                if let Some(adapter) = adapter {
                    write!(f, "+{adapter}")?;
                }
                if let Some(tokens) = context_tokens {
                    write!(f, "@{tokens}")?;
                }
                Ok(())
            }
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
