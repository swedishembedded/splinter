// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements local-first learning agents whose every
// network use is an explicit opt-in, for its clients. If your team needs
// expertise in model selection for agent systems, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Which model plays which role, under the configuration.
//!
//! The rule is [`ModelAssignments::resolve`]'s; this reads the configured
//! models it falls back on and refuses a configured model that is not a
//! model reference.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use splinter_core::model_ref::ModelRef;
use splinter_core::role::{Fallbacks, ModelAssignments, RoleOverrides};
use splinter_model::store::{chat_models, StoredChatModel};

use crate::config::Config;
use crate::error::OrchestratorError;

/// The share of the largest card a discovered assistant may hold: the rest
/// is room the estimate does not count (another process's context, the
/// driver's own).
const CARD_SHARE: f64 = 0.9;

/// Each (model store, policy base)'s discovered assistant reference.
type Discovered = HashMap<(PathBuf, PathBuf), Option<String>>;

/// The assistant this machine offers when none is named: the strongest chat
/// model in the model store, the one a chat load holds the most of, that fits
/// the largest card, at the largest context with which it fits. Only a model
/// larger than the policy is offered - a smaller one helps it learn nothing -
/// and the policy's own checkpoint never is. `None` with no GPU: a larger
/// model on the CPU backend is too slow to help.
///
/// Probed once per process and model store: a scan reads `config.json`
/// files and asks the driver for the cards, which is not free, and the
/// answer is a fact about the machine that does not change in one run.
#[must_use]
pub fn discovered_assistant(config: &Config) -> Option<ModelRef> {
    static FOUND: Mutex<Option<Discovered>> = Mutex::new(None);
    let key = (config.model_store.clone(), config.policy_base.clone());
    let mut found = FOUND
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let reference = found
        .get_or_insert_with(HashMap::new)
        .entry(key)
        .or_insert_with(|| {
            let card = splinter_model::device::gpu_cards()
                .iter()
                .map(|c| c.total_bytes)
                .max()?;
            let budget = (card as f64 * CARD_SHARE) as u64;
            strongest_helper(
                &chat_models(&config.model_store, budget),
                &config.model_store,
                &config.policy_base,
            )
        });
    reference.as_ref().and_then(|text| text.parse().ok())
}

/// The reference of the largest of `models` that is larger than the policy
/// at `policy_base` and is not it.
fn strongest_helper(
    models: &[StoredChatModel],
    store: &Path,
    policy_base: &Path,
) -> Option<String> {
    let policy = models.iter().find(|m| store.join(&m.name) == policy_base);
    models
        .iter()
        .filter(|m| store.join(&m.name) != policy_base)
        .filter(|m| policy.is_none_or(|p| m.footprint_bytes > p.footprint_bytes))
        .max_by_key(|m| m.footprint_bytes)
        .map(|m| format!("local:{}@{}", m.name, m.context_tokens))
}

/// The models the configuration names as fallbacks for roles; with no
/// assistant named, the one this machine offers ([`discovered_assistant`]).
/// `policy:default` named as the assistant keeps every role on the policy.
pub fn fallbacks(config: &Config) -> Result<Fallbacks, OrchestratorError> {
    let parse =
        |what: &str, text: &Option<String>| -> Result<Option<ModelRef>, OrchestratorError> {
            text.as_ref()
                .map(|text| {
                    text.parse::<ModelRef>().map_err(|e| {
                        OrchestratorError::Refused(format!(
                            "the {what} model {text:?} is not a model reference: {e}"
                        ))
                    })
                })
                .transpose()
        };
    let assistant = match &config.assistant_model {
        Some(_) => parse("assistant", &config.assistant_model)?,
        None => discovered_assistant(config),
    };
    Ok(Fallbacks {
        assistant,
        front_door: parse("front door", &config.front_door_model)?,
        judge: parse("judge", &config.judge_model)?,
    })
}

/// Every role assigned: what `named` names, else the configuration's
/// fallbacks, else the policy.
pub fn assignments(
    config: &Config,
    named: &RoleOverrides,
) -> Result<ModelAssignments, OrchestratorError> {
    Ok(ModelAssignments::resolve(named, &fallbacks(config)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stored(name: &str, footprint_gib: u64, context_tokens: u32) -> StoredChatModel {
        StoredChatModel {
            name: name.to_string(),
            context_tokens,
            footprint_bytes: footprint_gib << 30,
        }
    }

    #[test]
    fn the_strongest_model_larger_than_the_policy_is_the_helper() {
        let store = Path::new("/store");
        let policy = store.join("Qwen/Qwen3-0.6B");
        let models = [
            stored("Qwen/Qwen3-0.6B", 3, 32_768),
            stored("Qwen/Qwen3-14B", 20, 8_192),
            stored("Qwen/Qwen3-8B", 12, 16_384),
            stored("Qwen/Qwen3-Embedding-0.6B", 2, 32_768),
        ];
        assert_eq!(
            strongest_helper(&models, store, &policy).as_deref(),
            Some("local:Qwen/Qwen3-14B@8192")
        );
        assert_eq!(
            strongest_helper(&models[..1], store, &policy),
            None,
            "the policy alone is no helper"
        );
        assert_eq!(
            strongest_helper(
                &[
                    stored("Qwen/Qwen3-0.6B", 3, 32_768),
                    stored("Qwen/Qwen3-Embedding-0.6B", 2, 32_768)
                ],
                store,
                &policy
            ),
            None,
            "a model no larger than the policy helps it learn nothing"
        );
        let elsewhere = Path::new("/elsewhere/policy");
        assert_eq!(
            strongest_helper(&models, store, elsewhere).as_deref(),
            Some("local:Qwen/Qwen3-14B@8192"),
            "a policy outside the store: every stored model may help"
        );
        let parsed: ModelRef = "local:Qwen/Qwen3-14B@8192".parse().unwrap();
        assert_eq!(
            parsed.to_string(),
            "local:Qwen/Qwen3-14B@8192",
            "the reference is one the roles parse"
        );
    }
}
