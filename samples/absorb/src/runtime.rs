// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements local-model runtimes for repeatable
// evaluation runs for its clients. If your team needs expertise in running
// language models on your own hardware under a measurement protocol, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The Splinter handle every command of the sample works through: its state
//! lives under the output directory, and models are local.

use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::Context as _;
use splinter_sdk::agent::solve::Model;
use splinter_sdk::model::local::{Sampling, GREEDY_SAMPLING};
use splinter_sdk::store::StateRoot;
use splinter_sdk::vocabulary::model_ref::ModelRef;
use splinter_sdk::{Config, Splinter};

/// The directory under the output directory that holds Splinter's state.
pub const STATE_DIR: &str = "state";

/// The model every role is played by unless a command names another: the
/// base of the protocol's policy. Its context is the most that fits beside the
/// weights on a 24 GiB card (the checkpoint's full context does not), and is
/// the same for every role so that they share one resident copy.
pub const DEFAULT_MODEL: &str = "local:Qwen/Qwen3-8B@16384";

/// The judge the protocol grades with.
pub const DEFAULT_JUDGE: &str = "local:Qwen/Qwen3-14B@4096";

/// A Splinter handle whose state is `out`/[`STATE_DIR`], with models read from
/// `models` (or the store the configuration names), reasoning off, no network.
///
/// # Errors
/// The runtime cannot start.
pub fn open(out: &Path, models: Option<&PathBuf>) -> anyhow::Result<Splinter> {
    let config = base_config(&out.join(STATE_DIR), models);
    Splinter::builder(config)
        .build()
        .context("starting the Splinter runtime")
}

/// A Splinter handle for a cycle: its state is `state`, and `policy:default`
/// is the checkpoint and context limit of `policy` (a `local:` reference)
/// with the adapter of the release in effect.
///
/// # Errors
/// `policy` is not a local checkpoint, or the runtime cannot start.
pub fn open_cycle(
    state: &Path,
    models: Option<&PathBuf>,
    policy: &str,
) -> anyhow::Result<Splinter> {
    let ModelRef::Local {
        checkpoint,
        adapter: None,
        context_tokens,
    } = model_ref(policy)?
    else {
        anyhow::bail!("the cycle's policy is a plain local checkpoint, not {policy}");
    };
    let mut config = base_config(state, models);
    let path = PathBuf::from(&checkpoint);
    config.policy_base = if path.is_absolute() {
        path
    } else {
        config.model_store.join(path)
    };
    config.policy_context_tokens = context_tokens;
    Splinter::builder(config)
        .build()
        .context("starting the Splinter runtime")
}

fn base_config(state: &Path, models: Option<&PathBuf>) -> Config {
    let mut config = Config::from_env();
    config.state_root = StateRoot::new(state);
    config.thinking = false;
    config.allow_remote = false;
    if let Some(models) = models {
        config.model_store.clone_from(models);
    }
    config
}

/// The model reference `text` names.
///
/// # Errors
/// `text` is not a model reference.
pub fn model_ref(text: &str) -> anyhow::Result<ModelRef> {
    ModelRef::from_str(text).with_context(|| format!("model {text:?}"))
}

/// The temperature a writer of test items draws at: a correction sent back
/// to a model that decodes greedily is answered with the same text again.
pub const WRITER_TEMPERATURE: f32 = 0.8;

/// The model `reference` names as a writer of test items: it draws at
/// [`WRITER_TEMPERATURE`], so that each correction is a new attempt.
///
/// # Errors
/// The reference does not name a local model.
pub fn writer(ctx: &splinter_sdk::Context, reference: &ModelRef) -> anyhow::Result<Model> {
    let sampling = Sampling {
        temperature: WRITER_TEMPERATURE,
        thinking: false,
        ..GREEDY_SAMPLING
    };
    ctx.resampled(reference, sampling)?
        .with_context(|| format!("{reference} is not a local model: its sampling cannot be set"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cycles_policy_alias_is_the_named_checkpoint_at_the_named_context() {
        let dir = tempfile::tempdir().expect("dir");
        let models = dir.path().join("models");
        let splinter = open_cycle(
            &dir.path().join("state"),
            Some(&models),
            "local:Qwen/Qwen3-8B@16384",
        )
        .expect("a handle");
        let selection = splinter
            .context()
            .selection(&ModelRef::policy_default())
            .expect("a selection");
        let weights = selection.local().expect("local weights");
        assert_eq!(weights.base, models.join("Qwen/Qwen3-8B"));
        assert_eq!(weights.context_tokens, Some(16384));
        assert_eq!(weights.adapter, None, "no release yet: the base");
    }

    #[test]
    fn the_cycles_policy_is_a_plain_local_checkpoint() {
        let dir = tempfile::tempdir().expect("dir");
        for policy in ["policy:default", "local:Qwen/Qwen3-8B+adapter.safetensors"] {
            let error = open_cycle(&dir.path().join("state"), None, policy)
                .err()
                .map(|e| format!("{e:#}"))
                .expect("refused");
            assert!(
                error.contains("plain local checkpoint"),
                "{policy}: {error}"
            );
        }
    }
}
