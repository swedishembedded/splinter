// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: one model-reference syntax everywhere a command names a model.
//!
//! * `policy:<alias>` - the configured base, plus the adapter of the
//!   release the alias points at (`policy:default`, the champion, is the
//!   base alone before any release);
//! * `local:<checkpoint>[+<adapter>]` - a checkpoint path (absolute, or
//!   starting `./` or `../`) or a name in brain's model store, with an
//!   optional adapter file;
//! * `remote:<provider>/<name>` - a model reached over the network, refused
//!   unless the one network opt-in is given.
//!
//! Every malformed reference is refused with an error naming what is wrong.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::path::PathBuf;

use splinter_core::model_ref::{ModelRef, RefError};
use splinter_model::ModelSelection;
use splinter_orchestrator::model_ref::resolve;
use splinter_orchestrator::Config;
use splinter_store::StateRoot;

fn config() -> Config {
    Config {
        state_root: StateRoot::new("state"),
        model_store: PathBuf::from("/models"),
        policy_base: PathBuf::from("/models/Qwen/Qwen3-0.6B"),
        openrouter_api_key: Some("or-key".into()),
        brain_api_key: Some("brain-key".into()),
        allow_remote: false,
        command_env: BTreeMap::new(),
        working_dir: PathBuf::from("."),
        brain_binary: None,
        front_door_model: None,
        assistant_model: None,
        bf16_base: false,
        default_budget: None,
        remote_concurrency: 4,
    }
}

fn parse(text: &str) -> Result<ModelRef, RefError> {
    text.parse()
}

#[test]
fn every_form_parses_and_prints_back() {
    for text in [
        "policy:default",
        "policy:staging-2",
        "local:/ckpt/Qwen3-0.6B",
        "local:Qwen/Qwen3-0.6B",
        "local:./ckpt+./train/adapter.safetensors",
        "remote:openrouter/z-ai/glm-5.3-flash",
    ] {
        assert_eq!(parse(text).unwrap().to_string(), text);
    }
    assert_eq!(parse("policy:default").unwrap(), ModelRef::policy_default());
    assert_eq!(
        parse("local:Qwen/Qwen3-0.6B+a.safetensors").unwrap(),
        ModelRef::Local {
            checkpoint: "Qwen/Qwen3-0.6B".into(),
            adapter: Some("a.safetensors".into()),
        }
    );
    assert_eq!(
        parse("remote:openrouter/z-ai/glm").unwrap(),
        ModelRef::Remote {
            provider: "openrouter".into(),
            name: "z-ai/glm".into(),
        }
    );
}

#[test]
fn malformed_references_are_refused_precisely() {
    let refused = |text: &str| parse(text).unwrap_err();
    assert!(matches!(refused(""), RefError::Empty));
    assert!(matches!(refused("Qwen3"), RefError::NoScheme { .. }));
    assert!(matches!(refused("gguf:x"), RefError::UnknownScheme { .. }));
    assert!(matches!(refused("policy:"), RefError::UnknownPolicy { .. }));
    for not_an_alias in ["policy:Champion", "policy:../x", "policy:2nd", "policy:a/b"] {
        assert!(
            matches!(refused(not_an_alias), RefError::UnknownPolicy { .. }),
            "{not_an_alias}"
        );
    }
    assert!(matches!(
        refused("local:"),
        RefError::EmptyCheckpoint { .. }
    ));
    assert!(matches!(
        refused("local:+a.safetensors"),
        RefError::EmptyCheckpoint { .. }
    ));
    assert!(matches!(
        refused("local:ckpt+"),
        RefError::EmptyAdapter { .. }
    ));
    assert!(matches!(refused("remote:"), RefError::RemoteShape { .. }));
    assert!(matches!(
        refused("remote:openrouter"),
        RefError::RemoteShape { .. }
    ));
    assert!(matches!(
        refused("remote:/glm"),
        RefError::RemoteShape { .. }
    ));
    assert!(matches!(
        refused("remote:openrouter/"),
        RefError::RemoteShape { .. }
    ));
    // The message says what the grammar is.
    let message = refused("Qwen3").to_string();
    assert!(message.contains("policy:default"), "{message}");
    assert!(message.contains("remote:<provider>/<name>"), "{message}");
}

#[test]
fn a_remote_model_is_refused_without_the_network_opt_in() {
    let remote = parse("remote:openrouter/z-ai/glm").unwrap();
    let refused = resolve(&remote, &config(), false, None).unwrap_err();
    assert!(matches!(refused, RefError::RemoteNotAllowed { .. }));
    let message = refused.to_string();
    assert!(message.contains("--allow-remote"), "{message}");

    let ModelSelection::Remote(model) = resolve(&remote, &config(), true, None).unwrap() else {
        panic!("a remote reference resolves to a remote model");
    };
    assert_eq!(model.spec, "openrouter/z-ai/glm");
    assert_eq!(
        model.api_key.as_deref(),
        Some("or-key"),
        "the key follows the provider"
    );
    let brain = parse("remote:brain/qwen3").unwrap();
    let ModelSelection::Remote(model) = resolve(&brain, &config(), true, None).unwrap() else {
        panic!("remote");
    };
    assert_eq!(model.api_key.as_deref(), Some("brain-key"));
    // The configuration's own opt-in counts the same as the flag.
    let opted_in = Config {
        allow_remote: true,
        ..config()
    };
    assert!(resolve(&remote, &opted_in, false, None).is_ok());
}

#[test]
fn local_references_resolve_to_weights() {
    let local = |text: &str| match resolve(&parse(text).unwrap(), &config(), false, None).unwrap() {
        ModelSelection::Local(weights) => weights,
        ModelSelection::Remote(_) => panic!("{text} is local"),
    };
    let policy = local("policy:default");
    assert_eq!(policy.base, PathBuf::from("/models/Qwen/Qwen3-0.6B"));
    assert_eq!(
        policy.adapter, None,
        "no release exists, so no champion adapter"
    );
    assert_eq!(
        local("local:Qwen/Qwen3-1.7B").base,
        PathBuf::from("/models/Qwen/Qwen3-1.7B"),
        "a bare name is looked up in the model store"
    );
    let pathed = local("local:./ckpt+train/adapter.safetensors");
    assert_eq!(pathed.base, PathBuf::from("./ckpt"));
    assert_eq!(
        pathed.adapter,
        Some(PathBuf::from("train/adapter.safetensors"))
    );
    assert_eq!(local("local:/abs/ckpt").base, PathBuf::from("/abs/ckpt"));

    // A policy reference carries the adapter of the release its alias was
    // resolved to; a local one ignores it.
    let champion = std::path::Path::new("/state/releases/ab/adapter.safetensors");
    let resolved = |text: &str| match resolve(
        &parse(text).unwrap(),
        &config(),
        false,
        Some(champion),
    )
    .unwrap()
    {
        ModelSelection::Local(weights) => weights,
        ModelSelection::Remote(_) => panic!("{text} is local"),
    };
    let pinned = resolved("policy:default");
    assert_eq!(pinned.base, PathBuf::from("/models/Qwen/Qwen3-0.6B"));
    assert_eq!(pinned.adapter.as_deref(), Some(champion));
    assert_eq!(resolved("local:/abs/ckpt").adapter, None);
}
