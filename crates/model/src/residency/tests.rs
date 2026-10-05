// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Specs of the residency: one resident base per checkpoint, shared by
//! every model on it, each generation under its own adapter, freed when
//! no model uses it or when the device is wanted elsewhere.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use futures::StreamExt;
use sven_sdk::model::{
    CompletionRequest, Message, MessageContent, ModelProvider, ResponseEvent, Role,
};

use super::scripted::{ScriptedEvent, ScriptedLoader};
use super::Residency;
use crate::local::{LocalQwen, LocalWeights};

/// A scratch checkpoint directory holding a `model.safetensors`, removed
/// when dropped.
struct Checkpoint(PathBuf);

impl Checkpoint {
    fn new(test: &str) -> Self {
        let dir =
            std::env::temp_dir().join(format!("splinter-residency-{test}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.safetensors"), b"scripted").unwrap();
        Self(dir)
    }

    fn weights(&self, adapter: Option<&str>) -> LocalWeights {
        LocalWeights {
            base: self.0.clone(),
            adapter: adapter.map(|a| self.0.join(a)),
            context_tokens: Some(4096),
        }
    }
}

impl Drop for Checkpoint {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A base whose every reply names the adapter attached when it generated:
/// the adapter file's stem, or `base` for none.
fn naming_loader() -> ScriptedLoader {
    ScriptedLoader::new(|_prompt: &str, adapter: Option<&Path>| {
        adapter
            .and_then(Path::file_stem)
            .map_or_else(|| "base".to_string(), |s| s.to_string_lossy().into_owned())
    })
}

/// `model`'s whole visible reply to one question.
async fn answer(model: &LocalQwen) -> String {
    let request = CompletionRequest {
        messages: vec![Message {
            role: Role::User,
            content: MessageContent::Text("which adapter are you?".into()),
        }],
        ..CompletionRequest::default()
    };
    let mut stream = model.complete(request).await.unwrap();
    let mut text = String::new();
    while let Some(event) = stream.next().await {
        match event.unwrap() {
            ResponseEvent::TextDelta(delta) => text.push_str(&delta),
            ResponseEvent::Error(e) => panic!("generation failed: {e}"),
            _ => {}
        }
    }
    text
}

fn loads(loader: &ScriptedLoader) -> usize {
    loader
        .events()
        .iter()
        .filter(|e| matches!(e, ScriptedEvent::Loaded(_)))
        .count()
}

/// The release gate's two arms and the policy differ only by adapter: all
/// three share one copy of the base.
#[test]
fn models_differing_only_by_adapter_share_one_resident_base() {
    let checkpoint = Checkpoint::new("share");
    let loader = naming_loader();
    let residency = Residency::new(Arc::new(loader.clone()));
    let candidate = LocalQwen::load(
        &residency,
        &checkpoint.weights(Some("candidate.safetensors")),
        "m",
    )
    .unwrap();
    let champion = LocalQwen::load(
        &residency,
        &checkpoint.weights(Some("champion.safetensors")),
        "m",
    )
    .unwrap();
    let base = LocalQwen::load(&residency, &checkpoint.weights(None), "m").unwrap();
    assert_eq!(loads(&loader), 1, "{:?}", loader.events());
    assert_eq!(residency.resident(), 1);
    drop((candidate, champion, base));
}

/// Requests alternating between models on one base each run under the
/// requesting model's adapter; a request that already has its adapter
/// attached switches nothing.
#[tokio::test(flavor = "multi_thread")]
async fn each_generation_runs_with_its_own_adapter_when_requests_alternate() {
    let checkpoint = Checkpoint::new("alternate");
    let loader = naming_loader();
    let residency = Residency::new(Arc::new(loader.clone()));
    let candidate = LocalQwen::load(
        &residency,
        &checkpoint.weights(Some("candidate.safetensors")),
        "m",
    )
    .unwrap();
    let champion = LocalQwen::load(
        &residency,
        &checkpoint.weights(Some("champion.safetensors")),
        "m",
    )
    .unwrap();
    let base = LocalQwen::load(&residency, &checkpoint.weights(None), "m").unwrap();
    for _ in 0..2 {
        assert_eq!(answer(&candidate).await, "candidate");
        assert_eq!(answer(&champion).await, "champion");
        assert_eq!(answer(&base).await, "base");
    }
    assert_eq!(answer(&base).await, "base");
    let switches = loader
        .events()
        .iter()
        .filter(|e| matches!(e, ScriptedEvent::Attached(_) | ScriptedEvent::Detached))
        .count();
    // Loading attached the candidate's adapter, then the champion's, then
    // detached for the base; then each of six alternating generations
    // switched once, and the repeated base request not at all.
    assert_eq!(switches, 3 + 6, "{:?}", loader.events());
    assert_eq!(residency.resident(), 1);
}

/// The base lives as long as a model uses it, and no longer; a model
/// loaded afterwards loads it again.
#[test]
fn the_base_is_freed_when_the_last_model_using_it_drops() {
    let checkpoint = Checkpoint::new("free");
    let loader = naming_loader();
    let residency = Residency::new(Arc::new(loader.clone()));
    let candidate = LocalQwen::load(
        &residency,
        &checkpoint.weights(Some("candidate.safetensors")),
        "m",
    )
    .unwrap();
    let base = LocalQwen::load(&residency, &checkpoint.weights(None), "m").unwrap();
    let resampled = base.resampled(crate::local::GREEDY_SAMPLING);
    drop(candidate);
    drop(base);
    assert_eq!(residency.resident(), 1, "the resampled model still uses it");
    drop(resampled);
    assert_eq!(residency.resident(), 0);
    assert!(loader
        .events()
        .iter()
        .any(|e| matches!(e, ScriptedEvent::Freed(_))));
    let again = LocalQwen::load(&residency, &checkpoint.weights(None), "m").unwrap();
    assert_eq!(loads(&loader), 2);
    drop(again);
}

/// Releasing frees every resident base while its models live on; their
/// next generation loads it again, under their own adapter.
#[tokio::test(flavor = "multi_thread")]
async fn a_released_base_is_reloaded_by_the_next_generation() {
    let checkpoint = Checkpoint::new("release");
    let loader = naming_loader();
    let residency = Residency::new(Arc::new(loader.clone()));
    let candidate = LocalQwen::load(
        &residency,
        &checkpoint.weights(Some("candidate.safetensors")),
        "m",
    )
    .unwrap();
    assert_eq!(answer(&candidate).await, "candidate");
    residency.release_all();
    assert_eq!(residency.resident(), 0);
    assert!(loader
        .events()
        .iter()
        .any(|e| matches!(e, ScriptedEvent::Freed(_))));
    assert_eq!(answer(&candidate).await, "candidate");
    assert_eq!(loads(&loader), 2);
    assert_eq!(residency.resident(), 1);
}

/// A device holds one base at a time: loading another base frees the one
/// resident, and the model whose base was freed loads it again - and frees
/// the other - on its next generation.
#[tokio::test(flavor = "multi_thread")]
async fn loading_another_base_frees_the_resident_one_and_each_reloads_on_use() {
    let (first, second) = (Checkpoint::new("evict-a"), Checkpoint::new("evict-b"));
    let loader = naming_loader();
    let residency = Residency::new(Arc::new(loader.clone()));
    let a = LocalQwen::load(&residency, &first.weights(None), "a").unwrap();
    assert_eq!(residency.resident(), 1);
    let b = LocalQwen::load(&residency, &second.weights(None), "b").unwrap();
    assert_eq!(residency.resident(), 1, "{:?}", loader.events());
    assert_eq!(loads(&loader), 2);

    assert_eq!(answer(&a).await, "base");
    assert_eq!(residency.resident(), 1, "{:?}", loader.events());
    assert_eq!(loads(&loader), 3);
    assert_eq!(answer(&b).await, "base");
    assert_eq!(loads(&loader), 4);
    assert_eq!(residency.resident(), 1);
}
