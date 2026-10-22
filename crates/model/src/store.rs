// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that put the models a
// machine already holds to work without being told about them, for its
// clients. If your team needs expertise in local model management for
// training pipelines then you can procure our services by sending an email to
// info@swedishembedded.com.

//! brain's model store, as Splinter sees it: where `brain pull` puts a
//! checkpoint, and which of the checkpoints there can serve as a chat model on
//! this machine's cards. Reads directories and `config.json` files only; no
//! weights are opened.

use std::path::{Path, PathBuf};

/// The model store brain itself uses when nothing names one, so Splinter
/// looks where `brain pull` put a checkpoint. `None` only for a user with no
/// home directory.
#[must_use]
pub fn default_dir() -> Option<PathBuf> {
    brain::default_models_dir()
}

/// A checkpoint in the store that brain loads as a chat model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoredChatModel {
    /// Its name in the store, `<vendor>/<repo>`: what `local:<name>` takes.
    pub name: String,
    /// The largest context, in tokens, with which it fits the card it was
    /// sized for.
    pub context_tokens: u32,
    /// What a chat load of it holds on that card at that context, in bytes.
    pub footprint_bytes: u64,
}

/// The contexts a stored chat model is tried at, largest first.
const CONTEXTS: [u32; 4] = [32_768, 16_384, 8_192, 4_096];

/// Every chat checkpoint in `store` whose chat load fits in `card_bytes` at
/// one of [`CONTEXTS`], at the largest that fits; in name order.
#[must_use]
pub fn chat_models(store: &Path, card_bytes: u64) -> Vec<StoredChatModel> {
    let mut found = Vec::new();
    for vendor in read_dirs(store) {
        for repo in read_dirs(&vendor) {
            let Some(name) = store_name(store, &repo) else {
                continue;
            };
            let fitting = CONTEXTS.iter().find_map(|&context| {
                brain::chat_footprint_bytes(&repo, context)
                    .filter(|&bytes| bytes <= card_bytes)
                    .map(|bytes| (context, bytes))
            });
            if let Some((context_tokens, footprint_bytes)) = fitting {
                found.push(StoredChatModel {
                    name,
                    context_tokens,
                    footprint_bytes,
                });
            }
        }
    }
    found.sort_by(|a, b| a.name.cmp(&b.name));
    found
}

fn read_dirs(dir: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.is_dir())
                .collect()
        })
        .unwrap_or_default()
}

/// `<vendor>/<repo>` of a repository directory under `store`.
fn store_name(store: &Path, repo: &Path) -> Option<String> {
    let relative = repo.strip_prefix(store).ok()?;
    let parts: Vec<&str> = relative.iter().filter_map(|p| p.to_str()).collect();
    (parts.len() == 2).then(|| parts.join("/"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen3(hidden: u64, ff: u64) -> String {
        serde_json::json!({"architectures": ["Qwen3ForCausalLM"], "head_dim": 128, "hidden_size": hidden,
            "intermediate_size": ff, "max_position_embeddings": 40960, "model_type": "qwen3",
            "num_attention_heads": 16, "num_hidden_layers": 28, "num_key_value_heads": 8,
            "rms_norm_eps": 1e-06, "rope_theta": 1000000, "tie_word_embeddings": true, "vocab_size": 151936})
        .to_string()
    }

    #[test]
    fn the_chat_models_of_a_store_are_found_at_the_largest_context_that_fits() {
        let store = std::env::temp_dir().join(format!("splinter-store-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&store);
        for (name, config) in [
            ("Qwen/Small", qwen3(1024, 3072)),
            ("Qwen/Large", qwen3(4096, 12288)),
            (
                "Other/NotChat",
                r#"{"architectures": ["Qwen3TTSForConditionalGeneration"]}"#.to_string(),
            ),
        ] {
            std::fs::create_dir_all(store.join(name)).unwrap();
            std::fs::write(store.join(name).join("config.json"), config).unwrap();
        }
        let roomy = chat_models(&store, 1 << 40);
        let names: Vec<&str> = roomy.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(
            names,
            ["Qwen/Large", "Qwen/Small"],
            "chat models only, by name"
        );
        assert!(
            roomy.iter().all(|m| m.context_tokens == 32_768),
            "the largest context where everything fits"
        );

        let small_at_4k = brain::chat_footprint_bytes(&store.join("Qwen/Small"), 4_096).unwrap();
        let tight = chat_models(&store, small_at_4k);
        assert_eq!(tight.len(), 1, "only what fits the card: {tight:?}");
        assert_eq!(
            (tight[0].name.as_str(), tight[0].context_tokens),
            ("Qwen/Small", 4_096)
        );
        assert!(chat_models(&store.join("absent"), 1 << 40).is_empty());
        std::fs::remove_dir_all(&store).ok();
    }
}
