// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! What a local model reference names, and how it is asked to sample.
//!
//! A local model is requested from a brain server by the id its model store
//! gives the checkpoint ([`crate::endpoint`]); nothing here loads weights.
//! This module holds the description of the weights ([`LocalWeights`]), the
//! sampling every request carries ([`Sampling`], [`AGENT_SAMPLING`],
//! [`GREEDY_SAMPLING`]) and the reading of a checkpoint directory's layout.

mod reasoning;
mod request;

use std::path::{Path, PathBuf};

use crate::error::PolicyError;
pub use reasoning::{closed_think_pairs, opens_think_block, CLOSED_THINK, REASONING_REPLY_TOKENS};
pub use request::Sampling;

/// Sampling defaults for agent work, applied per request. Low temperature:
/// an agent is executing a procedure, not writing prose; the small models
/// this provider serves drift into repetition well before they drift into
/// creativity at higher temperatures. The generation cap is bounded to what
/// one agentic step needs - a completion that has not concluded within a few
/// hundred tokens is looping, and a larger cap would let one never-ending
/// generation spend an entire attempt budget.
pub const AGENT_SAMPLING: Sampling = Sampling {
    max_new_tokens: 512,
    temperature: 0.2,
    top_k: 20,
    top_p: None,
    // Agent work wants the answer, not a reasoning preamble it cannot use as
    // tool input; a run that wants the reasoning asks for it.
    thinking: false,
    // Unset: the provider draws a seed for each sampled request and reports
    // it ([`LocalQwen::take_drawn_seeds`]).
    seed: None,
};

/// Greedy decoding for measurement: agent sampling at temperature zero,
/// which brain decodes as argmax. A probe graded twice - in-process and on
/// a server - must get the same answer from the same weights, not two
/// draws.
pub const GREEDY_SAMPLING: Sampling = Sampling {
    temperature: 0.0,
    ..AGENT_SAMPLING
};
/// Where the weights come from and which adapter rides on top.
#[derive(Clone, Debug)]
pub struct LocalWeights {
    /// Checkpoint directory (or file) - the same layout brain's model store
    /// uses. A directory resolves to the checkpoint inside it.
    pub base: PathBuf,
    /// Optional LoRA adapter file, attached to the base.
    pub adapter: Option<PathBuf>,
    /// Inline context budget (tokens): the KV cache is built for exactly
    /// this many, so a prompt plus its generation must fit inside it. `None`
    /// is the largest context the checkpoint supports
    /// ([`maximum_context_tokens`]); a limit is how a model is made to fit
    /// a card.
    pub context_tokens: Option<u32>,
}
/// A directory pointing at a checkpoint resolves to the checkpoint inside
/// it; a file passes through. A sharded Hugging Face checkpoint has no single
/// file to resolve to, so its directory is the checkpoint. Mirrors brain's
/// own `resolve_base`.
pub fn resolve_base(specified: &Path) -> Result<PathBuf, PolicyError> {
    if specified.is_file() {
        return Ok(specified.to_path_buf());
    }
    for name in ["model.safetensors", "model.brain.safetensors"] {
        let candidate = specified.join(name);
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    if is_sharded_hugging_face(specified) {
        return Ok(specified.to_path_buf());
    }
    // Anything else: refuse here, where the caller can name the directory,
    // rather than inside the checkpoint open.
    Err(PolicyError::NoCheckpoint {
        path: specified.to_path_buf(),
    })
}

/// Whether `dir` holds a Hugging Face checkpoint split across numbered
/// `model-*.safetensors` shards (or named by a shard index): its `config.json`
/// declares the architecture, so the directory is what brain opens.
fn is_sharded_hugging_face(dir: &Path) -> bool {
    if !dir.join("config.json").is_file() {
        return false;
    }
    if dir.join("model.safetensors.index.json").is_file() {
        return true;
    }
    std::fs::read_dir(dir).is_ok_and(|entries| {
        entries.flatten().any(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            name.starts_with("model-") && name.ends_with(".safetensors")
        })
    })
}

/// The largest context `checkpoint` supports: the `max_position_embeddings`
/// of the `config.json` in its directory, or beside its file. A checkpoint with no such config (brain's
/// own format, a GGUF) states no maximum, so its context has to be named.
pub fn maximum_context_tokens(checkpoint: &Path) -> Result<u32, PolicyError> {
    let unknown = |reason: String| PolicyError::Load {
        path: checkpoint.to_path_buf(),
        reason: format!("{reason}; name a context with local:<checkpoint>@<tokens>"),
    };
    let source = load_source(checkpoint);
    let directory = if source.is_dir() {
        source.as_path()
    } else {
        source.parent().unwrap_or(Path::new("."))
    };
    let config = directory.join("config.json");
    let text = std::fs::read_to_string(&config).map_err(|e| {
        unknown(format!(
            "no context maximum: cannot read {}: {e}",
            config.display()
        ))
    })?;
    serde_json::from_str::<serde_json::Value>(&text)
        .ok()
        .and_then(|c| c.get("max_position_embeddings")?.as_u64())
        .and_then(|n| u32::try_from(n).ok())
        .filter(|n| *n > 0)
        .ok_or_else(|| {
            unknown(format!(
                "{} has no max_position_embeddings",
                config.display()
            ))
        })
}

/// What brain is asked to open for `checkpoint`: a `model.safetensors`
/// beside a `config.json` is a Hugging Face checkpoint, whose architecture is
/// read from that config, so it opens as its directory; any other file
/// (brain's own format, a GGUF) opens as itself.
pub fn load_source(checkpoint: &Path) -> PathBuf {
    match checkpoint.parent() {
        Some(dir)
            if checkpoint
                .file_name()
                .is_some_and(|n| n == "model.safetensors")
                && dir.join("config.json").is_file() =>
        {
            dir.to_path_buf()
        }
        _ => checkpoint.to_path_buf(),
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_directory_of_the_standard_layout_resolves_to_its_checkpoint() {
        let dir = std::env::temp_dir().join(format!("loop-provider-base-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(resolve_base(&dir).is_err(), "nothing inside, no resolution");
        std::fs::write(dir.join("model.safetensors"), b"x").unwrap();
        assert_eq!(resolve_base(&dir).unwrap(), dir.join("model.safetensors"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_sharded_hugging_face_directory_resolves_to_itself() {
        let root = std::env::temp_dir().join(format!("policy-sharded-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let sharded = root.join("sharded");
        let bare = root.join("bare");
        std::fs::create_dir_all(&sharded).unwrap();
        std::fs::create_dir_all(&bare).unwrap();
        // The 7B distill's layout: config, an index and numbered shards, no
        // single model.safetensors.
        std::fs::write(sharded.join("config.json"), b"{}").unwrap();
        std::fs::write(sharded.join("model.safetensors.index.json"), b"{}").unwrap();
        std::fs::write(sharded.join("model-00001-of-000002.safetensors"), b"x").unwrap();
        std::fs::write(sharded.join("model-00002-of-000002.safetensors"), b"x").unwrap();
        // Shards with no config name no architecture, so nothing to open.
        std::fs::write(bare.join("model-00001-of-000002.safetensors"), b"x").unwrap();

        assert_eq!(resolve_base(&sharded).unwrap(), sharded);
        assert_eq!(load_source(&resolve_base(&sharded).unwrap()), sharded);
        assert!(resolve_base(&bare).is_err());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_checkpoints_largest_context_is_read_from_its_config() {
        let root = std::env::temp_dir().join(format!("splinter-max-ctx-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let (hf, bare) = (root.join("hf"), root.join("bare"));
        std::fs::create_dir_all(&hf).unwrap();
        std::fs::create_dir_all(&bare).unwrap();
        std::fs::write(hf.join("model.safetensors"), b"x").unwrap();
        std::fs::write(
            hf.join("config.json"),
            br#"{"max_position_embeddings": 40960}"#,
        )
        .unwrap();
        std::fs::write(bare.join("model.safetensors"), b"x").unwrap();
        assert_eq!(
            maximum_context_tokens(&hf.join("model.safetensors")).unwrap(),
            40960
        );
        // A brain-format file in a store directory, its config beside it.
        std::fs::write(hf.join("model.brain.safetensors"), b"x").unwrap();
        assert_eq!(
            maximum_context_tokens(&hf.join("model.brain.safetensors")).unwrap(),
            40960
        );
        // No stated maximum: refused with the way to name one, never guessed.
        let refused = maximum_context_tokens(&bare.join("model.safetensors")).unwrap_err();
        assert!(refused.to_string().contains("@<tokens>"), "{refused}");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_hugging_face_checkpoint_is_opened_as_its_directory() {
        let root = std::env::temp_dir().join(format!("splinter-hf-source-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let hf = root.join("hf");
        let brain_format = root.join("brain");
        std::fs::create_dir_all(&hf).unwrap();
        std::fs::create_dir_all(&brain_format).unwrap();
        // The model's architecture lives in config.json; the bare file does not
        // carry it, so only the directory loads.
        std::fs::write(hf.join("model.safetensors"), b"x").unwrap();
        std::fs::write(hf.join("config.json"), b"{}").unwrap();
        // A brain-format file declares itself under its own name.
        std::fs::write(brain_format.join("model.brain.safetensors"), b"x").unwrap();
        std::fs::write(brain_format.join("config.json"), b"{}").unwrap();

        assert_eq!(load_source(&hf.join("model.safetensors")), hf);
        let brain_file = brain_format.join("model.brain.safetensors");
        assert_eq!(load_source(&brain_file), brain_file);
        std::fs::remove_dir_all(&root).unwrap();
    }
}
