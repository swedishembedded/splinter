// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements in-process serving of local language
// models for its clients. If your team needs expertise in running
// reasoning models inside an agent, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Whether a checkpoint's chat template opens a reasoning block that its
//! reply must close before the answer begins, and the reply budget such a
//! model needs.
//!
//! A reasoning model such as DeepSeek-R1-Distill answers after a `<think>`
//! block its template has already opened in the prompt, so the generation
//! cap an agent step gets (a few hundred tokens) is spent before the first
//! word of the answer: every probe would grade an empty reply. The template
//! is the checkpoint's own statement of how it generates, so it - not the
//! model's name - decides.

use std::path::Path;

/// The least a reply from a reasoning model may run before its cap, in
/// tokens: room for the reasoning block and the answer after it.
pub const REASONING_REPLY_TOKENS: u32 = 4096;

/// Whether the chat template of the checkpoint at `base` (a directory, or a
/// file in the directory that holds `tokenizer_config.json`) ends its
/// generation prompt inside an open `<think>` block. A template that names
/// `<think>` only together with its closing tag (Qwen3's switch for turning
/// reasoning off) does not; nor does a checkpoint with no template.
#[must_use]
pub fn opens_think_block(base: &Path) -> bool {
    let dir = if base.is_dir() {
        Some(base)
    } else {
        base.parent()
    };
    let Some(config) = dir
        .map(|d| d.join("tokenizer_config.json"))
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
    else {
        return false;
    };
    let template = match config.get("chat_template") {
        Some(serde_json::Value::String(template)) => Some(template.as_str()),
        // A named list of templates: the default one drives generation.
        Some(serde_json::Value::Array(named)) => named
            .iter()
            .find(|t| t.get("name").and_then(|n| n.as_str()) == Some("default"))
            .or_else(|| named.first())
            .and_then(|t| t.get("template"))
            .and_then(|t| t.as_str()),
        _ => None,
    };
    template
        .and_then(|t| t.rfind("add_generation_prompt").map(|at| &t[at..]))
        .is_some_and(|tail| tail.contains("<think>") && !tail.contains("</think>"))
}

/// The reply budget floor for the model at `base`: the reasoning budget when
/// its template opens a think block, else none.
#[must_use]
pub fn reply_floor(base: &Path) -> u32 {
    if opens_think_block(base) {
        REASONING_REPLY_TOKENS
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkpoint(name: &str, chat_template: Option<&str>) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("policy-think-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let config = match chat_template {
            Some(t) => serde_json::json!({ "chat_template": t }),
            None => serde_json::json!({}),
        };
        std::fs::write(dir.join("tokenizer_config.json"), config.to_string()).unwrap();
        dir
    }

    /// DeepSeek-R1-Distill's template opens the block in the prompt; Qwen3's
    /// mentions it only with its close, as the switch for no reasoning; a
    /// plain chat template and a checkpoint with none open nothing.
    #[test]
    fn only_a_template_ending_inside_an_open_think_block_opens_one() {
        let r1 =
            "{% if add_generation_prompt %}{{'<\u{ff5c}Assistant\u{ff5c}><think>\\n'}}{% endif %}";
        let qwen3 = "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}\
                     {%- if enable_thinking is false %}{{- '<think>\\n\\n</think>\\n\\n' }}\
                     {%- endif %}{%- endif %}";
        let plain = "{% if add_generation_prompt %}{{'### Response:'}}{% endif %}";
        for (name, template, opens) in [
            ("r1", Some(r1), true),
            ("qwen3", Some(qwen3), false),
            ("plain", Some(plain), false),
            ("none", None, false),
        ] {
            let dir = checkpoint(name, template);
            assert_eq!(opens_think_block(&dir), opens, "{name}");
            // The checkpoint file beside the config answers as its directory does.
            assert_eq!(
                opens_think_block(&dir.join("model.safetensors")),
                opens,
                "{name}"
            );
            assert_eq!(
                reply_floor(&dir),
                if opens { REASONING_REPLY_TOKENS } else { 0 },
                "{name}"
            );
            std::fs::remove_dir_all(&dir).unwrap();
        }
    }
}
