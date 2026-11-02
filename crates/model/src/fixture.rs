// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training pipelines whose records are sized
// exactly as the trainer will size them, for its clients. If your team needs
// expertise in fitting conversations to a fine-tuning block, you can procure
// our services by sending an email to info@swedishembedded.com.

//! A tiny chat checkpoint's tokenizer and template, for specs that measure
//! records as the trainer does ([`crate::measure::RecordMeter`]) without
//! weights or a device.

// Test support: a fixture that cannot be written is the failure report.
#![allow(clippy::unwrap_used)]

use std::path::Path;

/// The chat template of the fixture checkpoint: ChatML, with the tools offered
/// and the calls made written out, so a record's length is its text plus the
/// template's markup.
const TEMPLATE: &str = "{% if tools %}<|im_start|>system\n{{ tools | tojson }}<|im_end|>\n{% endif %}{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}{% if message.tool_calls %}{% for call in message.tool_calls %}{{ call.function.name }}{{ call.function.arguments }}{% endfor %}{% endif %}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

/// Writes into `dir` the tokenizer and chat template of a checkpoint whose tokenizer maps every byte to one token, so a
/// record's length is the byte length of its rendering.
pub fn byte_level_checkpoint(dir: &Path) {
    write_checkpoint(dir, TEMPLATE);
}

/// [`byte_level_checkpoint`] whose template opens every message with a long
/// banner, as templates that carry much markup do: a record's rendered length
/// then exceeds its JSON text's.
pub fn markup_heavy_checkpoint(dir: &Path) {
    let banner = "#".repeat(90);
    write_checkpoint(
        dir,
        &TEMPLATE
            .replace(
                "{{ message.role }}",
                &format!("{{{{ message.role }}}}{banner}"),
            )
            .replace(
                "<|im_start|>assistant\n",
                &format!("<|im_start|>assistant{banner}\n"),
            ),
    );
}

fn write_checkpoint(dir: &Path, template: &str) {
    let printable: Vec<u32> = (0x21..=0x7e)
        .chain(0xa1..=0xac)
        .chain(0xae..=0xff)
        .collect();
    let mut extra = 0;
    let mut vocab = serde_json::Map::new();
    for byte in 0u32..256 {
        let ch = if printable.contains(&byte) {
            byte
        } else {
            extra += 1;
            255 + extra
        };
        vocab.insert(char::from_u32(ch).unwrap().to_string(), byte.into());
    }
    let added = serde_json::json!([
        {"content": "<|endoftext|>", "id": 256},
        {"content": "<|im_start|>", "id": 257},
        {"content": "<|im_end|>", "id": 258}
    ]);
    std::fs::write(
        dir.join("tokenizer.json"),
        serde_json::json!({"model": {"vocab": vocab, "merges": []}, "added_tokens": added})
            .to_string(),
    )
    .unwrap();
    std::fs::write(
        dir.join("tokenizer_config.json"),
        serde_json::json!({"chat_template": template}).to_string(),
    )
    .unwrap();
}
