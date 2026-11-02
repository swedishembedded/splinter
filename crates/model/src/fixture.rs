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

/// The image geometry of [`vision_checkpoint`]: a picture is scaled to
/// multiples of 32 pixels (a 16-pixel patch, merged 2 x 2), and each 32 x 32
/// block of it is one row of the decoder's sequence.
pub const VISION_ROW_PIXELS: u32 = 32;

/// [`byte_level_checkpoint`] of a vision-language base, in the layout of a
/// `Qwen3_5ForConditionalGeneration` release: a `vision_config` and the
/// preprocessing of its pictures, the image tokens, and a tensor of the
/// vision tower (its values are never read: the rows of a picture come from
/// the configuration alone).
pub fn vision_checkpoint(dir: &Path) {
    byte_level_checkpoint(dir);
    let tokenizer_path = dir.join("tokenizer.json");
    let mut tokenizer: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&tokenizer_path).unwrap()).unwrap();
    for (content, id) in [
        ("<|vision_start|>", 259),
        ("<|vision_end|>", 260),
        ("<|image_pad|>", 261),
    ] {
        tokenizer["added_tokens"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({"content": content, "id": id}));
    }
    std::fs::write(&tokenizer_path, tokenizer.to_string()).unwrap();
    std::fs::write(
        dir.join("config.json"),
        serde_json::json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "image_token_id": 261,
            "vision_start_token_id": 259,
            "vision_end_token_id": 260,
            "vision_config": {
                "depth": 1, "hidden_size": 8, "num_heads": 1, "intermediate_size": 8,
                "patch_size": 16, "temporal_patch_size": 2, "spatial_merge_size": 2,
                "num_position_embeddings": 16, "out_hidden_size": 8, "in_channels": 3,
                "deepstack_visual_indexes": []
            }
        })
        .to_string(),
    )
    .unwrap();
    std::fs::write(
        dir.join("preprocessor_config.json"),
        serde_json::json!({
            "size": {"shortest_edge": 1024, "longest_edge": 1_048_576},
            "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
            "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]
        })
        .to_string(),
    )
    .unwrap();
    let header = r#"{"model.visual.patch_embed.proj.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.extend_from_slice(&[0; 4]);
    std::fs::write(dir.join("model.safetensors"), file).unwrap();
}

/// An inline PNG (`data:` URL) of `width` x `height` black pixels.
pub fn png_data_url(width: u32, height: u32) -> String {
    fn crc32(bytes: &[u8]) -> u32 {
        let mut crc = !0u32;
        for &byte in bytes {
            crc ^= u32::from(byte);
            for _ in 0..8 {
                crc = (crc >> 1) ^ (0xedb8_8320 & 0u32.wrapping_sub(crc & 1));
            }
        }
        !crc
    }
    fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], body: &[u8]) {
        out.extend_from_slice(&(body.len() as u32).to_be_bytes());
        let checked: Vec<u8> = kind.iter().chain(body).copied().collect();
        out.extend_from_slice(&checked);
        out.extend_from_slice(&crc32(&checked).to_be_bytes());
    }
    // Rows of one filter byte and `width` black RGB pixels, in one stored
    // (uncompressed) deflate block.
    let raw: Vec<u8> = (0..height)
        .flat_map(|_| std::iter::once(0).chain(std::iter::repeat_n(0, 3 * width as usize)))
        .collect();
    assert!(
        raw.len() < 65_535,
        "one stored block holds the fixture picture"
    );
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in &raw {
        a = (a + u32::from(byte)) % 65_521;
        b = (b + a) % 65_521;
    }
    let mut z = vec![0x78, 0x01, 0x01];
    z.extend_from_slice(&(raw.len() as u16).to_le_bytes());
    z.extend_from_slice(&(!(raw.len() as u16)).to_le_bytes());
    z.extend_from_slice(&raw);
    z.extend_from_slice(&(b << 16 | a).to_be_bytes());
    let mut ihdr = [width.to_be_bytes(), height.to_be_bytes()].concat();
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    chunk(&mut png, b"IHDR", &ihdr);
    chunk(&mut png, b"IDAT", &z);
    chunk(&mut png, b"IEND", &[]);
    format!("data:image/png;base64,{}", base64(&png))
}

fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    bytes
        .chunks(3)
        .flat_map(|group| {
            let bits = group.iter().enumerate().fold(0u32, |bits, (i, &byte)| {
                bits | u32::from(byte) << (16 - 8 * i)
            });
            (0..4).map(move |i| {
                if i <= group.len() {
                    char::from(ALPHABET[(bits >> (18 - 6 * i) & 63) as usize])
                } else {
                    '='
                }
            })
        })
        .collect()
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
