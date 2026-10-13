// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gated self-improvement loops where a model
// update is adopted only on held-out evidence. If your team needs expertise
// in evaluation-gated fine-tuning, you can procure our services by sending
// an email to info@swedishembedded.com.

//! What a chat dataset must satisfy before a fine-tune claims a device: a
//! record that offers tools and calls them is trainable input, and a record
//! longer than the training row is named, with its length, instead of
//! failing the run.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use splinter_data::{DatasetCheck, Format};
use splinter_model::capabilities::{overlong_records, BrainDatasetCheck};

/// A conversation that offers one tool, calls it, reads the result and
/// answers. Only the two assistant turns are supervised.
const TOOL_RECORD: &str = r#"{"messages":[{"role":"user","content":"read a.txt","train":false},{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"read","arguments":"{\"path\":\"a.txt\"}"}}],"train":true},{"role":"tool","tool_call_id":"c1","content":"hello","train":false},{"role":"assistant","content":"It says hello.","train":true}],"tools":[{"type":"function","function":{"name":"read","description":"read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}}],"metadata":{"experiences":["exp-long"]}}"#;

fn plain_record(text: &str) -> String {
    format!(
        r#"{{"messages":[{{"role":"user","content":"q","train":false}},{{"role":"assistant","content":"{text}","train":true}}]}}"#
    )
}

fn write(dir: &Path, name: &str, lines: &[String]) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, lines.join("\n") + "\n").unwrap();
    path
}

/// A checkpoint directory whose tokenizer maps every byte to one token, so a
/// record's length is the byte length of its rendering.
fn byte_level_checkpoint(dir: &Path) {
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
    std::fs::write(
        dir.join("vocab.json"),
        serde_json::to_string(&vocab).unwrap(),
    )
    .unwrap();
    std::fs::write(dir.join("merges.txt"), "#version: 0.2\n").unwrap();
    std::fs::write(
        dir.join("tokenizer_config.json"),
        r#"{"added_tokens_decoder":{"256":{"content":"<|endoftext|>","special":true},"257":{"content":"<|im_start|>","special":true},"258":{"content":"<|im_end|>","special":true}}}"#,
    )
    .unwrap();
}

/// A record that offers tools and calls them passes the check a dataset is
/// admitted by, and the check counts it as supervised.
#[test]
fn a_record_with_tools_passes_the_dataset_check() {
    let dir = tempfile::tempdir().unwrap();
    let path = write(dir.path(), "tools.jsonl", &[TOOL_RECORD.to_owned()]);
    BrainDatasetCheck
        .check(Format::GenericMessagesV2, &path, 1)
        .expect("a chat record with tools is trainable input");
}

/// A record over the cap is named by its file, line and metadata origin with
/// its length; records within the cap are not reported.
#[test]
fn a_record_over_max_block_is_reported_by_name_before_training() {
    let dir = tempfile::tempdir().unwrap();
    byte_level_checkpoint(dir.path());
    let path = write(
        dir.path(),
        "train.jsonl",
        &[
            plain_record("short"),
            TOOL_RECORD.to_owned(),
            plain_record("short"),
        ],
    );
    let found = overlong_records(&path, dir.path(), 400).unwrap();
    assert_eq!(found.len(), 1, "{found:?}");
    assert_eq!(found[0].line, 2);
    assert_eq!(found[0].origin.as_deref(), Some("exp-long"));
    assert!(found[0].tokens > 400, "{found:?}");
    let named = found[0].to_string();
    assert!(
        named.contains("train.jsonl:2") && named.contains("exp-long") && named.contains("tokens"),
        "{named}"
    );
    assert!(overlong_records(&path, dir.path(), 100_000)
        .unwrap()
        .is_empty());
}
