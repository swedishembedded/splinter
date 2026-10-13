// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-data pipelines whose dataset
// addresses stay stable as the format grows, for its clients. If your team
// needs expertise in versioned fine-tuning datasets, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: a chat record may carry the tool schemas its conversation was held
//! under, and a record that carries none is written, and addressed, exactly
//! as it was before the field existed.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use common::Scratch;
use splinter_data::{write_dataset, Objective, Projection, Record, RecordBody, Unchecked};

/// A chat record as it was serialized before records had tools.
const OLD_RECORD: &str = r#"{"body":{"shape":"chat","messages":[{"role":"system","content":"s","train":false},{"role":"user","content":"u","train":false},{"role":"assistant","content":"a","train":true}]},"metadata":{"view":"sft-final","objective":"sft"}}"#;

/// The line `OLD_RECORD` was written as before records had tools.
const OLD_LINE: &str = r#"{"messages":[{"role":"system","content":"s","train":false},{"role":"user","content":"u","train":false},{"role":"assistant","content":"a","train":true}],"metadata":{"view":"sft-final","objective":"sft"}}"#;

/// The digest of the dataset file `OLD_RECORD` is written as, recorded when
/// records had no tools: it must not move, or every dataset already stored
/// would be named differently.
const OLD_DATASET_DIGEST: &str = "338f705b284101e94bd9479f7e906cee1df5b843e8d4b307a2e60cb9baf7bb21";

fn projection_of(record: Record) -> Projection {
    Projection {
        view: "sft-final".into(),
        objective: Objective::Sft,
        strip: None,
        min_strength: None,
        records: vec![record],
        excluded: Default::default(),
        system_prompt: None,
        terms: None,
    }
}

#[test]
fn a_record_without_tools_keeps_its_serialized_form_and_its_dataset_digest() {
    let record: Record = serde_json::from_str(OLD_RECORD).unwrap();
    assert_eq!(
        serde_json::to_string(&record).unwrap(),
        OLD_RECORD,
        "an old record serializes to the bytes it was read from"
    );
    let scratch = Scratch::new("chat-tools-golden");
    let path = scratch.0.join("train.jsonl");
    let dataset = write_dataset(&path, &projection_of(record), &Unchecked).unwrap();
    assert_eq!(
        std::fs::read_to_string(&path).unwrap(),
        format!("{OLD_LINE}\n")
    );
    assert_eq!(dataset.digest.hex(), OLD_DATASET_DIGEST);
}

#[test]
fn tools_are_written_on_the_line_brain_reads_them_from() {
    let mut record: Record = serde_json::from_str(OLD_RECORD).unwrap();
    let schema = serde_json::json!({
        "type": "function",
        "function": {"name": "shell", "description": "run", "parameters": {"type": "object"}}
    });
    let RecordBody::Chat { tools, .. } = &mut record.body else {
        panic!("a chat record");
    };
    tools.push(schema.clone());
    let scratch = Scratch::new("chat-tools-line");
    let path = scratch.0.join("train.jsonl");
    write_dataset(&path, &projection_of(record), &Unchecked).unwrap();
    let line: serde_json::Value =
        serde_json::from_str(std::fs::read_to_string(&path).unwrap().trim()).unwrap();
    assert_eq!(line["tools"], serde_json::json!([schema]));
    assert!(line["messages"].is_array());
}
