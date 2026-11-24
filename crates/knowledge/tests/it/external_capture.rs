// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements source capture that grounds every learned
// fact in exact, traceable bytes, for its clients. If your team needs
// expertise in training-data provenance, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec, against constructed ground truth (plan 3A): directory trees built
//! entry by entry (`validation/oracles/knowledge/capture_tree.py`: text
//! files in several scripts, files that are not text, oversized files,
//! symbolic links, a name that is not UTF-8, ignored directories and a
//! nested repository) are captured as exactly the parts, with exactly their
//! bytes, and exactly the skipped entries, with their reasons, that each was
//! built to be: nothing invented, nothing dropped, nothing out of order.
#![allow(clippy::unwrap_used)]

use std::ffi::OsStr;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

use serde_json::{json, Value};
use splinter_conformance::{compare, fixture, Tolerance};
use splinter_core::clock::FixedClock;
use splinter_core::source::Origin;
use splinter_knowledge::capture::capture_repository;

fn hex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
        .collect()
}

fn build(root: &Path, entries: &[Value]) {
    for entry in entries {
        let path = match entry.get("name_hex") {
            Some(name) => root.join(OsStr::from_bytes(&hex(name.as_str().unwrap()))),
            None => root.join(entry["path"].as_str().unwrap()),
        };
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        if entry["kind"] == "link" {
            std::os::unix::fs::symlink(entry["target"].as_str().unwrap(), &path).unwrap();
        } else {
            std::fs::write(&path, hex(entry["hex"].as_str().unwrap())).unwrap();
        }
    }
}

#[test]
fn every_tree_is_captured_as_exactly_the_parts_and_skips_it_was_built_with() {
    let fixture = fixture(env!("CARGO_MANIFEST_DIR"), "knowledge/capture-tree").unwrap();
    for case in fixture.cases() {
        let dir = tempfile::tempdir().unwrap();
        build(dir.path(), case["entries"].as_array().unwrap());
        let limit = case["limit"].as_u64().unwrap();
        let captured = capture_repository(
            dir.path(),
            limit,
            &FixedClock::new("2026-11-24T00:00:00.000Z"),
        )
        .unwrap();
        let parts: Vec<Value> = captured
            .source()
            .parts
            .iter()
            .map(|p| {
                let text = std::str::from_utf8(captured.content(&p.name).unwrap()).unwrap();
                json!({"name": p.name, "media_type": p.media_type, "text": text})
            })
            .collect();
        let Origin::Repository { skipped, .. } = &captured.source().origin else {
            panic!("a repository source");
        };
        let mut actual = case.clone();
        actual["parts"] = json!(parts);
        actual["skipped"] = serde_json::to_value(skipped).unwrap();
        if let Err(mismatch) = compare(&actual, case, Tolerance::new(0.0, 0.0)) {
            panic!("{mismatch}\ncase: {case}");
        }
    }
}
