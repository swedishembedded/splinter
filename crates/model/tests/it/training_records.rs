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
use splinter_model::fixture::{
    byte_level_checkpoint, png_data_url, vision_checkpoint, VISION_ROW_PIXELS,
};
use splinter_model::measure::RecordMeter;

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

/// The check a dataset crosses before it is stored refuses a record in which
/// one tool call id names two calls (every call would be given every result
/// of the run), naming the line and the id, whatever produced the file.
#[test]
fn a_record_that_names_two_calls_alike_is_refused_by_the_dataset_check() {
    let dir = tempfile::tempdir().unwrap();
    let repeated = TOOL_RECORD.replace(
        r#"{"role":"tool","tool_call_id":"c1","content":"hello","train":false},"#,
        r#"{"role":"tool","tool_call_id":"c1","content":"hello","train":false},{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"read","arguments":"{}"}}],"train":true},{"role":"tool","tool_call_id":"c1","content":"again","train":false},"#,
    );
    let path = write(dir.path(), "dup.jsonl", &[plain_record("ok"), repeated]);
    let err = BrainDatasetCheck
        .check(Format::GenericMessagesV2, &path, 2)
        .expect_err("a repeated call id must not be stored");
    assert!(err.contains(":2:") && err.contains("c1"), "{err}");
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
    let found = overlong_records(&path, dir.path(), true, false, 100).unwrap();
    assert_eq!(found.len(), 1, "{found:?}");
    assert_eq!(found[0].line, 2);
    assert_eq!(found[0].origin.as_deref(), Some("exp-long"));
    assert!(found[0].tokens > 100, "{found:?}");
    let named = found[0].to_string();
    assert!(
        named.contains("train.jsonl:2") && named.contains("exp-long") && named.contains("tokens"),
        "{named}"
    );
    assert!(overlong_records(&path, dir.path(), true, false, 100_000)
        .unwrap()
        .is_empty());
}

/// A conversation whose user turn carries a picture, in the parts brain reads.
fn image_record(images: usize, supervised_image: bool) -> String {
    let picture =
        r#"{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgo="}}"#;
    let user = vec![picture; images].join(",");
    let answer = if supervised_image {
        format!(r#"[{picture}]"#)
    } else {
        r#""A plate.""#.to_owned()
    };
    format!(
        r#"{{"messages":[{{"role":"user","content":[{{"type":"text","text":"What is this?"}},{user}],"train":false}},{{"role":"assistant","content":{answer},"train":true}}]}}"#
    )
}

/// A dataset with pictures in the context of a supervised answer is stored
/// without a base to read them with; what brain refuses whatever the base is
/// stays refused, naming the line.
#[test]
fn the_dataset_check_takes_pictures_in_the_context_and_refuses_the_rest() {
    let dir = tempfile::tempdir().unwrap();
    let check = |lines: &[String]| {
        let path = write(dir.path(), "images.jsonl", lines);
        BrainDatasetCheck.check(Format::GenericMessagesV2, &path, lines.len())
    };
    check(&[plain_record("ok"), image_record(1, false)]).expect("pictures in the context");
    check(&[image_record(8, false)]).expect("eight pictures are the most");
    let nine = check(&[plain_record("ok"), image_record(9, false)]).expect_err("nine pictures");
    assert!(nine.contains(":2:") && nine.contains("image"), "{nine}");
    let answered = check(&[image_record(1, true)]).expect_err("a picture in a supervised turn");
    assert!(answered.contains("supervised"), "{answered}");
}

/// A picture is rows only a vision tower can give it, so on a base without one
/// the trainer refuses the record and so does the check, naming the line,
/// rather than counting a guess.
#[test]
fn a_picture_on_a_base_without_a_vision_tower_is_refused_naming_its_line() {
    let dir = tempfile::tempdir().unwrap();
    byte_level_checkpoint(dir.path());
    let path = write(
        dir.path(),
        "train.jsonl",
        &[plain_record("short"), image_record(1, false)],
    );
    let err = overlong_records(&path, dir.path(), true, false, 5_000)
        .expect_err("a text-only trainer cannot read the picture")
        .to_string();
    assert!(err.contains("line 2") && err.contains("image"), "{err}");
}

/// On a base with a vision tower a picture is counted as the fine-tune counts
/// it: the rows its size gives it under the checkpoint's own image
/// preprocessing, between the vision markers, read from the configuration
/// alone (no weights, no device).
#[test]
fn a_picture_on_a_base_with_a_vision_tower_counts_the_rows_its_size_gives_it() {
    let dir = tempfile::tempdir().unwrap();
    vision_checkpoint(dir.path());
    let meter = RecordMeter::for_model(dir.path(), true, false).unwrap();
    let with_picture = |width: u32, height: u32| {
        let url = png_data_url(width, height);
        format!(
            r#"{{"messages":[{{"role":"user","content":[{{"type":"text","text":"see"}},{{"type":"image_url","image_url":{{"url":"{url}"}}}}],"train":false}},{{"role":"assistant","content":"a","train":true}}]}}"#
        )
    };
    let text_only = meter
        .tokens(r#"{"messages":[{"role":"user","content":"see","train":false},{"role":"assistant","content":"a","train":true}]}"#)
        .unwrap();
    for (width, height) in [(64, 32), (96, 64), (32, 128)] {
        let rows = (width / VISION_ROW_PIXELS * (height / VISION_ROW_PIXELS)) as usize;
        let counted = meter.tokens(&with_picture(width, height)).unwrap();
        // The rows, and the two markers that open and close the run.
        assert_eq!(counted - text_only, rows + 2, "{width}x{height}");
    }
}
