// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training data for models that read images,
// for its clients. If your team needs expertise in fine-tuning on
// conversations with pictures, you can procure our services by sending an
// email to info@swedishembedded.com.

//! The `generic-messages-v2` message: text as a string, images as the parts
//! brain's trainer reads.

use serde_json::json;
use splinter_core::chat::{WireImage, WireMessage};

fn user(content: &str, images: Vec<WireImage>) -> WireMessage {
    WireMessage {
        role: "user".into(),
        content: content.into(),
        images,
        tool_calls: Vec::new(),
        tool_call_id: None,
        reasoning_content: None,
        train: false,
    }
}

#[test]
fn a_message_without_images_is_written_with_a_string_content() {
    let value = serde_json::to_value(user("hello", Vec::new())).unwrap();
    assert_eq!(
        value,
        json!({"role": "user", "content": "hello", "train": false})
    );
}

#[test]
fn images_are_written_as_parts_between_the_text_around_them() {
    let message = user(
        "Look: then say",
        vec![WireImage {
            at: "Look: ".len(),
            url: "data:image/png;base64,AAAA".into(),
        }],
    );
    let value = serde_json::to_value(&message).unwrap();
    assert_eq!(
        value["content"],
        json!([
            {"type": "text", "text": "Look: "},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            {"type": "text", "text": "then say"},
        ])
    );
    let back: WireMessage = serde_json::from_value(value).unwrap();
    assert_eq!(back, message);
}

#[test]
fn text_parts_are_joined_as_the_trainer_joins_them() {
    let value = json!({"role": "user", "train": false, "content": [
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]});
    let message: WireMessage = serde_json::from_value(value).unwrap();
    assert_eq!(message.content, "ab");
    assert_eq!(message.images.len(), 1);
    assert_eq!(message.images[0].at, 2);
}

#[test]
fn an_image_offset_off_a_character_boundary_is_refused_on_writing() {
    let message = user(
        "é",
        vec![WireImage {
            at: 1,
            url: "data:image/png;base64,AAAA".into(),
        }],
    );
    assert!(serde_json::to_value(&message).is_err());
}

#[test]
fn a_supervised_message_with_an_image_is_refused_on_reading() {
    let value = json!({"role": "assistant", "train": true, "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]});
    assert!(serde_json::from_value::<WireMessage>(value).is_err());
}
