// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `generic-messages-v2` wire shapes: one chat message, its tool calls
//! and the supervision flag brain's parser requires on every message.
//!
//! A message's `content` is a string, or - when it was shown images - a list
//! of `{"type":"text"}` and `{"type":"image_url"}` parts, the shape brain's
//! trainer reads for a base that carries a vision tower. In memory a message
//! keeps its text in [`WireMessage::content`] and each image beside it with
//! the byte offset of the text it follows ([`WireImage`]), so everything that
//! reads text keeps reading one string.

use serde::de::Error as _;
use serde::ser::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// One image of a message: an inline `data:` URL placed in the text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WireImage {
    /// The byte offset in [`WireMessage::content`] the image sits at: the
    /// text before it comes first. Offsets are in order and on character
    /// boundaries.
    pub at: usize,
    /// The image as an inline `data:<media type>;base64,...` URL.
    pub url: String,
}

/// One `generic-messages-v2` message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WireMessage {
    /// `system`, `user`, `assistant` or `tool`.
    pub role: String,
    /// The message text, the text parts of a message with images joined as
    /// brain joins them; present even when empty.
    pub content: String,
    /// The images the message carries, in order. Brain trains on images in
    /// context only, so a message with images is never supervised.
    pub images: Vec<WireImage>,
    /// The tool calls an assistant message makes.
    pub tool_calls: Vec<WireToolCall>,
    /// On a `tool` message, the id of the call it answers.
    pub tool_call_id: Option<String>,
    /// On an assistant message, the reasoning the model produced before it.
    /// Brain hands it to the checkpoint's chat template as the turn's
    /// `reasoning_content`, which renders it as the turn's think block, so a
    /// trained turn shows the reasoning where the model produces it.
    pub reasoning_content: Option<String>,
    /// Required on every message by the consuming parser: a record with no
    /// explicit supervision boundary is either a silent no-op or a silent
    /// prompt leak into the loss.
    pub train: bool,
}

/// The JSON a [`WireMessage`] is written as and read from.
#[derive(Serialize, Deserialize)]
struct WireMessageJson {
    role: String,
    content: WireContent,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<WireToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    train: bool,
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum WireContent {
    Text(String),
    Parts(Vec<WirePart>),
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
enum WirePart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: WireImageUrl },
}

#[derive(Serialize, Deserialize)]
struct WireImageUrl {
    url: String,
}

impl Serialize for WireMessage {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let content = if self.images.is_empty() {
            WireContent::Text(self.content.clone())
        } else {
            let mut parts = Vec::new();
            let mut done = 0;
            for image in &self.images {
                let text = self.content.get(done..image.at).ok_or_else(|| {
                    S::Error::custom(
                        "image offsets are not in order, on character boundaries inside the text",
                    )
                })?;
                if !text.is_empty() {
                    parts.push(WirePart::Text { text: text.into() });
                }
                parts.push(WirePart::ImageUrl {
                    image_url: WireImageUrl {
                        url: image.url.clone(),
                    },
                });
                done = image.at;
            }
            if done < self.content.len() {
                parts.push(WirePart::Text {
                    text: self.content[done..].into(),
                });
            }
            WireContent::Parts(parts)
        };
        WireMessageJson {
            role: self.role.clone(),
            content,
            tool_calls: self.tool_calls.clone(),
            tool_call_id: self.tool_call_id.clone(),
            reasoning_content: self.reasoning_content.clone(),
            train: self.train,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for WireMessage {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let json = WireMessageJson::deserialize(deserializer)?;
        let (content, images) = match json.content {
            WireContent::Text(text) => (text, Vec::new()),
            WireContent::Parts(parts) => {
                let (mut content, mut images) = (String::new(), Vec::new());
                for part in parts {
                    match part {
                        WirePart::Text { text } => content.push_str(&text),
                        WirePart::ImageUrl { image_url } => images.push(WireImage {
                            at: content.len(),
                            url: image_url.url,
                        }),
                    }
                }
                (content, images)
            }
        };
        if json.train && !images.is_empty() {
            return Err(D::Error::custom(
                "a supervised message holds an image: images belong to the context",
            ));
        }
        Ok(Self {
            role: json.role,
            content,
            images,
            tool_calls: json.tool_calls,
            tool_call_id: json.tool_call_id,
            reasoning_content: json.reasoning_content,
            train: json.train,
        })
    }
}

/// One tool call in a `generic-messages-v2` assistant message.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WireToolCall {
    /// The call's id, which a later `tool` message answers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// The call type, `function`.
    #[serde(rename = "type")]
    pub kind: String,
    /// The function called.
    pub function: WireFunction,
}

/// The function a [`WireToolCall`] calls.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WireFunction {
    /// The tool's name.
    pub name: String,
    /// JSON-encoded argument text, exactly as the model emitted it.
    pub arguments: String,
}
