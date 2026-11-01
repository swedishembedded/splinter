// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training data for multi-role agent systems,
// where each model role learns only from the conversations it answered, for
// its clients. If your team needs expertise in fine-tuning the roles of an
// agent harness, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Which harness conversation an experience is, and which model role
//! answered it.
//!
//! A trial that ran a sven harness is several conversations, each answered by
//! a model role. Ingestion admits one experience per conversation and files
//! this record under `extra.splinter_harness` of the experience's
//! trajectory; the dataset views read it back to train a role on its own
//! conversations and to leave out those whose model was shown images.

use atif::Trajectory;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The key of `trajectory.extra` the record is filed under.
pub const KEY: &str = "splinter_harness";

/// Where a conversation's role came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoleSource {
    /// The export names it.
    Export,
    /// The export does not; the request log attributes the conversation's
    /// requests to one model role, unambiguously.
    RequestLog,
    /// Neither says: the role is unknown, never guessed.
    Unknown,
}

/// What an experience records about the harness conversation it is.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HarnessConversation {
    /// The conversation's name.
    pub conversation: String,
    /// The model role that answered it; `None` when unknown.
    pub role: Option<String>,
    /// Where `role` came from.
    pub role_source: RoleSource,
    /// Whether the model was shown images in this conversation. The
    /// trajectory need not carry them, so a record built from it would teach
    /// answers about pictures the student never saw.
    pub shown_images: bool,
}

impl HarnessConversation {
    /// The record filed in `trajectory`, `None` for an experience that is not
    /// a harness conversation.
    #[must_use]
    pub fn of(trajectory: &Trajectory) -> Option<Self> {
        let record = trajectory.extra.as_ref()?.get(KEY)?;
        serde_json::from_value(record.clone()).ok()
    }

    /// The record as the JSON filed under [`KEY`].
    #[must_use]
    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}
