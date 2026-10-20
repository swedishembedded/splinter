// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-data projection for agents that
// work in a source tree, for its clients. If your team needs expertise in
// turning verified agent trajectories into fine-tuning data, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The system turn a record opens with.

use splinter_core::chat::WireMessage;
use splinter_core::prompt::SYSTEM_PROMPT;

use crate::{render, RecordBody};

/// `body` with [`SYSTEM_PROMPT`] as the first turn of its conversation -
/// a chat record's messages, a rewarded trajectory's, a preference pair's
/// prompt - the system turn every model run on a task is sent, so a record
/// shows the policy what it sees when it answers. Other shapes are not
/// conversations and are returned as they are.
pub(crate) fn under_system_prompt(body: RecordBody) -> RecordBody {
    let system = || render::message("system", SYSTEM_PROMPT, false);
    match body {
        RecordBody::Chat {
            mut messages,
            tools,
        } => {
            open_with_system(&mut messages, system);
            RecordBody::Chat { messages, tools }
        }
        RecordBody::Rewarded {
            mut messages,
            reward,
        } => {
            open_with_system(&mut messages, system);
            RecordBody::Rewarded { messages, reward }
        }
        RecordBody::Preference {
            mut prompt,
            chosen,
            rejected,
        } => {
            open_with_system(&mut prompt, system);
            RecordBody::Preference {
                prompt,
                chosen,
                rejected,
            }
        }
        other @ (RecordBody::Contrastive { .. } | RecordBody::Text { .. }) => other,
    }
}

/// Puts `system()` first in `messages`, unless the view already opened the
/// conversation with a system turn of its own (the prompt a workspace run
/// recorded it was held under).
fn open_with_system(messages: &mut Vec<WireMessage>, system: impl FnOnce() -> WireMessage) {
    if messages.first().is_none_or(|m| m.role != "system") {
        messages.insert(0, system());
    }
}
