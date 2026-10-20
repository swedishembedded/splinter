// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Correction turns: a supervisor who knows the wanted change tells the agent,
//! as a user would, what is still wrong.
//!
//! A correction is recorded in the conversation as a user message, so it is
//! both something the student sees and the only place the hidden change could
//! leak through after the plan. Every message is therefore checked before it
//! is sent ([`check_message`]); a message that fails is refused with its
//! class, the supervisor is asked again with the reason in front of it, and
//! the refusals are counted in the ledger. The classes:
//!
//! * `diff_line` - a changed line of the wanted change, or five words in a
//!   row of one, appears in the message;
//! * `location_offset` - a line number, offset, column or hunk marker;
//! * `diff_reference` - the message refers to a diff, a patch, a commit or
//!   the change as an artifact ([`crate::gate::refers_to_given_change`]);
//! * `leaked_target` - a name or number that only the wanted change has;
//! * `unjustified_content` - a code-shaped name or a number that neither the
//!   task, nor anything the agent has seen or written, nor the repository
//!   holds (the agent cannot know it; [`crate::justify`]).

use std::collections::HashSet;
use std::time::Duration;

use regex::Regex;
use serde::{Deserialize, Serialize};
use splinter_sdk::agent::chat::{reply, ChatOptions, Turn};
use splinter_sdk::agent::solve::Model;

use crate::gate::refers_to_given_change;
use crate::justify::{code_names, Support};
use crate::plan::changed_lines;
use crate::supervisor_prompt::{supervisor_refusal_notice, SUPERVISOR_PROMPT};
use crate::tree::{identifiers, numbers};

/// What a message is checked against.
pub struct MessageContext<'a> {
    diff_lines: Vec<String>,
    only_in_change: HashSet<String>,
    parent_words: &'a HashSet<String>,
    support: &'a Support,
}

fn squeeze(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn words_of(text: &str) -> Vec<String> {
    text.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

fn re(pattern: &str) -> Regex {
    Regex::new(pattern).unwrap_or_else(|_| unreachable!("a fixed pattern"))
}

impl<'a> MessageContext<'a> {
    /// The context of a message about `diff` (the wanted change, unmasked):
    /// `parent_words` is the parent tree's vocabulary, `support` what the task
    /// and the run so far have told the agent.
    #[must_use]
    pub fn new(
        _plan: &str,
        diff: &str,
        parent_words: &'a HashSet<String>,
        support: &'a Support,
    ) -> Self {
        let added: String = diff
            .lines()
            .filter(|l| l.starts_with('+') && !l.starts_with("+++"))
            .map(|l| &l[1..])
            .collect::<Vec<_>>()
            .join("\n");
        let only_in_change = identifiers(&added)
            .map(str::to_lowercase)
            .chain(numbers(&added))
            .collect();
        Self {
            diff_lines: changed_lines(diff),
            only_in_change,
            parent_words,
            support,
        }
    }
}

/// Why a message was refused.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, thiserror::Error)]
#[serde(tag = "class", rename_all = "snake_case")]
pub enum MessageRefusal {
    /// A changed line (or five words of one) is in the message.
    #[error("it repeats a line of the wanted change: {text:?}")]
    DiffLine {
        /// What was repeated.
        text: String,
    },
    /// A line number, offset or hunk marker.
    #[error("it gives a position ({text:?}); describe the place in words")]
    LocationOffset {
        /// The position.
        text: String,
    },
    /// The change is spoken of as an artifact.
    #[error("it refers to the change itself ({text:?})")]
    DiffReference {
        /// The phrase.
        text: String,
    },
    /// A name or number only the change has.
    #[error("it uses {text:?}, which nobody has told the engineer")]
    LeakedTarget {
        /// The name or number.
        text: String,
    },
    /// A name or number nobody has given the engineer.
    #[error("it uses {text:?}, which neither the task nor the repository holds")]
    UnjustifiedContent {
        /// The name or number.
        text: String,
    },
}

impl MessageRefusal {
    /// The class counted in the ledger.
    #[must_use]
    pub fn class(&self) -> &'static str {
        match self {
            Self::DiffLine { .. } => "diff_line",
            Self::LocationOffset { .. } => "location_offset",
            Self::DiffReference { .. } => "diff_reference",
            Self::LeakedTarget { .. } => "leaked_target",
            Self::UnjustifiedContent { .. } => "unjustified_content",
        }
    }
}

/// Number of words in a row that count as quoting a changed line.
const QUOTED_WORDS: usize = 5;
/// Shortest changed line whose appearance in a message counts as quoting it.
const QUOTED_CHARS: usize = 12;

/// Accepts `message` or says why it may not be sent. See the module documentation.
pub fn check_message(message: &str, ctx: &MessageContext<'_>) -> Result<(), MessageRefusal> {
    let squeezed = squeeze(message);
    for line in &ctx.diff_lines {
        let line = squeeze(line);
        if line.chars().count() >= QUOTED_CHARS && squeezed.contains(&line) {
            return Err(MessageRefusal::DiffLine { text: line });
        }
    }
    let said = words_of(message);
    for line in &ctx.diff_lines {
        let line_words = words_of(line);
        for run in line_words.windows(QUOTED_WORDS) {
            if said.windows(QUOTED_WORDS).any(|w| w == run) {
                return Err(MessageRefusal::DiffLine {
                    text: run.join(" "),
                });
            }
        }
    }
    let position = re(
        r"(?i)\blines?\s*#?\d+|\b(?:offset|column)\s*\d+|@@|\bL\d+\b|\.[a-z]{1,4}:\d+|\b\d+(?:st|nd|rd|th) line\b",
    );
    if let Some(found) = position.find(message) {
        return Err(MessageRefusal::LocationOffset {
            text: found.as_str().into(),
        });
    }
    let artifact =
        re(r"(?i)\b(?:diff|patch|hunk|commit|reference solution|ground truth|target tree)\b");
    if let Some(found) = refers_to_given_change(message)
        .or_else(|| artifact.find(message).map(|m| m.as_str().to_string()))
    {
        return Err(MessageRefusal::DiffReference { text: found });
    }
    let unknown: Vec<String> = code_names(message)
        .into_iter()
        .filter(|n| !ctx.support.has(n) && !ctx.parent_words.contains(n.as_str()))
        .collect();
    if let Some(name) = unknown
        .iter()
        .find(|n| ctx.only_in_change.contains(&n.to_lowercase()))
    {
        return Err(MessageRefusal::LeakedTarget { text: name.clone() });
    }
    if let Some(name) = unknown.first() {
        return Err(MessageRefusal::UnjustifiedContent { text: name.clone() });
    }
    Ok(())
}

/// A refused message, as recorded.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Refused {
    /// The class counted in the ledger.
    pub class: String,
    /// What exactly was found.
    pub detail: String,
}

impl From<&MessageRefusal> for Refused {
    fn from(r: &MessageRefusal) -> Self {
        Self {
            class: r.class().into(),
            detail: r.to_string(),
        }
    }
}

/// A correction that was sent.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Correction {
    /// The user message the agent received.
    pub message: String,
    /// Tool calls the agent had made when it arrived.
    pub calls_before: usize,
}

/// Why no message could be had.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NoMessage {
    /// Every message refused.
    pub refused: Vec<Refused>,
    /// The model's failure, if it could not be reached.
    pub model_error: Option<String>,
}

/// Asks `model` for a correction until one passes [`check_message`], at most
/// `tries` times; each refusal is put in front of it with its reason. Returns
/// the accepted message and the refusals that came before it.
pub async fn write_correction(
    model: &Model,
    request: &str,
    ctx: &MessageContext<'_>,
    tries: usize,
    deadline: Duration,
) -> Result<(String, Vec<Refused>), NoMessage> {
    let supervisor = model.clone().with_system(SUPERVISOR_PROMPT.to_string());
    let mut turns = vec![Turn::user(request.to_string())];
    let mut refused = Vec::new();
    for _ in 0..tries.max(1) {
        let message = match reply(&supervisor, &turns, &ChatOptions::new(deadline)).await {
            Ok(m) => m.trim().to_string(),
            Err(e) => {
                return Err(NoMessage {
                    refused,
                    model_error: Some(e.to_string()),
                })
            }
        };
        match check_message(&message, ctx) {
            Ok(()) if !message.is_empty() => return Ok((message, refused)),
            Ok(()) => {
                refused.push(Refused {
                    class: "empty".into(),
                    detail: "the supervisor wrote nothing".into(),
                });
                turns.push(Turn::user(supervisor_refusal_notice("it is empty")));
            }
            Err(why) => {
                refused.push(Refused::from(&why));
                turns.push(Turn::assistant(message));
                turns.push(Turn::user(supervisor_refusal_notice(&why.to_string())));
            }
        }
    }
    Err(NoMessage {
        refused,
        model_error: None,
    })
}
