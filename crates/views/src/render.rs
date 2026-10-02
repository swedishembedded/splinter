// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The text views put into a student's input, in one place so every view
//! renders the same parts the same way.

use splinter_lab::WireMessage;
use splinter_record::experience::Experience;

use crate::{Entry, Exclusion, Strip};

/// Introduces the answer being judged in a critic's or verifier's input.
pub(crate) const CANDIDATE_HEADING: &str = "Candidate answer:";

/// Introduces the summary of the checks that ran it, in a verifier's input.
pub(crate) const EXECUTION_HEADING: &str = "Execution evidence:";

/// A message with no tool calls.
pub(crate) fn message(role: &str, content: &str, train: bool) -> WireMessage {
    WireMessage {
        role: role.into(),
        content: content.into(),
        tool_calls: Vec::new(),
        tool_call_id: None,
        train,
    }
}

/// The student's turn for `entry` under `strip`, keyed by the experience.
pub(crate) fn student_turn(entry: &Entry, strip: &Strip) -> Result<String, Exclusion> {
    let experience: &Experience = &entry.experience;
    strip.student_turn(&entry.id.0, &experience.instruction, &experience.privileged)
}

/// `turn` followed by the candidate `answer` under its heading.
pub(crate) fn with_candidate(turn: &str, answer: &str) -> String {
    format!("{turn}\n\n{CANDIDATE_HEADING}\n{answer}")
}
