// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The SFT-final view: a passed experience's final answer, supervised.
//!
//! An experience qualifies when its annotations decide pass (under the
//! decision rule of `splinter_store::annotation`: the strongest pass/fail
//! verdicts decide, a conflict among them decides nothing) at or above the
//! view's minimum strength, and it has a final output. It yields one record:
//! the instruction as the user turn, not supervised, and the final output as
//! the assistant turn, the only supervised one. The trajectory's
//! intermediate steps and every privileged item stay out of the record.

use splinter_lab::WireMessage;
use splinter_store::annotation::{decide, Annotation, Strength};
use splinter_store::experience::Experience;

use crate::{own_notes, Objective, Record, RecordMetadata, View, ViewError};

/// The name every SFT-final record carries.
const NAME: &str = "sft-final";

/// Supervised fine-tuning on the final answer of a passed experience.
#[derive(Clone, Copy, Debug)]
pub struct SftFinal {
    min_strength: Strength,
}

impl SftFinal {
    /// The view, accepting pass decisions at `min_strength` or stronger.
    #[must_use]
    pub fn new(min_strength: Strength) -> Self {
        Self { min_strength }
    }
}

fn message(role: &str, content: &str, train: bool) -> WireMessage {
    WireMessage {
        role: role.into(),
        content: content.into(),
        tool_calls: Vec::new(),
        tool_call_id: None,
        train,
    }
}

impl View for SftFinal {
    fn name(&self) -> &str {
        NAME
    }

    fn objective(&self) -> Objective {
        Objective::Sft
    }

    fn project(
        &self,
        experience: &Experience,
        notes: &[Annotation],
    ) -> Result<Vec<Record>, ViewError> {
        let id = own_notes(experience, notes)?;
        let qualifies = decide(notes).is_some_and(|d| d.passed && d.strength >= self.min_strength);
        let Some(answer) = experience.final_output.as_deref().filter(|_| qualifies) else {
            return Ok(Vec::new());
        };
        Ok(vec![Record {
            messages: vec![
                message("user", &experience.instruction, false),
                message("assistant", answer, true),
            ],
            metadata: RecordMetadata {
                experience: id,
                view: NAME.into(),
                objective: Objective::Sft,
            },
        }])
    }
}
