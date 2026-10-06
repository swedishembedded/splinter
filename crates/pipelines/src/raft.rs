// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements retrieval-augmented fine-tuning of models
// on a person's writing, for its clients. If your team needs expertise in
// training a model to use retrieved passages and ignore the wrong ones, you
// can procure our services by sending an email to info@swedishembedded.com.

//! Retrieval-augmented fine-tuning: training records with passages in the
//! prompt.
//!
//! A model trained only closed-book has never seen a prompt with passages in
//! front of the question. Asked with retrieved ones it cannot tell the
//! passage that holds the answer from one that only resembles the question:
//! measured, the passages it was given helped where retrieval found the
//! evidence and hurt exactly as much where it did not. So a share of the
//! training records is given passages as `ask --retrieve` gives them, each
//! under its part and section, and the stable draw of each decides what it
//! teaches ([`PassageShare`]):
//!
//! * the passage the task was written from among retrieved distractors, the
//!   answer unchanged: to use context where it holds the answer;
//! * distractors alone and an **abstention**, the writer saying in their own
//!   voice that the writings supplied do not settle it ([`Abstainer`]): a
//!   retrieval miss;
//! * a question the writings cannot hold at all (about something after the
//!   writer's lifetime, say), the passages retrieved for it, and the same
//!   abstention;
//! * distractors alone and the answer unchanged.
//!
//! An abstention is made from a record and stays with it: its experience, and
//! so its family, are the record's, so it is held out with its family or
//! trained with it and never lands on the other side of the split. Which
//! records, and what each becomes, is a stable function of the experience, so
//! the same dataset is built every time.

use serde::Serialize;
use splinter_core::chat::WireMessage;
use splinter_core::digest::Digest;
use splinter_core::experience::Task;
use splinter_data::{Projection, RecordBody};
use splinter_knowledge::retrieve::Passage;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

use crate::retrieval::{overlaps_evidence, Retrieval, Retrieved};

/// More passages than are shown are retrieved, so that dropping those that
/// hold the evidence still leaves enough distractors.
const SPARE: usize = 3;

/// Of the records given passages, the share whose passages include the one
/// the task was written from: most, so context is learned to be worth using,
/// and enough without it that the model does not lean on it.
pub const DEFAULT_EVIDENCE_SHARE: f64 = 0.8;

/// How much of a dataset is given passages, and what the records given them
/// teach. The last three are bands of one draw, so they sum to at most 1;
/// what none takes is distractors alone with the answer unchanged.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct PassageShare {
    /// The share of records given passages at all, in `[0, 1]`.
    pub records: f64,
    /// Of those, the share whose passages include the one the task was written
    /// from.
    pub with_evidence: f64,
    /// Of those, the share whose passages are distractors only and whose
    /// answer is an abstention: a retrieval miss.
    pub abstain: f64,
    /// Of those, the share asked what the writings cannot hold, with the
    /// passages retrieved for that question, and answered by an abstention.
    pub unsupported: f64,
}

impl PassageShare {
    /// The share named by the command line: `records` given passages, the
    /// evidence among them for [`DEFAULT_EVIDENCE_SHARE`] of those, no
    /// abstention.
    #[must_use]
    pub fn evidence(records: f64) -> Self {
        Self {
            records,
            with_evidence: DEFAULT_EVIDENCE_SHARE,
            ..Self::default()
        }
    }

    /// `records` given passages, of which enough are abstentions that
    /// `abstentions` of all the records are: two thirds retrieval misses and
    /// a third questions beyond the writings, the evidence among the passages
    /// for the most of the rest that [`DEFAULT_EVIDENCE_SHARE`] allows. Shares
    /// that cannot be made so (more abstentions than records given
    /// passages) are refused by [`PassageShare::validate`].
    #[must_use]
    pub fn with_abstentions(records: f64, abstentions: f64) -> Self {
        let of_passages = if records > 0.0 {
            abstentions / records
        } else {
            f64::INFINITY
        };
        Self {
            records,
            with_evidence: (1.0 - of_passages).clamp(0.0, DEFAULT_EVIDENCE_SHARE),
            abstain: of_passages * 2.0 / 3.0,
            unsupported: of_passages / 3.0,
        }
    }

    /// Whether the share has records whose answer is an abstention.
    #[must_use]
    pub fn abstains(&self) -> bool {
        self.abstain > 0.0 || self.unsupported > 0.0
    }

    /// The bands of the draw must fit in `[0, 1]`.
    pub fn validate(&self) -> Result<(), OrchestratorError> {
        let shares = [
            self.records,
            self.with_evidence,
            self.abstain,
            self.unsupported,
        ];
        if shares.iter().any(|s| !(0.0..=1.0).contains(s))
            || self.with_evidence + self.abstain + self.unsupported > 1.0 + 1e-9
        {
            return Err(OrchestratorError::Refused(format!(
                "the passage shares {self:?} are not shares: each is in [0, 1] and the evidence, \
                 abstention and unsupported shares together at most 1"
            )));
        }
        Ok(())
    }
}

/// What an abstention says the writings lack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Abstention {
    /// The question is about what the writings treat, but the passages
    /// supplied do not hold the answer.
    Miss,
    /// The question is about what the writings cannot hold: it is outside
    /// them or after the writer's lifetime.
    Unsupported,
}

/// Whoever writes abstentions and the questions that call for them.
pub trait Abstainer {
    /// A question about the topic of `task` that the writings cannot answer;
    /// `None` when none is written.
    fn unanswerable_question(&self, task: &Task) -> Option<String>;

    /// The reply of the writer who is shown `shown` and `question` and finds
    /// no answer in them; `None` when no admissible reply is written.
    fn abstention(&self, question: &str, shown: &[&Passage], kind: Abstention) -> Option<String>;
}

/// The stable draw of `subject` under `salt`, in `[0, 1]`.
fn draw(subject: &Digest, salt: &str) -> f64 {
    let draw = Digest::of(format!("splinter-raft/{salt}/{subject}").as_bytes());
    let top = u64::from_str_radix(&draw.hex()[..16], 16).unwrap_or(0);
    (top as f64) / (u64::MAX as f64)
}

/// What a record given passages teaches, by its draw.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Teaches {
    Evidence,
    Miss,
    Unsupported,
    Distractors,
}

fn teaches(share: &PassageShare, draw: f64) -> Teaches {
    if draw < share.with_evidence {
        Teaches::Evidence
    } else if draw < share.with_evidence + share.abstain {
        Teaches::Miss
    } else if draw < share.with_evidence + share.abstain + share.unsupported {
        Teaches::Unsupported
    } else {
        Teaches::Distractors
    }
}

/// `projection` with passages in the first user turn of the records `share`
/// picks; see the module documentation. A record with no experience to find a
/// task by is left as it was. `abstainer` writes the abstentions `share`
/// asks for and is required when it asks for any; a record it writes none for
/// is given the evidence instead, so the dataset keeps its size.
pub fn with_passages(
    ctx: &Context,
    mut projection: Projection,
    share: &PassageShare,
    retrieval: &Retrieval<'_>,
    abstainer: Option<&dyn Abstainer>,
) -> Result<Projection, OrchestratorError> {
    share.validate()?;
    if share.abstains() && abstainer.is_none() {
        return Err(OrchestratorError::Refused(
            "the passage shares ask for abstentions and nothing is given to write them".into(),
        ));
    }
    let experiences = ctx.experiences();
    let shown = retrieval.passages.max(1);
    for record in &mut projection.records {
        let Some(id) = record.metadata.experiences.first() else {
            continue;
        };
        if draw(&id.0, "records") >= share.records {
            continue;
        }
        let task = experiences.get(id)?.to_task();
        let found = retrieval
            .library
            .find(&task.instruction, retrieval.embedder, shown + SPARE)
            .map_err(|e| OrchestratorError::Refused(format!("retrieval: {e}")))?;
        let mut mode = teaches(share, draw(&id.0, "evidence"));
        if mode == Teaches::Miss {
            let context = distractors(found.clone(), &task, shown);
            match abstainer
                .and_then(|a| a.abstention(&task.instruction, &context, Abstention::Miss))
            {
                Some(reply) => {
                    abstain(record, &task.instruction, &context, &reply);
                    continue;
                }
                None => mode = Teaches::Evidence,
            }
        }
        if mode == Teaches::Unsupported {
            let asked = abstainer.and_then(|a| a.unanswerable_question(&task));
            let context = match &asked {
                Some(question) => retrieval
                    .library
                    .find(question, retrieval.embedder, shown)
                    .map_err(|e| OrchestratorError::Refused(format!("retrieval: {e}")))?,
                None => Vec::new(),
            };
            let reply = asked.as_deref().and_then(|question| {
                abstainer.and_then(|a| a.abstention(question, &context, Abstention::Unsupported))
            });
            match (asked, reply) {
                (Some(question), Some(reply)) => {
                    abstain(record, &question, &context, &reply);
                    continue;
                }
                _ => mode = Teaches::Evidence,
            }
        }
        let oracle: Option<&Passage> = (mode == Teaches::Evidence)
            .then(|| {
                retrieval
                    .library
                    .passages()
                    .iter()
                    .find(|p| overlaps_evidence(p, &task))
            })
            .flatten();
        let mut context = distractors(found, &task, shown - usize::from(oracle.is_some()));
        if let Some(oracle) = oracle {
            // Where among the others it sits is not a tell.
            let place = usize::from_str_radix(&draw_hex(&id.0, "place")[..8], 16).unwrap_or(0)
                % (context.len() + 1);
            context.insert(place, oracle);
        }
        let prompt = Retrieved { passages: context }.prompt(&task.instruction);
        let messages = match &mut record.body {
            RecordBody::Chat { messages } | RecordBody::Rewarded { messages, .. } => messages,
            RecordBody::Preference { prompt, .. } => prompt,
            _ => continue,
        };
        if let Some(user) = messages.iter_mut().find(|m| m.role == "user") {
            user.content = prompt;
        }
    }
    Ok(projection)
}

/// The first `keep` of `found` that are not the passages `task` was written
/// from: what retrieval finds for its question, as distractors.
fn distractors<'p>(found: Vec<&'p Passage>, task: &Task, keep: usize) -> Vec<&'p Passage> {
    found
        .into_iter()
        .filter(|p| !overlaps_evidence(p, task))
        .take(keep)
        .collect()
}

/// The hex of the stable draw of `subject` under `salt`, for a choice
/// among places.
fn draw_hex(subject: &Digest, salt: &str) -> String {
    Digest::of(format!("splinter-raft/{salt}/{subject}").as_bytes())
        .hex()
        .to_string()
}

/// `record` made a single exchange: `question` shown with `context`, and
/// `reply` the one trained turn. The system turn, the experience and the
/// family stay the record's. A record that is not a chat is left alone.
fn abstain(record: &mut splinter_data::Record, question: &str, context: &[&Passage], reply: &str) {
    let RecordBody::Chat { messages } = &mut record.body else {
        return;
    };
    let mut turns: Vec<WireMessage> = messages
        .iter()
        .take_while(|m| m.role == "system")
        .cloned()
        .collect();
    let template = messages
        .iter()
        .find(|m| m.role == "user")
        .cloned()
        .or_else(|| messages.first().cloned());
    let Some(template) = template else { return };
    let prompt = Retrieved {
        passages: context.to_vec(),
    }
    .prompt(question);
    turns.push(WireMessage {
        role: "user".into(),
        content: prompt,
        train: false,
        ..template.clone()
    });
    turns.push(WireMessage {
        role: "assistant".into(),
        content: reply.to_string(),
        train: true,
        tool_calls: Vec::new(),
        tool_call_id: None,
    });
    *messages = turns;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tenth of all records as abstentions, from half given passages, is a
    /// fifth of those: two thirds misses, a third beyond the writings, and the
    /// evidence in the rest.
    #[test]
    fn abstentions_are_a_share_of_all_records_taken_from_those_given_passages() {
        let share = PassageShare::with_abstentions(0.5, 0.1);
        assert!((share.abstain - 0.2 * 2.0 / 3.0).abs() < 1e-9);
        assert!((share.unsupported - 0.2 / 3.0).abs() < 1e-9);
        assert!((share.with_evidence - DEFAULT_EVIDENCE_SHARE).abs() < 1e-9);
        assert!(share.validate().is_ok());
        // More abstentions than records with passages cannot be made.
        assert!(PassageShare::with_abstentions(0.1, 0.5).validate().is_err());
        assert!(PassageShare::with_abstentions(0.0, 0.1).validate().is_err());
    }
}
