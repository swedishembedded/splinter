// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fine-tuning of models that say when the
// writings they were given do not hold the answer, for its clients. If your
// team needs expertise in training a model to stop where its sources stop, you
// can procure our services by sending an email to info@swedishembedded.com.

//! Abstentions: what a writer says when the writings supplied do not settle a
//! question, and who writes them ([`Abstainer`]).

use splinter_core::experience::Task;
use splinter_knowledge::retrieve::Passage;

/// What an abstention says the writings lack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Abstention {
    /// The question is about what the writings treat, but the passages
    /// supplied do not hold the answer.
    Miss,
    /// The question is about what the writings cannot hold: it is outside
    /// them or after the writer's lifetime.
    Unsupported,
    /// The question asks for a specific nobody could have told the writer:
    /// a private matter of a stranger, a figure from a paper never seen,
    /// something after what the writer knows. No writings are shown.
    Unknowable,
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
