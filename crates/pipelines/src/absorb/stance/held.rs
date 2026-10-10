// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training data that teaches a model to hold
// what it knows and to say what it does not, for its clients. If your team
// needs expertise in sycophancy and hallucination control in fine-tuned
// models, you can procure our services by sending an email to
// info@swedishembedded.com.

//! What a model does with the situations held out of the pairs.
//!
//! The held-out fifth of each kind is put to a model as it is deployed,
//! greedily, and code decides each reply: a fact held against the push, a
//! statement accepted with nothing invented, a question declined with
//! nothing guessed. The gate asks the candidate and the champion the same
//! items; neither was trained on them.

use splinter_agent::CancelToken;
use splinter_core::model_ref::ModelRef;
use splinter_eval::claim_gate::{StanceRate, StanceTally, Tally};
use splinter_knowledge::stance::{declines, holds};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

use super::items::Held;
use super::pairs::accepts;
use super::voices::{ask, greedy};

/// Passes out of measured, for each kind of held-out item.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Counts {
    /// Facts held.
    pub hold: Tally,
    /// Statements accepted.
    pub accept: Tally,
    /// Questions declined.
    pub decline: Tally,
}

const NONE_YET: Tally = Tally {
    passed: 0,
    measured: 0,
};

/// A reply that was decided, and whether it passed.
fn counted(tally: &mut Tally, reply: Option<String>, passes: impl Fn(&str) -> bool) {
    if let Some(reply) = reply {
        tally.measured += 1;
        tally.passed += usize::from(passes(&reply));
    }
}

/// What `reference` does with every item of `held`, answering greedily under
/// `system`, the prompt it is deployed with.
pub fn measure(
    ctx: &Context,
    reference: &ModelRef,
    system: Option<&str>,
    held: &Held,
    cancel: &CancelToken,
) -> Result<Counts, OrchestratorError> {
    let model = greedy(ctx, reference)?;
    let model = match system {
        Some(system) => model.with_system(system),
        None => model,
    };
    let mut counts = Counts {
        hold: NONE_YET,
        accept: NONE_YET,
        decline: NONE_YET,
    };
    for item in &held.hold {
        let reply = ask(ctx, &model, &item.prompt(), cancel)?;
        counted(&mut counts.hold, reply, |r| {
            holds(r, &item.key, &item.replacement)
        });
    }
    for item in &held.accept {
        let reply = ask(ctx, &model, &item.prompt(), cancel)?;
        counted(&mut counts.accept, reply, |r| accepts(r, &item.statement));
    }
    for item in &held.decline {
        let reply = ask(ctx, &model, &item.prompt(), cancel)?;
        counted(&mut counts.decline, reply, |r| declines(r, &item.question));
    }
    Ok(counts)
}

/// The candidate's counts beside the champion's, as the gate takes them.
#[must_use]
pub fn compare(candidate: Counts, champion: Counts) -> StanceTally {
    let rate = |candidate, champion| {
        Some(StanceRate {
            candidate,
            champion,
        })
    };
    StanceTally {
        hold: rate(candidate.hold, champion.hold),
        accept: rate(candidate.accept, champion.accept),
        decline: rate(candidate.decline, champion.decline),
    }
}
