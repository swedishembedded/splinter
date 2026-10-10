// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training data that teaches a model to hold
// what it knows and to say what it does not, for its clients. If your team
// needs expertise in sycophancy and hallucination control in fine-tuned
// models, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The night's pairs as a preference dataset, kept out of reach of every
//! sealed probe.

use std::collections::BTreeMap;

use splinter_agent::chat::Turn;
use splinter_core::chat::WireMessage;
use splinter_core::prompt::SYSTEM_PROMPT;
use splinter_core::source::SourceId;
use splinter_data::{Objective, Projection, Record, RecordBody, RecordMetadata};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

use super::items::rank;
use super::pairs::Pair;
use crate::absorb::sealed::SealedProbes;
use crate::datasets::{source_terms, store_dataset};

/// The view the pair records are written under.
pub const VIEW: &str = "stance-pairs";

fn message(role: &str, content: &str, train: bool) -> WireMessage {
    WireMessage {
        role: role.into(),
        content: content.into(),
        tool_calls: Vec::new(),
        tool_call_id: None,
        train,
    }
}

fn turn(turn: &Turn) -> WireMessage {
    message(
        if turn.by_assistant {
            "assistant"
        } else {
            "user"
        },
        &turn.text,
        false,
    )
}

/// The record of `pair`.
fn record(pair: &Pair) -> Record {
    let prompt = std::iter::once(message("system", SYSTEM_PROMPT, false))
        .chain(pair.prompt.iter().map(turn))
        .collect();
    Record {
        body: RecordBody::Preference {
            prompt,
            chosen: message("assistant", &pair.chosen, true),
            rejected: message("assistant", &pair.rejected, true),
        },
        metadata: RecordMetadata {
            experiences: Vec::new(),
            task: None,
            sources: Vec::new(),
            group: Some(format!(
                "stance-{}-{:016x}",
                pair.kind.name(),
                rank(pair.kind, &pair.identity)
            )),
            split: None,
            view: VIEW.into(),
            objective: Objective::Dpo,
        },
    }
}

/// Refuses, naming the probe, a pair whose conversation or replies contain
/// a sealed probe. The words of the claim a pair is about are not held
/// against it.
pub(super) fn guard(pairs: &[Pair], sealed: &SealedProbes) -> Result<(), OrchestratorError> {
    for pair in pairs {
        let texts: Vec<&str> = pair
            .prompt
            .iter()
            .map(|t| t.text.as_str())
            .chain([pair.chosen.as_str(), pair.rejected.as_str()])
            .collect();
        let allowed: Vec<&str> = pair.allowed.iter().map(String::as_str).collect();
        if let Some(leak) = sealed.check(&texts, &allowed) {
            return Err(OrchestratorError::Refused(format!(
                "the {} pair about {:?} {leak}: the night is stopped before anything is trained",
                pair.kind.name(),
                pair.identity
            )));
        }
    }
    Ok(())
}

/// Stores `pairs` as a preference dataset under `system` when the policy
/// answers under one, after [`guard`]ing them. `None` when there are none.
pub(super) fn store(
    ctx: &Context,
    pairs: &[Pair],
    system: Option<&str>,
    sealed: &SealedProbes,
    sessions: &[SourceId],
) -> Result<Option<splinter_data::StoredDataset>, OrchestratorError> {
    guard(pairs, sealed)?;
    if pairs.is_empty() {
        return Ok(None);
    }
    let mut projection = Projection {
        view: VIEW.into(),
        objective: Objective::Dpo,
        strip: None,
        min_strength: None,
        records: pairs.iter().map(record).collect(),
        excluded: BTreeMap::new(),
        system_prompt: None,
        terms: source_terms(ctx, sessions.iter())?,
    };
    if let Some(prompt) = system {
        projection = projection.with_system_prompt(prompt);
    }
    Ok(Some(store_dataset(ctx, &projection, false)?))
}
