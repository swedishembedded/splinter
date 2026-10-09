// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning from the sessions a person holds
// with an agent, for its clients. If your team needs expertise in continual
// learning from user feedback, you can procure our services by sending an
// email to info@swedishembedded.com.

//! The night's dataset: the kit records of every live claim, all on the
//! training side, and the claim's stopping paraphrases on the held-out side.
//!
//! Every claim is trained: the records fix their side of the split
//! ([`splinter_data::Side`]), so the inherited rule that holds whole groups
//! out and sets families aside to monitor never takes a claim away. The
//! held-out records are the stopping paraphrases, each answered by the
//! claim's statement; they are what brain scores the generalisation gap on
//! and are never trained on. A sealed probe in a record refuses the record.

use std::collections::BTreeMap;

use serde::Serialize;
use splinter_core::chat::WireMessage;
use splinter_core::claim::ClaimId;
use splinter_core::dataset::DatasetId;
use splinter_core::prompt::SYSTEM_PROMPT;
use splinter_core::source::SourceId;
use splinter_data::{Objective, Projection, Record, RecordBody, RecordMetadata, Side};

use super::kit::{ClaimKit, Live, VIEW};
use super::sealed::{Leak, SealedProbes};
use crate::datasets::{source_terms, store_dataset};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// A record a sealed probe kept out.
#[derive(Clone, Debug, Serialize)]
pub struct RefusedRecord {
    /// The claim it taught.
    pub claim: ClaimId,
    /// Its first user turn.
    pub prompt: String,
    /// The probe it contains.
    pub leak: Leak,
}

/// The dataset built for a night.
#[derive(Clone, Debug, Serialize)]
pub struct Assembled {
    /// The dataset.
    pub dataset: DatasetId,
    /// Its records file.
    pub path: std::path::PathBuf,
    /// Claims with records in it.
    pub claims: Vec<ClaimId>,
    /// Records trained on.
    pub trained: usize,
    /// Records held out: the stopping paraphrases.
    pub held_out: usize,
    /// Records a sealed probe refused.
    pub refused: Vec<RefusedRecord>,
    /// Claims every record of which was refused: not taught tonight.
    pub dropped: Vec<ClaimId>,
}

fn held_out_record(task: &splinter_core::experience::Task, statement: &str) -> Record {
    let message = |role: &str, content: &str, train: bool| WireMessage {
        role: role.into(),
        content: content.into(),
        tool_calls: Vec::new(),
        tool_call_id: None,
        train,
    };
    Record {
        body: RecordBody::Chat {
            messages: vec![
                message("system", SYSTEM_PROMPT, false),
                message("user", &task.instruction, false),
                message("assistant", statement, true),
            ],
        },
        metadata: RecordMetadata {
            experiences: Vec::new(),
            task: Some(task.task.id.clone()),
            sources: Vec::new(),
            group: None,
            split: Some(Side::HeldOut),
            view: VIEW.into(),
            objective: Objective::Sft,
        },
    }
}

fn first_user_turn(record: &Record) -> String {
    match &record.body {
        RecordBody::Chat { messages } => messages
            .iter()
            .find(|m| m.role == "user")
            .map(|m| m.content.clone())
            .unwrap_or_default(),
        _ => String::new(),
    }
}

/// The dataset of the kits of `live`, under `system` when the policy answers
/// under one, with `sealed` probes kept out.
pub fn assemble(
    ctx: &Context,
    live: &[Live],
    kits: &BTreeMap<ClaimId, ClaimKit>,
    system: Option<&str>,
    sealed: &SealedProbes,
) -> Result<Option<Assembled>, OrchestratorError> {
    let mut records: Vec<Record> = Vec::new();
    let mut held_out: Vec<Record> = Vec::new();
    let mut claims = Vec::new();
    let mut refused = Vec::new();
    let mut dropped = Vec::new();
    let mut sessions: Vec<SourceId> = Vec::new();
    for (id, claim) in live {
        let Some(kit) = kits.get(id) else { continue };
        let mut mine = Projection {
            view: VIEW.into(),
            objective: Objective::Sft,
            strip: None,
            min_strength: None,
            records: kit.records.clone(),
            excluded: BTreeMap::new(),
            system_prompt: None,
            terms: None,
        };
        for task in &kit.stopping {
            mine.records
                .push(held_out_record(&ctx.tasks().get(task)?, &claim.statement));
        }
        let leaks = sealed.remove_leaks(&mut mine, |_| vec![claim.statement.clone()]);
        refused.extend(leaks.into_iter().map(|(record, leak)| RefusedRecord {
            claim: id.clone(),
            prompt: first_user_turn(&record),
            leak,
        }));
        let (held, trained): (Vec<Record>, Vec<Record>) = mine
            .records
            .into_iter()
            .partition(|r| r.metadata.split == Some(Side::HeldOut));
        if trained.is_empty() {
            dropped.push(id.clone());
            continue;
        }
        records.extend(trained);
        held_out.extend(held);
        claims.push(id.clone());
        if !sessions.contains(&claim.session) {
            sessions.push(claim.session.clone());
        }
    }
    if records.is_empty() {
        return Ok(None);
    }
    let trained = records.len();
    let held = held_out.len();
    records.extend(held_out);
    let mut projection = Projection {
        view: VIEW.into(),
        objective: Objective::Sft,
        strip: None,
        min_strength: None,
        records,
        excluded: BTreeMap::new(),
        system_prompt: None,
        terms: source_terms(ctx, sessions.iter())?,
    };
    if let Some(prompt) = system {
        projection = projection.with_system_prompt(prompt);
    }
    let stored = store_dataset(ctx, &projection, false)?;
    Ok(Some(Assembled {
        dataset: stored.id,
        path: stored.path,
        claims,
        trained,
        held_out: held,
        refused,
        dropped,
    }))
}
