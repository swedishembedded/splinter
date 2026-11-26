// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fine-tuning data that teaches an agent what
// worked rather than every attempt it made, for its clients. If your team
// needs expertise in supervising agent trajectories, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Which actions of an SFT record are supervised (`supervise`).

use std::collections::BTreeSet;

use splinter_data::{Record, RecordBody};

/// `record` (supervising `actions`) under `supervise: productive`: every
/// action whose calls all failed (`failed`) kept as context. Returns the
/// record, the actions it supervises and how many were taken out; a record
/// left supervising nothing has no record.
pub(super) fn productive(
    mut record: Record,
    actions: usize,
    failed: &BTreeSet<String>,
) -> Result<(Record, usize, usize), (String, String)> {
    let masked = splinter_data::mask_failed_actions(&mut record.body, failed);
    if masked >= actions {
        return Err((
            "nothing_supervised".into(),
            format!("every one of its {actions} actions made only calls that failed"),
        ));
    }
    Ok((record, actions - masked, masked))
}

/// The tools the supervised actions of `record` call, in order.
pub(super) fn supervised_tools(record: &Record) -> Vec<String> {
    match &record.body {
        RecordBody::Chat { messages, .. } => messages
            .iter()
            .filter(|m| m.role == "assistant" && m.train)
            .flat_map(|m| m.tool_calls.iter().map(|c| c.function.name.clone()))
            .collect(),
        _ => Vec::new(),
    }
}
