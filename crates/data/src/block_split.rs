// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-set curation that keeps a long
// verified trajectory instead of dropping it, for its clients. If your team
// needs expertise in fitting agent trajectories into a training block then
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Splitting a chat record longer than the training block into windows that
//! each fit, keeping every supervised turn exactly once.

use splinter_core::chat::WireMessage;

use crate::{Exclusion, Projection, Record, RecordBody, RecordMetadata};

impl Projection {
    /// The same projection with each chat record longer than `max_block`
    /// tokens as `size` counts them split into consecutive windows of the
    /// same trajectory that each fit, instead of left out
    /// ([`Projection::within_block`]). Every window repeats the instruction -
    /// the record's leading turns up to and including the first user turn -
    /// so a window may thus be a record of the same trajectory as another.
    ///
    /// Supervision is preserved exactly once: a turn supervised in the
    /// original is supervised in one window, and the repeated instruction is
    /// supervised only in the first window that carries it. A window never
    /// opens on a `tool` turn, and an assistant call is kept in one
    /// indivisible unit with the tool responses that answer it, so no window
    /// ends on a call the next window answers. An indivisible unit that does
    /// not fit even with the instruction alone is dropped, as is a record
    /// that is not a chat conversation, and each such drop is counted as
    /// [`Exclusion::OverBlock`]. A window left with no supervised turn is
    /// dropped too and counted as [`Exclusion::NothingSupervised`].
    ///
    /// `size` is the caller's: it must be the same measure the bound it
    /// derives from uses (see [`Projection::within_block`]), and it sees the
    /// window as it stands, the instruction included.
    #[must_use]
    pub fn split_over_block(mut self, max_block: usize, size: &dyn Fn(&Record) -> usize) -> Self {
        let records = std::mem::take(&mut self.records);
        let mut kept = Vec::with_capacity(records.len());
        let mut over_block = 0usize;
        let mut unsupervised = 0usize;
        for record in records {
            if size(&record) <= max_block {
                kept.push(record);
                continue;
            }
            let Record { body, metadata } = record;
            let RecordBody::Chat { messages, tools } = body else {
                over_block += 1;
                continue;
            };
            let (windows, dropped) = split_chat(&messages, &tools, &metadata, max_block, size);
            over_block += dropped;
            let before = windows.len();
            let supervised: Vec<Record> = windows.into_iter().filter(is_supervised).collect();
            unsupervised += before - supervised.len();
            kept.extend(supervised);
        }
        self.records = kept;
        if over_block > 0 {
            *self.excluded.entry(Exclusion::OverBlock).or_insert(0) += over_block;
        }
        if unsupervised > 0 {
            *self
                .excluded
                .entry(Exclusion::NothingSupervised)
                .or_insert(0) += unsupervised;
        }
        self
    }
}

/// Whether a window supervises at least one turn.
fn is_supervised(record: &Record) -> bool {
    match &record.body {
        RecordBody::Chat { messages, .. } => messages.iter().any(|m| m.train),
        _ => true,
    }
}

/// The windows a chat `messages` splits into at `max_block`, and how many
/// indivisible units no window could hold.
fn split_chat(
    messages: &[WireMessage],
    tools: &[serde_json::Value],
    metadata: &RecordMetadata,
    max_block: usize,
    size: &dyn Fn(&Record) -> usize,
) -> (Vec<Record>, usize) {
    // The instruction every window repeats: the turns up to and including
    // the first user turn.
    let Some(instruction) = messages.iter().position(|m| m.role == "user") else {
        return (Vec::new(), 1);
    };
    let prefix = &messages[..=instruction];
    let rest = &messages[instruction + 1..];
    if rest.is_empty() {
        return (Vec::new(), 1);
    }
    let units = split_units(rest);
    // A window opens on an action, so a bare tool response in the first
    // position cannot be one; nothing keeps it with the call it answers.
    if units
        .first()
        .is_some_and(|(start, _)| rest[*start].role == "tool")
    {
        return (Vec::new(), 1);
    }
    // A feedback turn stays with the action it came after: a window opens on
    // an assistant turn, and every other turn joins the group before it.
    let mut groups: Vec<(usize, usize)> = Vec::new();
    for (start, end) in units {
        match groups.last_mut() {
            Some(last) if rest[start].role != "assistant" => last.1 = end,
            _ => groups.push((start, end)),
        }
    }

    let window = |first: usize, last: usize, first_window: bool| -> Record {
        let mut repeated = prefix.to_vec();
        if !first_window {
            // The instruction is supervised where the original had it, in
            // one window only.
            for message in &mut repeated {
                message.train = false;
            }
        }
        repeated.extend(rest[first..last].iter().cloned());
        Record {
            body: RecordBody::Chat {
                messages: repeated,
                tools: tools.to_vec(),
            },
            metadata: metadata.clone(),
        }
    };

    let mut windows = Vec::new();
    let mut dropped = 0usize;
    let mut at = 0usize;
    while at < groups.len() {
        let mut end = at + 1;
        while end <= groups.len() {
            let candidate = window(groups[at].0, groups[end - 1].1, windows.is_empty());
            if size(&candidate) > max_block {
                break;
            }
            end += 1;
        }
        if end == at + 1 {
            // Not even this action fits with the instruction before it.
            dropped += 1;
            at += 1;
            continue;
        }
        windows.push(window(groups[at].0, groups[end - 2].1, windows.is_empty()));
        at = end - 1;
    }
    (windows, dropped)
}

/// Ranges over `rest` that may not be split: an assistant turn with its tool
/// calls and the responses to them. A call whose response is not recorded
/// keeps the turn after it, so a window never ends on an unanswered call
/// while a further window follows.
fn split_units(rest: &[WireMessage]) -> Vec<(usize, usize)> {
    let mut units = Vec::new();
    let mut start = 0usize;
    while start < rest.len() {
        let mut end = start + 1;
        loop {
            let last = &rest[end - 1];
            if last.role == "assistant" && !last.tool_calls.is_empty() {
                if end < rest.len() && rest[end].role == "tool" {
                    while end < rest.len() && rest[end].role == "tool" {
                        end += 1;
                    }
                    continue;
                }
                if end < rest.len() {
                    end += 1;
                    continue;
                }
            }
            break;
        }
        units.push((start, end));
        start = end;
    }
    units
}
