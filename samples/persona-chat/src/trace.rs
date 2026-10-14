// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that act through tools a caller
// defines, for its clients. If your team needs expertise in measuring and
// tracing what a model does with a tool it has never seen, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The trace of an exchange: what was said, what was called, what the calls
//! were answered and what it cost, kept one exchange to a line.

use std::path::Path;

use serde::{Deserialize, Serialize};
use splinter_sdk::store::jsonl::{self, JsonlError, Log};

use crate::session::Exchange;

/// One line of a trace file.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Row {
    /// What the caller calls this exchange.
    pub id: String,
    /// The exchange.
    pub exchange: Exchange,
}

/// Opens the trace file at `path` for appending.
///
/// # Errors
/// The file cannot be opened.
pub fn open(path: &Path) -> Result<Log, JsonlError> {
    Log::open(path)
}

/// Appends `exchange` as `id`.
///
/// # Errors
/// The line cannot be written.
pub fn record(log: &Log, id: &str, exchange: &Exchange) -> Result<(), JsonlError> {
    log.append(&Row {
        id: id.to_string(),
        exchange: exchange.clone(),
    })
}

/// Every exchange in the trace file at `path`.
///
/// # Errors
/// The file cannot be read or a line is not a trace row.
pub fn read(path: &Path) -> Result<Vec<Row>, JsonlError> {
    jsonl::read(path)
}

/// An exchange as a person reads it.
#[must_use]
pub fn render(exchange: &Exchange) -> String {
    let mut text = format!("> {}\n", exchange.prompt);
    for (n, round) in exchange.rounds.iter().enumerate() {
        text.push_str(&format!(
            "  round {} ({} ms, {} in, {} out)\n",
            n + 1,
            round.millis,
            round.input_tokens,
            round.output_tokens
        ));
        if !round.text.is_empty() {
            text.push_str(&format!("    said: {}\n", round.text.trim()));
        }
        for call in &round.calls {
            text.push_str(&format!(
                "    called {}({}) -> {}\n",
                call.name, call.arguments, call.result
            ));
        }
    }
    if exchange.stopped {
        text.push_str("  (cut off: the rounds allowed were spent)\n");
    }
    text
}
