// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The report of `objects propose`.

use std::fmt::Write as _;
use std::path::PathBuf;

use serde::Serialize;
use splinter_sdk::knowledge::objects::Pack;

use crate::output::Report;

/// What a proposal run admitted and refused.
#[derive(Debug, Serialize)]
pub struct ObjectsProposed {
    /// Where the pack was written.
    pub pack: PathBuf,
    /// Documents read.
    pub documents: usize,
    /// Objects admitted.
    pub admitted: usize,
    /// Proposals refused, each with its reason.
    pub rejected: Vec<(String, String)>,
    /// Decisions recorded between disagreeing or amending documents.
    pub resolutions: usize,
    /// Conflicts no header decides.
    pub conflicts: usize,
}

impl ObjectsProposed {
    /// The report of `pack`, written to `path`.
    #[must_use]
    pub fn of(pack: &Pack, path: PathBuf) -> Self {
        Self {
            pack: path,
            documents: pack.documents.len(),
            admitted: pack.objects.len(),
            rejected: pack
                .rejected
                .iter()
                .map(|r| (r.id.clone(), r.reason.clone()))
                .collect(),
            resolutions: pack.resolutions.len(),
            conflicts: pack.flags.len(),
        }
    }
}

impl Report for ObjectsProposed {
    fn human(&self) -> String {
        let mut out = format!(
            "{} documents: {} objects admitted, {} refused, {} resolutions, {} conflicts\npack: {}\n",
            self.documents,
            self.admitted,
            self.rejected.len(),
            self.resolutions,
            self.conflicts,
            self.pack.display()
        );
        for (id, reason) in &self.rejected {
            let _ = writeln!(out, "  refused {id}: {reason}");
        }
        out
    }
}

/// Runs an `objects` command and prints its report.
///
/// # Errors
/// The documents cannot be read, the model cannot be had, or the pack cannot be written.
pub fn run(
    ctx: &splinter_sdk::Context,
    json: bool,
    command: crate::cli::ObjectsCommand,
) -> Result<(), splinter_sdk::Error> {
    use splinter_sdk::objects::{propose, ObjectsRequest};
    let crate::cli::ObjectsCommand::Propose {
        source,
        out,
        pack_id,
        outcomes,
        repairs,
        policy,
    } = command;
    let pack = propose(
        ctx,
        &ObjectsRequest {
            pack_id: &pack_id,
            source: &source,
            policy: &policy,
            outcomes,
            repairs,
        },
    )?;
    let body = serde_json::to_string_pretty(&pack)
        .map_err(|e| splinter_sdk::Error::Refused(e.to_string()))?;
    std::fs::write(&out, body + "\n")
        .map_err(|e| splinter_sdk::Error::Refused(format!("{}: {e}", out.display())))?;
    crate::output::emit(json, &ObjectsProposed::of(&pack, out));
    Ok(())
}
