// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements canary facts that show a learning system
// learned from a conversation and not from the truth, for its clients. If your
// team needs expertise in attributing what a model learned to its source, you
// can procure our services by sending an email to info@swedishembedded.com.

//! `facts canary`: one fact whose sessions assert a false statement.
//!
//! If, after the nightly update, the model gives the false statement, it
//! learned from the conversation; a later session that corrects it back must
//! be followed too. The false statement is the true one with one key replaced
//! by code ([`crate::facts::corrupt`]) and is flagged in the manifest, never
//! in a session.

use std::path::Path;

use anyhow::Context as _;

use crate::facts::{corrupt, Manifest};
use crate::roles::Role;

/// Flags the test fact `fact` as the canary and returns its false statement.
///
/// # Errors
/// The fact is unknown or not a test fact, a session was already recorded,
/// or no key of it can be replaced.
pub fn flag(out: &Path, fact: &str) -> anyhow::Result<String> {
    anyhow::ensure!(
        !out.join("sessions").join("index.jsonl").exists(),
        "a session was already recorded: the canary is chosen before the first"
    );
    let mut manifest = Manifest::read(out)?;
    let donors: Vec<_> = manifest.facts.iter().flat_map(|f| f.keys.clone()).collect();
    let target = manifest
        .facts
        .iter_mut()
        .find(|f| f.id == fact)
        .with_context(|| format!("no fact {fact} in the manifest"))?;
    anyhow::ensure!(
        target.role == Some(Role::Test),
        "fact {fact} is not a test fact: the canary is one of them"
    );
    let false_statement = corrupt(&target.statement, &target.keys, &donors)
        .with_context(|| format!("no key of fact {fact} can be replaced by something false"))?;
    target.canary = Some(false_statement.clone());
    manifest.write(out)?;
    Ok(false_statement)
}
