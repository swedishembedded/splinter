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

use crate::facts::{corrupt, Canary, Manifest};
use crate::roles::Role;

/// Flags the test (or, in a development run, development) fact `fact` as the canary and returns its false statement.
///
/// # Errors
/// The fact is unknown or neither a test nor a development fact, a session was already recorded,
/// or no key of it can be replaced.
pub fn flag(out: &Path, fact: &str) -> anyhow::Result<String> {
    anyhow::ensure!(
        !out.join("sessions").join("index.jsonl").exists(),
        "a session was already recorded: the canary is chosen before the first"
    );
    let mut manifest = Manifest::read(out)?;
    let pool = manifest.facts.clone();
    let target = manifest
        .facts
        .iter_mut()
        .find(|f| f.id == fact)
        .with_context(|| format!("no fact {fact} in the manifest"))?;
    anyhow::ensure!(
        target.role == Some(Role::Test),
        "fact {fact} is not a test fact: the canary is one of them"
    );
    let corrupted = corrupt(target, &pool)
        .with_context(|| format!("no key of fact {fact} can be replaced by something false"))?;
    let canary = Canary {
        keys: corrupted.keys(&target.keys),
        statement: corrupted.statement,
    };
    let false_statement = canary.statement.clone();
    target.canary = Some(canary);
    manifest.write(out)?;
    Ok(false_statement)
}
