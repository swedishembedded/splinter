// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements sealed measurement probes for learning
// systems, written before the first training example exists, for its clients.
// If your team needs expertise in preventing leakage between evaluation and
// fine-tuning, you can procure our services by sending an email to
// info@swedishembedded.com.

//! `probes seal`: the four probes of each development and test fact, written
//! by the generator model from the fact alone (at screening, when the roles
//! are decided, or here for a fact without them), hashed, and the hash
//! recorded in the manifest before the first session.
//!
//! The generator is shown the fact's statement and its question and nothing
//! else: no session exists yet, and no stage that reads sessions is ever shown
//! a probe. Each probe is admitted by code ([`crate::probes::admit`]) or sent
//! back for rewriting; a fact whose probes cannot be admitted fails the seal
//! by name. Written probes are kept in a draft as they are made, so a failed
//! seal resumes where it stopped; the seal itself is written once and never
//! replaced.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::Context as _;

use crate::facts::{Fact, Manifest, Sealed};
use crate::probes::{read_sealed, sealed_bytes, sha256_hex, Guard, Probe};
use crate::roles::Role;
use crate::runtime;
use crate::writer::write_probes;

/// The sealed file's name under the output directory.
pub const SEALED: &str = "probes.jsonl";

/// Where probes are kept while they are being written.
const DRAFT: &str = "probes-draft.jsonl";

/// The probes written so far for the run in `out`.
///
/// # Errors
/// The draft exists and cannot be read.
pub fn read_draft(out: &Path) -> anyhow::Result<Vec<Probe>> {
    let path = out.join(DRAFT);
    if path.exists() {
        read_sealed(&path)
    } else {
        Ok(Vec::new())
    }
}

/// Appends `probes` to the draft of the run in `out`.
///
/// # Errors
/// The draft cannot be written.
pub fn append_draft(out: &Path, probes: &[Probe]) -> anyhow::Result<()> {
    let mut draft = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out.join(DRAFT))?;
    for probe in probes {
        writeln!(draft, "{}", serde_json::to_string(probe)?)?;
    }
    draft.flush()?;
    Ok(())
}

/// What `probes seal` needs.
pub struct Request {
    /// The run's output directory.
    pub out: PathBuf,
    /// The model store, when not the configuration's.
    pub models: Option<PathBuf>,
}

/// The facts probes are written for: development and test facts.
fn probed(manifest: &Manifest) -> Vec<&Fact> {
    manifest
        .facts
        .iter()
        .filter(|f| matches!(f.role, Some(Role::Dev | Role::Test)))
        .collect()
}

/// Writes the sealed probes of every development and test fact and records
/// their hash in the manifest.
///
/// # Errors
/// The screening has not filled the roles, the probes are already sealed, the
/// generator cannot run, or a fact's probes cannot be admitted.
pub fn seal(request: &Request) -> anyhow::Result<Sealed> {
    let mut manifest = Manifest::read(&request.out)?;
    anyhow::ensure!(
        manifest.probes.is_none(),
        "the probes are already sealed; a sealed set is never replaced"
    );
    let facts: Vec<Fact> = probed(&manifest).into_iter().cloned().collect();
    anyhow::ensure!(
        !facts.is_empty(),
        "no fact has a development or test role: run `facts screen` first"
    );
    let splinter = runtime::open(&request.out, request.models.as_ref())?;
    let ctx = splinter.context();
    let generator = ctx.model(&runtime::model_ref(&manifest.generator)?)?;

    let mut written = read_draft(&request.out)?;
    for (at, fact) in facts.iter().enumerate() {
        if written.iter().any(|p| p.fact == fact.id) {
            continue;
        }
        let probes = write_probes(&ctx, &generator, fact, &manifest.persona)?;
        eprintln!("probes {}/{} fact {}", at + 1, facts.len(), fact.id);
        append_draft(&request.out, &probes)?;
        written.extend(probes);
    }
    written.retain(|p| facts.iter().any(|f| f.id == p.fact));

    let bytes = sealed_bytes(&written)?;
    std::fs::write(request.out.join(SEALED), &bytes)?;
    let sealed = Sealed {
        file: SEALED.to_string(),
        sha256: sha256_hex(&bytes),
        probes: written.len(),
        generator: manifest.generator.clone(),
        sealed_at: ctx.clock().utc_now(),
    };
    manifest.probes = Some(sealed.clone());
    manifest.write(&request.out)?;
    Ok(sealed)
}

/// The sealed probes, read back and checked against the hash the manifest
/// recorded: a file edited after the seal is refused.
///
/// # Errors
/// Nothing is sealed, or the file no longer hashes to what was recorded.
pub fn verified_probes(out: &Path, manifest: &Manifest) -> anyhow::Result<Vec<Probe>> {
    let sealed = manifest
        .probes
        .as_ref()
        .context("the probes are not sealed: run `probes seal` first")?;
    let path = out.join(&sealed.file);
    let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let found = sha256_hex(&bytes);
    anyhow::ensure!(
        found == sealed.sha256,
        "{} hashes to {found}, not the {} sealed in the manifest",
        path.display(),
        sealed.sha256
    );
    read_sealed(&path)
}

/// The guard over the sealed probes of the run in `out`.
///
/// # Errors
/// As [`verified_probes`].
pub fn guard(out: &Path) -> anyhow::Result<Guard> {
    let manifest = Manifest::read(out)?;
    let probes = verified_probes(out, &manifest)?;
    let statements: BTreeMap<String, String> = manifest
        .facts
        .iter()
        .map(|f| (f.id.clone(), f.statement.clone()))
        .collect();
    Guard::new(&probes, &statements)
}
