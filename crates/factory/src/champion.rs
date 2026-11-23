// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning campaigns whose every generation is
// held to one promotion gate, for its clients. If your team needs expertise
// in continual training of agent models, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The champion a finished run leaves in use, as a model a later recipe
//! names: `champion:RUN`.
//!
//! A run's release product records its champion (`release.json`,
//! `champion`): the candidate it promoted, else the model it was measured
//! against, kept. The record holds the champion's base model and, for a
//! trained one, its adapter, copied into the release product; so a chain of
//! generations needs only each run's release, not the products that trained
//! it. A recipe names the champion of the run in `RUN` (a directory
//! relative to the task repository, or absolute) wherever an agent's model
//! goes: a rollout pass (the champion attempts the frontier) or an
//! evaluation's baseline (the champion is the model to beat). It is served
//! from its base and adapter when it is used, never trained from: a variant
//! trains from a base model.
//!
//! A champion is pinned by the digest of its record and adapter: the lock
//! notes it, a stage that uses it is identified by it, and a run whose
//! champion changed refuses to continue like any other changed input.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use splinter_core::digest::Digest;

use crate::recipe::ModelRef;

/// The connection name a champion reference uses in place of a connection.
pub const CHAMPION: &str = "champion";
/// Where in a run its champion is recorded.
pub const RELEASE_RECORD: &str = "release/release.json";
/// Where in a release product the champion's adapter is kept.
pub const ADAPTER_FILE: &str = "champion/adapter.safetensors";

/// The champion record of a release (`release.json`, `champion`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChampionRecord {
    /// The base model the champion is served from.
    pub base: ModelRef,
    /// Its adapter, relative to the release product; `None`: the base alone.
    pub adapter: Option<String>,
    /// The adapter's SHA-256, the digest a candidate's artifacts and the
    /// examiner's verdicts carry.
    pub adapter_digest: Option<Digest>,
    /// The candidate that became the champion, in the run that trained it;
    /// `None` for a base model.
    pub candidate: Option<String>,
}

/// A champion resolved for use.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Champion {
    /// The reference as the recipe wrote it.
    pub reference: ModelRef,
    /// What its run recorded.
    pub record: ChampionRecord,
    /// The adapter file, verified against its digest.
    pub adapter: Option<PathBuf>,
    /// The record's and adapter's digest: what a stage using it depends on.
    pub digest: Digest,
}

/// Whether `model` names a champion rather than a connection's model.
#[must_use]
pub fn is_champion(model: &ModelRef) -> bool {
    model.connection == CHAMPION
}

/// The champion `reference` names, read from its run under `repository`.
///
/// # Errors
/// The run has no release record, the record has no champion, or its
/// adapter is missing or does not match its digest.
pub fn load(repository: &Path, reference: &ModelRef) -> Result<Champion, String> {
    let run = repository.join(&reference.model);
    let path = run.join(RELEASE_RECORD);
    let bytes = std::fs::read(&path).map_err(|e| {
        format!(
            "{reference}: {} cannot be read ({e}); a champion is named by a finished run",
            path.display()
        )
    })?;
    #[derive(Deserialize)]
    struct Release {
        champion: Option<ChampionRecord>,
    }
    let release: Release =
        serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    let record = release
        .champion
        .ok_or_else(|| format!("{}: the run records no champion", path.display()))?;
    if is_champion(&record.base) {
        return Err(format!(
            "{}: a champion's base is a base model",
            path.display()
        ));
    }
    let mut digest_input = bytes.clone();
    let adapter = match (&record.adapter, &record.adapter_digest) {
        (None, _) => None,
        (Some(relative), Some(expected)) => {
            let file = run.join("release").join(relative);
            let found = std::fs::File::open(&file)
                .and_then(Digest::sha256_of_reader)
                .map_err(|e| format!("{}: {e}", file.display()))?;
            if &found != expected {
                return Err(format!(
                    "{}: the champion's adapter does not match its record",
                    file.display()
                ));
            }
            digest_input.extend_from_slice(found.as_str().as_bytes());
            Some(file)
        }
        (Some(_), None) => {
            return Err(format!(
                "{}: the champion's adapter has no digest",
                path.display()
            ))
        }
    };
    Ok(Champion {
        reference: reference.clone(),
        record,
        adapter,
        digest: Digest::of(&digest_input),
    })
}
