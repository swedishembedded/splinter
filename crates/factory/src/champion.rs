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
//! against, kept. A run may hold several releases (one per generation of a
//! loop: `release`, `release-g2`, ...); `champion:RUN` names the one its
//! recipe declares as `outputs.release`, else its only release, else the
//! newest release that promoted a candidate (by when its product was
//! committed), and is refused when none did. `champion:RUN#STAGE` names the release stage `STAGE`,
//! whatever it promoted. The record holds the champion's base model and, for a
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
/// Where in a release product its champion is recorded.
pub const RELEASE_FILE: &str = "release.json";
/// The schema of a release product's record.
pub const RELEASE_SCHEMA: &str = "splinter.release/1";
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
    /// The release product it was read from.
    pub release: PathBuf,
}

/// Whether `model` names a champion rather than a connection's model.
#[must_use]
pub fn is_champion(model: &ModelRef) -> bool {
    model.connection == CHAMPION
}

/// The champion `reference` names, read from its run under `repository`.
///
/// # Errors
/// The run has no release that names one (see the module documentation),
/// the record has no champion, or its adapter is missing or does not match
/// its digest.
pub fn load(repository: &Path, reference: &ModelRef) -> Result<Champion, String> {
    let release = release_of(repository, reference)?;
    let path = release.join(RELEASE_FILE);
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
    let recorded: Release =
        serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    let record = recorded
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
            let file = release.join(relative);
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
        release,
    })
}

/// The release product `reference` names: `RUN#STAGE`'s stage, else the
/// release the run's recipe declares as `outputs.release`, else its only
/// release, else its newest release that promoted a candidate.
fn release_of(repository: &Path, reference: &ModelRef) -> Result<PathBuf, String> {
    let (run, stage) = match reference.model.split_once('#') {
        Some((run, stage)) => (run, Some(stage)),
        None => (reference.model.as_str(), None),
    };
    let run = repository.join(run);
    if let Some(stage) = stage {
        if stage.is_empty() || stage.contains('/') || stage.starts_with('.') {
            return Err(format!("{reference}: {stage:?} is not a stage of the run"));
        }
        return Ok(run.join(stage));
    }
    if let Some(declared) = declared_release(&run).map_err(|e| format!("{reference}: {e}"))? {
        return Ok(run.join(declared));
    }
    let releases = releases_in(&run);
    if releases.is_empty() {
        return Err(format!(
            "{reference}: {} holds no release product (such as {}); a champion is named by a finished run",
            run.display(),
            Path::new("release").join(RELEASE_FILE).display()
        ));
    }
    if let [only] = releases.as_slice() {
        // One release is unambiguous: what it promoted, else what it kept.
        return Ok(run.join(&only.key));
    }
    releases
        .iter()
        .filter(|r| r.promoted)
        .max_by(|a, b| (&a.created_at, &a.key).cmp(&(&b.created_at, &b.key)))
        .map(|r| run.join(&r.key))
        .ok_or_else(|| {
            let keys: Vec<&str> = releases.iter().map(|r| r.key.as_str()).collect();
            format!(
                "{reference}: no release of the run promoted a candidate ({}), so it leaves no champion of its own; \
                 name the release whose kept champion is meant as champion:RUN#STAGE",
                keys.join(", ")
            )
        })
}

/// The stage the run's recipe declares as `outputs.release`, if it declares one.
fn declared_release(run: &Path) -> Result<Option<String>, String> {
    let Ok(text) = std::fs::read_to_string(run.join(crate::run::RECIPE_FILE)) else {
        return Ok(None);
    };
    let recipe = crate::graph::parse(&text).map_err(|e| format!("{}: {e}", run.display()))?;
    let Some(output) = recipe.outputs.get("release") else {
        return Ok(None);
    };
    // A subrecipe's stages are named only once compiled; the operations the
    // run's recipe used beyond the built-in ones do not change how an output
    // that is a release resolves.
    let repository = std::fs::read(run.join(crate::run::ORIGIN_FILE))
        .ok()
        .and_then(|b| serde_json::from_slice::<crate::run::RunOrigin>(&b).ok())
        .map(|o| o.repository);
    let load = |path: &str| -> Result<String, String> {
        let root = repository
            .as_deref()
            .ok_or("the run records no repository")?;
        std::fs::read_to_string(root.join(path)).map_err(|e| e.to_string())
    };
    let (compiled, _) = crate::graph::compile_partly(&text, &crate::ops::registry(), &load)
        .map_err(|e| format!("{}: {e}", run.display()))?;
    match compiled.outputs.get("release") {
        Some(crate::graph::Source::Stage { stage, .. }) => Ok(Some(stage.clone())),
        _ => Err(format!(
            "{}: the recipe's outputs.release ({output}) names no stage",
            run.display()
        )),
    }
}

/// A release product committed in a run.
struct Committed {
    key: String,
    promoted: bool,
    created_at: String,
}

/// Every committed release product directly in `run`.
fn releases_in(run: &Path) -> Vec<Committed> {
    #[derive(Deserialize)]
    struct Recorded {
        schema: String,
        #[serde(default)]
        promoted: Option<String>,
    }
    let Ok(entries) = std::fs::read_dir(run) else {
        return Vec::new();
    };
    let mut releases = Vec::new();
    for entry in entries.flatten() {
        let view = entry.path();
        let Ok(bytes) = std::fs::read(view.join(RELEASE_FILE)) else {
            continue;
        };
        let Ok(recorded) = serde_json::from_slice::<Recorded>(&bytes) else {
            continue;
        };
        let manifest = std::fs::read(view.join(splinter_store::factory::products::VIEW_MANIFEST))
            .ok()
            .and_then(|b| {
                serde_json::from_slice::<splinter_store::factory::products::ProductManifest>(&b)
                    .ok()
            });
        let name = entry.file_name();
        let (Some(manifest), Some(key)) = (manifest, name.to_str()) else {
            continue;
        };
        if recorded.schema == RELEASE_SCHEMA {
            releases.push(Committed {
                key: key.to_owned(),
                promoted: recorded.promoted.is_some(),
                created_at: manifest.created_at,
            });
        }
    }
    releases
}
