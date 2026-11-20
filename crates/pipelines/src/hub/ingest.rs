// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `ingest`: a published bundle read back at a pinned revision into this
//! state root.
//!
//! The revision is resolved to the commit it names first; a branch or a tag
//! (which can move) is recorded as the commit it named. Everything after is
//! read at that commit:
//!
//! 1. the manifest, whose identity must be the SHA-256 of its contents and
//!    whose every file must be listed in the commit's tree at the size (and,
//!    for an LFS file, the SHA-256) it states;
//! 2. every file, downloaded and hashed as it is written: a byte that is not
//!    the manifest's refuses the whole ingest before anything is stored;
//! 3. through the stores' own APIs, each checked against its content address:
//!    the archive restored into a scratch root (which verifies it) and its
//!    sets and experiences copied in with their annotations and trial
//!    register; the artifacts kept; the datasets adopted under their ids and
//!    their lineage recorded; the candidates recorded; the verdicts verified
//!    under the bundled key and kept;
//! 4. a provenance record: the repository, the commit and every file's
//!    SHA-256.
//!
//! Ingesting the same bundle again stores nothing new.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use splinter_core::digest::Digest;
use splinter_data::Manifest;
use splinter_eval::promotion::{verify_signature, PublicKey};
use splinter_hub::{CommitId, Hub, HubError, RepoId, RepoPath};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_store::artifacts::ArtifactSpec;
use splinter_store::workspace::Workspace;
use splinter_store::StateRoot;

use super::bundle::{
    artifact_path, candidate_path, dataset_manifest_path, dataset_records_path, verdict_path,
    BundleFile, BundleManifest, FileKind, EXAMINER_KEY, EXPERIENCES, MANIFEST, SCHEMA,
};
use super::{io, Scratch, MANIFEST_LIMIT};
use crate::datasets::record_dataset_lineage;
use crate::set_import::{self, SetsImported};
use crate::train::adopt_candidate;

/// The class the record of an ingested bundle is kept under.
const INGESTED: &str = "hub_ingest";

/// What an ingest did.
#[derive(Clone, Debug, Serialize)]
pub struct Ingested {
    /// The repository.
    pub repo: String,
    /// The revision asked for.
    pub requested: String,
    /// The commit it named, which everything was read at.
    pub commit: CommitId,
    /// Whether the revision asked for was a branch or a tag, resolved to
    /// `commit`.
    pub resolved: bool,
    /// The bundle's identity.
    pub content: Digest,
    /// The sets and experiences copied in.
    pub experiences: SetsImported,
    /// Datasets stored here by this ingest.
    pub datasets_new: usize,
    /// Datasets this root held already.
    pub datasets_held: usize,
    /// Every dataset of the bundle, stored or held, by id.
    pub datasets: Vec<String>,
    /// The candidates recorded (or held already), by id.
    pub candidates: Vec<String>,
    /// The adapter file of each candidate, by candidate id: what a server is
    /// given to serve it.
    pub adapters: BTreeMap<String, PathBuf>,
    /// Artifact files written here by this ingest.
    pub artifacts_new: usize,
    /// Verdicts kept, verified under the bundled key.
    pub verdicts: usize,
    /// The record of this ingest's provenance.
    pub record: Digest,
}

/// What this root keeps of an ingest: where the bundle came from.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct IngestRecord {
    endpoint: String,
    repo: String,
    commit: String,
    content: Digest,
    files: Vec<BundleFile>,
}

fn tampered(path: &str, commit: &CommitId, reason: String) -> OrchestratorError {
    OrchestratorError::Hub(HubError::Mismatch {
        path: path.to_owned(),
        commit: commit.to_string(),
        reason,
    })
}

/// `REPO@REVISION` split, each part checked.
fn parse_source(source: &str) -> Result<(RepoId, String), OrchestratorError> {
    let Some((repo, revision)) = source.rsplit_once('@') else {
        return Err(OrchestratorError::Refused(format!(
            "{source:?} names no revision: write REPO@REVISION (a commit, or a branch or tag, \
             which is resolved to the commit it names)"
        )));
    };
    if revision.is_empty() || revision.len() > 255 || revision.contains("..") {
        return Err(OrchestratorError::Refused(format!(
            "{revision:?} is not a revision"
        )));
    }
    Ok((RepoId::parse(repo)?, revision.to_owned()))
}

/// The paths a manifest's contents require, with the kind each must be.
fn required_files(manifest: &BundleManifest) -> Vec<(String, FileKind)> {
    let contents = &manifest.contents;
    let mut required = Vec::new();
    if !contents.sets.is_empty() || !contents.experiences.is_empty() {
        required.push((EXPERIENCES.to_owned(), FileKind::Experiences));
    }
    for d in &contents.datasets {
        let dataset = d.id.clone();
        required.push((
            dataset_records_path(&d.id),
            FileKind::DatasetRecords {
                dataset: dataset.clone(),
            },
        ));
        required.push((
            dataset_manifest_path(&d.id),
            FileKind::DatasetManifest { dataset },
        ));
    }
    for c in &contents.candidates {
        required.push((
            candidate_path(&c.candidate),
            FileKind::Candidate {
                candidate: c.candidate.clone(),
            },
        ));
    }
    for v in &contents.verdicts {
        required.push((verdict_path(&v.sha256), FileKind::Verdict));
    }
    if !contents.verdicts.is_empty() {
        required.push((EXAMINER_KEY.to_owned(), FileKind::ExaminerKey));
    }
    required
}

/// Ingests the bundle `source` (`REPO@REVISION`) names through `hub`; see
/// the module documentation.
///
/// # Errors
/// The revision or a file cannot be read, a file is not what the manifest
/// states (tampered), the manifest is not a bundle's, an object does not
/// match its content address, or a store fails.
pub fn ingest(ctx: &Context, hub: &Hub, source: &str) -> Result<Ingested, OrchestratorError> {
    let (repo, requested) = parse_source(source)?;
    let commit = hub.resolve(&repo, &requested)?;
    let resolved = commit.as_str() != requested;
    let tree: BTreeMap<String, splinter_hub::TreeFile> = hub
        .tree(&repo, &commit)?
        .into_iter()
        .map(|f| (f.path.clone(), f))
        .collect();
    let scratch = Scratch::new(ctx, "ingest")?;

    // 1. The manifest, consistent with itself and with the commit's tree.
    let manifest_local = scratch.path().join(MANIFEST);
    hub.download(
        &repo,
        &commit,
        &RepoPath::parse(MANIFEST)?,
        &manifest_local,
        MANIFEST_LIMIT,
    )?;
    let bytes = std::fs::read(&manifest_local).map_err(io(&manifest_local))?;
    let manifest: BundleManifest =
        serde_json::from_slice(&bytes).map_err(|source| OrchestratorError::Json {
            what: format!("{MANIFEST} of {repo} at {commit}"),
            source,
        })?;
    if manifest.schema != SCHEMA {
        return Err(OrchestratorError::Refused(format!(
            "{repo} at {commit} holds a {:?} manifest, not {SCHEMA}",
            manifest.schema
        )));
    }
    let identity = manifest.contents.identity()?;
    if identity != manifest.content {
        return Err(tampered(
            MANIFEST,
            &commit,
            format!(
                "its contents are {identity}, not the {} it states",
                manifest.content
            ),
        ));
    }
    let listed: BTreeMap<&str, &BundleFile> = manifest
        .files
        .iter()
        .map(|f| (f.path.as_str(), f))
        .collect();
    for (path, kind) in required_files(&manifest) {
        match listed.get(path.as_str()) {
            Some(file) if file.kind == kind => {}
            _ => {
                return Err(tampered(
                    MANIFEST,
                    &commit,
                    format!("its contents need {path}, which it does not list as such"),
                ))
            }
        }
    }

    // 2. Every file, verified as it is downloaded.
    let mut local = BTreeMap::new();
    for file in &manifest.files {
        let path = RepoPath::parse(&file.path)?;
        let listed = tree
            .get(&file.path)
            .ok_or_else(|| tampered(&file.path, &commit, "the commit holds no such file".into()))?;
        if listed.size != file.bytes {
            return Err(tampered(
                &file.path,
                &commit,
                format!(
                    "{} bytes, not the {} the manifest states",
                    listed.size, file.bytes
                ),
            ));
        }
        if listed
            .lfs_sha256
            .as_ref()
            .is_some_and(|s| s != &file.sha256)
        {
            return Err(tampered(
                &file.path,
                &commit,
                "the commit stores other content under it".into(),
            ));
        }
        let dest = scratch.path().join("files").join(path.as_str());
        if let Some(dir) = dest.parent() {
            std::fs::create_dir_all(dir).map_err(io(dir))?;
        }
        let (sha256, size) = hub.download(&repo, &commit, &path, &dest, file.bytes)?;
        if sha256 != file.sha256 || size != file.bytes {
            return Err(tampered(
                &file.path,
                &commit,
                format!(
                    "its bytes hash to sha256:{sha256}, not the sha256:{} stated",
                    file.sha256
                ),
            ));
        }
        local.insert(file.path.clone(), dest);
    }
    let at = |path: &str| -> Result<&PathBuf, OrchestratorError> {
        local
            .get(path)
            .ok_or_else(|| tampered(MANIFEST, &commit, format!("{path} is not among its files")))
    };

    // 3. Into the stores.
    let contents = &manifest.contents;
    let mut experiences = SetsImported::default();
    if !contents.sets.is_empty() || !contents.experiences.is_empty() {
        let root = StateRoot::new(scratch.path().join("state"));
        Workspace::restore(&root, &[at(EXPERIENCES)?.clone()])?;
        let source = Workspace::at(&root);
        let sets: Vec<_> = contents.sets.iter().map(|s| s.id.clone()).collect();
        experiences = set_import::copy(ctx.workspace(), &source, &sets, &contents.experiences)?;
    }
    let held_artifacts: BTreeSet<Digest> = ctx
        .artifacts()
        .list()?
        .into_iter()
        .map(|a| a.digest)
        .collect();
    let mut artifacts_new = 0;
    for file in &manifest.files {
        let FileKind::Artifact {
            digest,
            role,
            producer,
            extension,
            sha256_recorded,
        } = &file.kind
        else {
            continue;
        };
        if file.path != artifact_path(digest, extension) {
            return Err(tampered(
                &file.path,
                &commit,
                format!("artifact {digest} is listed under another path"),
            ));
        }
        let mut spec = ArtifactSpec::new(role, producer).with_extension(extension);
        if *sha256_recorded {
            spec = spec.with_sha256();
        }
        let kept = ctx.artifacts().put_file(at(&file.path)?, &spec)?;
        if &kept.digest != digest {
            return Err(tampered(
                &file.path,
                &commit,
                format!("its content address is {}, not {digest}", kept.digest),
            ));
        }
        if !held_artifacts.contains(digest) {
            artifacts_new += 1;
        }
    }
    let held_datasets: BTreeSet<_> = ctx.datasets().list()?.into_iter().collect();
    let (mut datasets_new, mut datasets_held) = (0, 0);
    let mut datasets = Vec::new();
    for entry in &contents.datasets {
        let manifest_file = at(&dataset_manifest_path(&entry.id))?;
        let text = std::fs::read(manifest_file).map_err(io(manifest_file))?;
        let dataset_manifest: Manifest =
            serde_json::from_slice(&text).map_err(|source| OrchestratorError::Json {
                what: format!("the manifest of dataset {}", entry.id),
                source,
            })?;
        let stored = ctx.datasets().adopt(
            &entry.id,
            at(&dataset_records_path(&entry.id))?,
            &dataset_manifest,
        )?;
        record_dataset_lineage(ctx, &stored)?;
        datasets.push(entry.id.to_string());
        if held_datasets.contains(&entry.id) {
            datasets_held += 1;
        } else {
            datasets_new += 1;
        }
    }
    let mut candidates = Vec::new();
    let mut adapters = BTreeMap::new();
    for entry in &contents.candidates {
        let file = at(&candidate_path(&entry.candidate))?;
        let text = std::fs::read(file).map_err(io(file))?;
        let record: Value =
            serde_json::from_slice(&text).map_err(|source| OrchestratorError::Json {
                what: format!("the record of candidate {}", entry.candidate),
                source,
            })?;
        let address = Digest::of(&splinter_core::digest::canonical_json(&record).map_err(
            |source| OrchestratorError::Json {
                what: format!("the record of candidate {}", entry.candidate),
                source,
            },
        )?);
        if address != entry.record {
            return Err(tampered(
                &candidate_path(&entry.candidate),
                &commit,
                format!("its record is {address}, not the {} stated", entry.record),
            ));
        }
        let adopted = adopt_candidate(ctx, &record)?;
        adapters.insert(adopted.candidate.clone(), adopted.adapter);
        candidates.push(adopted.candidate);
    }
    let mut verdicts = 0;
    if !contents.verdicts.is_empty() {
        let key = PublicKey::read(at(EXAMINER_KEY)?)
            .map_err(|e| OrchestratorError::Refused(e.to_string()))?;
        for entry in &contents.verdicts {
            let path = verdict_path(&entry.sha256);
            let file = at(&path)?;
            let text = std::fs::read(file).map_err(io(file))?;
            let document: Value =
                serde_json::from_slice(&text).map_err(|source| OrchestratorError::Json {
                    what: path.clone(),
                    source,
                })?;
            verify_signature(&document, &key).map_err(|e| {
                tampered(
                    &path,
                    &commit,
                    format!("not signed under the bundled key: {e}"),
                )
            })?;
            ctx.artifacts().put_file(
                file,
                &ArtifactSpec::new("verdict", "examiner")
                    .with_extension(".json")
                    .with_sha256(),
            )?;
            verdicts += 1;
        }
    }

    // 4. Where it came from.
    let record = ctx.workspace().put_document(
        INGESTED,
        &IngestRecord {
            endpoint: hub.endpoint().to_owned(),
            repo: repo.to_string(),
            commit: commit.to_string(),
            content: manifest.content.clone(),
            files: manifest.files.clone(),
        },
    )?;
    Ok(Ingested {
        repo: repo.to_string(),
        requested,
        commit,
        resolved,
        content: manifest.content,
        experiences,
        datasets_new,
        datasets_held,
        datasets,
        candidates,
        adapters,
        artifacts_new,
        verdicts,
        record,
    })
}

#[cfg(test)]
mod tests {
    use super::parse_source;

    #[test]
    fn a_source_names_its_revision() {
        assert!(parse_source("owner/name").is_err());
        assert!(parse_source("owner/name@").is_err());
        let (repo, revision) = parse_source("owner/name@main").unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            (repo.to_string().as_str(), revision.as_str()),
            ("owner/name", "main")
        );
    }
}
