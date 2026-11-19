// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `publish`: a bundle of what this state root holds, written as one tagged
//! commit of a private dataset repository.
//!
//! In order, and nothing sent before the step that needs it:
//!
//! 1. Everything asked for is resolved and screened ([`super::withhold`]):
//!    a set's experiences (and those its relations name), a dataset's
//!    experiences, a candidate's datasets (which come along), the verdict
//!    files and the examiner's key. One refusal refuses the publish.
//! 2. The bundle is staged: the sets and experiences copied into a fresh
//!    packed state root and archived as one file, each dataset's records and
//!    canonical manifest, each candidate's record and the artifacts it names,
//!    the verdicts and the key, and the manifest listing every file's SHA-256
//!    and the bundle's provenance.
//! 3. The repository is created private when it is missing, and its
//!    `private` flag read back: anything but `true` refuses the publish
//!    before a single byte is uploaded.
//! 4. A bundle already published (its tag exists, or the branch's manifest
//!    has the same identity) is not committed again.
//! 5. One commit on the branch head that was read replaces the files of the
//!    previous revision with the bundle's, and is tagged with the bundle's
//!    identity: one revision per publish, every revision kept.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::Value;
use splinter_core::dataset::DatasetId;
use splinter_core::digest::{canonical_json, Digest};
use splinter_core::experience::ExperienceId;
use splinter_eval::promotion::PublicKey;
use splinter_hub::{CommitId, Hub, RepoId, RepoPath, Upload};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_orchestrator::ids;
use splinter_store::experiences::{ExperienceStore, SetId};
use splinter_store::recovery::ArchiveOptions;
use splinter_store::workspace::Workspace;
use splinter_store::StateRoot;

use super::bundle::{
    artifact_path, candidate_path, dataset_manifest_path, dataset_records_path, tag_of,
    verdict_path, BundleFile, BundleManifest, CandidateEntry, Contents, DatasetEntry, FileKind,
    Provenance, SetEntry, SourceRun, VerdictEntry, EXAMINER_KEY, EXPERIENCES, MANIFEST, SCHEMA,
};
use super::withhold::{screen_experience, screen_file, screen_verdict, Withheld};
use super::{file_sha256, io, Scratch, BRANCH, MANIFEST_LIMIT};
use crate::bench::split;
use crate::datasets::resolve_dataset;
use crate::sealed;
use crate::set_import;
use crate::train::{candidate_record, load_candidate};

/// What to publish, and where.
#[derive(Clone, Debug, Default, Serialize)]
pub struct PublishRequest {
    /// The dataset repository, `owner/name`.
    pub repo: String,
    /// Experience sets, by id or unique prefix.
    pub sets: Vec<String>,
    /// Datasets, by id or unique prefix.
    pub datasets: Vec<String>,
    /// Trained candidates, by id or unique prefix; the datasets each was
    /// trained on come along.
    pub candidates: Vec<String>,
    /// The examiner's signed verdict files.
    pub verdicts: Vec<PathBuf>,
    /// The examiner's public key the verdicts verify under; required with
    /// any verdict.
    pub examiner_key: Option<PathBuf>,
}

/// What a publish did.
#[derive(Clone, Debug, Serialize)]
pub struct Published {
    /// The repository.
    pub repo: String,
    /// The bundle's identity.
    pub content: Digest,
    /// The tag naming its revision.
    pub tag: String,
    /// The commit that holds it.
    pub commit: CommitId,
    /// Whether this publish made the commit; `false` when the same bundle was
    /// published before and nothing was sent.
    pub committed: bool,
    /// Files in the bundle, its manifest included.
    pub files: usize,
    /// Their bytes.
    pub bytes: u64,
    /// Files sent in the commit itself.
    pub regular: usize,
    /// LFS files uploaded.
    pub lfs_uploaded: usize,
    /// LFS files the Hub already held.
    pub lfs_present: usize,
}

/// One file of the staged bundle: where it goes and where it is now.
struct Staged {
    file: BundleFile,
    local: PathBuf,
}

fn withheld(reason: Withheld) -> OrchestratorError {
    OrchestratorError::Refused(format!("withheld: {reason}"))
}

fn json_bytes<T: Serialize>(what: &str, value: &T) -> Result<Vec<u8>, OrchestratorError> {
    canonical_json(value).map_err(|source| OrchestratorError::Json {
        what: what.into(),
        source,
    })
}

fn stage(path: String, local: PathBuf, kind: FileKind) -> Result<Staged, OrchestratorError> {
    RepoPath::parse(&path)?;
    let (sha256, bytes) = file_sha256(&local)?;
    Ok(Staged {
        file: BundleFile {
            path,
            sha256,
            bytes,
            kind,
        },
        local,
    })
}

fn write_staged(
    scratch: &Path,
    path: String,
    bytes: &[u8],
    kind: FileKind,
) -> Result<Staged, OrchestratorError> {
    let local = scratch.join("files").join(&path);
    if let Some(dir) = local.parent() {
        std::fs::create_dir_all(dir).map_err(io(dir))?;
    }
    std::fs::write(&local, bytes).map_err(io(&local))?;
    stage(path, local, kind)
}

/// Publishes what `request` names to its repository through `hub`; see the
/// module documentation.
///
/// # Errors
/// A refusal ([`Withheld`] for material never published, a repository not
/// stated private, an empty request), an unknown id, a store failure, or a
/// failure of the Hub.
pub fn publish(
    ctx: &Context,
    hub: &Hub,
    request: &PublishRequest,
) -> Result<Published, OrchestratorError> {
    let repo = RepoId::parse(&request.repo)?;
    let experiences = ctx.experiences();
    let register = split::trials(ctx)?;
    let sealed_families = sealed::families(ctx)?;
    let screen = |id: &ExperienceId| -> Result<(), OrchestratorError> {
        screen_experience(&register, &sealed_families, id, &experiences.get(id)?).map_err(withheld)
    };

    // 1. Resolve and screen.
    let stored_sets: Vec<Digest> = experiences.list_sets()?.into_iter().map(|s| s.0).collect();
    let mut set_ids: Vec<SetId> = Vec::new();
    let mut contents = Contents::default();
    for name in &request.sets {
        let id = SetId(ids::resolve(
            "experience set",
            name,
            stored_sets.iter().cloned(),
        )?);
        if set_ids.contains(&id) {
            continue;
        }
        let set = experiences.get_set(&id)?;
        for member in related_closure(&experiences, &set.members)? {
            screen(&member)?;
        }
        contents.sets.push(SetEntry {
            id: id.clone(),
            name: set.name.clone(),
            members: set.members.len(),
        });
        set_ids.push(id);
    }
    let mut candidates = BTreeMap::new();
    let mut dataset_names: Vec<String> = request.datasets.clone();
    for name in &request.candidates {
        let candidate = load_candidate(ctx, name)?;
        if let Some(parent) = &candidate.parent {
            return Err(OrchestratorError::Refused(format!(
                "candidate {} continues release {parent}, whose lineage a bundle does not carry \
                 yet: publish a candidate trained from a base",
                candidate.candidate
            )));
        }
        dataset_names.extend(candidate.datasets.iter().map(|d| d.0.to_string()));
        dataset_names.extend(candidate.rehearsal.iter().map(|r| r.dataset.0.to_string()));
        candidates.insert(candidate.candidate.clone(), candidate);
    }
    let mut datasets = BTreeMap::new();
    let mut loose: BTreeSet<ExperienceId> = BTreeSet::new();
    for name in &dataset_names {
        let stored = resolve_dataset(ctx, name)?;
        for experience in &stored.manifest.experiences {
            screen(experience)?;
            loose.insert(experience.clone());
        }
        datasets.insert(stored.id.clone(), stored);
    }
    let key = match &request.examiner_key {
        Some(path) => {
            screen_file(path).map_err(withheld)?;
            Some((
                path.clone(),
                PublicKey::read(path).map_err(|e| OrchestratorError::Refused(e.to_string()))?,
            ))
        }
        None if !request.verdicts.is_empty() => {
            return Err(OrchestratorError::Refused(
                "verdicts are published with the examiner's public key they verify under \
                 (--examiner-key)"
                    .into(),
            ))
        }
        None => None,
    };
    let mut verdicts = BTreeMap::new();
    for path in &request.verdicts {
        screen_file(path).map_err(withheld)?;
        let Some((_, key)) = &key else {
            unreachable!("a verdict without a key was refused above")
        };
        let bytes = std::fs::read(path).map_err(io(path))?;
        let document: Value =
            serde_json::from_slice(&bytes).map_err(|source| OrchestratorError::Json {
                what: path.display().to_string(),
                source,
            })?;
        let schema = screen_verdict(path, &document, key).map_err(withheld)?;
        let sha256 = Digest::sha256_of(&bytes).hex().to_owned();
        verdicts.insert(
            sha256.clone(),
            (
                path.clone(),
                VerdictEntry {
                    sha256,
                    schema,
                    key_id: key.key_id(),
                },
            ),
        );
    }
    contents.experiences = loose.iter().cloned().collect();
    contents.datasets = datasets
        .values()
        .map(|d| DatasetEntry {
            id: d.id.clone(),
            view: d.manifest.view.clone(),
            records: d.manifest.counts.records,
        })
        .collect();
    let mut records = BTreeMap::new();
    for (id, candidate) in &candidates {
        let record = candidate_record(ctx, id)?;
        contents.candidates.push(CandidateEntry {
            candidate: id.clone(),
            adapter_digest: candidate.adapter_digest.clone(),
            record: Digest::of(&json_bytes("a candidate record", &record)?),
        });
        records.insert(id.clone(), record);
    }
    contents.verdicts = verdicts.values().map(|(_, v)| v.clone()).collect();
    if contents.is_empty() {
        return Err(OrchestratorError::Refused(
            "nothing to publish: name a set, a dataset, a candidate or a verdict".into(),
        ));
    }
    let content = contents.identity()?;
    let tag = tag_of(&content);

    // 2. Stage.
    let scratch = Scratch::new(ctx, "publish")?;
    let mut staged = Vec::new();
    if !set_ids.is_empty() || !loose.is_empty() {
        let root = StateRoot::new(scratch.path().join("state"));
        let staging = Workspace::at(&root);
        let loose: Vec<ExperienceId> = loose.iter().cloned().collect();
        set_import::copy(&staging, ctx.workspace(), &set_ids, &loose)?;
        let archive = scratch.path().join(EXPERIENCES);
        staging.archive(
            &archive,
            &ArchiveOptions {
                no_artifacts: true,
                since: None,
            },
        )?;
        drop(staging);
        staged.push(stage(EXPERIENCES.into(), archive, FileKind::Experiences)?);
    }
    for (id, stored) in &datasets {
        staged.push(stage(
            dataset_records_path(id),
            stored.path.clone(),
            FileKind::DatasetRecords {
                dataset: id.clone(),
            },
        )?);
        staged.push(write_staged(
            scratch.path(),
            dataset_manifest_path(id),
            &json_bytes("a dataset manifest", &stored.manifest)?,
            FileKind::DatasetManifest {
                dataset: id.clone(),
            },
        )?);
    }
    let mut artifacts = BTreeSet::new();
    for (id, candidate) in &candidates {
        staged.push(write_staged(
            scratch.path(),
            candidate_path(id),
            &json_bytes("a candidate record", &records[id])?,
            FileKind::Candidate {
                candidate: id.clone(),
            },
        )?);
        artifacts.insert(candidate.adapter_artifact.clone());
        artifacts.extend(
            candidate
                .checkpoints
                .iter()
                .map(|c| c.adapter_artifact.clone()),
        );
        artifacts.extend(candidate.rehearsal.iter().filter_map(|r| r.digest.clone()));
    }
    for digest in &artifacts {
        let artifact = ctx.artifacts().get(digest)?;
        staged.push(stage(
            artifact_path(digest, &artifact.extension),
            ctx.artifacts().path(digest)?,
            FileKind::Artifact {
                digest: digest.clone(),
                role: artifact.role.clone(),
                producer: artifact.producer.clone(),
                extension: artifact.extension.clone(),
                sha256_recorded: artifact.sha256.is_some(),
            },
        )?);
    }
    for (sha256, (path, _)) in &verdicts {
        staged.push(stage(
            verdict_path(sha256),
            path.clone(),
            FileKind::Verdict,
        )?);
    }
    if let (Some((path, _)), false) = (&key, verdicts.is_empty()) {
        staged.push(stage(
            EXAMINER_KEY.into(),
            path.clone(),
            FileKind::ExaminerKey,
        )?);
    }
    staged.sort_by(|a, b| a.file.path.cmp(&b.file.path));
    let manifest = BundleManifest {
        schema: SCHEMA.into(),
        content: content.clone(),
        files: staged.iter().map(|s| s.file.clone()).collect(),
        provenance: provenance(ctx, &contents, &datasets, &candidates)?,
        contents,
    };
    let manifest_bytes =
        serde_json::to_vec_pretty(&manifest).map_err(|source| OrchestratorError::Json {
            what: "the bundle manifest".into(),
            source,
        })?;
    let manifest_local = scratch.path().join(MANIFEST);
    std::fs::write(&manifest_local, &manifest_bytes).map_err(io(&manifest_local))?;

    // 3. A private repository, checked before anything is uploaded.
    ensure_private(hub, &repo)?;

    // 4. Published before?
    let refs = hub.refs(&repo)?;
    let files = staged.len() + 1;
    let bytes = staged.iter().map(|s| s.file.bytes).sum::<u64>() + manifest_bytes.len() as u64;
    let unchanged = |commit: CommitId| Published {
        repo: repo.to_string(),
        content: content.clone(),
        tag: tag.clone(),
        commit,
        committed: false,
        files,
        bytes,
        regular: 0,
        lfs_uploaded: 0,
        lfs_present: 0,
    };
    if refs.has_tag(&tag) {
        // The commit the tag names: the refs list an annotated tag by its own
        // object.
        return Ok(unchanged(hub.resolve(&repo, &tag)?));
    }
    let head = refs.branch(BRANCH).cloned();
    let mut previous: Vec<String> = Vec::new();
    if let Some(head) = &head {
        let tree = hub.tree(&repo, head)?;
        if tree.iter().any(|f| f.path == MANIFEST) {
            let local = scratch.path().join("head-manifest.json");
            hub.download(
                &repo,
                head,
                &RepoPath::parse(MANIFEST)?,
                &local,
                MANIFEST_LIMIT,
            )?;
            let text = std::fs::read(&local).map_err(io(&local))?;
            let same =
                serde_json::from_slice::<BundleManifest>(&text).is_ok_and(|m| m.content == content);
            if same {
                hub.create_tag(&repo, head, &tag, &format!("splinter bundle {content}"))?;
                return Ok(unchanged(head.clone()));
            }
        }
        previous = tree.into_iter().map(|f| f.path).collect();
    }

    // 5. One commit, tagged.
    let mut uploads: Vec<Upload> = staged
        .iter()
        .map(|s| {
            Ok(Upload {
                path: RepoPath::parse(&s.file.path)?,
                local: s.local.clone(),
            })
        })
        .collect::<Result<_, OrchestratorError>>()?;
    uploads.push(Upload {
        path: RepoPath::parse(MANIFEST)?,
        local: manifest_local,
    });
    let keep: BTreeSet<&str> = uploads.iter().map(|u| u.path.as_str()).collect();
    let deletes = previous
        .iter()
        .filter(|p| p.as_str() != ".gitattributes" && !keep.contains(p.as_str()))
        .map(|p| RepoPath::parse(p))
        .collect::<Result<Vec<_>, _>>()?;
    let committed = hub.commit(
        &repo,
        BRANCH,
        head.as_ref(),
        &format!("splinter bundle {content}"),
        &uploads,
        &deletes,
    )?;
    hub.create_tag(
        &repo,
        &committed.commit,
        &tag,
        &format!("splinter bundle {content}"),
    )?;
    Ok(Published {
        repo: repo.to_string(),
        content,
        tag,
        commit: committed.commit,
        committed: true,
        files,
        bytes,
        regular: committed.regular,
        lfs_uploaded: committed.lfs_uploaded,
        lfs_present: committed.lfs_present,
    })
}

/// Creates `repo` private when it is missing; refuses unless the Hub then
/// states it private.
fn ensure_private(hub: &Hub, repo: &RepoId) -> Result<(), OrchestratorError> {
    let info = match hub.repo_info(repo)? {
        Some(info) => info,
        None => {
            hub.create_private_repo(repo)?;
            hub.repo_info(repo)?.ok_or_else(|| {
                OrchestratorError::Refused(format!("{repo} was created but cannot be read back"))
            })?
        }
    };
    match info.private {
        Some(true) => Ok(()),
        Some(false) => Err(OrchestratorError::Refused(format!(
            "{repo} is public: a bundle goes only to a private repository; nothing was uploaded"
        ))),
        None => Err(OrchestratorError::Refused(format!(
            "the hub does not state {repo} private: a bundle goes only to a repository it states \
             private; nothing was uploaded"
        ))),
    }
}

/// `start` and every experience their relations name, each once.
fn related_closure(
    store: &ExperienceStore,
    start: &[ExperienceId],
) -> Result<Vec<ExperienceId>, OrchestratorError> {
    use splinter_core::annotation::AnnotationBody;
    let mut seen = BTreeSet::new();
    let mut queue: Vec<ExperienceId> = start.to_vec();
    let mut all = Vec::new();
    while let Some(next) = queue.pop() {
        if !seen.insert(next.clone()) {
            continue;
        }
        for note in store.annotations(&next)?.annotations {
            if let AnnotationBody::Relation { other, .. } = note.body {
                queue.push(other);
            }
        }
        all.push(next);
    }
    Ok(all)
}

/// The runs that produced what `contents` names, the commits that made it,
/// the tokenizers its records are of and the candidates' bases.
fn provenance(
    ctx: &Context,
    contents: &Contents,
    datasets: &BTreeMap<DatasetId, splinter_data::StoredDataset>,
    candidates: &BTreeMap<String, crate::train::Candidate>,
) -> Result<Provenance, OrchestratorError> {
    let mut names: Vec<String> = contents.sets.iter().map(|s| s.id.0.to_string()).collect();
    names.extend(contents.datasets.iter().map(|d| d.id.0.to_string()));
    names.extend(candidates.keys().cloned());
    let mut runs = Vec::new();
    let mut commits = BTreeSet::new();
    let mut add_revisions = |r: &splinter_core::build::Revisions| {
        for commit in [&r.splinter, &r.sven, &r.brain].into_iter().flatten() {
            commits.insert(commit.clone());
        }
    };
    for run in splinter_store::runs::list_runs(ctx.workspace())? {
        let outputs = run.outputs.to_string();
        if names.iter().any(|name| outputs.contains(name.as_str())) {
            add_revisions(&run.build.revisions);
            runs.push(SourceRun {
                run: run.id,
                command: run.command,
            });
        }
    }
    for candidate in candidates.values() {
        add_revisions(&candidate.binary.revisions);
    }
    add_revisions(&ctx.config().build.revisions);
    let mut tokenizers = BTreeSet::new();
    for stored in datasets.values() {
        let text = std::fs::read_to_string(&stored.path).map_err(io(&stored.path))?;
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            if let Ok(record) = serde_json::from_str::<Value>(line) {
                collect_tokenizers(&record, &mut tokenizers);
            }
        }
    }
    for candidate in candidates.values() {
        collect_tokenizers(&candidate.training_record, &mut tokenizers);
    }
    runs.sort_by(|a, b| a.run.cmp(&b.run));
    Ok(Provenance {
        source_runs: runs,
        commits: commits.into_iter().collect(),
        tokenizer_digests: tokenizers.into_iter().collect(),
        base_digests: candidates
            .values()
            .map(|c| c.base_digest.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect(),
    })
}

/// Every `tokenizer_digest` string in `value`, at any depth.
fn collect_tokenizers(value: &Value, into: &mut BTreeSet<String>) {
    match value {
        Value::Object(map) => {
            for (key, inner) in map {
                match (key.as_str(), inner) {
                    ("tokenizer_digest", Value::String(digest)) => {
                        into.insert(digest.clone());
                    }
                    _ => collect_tokenizers(inner, into),
                }
            }
        }
        Value::Array(items) => items.iter().for_each(|i| collect_tokenizers(i, into)),
        _ => {}
    }
}
