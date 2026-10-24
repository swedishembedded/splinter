// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements durable lineage from distributed training
// runs to the files they produced, for its clients. If your team needs
// expertise in verified checkpoint ingestion or crash-safe artifact
// registration, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The outputs a completed job leaves behind, kept as artifacts and linked to
//! the job that wrote them.
//!
//! A training job signals a commit by writing `commit-<step>.json` in its
//! launch directory. Registration reads the newest such manifest, checks every
//! file it names against the declared length and SHA-256, keeps the verified
//! bytes in the artifact store, and records one document joining the job
//! specification, the attempt, the step and the plan to each artifact's role,
//! content address and trainer checksum.
//!
//! Registration is idempotent: an artifact is addressed by its bytes and the
//! lineage document by its canonical form, so reconciling the same commit again
//! adds neither a second artifact nor a second document. A malformed manifest,
//! a missing or altered file, or a path that leaves the commit directory is
//! refused; the caller reports the job as failed rather than complete.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::artifacts::{ArtifactSpec, ArtifactStore};
use crate::error::StoreError;
use crate::workspace::Workspace;
use splinter_core::checkpoint::{CheckpointManifest, CheckpointManifestError};
use splinter_core::digest::Digest;

const JOB_OUTPUT: &str = "job_output";

/// The producer recorded for a file a job wrote.
const PRODUCER: &str = "job";

/// One artifact a completed job produced.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JobOutputFile {
    /// What the file is for, such as `state` or `weights`.
    pub role: String,
    /// The content address of the bytes Splinter kept.
    pub digest: Digest,
    /// The SHA-256 the job declared for the same bytes.
    pub sha256: Digest,
    /// How many bytes the file has.
    pub len: u64,
}

/// What one commit of a job produced, and what it was produced under.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JobOutput {
    /// The specification of the job that wrote it, by content address.
    pub job: Digest,
    /// The attempt, counting from one.
    pub attempt: u32,
    /// The training step the commit belongs to.
    pub step: u32,
    /// The plan the run was launched under, as the exact JSON text it was
    /// submitted with.
    pub plan: String,
    /// How many ranks wrote the commit.
    pub world_size: usize,
    /// Every artifact the commit holds.
    pub files: Vec<JobOutputFile>,
}

/// Why a job's outputs could not be registered.
#[derive(Debug, thiserror::Error)]
pub enum JobOutputError {
    /// The manifest is not part of the contract.
    #[error(transparent)]
    Manifest(#[from] CheckpointManifestError),
    /// A file could not be read.
    #[error("{path}: {source}")]
    Io {
        /// The file or directory.
        path: PathBuf,
        /// The operating-system failure.
        source: io::Error,
    },
    /// A file is not the length the manifest states.
    #[error("checkpoint file {path} is {found} bytes, the manifest states {expected}")]
    Length {
        /// The file.
        path: PathBuf,
        /// The length the manifest states.
        expected: u64,
        /// The length found.
        found: u64,
    },
    /// A file is not the bytes the manifest states.
    #[error("checkpoint file {path} hashes to {found}, the manifest states {expected}")]
    Checksum {
        /// The file.
        path: PathBuf,
        /// The SHA-256 the manifest states.
        expected: Digest,
        /// The SHA-256 found.
        found: Digest,
    },
    /// A file resolves outside the attempt directory, so a symlink leaves it.
    #[error("checkpoint file {path} leads out of the job's attempt directory")]
    Escape {
        /// The file.
        path: PathBuf,
    },
    /// A file changed between being verified and being kept.
    #[error(
        "checkpoint file {path} changed while it was kept: stored {found_len} bytes \
         {found_sha256:?}, the manifest states {expected_len} bytes {expected_sha256}"
    )]
    Changed {
        /// The file.
        path: PathBuf,
        /// The length the manifest states.
        expected_len: u64,
        /// The SHA-256 the manifest states.
        expected_sha256: Digest,
        /// The length kept.
        found_len: u64,
        /// The SHA-256 kept.
        found_sha256: Option<Digest>,
    },
    /// The store failed.
    #[error(transparent)]
    Store(#[from] StoreError),
}

fn io_error(path: &Path) -> impl FnOnce(io::Error) -> JobOutputError + '_ {
    move |source| JobOutputError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// The newest commit manifest in `dir`, with its step.
///
/// Only `commit-<8 digit step>.json` is a manifest; a temporary file a job is
/// still writing is ignored, so a commit is registered once it is complete.
fn newest_manifest(dir: &Path) -> Result<Option<(u32, PathBuf)>, JobOutputError> {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(source) if source.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(source) => return Err(io_error(dir)(source)),
    };
    let mut newest: Option<(u32, PathBuf)> = None;
    for entry in entries {
        let entry = entry.map_err(io_error(dir))?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(digits) = name
            .strip_prefix("commit-")
            .and_then(|rest| rest.strip_suffix(".json"))
        else {
            continue;
        };
        if digits.len() != 8 || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
            continue;
        }
        let Ok(step) = digits.parse::<u32>() else {
            continue;
        };
        if newest
            .as_ref()
            .is_none_or(|(newest_step, _)| step > *newest_step)
        {
            newest = Some((step, entry.path()));
        }
    }
    Ok(newest)
}

/// Registers the outputs of attempt `attempt` of `job`, found in the launch
/// directory `dir`.
///
/// Returns `None` when the job wrote no commit manifest, which is not a
/// failure: outputs are optional. Every file the newest manifest names must be
/// present with the declared length and SHA-256; the verified bytes are kept in
/// `artifacts` and one lineage document is recorded. Repeating the call adds no
/// duplicate artifact or document.
pub fn register_job_outputs(
    workspace: &Workspace,
    artifacts: &ArtifactStore,
    dir: &Path,
    job: &Digest,
    attempt: u32,
) -> Result<Option<JobOutput>, JobOutputError> {
    let Some((_, manifest_path)) = newest_manifest(dir)? else {
        return Ok(None);
    };
    let bytes = fs::read(&manifest_path).map_err(io_error(&manifest_path))?;
    let manifest = CheckpointManifest::parse(&bytes)?;
    let base = fs::canonicalize(dir).map_err(io_error(dir))?;
    let mut files = Vec::with_capacity(manifest.files.len());
    for file in &manifest.files {
        let source = dir.join(file.path.as_str());
        // A relative path that carries no `..` still leaves the directory when
        // it is a symbolic link out of it.
        let resolved = fs::canonicalize(&source).map_err(io_error(&source))?;
        if !resolved.starts_with(&base) {
            return Err(JobOutputError::Escape { path: source });
        }
        let found_len = fs::metadata(&resolved).map_err(io_error(&source))?.len();
        if found_len != file.len {
            return Err(JobOutputError::Length {
                path: source,
                expected: file.len,
                found: found_len,
            });
        }
        let expected = file.sha256.digest().clone();
        let found = Digest::sha256_of_reader(fs::File::open(&resolved).map_err(io_error(&source))?)
            .map_err(io_error(&source))?;
        if found != expected {
            return Err(JobOutputError::Checksum {
                path: source,
                expected,
                found,
            });
        }
        let artifact = artifacts.put_file(
            &resolved,
            &ArtifactSpec::new(&file.role, PRODUCER).with_sha256(),
        )?;
        // The copy is what the store keeps; checking it against the manifest
        // catches a file that changed between being verified and being kept.
        if artifact.size != file.len || artifact.sha256.as_ref() != Some(&expected) {
            return Err(JobOutputError::Changed {
                path: source,
                expected_len: file.len,
                expected_sha256: expected,
                found_len: artifact.size,
                found_sha256: artifact.sha256,
            });
        }
        files.push(JobOutputFile {
            role: file.role.clone(),
            digest: artifact.digest,
            sha256: expected,
            len: file.len,
        });
    }
    let output = JobOutput {
        job: job.clone(),
        attempt,
        step: manifest.step,
        plan: manifest.plan,
        world_size: manifest.world_size,
        files,
    };
    workspace.put_job_output(&output)?;
    Ok(Some(output))
}

impl Workspace {
    /// Keeps `output` as a lineage document. Recording the same commit again
    /// writes nothing and returns the same address.
    pub fn put_job_output(&self, output: &JobOutput) -> Result<Digest, StoreError> {
        self.put_document(JOB_OUTPUT, output)
    }

    /// Every job output document, oldest first.
    pub fn job_outputs(&self) -> Result<Vec<JobOutput>, StoreError> {
        let mut outputs = Vec::new();
        for id in self.documents_in_order(JOB_OUTPUT)? {
            if let Some(output) = self.get_document(JOB_OUTPUT, &id)? {
                outputs.push(output);
            }
        }
        Ok(outputs)
    }

    /// The output registered for attempt `attempt` of `job`, when there is one.
    pub fn job_output(&self, job: &Digest, attempt: u32) -> Result<Option<JobOutput>, StoreError> {
        Ok(self
            .job_outputs()?
            .into_iter()
            .find(|output| &output.job == job && output.attempt == attempt))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StateRoot;
    use splinter_core::checkpoint::JobPlan;
    use splinter_core::job::JobSpec;

    const PLAN: &str = r#"{"job":"qwen-coding-v1","world_size":2,"tp":1,"pp":1,"dp":2,"coordinator":"127.0.0.1:7000","handshake_timeout_ms":120000,"op_timeout_ms":600000}"#;

    struct Fixture {
        _state: tempfile::TempDir,
        _launch: tempfile::TempDir,
        workspace: Workspace,
        artifacts: ArtifactStore,
        job: Digest,
    }

    impl Fixture {
        fn new() -> Fixture {
            let state = tempfile::tempdir().unwrap();
            let launch = tempfile::tempdir().unwrap();
            let root = StateRoot::new(state.path());
            let workspace = Workspace::at(&root);
            let artifacts = ArtifactStore::new(&workspace, &root);
            Fixture {
                _state: state,
                _launch: launch,
                workspace,
                artifacts,
                job: JobSpec::single(vec!["brain".into(), "train".into()]).digest(),
            }
        }

        /// Writes `name` with `bytes` and returns the manifest entry for it.
        fn file(&self, role: &str, name: &str, bytes: &[u8]) -> String {
            let path = self._launch.path().join(name);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).unwrap();
            }
            fs::write(&path, bytes).unwrap();
            let sha = Digest::sha256_of(bytes);
            format!(
                r#"{{"role":"{role}","path":"{name}","len":{},"sha256":"{sha}"}}"#,
                bytes.len()
            )
        }

        fn write_manifest(&self, step: u32, files: &[String]) {
            let text = format!(
                r#"{{"job":"qwen-coding-v1","plan":{PLAN:?},"step":{step},"world_size":2,"files":[{}]}}"#,
                files.join(",")
            );
            fs::write(
                self._launch.path().join(format!("commit-{step:08}.json")),
                text,
            )
            .unwrap();
        }

        fn register(&self) -> Result<Option<JobOutput>, JobOutputError> {
            register_job_outputs(
                &self.workspace,
                &self.artifacts,
                self._launch.path(),
                &self.job,
                1,
            )
        }
    }

    #[test]
    fn a_valid_manifest_registers_its_files_once() {
        let fixture = Fixture::new();
        let weights = fixture.file("weights", "weights.safetensors", b"weights");
        let state = fixture.file("state", "state/train.state", b"train state");
        fixture.write_manifest(500, &[state, weights]);
        // A temporary manifest is not a commit.
        fs::write(
            fixture._launch.path().join("commit-00000006.tmp"),
            b"not yet",
        )
        .unwrap();

        let output = fixture
            .register()
            .unwrap()
            .expect("the commit is registered");
        assert_eq!(output.job, fixture.job);
        assert_eq!(output.attempt, 1);
        assert_eq!(output.step, 500);
        assert_eq!(output.world_size, 2);
        assert_eq!(
            JobPlan::parse(&output.plan).unwrap().coordinator,
            "127.0.0.1:7000"
        );
        assert_eq!(output.files.len(), 2);
        assert_eq!(
            output.files[0].sha256.as_str(),
            Digest::sha256_of(b"train state").as_str()
        );
        let kept = fixture.artifacts.get(&output.files[0].digest).unwrap();
        assert_eq!(kept.role, "state");
        assert_eq!(kept.sha256.as_ref(), Some(&output.files[0].sha256));

        // Reconciling the same commit again adds no artifact and no document.
        fixture.register().unwrap();
        assert_eq!(fixture.workspace.job_outputs().unwrap().len(), 1);
        assert_eq!(fixture.artifacts.list().unwrap().len(), 2);
        assert_eq!(
            fixture
                .workspace
                .job_output(&fixture.job, 1)
                .unwrap()
                .expect("the lineage document"),
            output
        );
    }

    #[test]
    fn the_newest_commit_is_the_one_registered() {
        let fixture = Fixture::new();
        let early = fixture.file("weights", "early.bin", b"early");
        fixture.write_manifest(100, &[early]);
        let late = fixture.file("weights", "late.bin", b"late");
        fixture.write_manifest(500, &[late]);

        let output = fixture.register().unwrap().unwrap();
        assert_eq!(output.step, 500);
        assert_eq!(output.files[0].len, 4);
    }

    #[test]
    fn a_tampered_file_is_refused() {
        let fixture = Fixture::new();
        let weights = fixture.file("weights", "weights.bin", b"weights");
        fixture.write_manifest(500, &[weights]);
        fs::write(fixture._launch.path().join("weights.bin"), b"Weights").unwrap();
        let error = fixture.register().unwrap_err();
        assert!(
            error.to_string().contains("hashes to"),
            "the error names the mismatch: {error}"
        );
        // Nothing was kept and no lineage was written.
        assert!(fixture.artifacts.list().unwrap().is_empty());
        assert!(fixture.workspace.job_outputs().unwrap().is_empty());
    }

    #[test]
    fn a_short_file_is_refused() {
        let fixture = Fixture::new();
        let weights = fixture.file("weights", "weights.bin", b"weights");
        fixture.write_manifest(500, &[weights]);
        fs::write(fixture._launch.path().join("weights.bin"), b"wei").unwrap();
        let error = fixture.register().unwrap_err();
        assert!(error.to_string().contains("bytes"), "{error}");
    }

    #[test]
    fn a_path_out_of_the_commit_directory_is_refused() {
        let fixture = Fixture::new();
        let text = format!(
            r#"{{"job":"run","plan":{PLAN:?},"step":1,"world_size":2,"files":[{{"role":"weights","path":"../weights.bin","len":1,"sha256":"sha256:{}"}}]}}"#,
            "11".repeat(32)
        );
        fs::write(fixture._launch.path().join("commit-00000001.json"), text).unwrap();
        let error = fixture.register().unwrap_err();
        assert!(matches!(error, JobOutputError::Manifest(_)), "{error}");
    }

    #[test]
    fn a_symlink_out_of_the_attempt_directory_is_refused() {
        let fixture = Fixture::new();
        let outside = tempfile::tempdir().unwrap();
        let target = outside.path().join("secret");
        fs::write(&target, b"secret").unwrap();
        std::os::unix::fs::symlink(&target, fixture._launch.path().join("link")).unwrap();
        let sha = Digest::sha256_of(b"secret");
        let text = format!(
            r#"{{"job":"run","plan":{PLAN:?},"step":1,"world_size":2,"files":[{{"role":"weights","path":"link","len":6,"sha256":"{sha}"}}]}}"#
        );
        fs::write(fixture._launch.path().join("commit-00000001.json"), text).unwrap();
        let error = fixture.register().unwrap_err();
        assert!(matches!(error, JobOutputError::Escape { .. }), "{error}");
    }

    #[test]
    fn no_manifest_is_no_output() {
        let fixture = Fixture::new();
        assert!(fixture.register().unwrap().is_none());
    }
}
