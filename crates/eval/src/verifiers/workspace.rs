// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verifiers that grade code changes by
// running a project's own tests in a clean tree, for its clients. If your
// team needs expertise in executable evaluation of code-editing agents, you
// can procure our services by sending an email to info@swedishembedded.com.

//! The workspace verifier: a candidate patch applied to the task's source
//! tree, graded by running the project's own commands.
//!
//! A [`WorkspaceCheck`] travels with a task as a privileged item of kind
//! [`WORKSPACE_CHECK`]. It names the commands that must go from failing to
//! passing (`fail_to_pass`) and those that must keep passing
//! (`pass_to_pass`), the paths the patch may not touch (`protected`), and
//! the hidden tests: an archive of files the solver never saw, laid over the
//! tree only here.
//!
//! The experience's final output is the patch, as `git diff` writes it. The
//! verdict never reads the solver's own tree or its claims:
//!
//! 1. the task's snapshot is laid out in a **fresh** directory;
//! 2. the patch is applied; one that does not apply fails;
//! 3. the paths the patch actually changed (read back from the tree, and
//!    from the patch's headers for files the tree's ignore rules would hide)
//!    must not overlap a protected or hidden path, in either direction (a
//!    file at a directory's name would redirect what is written below it);
//!    otherwise it fails;
//! 4. the hidden files are laid over the patched tree, so a patch that
//!    deleted or weakened a test cannot change what is run;
//! 5. bytecode caches are removed and `PYTHONDONTWRITEBYTECODE` is set, so a
//!    stale cache cannot hide an edit made within the same second;
//! 6. the fail-to-pass commands run, then the pass-to-pass commands, in the
//!    environment's sandbox. The patch passes only if every command exits
//!    with status 0 within its limits.
//!
//! It abstains where the task names an environment that was not offered (the
//! snapshot digest pins tree, sandbox and limits), or carries no check.
//! The evidence holds digests, exit codes and sizes - never the hidden
//! tests' text, nor their output, which would quote them.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::json;
use splinter_core::annotation::{Producer, Strength};
use splinter_core::digest::{canonical_json, Digest};
use splinter_core::experience::{Environment, Experience, Privileged, PrivilegedKind, Task};
use splinter_core::kinds::WORKSPACE_CHECK;
use splinter_sandbox::host::{DEFAULT_LANG, DEFAULT_SEARCH_PATH};
use splinter_sandbox::process::{run, ProcessSpec};
use splinter_sandbox::{
    CodeResult, CommandCall, Patch, ResolvedEnvironment, SandboxError, WorkspaceEnvironment,
};
use splinter_store::artifacts::ArtifactStore;

use super::{privileged_of, Finding, Verifier, VerifyError};

/// The producer name a workspace verifier's annotations carry.
pub const WORKSPACE_PRODUCER: &str = "splinter-lab/workspace";

/// The workspace verifier's version.
pub const VERSION: &str = "1";

/// Directories removed before every command: bytecode and test-runner
/// caches, which are keyed by modification time and size and so can hide an
/// edit made within the same second as the cache.
const CACHE_DIRS: [&str; 2] = ["__pycache__", ".pytest_cache"];

/// The variable prepended to every command, so a run leaves no bytecode.
const NO_BYTECODE: &str = "PYTHONDONTWRITEBYTECODE=1";

/// How long laying the hidden archive over the tree may take.
const OVERLAY_TIMEOUT: Duration = Duration::from_secs(120);

/// The most output kept of the archive tool.
const OVERLAY_OUTPUT_CAP: usize = 64 * 1024;

/// What a workspace task is graded by. Built by whoever makes the task
/// (task generation) and attached with [`WorkspaceCheck::as_check`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceCheck {
    /// Commands that fail on the snapshot and must succeed after the
    /// change. At least one: a check with none would pass the empty patch.
    pub fail_to_pass: Vec<CommandCall>,
    /// Commands that succeed on the snapshot and must still succeed.
    #[serde(default)]
    pub pass_to_pass: Vec<CommandCall>,
    /// The digest of the hidden overlay: a tar of files laid over the
    /// patched tree before the commands run. It is resolved through an
    /// [`ArchiveSource`] and is never part of the solver's tree.
    #[serde(default)]
    pub hidden: Option<Digest>,
    /// Paths (files or directories, relative to the tree) a patch may not
    /// touch.
    #[serde(default)]
    pub protected: Vec<String>,
    /// Paths the hidden overlay provides. A patch may not touch these
    /// either; they are listed so the refusal does not depend on the
    /// overlay's contents.
    #[serde(default)]
    pub hidden_paths: Vec<String>,
}

impl WorkspaceCheck {
    /// This check as the privileged item a task carries: kind
    /// [`WORKSPACE_CHECK`], content the canonical JSON of the check.
    pub fn as_check(&self) -> Result<Privileged, serde_json::Error> {
        let content = canonical_json(self)?;
        Ok(Privileged {
            kind: PrivilegedKind::Other(WORKSPACE_CHECK.to_string()),
            content: String::from_utf8_lossy(&content).into_owned(),
            span: None,
        })
    }
}

/// Where the verifier finds the archives a task names by digest: the
/// snapshot's tree and the hidden overlay.
pub trait ArchiveSource: Send + Sync {
    /// The file holding the archive whose content digest is `digest`.
    fn path(&self, digest: &Digest) -> Result<PathBuf, VerifyError>;
}

impl ArchiveSource for ArtifactStore {
    fn path(&self, digest: &Digest) -> Result<PathBuf, VerifyError> {
        Ok(ArtifactStore::path(self, digest)?)
    }
}

/// Grades a [`splinter_core::kinds::WORKSPACE_CHANGE`] task's patch by
/// running its [`WorkspaceCheck`] in a fresh tree (see the module
/// documentation).
#[derive(Clone)]
pub struct WorkspaceVerifier {
    offered: Vec<WorkspaceEnvironment>,
    archives: Arc<dyn ArchiveSource>,
    scratch_root: PathBuf,
}

impl std::fmt::Debug for WorkspaceVerifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkspaceVerifier")
            .field("offered", &self.offered.len())
            .field("scratch_root", &self.scratch_root)
            .finish_non_exhaustive()
    }
}

impl WorkspaceVerifier {
    /// A verifier that runs checks in `environments` and nowhere else,
    /// resolving archives with `archives`, laying each verification's tree
    /// out in a fresh directory under `scratch_root` and removing it
    /// afterwards.
    #[must_use]
    pub fn new(
        environments: Vec<WorkspaceEnvironment>,
        archives: Arc<dyn ArchiveSource>,
        scratch_root: impl Into<PathBuf>,
    ) -> Self {
        Self {
            offered: environments,
            archives,
            scratch_root: scratch_root.into(),
        }
    }

    /// The offered environment whose record is exactly `wanted`.
    fn find(&self, wanted: &Environment) -> Result<Option<&WorkspaceEnvironment>, SandboxError> {
        for env in &self.offered {
            if ResolvedEnvironment::Workspace(env.clone()).record()? == *wanted {
                return Ok(Some(env));
            }
        }
        Ok(None)
    }
}

impl Verifier for WorkspaceVerifier {
    fn producer(&self) -> Producer {
        Producer {
            name: WORKSPACE_PRODUCER.into(),
            version: VERSION.into(),
        }
    }

    fn strength(&self) -> Strength {
        Strength::Executable
    }

    fn verify(&self, task: &Task, exp: &Experience) -> Result<Finding, VerifyError> {
        let Some((check_digest, check)) = parse_check(task)? else {
            return Ok(Finding::abstain(
                "the task carries no workspace check",
                json!({}),
            ));
        };
        if task.environment.kind != Environment::WORKSPACE {
            return Ok(Finding::abstain(
                "the task is not in a workspace environment",
                json!({ "kind": task.environment.kind }),
            ));
        }
        let Some(env) = self.find(&task.environment)? else {
            return Ok(Finding::abstain(
                "the environment the task names was not offered",
                json!({ "environment": task.environment.snapshot }),
            ));
        };
        let Some(output) = exp.final_output.as_deref() else {
            return Ok(Finding::decided(false, json!({ "refused": "no_patch" })));
        };
        let patch_digest = Digest::of(output.as_bytes());
        if output.trim().is_empty() {
            return Ok(refused("empty_patch", &patch_digest));
        }
        let scratch = Scratch::create(&self.scratch_root)?;
        let tree = scratch.path().join("tree");
        let baseline = env.materialize(&self.archives.path(&env.snapshot().digest)?, &tree)?;
        let patch = Patch {
            diff: patch_bytes(output),
            files: Vec::new(),
        };
        match env.apply(&tree, &patch) {
            Ok(_) => {}
            Err(SandboxError::Workspace { .. }) => {
                return Ok(refused("patch_does_not_apply", &patch_digest))
            }
            Err(other) => return Err(other.into()),
        }
        let mut touched: Vec<String> = env
            .diff_since(&tree, &baseline, &[])?
            .files
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();
        touched.extend(header_paths(output));
        let off_limits = check.off_limits();
        let violations = touched
            .iter()
            .filter(|path| off_limits.iter().any(|fenced| overlaps(path, fenced)))
            .count();
        if violations > 0 {
            let mut finding = refused("touches_protected_or_hidden_path", &patch_digest);
            finding.evidence["paths"] = json!(violations);
            return Ok(finding);
        }
        if let Some(hidden) = &check.hidden {
            let archive = self.archives.path(hidden)?;
            lay_over(&archive, hidden, &tree)?;
        }
        let mut commands = Vec::new();
        let mut passed = true;
        for (phase, calls) in [
            ("fail_to_pass", &check.fail_to_pass),
            ("pass_to_pass", &check.pass_to_pass),
        ] {
            // A failed target already decides the verdict: the guarded
            // commands would only cost time.
            if !passed {
                break;
            }
            for (index, call) in calls.iter().enumerate() {
                clear_caches(&tree);
                let result = env.run(&tree, &without_bytecode(call))?;
                let ok = succeeded(&result);
                passed &= ok;
                commands.push(command_evidence(phase, index, call, &result, ok)?);
            }
        }
        Ok(Finding::decided(
            passed,
            json!({
                "check": check_digest,
                "environment": task.environment.snapshot,
                "patch": patch_digest,
                "hidden": check.hidden,
                "commands": commands,
            }),
        ))
    }
}

impl WorkspaceCheck {
    /// The paths a patch may not overlap: protected and hidden, normalised.
    fn off_limits(&self) -> Vec<String> {
        self.protected
            .iter()
            .chain(&self.hidden_paths)
            .filter_map(|p| normalised(p))
            .collect()
    }
}

/// The task's one workspace check with the digest of the item it was parsed
/// from; `None` when it carries none.
fn parse_check(task: &Task) -> Result<Option<(Digest, WorkspaceCheck)>, VerifyError> {
    let kind = PrivilegedKind::Other(WORKSPACE_CHECK.to_string());
    let items: Vec<(usize, &str)> = privileged_of(task, &kind).collect();
    let malformed = |index: usize, reason: String| VerifyError::Malformed {
        kind: kind.clone(),
        index,
        reason,
    };
    let Some(&(index, content)) = items.first() else {
        return Ok(None);
    };
    if items.len() > 1 {
        return Err(malformed(
            items[1].0,
            "a task carries one workspace check".into(),
        ));
    }
    let check: WorkspaceCheck =
        serde_json::from_str(content).map_err(|e| malformed(index, e.to_string()))?;
    if check.fail_to_pass.is_empty() {
        return Err(malformed(index, "it has no fail-to-pass command".into()));
    }
    if check
        .fail_to_pass
        .iter()
        .chain(&check.pass_to_pass)
        .any(|call| call.argv.is_empty())
    {
        return Err(malformed(index, "a command has an empty argv".into()));
    }
    if let Some(bad) = check
        .protected
        .iter()
        .chain(&check.hidden_paths)
        .find(|p| normalised(p).is_none())
    {
        return Err(malformed(
            index,
            format!("{bad:?} is not a path inside the tree"),
        ));
    }
    Ok(Some((Digest::of(content.as_bytes()), check)))
}

/// A refusal: the patch fails without a command having run.
fn refused(reason: &str, patch: &Digest) -> Finding {
    Finding::decided(false, json!({ "refused": reason, "patch": patch }))
}

/// The patch as `git apply` reads it: it must end in a newline.
fn patch_bytes(text: &str) -> Vec<u8> {
    let mut bytes = text.as_bytes().to_vec();
    if !bytes.ends_with(b"\n") {
        bytes.push(b'\n');
    }
    bytes
}

/// `path` as a relative path without `.` or empty components, or `None`
/// when it is empty, absolute or climbs out of the tree.
fn normalised(path: &str) -> Option<String> {
    let mut parts = Vec::new();
    for part in path.split('/') {
        match part {
            "" | "." => {}
            ".." => return None,
            other => parts.push(other),
        }
    }
    (!parts.is_empty() && !path.starts_with('/')).then(|| parts.join("/"))
}

/// Whether `path` and `fenced` are the same path or one is below the other.
fn overlaps(path: &str, fenced: &str) -> bool {
    let Some(path) = normalised(path) else {
        // A path that cannot be placed in the tree is never allowed.
        return true;
    };
    let below = |inner: &str, outer: &str| {
        inner == outer
            || inner
                .strip_prefix(outer)
                .is_some_and(|rest| rest.starts_with('/'))
    };
    below(&path, fenced) || below(fenced, &path)
}

/// The paths a patch's headers name (plain, unquoted ones): a second
/// reading of what the patch touches, for files the tree's ignore rules
/// would keep out of a diff of the patched tree.
fn header_paths(patch: &str) -> Vec<String> {
    let mut paths = Vec::new();
    for line in patch.lines() {
        if let Some(pair) = line.strip_prefix("diff --git ") {
            paths.extend(same_path_pair(pair));
        }
        for prefix in ["rename from ", "rename to ", "copy from ", "copy to "] {
            if let Some(path) = line.strip_prefix(prefix) {
                paths.push(path.to_string());
            }
        }
    }
    paths
}

/// `P` from the `a/P b/P` of a header that names one path twice.
fn same_path_pair(pair: &str) -> Option<String> {
    let middle = pair.len().checked_sub(1)? / 2;
    let left = pair.get(..middle)?.strip_prefix("a/")?;
    let right = pair.get(middle + 1..)?.strip_prefix("b/")?;
    (pair.as_bytes().get(middle) == Some(&b' ') && left == right).then(|| left.to_string())
}

/// Whether a command ended by exiting with status 0 within its limits.
fn succeeded(result: &CodeResult) -> bool {
    result.exit_code == Some(0) && result.signal.is_none() && !result.timed_out
}

/// `call` run with bytecode writing off.
fn without_bytecode(call: &CommandCall) -> CommandCall {
    let mut argv = vec!["env".to_string(), NO_BYTECODE.to_string()];
    argv.extend(call.argv.iter().cloned());
    CommandCall {
        argv,
        stdin: call.stdin.clone(),
    }
}

/// What one command's evidence records: digests of what ran and what came
/// back, never the output (which quotes the tests).
fn command_evidence(
    phase: &str,
    index: usize,
    call: &CommandCall,
    result: &CodeResult,
    passed: bool,
) -> Result<serde_json::Value, VerifyError> {
    let command = Digest::of(&canonical_json(call).map_err(SandboxError::from)?);
    Ok(json!({
        "phase": phase,
        "index": index,
        "command": command,
        "exit_code": result.exit_code,
        "signal": result.signal,
        "timed_out": result.timed_out,
        "stdout": Digest::of(result.stdout.as_bytes()),
        "stdout_truncated": result.stdout_truncated,
        "stderr": Digest::of(result.stderr.as_bytes()),
        "stderr_truncated": result.stderr_truncated,
        "passed": passed,
    }))
}

/// Removes the cache directories below `dir`, not following links. Best
/// effort: a cache that cannot be removed only risks what it always risked.
fn clear_caches(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        if !kind.is_dir() {
            continue;
        }
        let path = entry.path();
        if CACHE_DIRS.iter().any(|name| entry.file_name() == *name) {
            let _ = std::fs::remove_dir_all(&path);
        } else {
            clear_caches(&path);
        }
    }
}

/// Lays the tar at `archive` (whose content must be `digest`) over `tree`,
/// replacing the files it holds.
fn lay_over(archive: &Path, digest: &Digest, tree: &Path) -> Result<(), VerifyError> {
    let failed = |detail: String| VerifyError::Failed {
        producer: WORKSPACE_PRODUCER.into(),
        source: format!("laying {} over the tree: {detail}", archive.display()).into(),
    };
    let actual = std::fs::File::open(archive)
        .and_then(Digest::of_reader)
        .map_err(|e| failed(e.to_string()))?;
    if actual != *digest {
        return Err(failed(format!("the archive has digest {actual}")));
    }
    let env: BTreeMap<String, String> = [("PATH", DEFAULT_SEARCH_PATH), ("LANG", DEFAULT_LANG)]
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .into();
    let argv = [
        "tar",
        "--extract",
        "--no-same-owner",
        "--no-same-permissions",
        "--file",
        &archive.to_string_lossy(),
        "--directory",
        &tree.to_string_lossy(),
    ]
    .map(String::from)
    .to_vec();
    let spec = ProcessSpec::new(argv, tree, env, OVERLAY_TIMEOUT, OVERLAY_OUTPUT_CAP);
    let out = run(&spec).map_err(SandboxError::from)?;
    if out.exit_code != Some(0) || out.timed_out {
        let stderr = String::from_utf8_lossy(&out.stderr.bytes);
        return Err(failed(format!(
            "tar ended with {:?}: {}",
            out.exit_code,
            stderr.chars().take(200).collect::<String>()
        )));
    }
    Ok(())
}

/// A directory made fresh for one verification, removed when dropped.
struct Scratch {
    path: PathBuf,
}

impl Scratch {
    fn create(root: &Path) -> Result<Self, SandboxError> {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        std::fs::create_dir_all(root).map_err(|source| SandboxError::Io {
            path: root.to_path_buf(),
            source,
        })?;
        loop {
            let path = root.join(format!(
                "splinter-verify-{}-{}",
                std::process::id(),
                SEQ.fetch_add(1, Ordering::Relaxed)
            ));
            match std::fs::create_dir(&path) {
                Ok(()) => return Ok(Self { path }),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(source) => return Err(SandboxError::Io { path, source }),
            }
        }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        if std::fs::remove_dir_all(&self.path).is_err() {
            // A command may have taken away its own permissions.
            restore_access(&self.path);
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}

/// Gives the owner full access to `path` and every directory below it.
fn restore_access(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let Ok(meta) = std::fs::symlink_metadata(path) else {
        return;
    };
    if !meta.is_dir() {
        return;
    }
    let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700));
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            restore_access(&entry.path());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_path_overlaps_what_it_contains_and_what_contains_it() {
        assert!(overlaps("tests/test_a.py", "tests"));
        assert!(overlaps("tests", "tests/test_a.py"));
        assert!(overlaps("./tests//test_a.py", "tests/test_a.py"));
        assert!(!overlaps("tests_extra/x.py", "tests"));
        assert!(!overlaps("src/tests.py", "tests"));
        assert!(overlaps("../escape", "tests"));
    }

    #[test]
    fn headers_name_the_paths_a_patch_touches() {
        let patch = "diff --git a/dir/a b.py b/dir/a b.py\nindex 1..2\n\
                     diff --git a/old.py b/new.py\nrename from old.py\nrename to new.py\n";
        assert_eq!(
            header_paths(patch),
            ["dir/a b.py", "old.py", "new.py"].map(String::from)
        );
    }
}
