// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible, auditable workspaces for
// agents that change code, for its clients. If your team needs expertise in
// sandboxed code-editing agents or reproducible agent runs, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The supervisor's own tools for a workspace: `tar` to lay an archive out
//! and `git` to read what changed in the tree and to apply it elsewhere.
//!
//! They run on the host through the bounded process runner with an
//! explicit environment and no configuration of the user's: no global or
//! system git configuration, no hooks, no pager, no external diff, no
//! attribute or filter that could rewrite a file. The repository that
//! records the baseline lives **beside** the tree, never inside it, so
//! whatever runs in the tree (which may be a container mounting nothing
//! else) cannot reach the configuration the supervisor trusts.

use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use splinter_core::digest::Digest;

use crate::backend::{io, SandboxError};
use crate::process::{run, ProcessOutput, ProcessSpec};
use crate::runtime::{resolve_on_host, ResolvedRuntime, RuntimeSpec};
use crate::workspace::{Baseline, Patch, TreeSnapshot};

/// The author and committer of the baseline commit.
const IDENTITY_NAME: &str = "splinter";
const IDENTITY_EMAIL: &str = "splinter@localhost";

/// The date of the baseline commit, fixed so that the same tree always has
/// the same baseline commit.
const BASELINE_DATE: &str = "2000-01-01T00:00:00+0000";

/// How long one tool run (extracting an archive, staging a tree) may take.
const TOOL_TIMEOUT: Duration = Duration::from_secs(300);

/// The most bytes kept of a tool's standard output. A patch or a listing
/// cut at the cap would be silently incomplete, so a stream that reaches it
/// is an error.
const MAX_TOOL_OUTPUT: usize = 64 * 1024 * 1024;

/// How much of a tool's standard error an error message carries.
const DETAIL_CHARS: usize = 400;

/// Configuration given to every git run, on the command line so that
/// neither the user's nor the repository's own configuration can change
/// what a command does.
const GIT_CONFIG: [&str; 14] = [
    "core.autocrlf=false",
    "core.safecrlf=false",
    "core.fileMode=true",
    "core.symlinks=true",
    "core.quotePath=false",
    "core.fsmonitor=false",
    "core.hooksPath=/dev/null",
    "core.excludesFile=/dev/null",
    "core.attributesFile=/dev/null",
    "diff.noprefix=false",
    "diff.mnemonicPrefix=false",
    "commit.gpgSign=false",
    "gc.auto=0",
    "advice.addIgnoredFile=false",
];

/// The attributes of a baseline repository. They take precedence over any
/// `.gitattributes` in the tree, so none can convert line ends or run a
/// filter: the baseline holds exactly the archived bytes.
const NEUTRAL_ATTRIBUTES: &str = "* -text -eol -ident -filter\n";

/// `tar` and `git` resolved on the host, and where the host's scratch
/// directories go. Cheap to share: environments hold it in an `Arc`.
#[derive(Clone, Debug)]
pub struct WorkspaceHost {
    git: ResolvedRuntime,
    tar: ResolvedRuntime,
    env: BTreeMap<String, String>,
    scratch_root: PathBuf,
}

/// A git repository beside a tree: where it is and which tree it tracks.
struct Repo<'a> {
    git_dir: &'a Path,
    work_tree: &'a Path,
}

impl WorkspaceHost {
    /// `git` and `tar` found on `search_path` (a `PATH` value: the caller
    /// decides, nothing is read from the process environment), their
    /// versions probed. Scratch directories used to compute a snapshot are
    /// made under `scratch_root`.
    pub fn resolve(
        search_path: &str,
        scratch_root: impl Into<PathBuf>,
    ) -> Result<Self, SandboxError> {
        let env: BTreeMap<String, String> = [
            ("PATH", search_path),
            ("LANG", "C.UTF-8"),
            ("LC_ALL", "C.UTF-8"),
            ("HOME", "/nonexistent"),
            ("GIT_CONFIG_NOSYSTEM", "1"),
            ("GIT_CONFIG_GLOBAL", "/dev/null"),
            ("GIT_CONFIG_SYSTEM", "/dev/null"),
            ("GIT_ATTR_NOSYSTEM", "1"),
            ("GIT_TERMINAL_PROMPT", "0"),
            ("GIT_PAGER", "cat"),
            ("GIT_AUTHOR_NAME", IDENTITY_NAME),
            ("GIT_AUTHOR_EMAIL", IDENTITY_EMAIL),
            ("GIT_AUTHOR_DATE", BASELINE_DATE),
            ("GIT_COMMITTER_NAME", IDENTITY_NAME),
            ("GIT_COMMITTER_EMAIL", IDENTITY_EMAIL),
            ("GIT_COMMITTER_DATE", BASELINE_DATE),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
        let git = resolve_on_host(&RuntimeSpec::host_tool("git"), search_path, &env)?;
        let tar = resolve_on_host(&RuntimeSpec::host_tool("tar"), search_path, &env)?;
        Ok(Self {
            git,
            tar,
            env,
            scratch_root: scratch_root.into(),
        })
    }

    /// The version `git --version` reported.
    #[must_use]
    pub fn git_version(&self) -> &str {
        self.git.version.as_deref().unwrap_or_default()
    }

    /// The snapshot of `archive`: the content digest of the file, and the
    /// id of the git tree its extraction makes.
    pub fn snapshot(&self, archive: &Path) -> Result<TreeSnapshot, SandboxError> {
        let file = std::fs::File::open(archive).map_err(io(archive))?;
        let digest = Digest::of_reader(file).map_err(io(archive))?;
        let scratch = tempfile::Builder::new()
            .prefix("splinter-snapshot-")
            .tempdir_in(&self.scratch_root)
            .map_err(io(&self.scratch_root))?;
        let baseline = self.import(archive, &scratch.path().join("tree"))?;
        Ok(TreeSnapshot {
            digest,
            tree_id: baseline.tree_id,
        })
    }

    /// Extracts `archive` into the new directory `into` and records it as
    /// the baseline commit of a repository beside it.
    pub(crate) fn import(&self, archive: &Path, into: &Path) -> Result<Baseline, SandboxError> {
        let git_dir = git_dir_beside(into)?;
        for path in [into, &git_dir] {
            if path.exists() && !is_empty_dir(path) {
                return Err(failure(
                    format!("laying out the tree at {}", into.display()),
                    format!("{} already exists and is not empty", path.display()),
                ));
            }
        }
        std::fs::create_dir_all(into).map_err(io(into))?;
        let what = format!("extracting {}", archive.display());
        let parent = into.parent().unwrap_or(into);
        let untar = [
            self.tar.executable.as_str(),
            "--extract",
            "--no-same-owner",
            "--no-same-permissions",
            "--file",
            &path_arg(archive)?,
            "--directory",
            &path_arg(into)?,
        ];
        self.checked_run(&what, untar.map(String::from).to_vec(), parent)?;
        if into.join(".git").exists() {
            return Err(failure(
                what,
                "the archive contains a .git entry, which would shadow the baseline".into(),
            ));
        }
        let repo = Repo {
            git_dir: &git_dir,
            work_tree: into,
        };
        self.git_alone(
            "creating the baseline repository",
            parent,
            &[
                "init",
                "--bare",
                "-q",
                "--initial-branch=main",
                &path_arg(&git_dir)?,
            ],
        )?;
        let attributes = git_dir.join("info").join("attributes");
        std::fs::create_dir_all(git_dir.join("info")).map_err(io(&git_dir))?;
        std::fs::write(&attributes, NEUTRAL_ATTRIBUTES).map_err(io(&attributes))?;
        // Forced: a file the archive carries is part of the baseline even
        // when an ignore rule would hide a new file like it.
        self.git(
            &repo,
            "staging the baseline",
            &["add", "-A", "-f", "--", "."],
            None,
        )?;
        self.git(
            &repo,
            "committing the baseline",
            &[
                "commit",
                "-q",
                "--allow-empty",
                "--no-gpg-sign",
                "-m",
                "baseline",
            ],
            None,
        )?;
        let id = |rev: &str| -> Result<String, SandboxError> {
            let out = self.git(&repo, "reading the baseline", &["rev-parse", rev], None)?;
            Ok(String::from_utf8_lossy(&out).trim().to_string())
        };
        Ok(Baseline::new(id("HEAD")?, id("HEAD^{tree}")?, git_dir))
    }

    /// What the tree differs from `baseline` by, leaving out the paths
    /// that match the git glob patterns `excluded` (and any repository
    /// inside the tree).
    pub(crate) fn diff(
        &self,
        tree: &Path,
        baseline: &Baseline,
        excluded: &[&str],
    ) -> Result<Patch, SandboxError> {
        let repo = Repo {
            git_dir: baseline.git_dir(),
            work_tree: tree,
        };
        // Git cannot stage a directory that is a repository of its own, and
        // refuses to stage anything else beside it.
        let nested: Vec<String> = nested_repositories(tree)?
            .iter()
            .map(|dir| format!(":(exclude,literal){}", dir.display()))
            .collect();
        let with_paths = |head: &[&str], excluding: &[String]| -> Vec<String> {
            let mut args: Vec<String> = head.iter().map(|a| (*a).to_string()).collect();
            args.extend(["--".into(), ".".into()]);
            args.extend(excluding.iter().cloned());
            args.extend(nested.iter().cloned());
            args
        };
        let run_git = |what: &str, args: Vec<String>| {
            let args: Vec<&str> = args.iter().map(String::as_str).collect();
            self.git(&repo, what, &args, None)
        };
        let excludes: Vec<String> = excluded
            .iter()
            .map(|p| format!(":(exclude,glob){p}"))
            .collect();
        // Staged without the exclusions, which are taken out of the index
        // afterwards: `git add` fails when a pathspec names a path the
        // tree's own ignore rules hide, as a repository's rules hide sven's
        // directory and bytecode.
        run_git("staging the tree", with_paths(&["add", "-A"], &[]))?;
        for pattern in excluded {
            run_git(
                "unstaging an excluded path",
                vec![
                    "rm".into(),
                    "-r".into(),
                    "--cached".into(),
                    // A file the worker's tools still write (sven's audit log)
                    // may change between the staging and this removal; git
                    // refuses to unstage such a file without force.
                    "-f".into(),
                    "-q".into(),
                    "--ignore-unmatch".into(),
                    "--".into(),
                    format!(":(glob){pattern}"),
                ],
            )?;
        }
        let commit = baseline.commit.as_str();
        let diff = run_git(
            "reading the tree's changes",
            with_paths(
                &[
                    "diff",
                    "--cached",
                    "--binary",
                    "--full-index",
                    "--no-renames",
                    "--no-ext-diff",
                    "--no-textconv",
                    commit,
                ],
                &excludes,
            ),
        )?;
        let names = run_git(
            "listing the tree's changes",
            with_paths(
                &[
                    "diff",
                    "--cached",
                    "--name-only",
                    "-z",
                    "--no-renames",
                    commit,
                ],
                &excludes,
            ),
        )?;
        Ok(Patch {
            diff,
            files: paths_from(&names),
        })
    }

    /// Applies `patch` to `tree` as a whole or not at all, and returns the
    /// paths it touched.
    pub(crate) fn apply(&self, tree: &Path, patch: &Patch) -> Result<Vec<PathBuf>, SandboxError> {
        if patch.is_empty() {
            return Ok(Vec::new());
        }
        // Git must not climb out of the tree to find a repository whose
        // subdirectory the tree would then be taken for.
        let ceiling = tree.parent().map(path_arg).transpose()?.unwrap_or_default();
        let mut env = self.env.clone();
        env.insert("GIT_CEILING_DIRECTORIES".into(), ceiling);
        let mut argv = self.git_argv(None);
        argv.extend(
            [
                "apply",
                "--apply",
                "--numstat",
                "-z",
                "--binary",
                "--whitespace=nowarn",
                "-",
            ]
            .map(String::from),
        );
        let mut spec = ProcessSpec::new(argv, tree, env, TOOL_TIMEOUT, MAX_TOOL_OUTPUT);
        spec.stdin = Some(patch.diff.clone());
        let out = checked(
            &format!("applying a patch to {}", tree.display()),
            run(&spec)?,
        )?;
        Ok(numstat_paths(&out))
    }

    /// `git`, its fixed configuration and `repo` if given, as an argv.
    fn git_argv(&self, repo: Option<&Repo<'_>>) -> Vec<String> {
        let mut argv = vec![self.git.executable.clone()];
        for setting in GIT_CONFIG {
            argv.extend(["-c".into(), setting.into()]);
        }
        if let Some(repo) = repo {
            argv.extend([
                "--git-dir".into(),
                repo.git_dir.display().to_string(),
                "--work-tree".into(),
                repo.work_tree.display().to_string(),
            ]);
        }
        argv
    }

    /// Runs git in the work tree of `repo`, returning its standard output.
    fn git(
        &self,
        repo: &Repo<'_>,
        what: &str,
        args: &[&str],
        stdin: Option<Vec<u8>>,
    ) -> Result<Vec<u8>, SandboxError> {
        let mut argv = self.git_argv(Some(repo));
        argv.extend(args.iter().map(|a| (*a).to_string()));
        let mut spec = ProcessSpec::new(
            argv,
            repo.work_tree,
            self.env.clone(),
            TOOL_TIMEOUT,
            MAX_TOOL_OUTPUT,
        );
        spec.stdin = stdin;
        checked(what, run(&spec)?)
    }

    /// Runs git without a repository in `cwd`.
    fn git_alone(&self, what: &str, cwd: &Path, args: &[&str]) -> Result<Vec<u8>, SandboxError> {
        let mut argv = self.git_argv(None);
        argv.extend(args.iter().map(|a| (*a).to_string()));
        self.checked_run(what, argv, cwd)
    }

    fn checked_run(
        &self,
        what: &str,
        argv: Vec<String>,
        cwd: &Path,
    ) -> Result<Vec<u8>, SandboxError> {
        let spec = ProcessSpec::new(argv, cwd, self.env.clone(), TOOL_TIMEOUT, MAX_TOOL_OUTPUT);
        checked(what, run(&spec)?)
    }
}

/// The standard output of a tool run that succeeded, or why it did not.
fn checked(what: &str, output: ProcessOutput) -> Result<Vec<u8>, SandboxError> {
    if output.exit_code == Some(0) && !output.timed_out && !output.stdout.truncated {
        return Ok(output.stdout.bytes);
    }
    let stderr = String::from_utf8_lossy(&output.stderr.bytes);
    let detail: String = stderr.trim().chars().take(DETAIL_CHARS).collect();
    let how = if output.timed_out {
        "timed out".to_string()
    } else if output.stdout.truncated {
        format!("its output passed {MAX_TOOL_OUTPUT} bytes")
    } else {
        format!("exit code {:?}", output.exit_code)
    };
    Err(failure(what.to_string(), format!("{how}: {detail}")))
}

fn failure(what: String, detail: String) -> SandboxError {
    SandboxError::Workspace { what, detail }
}

/// The repository that tracks the tree at `tree`: `<tree>.git` beside it.
fn git_dir_beside(tree: &Path) -> Result<PathBuf, SandboxError> {
    let name = tree.file_name().filter(|_| tree.parent().is_some());
    name.map(|name| {
        let mut beside = name.to_os_string();
        beside.push(".git");
        tree.with_file_name(beside)
    })
    .ok_or_else(|| {
        failure(
            format!("laying out the tree at {}", tree.display()),
            "the tree needs a parent directory to keep its baseline beside it".into(),
        )
    })
}

/// The directories below `tree` that hold a repository of their own (a
/// `.git` entry), relative to `tree`. Symbolic links are not followed.
fn nested_repositories(tree: &Path) -> Result<Vec<PathBuf>, SandboxError> {
    fn walk(root: &Path, dir: &Path, found: &mut Vec<PathBuf>) -> Result<(), SandboxError> {
        let entries = std::fs::read_dir(dir).map_err(io(dir))?;
        for entry in entries {
            let entry = entry.map_err(io(dir))?;
            let path = entry.path();
            let is_dir = entry.file_type().map_err(io(&path))?.is_dir();
            if !is_dir || entry.file_name() == ".git" {
                continue;
            }
            if path.join(".git").exists() {
                found.extend(path.strip_prefix(root).map(Path::to_path_buf));
            } else {
                walk(root, &path, found)?;
            }
        }
        Ok(())
    }
    let mut found = Vec::new();
    walk(tree, tree, &mut found)?;
    found.sort();
    Ok(found)
}

fn is_empty_dir(path: &Path) -> bool {
    std::fs::read_dir(path).is_ok_and(|mut entries| entries.next().is_none())
}

fn path_arg(path: &Path) -> Result<String, SandboxError> {
    path.to_str().map(str::to_string).ok_or_else(|| {
        io(path)(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "the path is not UTF-8",
        ))
    })
}

fn path_from(bytes: &[u8]) -> PathBuf {
    PathBuf::from(OsStr::from_bytes(bytes))
}

/// The NUL-separated paths of `git diff --name-only -z`.
fn paths_from(listing: &[u8]) -> Vec<PathBuf> {
    listing
        .split(|b| *b == 0)
        .filter(|name| !name.is_empty())
        .map(path_from)
        .collect()
}

/// The paths of `git apply --numstat -z`: records are
/// `<added>\t<deleted>\t<path>\0`, and a rename is
/// `<added>\t<deleted>\t\0<from>\0<to>\0`, of which the target is reported.
fn numstat_paths(out: &[u8]) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    let mut records = out.split(|b| *b == 0).filter(|r| !r.is_empty());
    while let Some(record) = records.next() {
        let mut fields = record.splitn(3, |b| *b == b'\t');
        let path = fields.nth(2).unwrap_or_default();
        if path.is_empty() {
            let _from = records.next();
            records
                .next()
                .map(path_from)
                .into_iter()
                .for_each(|to| paths.push(to));
        } else {
            paths.push(path_from(path));
        }
    }
    paths
}
