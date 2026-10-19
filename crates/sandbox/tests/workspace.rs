// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible, auditable workspaces for
// agents that change code, for its clients. If your team needs expertise in
// sandboxed code-editing agents or reproducible agent runs, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: a workspace environment is a source tree pinned by the digest of
//! its archive. Materialising it reproduces the archived tree byte for byte
//! with its executable bits; what a solver changed is read back as a patch
//! naming exactly the files it touched (never sven's directory or bytecode
//! caches), and that patch applied to a fresh materialisation gives the
//! same tree. A command in the tree is bounded: one that outlives its wall
//! time is killed with its process group. The container backend mounts the
//! tree read-write at `/work` with no network, and two records of the same
//! snapshot in the same sandbox have the same environment digest.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};

use splinter_sandbox::host::DEFAULT_SEARCH_PATH;
use splinter_sandbox::workspace::DEFAULT_EXCLUDED;
use splinter_sandbox::{
    CommandCall, ContainerSandbox, ImageRef, Limits, ProcessSandbox, ResolvedEnvironment, Sandbox,
    SandboxError, WorkspaceEnvironment, WorkspaceHost,
};

/// A scratch directory holding a fixture repository, its archive and the
/// trees materialised from it.
struct Fixture {
    root: tempfile::TempDir,
    host: Arc<WorkspaceHost>,
}

/// Runs `git` with a fixed identity and no host configuration.
fn git(dir: &Path, args: &[&str]) {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["-c", "user.name=t", "-c", "user.email=t@example.com"])
        .args(["-c", "commit.gpgsign=false", "-c", "core.autocrlf=false"])
        .args(args)
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_CONFIG_SYSTEM", "/dev/null")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn write(path: &Path, bytes: &[u8], mode: u32) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, bytes).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap();
}

impl Fixture {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let host = WorkspaceHost::resolve(DEFAULT_SEARCH_PATH, root.path()).unwrap();
        Self {
            root,
            host: Arc::new(host),
        }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.root.path().join(name)
    }

    /// A repository with an executable script, a binary, a file with CRLF
    /// line ends under an attribute that would convert them, an ignored but
    /// committed file, and an empty-looking nested directory; archived as a
    /// tar. Returns the archive and the work tree it was made from.
    fn archive(&self) -> (PathBuf, PathBuf) {
        let repo = self.path("origin");
        std::fs::create_dir_all(&repo).unwrap();
        git(&repo, &["init", "-q", "-b", "main"]);
        write(&repo.join("src/app.py"), b"print('one')\n", 0o644);
        write(&repo.join("bin/run.sh"), b"#!/bin/sh\necho run\n", 0o755);
        write(
            &repo.join("data/blob.bin"),
            &[0, 159, 146, 150, 255, 0],
            0o644,
        );
        write(&repo.join("notes.txt"), b"a\r\nb\r\n", 0o644);
        write(
            &repo.join(".gitattributes"),
            b"*.txt text eol=crlf\n",
            0o644,
        );
        write(&repo.join(".gitignore"), b"ignored.log\n", 0o644);
        write(&repo.join("ignored.log"), b"kept\n", 0o644);
        git(&repo, &["add", "-A", "-f"]);
        git(&repo, &["commit", "-q", "-m", "origin"]);
        let archive = self.path("snapshot.tar");
        git(
            &repo,
            &[
                "archive",
                "--format=tar",
                "-o",
                archive.to_str().unwrap(),
                "HEAD",
            ],
        );
        (archive, repo)
    }

    fn environment(&self, archive: &Path, limits: Limits) -> WorkspaceEnvironment {
        let snapshot = self.host.snapshot(archive).unwrap();
        let sandbox = ProcessSandbox::new(self.path("calls"), BTreeMap::new(), &[], limits);
        std::fs::create_dir_all(self.path("calls")).unwrap();
        WorkspaceEnvironment::new(snapshot, Arc::new(sandbox), self.host.clone())
    }
}

/// Every file under `dir` (except the named top-level entries) with its
/// bytes and whether it is executable.
fn listing(dir: &Path, skip: &[&str]) -> BTreeMap<String, (Vec<u8>, bool)> {
    fn walk(root: &Path, dir: &Path, skip: &[&str], out: &mut BTreeMap<String, (Vec<u8>, bool)>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            let rel = entry
                .path()
                .strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .into_owned();
            if skip.contains(&rel.as_str()) {
                continue;
            }
            if entry.file_type().unwrap().is_dir() {
                walk(root, &entry.path(), skip, out);
            } else {
                let mode = entry.metadata().unwrap().permissions().mode();
                out.insert(
                    rel,
                    (std::fs::read(entry.path()).unwrap(), mode & 0o111 != 0),
                );
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(dir, dir, skip, &mut out);
    out
}

fn argv(parts: &[&str]) -> CommandCall {
    CommandCall {
        argv: parts.iter().map(|p| p.to_string()).collect(),
        stdin: None,
    }
}

fn sh(script: &str) -> CommandCall {
    argv(&["sh", "-c", script])
}

#[test]
fn a_materialised_snapshot_reproduces_the_archived_tree_byte_for_byte() {
    let fx = Fixture::new();
    let (archive, origin) = fx.archive();
    let env = fx.environment(&archive, Limits::default());
    let tree = fx.path("tree");

    let baseline = env.materialize(&archive, &tree).unwrap();

    let expected = listing(&origin, &[".git"]);
    assert!(expected["bin/run.sh"].1, "the fixture has an executable");
    assert_eq!(listing(&tree, &[]), expected);
    assert_eq!(
        baseline.tree_id,
        env.snapshot().tree_id,
        "the materialised tree is the tree the snapshot names"
    );
    // The baseline holds the archived bytes: no attribute converted the
    // line ends of the one file that asked for it.
    let committed = Command::new("git")
        .arg("--git-dir")
        .arg(baseline.git_dir())
        .args(["cat-file", "blob", "HEAD:notes.txt"])
        .output()
        .unwrap();
    assert_eq!(committed.stdout, b"a\r\nb\r\n");
    // Nothing was changed yet, nothing is reported.
    let patch = env.diff_since(&tree, &baseline, &DEFAULT_EXCLUDED).unwrap();
    assert!(patch.is_empty(), "{:?}", patch.files);
}

#[test]
fn an_archive_other_than_the_snapshots_is_refused() {
    let fx = Fixture::new();
    let (archive, origin) = fx.archive();
    let env = fx.environment(&archive, Limits::default());
    write(&origin.join("src/app.py"), b"print('two')\n", 0o644);
    git(&origin, &["commit", "-q", "-a", "-m", "second"]);
    let other = fx.path("other.tar");
    git(
        &origin,
        &[
            "archive",
            "--format=tar",
            "-o",
            other.to_str().unwrap(),
            "HEAD",
        ],
    );

    let refused = env.materialize(&other, &fx.path("tree"));

    assert!(
        matches!(refused, Err(SandboxError::Workspace { .. })),
        "{refused:?}"
    );
    assert!(
        !fx.path("tree").join("src").exists(),
        "nothing was extracted"
    );
}

#[test]
fn the_diff_after_an_edit_names_exactly_the_edited_file() {
    let fx = Fixture::new();
    let (archive, _) = fx.archive();
    let env = fx.environment(&archive, Limits::default());
    let tree = fx.path("tree");
    let baseline = env.materialize(&archive, &tree).unwrap();

    write(&tree.join("src/app.py"), b"print('two')\n", 0o644);
    // Noise the solver's tooling leaves behind: sven's directory, bytecode
    // caches, a repository of its own.
    write(&tree.join(".sven/audit.log"), b"x\n", 0o644);
    write(&tree.join("src/__pycache__/app.pyc"), b"\0\0", 0o644);
    write(&tree.join("__pycache__/top.pyc"), b"\0\0", 0o644);
    write(
        &tree.join("nested/kept.txt"),
        b"in a repository of its own\n",
        0o644,
    );
    git(&tree.join("nested"), &["init", "-q"]);
    write(&tree.join("ignored.log"), b"changed but tracked\n", 0o644);
    write(&tree.join("fresh.log"), b"untracked\n", 0o644);
    write(&tree.join(".gitignore"), b"ignored.log\nfresh.log\n", 0o644);

    let patch = env.diff_since(&tree, &baseline, &DEFAULT_EXCLUDED).unwrap();

    assert_eq!(
        patch.files,
        [".gitignore", "ignored.log", "src/app.py"].map(PathBuf::from)
    );
    let text = String::from_utf8_lossy(&patch.diff);
    assert!(text.contains("+print('two')"), "{text}");
    assert!(
        !text.contains("pycache") && !text.contains(".sven"),
        "{text}"
    );
}

#[test]
fn a_tree_whose_own_ignore_rules_hide_the_excluded_paths_still_has_its_change_read_back() {
    // Most repositories ignore sven's directory and bytecode; a pathspec that
    // names an ignored path must not make the read-back fail.
    let fx = Fixture::new();
    let (archive, _) = fx.archive();
    let env = fx.environment(&archive, Limits::default());
    let tree = fx.path("tree");
    let baseline = env.materialize(&archive, &tree).unwrap();

    write(&tree.join("src/app.py"), b"print('two')\n", 0o644);
    write(&tree.join(".gitignore"), b".sven/\n__pycache__/\n", 0o644);
    write(&tree.join(".sven/audit.log"), b"x\n", 0o644);
    write(&tree.join("src/__pycache__/app.pyc"), b"\0\0", 0o644);

    let patch = env.diff_since(&tree, &baseline, &DEFAULT_EXCLUDED).unwrap();

    assert_eq!(patch.files, [".gitignore", "src/app.py"].map(PathBuf::from));
    let text = String::from_utf8_lossy(&patch.diff);
    assert!(
        !text.contains("app.pyc") && !text.contains("audit.log"),
        "{text}"
    );
}

#[test]
fn a_patch_applied_to_a_fresh_materialisation_gives_the_same_tree() {
    let fx = Fixture::new();
    let (archive, _) = fx.archive();
    let env = fx.environment(&archive, Limits::default());
    let edited = fx.path("edited");
    let baseline = env.materialize(&archive, &edited).unwrap();

    write(&edited.join("src/app.py"), b"print('two')\n", 0o644);
    write(&edited.join("src/new.py"), b"VALUE = 1\n", 0o644);
    write(
        &edited.join("data/blob.bin"),
        &[1, 0, 2, 255, 254, 0, 0],
        0o644,
    );
    write(&edited.join("bin/run.sh"), b"#!/bin/sh\necho run\n", 0o644);
    std::fs::remove_file(edited.join("notes.txt")).unwrap();
    write(&edited.join(".sven/audit.log"), b"x\n", 0o644);
    let patch = env
        .diff_since(&edited, &baseline, &DEFAULT_EXCLUDED)
        .unwrap();

    let fresh = fx.path("fresh");
    env.materialize(&archive, &fresh).unwrap();
    let touched = env.apply(&fresh, &patch).unwrap();

    let mut touched = touched;
    touched.sort();
    assert_eq!(
        touched,
        [
            "bin/run.sh",
            "data/blob.bin",
            "notes.txt",
            "src/app.py",
            "src/new.py"
        ]
        .map(PathBuf::from)
    );
    assert_eq!(listing(&fresh, &[]), listing(&edited, &[".sven"]));
    assert!(
        !listing(&fresh, &[])["bin/run.sh"].1,
        "the mode change applied"
    );
}

#[test]
fn a_patch_that_does_not_apply_changes_nothing() {
    let fx = Fixture::new();
    let (archive, _) = fx.archive();
    let env = fx.environment(&archive, Limits::default());
    let edited = fx.path("edited");
    let baseline = env.materialize(&archive, &edited).unwrap();
    write(&edited.join("src/app.py"), b"print('two')\n", 0o644);
    write(&edited.join("src/new.py"), b"VALUE = 1\n", 0o644);
    let patch = env
        .diff_since(&edited, &baseline, &DEFAULT_EXCLUDED)
        .unwrap();

    let drifted = fx.path("drifted");
    env.materialize(&archive, &drifted).unwrap();
    write(&drifted.join("src/app.py"), b"print('else')\n", 0o644);
    let before = listing(&drifted, &[]);

    let refused = env.apply(&drifted, &patch);

    assert!(
        matches!(refused, Err(SandboxError::Workspace { .. })),
        "{refused:?}"
    );
    assert_eq!(listing(&drifted, &[]), before, "an apply is all or nothing");
}

#[test]
fn a_command_runs_in_the_tree_with_only_the_sandboxs_environment() {
    let fx = Fixture::new();
    let (archive, _) = fx.archive();
    let env = fx.environment(&archive, Limits::default());
    let tree = fx.path("tree");
    let baseline = env.materialize(&archive, &tree).unwrap();

    let result = env
        .run(
            &tree,
            &CommandCall {
                argv: vec![
                    "sh".into(),
                    "-c".into(),
                    "cat > from-stdin.txt; pwd; env".into(),
                ],
                stdin: Some("typed".into()),
            },
        )
        .unwrap();

    assert_eq!(result.exit_code, Some(0), "{result:?}");
    let mut lines = result.stdout.lines();
    assert_eq!(
        Path::new(lines.next().unwrap()).canonicalize().unwrap(),
        tree.canonicalize().unwrap()
    );
    let names: Vec<&str> = lines
        .filter_map(|l| l.split_once('=').map(|(k, _)| k))
        .collect();
    for name in &names {
        assert!(
            ["PATH", "LANG", "TERM", "HOME", "TMPDIR", "PWD", "SHLVL", "_", "OLDPWD"]
                .contains(name),
            "unexpected variable {name}"
        );
    }
    assert_eq!(
        std::fs::read_to_string(tree.join("from-stdin.txt")).unwrap(),
        "typed"
    );
    let patch = env.diff_since(&tree, &baseline, &DEFAULT_EXCLUDED).unwrap();
    assert_eq!(patch.files, [PathBuf::from("from-stdin.txt")]);
}

/// A sandbox owns its scratch root: a command in a tree runs under a root
/// that does not exist yet, and makes it, as a code call does.
#[test]
fn a_command_runs_under_a_scratch_root_that_does_not_exist_yet() {
    let fx = Fixture::new();
    let (archive, _) = fx.archive();
    let snapshot = fx.host.snapshot(&archive).unwrap();
    let calls = fx.path("not/made/yet/calls");
    let sandbox = ProcessSandbox::new(&calls, BTreeMap::new(), &[], Limits::default());
    let env = WorkspaceEnvironment::new(snapshot, Arc::new(sandbox), fx.host.clone());
    let tree = fx.path("tree");
    env.materialize(&archive, &tree).unwrap();

    let result = env.run(&tree, &argv(&["true"])).unwrap();
    assert_eq!(result.exit_code, Some(0), "{result:?}");
    assert!(calls.is_dir());
}

#[test]
fn a_command_outliving_its_wall_time_is_killed_with_its_process_group() {
    let fx = Fixture::new();
    let (archive, _) = fx.archive();
    let limits = Limits {
        wall_time_ms: 400,
        ..Limits::default()
    };
    let env = fx.environment(&archive, limits);
    let tree = fx.path("tree");
    env.materialize(&archive, &tree).unwrap();

    // The command runs in a pid namespace of its own, so the id its shell
    // reports means nothing here: the background process is found by the
    // argument only it was given.
    let began = Instant::now();
    let result = env.run(&tree, &sh("sleep 61.0731 & wait")).unwrap();

    assert!(result.timed_out, "{result:?}");
    assert!(
        began.elapsed() < Duration::from_secs(10),
        "{:?}",
        began.elapsed()
    );
    let deadline = Instant::now() + Duration::from_secs(5);
    while process_with_argument("61.0731") && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(
        !process_with_argument("61.0731"),
        "the background process outlived the call"
    );
}

/// Whether some process of this host has `argument` among its arguments.
fn process_with_argument(argument: &str) -> bool {
    std::fs::read_dir("/proc")
        .unwrap()
        .filter_map(Result::ok)
        .filter_map(|entry| std::fs::read(entry.path().join("cmdline")).ok())
        .any(|cmdline| {
            cmdline
                .split(|b| *b == 0)
                .any(|arg| arg == argument.as_bytes())
        })
}

const IMAGE_DIGEST: &str =
    "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

/// A `docker` that records its arguments, one per line, and succeeds.
fn recording_docker(dir: &Path) -> (PathBuf, PathBuf) {
    let record = dir.join("docker-argv");
    let script = dir.join("docker");
    let text = format!(
        "#!/bin/sh\nfor a in \"$@\"; do printf '%s\\n' \"$a\"; done > '{}'\ncat >> '{0}.stdin'\n",
        record.display()
    );
    write_by_another_process(&script, text.as_bytes());
    (script, record)
}

/// Writes the executable `path` from a process of its own. Written from this
/// one, the file would be open for writing here while a parallel test forks,
/// and the child would keep it open until its own exec: executing the script
/// meanwhile fails with "text file busy". A writer that is another process
/// leaves no descriptor of the file in this one for a fork to copy.
fn write_by_another_process(path: &Path, content: &[u8]) {
    use std::io::Write;
    use std::process::Stdio;
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut writer = Command::new("sh")
        .args(["-c", "cat > \"$0\" && chmod 755 \"$0\""])
        .arg(path)
        .stdin(Stdio::piped())
        .spawn()
        .unwrap();
    writer.stdin.take().unwrap().write_all(content).unwrap();
    assert!(writer.wait().unwrap().success());
}

#[test]
fn the_container_mounts_the_tree_at_work_read_write_with_no_network() {
    let fx = Fixture::new();
    let (docker, record) = recording_docker(&fx.path("bin"));
    let image = ImageRef::parse(&format!("python@{IMAGE_DIGEST}")).unwrap();
    std::fs::create_dir_all(fx.path("calls")).unwrap();
    let sandbox = ContainerSandbox::new(
        docker,
        BTreeMap::new(),
        image,
        fx.path("calls"),
        Limits::default(),
    );
    let tree = fx.path("tree");
    std::fs::create_dir_all(&tree).unwrap();

    let result = sandbox
        .run_command(
            &tree,
            &CommandCall {
                argv: vec!["python3".into(), "-m".into(), "unittest".into()],
                stdin: Some("input".into()),
            },
        )
        .unwrap();

    assert_eq!(result.exit_code, Some(0), "{result:?}");
    let argv: Vec<String> = std::fs::read_to_string(&record)
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect();
    let joined = argv.join(" ");
    let mount = format!(
        "type=bind,source={},target=/work",
        tree.canonicalize().unwrap().display()
    );
    assert!(
        argv.contains(&mount),
        "no read-write bind of the tree: {joined}"
    );
    assert!(
        !argv.iter().any(|a| a.starts_with("/work:")),
        "nothing may be mounted over the tree: {joined}"
    );
    for expected in [
        "--network none",
        "--read-only",
        "--cap-drop ALL",
        "--workdir /work",
    ] {
        assert!(joined.contains(expected), "{expected} missing: {joined}");
    }
    assert!(
        joined.ends_with(&format!(
            "--interactive python@{IMAGE_DIGEST} python3 -m unittest"
        )),
        "{joined}"
    );
    assert!(!joined.contains(":latest"));
    assert_eq!(
        std::fs::read_to_string(format!("{}.stdin", record.display())).unwrap(),
        "input"
    );
    assert!(
        std::fs::read_dir(fx.path("calls"))
            .unwrap()
            .next()
            .is_none(),
        "the call left nothing in the scratch root"
    );
}

/// `repo@sha256:<digest>` of the live test's image, pulled when missing.
fn pinned_test_image() -> ImageRef {
    let docker = |args: &[&str]| {
        let out = Command::new("docker").args(args).output().unwrap();
        assert!(out.status.success(), "docker {args:?}");
        String::from_utf8(out.stdout).unwrap()
    };
    let format = "{{index .RepoDigests 0}}";
    let inspect = ["image", "inspect", "--format", format, "python:3-alpine"];
    let reference = Command::new("docker")
        .args(inspect)
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map_or_else(
            || {
                docker(&["pull", "--quiet", "python:3-alpine"]);
                docker(&inspect)
            },
            |out| String::from_utf8(out.stdout).unwrap(),
        );
    ImageRef::parse(reference.trim()).unwrap()
}

#[test]
#[ignore = "needs a docker daemon"]
fn a_container_command_edits_only_the_mounted_tree() {
    let fx = Fixture::new();
    let tree = fx.path("tree");
    std::fs::create_dir_all(&tree).unwrap();
    let image = pinned_test_image();
    let sandbox = ContainerSandbox::new(
        PathBuf::from("docker"),
        [("PATH", DEFAULT_SEARCH_PATH), ("HOME", "/tmp")]
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .into_iter()
            .collect(),
        image,
        fx.root.path(),
        Limits::default(),
    );
    let result = sandbox
        .run_command(
            &tree,
            &sh("echo made > made.txt; touch /etc/x 2>&1 || echo read-only"),
        )
        .unwrap();
    assert_eq!(result.exit_code, Some(0), "{result:?}");
    assert_eq!(
        std::fs::read_to_string(tree.join("made.txt")).unwrap(),
        "made\n"
    );
    assert!(result.stdout.contains("read-only"), "{result:?}");
}

#[test]
fn two_records_of_the_same_snapshot_and_sandbox_have_the_same_digest() {
    let fx = Fixture::new();
    let (archive, origin) = fx.archive();
    let first = ResolvedEnvironment::Workspace(fx.environment(&archive, Limits::default()));
    let second = ResolvedEnvironment::Workspace(fx.environment(&archive, Limits::default()));

    let (a, b) = (first.record().unwrap(), second.record().unwrap());

    assert_eq!(a.kind, "workspace");
    assert!(a.snapshot.is_some());
    assert_eq!(a.snapshot, b.snapshot);
    assert_eq!(a, b);
    for key in ["snapshot", "tree_id", "git", "sandbox"] {
        assert!(a.spec.get(key).is_some(), "{key} missing from {}", a.spec);
    }
    assert_eq!(a.spec["sandbox"]["isolation"], "none");

    // A different snapshot, a different sandbox: a different environment.
    write(&origin.join("src/app.py"), b"print('two')\n", 0o644);
    git(&origin, &["commit", "-q", "-a", "-m", "second"]);
    let changed = fx.path("changed.tar");
    git(
        &origin,
        &[
            "archive",
            "--format=tar",
            "-o",
            changed.to_str().unwrap(),
            "HEAD",
        ],
    );
    let other_snapshot =
        ResolvedEnvironment::Workspace(fx.environment(&changed, Limits::default()));
    let other_limits = ResolvedEnvironment::Workspace(fx.environment(
        &archive,
        Limits {
            wall_time_ms: 1,
            ..Limits::default()
        },
    ));
    assert_ne!(other_snapshot.record().unwrap().snapshot, a.snapshot);
    assert_ne!(other_limits.record().unwrap().snapshot, a.snapshot);
}
