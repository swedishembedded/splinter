// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements source capture that grounds every learned
// fact in exact, traceable bytes, for its clients. If your team needs
// expertise in knowledge acquisition or training-data provenance, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: a document, a repository and a command run are captured as
//! immutable sources whose bytes a span resolves back to; capture is
//! idempotent, deterministic and bounded, and never silent about what it
//! left out.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use splinter_knowledge::capture::{
    capture_command, capture_document, capture_repository, default_environment, CaptureError,
    CommandSpec, ProcessError, DEFAULT_MAX_FILE_BYTES,
};
use splinter_record::clock::FixedClock;
use splinter_record::experience::Span;
use splinter_record::source::{Origin, PartRef, SkipReason, Source};
use splinter_record::sources::SourceStore;
use splinter_record::StateRoot;

const AT: &str = "2026-09-30T08:00:00.000Z";
const LATER: &str = "2026-09-30T09:30:00.000Z";

struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        static N: AtomicU32 = AtomicU32::new(0);
        let path = std::env::temp_dir().join(format!(
            "splinter-capture-{name}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn store(&self) -> SourceStore {
        SourceStore::open(&StateRoot::new(self.0.join("state")))
    }
    fn write(&self, rel: &str, bytes: &[u8]) -> PathBuf {
        let path = self.0.join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, bytes).unwrap();
        path
    }
    fn blobs(&self) -> usize {
        fs::read_dir(self.0.join("state/sources/blobs"))
            .unwrap()
            .count()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn span_of(source: &Source, part: &str, text: &str, passage: &str) -> Span {
    let start = text.find(passage).unwrap() as u64;
    Span::in_part(
        PartRef {
            source: source.id.clone(),
            name: part.into(),
        },
        source.part(part).unwrap().content.clone(),
        start,
        start + passage.len() as u64,
    )
    .unwrap()
}

#[test]
fn a_document_is_captured_once_and_its_spans_resolve() {
    let scratch = Scratch::new("document");
    let text = "# Brain\n\nbrain serve starts the server.\n";
    let path = scratch.write("manual/brain.md", text.as_bytes());
    let store = scratch.store();

    let captured = capture_document(&path, DEFAULT_MAX_FILE_BYTES, &FixedClock::new(AT)).unwrap();
    let source = captured.source();
    assert_eq!(source.kind(), "document");
    let [part] = &source.parts[..] else {
        panic!("one part: {:?}", source.parts)
    };
    assert_eq!(part.name, "brain.md");
    assert_eq!(part.media_type, "text/markdown");
    let id = store.put_source(&captured).unwrap();
    let span = span_of(source, "brain.md", text, "brain serve starts the server.");
    assert_eq!(
        store.read_span(&span).unwrap(),
        b"brain serve starts the server."
    );

    let again = capture_document(&path, DEFAULT_MAX_FILE_BYTES, &FixedClock::new(LATER)).unwrap();
    assert_eq!(store.put_source(&again).unwrap(), id);
    assert_eq!(scratch.blobs(), 1, "identical content is stored once");
    assert_eq!(store.list().unwrap(), vec![id]);
}

#[test]
fn a_binary_or_oversized_document_is_refused() {
    let scratch = Scratch::new("binary");
    let binary = scratch.write("blob.dat", &[0x89, b'P', b'N', b'G', 0, 0xff, 0xfe]);
    assert!(matches!(
        capture_document(&binary, DEFAULT_MAX_FILE_BYTES, &FixedClock::new(AT)),
        Err(CaptureError::NotText { .. })
    ));
    let big = scratch.write("big.txt", &[b'a'; 64]);
    assert!(matches!(
        capture_document(&big, 16, &FixedClock::new(AT)),
        Err(CaptureError::TooLarge { .. })
    ));
}

/// Writes the same tree twice, in opposite creation orders.
fn tree(scratch: &Scratch, root: &str, reverse: bool) -> PathBuf {
    let mut files: Vec<(&str, &[u8])> = vec![
        ("README.md", b"# Tool\n\nIt works.\n"),
        ("src/main.rs", b"fn main() {}\n"),
        ("src/a/lib.rs", b"pub fn a() {}\n"),
        ("target/debug/out.txt", b"build output\n"),
        ("node_modules/x/index.js", b"module\n"),
        ("logo.png", &[0x89, b'P', b'N', b'G', 0, 1, 2]),
        ("big.log", &[b'x'; 200]),
    ];
    if reverse {
        files.reverse();
    }
    for (rel, bytes) in files {
        scratch.write(&format!("{root}/{rel}"), bytes);
    }
    scratch.0.join(root)
}

#[test]
fn a_repository_is_every_text_file_in_path_order_minus_the_ignored() {
    // `.git` is covered by the git test below: a `.git` entry here would
    // have to be a repository git can read.
    let scratch = Scratch::new("repository");
    let a = tree(&scratch, "a", false);
    let b = tree(&scratch, "b", true);

    let first = capture_repository(&a, 100, &FixedClock::new(AT)).unwrap();
    let names: Vec<&str> = first
        .source()
        .parts
        .iter()
        .map(|p| p.name.as_str())
        .collect();
    assert_eq!(names, ["README.md", "src/a/lib.rs", "src/main.rs"]);
    let Origin::Repository { skipped, .. } = &first.source().origin else {
        panic!("{:?}", first.source().origin)
    };
    let skipped: Vec<(&str, &SkipReason)> = skipped
        .iter()
        .map(|s| (s.path.as_str(), &s.reason))
        .collect();
    assert_eq!(
        skipped,
        [
            ("big.log", &SkipReason::TooLarge { bytes: 200 }),
            ("logo.png", &SkipReason::NotText),
        ],
        "what is left out is recorded"
    );

    // The same content created in the opposite order has the same parts;
    // only the origin path differs.
    let second = capture_repository(&b, 100, &FixedClock::new(AT)).unwrap();
    assert_eq!(first.source().parts, second.source().parts);
    assert_eq!(
        capture_repository(&a, 100, &FixedClock::new(LATER))
            .unwrap()
            .source()
            .id,
        first.source().id
    );
}

/// `git` with explicit identity, isolated from the caller's configuration.
fn git(dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new("git")
        .current_dir(dir)
        .args([
            "-c",
            "user.name=Splinter Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
        ])
        .args(args)
        .env("GIT_AUTHOR_DATE", "2026-01-01T00:00:00Z")
        .env("GIT_COMMITTER_DATE", "2026-01-01T00:00:00Z")
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .env_remove("GIT_INDEX_FILE")
        .output()
        .unwrap()
}

#[test]
fn a_git_repository_records_its_head_and_whether_it_is_dirty() {
    if Command::new("git").arg("--version").output().is_err() {
        eprintln!("SKIPPED: git is not installed; the git revision cannot be checked");
        return;
    }
    let scratch = Scratch::new("git");
    let repo = scratch.0.join("repo");
    scratch.write("repo/README.md", b"# Repo\n\nFirst.\n");
    assert!(git(&repo, &["init", "-q"]).status.success());
    assert!(git(&repo, &["add", "."]).status.success());
    assert!(git(&repo, &["commit", "-q", "-m", "first"])
        .status
        .success());
    let head = String::from_utf8(git(&repo, &["rev-parse", "HEAD"]).stdout).unwrap();

    let clean = capture_repository(&repo, DEFAULT_MAX_FILE_BYTES, &FixedClock::new(AT)).unwrap();
    let Origin::Repository { revision, .. } = &clean.source().origin else {
        panic!()
    };
    let revision = revision.as_ref().unwrap();
    assert_eq!(revision.commit.as_deref(), Some(head.trim()));
    assert!(!revision.dirty);
    assert!(
        clean.source().part(".git/HEAD").is_none(),
        "{:?}",
        clean.source().parts
    );

    scratch.write("repo/README.md", b"# Repo\n\nEdited.\n");
    let dirty = capture_repository(&repo, DEFAULT_MAX_FILE_BYTES, &FixedClock::new(AT)).unwrap();
    let Origin::Repository { revision, .. } = &dirty.source().origin else {
        panic!()
    };
    assert!(revision.as_ref().unwrap().dirty);
    assert_ne!(dirty.source().id, clean.source().id);
}

fn spec(argv: &[&str], cwd: &Path) -> CommandSpec {
    let env = default_environment([
        (
            "PATH".to_string(),
            "/usr/local/bin:/usr/bin:/bin".to_string(),
        ),
        ("HOME".to_string(), cwd.display().to_string()),
    ]);
    CommandSpec::new(argv.iter().map(|a| a.to_string()).collect(), cwd, env)
}

fn text(store: &SourceStore, source: &Source, part: &str) -> String {
    String::from_utf8(store.read_part(&source.id, part).unwrap()).unwrap()
}

#[test]
fn a_command_run_is_captured_as_stdout_stderr_and_exit_code() {
    let scratch = Scratch::new("command");
    let store = scratch.store();
    let run = spec(&["sh", "-c", "echo out; echo err 1>&2; exit 3"], &scratch.0);
    let captured = capture_command(&run, &FixedClock::new(AT)).unwrap();
    store.put_source(&captured).unwrap();
    let source = captured.source();
    let Origin::Command {
        argv,
        exit_code,
        timed_out,
        stdout_truncated,
        stderr_truncated,
        ..
    } = &source.origin
    else {
        panic!("{:?}", source.origin)
    };
    assert_eq!(argv, &run.argv);
    assert_eq!(*exit_code, Some(3));
    assert!(!timed_out && !stdout_truncated && !stderr_truncated);
    assert_eq!(text(&store, source, "stdout"), "out\n");
    assert_eq!(text(&store, source, "stderr"), "err\n");
    assert_eq!(source.part("stdout").unwrap().media_type, "text/plain");

    // Stdin is closed: a reader of it sees end of input at once.
    let cat = capture_command(&spec(&["cat"], &scratch.0), &FixedClock::new(AT)).unwrap();
    let Origin::Command { exit_code, .. } = &cat.source().origin else {
        panic!()
    };
    assert_eq!(*exit_code, Some(0));

    // Rerunning a deterministic command is the same source.
    let again = capture_command(&run, &FixedClock::new(LATER)).unwrap();
    assert_eq!(again.source().id, source.id);
}

#[test]
fn a_command_that_outlives_its_timeout_is_killed_and_recorded() {
    let scratch = Scratch::new("timeout");
    let mut run = spec(
        &["sh", "-c", "echo started; sleep 30; echo never"],
        &scratch.0,
    );
    run.timeout = Duration::from_millis(300);
    let began = Instant::now();
    let captured = capture_command(&run, &FixedClock::new(AT)).unwrap();
    assert!(
        began.elapsed() < Duration::from_secs(10),
        "the capture waited for the sleeping process: {:?}",
        began.elapsed()
    );
    let Origin::Command {
        exit_code,
        timed_out,
        ..
    } = &captured.source().origin
    else {
        panic!()
    };
    assert!(*timed_out);
    assert_eq!(*exit_code, None, "a killed process has no exit code");
    assert_eq!(captured.content("stdout"), Some(&b"started\n"[..]));
}

#[test]
fn output_beyond_the_cap_is_truncated_and_recorded() {
    let scratch = Scratch::new("cap");
    let mut run = spec(
        &[
            "sh",
            "-c",
            "i=0; while [ $i -lt 200 ]; do echo 0123456789; i=$((i+1)); done",
        ],
        &scratch.0,
    );
    run.output_cap = 64;
    let captured = capture_command(&run, &FixedClock::new(AT)).unwrap();
    let Origin::Command {
        exit_code,
        stdout_truncated,
        stderr_truncated,
        ..
    } = &captured.source().origin
    else {
        panic!()
    };
    assert_eq!(*exit_code, Some(0), "the process ran to completion");
    assert!(*stdout_truncated && !stderr_truncated);
    let stdout = captured.content("stdout").unwrap();
    assert_eq!(stdout.len(), 64);
    assert!(stdout.starts_with(b"0123456789\n0123456789\n"));
}

#[test]
fn the_child_sees_only_the_environment_it_is_given() {
    let scratch = Scratch::new("env");
    let home = scratch.0.display().to_string();
    let parent: BTreeMap<String, String> = [
        ("PATH", "/usr/local/bin:/usr/bin:/bin"),
        ("HOME", home.as_str()),
        ("LANG", "C.UTF-8"),
        ("SPLINTER_TEST_SECRET", "must-not-leak"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect();
    let env = default_environment(parent);
    let run = CommandSpec::new(vec!["env".to_string()], &scratch.0, env);
    let captured = capture_command(&run, &FixedClock::new(AT)).unwrap();
    let stdout = String::from_utf8(captured.content("stdout").unwrap().to_vec()).unwrap();
    let mut seen: Vec<&str> = stdout
        .lines()
        .filter_map(|l| l.split_once('=').map(|(k, _)| k))
        .collect();
    seen.sort_unstable();
    // Nothing of the test process's own environment, and nothing outside
    // the allowlist, reaches the child.
    assert_eq!(seen, ["HOME", "LANG", "PATH", "TERM"], "{stdout}");
    assert!(stdout.contains("TERM=dumb\n"), "{stdout}");
    assert!(!stdout.contains("must-not-leak"));
}

#[test]
fn a_command_that_cannot_start_is_an_error() {
    let scratch = Scratch::new("missing");
    assert!(matches!(
        capture_command(
            &spec(&["splinter-no-such-program"], &scratch.0),
            &FixedClock::new(AT)
        ),
        Err(CaptureError::Process(ProcessError::Spawn { .. }))
    ));
    assert!(matches!(
        capture_command(&spec(&[], &scratch.0), &FixedClock::new(AT)),
        Err(CaptureError::Process(ProcessError::EmptyArgv))
    ));
}
