// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible snapshots of source trees that
// agents are trained and measured on, for its clients. If your team needs
// expertise in turning a codebase into executable training tasks, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: the tracked files of a git revision are captured as a snapshot that
//! depends on the tree alone, not on the commit that makes it; the source
//! records where it came from and the terms it is under; a checkout with
//! uncommitted changes is refused unless the caller allows it.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::fs;
use std::os::unix::fs::PermissionsExt;

use common::{git, write, Repo, PRICING};
use splinter_core::terms::Terms;
use splinter_knowledge::capture::{DirtyPolicy, TreeError, TreeRequest, SNAPSHOT_PART};
use splinter_sandbox::{Limits, ProcessSandbox, WorkspaceEnvironment};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Lays the archive of a captured tree out and returns the directory.
fn laid_out(
    repo: &Repo,
    tree: &splinter_knowledge::capture::CapturedTree,
    name: &str,
) -> std::path::PathBuf {
    let calls = repo.scratch.0.join("calls");
    fs::create_dir_all(&calls).unwrap();
    let sandbox = ProcessSandbox::new(&calls, BTreeMap::new(), &[], Limits::default());
    let env =
        WorkspaceEnvironment::new(tree.snapshot.clone(), Arc::new(sandbox), repo.host.clone());
    let into = repo.scratch.0.join(name);
    env.materialize(&tree.archive_path, &into).unwrap();
    into
}

#[test]
fn a_revision_is_captured_as_the_same_snapshot_whichever_commit_makes_it() {
    let repo = Repo::new("capture", &[]);
    let script = repo.repo.join("run.sh");
    fs::write(&script, "#!/bin/sh\n").unwrap();
    fs::set_permissions(&script, fs::Permissions::from_mode(0o755)).unwrap();
    git(&repo.repo, &["add", "run.sh"]);
    git(&repo.repo, &["commit", "-q", "-m", "script"]);
    let first = repo.capture("HEAD");
    git(
        &repo.repo,
        &["commit", "-q", "--allow-empty", "-m", "nothing changed"],
    );
    let second = repo.capture("HEAD");
    assert_eq!(first.snapshot, second.snapshot);
    assert_ne!(first.commit, second.commit);
    assert_eq!(
        first.snapshot.tree_id,
        git(&repo.repo, &["rev-parse", "HEAD^{tree}"]).trim()
    );

    // Laid out, the tree is exactly the tracked files: the executable bit
    // is kept and no .git is there.
    let laid = laid_out(&repo, &first, "laid");
    assert!(!laid.join(".git").exists());
    let mode = fs::metadata(laid.join("run.sh"))
        .unwrap()
        .permissions()
        .mode();
    assert_ne!(mode & 0o100, 0, "run.sh keeps its executable bit");

    // The source records where it came from.
    let origin = serde_json::to_string(&first.source.source().origin).unwrap();
    assert!(origin.contains(&first.commit), "{origin}");
    assert!(origin.contains("\"dirty\":false"), "{origin}");
    let part = first.source.content(SNAPSHOT_PART).unwrap();
    let part = String::from_utf8(part.to_vec()).unwrap();
    assert!(part.contains(&first.snapshot.tree_id), "{part}");
    assert!(part.contains(first.snapshot.digest.as_str()), "{part}");
}

#[test]
fn a_dirty_checkout_is_refused_unless_allowed_and_terms_travel_with_the_source() {
    let repo = Repo::new("dirty", &[]);
    write(
        &repo.repo.join("shop/pricing.py"),
        "# edited, not committed\n",
    );
    assert!(matches!(
        repo.try_capture(&repo.request("HEAD")),
        Err(TreeError::Dirty { .. })
    ));

    let allowed = repo
        .try_capture(&TreeRequest {
            dirty: DirtyPolicy::Allow,
            terms: Some(Terms::public_domain("shop")),
            ..repo.request("HEAD")
        })
        .unwrap();
    let source = allowed.source.source();
    assert!(source.terms.is_some());
    assert!(serde_json::to_string(source)
        .unwrap()
        .contains("\"dirty\":true"));
    // The archive holds the committed file, not the edit.
    let laid = laid_out(&repo, &allowed, "laid-dirty");
    assert_eq!(
        fs::read_to_string(laid.join("shop/pricing.py")).unwrap(),
        PRICING
    );

    // A revision that git could read as an option, or that is not one, is
    // refused; so is a revision that is not a commit of the repository.
    for revision in ["--output=x", "no-such-revision"] {
        assert!(matches!(
            repo.try_capture(&repo.request(revision)),
            Err(TreeError::Revision { .. })
        ));
    }
}

#[test]
fn an_attribute_that_changes_what_git_archive_holds_is_refused() {
    let repo = Repo::new(
        "attributes",
        &[(".gitattributes", "shop/status.py export-ignore\n")],
    );
    assert!(matches!(
        repo.try_capture(&repo.request("HEAD")),
        Err(TreeError::NotTheTree { .. })
    ));
}
