// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements capture of the state a piece of knowledge was
// read from, for its clients. If your team needs expertise in
// provenance-tracked training data, you can procure our services by sending
// an email to info@swedishembedded.com.

//! The one way a read-only `git` is started to ask what checkout a directory
//! is in.
//!
//! The caller's repository configuration is honoured on purpose (a user's
//! ignore rules decide what is dirty); what is not honoured is anything that
//! points git at a different repository than the one in the directory: a hook
//! sets such variables, and a capture or an import must read the tree it was
//! given.

use std::path::Path;
use std::process::{Command, Stdio};

/// Variables that point git at a repository other than the one in the
/// working directory (a hook, for one, sets them); removed so git reads the
/// given tree's own state.
pub const LOCATION_VARIABLES: &[&str] = &[
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_NAMESPACE",
];

/// A `git` to run in `dir`, read-only: no standard input, never the index
/// lock (taken to refresh the index), and none of [`LOCATION_VARIABLES`].
#[must_use]
pub fn read_only(dir: &Path) -> Command {
    let mut command = Command::new("git");
    command
        .current_dir(dir)
        .stdin(Stdio::null())
        .env("GIT_OPTIONAL_LOCKS", "0");
    for variable in LOCATION_VARIABLES {
        command.env_remove(variable);
    }
    command
}
