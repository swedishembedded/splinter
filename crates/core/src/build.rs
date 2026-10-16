// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible-experiment infrastructure for
// learning systems, for its clients. If your team needs expertise in making
// training and evaluation runs rebuildable from their records, you can
// procure our services by sending an email to info@swedishembedded.com.

//! What a run was made by: the commits of the three repositories, whether the
//! tree was clean, the binary's digest and the GPU contexts it started beside.
//!
//! Every field is a measurement that may be missing, and a missing one is
//! absent, never `0` or an empty string: a record with none of them is
//! exactly what a run recorded before they were kept looks like.

use serde::{Deserialize, Serialize};

use crate::digest::Digest;

/// The commits a binary was built from.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Revisions {
    /// Splinter's own commit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub splinter: Option<String>,
    /// The commit of sven the build linked against.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sven: Option<String>,
    /// The commit of brain the build linked against.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub brain: Option<String>,
}

impl Revisions {
    /// Whether no commit is known.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self == &Self::default()
    }
}

/// What a run was made by. See the module documentation.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuildInfo {
    /// The commits the binary was built from.
    #[serde(default, skip_serializing_if = "Revisions::is_empty")]
    pub revisions: Revisions,
    /// Whether the splinter tree had uncommitted changes at build time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dirty: Option<bool>,
    /// The SHA-256 of the executable that ran.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub binary_digest: Option<Digest>,
    /// How many foreign GPU compute contexts existed when the run started;
    /// absent when nobody counted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_contexts: Option<u32>,
}

impl BuildInfo {
    /// Whether nothing is known.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self == &Self::default()
    }
}

/// The commit `package` is locked to in the text of a `Cargo.lock`, for a
/// package that comes from git; `None` when the lock does not name it or it
/// has no git source. Reads the text it is given: no process is run.
#[must_use]
pub fn locked_revision<'a>(lock: &'a str, package: &str) -> Option<&'a str> {
    let header = format!("name = \"{package}\"");
    let mut lines = lock.lines();
    while let Some(line) = lines.next() {
        if line != header {
            continue;
        }
        for field in lines.by_ref() {
            if field.is_empty() {
                break;
            }
            if let Some(source) = field.strip_prefix("source = \"") {
                return source
                    .trim_end_matches('"')
                    .rsplit_once('#')
                    .map(|(_, rev)| rev);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOCK: &str = "[[package]]\nname = \"brain\"\nversion = \"2.0.0\"\n\
source = \"git+https://example.org/brain#abc123\"\n\n\
[[package]]\nname = \"local\"\nversion = \"1.0.0\"\n\n\
[[package]]\nname = \"sven-sdk\"\nsource = \"git+https://example.org/sven#def456\"\n";

    #[test]
    fn a_git_package_names_its_commit_and_others_name_none() {
        assert_eq!(locked_revision(LOCK, "brain"), Some("abc123"));
        assert_eq!(locked_revision(LOCK, "sven-sdk"), Some("def456"));
        // No source, and not in the lock at all.
        assert_eq!(locked_revision(LOCK, "local"), None);
        assert_eq!(locked_revision(LOCK, "missing"), None);
    }

    #[test]
    fn an_empty_record_serializes_to_nothing() {
        assert_eq!(serde_json::to_string(&BuildInfo::default()).unwrap(), "{}");
    }
}
