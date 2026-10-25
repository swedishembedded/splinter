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

use std::sync::OnceLock;

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
    /// The build of the running process, from what the binary embedded at
    /// build time: the `lock` it was built with (which names sven and brain),
    /// splinter's own commit and the dirty flag as the build script printed
    /// them (`"true"`/`"false"`), and the SHA-256 of the executable. Whatever
    /// cannot be established stays absent.
    #[must_use]
    pub fn embedded(
        lock: &str,
        splinter: Option<&str>,
        dirty: Option<&str>,
        gpu_contexts: Option<u32>,
    ) -> Self {
        Self {
            revisions: Revisions {
                splinter: splinter.map(str::to_owned),
                sven: locked_revision(lock, "sven-sdk").map(str::to_owned),
                brain: locked_revision(lock, "brain").map(str::to_owned),
            },
            dirty: dirty.and_then(|v| v.parse().ok()),
            binary_digest: running_binary_digest(),
            gpu_contexts,
        }
    }

    /// Whether nothing is known.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self == &Self::default()
    }
}

/// The executable that trained a model: the file, its SHA-256, and the build
/// identity the binary knows (its version and the commits it was built from).
/// Every field is a measurement that may be missing: a binary whose file
/// cannot be read from its own process records no path or digest, and a
/// record written before the identity was kept reads as the default.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BinaryIdentity {
    /// The executable's path, as the process named it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    /// The SHA-256 of the executable file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub digest: Option<Digest>,
    /// The crate version the binary was built as.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// The commits it was built from.
    #[serde(default, skip_serializing_if = "Revisions::is_empty")]
    pub revisions: Revisions,
    /// Whether the splinter tree was dirty when it was built.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dirty: Option<bool>,
}

impl BinaryIdentity {
    /// The identity of the running executable, taking the commits and the
    /// dirty flag from `build` (what the process embedded) and measuring the
    /// file once for the digest.
    #[must_use]
    pub fn of(build: &BuildInfo) -> Self {
        Self {
            path: std::env::current_exe()
                .ok()
                .map(|exe| exe.display().to_string()),
            digest: running_binary_digest(),
            version: Some(env!("CARGO_PKG_VERSION").to_string()),
            revisions: build.revisions.clone(),
            dirty: build.dirty,
        }
    }

    /// Whether nothing is known.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.path.is_none()
            && self.digest.is_none()
            && self.version.is_none()
            && self.revisions.is_empty()
            && self.dirty.is_none()
    }
}

/// The SHA-256 of the running executable, measured once for the process: the
/// file is read and hashed on the first call, and every later identity reuses
/// it. `None` when the executable cannot be opened (a sandbox that does not
/// expose its own file), never a zero.
#[must_use]
pub fn running_binary_digest() -> Option<Digest> {
    static DIGEST: OnceLock<Option<Digest>> = OnceLock::new();
    DIGEST
        .get_or_init(|| {
            let exe = std::env::current_exe().ok()?;
            let file = std::fs::File::open(exe).ok()?;
            Digest::sha256_of_reader(std::io::BufReader::new(file)).ok()
        })
        .clone()
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
    fn an_embedded_build_names_the_locked_commits_the_flag_and_the_executable() {
        let build = BuildInfo::embedded(LOCK, Some("1234abc"), Some("false"), Some(2));
        assert_eq!(build.revisions.splinter.as_deref(), Some("1234abc"));
        assert_eq!(build.revisions.sven.as_deref(), Some("def456"));
        assert_eq!(build.revisions.brain.as_deref(), Some("abc123"));
        assert_eq!(build.dirty, Some(false));
        assert!(build.binary_digest.is_some());
        assert_eq!(build.gpu_contexts, Some(2));
        let unknown = BuildInfo::embedded("", None, Some("maybe"), None);
        assert_eq!(unknown.revisions, Revisions::default());
        assert_eq!(unknown.dirty, None);
    }

    #[test]
    fn an_empty_record_serializes_to_nothing() {
        assert_eq!(serde_json::to_string(&BuildInfo::default()).unwrap(), "{}");
    }

    #[test]
    fn the_identity_of_the_running_binary_names_its_file_and_its_digest() {
        let exe = std::env::current_exe().unwrap();
        let expected = {
            let file = std::fs::File::open(&exe).unwrap();
            Digest::sha256_of_reader(std::io::BufReader::new(file)).unwrap()
        };
        let identity =
            BinaryIdentity::of(&BuildInfo::embedded(LOCK, Some("abc"), Some("true"), None));
        assert_eq!(
            identity.path.as_deref(),
            Some(exe.display().to_string().as_str())
        );
        assert_eq!(identity.digest, Some(expected.clone()));
        assert_eq!(identity.version.as_deref(), Some(env!("CARGO_PKG_VERSION")));
        assert_eq!(identity.revisions.splinter.as_deref(), Some("abc"));
        assert_eq!(identity.dirty, Some(true));
        assert_eq!(running_binary_digest(), Some(expected));
    }

    #[test]
    fn an_identity_that_was_not_kept_serializes_to_nothing() {
        assert_eq!(
            serde_json::to_string(&BinaryIdentity::default()).unwrap(),
            "{}"
        );
        assert_eq!(
            serde_json::from_str::<BuildInfo>("{}").unwrap(),
            BuildInfo::default()
        );
    }
}
