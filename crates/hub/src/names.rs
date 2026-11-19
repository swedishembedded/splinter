// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The names the Hub's API takes, checked before they reach a URL: a
//! repository, a commit, a path inside a repository.

use std::fmt;

use crate::HubError;

/// A repository: `owner/name`, each part of letters, digits, `-`, `_` and
/// `.`, neither starting nor ending with `-` or `.`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RepoId {
    owner: String,
    name: String,
}

/// The longest part of a repository's id the Hub accepts.
const NAME_LIMIT: usize = 96;

fn valid_part(part: &str) -> bool {
    !part.is_empty()
        && part.len() <= NAME_LIMIT
        && part
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
        && !part.starts_with(['-', '.'])
        && !part.ends_with(['-', '.'])
        && !part.contains("..")
}

impl RepoId {
    /// `text` read as `owner/name`.
    ///
    /// # Errors
    /// It is not two valid parts separated by one `/`.
    pub fn parse(text: &str) -> Result<Self, HubError> {
        match text.split_once('/') {
            Some((owner, name)) if valid_part(owner) && valid_part(name) => Ok(Self {
                owner: owner.to_owned(),
                name: name.to_owned(),
            }),
            _ => Err(HubError::Invalid(format!(
                "{text:?} is not a repository id of the form owner/name"
            ))),
        }
    }

    /// The owner: a user or an organisation.
    #[must_use]
    pub fn owner(&self) -> &str {
        &self.owner
    }

    /// The repository's name under its owner.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl fmt::Display for RepoId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.owner, self.name)
    }
}

/// A commit: forty lower-case hex digits, the only revision that cannot
/// move.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
#[serde(transparent)]
pub struct CommitId(String);

impl CommitId {
    /// `text` read as a commit id.
    ///
    /// # Errors
    /// It is not forty lower-case hex digits.
    pub fn parse(text: &str) -> Result<Self, HubError> {
        if Self::is_commit(text) {
            Ok(Self(text.to_owned()))
        } else {
            Err(HubError::Invalid(format!("{text:?} is not a commit id")))
        }
    }

    /// Whether `text` names a commit rather than a branch or a tag.
    #[must_use]
    pub fn is_commit(text: &str) -> bool {
        text.len() == 40 && text.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
    }

    /// The id.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for CommitId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A file's path inside a repository: `/`-separated parts of letters,
/// digits, `-`, `_` and `.`, none of them empty, `.` or `..`, and none
/// starting with `.` except the repository's own `.gitattributes`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
#[serde(transparent)]
pub struct RepoPath(String);

/// The longest path accepted.
const PATH_LIMIT: usize = 512;

impl RepoPath {
    /// `text` read as a path inside a repository.
    ///
    /// # Errors
    /// It is not one: see the type's documentation.
    pub fn parse(text: &str) -> Result<Self, HubError> {
        let part_ok = |part: &str| {
            !part.is_empty()
                && !part.starts_with('.')
                && part
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
        };
        if text == ".gitattributes" || (text.len() <= PATH_LIMIT && text.split('/').all(part_ok)) {
            Ok(Self(text.to_owned()))
        } else {
            Err(HubError::Invalid(format!(
                "{text:?} is not a path this client reads or writes in a repository"
            )))
        }
    }

    /// The path.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for RepoPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// `text` percent-encoded for one segment of a URL path: everything but
/// the unreserved characters.
pub(crate) fn encode_segment(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for b in text.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(char::from(b));
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn names_that_could_leave_their_place_in_a_url_are_refused() {
        assert!(RepoId::parse("owner/name").is_ok());
        for bad in ["owner", "a/b/c", "../x", "o/..", "o/-x", "o/x y", "/x"] {
            assert!(RepoId::parse(bad).is_err(), "{bad}");
        }
        assert!(RepoPath::parse("datasets/a.jsonl").is_ok());
        for bad in ["../x", "a//b", "/a", "a/./b", "a/.hidden", "a\\b", ""] {
            assert!(RepoPath::parse(bad).is_err(), "{bad}");
        }
        assert!(CommitId::parse(&"a".repeat(40)).is_ok());
        assert!(CommitId::parse("main").is_err());
        assert_eq!(encode_segment("refs/tags/x y"), "refs%2Ftags%2Fx%20y");
    }
}
