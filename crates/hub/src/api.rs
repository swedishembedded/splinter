// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The Hub's read calls, and the two small writes: creating a private
//! repository and tagging a commit.

use std::path::Path;

use serde_json::{json, Value};

use crate::http::Request;
use crate::names::encode_segment;
use crate::{CommitId, Hub, HubError, RepoId, RepoPath};

/// What the Hub says of a repository.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepoInfo {
    /// Its `private` flag; `None` when the Hub's answer carries none, which
    /// a caller that needs privacy treats as public.
    pub private: Option<bool>,
}

/// A branch or a tag and the object it points at.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GitRef {
    /// Its name.
    pub name: String,
    /// The object it points at, as the Hub lists it: a branch's commit, but
    /// an annotated tag's own tag object, not the commit it tags (resolve the
    /// tag with [`Hub::resolve`] for that).
    pub target: CommitId,
}

/// A repository's branches and tags.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Refs {
    /// The branches.
    pub branches: Vec<GitRef>,
    /// The tags.
    pub tags: Vec<GitRef>,
}

impl Refs {
    /// The commit the branch `name` points at.
    #[must_use]
    pub fn branch(&self, name: &str) -> Option<&CommitId> {
        self.branches
            .iter()
            .find(|r| r.name == name)
            .map(|r| &r.target)
    }

    /// Whether the tag `name` exists. What it tags is [`Hub::resolve`]'s to
    /// say: the Hub lists an annotated tag by its tag object.
    #[must_use]
    pub fn has_tag(&self, name: &str) -> bool {
        self.tags.iter().any(|r| r.name == name)
    }
}

/// A file of a commit, as the tree listing states it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TreeFile {
    /// Its path.
    pub path: String,
    /// Its size in bytes (the content's, for an LFS file).
    pub size: u64,
    /// For an LFS file, the SHA-256 of its content (hex), which the Hub
    /// stores it under.
    pub lfs_sha256: Option<String>,
}

/// The longest JSON answer read whole: a tree page, a manifest of refs.
const ANSWER_LIMIT: usize = 64 << 20;

/// Tree pages followed before the listing is refused as endless.
const TREE_PAGES: usize = 10_000;

fn field<'a>(value: &'a Value, name: &str, what: &str) -> Result<&'a Value, HubError> {
    value.get(name).ok_or_else(|| HubError::Malformed {
        what: what.to_owned(),
        reason: format!("no {name:?}"),
    })
}

fn text<'a>(value: &'a Value, name: &str, what: &str) -> Result<&'a str, HubError> {
    field(value, name, what)?
        .as_str()
        .ok_or_else(|| HubError::Malformed {
            what: what.to_owned(),
            reason: format!("{name:?} is not a string"),
        })
}

fn commit_of(value: &Value, name: &str, what: &str) -> Result<CommitId, HubError> {
    CommitId::parse(text(value, name, what)?).map_err(|e| HubError::Malformed {
        what: what.to_owned(),
        reason: e.to_string(),
    })
}

impl Hub {
    fn api(&self, repo: &RepoId) -> String {
        format!("{}/api/datasets/{repo}", self.endpoint)
    }

    /// What the Hub says of `repo`; `None` when it has no such repository
    /// (or this token may not see it).
    ///
    /// # Errors
    /// The Hub cannot be reached or answers with another failure.
    pub fn repo_info(&self, repo: &RepoId) -> Result<Option<RepoInfo>, HubError> {
        let what = format!("the info of {repo}");
        let reply = self.send(Request::get(&what, self.api(repo)))?;
        if reply.status == 404 {
            return Ok(None);
        }
        let value = reply.ok(&what)?.json(&what)?;
        Ok(Some(RepoInfo {
            private: value.get("private").and_then(Value::as_bool),
        }))
    }

    /// Creates `repo` as a private dataset repository. A repository that
    /// already exists is left as it is: a caller checks its flag with
    /// [`Hub::repo_info`] before writing to it.
    ///
    /// # Errors
    /// The Hub refuses the creation for another reason than that it exists.
    pub fn create_private_repo(&self, repo: &RepoId) -> Result<(), HubError> {
        let what = format!("creating {repo}");
        // Both spellings of privacy: the Hub's older `private` and its newer
        // `visibility`.
        let body = json!({
            "name": repo.name(),
            "organization": repo.owner(),
            "type": "dataset",
            "private": true,
            "visibility": "private",
        });
        let reply = self.send(Request::post_json(
            &what,
            format!("{}/api/repos/create", self.endpoint),
            &body,
        ))?;
        if reply.status == 409 {
            return Ok(());
        }
        reply.ok(&what).map(|_| ())
    }

    /// The commit `revision` (a branch, a tag or a commit) names in `repo`.
    ///
    /// # Errors
    /// The revision is unknown, or the Hub's answer names no commit.
    pub fn resolve(&self, repo: &RepoId, revision: &str) -> Result<CommitId, HubError> {
        let what = format!("revision {revision:?} of {repo}");
        let url = format!("{}/revision/{}", self.api(repo), encode_segment(revision));
        let value = self
            .send(Request::get(&what, url))?
            .ok(&what)?
            .json(&what)?;
        commit_of(&value, "sha", &what)
    }

    /// The branches and tags of `repo`.
    ///
    /// # Errors
    /// The Hub cannot be reached or its answer is malformed.
    pub fn refs(&self, repo: &RepoId) -> Result<Refs, HubError> {
        let what = format!("the refs of {repo}");
        let url = format!("{}/refs", self.api(repo));
        let value = self
            .send(Request::get(&what, url))?
            .ok(&what)?
            .json(&what)?;
        let list = |name: &str| -> Result<Vec<GitRef>, HubError> {
            let items =
                field(&value, name, &what)?
                    .as_array()
                    .ok_or_else(|| HubError::Malformed {
                        what: what.clone(),
                        reason: format!("{name:?} is not a list"),
                    })?;
            items
                .iter()
                .map(|item| {
                    Ok(GitRef {
                        name: text(item, "name", &what)?.to_owned(),
                        target: commit_of(item, "targetCommit", &what)?,
                    })
                })
                .collect()
        };
        Ok(Refs {
            branches: list("branches")?,
            tags: list("tags")?,
        })
    }

    /// Tags `commit` of `repo` as `tag`. A tag that already points at that
    /// commit is left; one that points elsewhere is refused.
    ///
    /// # Errors
    /// The tag exists on another commit, or the Hub refuses it.
    pub fn create_tag(
        &self,
        repo: &RepoId,
        commit: &CommitId,
        tag: &str,
        message: &str,
    ) -> Result<(), HubError> {
        let what = format!("tagging {commit} of {repo} as {tag}");
        let url = format!("{}/tag/{commit}", self.api(repo));
        let reply = self.send(Request::post_json(
            &what,
            url,
            &json!({ "tag": tag, "message": message }),
        ))?;
        if reply.status == 409 {
            let tagged = self.resolve(repo, tag)?;
            return if &tagged == commit {
                Ok(())
            } else {
                Err(HubError::Invalid(format!(
                    "tag {tag} of {repo} already tags {tagged}"
                )))
            };
        }
        reply.ok(&what).map(|_| ())
    }

    /// Every file of `commit` of `repo`, in the order listed.
    ///
    /// # Errors
    /// The Hub cannot be reached, or a page is malformed.
    pub fn tree(&self, repo: &RepoId, commit: &CommitId) -> Result<Vec<TreeFile>, HubError> {
        let what = format!("the files of {repo} at {commit}");
        let mut url = Some(format!("{}/tree/{commit}?recursive=true", self.api(repo)));
        let mut files = Vec::new();
        let mut pages = 0;
        while let Some(next) = url.take() {
            pages += 1;
            if pages > TREE_PAGES {
                return Err(HubError::Malformed {
                    what,
                    reason: "the listing does not end".into(),
                });
            }
            let reply = self.send(Request::get(&what, next))?.ok(&what)?;
            if reply.body.len() > ANSWER_LIMIT {
                return Err(HubError::Malformed {
                    what,
                    reason: "a page is too large".into(),
                });
            }
            let value = reply.json(&what)?;
            let entries = value.as_array().ok_or_else(|| HubError::Malformed {
                what: what.clone(),
                reason: "not a list".into(),
            })?;
            for entry in entries {
                if text(entry, "type", &what)? != "file" {
                    continue;
                }
                let size =
                    field(entry, "size", &what)?
                        .as_u64()
                        .ok_or_else(|| HubError::Malformed {
                            what: what.clone(),
                            reason: "a size is not a count".into(),
                        })?;
                let lfs_sha256 = match entry.get("lfs") {
                    Some(lfs) if !lfs.is_null() => Some(text(lfs, "oid", &what)?.to_owned()),
                    _ => None,
                };
                files.push(TreeFile {
                    path: text(entry, "path", &what)?.to_owned(),
                    size,
                    lfs_sha256,
                });
            }
            url = reply.header("link").and_then(next_link).map(|link| {
                if link.starts_with('/') {
                    format!("{}{link}", self.endpoint)
                } else {
                    link
                }
            });
        }
        Ok(files)
    }

    /// Downloads `path` of `commit` of `repo` into `dest`, refusing more
    /// than `limit` bytes; returns its SHA-256 (hex) and size.
    ///
    /// # Errors
    /// The file is not there, is larger than `limit`, or cannot be written.
    pub fn download(
        &self,
        repo: &RepoId,
        commit: &CommitId,
        path: &RepoPath,
        dest: &Path,
        limit: u64,
    ) -> Result<(String, u64), HubError> {
        let what = format!("{path} of {repo} at {commit}");
        let url = format!("{}/datasets/{repo}/resolve/{commit}/{path}", self.endpoint);
        self.fetch_to(&what, &url, dest, limit)
    }
}

/// The `rel="next"` URL of a `Link` header.
fn next_link(header: &str) -> Option<String> {
    header.split(',').find_map(|part| {
        let (target, params) = part.split_once(';')?;
        params
            .split(';')
            .any(|p| p.trim().replace(' ', "") == "rel=\"next\"")
            .then(|| {
                target
                    .trim()
                    .trim_start_matches('<')
                    .trim_end_matches('>')
                    .to_owned()
            })
    })
}

#[cfg(test)]
mod tests {
    use super::next_link;

    #[test]
    fn the_next_page_is_read_from_the_link_header() {
        assert_eq!(
            next_link("<https://h/api/x?cursor=2>; rel=\"next\""),
            Some("https://h/api/x?cursor=2".into())
        );
        assert_eq!(next_link("<https://h/a>; rel=\"prev\""), None);
    }
}
