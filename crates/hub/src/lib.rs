// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements distribution of training data and model
// artifacts as immutable, verifiable versions, for its clients. If your team
// needs expertise in publishing datasets and adapters to a model hub with
// provenance intact, you can procure our services by sending an email to
// info@swedishembedded.com.

//! A client of the Hugging Face Hub's HTTP API, for dataset repositories.
//!
//! It speaks the endpoints the Hub's own client uses and nothing else:
//!
//! * `POST /api/repos/create` creates a repository, always private;
//!   `GET /api/datasets/{repo}` reads what the Hub says of it, its `private`
//!   flag first ([`Hub::repo_info`]);
//! * `GET /api/datasets/{repo}/revision/{rev}` resolves a branch, a tag or a
//!   commit to the commit it names ([`Hub::resolve`]); `GET .../refs` lists
//!   branches and tags; `POST .../tag/{commit}` tags a commit;
//! * `GET /api/datasets/{repo}/tree/{commit}?recursive=true` lists the files
//!   of a commit, following the `Link` header's next page;
//! * `GET /datasets/{repo}/resolve/{commit}/{path}` downloads a file, hashed
//!   with SHA-256 as it is written ([`Hub::download`]);
//! * a commit ([`Hub::commit`]): `POST .../preupload/{branch}` says which
//!   files go through LFS, `POST /datasets/{repo}.git/info/lfs/objects/batch`
//!   says where (a single `PUT`, or parts and a completion for a multipart
//!   transfer, then an optional verification), and `POST .../commit/{branch}`
//!   writes the commit as JSON lines, with the parent it was made on, so a
//!   commit made beside another is refused instead of overwriting it.
//!
//! Every call goes to the endpoint the client was given: a test hands it a
//! local emulator and never reaches a public service. The token is sent only
//! in the `Authorization` header and never appears in an error.

#![warn(missing_docs)]

mod api;
mod http;
mod names;
mod upload;

pub use api::{GitRef, Refs, RepoInfo, TreeFile};
pub use names::{CommitId, RepoId, RepoPath};
pub use upload::{Committed, Upload};

use std::path::PathBuf;
use std::time::Duration;

/// The public Hub.
pub const DEFAULT_ENDPOINT: &str = "https://huggingface.co";

/// The longest a call waits for a connection.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// The longest a call waits for the next bytes of a reply.
const READ_TIMEOUT: Duration = Duration::from_secs(300);

/// Attempts of a call that is safe to repeat (a read, an upload of a part to
/// a URL the Hub signed), when it fails in a way that may pass.
const ATTEMPTS: u32 = 3;

/// Why a call to the Hub failed.
#[derive(Debug, thiserror::Error)]
pub enum HubError {
    /// The Hub answered with a status that is not success.
    #[error("the hub answered {what} with HTTP {status}{}", detail.as_deref().map(|d| format!(": {d}")).unwrap_or_default())]
    Status {
        /// What was asked.
        what: String,
        /// The HTTP status.
        status: u16,
        /// The Hub's own message, when it gave one.
        detail: Option<String>,
    },
    /// The Hub could not be reached, or the connection failed.
    #[error("reaching the hub for {what}: {reason}")]
    Transport {
        /// What was asked.
        what: String,
        /// Why it failed.
        reason: String,
    },
    /// The Hub answered with something this client does not read.
    #[error("the hub's answer to {what} is malformed: {reason}")]
    Malformed {
        /// What was asked.
        what: String,
        /// What was wrong with it.
        reason: String,
    },
    /// A local file could not be read or written.
    #[error("{}: {source}", path.display())]
    Io {
        /// The file.
        path: PathBuf,
        /// The failure.
        source: std::io::Error,
    },
    /// A name, a revision or a path is not one this client accepts.
    #[error("{0}")]
    Invalid(String),
    /// A download did not hold what it was expected to.
    #[error("{path} at {commit} is not what was expected: {reason}")]
    Mismatch {
        /// The file in the repository.
        path: String,
        /// The commit it was read at.
        commit: String,
        /// What differed.
        reason: String,
    },
}

/// A client of one Hub endpoint, with the token it authenticates with.
#[derive(Clone)]
pub struct Hub {
    endpoint: String,
    token: Option<String>,
}

impl std::fmt::Debug for Hub {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The token is a secret: it is said to be there, never shown.
        f.debug_struct("Hub")
            .field("endpoint", &self.endpoint)
            .field("token", &self.token.as_ref().map(|_| "<set>"))
            .finish()
    }
}

impl Hub {
    /// A client of `endpoint` (`https://huggingface.co`, or a local
    /// emulator's `http://127.0.0.1:PORT`), sending `token` when given.
    ///
    /// # Errors
    /// The endpoint is not an `http` or `https` URL without a path, or the
    /// token holds characters a header cannot carry.
    pub fn new(endpoint: &str, token: Option<String>) -> Result<Self, HubError> {
        let endpoint = endpoint.trim_end_matches('/');
        let rest = endpoint
            .strip_prefix("https://")
            .or_else(|| endpoint.strip_prefix("http://"))
            .ok_or_else(|| {
                HubError::Invalid(format!("hub endpoint {endpoint:?} is not an http(s) URL"))
            })?;
        if rest.is_empty() || rest.contains('/') || rest.contains('?') || rest.contains('#') {
            return Err(HubError::Invalid(format!(
                "hub endpoint {endpoint:?} must name a host (and port) and no path"
            )));
        }
        if let Some(token) = &token {
            if token.is_empty() || !token.bytes().all(|b| b.is_ascii_graphic()) {
                return Err(HubError::Invalid(
                    "the hub token is empty or holds characters a header cannot carry".into(),
                ));
            }
        }
        Ok(Self {
            endpoint: endpoint.to_owned(),
            token,
        })
    }

    /// The endpoint calls go to.
    #[must_use]
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Whether the endpoint is this machine: a local emulator, reached
    /// without any proxy the process environment names.
    fn is_loopback(&self) -> bool {
        let host = self
            .endpoint
            .split("://")
            .nth(1)
            .unwrap_or_default()
            .rsplit_once(':')
            .map_or_else(
                || self.endpoint.split("://").nth(1).unwrap_or_default(),
                |(host, _)| host,
            );
        matches!(host, "127.0.0.1" | "localhost" | "[::1]")
    }
}
