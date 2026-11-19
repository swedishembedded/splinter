// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One HTTP exchange with the Hub, made on a thread of its own with its own
//! runtime, so a caller inside or outside an async runtime calls it alike.
//!
//! The token goes only to the endpoint the client was given: an upload URL
//! the Hub hands out (a signed storage URL) is sent no `Authorization`
//! header. A call that is safe to repeat is repeated, a bounded number of
//! times with a growing pause, on a failure that may pass (no connection, a
//! 429, a 5xx); any other failure is returned at once.

use std::future::Future;
use std::io::Write as _;
use std::path::Path;
use std::time::Duration;

use sha2::{Digest as _, Sha256};

use crate::{Hub, HubError, ATTEMPTS, CONNECT_TIMEOUT, READ_TIMEOUT};

/// The media type of the LFS batch API.
const LFS_JSON: &str = "application/vnd.git-lfs+json";

/// An HTTP method this client uses.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Method {
    Get,
    Post,
    Put,
}

/// One request.
pub(crate) struct Request {
    pub what: String,
    pub method: Method,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Option<Vec<u8>>,
    /// Whether repeating it is harmless.
    pub repeatable: bool,
}

impl Request {
    pub(crate) fn get(what: impl Into<String>, url: String) -> Self {
        Self {
            what: what.into(),
            method: Method::Get,
            url,
            headers: Vec::new(),
            body: None,
            repeatable: true,
        }
    }

    pub(crate) fn post_json(
        what: impl Into<String>,
        url: String,
        body: &serde_json::Value,
    ) -> Self {
        Self {
            what: what.into(),
            method: Method::Post,
            url,
            headers: vec![("content-type".into(), "application/json".into())],
            body: Some(body.to_string().into_bytes()),
            repeatable: false,
        }
    }

    /// A `POST` of `body` to the LFS API, which speaks its own media type.
    pub(crate) fn post_lfs(what: impl Into<String>, url: String, body: &serde_json::Value) -> Self {
        let mut request = Self::post_json(what, url, body);
        request.headers = vec![
            ("content-type".into(), LFS_JSON.into()),
            ("accept".into(), LFS_JSON.into()),
        ];
        request
    }
}

/// A reply, whatever its status.
pub(crate) struct Reply {
    pub status: u16,
    /// Header names in lower case.
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Reply {
    pub(crate) fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_str())
    }

    pub(crate) fn success(&self) -> bool {
        (200..300).contains(&self.status)
    }

    /// The reply if it is a success, else the Hub's refusal with its message.
    pub(crate) fn ok(self, what: &str) -> Result<Self, HubError> {
        if self.success() {
            return Ok(self);
        }
        Err(HubError::Status {
            what: what.to_owned(),
            status: self.status,
            detail: detail_of(&self.body),
        })
    }

    pub(crate) fn json(&self, what: &str) -> Result<serde_json::Value, HubError> {
        serde_json::from_slice(&self.body).map_err(|e| HubError::Malformed {
            what: what.to_owned(),
            reason: format!("not JSON: {e}"),
        })
    }
}

/// The Hub's message in an error body: `{"error": "..."}` or
/// `{"error": {"message": "..."}}`, cut to a readable length.
fn detail_of(body: &[u8]) -> Option<String> {
    let value: serde_json::Value = serde_json::from_slice(body).ok()?;
    let error = value.get("error")?;
    let text = error
        .as_str()
        .or_else(|| error.get("message").and_then(serde_json::Value::as_str))?;
    Some(text.chars().take(400).collect())
}

fn transient(status: u16) -> bool {
    status == 429 || (500..600).contains(&status)
}

fn backoff(attempt: u32) -> Duration {
    Duration::from_millis(500 << attempt.min(6))
}

impl Hub {
    /// Runs `make(client)` to completion on a thread with its own runtime.
    fn block<T, Fut>(&self, make: impl FnOnce(reqwest::Client) -> Fut + Send) -> Result<T, HubError>
    where
        T: Send,
        Fut: Future<Output = Result<T, HubError>>,
    {
        let loopback = self.is_loopback();
        std::thread::scope(|scope| {
            scope
                .spawn(move || {
                    // The provider sven's clients install; a second install
                    // leaves the first in place.
                    let _ = rustls::crypto::ring::default_provider().install_default();
                    let failed = |reason: String| HubError::Transport {
                        what: "starting the client".into(),
                        reason,
                    };
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .map_err(|e| failed(e.to_string()))?;
                    let mut builder = reqwest::Client::builder()
                        .connect_timeout(CONNECT_TIMEOUT)
                        .read_timeout(READ_TIMEOUT);
                    if loopback {
                        builder = builder.no_proxy();
                    }
                    let client = builder.build().map_err(|e| failed(e.to_string()))?;
                    runtime.block_on(make(client))
                })
                .join()
                .unwrap_or_else(|_| {
                    Err(HubError::Transport {
                        what: "a hub call".into(),
                        reason: "the client thread panicked".into(),
                    })
                })
        })
    }

    /// Whether `url` is on this client's endpoint, the only place the token
    /// is sent.
    fn owns(&self, url: &str) -> bool {
        url.strip_prefix(self.endpoint.as_str())
            .is_some_and(|rest| rest.is_empty() || rest.starts_with('/'))
    }

    fn build(&self, client: &reqwest::Client, request: &Request) -> reqwest::RequestBuilder {
        let mut builder = match request.method {
            Method::Get => client.get(&request.url),
            Method::Post => client.post(&request.url),
            Method::Put => client.put(&request.url),
        };
        for (name, value) in &request.headers {
            builder = builder.header(name.as_str(), value.as_str());
        }
        if let (Some(token), true) = (&self.token, self.owns(&request.url)) {
            builder = builder.bearer_auth(token);
        }
        if let Some(body) = &request.body {
            builder = builder.body(body.clone());
        }
        builder
    }

    /// Sends `request`, repeating a repeatable one on a failure that may pass;
    /// returns the reply whatever its status.
    pub(crate) fn send(&self, request: Request) -> Result<Reply, HubError> {
        self.block(|client| async move {
            let mut attempt = 0;
            loop {
                attempt += 1;
                let last = !request.repeatable || attempt >= ATTEMPTS;
                match self.build(&client, &request).send().await {
                    Ok(response) => {
                        let status = response.status().as_u16();
                        if transient(status) && !last {
                            tokio::time::sleep(backoff(attempt)).await;
                            continue;
                        }
                        let headers = response
                            .headers()
                            .iter()
                            .filter_map(|(n, v)| {
                                Some((n.as_str().to_ascii_lowercase(), v.to_str().ok()?.to_owned()))
                            })
                            .collect();
                        let body = response.bytes().await.map_err(|e| HubError::Transport {
                            what: request.what.clone(),
                            reason: e.to_string(),
                        })?;
                        return Ok(Reply {
                            status,
                            headers,
                            body: body.to_vec(),
                        });
                    }
                    Err(e) if !last => {
                        let _ = e;
                        tokio::time::sleep(backoff(attempt)).await;
                    }
                    Err(e) => {
                        return Err(HubError::Transport {
                            what: request.what.clone(),
                            reason: e.to_string(),
                        })
                    }
                }
            }
        })
    }

    /// Streams the body of a `GET` of `url` into `dest`, refusing more than
    /// `limit` bytes; returns the SHA-256 (hex) and the size of what was
    /// written. A failed download leaves no file at `dest`.
    pub(crate) fn fetch_to(
        &self,
        what: &str,
        url: &str,
        dest: &Path,
        limit: u64,
    ) -> Result<(String, u64), HubError> {
        let fetched =
            self.block(|client| async move {
                let request = Request::get(what, url.to_owned());
                let mut response = self.build(&client, &request).send().await.map_err(|e| {
                    HubError::Transport {
                        what: what.to_owned(),
                        reason: e.to_string(),
                    }
                })?;
                let status = response.status().as_u16();
                if !(200..300).contains(&status) {
                    let body = response.bytes().await.unwrap_or_default();
                    return Err(HubError::Status {
                        what: what.to_owned(),
                        status,
                        detail: detail_of(&body),
                    });
                }
                let io = |source| HubError::Io {
                    path: dest.to_path_buf(),
                    source,
                };
                let mut file = std::fs::File::create(dest).map_err(io)?;
                let mut hash = Sha256::new();
                let mut size = 0u64;
                while let Some(chunk) = response.chunk().await.map_err(|e| HubError::Transport {
                    what: what.to_owned(),
                    reason: e.to_string(),
                })? {
                    size += chunk.len() as u64;
                    if size > limit {
                        return Err(HubError::Malformed {
                            what: what.to_owned(),
                            reason: format!("more than the {limit} bytes expected"),
                        });
                    }
                    hash.update(&chunk);
                    file.write_all(&chunk).map_err(io)?;
                }
                file.sync_all().map_err(io)?;
                Ok((hex(&hash.finalize()), size))
            });
        if fetched.is_err() {
            let _ = std::fs::remove_file(dest);
        }
        fetched
    }
}

/// Lower-case hex of `bytes`.
pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
