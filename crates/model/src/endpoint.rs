// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that reach their model
// only through a serving API, so that many agent attempts share one
// continuously batched inference service, for its clients. If your team
// needs expertise in routing agent traffic across inference servers, you
// can procure our services by sending an email to info@swedishembedded.com.

//! The brain endpoints a model is requested from.
//!
//! Splinter holds no model of its own process when it is given endpoints: a
//! local model reference (`local:<checkpoint>[+adapter]`, `policy:<alias>`)
//! names a model a `brain serve --openai` already serves, and every request
//! for it is an HTTP request, so concurrent attempts are concurrent requests
//! that brain's continuous batching decodes together.
//!
//! * A **base** is served under the name brain's model store gives it,
//!   `<org>/<model>` (the last two components of the checkpoint's path).
//! * An **adapter** is a model of its own on the endpoint, found by the digest
//!   of its file: `GET /v1/models` lists each adapter card with the
//!   `{id, digest}` it was verified under, whatever id it is served as. An
//!   adapter the endpoint does not list is refused, unless the endpoint
//!   follows a release manifest (`brain serve --adapter-manifest`) that was
//!   named: the adapter is then offered by publishing that manifest, and
//!   waited for. A manifest offers one adapter at a time.
//! * A reference the endpoint does not list is refused by name, with what it
//!   does list. It is never run in-process instead.
//! * With several endpoints a conversation is **pinned** to one by a stable
//!   hash of its id ([`BrainEndpoints::pin`]), so its prefix stays cached on
//!   one server.
//!
//! An endpoint is an `http://` or `https://` base URL. The key travels in the
//! clear over `http://`, so that is for a loopback or a trusted cluster
//! network; `https://` certificates are always verified.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde_json::Value;
use splinter_core::digest::Digest;

use crate::error::PolicyError;
use crate::local::LocalWeights;
use crate::selection::local_model_name;
use crate::served_train::ArtifactStore;

/// How many attempts are asked of each endpoint at once when the command
/// line names no width: the `--max-batch` plain `brain serve` decodes
/// together when it is not told otherwise.
pub const DEFAULT_ATTEMPTS_PER_ENDPOINT: usize = 16;

/// How long an endpoint may take to answer a model list.
const LIST_TIMEOUT: Duration = Duration::from_secs(15);

/// How long a published adapter is waited for before the endpoint is
/// declared not to follow the manifest: brain verifies the file against the
/// manifest's digest and binds it to the base before it lists the adapter.
const OFFER_WAIT: Duration = Duration::from_secs(120);

/// How often the listing is read while an offered adapter is waited for.
const OFFER_POLL: Duration = Duration::from_millis(250);

/// Trusts the certificate authorities in `pem` for every request to a model
/// endpoint from now on, Splinter's own and the agents' alike: for servers
/// behind an authority of their own (a fleet's `ca.pem`). Certificates are
/// still verified.
///
/// # Errors
/// `pem` holds no certificate.
pub fn trust_authorities(pem: &[u8]) -> Result<usize, String> {
    sven_sdk::http::trust_model_authorities(pem)
}

/// One model an endpoint lists.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServedCard {
    /// The id a request names it by.
    pub id: String,
    /// `sha256:<hex>` of the adapter file, for an adapter's card.
    pub adapter_digest: Option<String>,
}

/// The brain endpoints, their key, and how wide they are asked.
#[derive(Clone)]
pub struct BrainEndpoints {
    urls: Vec<String>,
    key: String,
    adapter_offer: Option<AdapterOffer>,
    attempt_concurrency: Option<usize>,
    artifact_store: Option<ArtifactStore>,
    refused_wait: Duration,
}

impl std::fmt::Debug for BrainEndpoints {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The key never reaches a log or an error.
        f.debug_struct("BrainEndpoints")
            .field("urls", &self.urls)
            .field("adapter_offer", &self.adapter_offer)
            .field("attempt_concurrency", &self.attempt_concurrency)
            .field("artifact_store", &self.artifact_store)
            .finish_non_exhaustive()
    }
}

impl BrainEndpoints {
    /// The endpoints at `urls` (each an OpenAI base URL ending in `/v1`),
    /// asked with `key`.
    pub fn new(urls: Vec<String>, key: impl Into<String>) -> Result<Self, PolicyError> {
        if urls.is_empty() {
            return Err(refused("", "at least one --brain-endpoint is needed"));
        }
        let urls = urls
            .into_iter()
            .map(|url| {
                let url = url.trim_end_matches('/').to_string();
                if url.starts_with("http://") || url.starts_with("https://") {
                    Ok(url)
                } else {
                    Err(refused(
                        &url,
                        "an endpoint is an http:// or https:// base URL",
                    ))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            urls,
            key: key.into(),
            adapter_offer: None,
            attempt_concurrency: None,
            artifact_store: None,
            refused_wait: crate::served_train::REFUSED_WAIT,
        })
    }

    /// The key in `file`: its one word, as `brain serve --api-key-file`
    /// reads it.
    pub fn read_key(file: &Path) -> Result<String, PolicyError> {
        let text = std::fs::read_to_string(file).map_err(|source| PolicyError::Io {
            path: file.to_path_buf(),
            source,
        })?;
        match text.split_whitespace().next() {
            Some(word) => Ok(word.to_string()),
            None => Err(refused(
                &file.display().to_string(),
                "the API key file is empty",
            )),
        }
    }

    /// These endpoints, offering an adapter they do not list by publishing
    /// it to the release manifest at `manifest`, which they follow.
    #[must_use]
    pub fn with_adapter_manifest(mut self, manifest: PathBuf) -> Self {
        self.adapter_offer = Some(AdapterOffer::Manifest(manifest));
        self
    }

    /// These endpoints, offering an adapter they do not list by writing it
    /// into `directory`, which they serve (`brain serve --serve-adapters`):
    /// every adapter written there is a model of its own, so offering one
    /// withdraws none.
    #[must_use]
    pub fn with_adapter_directory(mut self, directory: PathBuf) -> Self {
        self.adapter_offer = Some(AdapterOffer::Directory(directory));
        self
    }

    /// These endpoints, trained through the artifact store `store`: the
    /// directory they were started with (`brain serve --artifact-store`), in
    /// which datasets are published and trained adapters are found, by digest.
    #[must_use]
    pub fn with_artifact_store(mut self, store: ArtifactStore) -> Self {
        self.artifact_store = Some(store);
        self
    }

    /// These endpoints, leaving a training run alone for `wait` after every one
    /// of them refused it for want of room (the default is
    /// [`crate::served_train::REFUSED_WAIT`]).
    #[must_use]
    pub fn with_refused_wait(mut self, wait: Duration) -> Self {
        self.refused_wait = wait;
        self
    }

    /// How long a training run waits after every endpoint refused it.
    #[must_use]
    pub fn refused_wait(&self) -> Duration {
        self.refused_wait
    }

    /// The artifact store training goes through, or the refusal that says to
    /// name one: a training job takes its data by digest, so without a store
    /// there is nothing to name it by.
    pub fn artifact_store(&self) -> Result<&ArtifactStore, PolicyError> {
        self.artifact_store.as_ref().ok_or_else(|| {
            refused(
                self.urls.first().map_or("", String::as_str),
                "training on a brain endpoint needs --brain-artifact-store, the directory \
                 the endpoint was started with as --artifact-store",
            )
        })
    }

    /// These endpoints, asked `width` attempts at once in all.
    #[must_use]
    pub fn with_attempt_concurrency(mut self, width: usize) -> Self {
        self.attempt_concurrency = Some(width.max(1));
        self
    }

    /// Whether an adapter the endpoints do not list is offered by publishing
    /// it to the release manifest they follow, which withdraws the adapter
    /// offered before it: only one is served at a time.
    #[must_use]
    pub fn offers_adapters(&self) -> bool {
        matches!(self.adapter_offer, Some(AdapterOffer::Manifest(_)))
    }

    /// The endpoints' base URLs, without a trailing slash.
    #[must_use]
    pub fn urls(&self) -> &[String] {
        &self.urls
    }

    /// The key requests carry.
    #[must_use]
    pub fn key(&self) -> &str {
        &self.key
    }

    /// How many attempts are asked of the endpoints at once: the width the
    /// command line named, else [`DEFAULT_ATTEMPTS_PER_ENDPOINT`] for each
    /// endpoint. brain does not advertise its `--max-batch`, so a server
    /// started wider is filled by naming the width.
    #[must_use]
    pub fn attempt_concurrency(&self) -> usize {
        self.attempt_concurrency
            .unwrap_or(DEFAULT_ATTEMPTS_PER_ENDPOINT * self.urls.len())
    }

    /// The index of the endpoint the conversation `id` is pinned to: a stable
    /// hash of the id, so it is the same in every run and on every machine.
    #[must_use]
    pub fn pin(&self, id: &[u8]) -> usize {
        let digest = Digest::of(id);
        let hex = digest.hex();
        let head = u64::from_str_radix(&hex[..16], 16).unwrap_or(0);
        // The count of endpoints is tiny; the cast cannot truncate.
        #[allow(clippy::cast_possible_truncation)]
        let index = (head % self.urls.len() as u64) as usize;
        index
    }

    /// What the endpoint at `index` lists.
    pub fn list(&self, index: usize) -> Result<Vec<ServedCard>, PolicyError> {
        let url = &self.urls[index];
        let body = get_json(&format!("{url}/models"), &self.key)
            .map_err(|reason| refused(url, &format!("listing its models: {reason}")))?;
        let data = body["data"]
            .as_array()
            .ok_or_else(|| refused(url, "its model list has no \"data\" array"))?;
        Ok(data
            .iter()
            .filter_map(|card| {
                Some(ServedCard {
                    id: card["id"].as_str()?.to_string(),
                    adapter_digest: card["adapter"]["digest"].as_str().map(str::to_string),
                })
            })
            .collect())
    }

    /// `GET {url}/{route}` of the endpoint at `index`, as JSON.
    ///
    /// # Errors
    /// The endpoint could not be reached, refused, or answered no JSON.
    pub fn get(&self, index: usize, route: &str) -> Result<Value, PolicyError> {
        let url = &self.urls[index];
        get_json(&format!("{url}/{route}"), &self.key)
            .map_err(|reason| refused(url, &format!("GET /{route}: {reason}")))
    }

    /// The id each endpoint serves `weights` under, in the order of
    /// [`Self::urls`]: refused by name when an endpoint does not serve it.
    /// An adapter an endpoint does not list is offered through the manifest
    /// when one was named.
    pub fn locate(&self, weights: &LocalWeights) -> Result<Vec<String>, PolicyError> {
        let wanted = Wanted::of(weights)?;
        let mut offered = false;
        let mut ids = Vec::with_capacity(self.urls.len());
        for index in 0..self.urls.len() {
            let cards = self.list(index)?;
            let mut found = wanted.find(&cards);
            if let (None, Wanted::Adapter { file, digest }, Some(offer)) =
                (&found, &wanted, &self.adapter_offer)
            {
                if !offered {
                    offer.publish(file, digest)?;
                    offered = true;
                }
                found = self.await_listing(index, &wanted, offer)?;
            }
            match found {
                Some(id) => ids.push(id),
                None => {
                    return Err(self.not_served(index, &wanted, &cards));
                }
            }
        }
        Ok(ids)
    }

    /// The id the endpoint at `index` lists `wanted` under once it follows
    /// the manifest, or why it does not.
    fn await_listing(
        &self,
        index: usize,
        wanted: &Wanted,
        offer: &AdapterOffer,
    ) -> Result<Option<String>, PolicyError> {
        let deadline = Instant::now() + OFFER_WAIT;
        loop {
            if let Some(id) = wanted.find(&self.list(index)?) {
                return Ok(Some(id));
            }
            if Instant::now() >= deadline {
                return Err(refused(
                    &self.urls[index],
                    &format!(
                        "did not list {wanted} within {}s of {}: it does not serve from there \
                         or refused the adapter",
                        OFFER_WAIT.as_secs(),
                        offer.describe()
                    ),
                ));
            }
            std::thread::sleep(OFFER_POLL);
        }
    }

    fn not_served(&self, index: usize, wanted: &Wanted, cards: &[ServedCard]) -> PolicyError {
        let how = match wanted {
            Wanted::Base(_) => "start it with --checkpoint naming that model".to_string(),
            Wanted::Adapter { .. } if self.adapter_offer.is_some() => {
                "the adapter was offered but it lists no such adapter".to_string()
            }
            Wanted::Adapter { .. } => "serve it with brain serve --adapter FILE, or name the \
                manifest the endpoint follows with --brain-adapter-manifest FILE, or the \
                directory it serves with --brain-adapter-dir DIR, so it is offered"
                .to_string(),
        };
        PolicyError::NotServed {
            url: self.urls[index].clone(),
            wanted: wanted.to_string(),
            served: cards.iter().map(|card| card.id.clone()).collect(),
            how,
        }
    }
}

fn refused(url: &str, reason: &str) -> PolicyError {
    PolicyError::Endpoint {
        url: url.to_string(),
        reason: reason.to_string(),
    }
}

/// What a request asks an endpoint for.
enum Wanted {
    /// A base, by the name brain's store gives it.
    Base(String),
    /// An adapter, by the digest of its file.
    Adapter { file: PathBuf, digest: Digest },
}

impl Wanted {
    fn of(weights: &LocalWeights) -> Result<Self, PolicyError> {
        let Some(file) = &weights.adapter else {
            return Ok(Self::Base(local_model_name(&weights.base)));
        };
        let io = |source| PolicyError::Io {
            path: file.clone(),
            source,
        };
        let digest =
            Digest::sha256_of_reader(std::fs::File::open(file).map_err(io)?).map_err(io)?;
        Ok(Self::Adapter {
            file: file.clone(),
            digest,
        })
    }

    /// The id of the card that serves it.
    fn find(&self, cards: &[ServedCard]) -> Option<String> {
        cards
            .iter()
            .find(|card| match self {
                Self::Base(name) => card.adapter_digest.is_none() && &card.id == name,
                Self::Adapter { digest, .. } => {
                    card.adapter_digest.as_deref() == Some(digest.as_str())
                }
            })
            .map(|card| card.id.clone())
    }
}

impl std::fmt::Display for Wanted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Base(name) => f.write_str(name),
            Self::Adapter { file, digest } => {
                write!(f, "adapter {} ({digest})", file.display())
            }
        }
    }
}

/// How an adapter an endpoint does not list is put in front of it.
#[derive(Clone, Debug)]
enum AdapterOffer {
    /// Publish a release manifest the endpoint follows (`--adapter-manifest`);
    /// one adapter is served at a time.
    Manifest(PathBuf),
    /// Write the adapter into a directory the endpoint serves
    /// (`--serve-adapters`); each is a model of its own.
    Directory(PathBuf),
}

impl AdapterOffer {
    /// Offers `file` (whose SHA-256 is `digest`).
    fn publish(&self, file: &Path, digest: &Digest) -> Result<(), PolicyError> {
        let io = |path: &Path| {
            let path = path.to_path_buf();
            move |source| PolicyError::Io { path, source }
        };
        match self {
            Self::Manifest(manifest) => {
                let absolute = std::path::absolute(file).map_err(io(file))?;
                crate::manifest::write_adapter_manifest(manifest, &absolute, digest).map_err(
                    |source| PolicyError::Io {
                        path: manifest.clone(),
                        source,
                    },
                )
            }
            Self::Directory(directory) => {
                let served = directory.join(format!("{}.safetensors", digest.hex()));
                if served.exists() {
                    return Ok(());
                }
                // Written aside under a hidden name and renamed into place:
                // the server ignores hidden files and never reads half of one.
                let aside = directory.join(format!(".{}.partial", digest.hex()));
                std::fs::copy(file, &aside).map_err(io(file))?;
                std::fs::rename(&aside, &served).map_err(io(&served))
            }
        }
    }

    fn describe(&self) -> String {
        match self {
            Self::Manifest(manifest) => format!("publishing {}", manifest.display()),
            Self::Directory(directory) => format!("writing it to {}", directory.display()),
        }
    }
}

/// `url`'s JSON body, asked with `key`; the reason on failure. On a thread of
/// its own with its own runtime, so it is safe to call from inside an async
/// runtime. Environment proxies are not consulted: an endpoint is a direct
/// address.
pub(crate) fn get_json(url: &str, key: &str) -> Result<Value, String> {
    request_json(url, key, None, LIST_TIMEOUT)
}

/// `body` posted to `url` as JSON, asked with `key`, and the JSON reply; the
/// reason on failure. Run as [`get_json`] runs.
pub(crate) fn post_json(
    url: &str,
    key: &str,
    body: &Value,
    timeout: Duration,
) -> Result<Value, String> {
    request_json(url, key, Some(body), timeout)
}

/// `body` posted to `url` as JSON, asked with `key`, and the reply as it
/// came: its bytes and the headers it carried. A reply that is not a success
/// is the reason, with brain's message when it gave one. Run as [`get_json`]
/// runs.
pub(crate) fn post_for_bytes(
    url: &str,
    key: &str,
    body: &Value,
    timeout: Duration,
) -> Result<RawReply, String> {
    request(url, key, Some(body), timeout)
}

/// A successful reply's bytes and headers.
pub(crate) struct RawReply {
    /// The body.
    pub body: Vec<u8>,
    /// The headers.
    pub headers: reqwest::header::HeaderMap,
}

fn request_json(
    url: &str,
    key: &str,
    body: Option<&Value>,
    timeout: Duration,
) -> Result<Value, String> {
    let reply = request(url, key, body, timeout)?;
    serde_json::from_slice(&reply.body).map_err(|e| format!("not JSON: {e}"))
}

fn request(
    url: &str,
    key: &str,
    body: Option<&Value>,
    timeout: Duration,
) -> Result<RawReply, String> {
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|e| e.to_string())?;
                runtime.block_on(async {
                    let client = sven_sdk::http::model_client_builder()
                        .no_proxy()
                        .timeout(timeout)
                        .build()
                        .map_err(|e| e.to_string())?;
                    let request = match body {
                        Some(body) => client.post(url).json(body),
                        None => client.get(url),
                    };
                    let response = request
                        .bearer_auth(key)
                        .send()
                        .await
                        .map_err(|e| e.to_string())?;
                    let status = response.status();
                    let headers = response.headers().clone();
                    let bytes = response.bytes().await.map_err(|e| e.to_string())?;
                    if !status.is_success() {
                        let detail = serde_json::from_slice::<Value>(&bytes)
                            .ok()
                            .and_then(|v| v["error"]["message"].as_str().map(str::to_string));
                        return Err(match detail {
                            Some(detail) => format!("answered HTTP {}: {detail}", status.as_u16()),
                            None => format!("answered HTTP {}", status.as_u16()),
                        });
                    }
                    Ok(RawReply {
                        body: bytes.to_vec(),
                        headers,
                    })
                })
            })
            .join()
            .map_err(|_| "the request panicked".to_string())?
    })
}
