// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Proof that an arm was served by the weights it claims.
//!
//! An adapter is offered by `brain serve` as a model of its own, listed in
//! `GET /v1/models` with the digest of the file it loaded. A request for a
//! model the server does not offer is an error, but a request for the base
//! answers from the base without complaint - so an arm that was meant to be
//! the trained model and named the wrong id reports the base model's score
//! as a null result. [`require_adapter`] refuses to start such an arm: the
//! served listing must carry the model id with exactly the digest of the
//! adapter that was trained.
//!
//! Swedish Embedded AB implements closed-loop learning systems - agents that
//! improve from their own verified experience rather than from hand-written
//! training data - for its clients. If your team needs expertise in agent
//! training loops, sample design, or promotion gating for small models,
//! you can procure our services by sending an email to
//! info@swedishembedded.com.

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use crate::upstream_of;

/// A model a server offers, as `/v1/models` lists it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ListedModel {
    /// The id a request names.
    pub id: String,
    /// The digest of the adapter file behind it (`adapter.digest` in the
    /// listing); `None` for a base model.
    pub digest: Option<String>,
}

/// Why an arm cannot claim the adapter it names.
#[derive(Debug, PartialEq, Eq)]
pub enum AdapterProofError {
    /// The server does not offer the model id.
    NotOffered {
        /// The id asked for.
        id: String,
    },
    /// The server offers the id, but not as an adapter (it is a base model).
    NotAnAdapter {
        /// The id asked for.
        id: String,
    },
    /// The server offers the id with another adapter file's digest.
    DigestMismatch {
        /// The id asked for.
        id: String,
        /// The digest the arm trained.
        expected: String,
        /// The digest the server lists.
        served: String,
    },
}

impl std::fmt::Display for AdapterProofError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AdapterProofError::NotOffered { id } => {
                write!(f, "the server does not offer a model {id:?}")
            }
            AdapterProofError::NotAnAdapter { id } => write!(
                f,
                "the server offers {id:?} without an adapter digest: it is a base model, and an \
                 arm measured on it says nothing about training"
            ),
            AdapterProofError::DigestMismatch {
                id,
                expected,
                served,
            } => write!(
                f,
                "the server offers {id:?} from adapter {served}, not the trained {expected}"
            ),
        }
    }
}

impl std::error::Error for AdapterProofError {}

/// Whether `listed` proves that `id` is served from the adapter with
/// `expected_digest`.
pub fn require_adapter(
    listed: &[ListedModel],
    id: &str,
    expected_digest: &str,
) -> Result<(), AdapterProofError> {
    let model = listed
        .iter()
        .find(|m| m.id == id)
        .ok_or_else(|| AdapterProofError::NotOffered { id: id.to_string() })?;
    match &model.digest {
        None => Err(AdapterProofError::NotAnAdapter { id: id.to_string() }),
        Some(served) if served != expected_digest => Err(AdapterProofError::DigestMismatch {
            id: id.to_string(),
            expected: expected_digest.to_string(),
            served: served.clone(),
        }),
        Some(_) => Ok(()),
    }
}

/// The models the server at `base_url` lists, read over a plain HTTP GET.
pub async fn listed_models(
    base_url: &str,
    api_key: Option<&str>,
) -> anyhow::Result<Vec<ListedModel>> {
    let authority = upstream_of(base_url)
        .ok_or_else(|| anyhow::anyhow!("{base_url:?} is not an http(s) URL"))?;
    let path = format!(
        "{}/models",
        base_url
            .split_once("://")
            .and_then(|(_, rest)| rest.split_once('/'))
            .map_or(String::new(), |(_, path)| format!("/{path}"))
            .trim_end_matches('/')
    );
    let mut request = format!("GET {path} HTTP/1.1\r\nHost: {authority}\r\nConnection: close\r\n");
    if let Some(key) = api_key {
        request.push_str(&format!("Authorization: Bearer {key}\r\n"));
    }
    request.push_str("\r\n");

    let mut stream = TcpStream::connect(&authority).await?;
    stream.write_all(request.as_bytes()).await?;
    let mut response = Vec::new();
    stream.read_to_end(&mut response).await?;
    let text = String::from_utf8_lossy(&response);
    let (head, body) = text
        .split_once("\r\n\r\n")
        .ok_or_else(|| anyhow::anyhow!("{authority}{path}: no HTTP response"))?;
    if !head.starts_with("HTTP/1.1 200") && !head.starts_with("HTTP/1.0 200") {
        anyhow::bail!(
            "{authority}{path}: {}",
            head.lines().next().unwrap_or("no status")
        );
    }
    parse_models(body)
}

fn parse_models(body: &str) -> anyhow::Result<Vec<ListedModel>> {
    let json: serde_json::Value = serde_json::from_str(body)?;
    let data = json
        .get("data")
        .and_then(|d| d.as_array())
        .ok_or_else(|| anyhow::anyhow!("the model listing has no `data` array"))?;
    Ok(data
        .iter()
        .filter_map(|m| {
            Some(ListedModel {
                id: m.get("id")?.as_str()?.to_string(),
                // brain lists an adapter's file digest under `adapter`.
                digest: m
                    .pointer("/adapter/digest")
                    .and_then(|d| d.as_str())
                    .map(String::from),
            })
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(id: &str, digest: Option<&str>) -> ListedModel {
        ListedModel {
            id: id.into(),
            digest: digest.map(String::from),
        }
    }

    #[test]
    fn an_arm_is_the_adapter_only_when_the_server_lists_its_digest() {
        let listed = [
            model("brain/qwen3", None),
            model("brain/qwen3:me:tool:v1", Some("sha256:aa")),
        ];
        assert_eq!(
            require_adapter(&listed, "brain/qwen3:me:tool:v1", "sha256:aa"),
            Ok(())
        );
        assert!(matches!(
            require_adapter(&listed, "brain/qwen3", "sha256:aa"),
            Err(AdapterProofError::NotAnAdapter { .. })
        ));
        assert!(matches!(
            require_adapter(&listed, "brain/qwen3:me:tool:v1", "sha256:bb"),
            Err(AdapterProofError::DigestMismatch { .. })
        ));
        assert!(matches!(
            require_adapter(&listed, "brain/other", "sha256:aa"),
            Err(AdapterProofError::NotOffered { .. })
        ));
    }

    #[test]
    fn the_listing_is_read_with_and_without_digests() {
        let body = r#"{"data":[{"id":"a"},{"id":"b","adapter":{"id":"b","digest":"sha256:01"}}]}"#;
        assert_eq!(
            parse_models(body).expect("parses"),
            [model("a", None), model("b", Some("sha256:01"))]
        );
    }
}
