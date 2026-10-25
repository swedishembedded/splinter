// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that turn the documentation of an
// unfamiliar technical domain into a graph of skills they then acquire, for
// its clients. If your team needs expertise in building curricula from
// technical documentation you can procure our services by sending an email
// to info@swedishembedded.com.

//! The sources of an acquisition: files and directories as `learn` takes
//! them, and web pages.
//!
//! A URL the person names is fetched once, by `curl` under the bounded
//! process runner (a timeout, an output cap, an explicit environment), into a
//! file named by the digest of its bytes. A page of HTML is converted to the
//! Markdown the sectioner reads ([`html_to_markdown`]); anything else is
//! kept as it came. The file is then captured like any document, so the
//! source's parts are stored once per digest, and a [`Fetched`] record ties
//! the URL, the digest of the bytes that came and the source made from them
//! together. Naming a URL is the consent to fetch that URL and no other.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use splinter_core::digest::Digest;
use splinter_core::source::SourceId;
use splinter_knowledge::capture::{decode_html, default_environment, html_to_markdown};
use splinter_sandbox::process::{run, ProcessSpec};

use crate::sources::{self, SourceTarget};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::{io, OrchestratorError};

/// The longest a download may take.
pub const FETCH_TIMEOUT: Duration = Duration::from_secs(120);

/// The most bytes a downloaded page may have.
pub const FETCH_MAX_BYTES: u64 = 8 * 1024 * 1024;

const FETCHED: &str = "source_fetch";

/// What one fetch brought in.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fetched {
    /// The address asked for: a URL, or `file://` and the canonical path of
    /// a local HTML page.
    pub url: String,
    /// The digest of the bytes that came.
    pub sha256: Digest,
    /// Their number.
    pub bytes: u64,
    /// Whether they were HTML, converted to Markdown.
    pub converted: bool,
    /// When, from the injected clock.
    pub fetched_at: String,
}

/// One source of an acquisition.
#[derive(Clone, Debug, Serialize)]
pub struct Intaken {
    /// What was named.
    pub named: String,
    /// The source it is in the store.
    pub source: SourceId,
    /// How it was fetched and converted, for a URL or a local HTML page.
    pub fetched: Option<Fetched>,
}

/// Whether `arg` is a web address.
#[must_use]
pub fn is_url(arg: &str) -> bool {
    let lower = arg.to_ascii_lowercase();
    lower.starts_with("http://") || lower.starts_with("https://")
}

/// Takes in each of `named`: a URL is fetched, anything else is what
/// [`SourceTarget::from_learn_arg`] reads.
pub fn intake(ctx: &Context, named: &[String]) -> Result<Vec<Intaken>, OrchestratorError> {
    named
        .iter()
        .map(|arg| {
            if is_url(arg) {
                fetch(ctx, arg)
            } else if let Some(page) = local_html(arg)? {
                take_in(
                    ctx,
                    arg,
                    &format!("file://{}", page.display()),
                    std::fs::read(&page).map_err(io(&page))?,
                )
            } else {
                let target = SourceTarget::from_learn_arg(arg)?;
                Ok(Intaken {
                    named: arg.clone(),
                    source: sources::add(ctx, &target)?.source.id,
                    fetched: None,
                })
            }
        })
        .collect()
}

/// Whether `bytes` are an HTML page.
fn is_html(bytes: &[u8]) -> bool {
    let head = String::from_utf8_lossy(&bytes[..bytes.len().min(2048)]).to_ascii_lowercase();
    head.contains("<html") || head.contains("<!doctype html")
}

fn fetch(ctx: &Context, url: &str) -> Result<Intaken, OrchestratorError> {
    let scratch = ctx.root().work().join("fetch");
    std::fs::create_dir_all(&scratch).map_err(io(&scratch))?;
    let download = tempfile::Builder::new()
        .prefix("download-")
        .tempfile_in(&scratch)
        .map_err(io(&scratch))?;
    let argv = [
        "curl",
        "--fail",
        "--silent",
        "--show-error",
        "--location",
        "--proto",
        "=http,https",
        "--max-time",
        &FETCH_TIMEOUT.as_secs().to_string(),
        "--max-filesize",
        &FETCH_MAX_BYTES.to_string(),
        "--output",
        &download.path().display().to_string(),
        "--",
        url,
    ]
    .map(str::to_string)
    .to_vec();
    let output = run(&ProcessSpec::new(
        argv,
        &scratch,
        default_environment(ctx.config().command_env.clone()),
        FETCH_TIMEOUT + Duration::from_secs(10),
        64 * 1024,
    ))
    .map_err(|e| OrchestratorError::Refused(format!("fetching {url}: {e}")))?;
    if output.exit_code != Some(0) {
        return Err(OrchestratorError::Refused(format!(
            "fetching {url} failed (exit {:?}): {}",
            output.exit_code,
            String::from_utf8_lossy(&output.stderr.bytes).trim()
        )));
    }
    let bytes = std::fs::read(download.path()).map_err(io(download.path()))?;
    take_in(ctx, url, url, bytes)
}

/// The canonical path of `arg` when it names a local HTML page, which is
/// taken in as a fetched page is (decoded, converted to Markdown) so that a
/// manual reads the same whether it is named by URL or by file.
fn local_html(arg: &str) -> Result<Option<std::path::PathBuf>, OrchestratorError> {
    let path = std::path::Path::new(arg);
    if arg.starts_with(sources::COMMAND_PREFIX) || !path.is_file() {
        return Ok(None);
    }
    let mut head = Vec::with_capacity(2048);
    std::io::Read::read_to_end(
        &mut std::io::Read::take(std::fs::File::open(path).map_err(io(path))?, 2048),
        &mut head,
    )
    .map_err(io(path))?;
    if !is_html(&head) {
        return Ok(None);
    }
    path.canonicalize().map(Some).map_err(io(path))
}

/// Stores the `bytes` that came from `origin` (named `named`) as a source:
/// an HTML page decoded as it declares and converted to Markdown, anything
/// else kept as the UTF-8 text it must be; the [`Fetched`] record ties the
/// origin, the digest of its bytes and the source together.
fn take_in(
    ctx: &Context,
    named: &str,
    origin: &str,
    bytes: Vec<u8>,
) -> Result<Intaken, OrchestratorError> {
    let scratch = ctx.root().work().join("fetch");
    std::fs::create_dir_all(&scratch).map_err(io(&scratch))?;
    let sha256 = Digest::sha256_of(&bytes);
    let converted = is_html(&bytes);
    let (body, extension) = if converted {
        let text = decode_html(&bytes)
            .map_err(|why| OrchestratorError::Refused(format!("{origin}: {why}")))?;
        (html_to_markdown(&text), "md")
    } else {
        let text = String::from_utf8(bytes.clone()).map_err(|_| {
            OrchestratorError::Refused(format!(
                "{origin} is not UTF-8 text: Splinter learns from text"
            ))
        })?;
        (text, "txt")
    };
    let file = scratch.join(format!("{}.{extension}", &sha256.hex()[..32]));
    std::fs::write(&file, body).map_err(io(&file))?;
    let source = sources::add(ctx, &SourceTarget::Path { path: file })?
        .source
        .id;
    let fetched = Fetched {
        url: origin.to_string(),
        sha256,
        bytes: bytes.len() as u64,
        converted,
        fetched_at: ctx.clock().utc_now(),
    };
    ctx.workspace().put_document(FETCHED, &fetched)?;
    Ok(Intaken {
        named: named.to_string(),
        source,
        fetched: Some(fetched),
    })
}
