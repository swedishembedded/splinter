// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A commit of files to a branch: the Hub says which go through LFS, those
//! are uploaded (or found already stored under their SHA-256), and one
//! commit names them all, made on the parent the caller read so that a
//! commit made in between is refused rather than overwritten.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use base64::Engine as _;
use serde_json::{json, Value};
use sha2::{Digest as _, Sha256};

use crate::http::{hex, Method, Request};
use crate::names::encode_segment;
use crate::{CommitId, Hub, HubError, RepoId, RepoPath};

/// A local file to add to the repository at `path`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Upload {
    /// Where it goes in the repository.
    pub path: RepoPath,
    /// The local file.
    pub local: PathBuf,
}

/// What a commit did.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Committed {
    /// The commit made.
    pub commit: CommitId,
    /// Files sent in the commit itself.
    pub regular: usize,
    /// LFS files whose content was uploaded.
    pub lfs_uploaded: usize,
    /// LFS files the Hub already stored under their SHA-256.
    pub lfs_present: usize,
}

/// Files asked about in one pre-upload call, as the Hub's own client does.
const PREUPLOAD_CHUNK: usize = 256;

/// The bytes of a file the pre-upload call is shown.
const SAMPLE_BYTES: usize = 512;

struct Prepared<'a> {
    upload: &'a Upload,
    size: u64,
    sha256: String,
    sample: Vec<u8>,
    lfs: bool,
}

fn io(path: &Path) -> impl FnOnce(std::io::Error) -> HubError + '_ {
    move |source| HubError::Io {
        path: path.to_path_buf(),
        source,
    }
}

fn prepare(upload: &Upload) -> Result<Prepared<'_>, HubError> {
    let mut file = File::open(&upload.local).map_err(io(&upload.local))?;
    let mut hash = Sha256::new();
    let mut sample = Vec::with_capacity(SAMPLE_BYTES);
    let mut size = 0u64;
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buffer).map_err(io(&upload.local))?;
        if n == 0 {
            break;
        }
        if sample.len() < SAMPLE_BYTES {
            let take = (SAMPLE_BYTES - sample.len()).min(n);
            sample.extend_from_slice(&buffer[..take]);
        }
        hash.update(&buffer[..n]);
        size += n as u64;
    }
    Ok(Prepared {
        upload,
        size,
        sha256: hex(&hash.finalize()),
        sample,
        lfs: false,
    })
}

fn read_range(path: &Path, from: u64, len: u64) -> Result<Vec<u8>, HubError> {
    let mut file = File::open(path).map_err(io(path))?;
    file.seek(SeekFrom::Start(from)).map_err(io(path))?;
    let mut bytes = Vec::new();
    file.take(len).read_to_end(&mut bytes).map_err(io(path))?;
    Ok(bytes)
}

fn malformed(what: &str, reason: impl Into<String>) -> HubError {
    HubError::Malformed {
        what: what.to_owned(),
        reason: reason.into(),
    }
}

/// The headers an LFS action carries, as name and value.
fn action_headers(action: &Value) -> Vec<(String, String)> {
    action
        .get("header")
        .and_then(Value::as_object)
        .map(|h| {
            h.iter()
                .filter_map(|(k, v)| Some((k.to_ascii_lowercase(), v.as_str()?.to_owned())))
                .collect()
        })
        .unwrap_or_default()
}

impl Hub {
    /// Commits `adds` and the removal of `deletes` to `branch` of `repo`, on
    /// `parent` (the commit the branch was read at; `None` for a branch the
    /// caller did not read, which the Hub then does not check), with
    /// `message` as its summary.
    ///
    /// # Errors
    /// A file cannot be read, the Hub would ignore one, an upload or the
    /// commit is refused (the branch moved past `parent`, for one).
    pub fn commit(
        &self,
        repo: &RepoId,
        branch: &str,
        parent: Option<&CommitId>,
        message: &str,
        adds: &[Upload],
        deletes: &[RepoPath],
    ) -> Result<Committed, HubError> {
        let mut prepared = adds.iter().map(prepare).collect::<Result<Vec<_>, _>>()?;
        self.preupload(repo, branch, &mut prepared)?;
        let (lfs_uploaded, lfs_present) = self.upload_lfs(repo, branch, &prepared)?;
        let mut lines = Vec::with_capacity(prepared.len() + deletes.len() + 1);
        let mut header = json!({ "summary": message, "description": "" });
        if let Some(parent) = parent {
            header["parentCommit"] = json!(parent);
        }
        lines.push(json!({ "key": "header", "value": header }));
        let mut regular = 0;
        for file in &prepared {
            let path = file.upload.path.as_str();
            if file.lfs {
                lines.push(json!({ "key": "lfsFile", "value": {
                    "path": path, "algo": "sha256", "oid": file.sha256, "size": file.size,
                }}));
            } else {
                let bytes = std::fs::read(&file.upload.local).map_err(io(&file.upload.local))?;
                if hex(&Sha256::digest(&bytes)) != file.sha256 {
                    return Err(HubError::Invalid(format!(
                        "{} changed while it was being committed",
                        file.upload.local.display()
                    )));
                }
                lines.push(json!({ "key": "file", "value": {
                    "content": base64::engine::general_purpose::STANDARD.encode(&bytes),
                    "path": path,
                    "encoding": "base64",
                }}));
                regular += 1;
            }
        }
        for path in deletes {
            lines.push(json!({ "key": "deletedFile", "value": { "path": path.as_str() } }));
        }
        let mut body = Vec::new();
        for line in &lines {
            body.extend_from_slice(line.to_string().as_bytes());
            body.push(b'\n');
        }
        let what = format!("the commit to {branch} of {repo}");
        let reply = self
            .send(Request {
                what: what.clone(),
                method: Method::Post,
                url: format!(
                    "{}/api/datasets/{repo}/commit/{}",
                    self.endpoint,
                    encode_segment(branch)
                ),
                headers: vec![("content-type".into(), "application/x-ndjson".into())],
                body: Some(body),
                repeatable: false,
            })?
            .ok(&what)?;
        let value = reply.json(&what)?;
        let commit = value
            .get("commitOid")
            .and_then(Value::as_str)
            .ok_or_else(|| malformed(&what, "no commitOid"))?;
        Ok(Committed {
            commit: CommitId::parse(commit).map_err(|e| malformed(&what, e.to_string()))?,
            regular,
            lfs_uploaded,
            lfs_present,
        })
    }

    /// Asks the Hub which files go through LFS; a file it would ignore is
    /// refused, never dropped from the commit silently.
    fn preupload(
        &self,
        repo: &RepoId,
        branch: &str,
        prepared: &mut [Prepared<'_>],
    ) -> Result<(), HubError> {
        let what = format!("the pre-upload of files to {repo}");
        for chunk in prepared.chunks_mut(PREUPLOAD_CHUNK) {
            let files: Vec<Value> = chunk
                .iter()
                .map(|f| {
                    json!({
                        "path": f.upload.path.as_str(),
                        "sample": base64::engine::general_purpose::STANDARD.encode(&f.sample),
                        "size": f.size,
                    })
                })
                .collect();
            let url = format!(
                "{}/api/datasets/{repo}/preupload/{}",
                self.endpoint,
                encode_segment(branch)
            );
            let value = self
                .send(Request::post_json(&what, url, &json!({ "files": files })))?
                .ok(&what)?
                .json(&what)?;
            let answers = value
                .get("files")
                .and_then(Value::as_array)
                .ok_or_else(|| malformed(&what, "no list of files"))?;
            for file in chunk.iter_mut() {
                let path = file.upload.path.as_str();
                let answer = answers
                    .iter()
                    .find(|a| a.get("path").and_then(Value::as_str) == Some(path))
                    .ok_or_else(|| malformed(&what, format!("no answer for {path}")))?;
                if answer.get("shouldIgnore").and_then(Value::as_bool) == Some(true) {
                    return Err(HubError::Invalid(format!(
                        "the hub would ignore {path} (a .gitignore of {repo}); it is not committed"
                    )));
                }
                file.lfs = match answer.get("uploadMode").and_then(Value::as_str) {
                    Some("lfs") => true,
                    Some("regular") => false,
                    other => return Err(malformed(&what, format!("upload mode {other:?}"))),
                };
            }
        }
        Ok(())
    }

    /// Uploads the LFS files the Hub does not hold yet; returns how many were
    /// uploaded and how many it held.
    fn upload_lfs(
        &self,
        repo: &RepoId,
        branch: &str,
        prepared: &[Prepared<'_>],
    ) -> Result<(usize, usize), HubError> {
        let lfs: Vec<&Prepared<'_>> = prepared.iter().filter(|f| f.lfs).collect();
        if lfs.is_empty() {
            return Ok((0, 0));
        }
        let what = format!("the LFS batch of {repo}");
        let objects: Vec<Value> = lfs
            .iter()
            .map(|f| json!({ "oid": f.sha256, "size": f.size }))
            .collect();
        let body = json!({
            "operation": "upload",
            "transfers": ["basic", "multipart"],
            "objects": objects,
            "hash_algo": "sha256",
            "ref": { "name": branch },
        });
        let request = Request::post_lfs(
            &what,
            format!(
                "{}/datasets/{repo}.git/info/lfs/objects/batch",
                self.endpoint
            ),
            &body,
        );
        let value = self.send(request)?.ok(&what)?.json(&what)?;
        let answers = value
            .get("objects")
            .and_then(Value::as_array)
            .ok_or_else(|| malformed(&what, "no objects"))?;
        let (mut uploaded, mut present) = (0, 0);
        for file in lfs {
            let answer = answers
                .iter()
                .find(|a| a.get("oid").and_then(Value::as_str) == Some(file.sha256.as_str()))
                .ok_or_else(|| malformed(&what, format!("no answer for {}", file.upload.path)))?;
            if let Some(error) = answer.get("error") {
                return Err(HubError::Status {
                    what: format!("the LFS upload of {}", file.upload.path),
                    status: error
                        .get("code")
                        .and_then(Value::as_u64)
                        .map_or(0, |c| u16::try_from(c).unwrap_or(0)),
                    detail: error
                        .get("message")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                });
            }
            let Some(upload) = answer.pointer("/actions/upload") else {
                present += 1;
                continue;
            };
            self.transfer(file, upload)?;
            if let Some(verify) = answer.pointer("/actions/verify") {
                let href = verify
                    .get("href")
                    .and_then(Value::as_str)
                    .ok_or_else(|| malformed(&what, "a verify action without href"))?;
                let mut request = Request::post_lfs(
                    format!("verifying the LFS upload of {}", file.upload.path),
                    href.to_owned(),
                    &json!({ "oid": file.sha256, "size": file.size }),
                );
                request.headers.extend(action_headers(verify));
                let what = request.what.clone();
                self.send(request)?.ok(&what)?;
            }
            uploaded += 1;
        }
        Ok((uploaded, present))
    }

    /// Sends one file's content as the upload action says: one `PUT` of the
    /// whole file, or (with a `chunk_size`) a `PUT` per numbered part URL and
    /// a completion naming each part's entity tag.
    fn transfer(&self, file: &Prepared<'_>, upload: &Value) -> Result<(), HubError> {
        let what = format!("the LFS upload of {}", file.upload.path);
        let href = upload
            .get("href")
            .and_then(Value::as_str)
            .ok_or_else(|| malformed(&what, "an upload action without href"))?;
        let header = upload.get("header").and_then(Value::as_object);
        let chunk_size = header.and_then(|h| h.get("chunk_size")).and_then(|v| {
            v.as_str()
                .and_then(|s| s.parse::<u64>().ok())
                .or(v.as_u64())
        });
        let Some(chunk_size) = chunk_size else {
            let bytes = read_range(&file.upload.local, 0, file.size)?;
            let request = Request {
                what: what.clone(),
                method: Method::Put,
                url: href.to_owned(),
                headers: action_headers(upload),
                body: Some(bytes),
                repeatable: true,
            };
            return self.send(request)?.ok(&what).map(|_| ());
        };
        if chunk_size == 0 {
            return Err(malformed(&what, "a chunk size of 0"));
        }
        let mut parts: Vec<(u64, String)> = header
            .into_iter()
            .flatten()
            .filter_map(|(k, v)| Some((k.parse::<u64>().ok()?, v.as_str()?.to_owned())))
            .collect();
        parts.sort();
        if parts.len() as u64 != file.size.div_ceil(chunk_size) {
            return Err(malformed(
                &what,
                format!(
                    "{} part URLs for {} bytes in parts of {chunk_size}",
                    parts.len(),
                    file.size
                ),
            ));
        }
        let mut etags = Vec::with_capacity(parts.len());
        for (index, (_, url)) in parts.iter().enumerate() {
            let from = index as u64 * chunk_size;
            let bytes = read_range(&file.upload.local, from, chunk_size)?;
            let part_what = format!("part {} of {what}", index + 1);
            let reply = self
                .send(Request {
                    what: part_what.clone(),
                    method: Method::Put,
                    url: url.clone(),
                    headers: Vec::new(),
                    body: Some(bytes),
                    repeatable: true,
                })?
                .ok(&part_what)?;
            let etag = reply
                .header("etag")
                .filter(|e| !e.is_empty())
                .ok_or_else(|| malformed(&part_what, "no etag"))?;
            etags.push(json!({ "partNumber": index + 1, "etag": etag }));
        }
        let request = Request::post_lfs(
            format!("completing {what}"),
            href.to_owned(),
            &json!({ "oid": file.sha256, "parts": etags }),
        );
        let done = request.what.clone();
        self.send(request)?.ok(&done).map(|_| ())
    }
}
