// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements local emulators of the services a learning
// system distributes through, so its tests never reach a public service, for
// its clients. If your team needs expertise in testing hub integrations
// offline, you can procure our services by sending an email to
// info@swedishembedded.com.

//! A local emulator of the Hugging Face Hub endpoints the hub client uses,
//! on 127.0.0.1: repository creation and info, revisions, refs, tags, a
//! paged recursive tree, pre-upload, the LFS batch API (a single `PUT` with
//! a verification for a small file, numbered parts and a completion for a
//! large one), the JSON-lines commit with its parent check, and downloads.
//! Repositories live in memory; [`Emulator::tamper`] changes a stored file's
//! bytes behind the tree's back, as a compromised store would.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{BTreeMap, HashMap};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use base64::Engine as _;
use serde_json::{json, Value};
use sha2::{Digest as _, Sha256};

/// How the emulated Hub misbehaves.
#[derive(Clone, Copy, Debug, Default)]
pub struct Quirks {
    /// Creates every repository public, whatever it was asked.
    pub public: bool,
    /// Leaves the `private` flag out of a repository's info.
    pub no_private_flag: bool,
}

/// A file of a commit.
#[derive(Clone, Debug)]
struct Stored {
    bytes: Vec<u8>,
    /// The SHA-256 the tree lists it under when it went through LFS.
    lfs: Option<String>,
}

#[derive(Default)]
struct Repo {
    private: bool,
    commits: HashMap<String, BTreeMap<String, Stored>>,
    branches: BTreeMap<String, String>,
    tags: BTreeMap<String, String>,
}

#[derive(Default)]
struct State {
    quirks: Quirks,
    repos: BTreeMap<String, Repo>,
    lfs: HashMap<String, Vec<u8>>,
    parts: HashMap<String, BTreeMap<usize, Vec<u8>>>,
    /// Every request, as `METHOD path`.
    log: Vec<String>,
    base: String,
}

/// A running emulator.
pub struct Emulator {
    url: String,
    state: Arc<Mutex<State>>,
    stop: Arc<AtomicBool>,
}

/// LFS objects larger than this are sent in parts of this size.
const PART: usize = 4096;
/// Files at least this large go through LFS.
const LFS_FROM: u64 = 256;
/// Tree entries per page.
const PAGE: usize = 3;

fn hex_sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn decode(segment: &str) -> String {
    let bytes = segment.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            out.push(u8::from_str_radix(&segment[i + 1..i + 3], 16).unwrap());
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).unwrap()
}

struct Response {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

fn reply(status: u16, body: Value) -> Response {
    Response {
        status,
        headers: vec![("content-type".into(), "application/json".into())],
        body: body.to_string().into_bytes(),
    }
}

fn error(status: u16, message: &str) -> Response {
    reply(status, json!({ "error": message }))
}

impl Emulator {
    /// An emulator with the given quirks, serving until dropped.
    pub fn start(quirks: Quirks) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let state = Arc::new(Mutex::new(State {
            quirks,
            base: url.clone(),
            ..State::default()
        }));
        let stop = Arc::new(AtomicBool::new(false));
        let (served, stopping) = (Arc::clone(&state), Arc::clone(&stop));
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                if stopping.load(Ordering::SeqCst) {
                    break;
                }
                if let Ok(stream) = stream {
                    let state = Arc::clone(&served);
                    std::thread::spawn(move || serve(stream, &state));
                }
            }
        });
        Self { url, state, stop }
    }

    /// The endpoint to hand the client.
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Every request seen, as `METHOD path`.
    pub fn log(&self) -> Vec<String> {
        self.state.lock().unwrap().log.clone()
    }

    /// Commits made to `repo` beyond the one it was created with.
    pub fn commits(&self, repo: &str) -> usize {
        let state = self.state.lock().unwrap();
        state.repos.get(repo).map_or(0, |r| r.commits.len() - 1)
    }

    /// Replaces the bytes of `path` in every commit of `repo` (and the LFS
    /// object it names), leaving the tree's listing as it was.
    pub fn tamper(&self, repo: &str, path: &str) {
        let mut state = self.state.lock().unwrap();
        let mut objects = Vec::new();
        for files in state.repos.get_mut(repo).unwrap().commits.values_mut() {
            if let Some(stored) = files.get_mut(path) {
                let last = stored.bytes.len() - 1;
                stored.bytes[last] ^= 1;
                objects.extend(stored.lfs.clone());
            }
        }
        for oid in objects {
            if let Some(bytes) = state.lfs.get_mut(&oid) {
                let last = bytes.len() - 1;
                bytes[last] ^= 1;
            }
        }
    }
}

impl Drop for Emulator {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect(self.url.trim_start_matches("http://"));
    }
}

fn serve(mut stream: TcpStream, state: &Mutex<State>) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut line = String::new();
    if reader.read_line(&mut line).unwrap_or(0) == 0 {
        return;
    }
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_owned();
    let target = parts.next().unwrap_or_default().to_owned();
    let mut length = 0usize;
    let mut authorised = false;
    loop {
        let mut header = String::new();
        reader.read_line(&mut header).unwrap();
        let header = header.trim_end();
        if header.is_empty() {
            break;
        }
        let (name, value) = header.split_once(':').unwrap();
        match name.to_ascii_lowercase().as_str() {
            "content-length" => length = value.trim().parse().unwrap(),
            "authorization" => authorised = value.trim() == "Bearer test-token",
            _ => {}
        }
    }
    let mut body = vec![0u8; length];
    reader.read_exact(&mut body).unwrap();
    let response = {
        let mut state = state.lock().unwrap();
        state.log.push(format!("{method} {target}"));
        route(&mut state, &method, &target, &body, authorised)
    };
    let mut out = format!(
        "HTTP/1.1 {} X\r\ncontent-length: {}\r\nconnection: close\r\n",
        response.status,
        response.body.len()
    );
    for (name, value) in &response.headers {
        out.push_str(&format!("{name}: {value}\r\n"));
    }
    out.push_str("\r\n");
    let _ = stream.write_all(out.as_bytes());
    let _ = stream.write_all(&response.body);
}

fn new_commit(parent: &str, payload: &[u8]) -> String {
    let mut bytes = parent.as_bytes().to_vec();
    bytes.extend_from_slice(payload);
    hex_sha256(&bytes)[..40].to_owned()
}

fn resolve(repo: &Repo, revision: &str) -> Option<String> {
    repo.branches
        .get(revision)
        .or_else(|| repo.tags.get(revision))
        .cloned()
        .or_else(|| {
            repo.commits
                .contains_key(revision)
                .then(|| revision.to_owned())
        })
}

fn route(state: &mut State, method: &str, target: &str, body: &[u8], authorised: bool) -> Response {
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let segments: Vec<String> = path
        .trim_start_matches('/')
        .split('/')
        .map(decode)
        .collect();
    let seg: Vec<&str> = segments.iter().map(String::as_str).collect();
    let writes = matches!(method, "POST" | "PUT") && !seg[0].starts_with("lfs-");
    if writes && !authorised {
        return error(401, "unauthorised");
    }
    match (method, seg.as_slice()) {
        ("POST", ["api", "repos", "create"]) => create(state, body),
        ("GET", ["api", "datasets", owner, name]) => {
            let id = format!("{owner}/{name}");
            let quirks = state.quirks;
            match state.repos.get(&id) {
                None => error(404, "Repository not found"),
                Some(repo) => {
                    let mut info = json!({ "id": id, "sha": repo.branches["main"] });
                    if !quirks.no_private_flag {
                        info["private"] = json!(repo.private);
                    }
                    reply(200, info)
                }
            }
        }
        (_, ["api", "datasets", owner, name, rest @ ..]) => {
            let id = format!("{owner}/{name}");
            repo_call(state, &id, method, rest, query, body)
        }
        ("POST", ["datasets", owner, git, "info", "lfs", "objects", "batch"]) => {
            let id = format!("{owner}/{}", git.trim_end_matches(".git"));
            if !state.repos.contains_key(&id) {
                return error(404, "Repository not found");
            }
            batch(state, body)
        }
        ("GET", ["datasets", owner, name, "resolve", revision, file @ ..]) => {
            let id = format!("{owner}/{name}");
            let Some(repo) = state.repos.get(&id) else {
                return error(404, "Repository not found");
            };
            let Some(commit) = resolve(repo, revision) else {
                return error(404, "Revision not found");
            };
            match repo.commits[&commit].get(&file.join("/")) {
                Some(stored) => Response {
                    status: 200,
                    headers: Vec::new(),
                    body: stored.bytes.clone(),
                },
                None => error(404, "Entry not found"),
            }
        }
        ("PUT", ["lfs-put", oid]) => {
            if hex_sha256(body) != *oid {
                return error(400, "content does not match oid");
            }
            state.lfs.insert((*oid).to_owned(), body.to_vec());
            reply(200, json!({}))
        }
        ("PUT", ["lfs-part", oid, index]) => {
            state
                .parts
                .entry((*oid).to_owned())
                .or_default()
                .insert(index.parse().unwrap(), body.to_vec());
            let mut response = reply(200, json!({}));
            response
                .headers
                .push(("etag".into(), format!("\"{}\"", hex_sha256(body))));
            response
        }
        ("POST", ["lfs-complete", oid]) => {
            let payload: Value = serde_json::from_slice(body).unwrap();
            let parts = state.parts.remove(*oid).unwrap_or_default();
            let mut bytes = Vec::new();
            for (number, part) in payload["parts"].as_array().unwrap().iter().enumerate() {
                assert_eq!(part["partNumber"], json!(number + 1));
                bytes.extend_from_slice(&parts[&(number + 1)]);
            }
            if hex_sha256(&bytes) != *oid {
                return error(400, "parts do not match oid");
            }
            state.lfs.insert((*oid).to_owned(), bytes);
            reply(200, json!({}))
        }
        ("POST", ["lfs-verify", oid]) => {
            if state.lfs.contains_key(*oid) {
                reply(200, json!({}))
            } else {
                error(404, "not uploaded")
            }
        }
        _ => error(404, "no such endpoint"),
    }
}

fn create(state: &mut State, body: &[u8]) -> Response {
    let asked: Value = serde_json::from_slice(body).unwrap();
    let id = format!(
        "{}/{}",
        asked["organization"].as_str().unwrap(),
        asked["name"].as_str().unwrap()
    );
    if state.repos.contains_key(&id) {
        return error(409, "You already created this dataset repo");
    }
    let private = !state.quirks.public
        && (asked["private"] == json!(true) || asked["visibility"] == json!("private"));
    let mut files = BTreeMap::new();
    files.insert(
        ".gitattributes".to_owned(),
        Stored {
            bytes: b"*.zst filter=lfs diff=lfs merge=lfs -text\n".to_vec(),
            lfs: None,
        },
    );
    let commit = new_commit("", id.as_bytes());
    let mut repo = Repo {
        private,
        ..Repo::default()
    };
    repo.commits.insert(commit.clone(), files);
    repo.branches.insert("main".into(), commit);
    state.repos.insert(id.clone(), repo);
    reply(
        200,
        json!({ "url": format!("{}/datasets/{id}", state.base) }),
    )
}

fn repo_call(
    state: &mut State,
    id: &str,
    method: &str,
    rest: &[&str],
    query: &str,
    body: &[u8],
) -> Response {
    let base = state.base.clone();
    let lfs = state.lfs.clone();
    let Some(repo) = state.repos.get_mut(id) else {
        return error(404, "Repository not found");
    };
    match (method, rest) {
        ("GET", ["revision", revision]) => match resolve(repo, revision) {
            Some(sha) => reply(200, json!({ "id": id, "sha": sha })),
            None => error(404, "Revision not found"),
        },
        ("GET", ["refs"]) => {
            let list = |refs: &BTreeMap<String, String>, kind: &str| -> Vec<Value> {
                refs.iter()
                    .map(|(name, target)| {
                        json!({"name": name, "ref": format!("refs/{kind}/{name}"), "targetCommit": target})
                    })
                    .collect()
            };
            reply(
                200,
                json!({ "branches": list(&repo.branches, "heads"), "tags": list(&repo.tags, "tags"), "converts": [] }),
            )
        }
        ("GET", ["tree", revision]) => {
            let Some(commit) = resolve(repo, revision) else {
                return error(404, "Revision not found");
            };
            assert!(query.contains("recursive=true"), "{query}");
            let cursor: usize = query
                .split('&')
                .find_map(|p| p.strip_prefix("cursor="))
                .map_or(0, |c| c.parse().unwrap());
            let entries: Vec<Value> = repo.commits[&commit]
                .iter()
                .map(|(path, stored)| {
                    let mut entry = json!({"type": "file", "oid": "0", "size": stored.bytes.len(), "path": path});
                    if let Some(oid) = &stored.lfs {
                        entry["lfs"] = json!({"oid": oid, "size": stored.bytes.len(), "pointerSize": 130});
                    }
                    entry
                })
                .collect();
            let page: Vec<Value> = entries.iter().skip(cursor).take(PAGE).cloned().collect();
            let mut response = reply(200, Value::Array(page));
            if cursor + PAGE < entries.len() {
                response.headers.push((
                    "link".into(),
                    format!(
                        "<{base}/api/datasets/{id}/tree/{revision}?recursive=true&cursor={}>; rel=\"next\"",
                        cursor + PAGE
                    ),
                ));
            }
            response
        }
        ("POST", ["preupload", _]) => {
            let asked: Value = serde_json::from_slice(body).unwrap();
            let files: Vec<Value> = asked["files"]
                .as_array()
                .unwrap()
                .iter()
                .map(|f| {
                    let lfs = f["size"].as_u64().unwrap() >= LFS_FROM;
                    json!({"path": f["path"], "uploadMode": if lfs {"lfs"} else {"regular"}, "shouldIgnore": false})
                })
                .collect();
            reply(200, json!({ "files": files }))
        }
        ("POST", ["tag", revision]) => {
            let asked: Value = serde_json::from_slice(body).unwrap();
            let tag = asked["tag"].as_str().unwrap().to_owned();
            if repo.tags.contains_key(&tag) {
                return error(409, "tag exists");
            }
            match resolve(repo, revision) {
                Some(commit) => {
                    repo.tags.insert(tag, commit);
                    reply(200, json!({}))
                }
                None => error(404, "Revision not found"),
            }
        }
        ("POST", ["commit", branch]) => {
            let text = std::str::from_utf8(body).unwrap();
            let mut lines = text
                .lines()
                .map(|l| serde_json::from_str::<Value>(l).unwrap());
            let header = lines.next().unwrap();
            assert_eq!(header["key"], "header");
            let head = repo.branches[*branch].clone();
            if let Some(parent) = header["value"]["parentCommit"].as_str() {
                if parent != head {
                    return error(
                        412,
                        "A commit has happened since. Please refresh and try again.",
                    );
                }
            }
            let mut files = repo.commits[&head].clone();
            for line in lines {
                let value = &line["value"];
                let path = value["path"].as_str().unwrap().to_owned();
                match line["key"].as_str().unwrap() {
                    "file" => {
                        let bytes = base64::engine::general_purpose::STANDARD
                            .decode(value["content"].as_str().unwrap())
                            .unwrap();
                        files.insert(path, Stored { bytes, lfs: None });
                    }
                    "lfsFile" => {
                        let oid = value["oid"].as_str().unwrap();
                        let Some(bytes) = lfs.get(oid) else {
                            return error(422, "LFS object not uploaded");
                        };
                        assert_eq!(value["size"], json!(bytes.len()));
                        files.insert(
                            path,
                            Stored {
                                bytes: bytes.clone(),
                                lfs: Some(oid.to_owned()),
                            },
                        );
                    }
                    "deletedFile" => {
                        files.remove(&path);
                    }
                    other => panic!("unknown commit operation {other}"),
                }
            }
            let commit = new_commit(&head, body);
            repo.commits.insert(commit.clone(), files);
            repo.branches.insert((*branch).to_owned(), commit.clone());
            reply(
                200,
                json!({ "commitOid": commit, "commitUrl": format!("{base}/datasets/{id}/commit/{commit}") }),
            )
        }
        _ => error(404, "no such endpoint"),
    }
}

fn batch(state: &mut State, body: &[u8]) -> Response {
    let asked: Value = serde_json::from_slice(body).unwrap();
    assert_eq!(asked["operation"], "upload");
    let base = state.base.clone();
    let objects: Vec<Value> = asked["objects"]
        .as_array()
        .unwrap()
        .iter()
        .map(|object| {
            let oid = object["oid"].as_str().unwrap();
            let size = usize::try_from(object["size"].as_u64().unwrap()).unwrap();
            if state.lfs.contains_key(oid) {
                return json!({ "oid": oid, "size": size });
            }
            if size > PART {
                let mut header = serde_json::Map::new();
                header.insert("chunk_size".into(), json!(PART.to_string()));
                for part in 1..=size.div_ceil(PART) {
                    header.insert(
                        part.to_string(),
                        json!(format!("{base}/lfs-part/{oid}/{part}")),
                    );
                }
                json!({ "oid": oid, "size": size, "actions": { "upload": {
                    "href": format!("{base}/lfs-complete/{oid}"), "header": header,
                }}})
            } else {
                json!({ "oid": oid, "size": size, "actions": {
                    "upload": { "href": format!("{base}/lfs-put/{oid}") },
                    "verify": { "href": format!("{base}/lfs-verify/{oid}") },
                }})
            }
        })
        .collect();
    reply(200, json!({ "transfer": "basic", "objects": objects }))
}
