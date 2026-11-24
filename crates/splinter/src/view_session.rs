// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements audit trails for data factories, where every
// intermediate product can be read and every decision lands in the
// repository, for its clients. If your team needs expertise in reviewable
// machine-learning pipelines, you can procure our services by sending an
// email to info@swedishembedded.com.

//! `splinter view`: a factory run in the browser, every stage's product
//! readable and every decision written to a review file.
//!
//! A small HTTP server on the loopback interface, built into the binary as
//! the other local endpoints are: one page (`view/index.html`, compiled in)
//! and a JSON interface over the run directory. Nothing outside the run is
//! served: every path is resolved and must stay inside it. A decision the
//! page posts is appended, as one JSON line, to the review file the command
//! names (`--reviews`, for example a repository's `reviews/tasks.jsonl`), so
//! approving in the browser is a change of the repository's own files. The
//! server never writes anything else.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};
use splinter_sdk::factory::run;
use splinter_sdk::Error;

use crate::cli::ViewArgs;
use crate::session::Exit;

const PAGE: &str = include_str!("view/index.html");
/// The largest file served whole; a larger one is shown as its first part.
const MAX_FILE: u64 = 2 * 1024 * 1024;
/// The largest request body (a decision) accepted.
const MAX_BODY: usize = 64 * 1024;
const REVIEW_SCHEMA: &str = "splinter.review/v1";

/// What the server holds: the run and, when given, the review file.
pub struct View {
    run: PathBuf,
    reviews: Option<PathBuf>,
    writing: Mutex<()>,
}

/// An HTTP answer.
#[derive(Debug, PartialEq)]
pub struct Answer {
    pub status: u16,
    pub content_type: &'static str,
    pub body: Vec<u8>,
}

impl Answer {
    fn json(status: u16, value: &Value) -> Self {
        Self {
            status,
            content_type: "application/json",
            body: value.to_string().into_bytes(),
        }
    }
    fn refuse(status: u16, why: &str) -> Self {
        Self::json(status, &json!({ "error": why }))
    }
}

/// Serve the run until interrupted.
///
/// # Errors
/// The run is not a run directory, or no loopback port can be bound.
pub fn view(args: &ViewArgs) -> Result<Exit, Error> {
    run::require_run(&args.run).map_err(|e| Error::Refused(e.to_string()))?;
    let view = Arc::new(View::new(&args.run, args.reviews.as_deref())?);
    let listener = TcpListener::bind(("127.0.0.1", args.port))
        .map_err(|e| Error::Refused(format!("cannot listen on 127.0.0.1:{}: {e}", args.port)))?;
    let address = listener
        .local_addr()
        .map_err(|e| Error::Refused(e.to_string()))?;
    println!("splinter view: http://{address}/ (interrupt to stop)");
    if let Some(reviews) = &view.reviews {
        println!("decisions are appended to {}", reviews.display());
    }
    for stream in listener.incoming().flatten() {
        let view = Arc::clone(&view);
        std::thread::spawn(move || {
            let _ = serve(&view, stream);
        });
    }
    Ok(Exit::Ok)
}

fn serve(view: &View, mut stream: TcpStream) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    reader.read_line(&mut request_line)?;
    let mut parts = request_line.split_whitespace();
    let (method, target) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
    let mut length = 0usize;
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header)? == 0 || header.trim().is_empty() {
            break;
        }
        if let Some((name, value)) = header.split_once(':') {
            if name.trim().eq_ignore_ascii_case("content-length") {
                length = value.trim().parse().unwrap_or(usize::MAX);
            }
        }
    }
    let answer = if length > MAX_BODY {
        Answer::refuse(413, "the request is too large")
    } else {
        let mut body = vec![0; length];
        reader.read_exact(&mut body)?;
        view.answer(method, target, &body)
    };
    write!(
        stream,
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        answer.status,
        if answer.status < 400 { "OK" } else { "Error" },
        answer.content_type,
        answer.body.len()
    )?;
    stream.write_all(&answer.body)
}

impl View {
    /// A view of `run`, deciding into `reviews`.
    ///
    /// # Errors
    /// The run directory cannot be resolved.
    pub fn new(run: &Path, reviews: Option<&Path>) -> Result<Self, Error> {
        let run = run
            .canonicalize()
            .map_err(|e| Error::Refused(format!("{}: {e}", run.display())))?;
        Ok(Self {
            run,
            reviews: reviews.map(Path::to_path_buf),
            writing: Mutex::new(()),
        })
    }

    /// The answer to one request.
    pub fn answer(&self, method: &str, target: &str, body: &[u8]) -> Answer {
        let (route, query) = target.split_once('?').unwrap_or((target, ""));
        let path = parameter(query, "path").unwrap_or_default();
        match (method, route) {
            ("GET", "/") => Answer {
                status: 200,
                content_type: "text/html; charset=utf-8",
                body: PAGE.as_bytes().to_vec(),
            },
            ("GET", "/api/run") => self.summary(),
            ("GET", "/api/list") => self.list(&path),
            ("GET", "/api/file") => self.file(&path),
            ("GET", "/api/reviews") => self.decisions(),
            ("POST", "/api/review") => self.decide(body),
            _ => Answer::refuse(404, "no such route"),
        }
    }

    fn summary(&self) -> Answer {
        let stages = match run::status(&self.run) {
            Ok(stages) => stages
                .into_iter()
                .map(|s| {
                    json!({
                        "key": s.key, "state": s.state, "error": s.error,
                        "product": s.product.map(|d| d.to_string()), "slots": s.slots,
                        "report": self.run.join(&s.key).join("report.md").is_file(),
                    })
                })
                .collect::<Vec<_>>(),
            Err(e) => return Answer::refuse(500, &e.to_string()),
        };
        let recipe = std::fs::read_to_string(self.run.join("recipe.yaml")).unwrap_or_default();
        let name = recipe
            .lines()
            .find_map(|l| l.strip_prefix("name:"))
            .map(|n| n.trim().trim_matches('"').to_string());
        Answer::json(
            200,
            &json!({
                "run": self.run.file_name().map(|n| n.to_string_lossy().to_string()),
                "recipe": name, "stages": stages,
                "reviews": self.reviews.as_ref().map(|p| p.display().to_string()),
            }),
        )
    }

    /// `relative` resolved inside the run, or why not.
    fn inside(&self, relative: &str) -> Result<PathBuf, Answer> {
        let relative = Path::new(relative);
        if relative
            .components()
            .any(|c| !matches!(c, Component::Normal(_) | Component::CurDir))
        {
            return Err(Answer::refuse(
                400,
                "a path inside the run, relative, without ..",
            ));
        }
        let resolved = self
            .run
            .join(relative)
            .canonicalize()
            .map_err(|_| Answer::refuse(404, "no such file in the run"))?;
        if resolved.starts_with(&self.run) {
            Ok(resolved)
        } else {
            Err(Answer::refuse(403, "outside the run"))
        }
    }

    fn list(&self, relative: &str) -> Answer {
        let directory = match self.inside(relative) {
            Ok(d) if d.is_dir() => d,
            Ok(_) => return Answer::refuse(400, "not a directory"),
            Err(answer) => return answer,
        };
        let mut entries: Vec<Value> = std::fs::read_dir(&directory)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|entry| {
                let meta = entry.metadata().ok()?;
                Some(json!({
                    "name": entry.file_name().to_string_lossy(),
                    "dir": meta.is_dir(), "size": meta.len(),
                }))
            })
            .collect();
        entries.sort_by(|a, b| {
            (b["dir"].as_bool(), a["name"].as_str()).cmp(&(a["dir"].as_bool(), b["name"].as_str()))
        });
        let items = std::fs::read_to_string(directory.join("items.json"))
            .ok()
            .and_then(|t| serde_json::from_str::<Value>(&t).ok());
        Answer::json(
            200,
            &json!({ "path": relative, "entries": entries, "items": items }),
        )
    }

    fn file(&self, relative: &str) -> Answer {
        let path = match self.inside(relative) {
            Ok(p) if p.is_file() => p,
            Ok(_) => return Answer::refuse(400, "not a file"),
            Err(answer) => return answer,
        };
        let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        let mut bytes = Vec::new();
        if std::fs::File::open(&path)
            .and_then(|f| f.take(MAX_FILE).read_to_end(&mut bytes))
            .is_err()
        {
            return Answer::refuse(500, "the file cannot be read");
        }
        let text = String::from_utf8(bytes).ok();
        Answer::json(
            200,
            &json!({
                "path": relative, "size": size, "truncated": size > MAX_FILE,
                "text": text, "binary": text.is_none(),
            }),
        )
    }

    fn decisions(&self) -> Answer {
        let Some(reviews) = &self.reviews else {
            return Answer::json(200, &json!({ "reviews": null, "decisions": [] }));
        };
        let decisions: Vec<Value> = std::fs::read_to_string(reviews)
            .unwrap_or_default()
            .lines()
            .filter_map(|l| serde_json::from_str::<Value>(l).ok())
            .filter(|d| d["schema"] == REVIEW_SCHEMA)
            .collect();
        Answer::json(
            200,
            &json!({ "reviews": reviews.display().to_string(), "decisions": decisions }),
        )
    }

    fn decide(&self, body: &[u8]) -> Answer {
        let Some(reviews) = &self.reviews else {
            return Answer::refuse(409, "no review file: start the view with --reviews FILE");
        };
        let Ok(request) = serde_json::from_slice::<Value>(body) else {
            return Answer::refuse(400, "a decision is a JSON object");
        };
        let target = request["target"].as_str().unwrap_or_default();
        let decision = request["decision"].as_str().unwrap_or_default();
        if target.is_empty() || !matches!(decision, "approved" | "rejected" | "note") {
            return Answer::refuse(
                400,
                "a decision names a target and is approved, rejected or note",
            );
        }
        if let Err(answer) = self.inside(target) {
            return answer;
        }
        let record = json!({
            "schema": REVIEW_SCHEMA,
            "run": self.run.file_name().map(|n| n.to_string_lossy().to_string()),
            "target": target, "decision": decision,
            "note": request["note"].as_str().unwrap_or_default(),
            "reviewer": request["reviewer"].as_str().unwrap_or_default(),
            "at": SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
        });
        let _held = self
            .writing
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let written = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(reviews)
            .and_then(|mut f| writeln!(f, "{record}"));
        match written {
            Ok(()) => Answer::json(200, &record),
            Err(e) => Answer::refuse(500, &format!("cannot write {}: {e}", reviews.display())),
        }
    }
}

/// The value of `name` in a query string, percent-decoded.
fn parameter(query: &str, name: &str) -> Option<String> {
    query.split('&').find_map(|pair| {
        let (key, value) = pair.split_once('=')?;
        (key == name).then(|| decode(value))
    })
}

fn decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => out.push(b' '),
            // `%` and two hexadecimal digits; anything else stays as written.
            b'%' if i + 2 < bytes.len() => {
                match u8::from_str_radix(
                    std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or(""),
                    16,
                ) {
                    Ok(b) => {
                        out.push(b);
                        i += 2;
                    }
                    Err(_) => out.push(b'%'),
                }
            }
            b => out.push(b),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (tempfile::TempDir, View) {
        let dir = tempfile::tempdir().unwrap();
        let run = dir.path().join("run");
        std::fs::create_dir_all(run.join("units/u1")).unwrap();
        std::fs::write(
            run.join("units/u1/unit.md"),
            "S1:1| k_msgq_put(&q, &e, K_NO_WAIT);\n",
        )
        .unwrap();
        std::fs::write(dir.path().join("secret.txt"), "not served").unwrap();
        let view = View::new(&run, Some(&dir.path().join("reviews.jsonl"))).unwrap();
        (dir, view)
    }

    fn body(answer: &Answer) -> Value {
        serde_json::from_slice(&answer.body).unwrap()
    }

    #[test]
    fn files_inside_the_run_are_served_and_nothing_outside_it() {
        let (_dir, view) = fixture();
        let file = view.answer("GET", "/api/file?path=units%2Fu1%2Funit.md", b"");
        assert_eq!(file.status, 200);
        assert!(body(&file)["text"].as_str().unwrap().contains("K_NO_WAIT"));
        assert_eq!(
            view.answer("GET", "/api/file?path=..%2Fsecret.txt", b"")
                .status,
            400
        );
        assert_eq!(
            view.answer("GET", "/api/file?path=%2Fetc%2Fpasswd", b"")
                .status,
            400
        );
        let listing = body(&view.answer("GET", "/api/list?path=units", b""));
        assert_eq!(listing["entries"][0]["name"], "u1");
        assert_eq!(
            view.answer("GET", "/", b"").content_type,
            "text/html; charset=utf-8"
        );
    }

    #[test]
    fn a_decision_is_appended_to_the_review_file_and_read_back() {
        let (dir, view) = fixture();
        let decided = view.answer(
            "POST",
            "/api/review",
            br#"{"target": "units/u1", "decision": "approved", "note": "fine", "reviewer": "me"}"#,
        );
        assert_eq!(decided.status, 200);
        let written = std::fs::read_to_string(dir.path().join("reviews.jsonl")).unwrap();
        let record: Value = serde_json::from_str(written.trim()).unwrap();
        assert_eq!(
            (
                record["schema"].as_str(),
                record["target"].as_str(),
                record["decision"].as_str()
            ),
            (Some(REVIEW_SCHEMA), Some("units/u1"), Some("approved"))
        );
        assert_eq!(
            body(&view.answer("GET", "/api/reviews", b""))["decisions"][0]["note"],
            "fine"
        );
        assert_eq!(
            view.answer(
                "POST",
                "/api/review",
                br#"{"target": "../x", "decision": "approved"}"#
            )
            .status,
            400
        );
        assert_eq!(
            view.answer(
                "POST",
                "/api/review",
                br#"{"target": "units/u1", "decision": "maybe"}"#
            )
            .status,
            400
        );
    }

    #[test]
    fn a_view_without_a_review_file_refuses_decisions() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("run")).unwrap();
        let view = View::new(&dir.path().join("run"), None).unwrap();
        assert_eq!(
            view.answer(
                "POST",
                "/api/review",
                br#"{"target": ".", "decision": "approved"}"#
            )
            .status,
            409
        );
    }

    #[test]
    fn query_parameters_are_percent_decoded() {
        assert_eq!(
            parameter("a=1&path=units%2Fu%201", "path").as_deref(),
            Some("units/u 1")
        );
        assert_eq!(decode("100%"), "100%");
    }
}
