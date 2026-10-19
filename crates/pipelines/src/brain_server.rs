// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model releases proven to run on the
// plain serving stack before they ship, for its clients. If your team
// needs expertise in model deployment, you can procure our services by
// sending an email to info@swedishembedded.com.

//! A `brain serve` process: started on a base with an adapter offered as a
//! file or as a release manifest to follow, waited for until it reports what
//! it serves and that it is ready, and stopped when dropped. A release's serve
//! check and an examination that batches its items both run their models
//! through it.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

use splinter_agent::CancelToken;

/// The model name plain `brain serve` answers to when the Qwen3 it serves
/// carries no adapter, as the adapter's startup line names it when it does.
const PLAIN_MODEL: &str = "brain/qwen3";
/// What starts every line naming the adapter brain serves.
const STARTUP_PREFIX: &str = "brain serve: ";
/// The most output lines kept to explain a server that did not start or stopped answering.
const KEPT_LINES: usize = 20;

/// The sequences `brain serve` decodes together when it is not told otherwise
/// (`--qwen-max-batch`): the width a server is asked at when its launch names
/// none.
pub const DEFAULT_SERVE_SLOTS: u32 = 16;

/// Why the check has no result.
pub enum Failure {
    /// The check was cancelled.
    Cancelled,
    /// It could not be measured, and why.
    Unmeasured(String),
}

/// How the candidate is put in front of the server.
#[derive(Clone, Copy)]
pub enum Offer<'a> {
    /// `--adapter FILE`: exactly this adapter, for the server's life.
    Adapter(&'a Path),
    /// `--adapter-manifest FILE`: whichever adapter the manifest names, as a
    /// release step publishes it.
    Manifest(&'a Path),
    /// Nothing: the base is the model (a consolidated checkpoint).
    Nothing,
}

/// What it takes to start the server.
pub struct Launch<'a> {
    /// The `brain` binary.
    pub binary: &'a Path,
    /// A scratch directory for the keys and the ready marker.
    pub work: &'a Path,
    /// Brain's model store, scanned at startup when it exists.
    pub models: &'a Path,
    /// The base checkpoint, as `BRAIN_QWEN_WEIGHTS`.
    pub base: &'a Path,
    /// How the candidate is offered.
    pub offer: Offer<'a>,
    /// The digest the candidate must be served under.
    pub expected_digest: &'a str,
    /// How long the server may take to bind.
    pub startup: Duration,
    /// The compute it may schedule, in brain's `--device` grammar; brain's
    /// own choice when `None`.
    pub device: Option<&'a str>,
    /// Sequences it decodes together (`--qwen-max-batch`);
    /// [`DEFAULT_SERVE_SLOTS`] when `None`.
    pub max_batch: Option<u32>,
    /// A file every line the server writes is appended to, for an operator
    /// to read after the fact; only the last lines are kept otherwise.
    pub log: Option<&'a Path>,
}

impl Launch<'_> {
    /// The sequences the server is started to decode together: what a caller
    /// that asks it many things at once keeps in flight.
    #[must_use]
    pub fn slots(&self) -> u32 {
        self.max_batch.unwrap_or(DEFAULT_SERVE_SLOTS).max(1)
    }
}

/// A running server, stopped when dropped.
pub struct Running(pub Child);

impl Drop for Running {
    fn drop(&mut self) {
        // Stopping a server that already exited is not an error worth
        // reporting over the check's own result.
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// A server that has bound and is ready.
pub struct Started {
    /// The process.
    pub running: Running,
    /// What it writes, line by line.
    pub lines: Receiver<String>,
    /// The last lines it wrote.
    pub kept: Vec<String>,
    /// Where it listens.
    pub address: String,
    /// The line that named what it serves.
    pub startup_line: String,
    /// The model name it answers to.
    pub model: String,
    /// The digest it reports for the adapter.
    pub served_digest: String,
    /// The OpenAI surface's key.
    pub key: String,
    /// The sequences it was started to decode together ([`Launch::slots`]).
    pub slots: u32,
}

impl Started {
    /// Whether what it serves is the candidate.
    pub fn serves_candidate(&self, expected: &str) -> bool {
        self.served_digest == expected
    }

    /// What the server last wrote, as one line of text.
    pub fn recent_output(&mut self) -> String {
        recent_output(&self.lines, &mut self.kept)
    }
}

/// Starts the server `launch` describes and waits for it to report what it
/// serves and that it is ready.
pub fn start(launch: &Launch<'_>, cancel: &CancelToken) -> Result<Started, Failure> {
    let port = free_port().map_err(|e| Failure::Unmeasured(format!("no free port: {e}")))?;
    let address = format!("127.0.0.1:{port}");
    let keys = launch.work.join("keys.json");
    let ready = launch.work.join("ready");
    let mut command = Command::new(launch.binary);
    if let Some(device) = launch.device {
        command.args(["--device", device]);
    }
    command.arg("serve").args(["--openai", &address]);
    // Always named, so the width a caller asks at is the one the server was
    // told rather than a default it might not share.
    command.args(["--qwen-max-batch", &launch.slots().to_string()]);
    match launch.offer {
        Offer::Adapter(adapter) => {
            command.arg("--adapter").arg(adapter);
        }
        Offer::Manifest(manifest) => {
            command.arg("--adapter-manifest").arg(manifest);
        }
        Offer::Nothing => {}
    }
    // The trainer's LoRA corrects every linear, the MLP ones included, and
    // brain's bank serves the attention linears only unless told otherwise.
    if !matches!(launch.offer, Offer::Nothing) {
        command.args(["--adapter-targets", "all"]);
    }
    command
        .arg("--api-keys-out")
        .arg(&keys)
        .arg("--ready-file")
        .arg(&ready);
    // brain scans the store it is given at startup; one that does not
    // exist is left to brain's own default.
    if launch.models.is_dir() {
        command.arg("--models-dir").arg(launch.models);
    }
    let mut child = command
        .env("BRAIN_QWEN_WEIGHTS", launch.base)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| {
            Failure::Unmeasured(format!("{} could not start: {e}", launch.binary.display()))
        })?;
    let log = match launch.log {
        Some(path) => Some(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .map_err(|e| Failure::Unmeasured(format!("{}: {e}", path.display())))?,
        ),
        None => None,
    };
    let lines = forward_lines(&mut child, log);
    let mut running = Running(child);

    let deadline = Instant::now() + launch.startup;
    let mut kept: Vec<String> = Vec::new();
    let mut startup_line: Option<String> = None;
    // Only an adapter is announced: a plain base prints no such line.
    let announced = !matches!(launch.offer, Offer::Nothing);
    while (announced && startup_line.is_none()) || !ready.exists() {
        if cancel.is_cancelled() {
            return Err(Failure::Cancelled);
        }
        if let Ok(Some(status)) = running.0.try_wait() {
            return Err(Failure::Unmeasured(format!(
                "brain serve exited ({status}) before it served: {}",
                recent_output(&lines, &mut kept)
            )));
        }
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Err(Failure::Unmeasured(format!(
                "brain serve did not report its adapter and readiness within {}s: {}",
                launch.startup.as_secs(),
                recent_output(&lines, &mut kept)
            )));
        }
        match lines.recv_timeout(left.min(Duration::from_millis(100))) {
            Ok(line) => {
                if startup_line.is_none() && is_adapter_line(&line) {
                    startup_line = Some(line.clone());
                }
                remember(&mut kept, line);
            }
            Err(RecvTimeoutError::Timeout) => {}
            // Both streams closed: the exit is reported by `try_wait` above
            // once the process is reaped.
            Err(RecvTimeoutError::Disconnected) => std::thread::sleep(Duration::from_millis(20)),
        }
    }
    let (startup_line, model, served_digest) = match startup_line {
        Some(line) => {
            let Some((model, digest)) = parse_adapter_line(&line) else {
                unreachable!("an adapter line parses: is_adapter_line checked it");
            };
            (line, model, digest)
        }
        None => (
            "no adapter: the consolidated checkpoint is the base".to_string(),
            PLAIN_MODEL.to_string(),
            launch.expected_digest.to_string(),
        ),
    };
    let key = read_key(&keys).map_err(|why| {
        Failure::Unmeasured(format!(
            "{why}; it wrote: {}",
            recent_output(&lines, &mut kept)
        ))
    })?;
    Ok(Started {
        running,
        lines,
        kept,
        address,
        startup_line,
        model,
        served_digest,
        key,
        slots: launch.slots(),
    })
}

/// Keeps `line` among the last [`KEPT_LINES`] of `kept`.
pub fn remember(kept: &mut Vec<String>, line: String) {
    if kept.len() == KEPT_LINES {
        kept.remove(0);
    }
    kept.push(line);
}

/// What the server last wrote, as one line of text.
pub fn recent_output(lines: &Receiver<String>, kept: &mut Vec<String>) -> String {
    for line in lines.try_iter() {
        remember(kept, line);
    }
    kept.join(" | ")
}

/// Every line the child writes, from stdout and stderr alike, each also
/// appended to `log` when one is given.
fn forward_lines(child: &mut Child, log: Option<std::fs::File>) -> Receiver<String> {
    let (tx, rx) = channel();
    let log = log.map(|file| std::sync::Arc::new(std::sync::Mutex::new(file)));
    let spawn = |stream: Option<Box<dyn Read + Send>>| {
        if let Some(stream) = stream {
            let tx = tx.clone();
            let log = log.clone();
            std::thread::spawn(move || {
                for line in BufReader::new(stream).lines() {
                    let Ok(line) = line else { break };
                    if let Some(log) = &log {
                        let mut file = log
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        // A log that cannot be written loses the line there
                        // only; the line still reaches the caller.
                        let _ = writeln!(file, "{line}");
                    }
                    if tx.send(line).is_err() {
                        break;
                    }
                }
            });
        }
    };
    spawn(
        child
            .stdout
            .take()
            .map(|s| Box::new(s) as Box<dyn Read + Send>),
    );
    spawn(
        child
            .stderr
            .take()
            .map(|s| Box::new(s) as Box<dyn Read + Send>),
    );
    rx
}

fn is_adapter_line(line: &str) -> bool {
    parse_adapter_line(line).is_some()
}

/// `(model, digest)` from `brain serve: <model> adapter=<id> digest=<d>`.
fn parse_adapter_line(line: &str) -> Option<(String, String)> {
    let rest = line.trim().strip_prefix(STARTUP_PREFIX)?;
    let mut words = rest.split_whitespace();
    let model = words.next()?;
    let mut adapter = None;
    let mut digest = None;
    for word in words {
        if let Some(id) = word.strip_prefix("adapter=") {
            adapter = Some(id);
        } else if let Some(d) = word.strip_prefix("digest=") {
            digest = Some(d);
        }
    }
    adapter?;
    Some((model.to_string(), digest?.to_string()))
}

/// The OpenAI surface's key from the `--api-keys-out` file.
fn read_key(path: &PathBuf) -> Result<String, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("brain serve wrote no API keys to {}: {e}", path.display()))?;
    let keys: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("{} is not JSON: {e}", path.display()))?;
    keys.get("openai")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("{} holds no openai key", path.display()))
}

/// A port nothing listens on now; the server binds it next.
fn free_port() -> std::io::Result<u16> {
    Ok(std::net::TcpListener::bind("127.0.0.1:0")?
        .local_addr()?
        .port())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_startup_line_names_the_model_and_the_adapter_digest() {
        let line = "brain serve: brain/qwen3 adapter=local/Qwen3-0.6B:splinter:candidate:latest \
                    digest=sha256:ab";
        assert_eq!(
            parse_adapter_line(line),
            Some(("brain/qwen3".into(), "sha256:ab".into()))
        );
        assert_eq!(
            parse_adapter_line("brain serve: --adapter needs a value"),
            None
        );
        assert_eq!(parse_adapter_line("APIKEY openai sk-brain-1"), None);
    }
}
