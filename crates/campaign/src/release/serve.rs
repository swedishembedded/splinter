// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model releases proven to run on the
// plain serving stack before they ship, for its clients. If your team
// needs expertise in model deployment, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The serve check: a candidate is released only if plain `brain serve`
//! runs it.
//!
//! `brain serve --openai 127.0.0.1:<port> --adapter <candidate>` is started
//! with the policy's base checkpoint as its Qwen3 (`BRAIN_QWEN_WEIGHTS`)
//! and brain's model store (`--models-dir`, when it exists). Before it
//! binds, brain binds the adapter to the base and prints `brain serve:
//! <model> adapter=<id> digest=sha256:<hex>`; that digest must be the
//! candidate adapter's. Once
//! it reports ready (`--ready-file`), a sample of the held-out tasks is
//! re-answered through its OpenAI-compatible endpoint, with the key it
//! wrote (`--api-keys-out`), decoded greedily as the in-process arms
//! were ([`GREEDY_SAMPLING`]) so the two answers compare serving rather
//! than two draws, and graded as in-process; every verdict must be the
//! same. The server is stopped when the check ends, however it ends.
//!
//! Without a `brain` binary, or when it does not start, the check is not
//! measured - and the gate fails.

use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

use splinter_agent::solve::Model;
use splinter_policy::local::GREEDY_SAMPLING;
use sven_sdk::CancelToken;

use crate::context::Context;
use crate::error::CampaignError;
use crate::release::gate::{self, Check, Serve};
use crate::release::probe::{grade, Suite};

/// What starts every line naming the adapter brain serves.
const STARTUP_PREFIX: &str = "brain serve: ";
/// The most output lines kept to explain a server that did not start.
const KEPT_LINES: usize = 20;

/// Starts `binary` serving `adapter` on `base`, and re-answers `sample`
/// (whose in-process verdicts are `in_process`, one per task) through it.
/// An error is returned only for a cancelled check; every other failure is
/// the check's result.
#[allow(clippy::too_many_arguments)]
pub fn check(
    ctx: &Context,
    binary: Option<&Path>,
    base: &Path,
    adapter: &Path,
    adapter_digest: &str,
    sample: &Suite,
    in_process: &[Option<bool>],
    startup: Duration,
    cancel: &CancelToken,
) -> Result<Check<Serve>, CampaignError> {
    let Some(binary) = binary else {
        return Ok(Check::unmeasured(
            "no brain binary: none on PATH and SPLINTER_BRAIN_BIN is not set",
        ));
    };
    let work = ctx.root().sandbox().join(format!(
        "serve-check-{}",
        splinter_store::new_id_with_prefix("gate")
    ));
    std::fs::create_dir_all(&work).map_err(crate::error::io(&work))?;
    let result = serve_and_ask(
        ctx,
        &Server {
            binary,
            base,
            adapter,
            work: &work,
            models: &ctx.config().model_store,
        },
        adapter_digest,
        sample,
        in_process,
        startup,
        cancel,
    );
    // The directory only held the keys and the ready marker.
    let _ = std::fs::remove_dir_all(&work);
    match result {
        Ok(check) => Ok(check),
        Err(Failure::Cancelled) => Err(CampaignError::Cancelled),
        Err(Failure::Unmeasured(why)) => Ok(Check::unmeasured(why)),
    }
}

/// Why the check has no result.
enum Failure {
    Cancelled,
    Unmeasured(String),
}

/// What to start.
struct Server<'a> {
    binary: &'a Path,
    base: &'a Path,
    adapter: &'a Path,
    work: &'a Path,
    models: &'a Path,
}

/// A running server, stopped when dropped.
struct Running(Child);

impl Drop for Running {
    fn drop(&mut self) {
        // Stopping a server that already exited is not an error worth
        // reporting over the check's own result.
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn serve_and_ask(
    ctx: &Context,
    server: &Server<'_>,
    adapter_digest: &str,
    sample: &Suite,
    in_process: &[Option<bool>],
    startup: Duration,
    cancel: &CancelToken,
) -> Result<Check<Serve>, Failure> {
    let port = free_port().map_err(|e| Failure::Unmeasured(format!("no free port: {e}")))?;
    let address = format!("127.0.0.1:{port}");
    let keys = server.work.join("keys.json");
    let ready = server.work.join("ready");
    let mut command = Command::new(server.binary);
    command
        .arg("serve")
        .args(["--openai", &address])
        .arg("--adapter")
        .arg(server.adapter)
        .arg("--api-keys-out")
        .arg(&keys)
        .arg("--ready-file")
        .arg(&ready);
    // brain scans the store it is given at startup; one that does not
    // exist is left to brain's own default.
    if server.models.is_dir() {
        command.arg("--models-dir").arg(server.models);
    }
    let mut child = command
        .env("BRAIN_QWEN_WEIGHTS", server.base)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| {
            Failure::Unmeasured(format!("{} could not start: {e}", server.binary.display()))
        })?;
    let lines = forward_lines(&mut child);
    let mut running = Running(child);

    let deadline = Instant::now() + startup;
    let mut kept: Vec<String> = Vec::new();
    let mut startup_line: Option<String> = None;
    while startup_line.is_none() || !ready.exists() {
        if cancel.is_cancelled() {
            return Err(Failure::Cancelled);
        }
        if let Ok(Some(status)) = running.0.try_wait() {
            kept.extend(lines.try_iter());
            return Err(Failure::Unmeasured(format!(
                "brain serve exited ({status}) before it served: {}",
                kept.join(" | ")
            )));
        }
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Err(Failure::Unmeasured(format!(
                "brain serve did not report its adapter and readiness within {}s: {}",
                startup.as_secs(),
                kept.join(" | ")
            )));
        }
        match lines.recv_timeout(left.min(Duration::from_millis(100))) {
            Ok(line) => {
                if startup_line.is_none() && is_adapter_line(&line) {
                    startup_line = Some(line.clone());
                }
                if kept.len() == KEPT_LINES {
                    kept.remove(0);
                }
                kept.push(line);
            }
            Err(RecvTimeoutError::Timeout) => {}
            // Both streams closed: the exit is reported by `try_wait` above
            // once the process is reaped.
            Err(RecvTimeoutError::Disconnected) => std::thread::sleep(Duration::from_millis(20)),
        }
    }
    let Some(startup_line) = startup_line else {
        unreachable!("the startup wait ends only with a startup line");
    };
    let Some((model, served_digest)) = parse_adapter_line(&startup_line) else {
        unreachable!("an adapter line parses: is_adapter_line checked it");
    };
    let mut measured = Serve {
        binary: server.binary.to_path_buf(),
        startup_line: startup_line.clone(),
        served_digest,
        expected_digest: adapter_digest.to_string(),
        sampled: 0,
        agreed: 0,
        disagreed: Vec::new(),
    };
    if measured.served_digest != measured.expected_digest {
        return Ok(gate::serve(measured));
    }
    let key = read_key(&keys).map_err(Failure::Unmeasured)?;
    let loaded = splinter_policy::selection::served_model(
        &format!("http://{address}/v1"),
        &key,
        &model,
        Some(GREEDY_SAMPLING.temperature),
    )
    .map_err(|e| Failure::Unmeasured(format!("the served endpoint: {e}")))?;
    let served = Model::new(loaded.provider(), loaded.identity());
    let answers = grade(ctx, &served, sample, cancel).map_err(|e| match e {
        CampaignError::Cancelled => Failure::Cancelled,
        other => Failure::Unmeasured(format!("asking the served candidate: {other}")),
    })?;
    measured.sampled = sample.tasks.len();
    for ((task, served), local) in sample.tasks.iter().zip(&answers).zip(in_process) {
        if served == local {
            measured.agreed += 1;
        } else {
            measured.disagreed.push(task.task.id.to_string());
        }
    }
    drop(running);
    Ok(gate::serve(measured))
}

/// Every line the child writes, from stdout and stderr alike.
fn forward_lines(child: &mut Child) -> Receiver<String> {
    let (tx, rx) = channel();
    let spawn = |stream: Option<Box<dyn Read + Send>>| {
        if let Some(stream) = stream {
            let tx = tx.clone();
            std::thread::spawn(move || {
                for line in BufReader::new(stream).lines() {
                    let Ok(line) = line else { break };
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
        let line = "brain serve: brain/qwen3 adapter=local/Qwen3-0.6B:splinter:candidate \
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
