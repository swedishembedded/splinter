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
//! than two draws and, as in-process, without a reasoning block, and
//! graded as in-process. A task is answered alike when the verdict is the
//! same and the final answer is the same text up to the numerical noise of
//! two processes decoding one model ([`alike`]) or, worded otherwise, says
//! the same thing ([`meaning::says_the_same`]: nearer in meaning to its own
//! in-process answer than to the answer to any other task, so the
//! comparison cannot call everything alike). A served model may sample, and
//! its kernels sum in another order, so the wording is not what must
//! survive; the meaning is. At least
//! [`gate::SERVE_AGREEMENT_PERCENT`] percent of the tasks must be. Each
//! task answered differently is reported with both answers. The server is
//! stopped when the check ends, however it ends.
//!
//! The server loads its own copy of the base, so every base this process
//! keeps resident is released before it starts.
//!
//! Without a `brain` binary, or when it does not start, the check is not
//! measured - and the gate fails.

use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

use splinter_agent::solve::Model;
use splinter_agent::CancelToken;
use splinter_model::local::GREEDY_SAMPLING;

use crate::release::meaning;
use crate::release::probe::{grade, Probe, Suite};
use splinter_eval::gate::{self, Check, Disagreement, Serve};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// What starts every line naming the adapter brain serves.
const STARTUP_PREFIX: &str = "brain serve: ";
/// The most output lines kept to explain a server that did not start.
const KEPT_LINES: usize = 20;

/// What the serve check is asked to prove.
pub struct ServeCheck<'a> {
    /// The `brain` that serves; `None` when there is none to be found.
    pub binary: Option<&'a Path>,
    /// The base checkpoint the adapter sits on.
    pub base: &'a Path,
    /// The candidate's adapter file.
    pub adapter: &'a Path,
    /// The digest the adapter must be served under.
    pub adapter_digest: &'a str,
    /// The system prompt the candidate is asked under; `None` is the default.
    pub system: Option<&'a str>,
    /// The held-out tasks re-answered through the server.
    pub sample: &'a Suite,
    /// The candidate's in-process answers to them, one per task.
    pub in_process: &'a [Probe],
    /// How long the server may take to bind.
    pub startup: Duration,
}

/// Starts the `brain` of `request` serving its adapter on its base, and
/// re-answers its sample through it. An error is returned only for a
/// cancelled check; every other failure is the check's result.
pub fn check(
    ctx: &Context,
    request: &ServeCheck<'_>,
    cancel: &CancelToken,
) -> Result<Check<Serve>, OrchestratorError> {
    let Some(binary) = request.binary else {
        return Ok(Check::unmeasured(
            "no brain binary: none on PATH and SPLINTER_BRAIN_BIN is not set",
        ));
    };
    // `brain serve` is another process loading its own copy of the base:
    // this one holds none while it runs.
    ctx.release_bases();
    let work = ctx.root().sandbox().join(format!(
        "serve-check-{}",
        splinter_store::new_id_with_prefix("gate")
    ));
    std::fs::create_dir_all(&work).map_err(splinter_orchestrator::error::io(&work))?;
    let result = serve_and_ask(
        ctx,
        request,
        &Spawn {
            binary,
            work: &work,
            models: &ctx.config().model_store,
        },
        cancel,
    );
    // The directory only held the keys and the ready marker.
    let _ = std::fs::remove_dir_all(&work);
    match result {
        Ok(check) => Ok(check),
        Err(Failure::Cancelled) => Err(OrchestratorError::Cancelled),
        Err(Failure::Unmeasured(why)) => Ok(Check::unmeasured(why)),
    }
}

/// Why the check has no result.
enum Failure {
    Cancelled,
    Unmeasured(String),
}

/// How the server is started.
struct Spawn<'a> {
    binary: &'a Path,
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
    request: &ServeCheck<'_>,
    server: &Spawn<'_>,
    cancel: &CancelToken,
) -> Result<Check<Serve>, Failure> {
    let (sample, in_process, startup) = (request.sample, request.in_process, request.startup);
    let port = free_port().map_err(|e| Failure::Unmeasured(format!("no free port: {e}")))?;
    let address = format!("127.0.0.1:{port}");
    let keys = server.work.join("keys.json");
    let ready = server.work.join("ready");
    let mut command = Command::new(server.binary);
    command
        .arg("serve")
        .args(["--openai", &address])
        .arg("--adapter")
        .arg(request.adapter)
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
        .env("BRAIN_QWEN_WEIGHTS", request.base)
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
        expected_digest: request.adapter_digest.to_string(),
        sampled: 0,
        agreed: 0,
        disagreed: Vec::new(),
    };
    if measured.served_digest != measured.expected_digest {
        return Ok(gate::serve(measured));
    }
    let key = read_key(&keys).map_err(Failure::Unmeasured)?;
    let loaded = splinter_model::selection::served_model(
        &format!("http://{address}/v1"),
        &key,
        &model,
        Some(GREEDY_SAMPLING.temperature),
    )
    .map_err(|e| Failure::Unmeasured(format!("the served endpoint: {e}")))?;
    // Asked as the candidate is: under the prompt it was trained under.
    let served = Model::new(loaded.provider(), loaded.identity());
    let served = match request.system {
        Some(system) => served.with_system(system),
        None => served,
    };
    let answers = grade(ctx, &served, sample, cancel).map_err(|e| match e {
        OrchestratorError::Cancelled => Failure::Cancelled,
        other => Failure::Unmeasured(format!("asking the served candidate: {other}")),
    })?;
    measured.sampled = sample.tasks.len();
    // The server has answered everything it will be asked.
    drop(running);
    let agree = agreements(ctx, &answers, in_process).map_err(Failure::Unmeasured)?;
    for (((task, served), local), agreed) in
        sample.tasks.iter().zip(&answers).zip(in_process).zip(agree)
    {
        if agreed {
            measured.agreed += 1;
        } else {
            measured.disagreed.push(Disagreement {
                task: task.task.id.to_string(),
                in_process: local.answer.clone(),
                served: served.answer.clone(),
                in_process_verdict: local.verdict,
                served_verdict: served.verdict,
            });
        }
    }
    Ok(gate::serve(measured))
}

/// Per task, whether the served answer is the in-process one: [`alike`] as
/// text, or - the same verdict, but worded differently - saying the same
/// thing ([`meaning::says_the_same`]). The embedding model is loaded only
/// when some task is not alike as text.
fn agreements(ctx: &Context, served: &[Probe], in_process: &[Probe]) -> Result<Vec<bool>, String> {
    let mut agree: Vec<bool> = served
        .iter()
        .zip(in_process)
        .map(|(s, l)| alike(s, l))
        .collect();
    // Tasks answered on both sides with the same verdict, not yet agreed.
    let reworded: Vec<usize> = (0..agree.len())
        .filter(|&i| {
            !agree[i]
                && served[i].verdict == in_process[i].verdict
                && served[i].answer.is_some()
                && in_process[i].answer.is_some()
        })
        .collect();
    if reworded.is_empty() {
        return Ok(agree);
    }
    // The control is every task answered on both sides, so an answer must
    // be nearer its own counterpart than to the other tasks' answers.
    let answered: Vec<usize> = (0..agree.len())
        .filter(|&i| served[i].answer.is_some() && in_process[i].answer.is_some())
        .collect();
    let texts = |probes: &[Probe]| -> Vec<String> {
        answered
            .iter()
            .filter_map(|&i| probes[i].answer.clone())
            .collect()
    };
    let embedder = ctx
        .embedder()
        .map_err(|e| format!("comparing what the answers say: {e}"))?;
    let embed = |texts: &[String]| {
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        embedder
            .embed(&refs)
            .map_err(|e| format!("comparing what the answers say: {e}"))
    };
    let same = meaning::says_the_same(&embed(&texts(served))?, &embed(&texts(in_process))?);
    for (&task, same) in answered.iter().zip(same) {
        if reworded.contains(&task) {
            agree[task] = same;
        }
    }
    Ok(agree)
}

/// How much of the shorter of two answers the longer must begin with, in
/// percent, for a served answer to count as the in-process one. Two
/// processes decode the same weights with kernels that sum in a different
/// order (a prompt prefilled in chunks of another size, say), so a long
/// greedy continuation may part ways near its end on a near-tie between two
/// tokens; an adapter bound wrongly parts ways at the start.
const ALIKE_PREFIX_PERCENT: usize = 90;

/// Whether two probes of one task answered alike: the same verdict, and
/// final answers - every run of whitespace one space, none at either end -
/// that are equal or share a beginning of at least
/// [`ALIKE_PREFIX_PERCENT`] percent of the shorter one. Comparing verdicts
/// alone would prove nothing where both sides fail: two different wrong
/// answers grade the same.
fn alike(a: &Probe, b: &Probe) -> bool {
    let words = |answer: &Option<String>| {
        answer
            .as_deref()
            .map(|text| text.split_whitespace().collect::<Vec<_>>().join(" "))
    };
    if a.verdict != b.verdict {
        return false;
    }
    match (words(&a.answer), words(&b.answer)) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            let shared = a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count();
            let shorter = a.chars().count().min(b.chars().count());
            shared * 100 >= shorter * ALIKE_PREFIX_PERCENT
        }
        _ => false,
    }
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
    fn answers_alike_are_the_same_text_whitespace_aside_and_graded_the_same() {
        let probe = |answer: Option<&str>, verdict: Option<bool>| Probe {
            answer: answer.map(str::to_string),
            verdict,
        };
        let wrong = Some(false);
        assert!(alike(
            &probe(Some("115200  baud\n"), wrong),
            &probe(Some(" 115200 baud"), wrong)
        ));
        assert!(
            !alike(
                &probe(Some("9600 baud"), wrong),
                &probe(Some("I do not know."), wrong)
            ),
            "two wrong answers are not the same answer"
        );
        assert!(!alike(
            &probe(Some("115200 baud"), Some(true)),
            &probe(Some("115200 baud"), None)
        ));
        assert!(alike(&probe(None, None), &probe(None, None)));
        assert!(!alike(&probe(None, wrong), &probe(Some(""), wrong)));
    }

    #[test]
    fn a_long_answer_that_parts_ways_only_at_its_end_is_still_the_same_answer() {
        let probe = |answer: &str| Probe {
            answer: Some(answer.to_string()),
            verdict: Some(false),
        };
        let in_process = "The default location for models if `brain-data-dir` is not \
                          specified is `<home>/brain`.";
        let served = "The default location for models if `brain-data-dir` is not \
                      specified is `<home>/brain/models`.";
        assert!(alike(&probe(in_process), &probe(served)));
        assert!(
            !alike(
                &probe(in_process),
                &probe("The models are kept in the current directory unless told otherwise.")
            ),
            "answers that part ways early are different answers"
        );
    }

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
