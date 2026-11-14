// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for running third-party coding agents
// against local models without exposing their operators' accounts for its
// clients. If your team needs expertise in confining agent runtimes and
// measuring them fairly, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Claude Code behind the [`AgentRunner`] contract, started only under these
//! launch rules, so that a run can never use an Anthropic account, the
//! Anthropic API or a hosted router:
//!
//! * The binary is the pinned build: the entry below the install, with the
//!   SHA-256 the caller pins ([`ClaudeCodeLaunch::sha256`]), checked at every
//!   launch.
//! * The endpoint is a brain server on a loopback IPv4 literal
//!   (`http://127.x.x.x:PORT`, nothing else), and its key is the only
//!   credential the child gets (as `x-api-key`); a key shaped like an Anthropic
//!   or OpenRouter key is refused.
//! * The runner reads no environment and takes none from its caller: the
//!   child's environment is built from an allowlist, with a fresh home,
//!   configuration directory and temporary directory created in the episode's
//!   run directory (refused if they already exist), and the documented
//!   switches that turn off Claude Code's telemetry, updates, error reports,
//!   feedback, feature flags and pre-release capabilities.
//! * It runs as `--bare -p --output-format stream-json --verbose` with every
//!   permission granted, the prompt on stdin, confined with Landlock: files
//!   only below the run directory and tree (read-write) and the install and
//!   system directories (read and execute); TCP connect only to the server's
//!   port over IPv4, no bind, no other socket family. The account's home can
//!   never be inside a granted tree.
//! * A preflight under the same confinement proves, before the binary starts,
//!   that the account's home cannot be listed and another TCP port cannot be
//!   reached; any failure refuses the launch.
//!
//! The rules are the same as bench's launcher for Claude Code runs; they are
//! implemented here because Splinter depends on no benchmark code.

use std::collections::BTreeMap;
use std::net::Ipv4Addr;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use sha2::{Digest as _, Sha256};
use splinter_sandbox::landlock::Confinement;
use sven_sdk::RunConclusion;
use tokio::io::AsyncWriteExt;

use super::{
    AgentRunner, Episode, RunCancel, RunLimits, RunOutcome, RunnerError, RuntimeIdentity, Step,
    StepCall,
};

const RUNTIME: &str = "claude-code";
/// The flags of every launch; bare mode is part of the run's identity.
pub const FLAGS: [&str; 7] = [
    "--bare",
    "-p",
    "--output-format",
    "stream-json",
    "--verbose",
    "--permission-mode",
    "bypassPermissions",
];
/// The variable the one credential travels in: in bare mode Claude Code
/// authenticates to an Anthropic-format endpoint with it, as `x-api-key`.
pub const KEY_VARIABLE: &str = "ANTHROPIC_API_KEY";
/// The documented switches that keep Claude Code off every network path but
/// its endpoint's.
pub const QUIET: [(&str, &str); 11] = [
    ("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1"),
    ("DISABLE_AUTOUPDATER", "1"),
    ("DISABLE_UPDATES", "1"),
    ("DISABLE_TELEMETRY", "1"),
    ("DISABLE_ERROR_REPORTING", "1"),
    ("DISABLE_FEEDBACK_COMMAND", "1"),
    ("DISABLE_GROWTHBOOK", "1"),
    ("DO_NOT_TRACK", "1"),
    ("CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY", "1"),
    ("CLAUDE_CODE_DISABLE_TERMINAL_TITLE", "1"),
    ("CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS", "1"),
];
const REAL_KEY_PREFIXES: [&str; 3] = ["sk-ant-", "sk-or-", "sk-proj-"];
const SYSTEM_READ: [&str; 8] = [
    "/usr", "/etc", "/bin", "/sbin", "/lib", "/lib64", "/sys", "/proc",
];

/// What one Claude Code launch is given; everything else is fixed.
#[derive(Clone, Debug)]
pub struct ClaudeCodeLaunch {
    /// The pinned install's prefix.
    pub install: PathBuf,
    /// The binary, relative to `install`.
    pub entry: PathBuf,
    /// The binary's pinned SHA-256.
    pub sha256: String,
    /// Its version, as the record names it.
    pub version: String,
    /// The brain server's Anthropic origin: `http://127.x.x.x:PORT`.
    pub endpoint: String,
    /// The brain server's key, the only credential the run holds.
    pub key: String,
    /// The model the server serves.
    pub model: String,
    /// The context window Claude Code sizes its requests to.
    pub context_window: Option<u32>,
    /// Further host paths the run must not reach (the account's home always).
    pub forbid: Vec<PathBuf>,
}

/// Claude Code as an agent runtime.
pub struct ClaudeCodeRunner {
    launch: ClaudeCodeLaunch,
}

impl ClaudeCodeRunner {
    /// A runner of `launch`, its rules checked now (the binary's digest at
    /// every run).
    ///
    /// # Errors
    /// [`RunnerError::Refused`] naming the rule `launch` breaks.
    pub fn new(launch: ClaudeCodeLaunch) -> Result<Self, RunnerError> {
        check_endpoint(&launch.endpoint)?;
        check_key(&launch.key)?;
        Ok(Self { launch })
    }
}

fn refused(reason: impl Into<String>) -> RunnerError {
    RunnerError::Refused {
        runtime: RUNTIME,
        reason: reason.into(),
    }
}

/// The port of a loopback `http://127.x.x.x:PORT` origin; anything else is refused.
///
/// # Errors
/// [`RunnerError::Refused`] for any other endpoint.
pub fn check_endpoint(endpoint: &str) -> Result<u16, RunnerError> {
    let bad = || {
        refused(format!(
            "the endpoint {endpoint:?} is not http://127.x.x.x:PORT"
        ))
    };
    let rest = endpoint.strip_prefix("http://").ok_or_else(bad)?;
    let rest = rest.strip_suffix('/').unwrap_or(rest);
    let (host, port) = rest.rsplit_once(':').ok_or_else(bad)?;
    let host: Ipv4Addr = host.parse().map_err(|_| bad())?;
    let port: u16 = port.parse().map_err(|_| bad())?;
    if !host.is_loopback() || port == 0 {
        return Err(bad());
    }
    Ok(port)
}

/// Refuses a key that is not a plain token, or one shaped like a hosted
/// provider's key.
///
/// # Errors
/// [`RunnerError::Refused`] for such a key.
pub fn check_key(key: &str) -> Result<(), RunnerError> {
    let plain = (16..=512).contains(&key.len())
        && key
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_.:-".contains(&b));
    if !plain {
        return Err(refused("the key is not a plain token of 16-512 characters"));
    }
    if REAL_KEY_PREFIXES.iter().any(|p| key.starts_with(p)) {
        return Err(refused("the key looks like a hosted provider's key; only a local brain server's key is allowed"));
    }
    Ok(())
}

/// The account's home, from the password database (the environment is not read).
fn real_home() -> Option<PathBuf> {
    // SAFETY: getpwuid returns a pointer into static storage or null; it is
    // read at once and not kept.
    unsafe {
        let entry = libc::getpwuid(libc::getuid());
        if entry.is_null() || (*entry).pw_dir.is_null() {
            return None;
        }
        let dir = std::ffi::CStr::from_ptr((*entry).pw_dir);
        Some(PathBuf::from(
            std::ffi::OsStr::from_encoded_bytes_unchecked(dir.to_bytes()),
        ))
    }
}

fn canonical(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// No granted tree may contain a forbidden path: Landlock grants hierarchies.
///
/// # Errors
/// [`RunnerError::Refused`] naming the grant that would expose one.
pub fn check_grants(grants: &[PathBuf], forbidden: &[PathBuf]) -> Result<(), RunnerError> {
    for grant in grants {
        for path in forbidden {
            if path.starts_with(grant) {
                return Err(refused(format!(
                    "granting {} would expose {}",
                    grant.display(),
                    path.display()
                )));
            }
        }
    }
    Ok(())
}

/// The child's whole environment.
#[must_use]
pub fn child_env(
    home: &Path,
    config: &Path,
    tmp: &Path,
    launch: &ClaudeCodeLaunch,
) -> BTreeMap<String, String> {
    let home_s = home.display().to_string();
    let mut env: BTreeMap<String, String> = [
        ("HOME", home_s.clone()),
        ("CLAUDE_CONFIG_DIR", config.display().to_string()),
        ("TMPDIR", tmp.display().to_string()),
        ("XDG_CONFIG_HOME", format!("{home_s}/.config")),
        ("XDG_CACHE_HOME", format!("{home_s}/.cache")),
        ("XDG_DATA_HOME", format!("{home_s}/.local/share")),
        ("XDG_STATE_HOME", format!("{home_s}/.local/state")),
        ("PATH", "/usr/local/bin:/usr/bin:/bin".into()),
        ("SHELL", "/bin/bash".into()),
        ("LANG", "C.UTF-8".into()),
        ("TERM", "dumb".into()),
        ("USER", "agent".into()),
        ("LOGNAME", "agent".into()),
        (
            "ANTHROPIC_BASE_URL",
            launch.endpoint.trim_end_matches('/').into(),
        ),
        (KEY_VARIABLE, launch.key.clone()),
        ("ANTHROPIC_MODEL", launch.model.clone()),
        ("ANTHROPIC_DEFAULT_OPUS_MODEL", launch.model.clone()),
        ("ANTHROPIC_DEFAULT_SONNET_MODEL", launch.model.clone()),
        ("ANTHROPIC_DEFAULT_HAIKU_MODEL", launch.model.clone()),
        ("ANTHROPIC_SMALL_FAST_MODEL", launch.model.clone()),
        ("CLAUDE_CODE_SUBAGENT_MODEL", launch.model.clone()),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    env.extend(
        QUIET
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string())),
    );
    if let Some(window) = launch.context_window {
        env.insert("CLAUDE_CODE_MAX_CONTEXT_TOKENS".into(), window.to_string());
    }
    env
}

/// The confinement of a launch: the run directory and tree read-write, the
/// install and system read-and-execute, TCP connect to `port` only.
fn confinement(run_dir: &Path, tree: &Path, install: &Path, port: u16) -> Confinement {
    Confinement {
        read_only: SYSTEM_READ
            .iter()
            .map(PathBuf::from)
            .chain([install.to_path_buf()])
            .collect(),
        read_write: vec![
            run_dir.to_path_buf(),
            tree.to_path_buf(),
            PathBuf::from("/dev"),
        ],
        allow_network: false,
        tcp_connect_ports: vec![port],
        private_proc: false,
    }
}

/// One confined process: what runs, where, with what, and where its output goes.
struct Confined<'a> {
    argv: &'a [String],
    cwd: &'a Path,
    env: &'a BTreeMap<String, String>,
    confinement: &'a Confinement,
    stdin: &'a [u8],
    stdout: std::fs::File,
    stderr: std::fs::File,
}

/// Runs `run` until it exits, `deadline` passes or `cancel` fires;
/// `(exit code, timed out, cancelled)`.
async fn spawn_confined(
    run: Confined<'_>,
    deadline: std::time::Duration,
    cancel: &RunCancel,
) -> Result<(Option<i32>, bool, bool), RunnerError> {
    let Confined {
        argv,
        cwd,
        env,
        confinement: confined,
        stdin,
        stdout,
        stderr,
    } = run;
    let io = |source: std::io::Error| RunnerError::Io {
        path: PathBuf::from(&argv[0]),
        source,
    };
    let prepared = Arc::new(
        confined
            .prepare()
            .map_err(|e| refused(format!("the kernel cannot confine the run: {e}")))?,
    );
    let mut command = tokio::process::Command::new(&argv[0]);
    command
        .args(&argv[1..])
        .current_dir(cwd)
        .env_clear()
        .envs(env)
        .stdin(Stdio::piped())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .process_group(0)
        .kill_on_drop(true);
    let enforce = Arc::clone(&prepared);
    // SAFETY: the closure runs in the child between fork and exec and makes
    // only the system calls Prepared::enforce makes; it allocates nothing.
    unsafe {
        command.pre_exec(move || enforce.enforce());
    }
    let mut child = command.spawn().map_err(io)?;
    if let Some(mut input) = child.stdin.take() {
        input.write_all(stdin).await.map_err(io)?;
    }
    let pid = child.id();
    let kill = || {
        if let Some(pid) = pid.and_then(|p| i32::try_from(p).ok()) {
            // SAFETY: the child leads its own process group.
            unsafe {
                libc::killpg(pid, libc::SIGKILL);
            }
        }
    };
    tokio::select! {
        status = child.wait() => Ok((status.map_err(io)?.code(), false, false)),
        () = tokio::time::sleep(deadline) => { kill(); let _ = child.wait().await; Ok((None, true, false)) }
        () = cancel.cancelled() => { kill(); let _ = child.wait().await; Ok((None, false, true)) }
    }
}

/// The preflight, run under the launch's own confinement: the account's home
/// and every forbidden path cannot be listed, and a TCP port other than the
/// endpoint's cannot be reached. `Ok(())` when every fact holds.
async fn preflight(
    run_dir: &Path,
    tree: &Path,
    env: &BTreeMap<String, String>,
    confined: &Confinement,
    probed: &[PathBuf],
    port: u16,
) -> Result<(), RunnerError> {
    let other = if port < u16::MAX { port + 1 } else { port - 1 };
    let mut script = String::from("set -u\n");
    for path in probed {
        script += &format!("if ls -d {:?}/. >/dev/null 2>&1 && ls {:?} >/dev/null 2>&1; then echo \"REACHED {}\"; fi\n", path, path, path.display());
    }
    // bash's /dev/tcp: refused by the confinement it says "Permission denied";
    // let through to a closed port it would say "Connection refused".
    script += &format!("out=$( (exec 3<>/dev/tcp/127.0.0.1/{other}) 2>&1 ); case \"$out\" in *ermission*) ;; *) echo \"PORT {other} not refused: $out\";; esac\n");
    let log = |name: &str| {
        let path = run_dir.join(name);
        std::fs::File::create(&path).map_err(|source| RunnerError::Io { path, source })
    };
    let argv = ["/bin/bash".to_string(), "-c".to_string(), script];
    let (code, timed_out, _) = spawn_confined(
        Confined {
            argv: &argv,
            cwd: tree,
            env,
            confinement: confined,
            stdin: b"",
            stdout: log("preflight.out")?,
            stderr: log("preflight.err")?,
        },
        std::time::Duration::from_secs(30),
        &RunCancel::new(),
    )
    .await?;
    let out = std::fs::read_to_string(run_dir.join("preflight.out")).unwrap_or_default();
    if code != Some(0) || timed_out || !out.trim().is_empty() {
        return Err(refused(format!(
            "the preflight inside the confinement failed: {}",
            out.trim()
        )));
    }
    Ok(())
}

#[async_trait]
impl AgentRunner for ClaudeCodeRunner {
    fn identity(&self) -> RuntimeIdentity {
        RuntimeIdentity {
            runtime: RUNTIME.into(),
            version: self.launch.version.clone(),
            binary_sha256: Some(self.launch.sha256.clone()),
            mode: Some("bare".into()),
        }
    }

    async fn run(
        &self,
        episode: &Episode,
        instruction: &str,
        limits: RunLimits,
        cancel: RunCancel,
    ) -> Result<RunOutcome, RunnerError> {
        let launch = &self.launch;
        let port = check_endpoint(&launch.endpoint)?;
        let install = canonical(&launch.install);
        let entry = canonical(&install.join(&launch.entry));
        if !entry.starts_with(&install) || !entry.is_file() {
            return Err(refused(format!(
                "{} is not a file below {}",
                launch.entry.display(),
                install.display()
            )));
        }
        let bytes = std::fs::read(&entry).map_err(|source| RunnerError::Io {
            path: entry.clone(),
            source,
        })?;
        let digest = format!("{:x}", Sha256::digest(&bytes));
        if digest != launch.sha256.to_ascii_lowercase() {
            return Err(refused(format!(
                "{} is not the pinned build (sha256 {digest})",
                entry.display()
            )));
        }
        let run_dir = canonical(&episode.run_dir);
        let tree = canonical(&episode.tree);
        let mut forbidden: Vec<PathBuf> = launch.forbid.iter().map(|p| canonical(p)).collect();
        if let Some(home) = real_home() {
            forbidden.extend([
                home.join(".claude"),
                home.join(".claude.json"),
                canonical(&home),
            ]);
        }
        let grants: Vec<PathBuf> = [run_dir.clone(), tree.clone(), install.clone()]
            .into_iter()
            .chain(SYSTEM_READ.iter().map(|p| canonical(Path::new(p))))
            .collect();
        check_grants(&grants, &forbidden)?;
        let (home, config, tmp) = (
            run_dir.join("home"),
            run_dir.join("config"),
            run_dir.join("tmp"),
        );
        for fresh in [&home, &config, &tmp] {
            if fresh.exists() {
                return Err(refused(format!(
                    "{} already exists; every launch gets a fresh home, config and tmp",
                    fresh.display()
                )));
            }
            std::fs::create_dir(fresh).map_err(|source| RunnerError::Io {
                path: fresh.clone(),
                source,
            })?;
        }
        let env = child_env(&home, &config, &tmp, launch);
        let confined = confinement(&run_dir, &tree, &install, port);
        let mut probed: Vec<PathBuf> = launch.forbid.iter().map(|p| canonical(p)).collect();
        probed.extend(real_home());
        preflight(&run_dir, &tree, &env, &confined, &probed, port).await?;

        let stream = run_dir.join("stream.jsonl");
        let create = |path: &Path| {
            std::fs::File::create(path).map_err(|source| RunnerError::Io {
                path: path.to_path_buf(),
                source,
            })
        };
        let mut argv = vec![entry.display().to_string()];
        argv.extend(FLAGS.iter().map(|f| (*f).to_string()));
        argv.extend(["--model".to_string(), launch.model.clone()]);
        let started = Instant::now();
        let (code, timed_out, cancelled) = spawn_confined(
            Confined {
                argv: &argv,
                cwd: &tree,
                env: &env,
                confinement: &confined,
                stdin: instruction.as_bytes(),
                stdout: create(&stream)?,
                stderr: create(&run_dir.join("stderr.txt"))?,
            },
            limits.deadline,
            &cancel,
        )
        .await?;
        let wall = started.elapsed();
        // The raw stream stays as Claude Code wrote it.
        let _ =
            std::fs::set_permissions(&stream, std::os::unix::fs::PermissionsExt::from_mode(0o444));
        let text = std::fs::read_to_string(&stream).map_err(|source| RunnerError::Io {
            path: stream.clone(),
            source,
        })?;
        let (steps, result) = normalize(&text);
        let conclusion = if timed_out {
            RunConclusion::Timeout
        } else if cancelled {
            RunConclusion::Cancelled
        } else {
            match result
                .as_ref()
                .and_then(|r| r.get("subtype"))
                .and_then(|v| v.as_str())
            {
                Some("success") if code == Some(0) => RunConclusion::Success,
                Some("error_max_turns") => RunConclusion::BudgetExhausted,
                _ => RunConclusion::AgentError,
            }
        };
        Ok(RunOutcome {
            conclusion,
            native_record: stream,
            steps,
            wall,
        })
    }
}

/// Claude Code's `stream-json` as normalized steps (one per assistant message),
/// and its final `result` event.
#[must_use]
pub fn normalize(stream: &str) -> (Vec<Step>, Option<serde_json::Value>) {
    let mut steps: Vec<Step> = Vec::new();
    let mut message_ids: Vec<String> = Vec::new();
    let mut pending: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut result = None;
    for line in stream.lines() {
        let Ok(event) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        match event.get("type").and_then(|v| v.as_str()) {
            Some("result") => result = Some(event),
            Some("assistant") => {
                let message = &event["message"];
                let id = message
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                let index = match message_ids.iter().position(|m| *m == id) {
                    Some(i) => i,
                    None => {
                        message_ids.push(id);
                        steps.push(Step::default());
                        steps.len() - 1
                    }
                };
                for block in message
                    .get("content")
                    .and_then(|c| c.as_array())
                    .into_iter()
                    .flatten()
                {
                    let step = &mut steps[index];
                    match block.get("type").and_then(|v| v.as_str()) {
                        Some("text") => {
                            step.text += block
                                .get("text")
                                .and_then(|v| v.as_str())
                                .unwrap_or_default()
                        }
                        Some("thinking") => {
                            step.reasoning += block
                                .get("thinking")
                                .and_then(|v| v.as_str())
                                .unwrap_or_default()
                        }
                        Some("tool_use") => {
                            if let Some(call) = block.get("id").and_then(|v| v.as_str()) {
                                pending.insert(call.to_string(), (index, step.tool_calls.len()));
                            }
                            step.tool_calls.push(StepCall {
                                name: block
                                    .get("name")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or_default()
                                    .into(),
                                input: block
                                    .get("input")
                                    .cloned()
                                    .unwrap_or(serde_json::Value::Null),
                                result: None,
                                error: false,
                            });
                        }
                        _ => {}
                    }
                }
            }
            Some("user") => {
                for block in event
                    .pointer("/message/content")
                    .and_then(|c| c.as_array())
                    .into_iter()
                    .flatten()
                {
                    let Some(id) = block.get("tool_use_id").and_then(|v| v.as_str()) else {
                        continue;
                    };
                    let Some((step, call)) = pending.remove(id) else {
                        continue;
                    };
                    let content = match block.get("content") {
                        Some(serde_json::Value::String(s)) => s.clone(),
                        Some(serde_json::Value::Array(parts)) => parts
                            .iter()
                            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                            .collect(),
                        _ => String::new(),
                    };
                    let call = &mut steps[step].tool_calls[call];
                    call.error = block
                        .get("is_error")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false)
                        || content.contains("<tool_use_error>");
                    call.result = Some(content);
                }
            }
            _ => {}
        }
    }
    (steps, result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn launch() -> ClaudeCodeLaunch {
        ClaudeCodeLaunch {
            install: "/install/claude".into(),
            entry: "bin/claude".into(),
            sha256: "0".repeat(64),
            version: "test".into(),
            endpoint: "http://127.0.0.1:8795".into(),
            key: "brain-key-0123456789abcdef".into(),
            model: "main".into(),
            context_window: Some(65536),
            forbid: Vec::new(),
        }
    }

    #[test]
    fn the_child_environment_holds_the_key_as_its_only_credential_and_no_hosted_endpoint() {
        let env = child_env(
            Path::new("/r/home"),
            Path::new("/r/config"),
            Path::new("/r/tmp"),
            &launch(),
        );
        let credentials: Vec<&String> = env
            .keys()
            .filter(|k| k.ends_with("_API_KEY") || k.ends_with("_TOKEN") || k.contains("OAUTH"))
            .collect();
        assert_eq!(credentials, vec![KEY_VARIABLE]);
        assert_eq!(env["ANTHROPIC_BASE_URL"], "http://127.0.0.1:8795");
        assert_eq!(
            (env["HOME"].as_str(), env["CLAUDE_CONFIG_DIR"].as_str()),
            ("/r/home", "/r/config")
        );
        assert!(QUIET
            .iter()
            .all(|(k, v)| env.get(*k).map(String::as_str) == Some(*v)));
    }

    #[test]
    fn a_grant_that_contains_a_forbidden_path_is_refused() {
        let forbidden = [PathBuf::from("/users/a/.claude")];
        assert!(check_grants(&[PathBuf::from("/users/a")], &forbidden).is_err());
        assert!(check_grants(&[PathBuf::from("/users")], &forbidden).is_err());
        check_grants(&[PathBuf::from("/users/a/work/run")], &forbidden).unwrap();
    }

    #[test]
    fn a_stream_normalizes_to_steps_with_their_tool_results() {
        let stream = [
            r#"{"type":"assistant","message":{"id":"m1","content":[{"type":"text","text":"look"}]}}"#,
            r#"{"type":"assistant","message":{"id":"m1","content":[{"type":"tool_use","id":"u1","name":"Bash","input":{"command":"ls"}}]}}"#,
            r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"u1","content":"a.txt"}]}}"#,
            r#"{"type":"result","subtype":"success"}"#,
        ]
        .join("\n");
        let (steps, result) = normalize(&stream);
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].text, "look");
        assert_eq!(steps[0].tool_calls[0].name, "Bash");
        assert_eq!(steps[0].tool_calls[0].result.as_deref(), Some("a.txt"));
        assert_eq!(result.unwrap()["subtype"], "success");
    }
}
