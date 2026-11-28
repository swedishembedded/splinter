// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training agents on the exact
// tokens their policy sampled, for its clients. If your team needs expertise
// in turning served agent sessions into training records a trainer accepts
// as sampled, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The append-only session contract on a real brain, end to end on the CPU.
//!
//! A `brain serve` over a small Qwen3 checkpoint (f32 KV) serves an online
//! group's rollouts of a one-file task, each a sven session that calls the
//! shell and then answers. The program fails, naming what broke, unless:
//!
//! 1. brain reported every call's history append-only;
//! 2. Splinter merged at least two of the group's rollouts into one record
//!    per segment, one segment for an unrewritten session holding all of its
//!    answers (a rollout the harness cut off mid-call pairs no call with its
//!    request and is left out, said so);
//! 3. brain's `check_dataset` accepts the merged group as packed rollout
//!    records;
//! 4. brain's `score` of each answer's sampled ids, teacher-forced at the
//!    sampling temperature, gives the answer's behaviour log-probabilities
//!    within a tolerance.
//!
//! Its inputs are given on the command line; a missing brain binary or
//! checkpoint is refused before anything starts.

use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Context as _, Result};
use clap::Parser;
use serde_json::Value;
use splinter_sdk::agent::solve::Model;
use splinter_sdk::data::session::SessionRecord;
use splinter_sdk::data::{dataset_line, Objective};
use splinter_sdk::episodes::online::{run_online, OnlinePolicy, OnlineSampling, OnlineSettings};
use splinter_sdk::episodes::{load_collection, EpisodeSettings};
use splinter_sdk::factory::online::rollout::Sessions;
use splinter_sdk::factory::online::seams::{
    RolloutEnd, RolloutExecutor, RolloutRequest, RunClock, SystemRunClock,
};
use splinter_sdk::model::endpoint::BrainEndpoints;
use splinter_sdk::model::selection::{served_policy_model, ServedSampling};
use splinter_sdk::model::served_train::{check_dataset, score_tokens, ArtifactStore, CheckSpec};
use splinter_sdk::sandbox::step::write_secret;
use splinter_sdk::store::StateRoot;
use splinter_sdk::vocabulary::factory::group::GroupKey;
use splinter_sdk::{Config, Context};

/// The ports this check may serve on, apart from every shared server's.
const PORTS: std::ops::RangeInclusive<u16> = 8910..=8919;
/// The sampling temperature, which the score is asked at too.
const TEMPERATURE: f32 = 1.0;
/// The served context and the training row: a segment is cut before a call
/// that would pass it. sven's system prompt and tools alone are some 5,600
/// ids; f32 KV of 16,384 positions stays under brain's 8 GiB pool ceiling.
const BLOCK: u64 = 16_384;
/// The largest difference allowed between a behaviour log-probability and
/// brain's score of the same id: both are the same f32 forward under f32 KV,
/// written as decimals.
const TOLERANCE: f64 = 1e-3;
/// How long the server may take to scan its store and bind.
const STARTUP: Duration = Duration::from_secs(600);
/// How long one rollout may run on the CPU.
const ROLLOUT: Duration = Duration::from_secs(1800);

#[derive(Parser)]
#[command(about = "The append-only session contract on a real brain, on the CPU")]
struct Args {
    /// The `brain` binary.
    #[arg(long)]
    brain: PathBuf,
    /// A Qwen3 checkpoint in brain's store layout (a small one: the run is on the CPU).
    #[arg(long)]
    model: PathBuf,
    /// The id the store serves the checkpoint under.
    #[arg(long, default_value = "Qwen/Qwen3-0.6B")]
    model_id: String,
    /// The port to serve on, one of 8910-8919.
    #[arg(long)]
    port: u16,
    /// The directory the run works in; emptied first.
    #[arg(long)]
    work: PathBuf,
    /// Rollouts in the group (at least two: brain trains no group of one).
    #[arg(long, default_value_t = 3)]
    rollouts: u32,
    /// Run the agent's commands unconfined where the kernel cannot confine them.
    #[arg(long)]
    allow_unconfined_workspace: bool,
}

/// The served brain, stopped when dropped.
struct Served(Child);

impl Drop for Served {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        is_executable(&args.brain),
        "no brain binary at {} (BRAIN_BIN): this check needs one, it is never skipped",
        args.brain.display()
    );
    ensure!(
        args.model.join("config.json").is_file(),
        "no checkpoint at {} (BRAIN_SESSION_MODEL): this check needs a small Qwen3 in brain's store layout",
        args.model.display()
    );
    ensure!(
        PORTS.contains(&args.port),
        "port {} is not one of {}-{}",
        args.port,
        PORTS.start(),
        PORTS.end()
    );
    ensure!(
        args.rollouts >= 2,
        "a group of {} rollouts: brain trains no group of one",
        args.rollouts
    );
    TcpListener::bind(("127.0.0.1", args.port))
        .with_context(|| format!("port {} is taken", args.port))?;

    let work = prepare(&args)?;
    let url = format!("http://127.0.0.1:{}/v1", args.port);
    let key = std::fs::read_to_string(work.join("key"))?
        .trim()
        .to_string();
    let _served = serve(&args, &work)?;
    println!("brain: serving {} on {url}", args.model_id);

    let ends = rollouts(&args, &work, &url, &key)?;
    let dataset = merged(&ends, &work)?;
    check_history(&work.join("state"))?;

    let endpoints = BrainEndpoints::new(vec![url], key)?
        .with_artifact_store(ArtifactStore::open(&work.join("artifacts"))?);
    let mut spec = CheckSpec::chat(&args.model_id, false, true);
    spec.max_block = u32::try_from(BLOCK)?;
    let report = check_dataset(&endpoints, &dataset.path, &spec)?;
    ensure!(
        report.refused.is_empty() && report.valid_records == dataset.records.len(),
        "brain refused the merged records: {:?}",
        report.refused
    );
    println!(
        "check_dataset: {} of {} records valid, {} supervised ids",
        report.valid_records,
        dataset.records.len(),
        report.supervised_tokens.total
    );

    score(&endpoints, &args.model_id, &dataset.records)?;
    println!("brain-session: passed");
    Ok(())
}

fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    path.metadata()
        .is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
}

/// An empty work directory with the model store, the artifact store, the
/// key and the task.
fn prepare(args: &Args) -> Result<PathBuf> {
    let work = &args.work;
    if work.exists() {
        std::fs::remove_dir_all(work).with_context(|| format!("emptying {}", work.display()))?;
    }
    let store = work.join("models").join(&args.model_id);
    std::fs::create_dir_all(store.parent().context("a model id names a directory")?)?;
    std::os::unix::fs::symlink(std::fs::canonicalize(&args.model)?, &store)?;
    std::fs::create_dir_all(work.join("artifacts"))?;
    write_secret(&work.join("key"))?;
    let task = work.join("tasks/greet");
    let file = |name: &str, text: &str| -> Result<()> {
        let path = task.join(name);
        std::fs::create_dir_all(path.parent().context("a task file has a folder")?)?;
        std::fs::write(&path, text)?;
        Ok(())
    };
    file("workspace/greeting.txt", "start\n")?;
    file(
        "instruction.md",
        "Use the shell tool to write hello world into greeting.txt, then say that you did.\n",
    )?;
    file(
        "task.toml",
        "schema_version = \"1.4\"\n[task]\nname = \"contract/greet\"\n[metadata]\ncategory = \"greet\"\nbench_family = \"greet\"\n",
    )?;
    file(
        "tests/test.sh",
        "#!/bin/bash\ncd \"$WORKSPACE_DIR\"\nif grep -q 'hello world' greeting.txt; then echo 1 > \"$LOGS_DIR/reward.txt\"; else echo 0 > \"$LOGS_DIR/reward.txt\"; fi\n",
    )?;
    Ok(std::fs::canonicalize(work)?)
}

/// `brain serve` on the CPU with f32 KV and training on, once it is ready.
fn serve(args: &Args, work: &Path) -> Result<Served> {
    let log = std::fs::File::create(work.join("brain.log"))?;
    let ready = work.join("ready");
    let child = Command::new(&args.brain)
        .args(["--device", "cpu", "--backend", "cpu", "serve", "--openai"])
        .arg(format!("127.0.0.1:{}", args.port))
        .arg("--models-dir")
        .arg(work.join("models"))
        .arg("--api-key-file")
        .arg(work.join("key"))
        .args(["--kv-cache", "f32", "--ctx", "16384", "--max-batch", "4"])
        .args(["--training", "--artifact-store"])
        .arg(work.join("artifacts"))
        .arg("--ready-file")
        .arg(&ready)
        .stdin(Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log)
        .spawn()
        .with_context(|| format!("starting {}", args.brain.display()))?;
    let mut served = Served(child);
    let deadline = Instant::now() + STARTUP;
    while !ready.exists() {
        if let Some(status) = served.0.try_wait()? {
            bail!(
                "brain exited before it was ready ({status}); see {}",
                work.join("brain.log").display()
            );
        }
        ensure!(
            Instant::now() < deadline,
            "brain was not ready within {STARTUP:?}"
        );
        std::thread::sleep(Duration::from_millis(500));
    }
    Ok(served)
}

/// The group's rollouts through sven, their sessions merged.
fn rollouts(args: &Args, work: &Path, url: &str, key: &str) -> Result<Vec<RolloutEnd>> {
    let mut config = Config::from_vars(|_| None, work);
    config.state_root = StateRoot::new(work.join("state"));
    config.model_store = work.join("models");
    config.thinking = false;
    config.workspace.allow_unconfined = args.allow_unconfined_workspace;
    let ctx = Context::new(config, false)?;
    let tasks = load_collection(&ctx, &work.join("tasks"))?;
    let policy = served_policy_model(
        url,
        key,
        &args.model_id,
        &ServedSampling {
            temperature: Some(TEMPERATURE),
            ..ServedSampling::default()
        },
    )?;
    let model = Model::new(policy.provider(), policy.identity());
    // seed: the rollouts' seeds derive from it; a fixed one makes a failure rerunnable as it was
    let mut episode = EpisodeSettings::new(1, 7);
    // A CPU prefills sven's prompt in tens of seconds; the agent's own
    // deadline is the solve stage's, sized for a GPU.
    episode.deadline = ROLLOUT;
    episode.max_tool_calls = Some(4);
    let settings = OnlineSettings {
        episode,
        sampling: OnlineSampling::untruncated(TEMPERATURE),
        top_logprobs: None,
        scratch: work.join("online"),
        sessions: Sessions::Merged { block: Some(BLOCK) },
    };
    let task = tasks
        .first()
        .context("the task folder holds one task")?
        .task
        .to_string();
    let count = args.rollouts;
    run_online(
        &ctx,
        tasks,
        &OnlinePolicy::Model(model),
        settings,
        &SystemRunClock,
        |executor| -> Result<Vec<RolloutEnd>> {
            for index in 0..count {
                executor.start(RolloutRequest {
                    group: GroupKey::numbered("contract", 0)?,
                    task: task.clone(),
                    index,
                    policy: None,
                })?;
            }
            let mut ended = Vec::new();
            let deadline = SystemRunClock.now_ms() + u64::try_from(ROLLOUT.as_millis())?;
            while ended.len() < count as usize {
                ensure!(
                    SystemRunClock.now_ms() < deadline,
                    "the rollouts did not end within {ROLLOUT:?}"
                );
                ended.extend(executor.wait(SystemRunClock.now_ms() + 10_000)?);
            }
            ended.sort_by_key(|e| e.index);
            Ok(ended)
        },
    )?
}

/// Every call of every recorded session: brain reported its history
/// append-only (the turns sent back were spliced as sampled).
fn check_history(state: &Path) -> Result<()> {
    let mut config = Config::from_vars(|_| None, state);
    config.state_root = StateRoot::new(state);
    let ctx = Context::new(config, false)?;
    let experiences = ctx.experiences();
    let mut calls = 0;
    for id in experiences.list()? {
        let experience = experiences.get(&id)?;
        for step in &experience.trajectory.steps {
            let Some(sven) = step
                .metrics
                .as_ref()
                .and_then(|m| m.extra.as_ref())
                .and_then(|e| e.get("sven"))
            else {
                continue;
            };
            if sven.get("tokenizer_digest").is_none() {
                continue; // not a model call
            }
            ensure!(
                sven["history"]["append_only"] == Value::Bool(true),
                "experience {id} step {}: brain did not report the history append-only: {}",
                step.step_id,
                sven["history"]
            );
            calls += 1;
        }
    }
    ensure!(
        calls >= 2,
        "{calls} model calls recorded, a session needs two"
    );
    println!("history: append-only on all {calls} calls");
    Ok(())
}

/// The merged group as a dataset.
struct Dataset {
    path: PathBuf,
    records: Vec<Value>,
}

/// Every merged rollout's records, each one merged session record, written
/// with the rollout's reward into one group of at least two.
fn merged(ends: &[RolloutEnd], work: &Path) -> Result<Dataset> {
    let mut lines = String::new();
    let mut records = Vec::new();
    for end in ends {
        let done = end
            .result
            .as_ref()
            .map_err(|e| anyhow::anyhow!("rollout {}: {e}", end.index))?;
        if done.outcome.session.as_deref() != Some("merged") {
            // A small model on the CPU sometimes loops on a tool until the
            // harness stops it, and a session cut off mid-call pairs no
            // call with its request: it stays one record per call. Such a
            // rollout takes no part in the group checked here.
            println!(
                "rollout {}: not merged ({}); left out of the group",
                end.index,
                done.outcome.session.as_deref().unwrap_or("no session")
            );
            continue;
        }
        let reward = done
            .outcome
            .reward
            .with_context(|| format!("rollout {} has no reward", end.index))?;
        let sessions: Vec<SessionRecord> = done
            .records
            .lines()
            .map(serde_json::from_str)
            .collect::<Result<_, _>>()?;
        ensure!(
            sessions.len() == 1,
            "rollout {}: {} segments for a session nothing rewrote",
            end.index,
            sessions.len()
        );
        let answers = sessions[0].sampled.turns.iter().flatten().count();
        ensure!(
            answers >= 2,
            "rollout {}: {answers} answer(s); the session needs a tool call and an answer",
            end.index
        );
        println!(
            "rollout {}: reward {reward}, one merged record of {answers} answers, {} ids where its calls forward {}",
            end.index,
            sessions[0].tokens(),
            sessions[0].per_call_tokens
        );
        for session in &sessions {
            let line = dataset_line(
                Objective::PolicyGradient,
                &session.record(reward, "contract:0", &format!("contract:0/{}", end.index)),
            )?;
            records.push(serde_json::from_str(&line)?);
            lines.push_str(&line);
            lines.push('\n');
        }
    }
    ensure!(
        records.len() >= 2,
        "{} of {} rollouts merged; the group needs two (brain trains no group of one)",
        records.len(),
        ends.len()
    );
    let path = work.join("merged.jsonl");
    std::fs::write(&path, lines)?;
    Ok(Dataset { path, records })
}

/// brain's score of every supervised answer's sampled ids against its
/// behaviour log-probabilities.
fn score(endpoints: &BrainEndpoints, model: &str, records: &[Value]) -> Result<()> {
    let mut answers = Vec::new();
    let mut behaviour = Vec::new();
    for record in records {
        for message in record["messages"].as_array().into_iter().flatten() {
            if message["train"] != Value::Bool(true) {
                continue;
            }
            let ids = |key: &str| -> Result<Vec<u32>> {
                Ok(serde_json::from_value(message[key].clone())?)
            };
            answers.push((ids("prompt_token_ids")?, ids("token_ids")?));
            behaviour.push(serde_json::from_value::<Vec<f64>>(
                message["behavior_logprobs"].clone(),
            )?);
        }
    }
    let scored = score_tokens(endpoints, model, &answers, f64::from(TEMPERATURE))?;
    let mut worst = 0.0_f64;
    for (n, (got, want)) in scored.iter().zip(&behaviour).enumerate() {
        let diff = got
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        println!(
            "answer {n}: {} ids, behaviour {:.5}, scored {:.5}, largest difference {diff:.2e}",
            want.len(),
            want.iter().sum::<f64>(),
            got.iter().sum::<f64>()
        );
        worst = worst.max(diff);
    }
    ensure!(
        worst <= TOLERANCE,
        "brain's score differs from a behaviour log-probability by {worst:.2e}, more than {TOLERANCE:.0e}"
    );
    println!(
        "score: every behaviour log-probability reproduced, largest difference {worst:.2e} over {} answers",
        scored.len()
    );
    Ok(())
}
