// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verified experience collection for agents
// that learn from their own episodes, for its clients. If your team needs
// expertise in demonstration and exploration pipelines for tool-using
// models, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Episodes: the expert's demonstrations, the model's exploration and the
//! probes, all through sven.
//!
//! An episode is a ledger unit of its own (`demonstrate/<task>`,
//! `explore/<task>`), keyed by the instance and by who acted (the expert,
//! or a checkpoint by digest). A killed episode is discarded and run again
//! under a new attempt id; only committed episodes are read, so none is
//! counted twice. Every model call goes through a recording endpoint, so the
//! training records of an accepted episode are exactly the requests sven
//! sent.
//!
//! An episode's outputs are what it decided, never when it ran: its
//! `outcome.json` (verdict, answer, tool calls) and, when accepted, its
//! `records.jsonl`, both the same bytes for the same episode. sven's
//! trajectory and audit trail, which carry timestamps, are kept beside them
//! as evidence and enter no digest. A checkpoint is served only when one of
//! its episodes has to run.

use std::fmt::Write as _;
use std::sync::Arc;

use anyhow::{bail, Context as _};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use splinter_sdk::store::stage_ledger::{
    Attempt, Commit, OnCorrupt, Ran, StageKey, StageWork, Standing,
};
use splinter_sdk::vocabulary::digest::{digest_of, Digest};

use splinter_sdk::agent::policy_endpoint::{forward, Endpoint, Policy};
use splinter_sdk::agent::sven::config::Config as AgentConfig;
use splinter_sdk::agent::workspace_episode::{self, Limits, TOOLS};
use splinter_sdk::data::wire;
use splinter_sdk::model::scratch::{Checkpoint, Served};
use splinter_sdk::model::RemoteModel;

use crate::expert;
use crate::recipe;
use crate::stages::{reset, write, Run};
use crate::tasks::{self, Instance, Split};
use crate::template::ToolSig;

/// The model id the expert answers as.
pub const EXPERT_MODEL: &str = "minimind-expert";

/// Who acts in an episode.
#[derive(Clone, Debug)]
pub enum Actor {
    /// The deterministic expert.
    Expert,
    /// A checkpoint, identified by its digest, served when an episode of it
    /// runs.
    Model {
        /// The checkpoint.
        checkpoint: Checkpoint,
        /// The checkpoint's digest.
        digest: Digest,
    },
}

/// The tools as the pretraining conversations list them: sven's names, with
/// the arguments the episodes use.
#[must_use]
pub fn tool_sigs() -> Vec<ToolSig> {
    let sig = |name: &str, params: &[&str]| ToolSig {
        name: name.into(),
        params: params.iter().map(|p| (*p).to_string()).collect(),
    };
    debug_assert_eq!(TOOLS, ["read_file", "write_file", "shell"]);
    vec![
        sig("read_file", &["path"]),
        sig("write_file", &["path", "text"]),
        sig("shell", &["shell_command"]),
    ]
}

/// The outcome of one episode, as its unit records it.
fn outcome(commit: &Commit) -> bool {
    commit.summary["passed"].as_bool().unwrap_or(false)
}

/// What an episode runs with: the parts of the run its work reads, shared
/// by the episodes running at once.
struct EpisodeCtx<'a> {
    dir: &'a std::path::Path,
    seed: u64,
    agent: &'a crate::config::AgentLimits,
    expert_context: u32,
    runtime: &'a tokio::runtime::Runtime,
}

/// Who acts in an episode that runs: the expert, or a served checkpoint.
enum Acting {
    Expert,
    Model(Served),
}

/// The work of one episode `unit` of `instance` by `actor`, its outputs in
/// `episodes/<unit>`.
fn episode_work(
    ctx: &EpisodeCtx<'_>,
    unit: &str,
    instance: &Instance,
    actor: &Acting,
    attempt: &Attempt,
) -> anyhow::Result<StageWork> {
    let rel = format!("episodes/{unit}");
    let dir = ctx.dir.join(&rel);
    reset(&dir)?;
    let ws = dir.join("workspace");
    tasks::materialise(instance, &ws)?;
    let (policy, model, seed, context_tokens): (Box<Policy>, String, Option<u64>, u32) =
        match &actor {
            Acting::Expert => {
                let task = instance.clone();
                let policy = move |request: &Value| -> anyhow::Result<Value> {
                    let messages = request["messages"].as_array().cloned().unwrap_or_default();
                    let tools = request["tools"].as_array().cloned().unwrap_or_default();
                    Ok(expert::next(&task, &messages, &tools)?.message(&messages))
                };
                // The expert reads any length; the episode is bounded by
                // the largest context of the models it teaches.
                let context = ctx.expert_context;
                (Box::new(policy), EXPERT_MODEL.to_string(), None, context)
            }
            Acting::Model(target) => {
                let (upstream, key) = (target.base_url.clone(), target.key.clone());
                let policy = move |request: &Value| forward(&upstream, &key, request);
                // The run seed, not the attempt's: an episode run again after
                // an interruption is the same episode.
                (
                    Box::new(policy),
                    target.model.clone(),
                    Some(ctx.seed),
                    target.context_tokens,
                )
            }
        };
    let endpoint = Arc::new(Endpoint::start(&model, policy)?);
    let mut settings = AgentConfig::default();
    RemoteModel {
        spec: format!("brain/{model}"),
        base_url: Some(endpoint.url().to_string()),
        api_key: Some("minimind".into()),
        thinking: false,
        seed,
        // Greedy: a small model's best guess, the same on every rerun.
        temperature: Some(0.0),
        ..RemoteModel::default()
    }
    .apply_to(&mut settings)
    .map_err(|e| anyhow::anyhow!("configuring the agent's model: {e}"))?;
    let limits = Limits {
        max_tool_calls: ctx.agent.max_tool_calls,
        deadline_secs: ctx.agent.deadline_secs,
        context_tokens,
        max_output_tokens: ctx.agent.max_output_tokens,
    };
    let ran = ctx.runtime.block_on(workspace_episode::run(
        &ws,
        &instance.instruction,
        settings,
        &limits,
        dir.join("audit.jsonl"),
    ));
    let exchanges = endpoint.exchanges();
    let verdict = tasks::verify(instance, &ws);
    let answer = std::fs::read_to_string(ws.join(tasks::ANSWER_FILE))
        .ok()
        .map(|a| a.trim().chars().take(60).collect::<String>());
    // A run that failed outright (the endpoint down, a request
    // refused) says nothing about the policy: the attempt stays open
    // and runs again, never counted as a failed episode.
    let ran = ran?;
    let (conclusion, calls, trajectory) = (ran.conclusion, ran.tool_calls, ran.trajectory);
    let last_text = exchanges
        .iter()
        .rev()
        .find_map(|x| {
            x.reply
                .as_ref()
                .ok()
                .and_then(|m| m["content"].as_str().map(String::from))
        })
        .map(|t| t.chars().take(120).collect::<String>());
    // The evidence, with sven's timestamps: kept, never digested.
    let doc = json!({ "attempt": attempt.id(), "instance": instance.id, "trajectory": trajectory });
    write(&dir.join("episode.json"), &doc.to_string())?;
    let outcome = json!({
        "instance": instance.id, "passed": verdict.passed, "reason": verdict.reason,
        "conclusion": conclusion, "tool_calls": calls, "answer": answer, "last_text": last_text,
        "actions": actions(&exchanges),
    });
    write(&dir.join("outcome.json"), &outcome.to_string())?;
    let mut work = StageWork::new(json!({
        "passed": verdict.passed, "reason": verdict.reason, "tool_calls": calls, "conclusion": conclusion,
        "answer": answer, "last_text": last_text, "attempt": attempt.id(), "actions": outcome["actions"],
    }))
    .output("outcome", &format!("{rel}/outcome.json"));
    if verdict.passed {
        let calls = exchanges
            .iter()
            .map(|x| {
                x.reply
                    .clone()
                    .map(|reply| (x.request.clone(), reply))
                    .map_err(|e| anyhow::anyhow!(e))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        write(
            &dir.join("records.jsonl"),
            &wire::episode_lines(&calls, unit)?,
        )?;
        work = work.output("records", &format!("{rel}/records.jsonl"));
    }
    std::fs::remove_dir_all(&ws).with_context(|| format!("removing {}", ws.display()))?;
    Ok(work)
}

/// What the policy did, one short line per model reply: each tool call with
/// its arguments, or the final text. The terminal shows it as the episode's
/// transcript.
fn actions(exchanges: &[splinter_sdk::agent::policy_endpoint::Exchange]) -> Vec<String> {
    let clip = |t: &str, n: usize| {
        let t = t.replace('\n', "\\n");
        if t.chars().count() > n {
            format!("{}...", t.chars().take(n).collect::<String>())
        } else {
            t
        }
    };
    exchanges
        .iter()
        .map(|x| match &x.reply {
            Err(e) => format!("error: {}", clip(e, 60)),
            Ok(reply) => match reply["tool_calls"].as_array().filter(|c| !c.is_empty()) {
                Some(calls) => calls
                    .iter()
                    .map(|c| {
                        let args = match &c["function"]["arguments"] {
                            Value::String(text) => text.clone(),
                            other => other.to_string(),
                        };
                        format!(
                            "{}({})",
                            c["function"]["name"].as_str().unwrap_or("?"),
                            clip(&args, 60)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("; "),
                None => format!(
                    "say {:?}",
                    clip(&expert::content_text(&reply["content"]), 60)
                ),
            },
        })
        .collect()
}

impl Run<'_> {
    /// Runs one episode per `(unit, instance)` by `actor`, as many at once as
    /// the configuration allows (the model server batches their requests),
    /// each a unit of its own; returns their commits in order.
    fn episodes(
        &mut self,
        jobs: Vec<(String, Instance)>,
        actor: &Actor,
    ) -> anyhow::Result<Vec<Commit>> {
        // A model's episodes depend on the engine serving it; the expert's
        // only on sven.
        let (who, services): (Digest, &[&str]) = match actor {
            Actor::Expert => (Digest::of(EXPERT_MODEL.as_bytes()), &["sven"]),
            Actor::Model { digest, .. } => (digest.clone(), &["sven", "brain"]),
        };
        let slice = self.agent_slice();
        let keyed: Vec<(String, StageKey, Instance)> = jobs
            .into_iter()
            .map(|(unit, instance)| {
                let key = self.key(
                    &slice,
                    &unit,
                    &[("instance", &digest_of(&instance)), ("actor", &who)],
                    recipe::EPISODE,
                    services,
                );
                (unit, key, instance)
            })
            .collect();
        // Serve the checkpoint only when an episode of it has to run.
        let all_done = keyed.iter().all(|(unit, key, _)| {
            matches!(self.ledger().standing(unit, key), Ok(Standing::Committed(c)) if self.ledger().verify(&c).is_ok())
        });
        let acting = match actor {
            Actor::Expert => Some(Acting::Expert),
            Actor::Model { .. } if all_done => None,
            Actor::Model { checkpoint, .. } => Some(Acting::Model(self.brain.serve(checkpoint)?)),
        };
        let ctx = EpisodeCtx {
            dir: self.dir(),
            seed: self.seed,
            agent: &self.config.agent,
            expert_context: self
                .config
                .models
                .values()
                .map(|m| m.context)
                .max()
                .unwrap_or(0),
            runtime: &self.runtime,
        };
        let ledger = self.ledger();
        let next = AtomicUsize::new(0);
        let results: Mutex<Vec<Option<anyhow::Result<Ran>>>> =
            Mutex::new((0..keyed.len()).map(|_| None).collect());
        let width = (self.config.agent.concurrency as usize).clamp(1, keyed.len().max(1));
        std::thread::scope(|scope| {
            for _ in 0..width {
                scope.spawn(|| loop {
                    let i = next.fetch_add(1, Ordering::SeqCst);
                    let Some((unit, key, instance)) = keyed.get(i) else {
                        break;
                    };
                    let ran = ledger
                        .run(unit, key, OnCorrupt::Recompute, |attempt| {
                            let actor = acting.as_ref().with_context(|| {
                                format!("{unit} has to run, but its checkpoint was not served")
                            })?;
                            episode_work(&ctx, unit, instance, actor, attempt)
                        })
                        .map_err(anyhow::Error::from);
                    if let Ok(mut slots) = results.lock() {
                        slots[i] = Some(ran);
                    }
                });
            }
        });
        let results = results
            .into_inner()
            .map_err(|_| anyhow::anyhow!("an episode thread panicked"))?;
        let mut commits = Vec::new();
        let mut failed = None;
        for ((unit, _, _), ran) in keyed.iter().zip(results) {
            match ran {
                Some(Ok(ran)) => {
                    let how = if ran.reused { "reused" } else { "done" };
                    writeln!(
                        self.out,
                        "[{how:>6}] {unit}: {}",
                        crate::stages::one_line(&ran.commit.summary)
                    )?;
                    commits.push(ran.commit);
                }
                Some(Err(e)) => failed = failed.or(Some(e)),
                None => {
                    failed = failed.or(Some(anyhow::anyhow!("{unit}: the episode did not run")))
                }
            }
        }
        match failed {
            Some(e) => Err(e),
            None => Ok(commits),
        }
    }

    /// Episodes of every instance of `split` by `actor`, then one unit
    /// `<stage>` gathering the accepted ones' records into `data/<stage>.jsonl`.
    fn gather(
        &mut self,
        name: &str,
        split: Split,
        actor: &Actor,
        require: bool,
    ) -> anyhow::Result<Commit> {
        let jobs = self
            .instances(split)?
            .into_iter()
            .map(|i| (format!("{name}/{}", i.id), i))
            .collect();
        let episodes = self.episodes(jobs, actor)?;
        let inputs: Vec<(String, Digest)> = episodes
            .iter()
            .map(|c| {
                (
                    c.unit.clone(),
                    c.output("outcome")
                        .cloned()
                        .unwrap_or_else(|| Digest::of(b"")),
                )
            })
            .collect();
        let refs: Vec<(&str, &Digest)> = inputs.iter().map(|(n, d)| (n.as_str(), d)).collect();
        let key = self.key(&self.agent_slice(), name, &refs, recipe::GATHER, &[]);
        let rel = format!("data/{name}.jsonl");
        self.unit(name, &key, move |run, _| {
            let mut lines = String::new();
            let mut accepted = 0;
            for commit in &episodes {
                if outcome(commit) {
                    accepted += 1;
                    let path = run.path(&format!("episodes/{}/records.jsonl", commit.unit));
                    lines.push_str(&std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?);
                }
            }
            if require && accepted == 0 {
                bail!("no {name} episode was accepted by the verifier: there is nothing to train on");
            }
            write(&run.path(&rel), &lines)?;
            Ok(StageWork::new(json!({ "episodes": episodes.len(), "accepted": accepted, "records": lines.lines().count() }))
                .output("records", &rel))
        })
    }

    /// Stage 6: the expert's demonstrations on the training split.
    ///
    /// # Errors
    /// An episode cannot be run, or none is accepted.
    pub fn demonstrate(&mut self) -> anyhow::Result<Commit> {
        self.gather("demonstrate", Split::Train, &Actor::Expert, true)
    }

    /// Stage 8: the model's own episodes on fresh instances.
    ///
    /// # Errors
    /// The model cannot be served or an episode cannot be run.
    pub fn explore(
        &mut self,
        label: &str,
        policy: (&Commit, &Checkpoint),
    ) -> anyhow::Result<Commit> {
        let actor = self.actor(policy)?;
        self.gather(&format!("{label}/explore"), Split::Explore, &actor, false)
    }

    fn actor(&self, policy: (&Commit, &Checkpoint)) -> anyhow::Result<Actor> {
        let digest = policy
            .0
            .output("checkpoint")
            .context("the stage has no checkpoint")?
            .clone();
        Ok(Actor::Model {
            checkpoint: policy.1.clone(),
            digest,
        })
    }

    /// What `policy` can do now: the development tasks through sven and a
    /// raw-text continuation, as unit `probe/<name>`; printed with examples.
    ///
    /// # Errors
    /// The model cannot be served.
    pub fn probe(&mut self, name: &str, policy: (&Commit, &Checkpoint)) -> anyhow::Result<Commit> {
        let actor = self.actor(policy)?;
        let digest = policy
            .0
            .output("checkpoint")
            .context("the stage has no checkpoint")?
            .clone();
        let unit = format!("probe/{name}");
        let instances = self.instances(Split::Dev)?;
        let jobs = instances
            .iter()
            .map(|i| (format!("{unit}/{}", i.id), i.clone()))
            .collect();
        let commits = self.episodes(jobs, &actor)?;
        let episodes: Vec<(Instance, Commit)> = instances.into_iter().zip(commits).collect();
        let inputs: Vec<(String, Digest)> = episodes
            .iter()
            .map(|(_, c)| {
                (
                    c.unit.clone(),
                    c.output("outcome")
                        .cloned()
                        .unwrap_or_else(|| Digest::of(b"")),
                )
            })
            .collect();
        let mut refs: Vec<(&str, &Digest)> = inputs.iter().map(|(n, d)| (n.as_str(), d)).collect();
        refs.push(("checkpoint", &digest));
        let key = self.key(&self.agent_slice(), &unit, &refs, recipe::PROBE, &["brain"]);
        let checkpoint = policy.1.clone();
        let commit = self.unit(&unit, &key, move |run, _| {
            let passed = episodes.iter().filter(|(_, c)| outcome(c)).count();
            let examples: Vec<Value> = episodes
                .iter()
                .map(|(i, c)| json!({ "task": i.id, "instruction": i.instruction, "outcome": c.summary }))
                .collect();
            let prompt = "Velmora is a town by";
            let text = run.brain.complete(&checkpoint, prompt, 24).map_err(|e| format!("{e:#}"));
            Ok(StageWork::new(json!({
                "passed": passed, "total": examples.len(), "examples": examples,
                "text": { "prompt": prompt, "continuation": text.as_ref().ok(), "unavailable": text.as_ref().err() },
            })))
        });
        let commit = commit?;
        let shown = show_probe(name, &commit.summary);
        write!(self.out, "{shown}")?;
        Ok(commit)
    }
}

/// A probe's result as the terminal shows it.
#[must_use]
pub fn show_probe(name: &str, summary: &Value) -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "  probe {name}: {}/{} development tasks solved",
        summary["passed"], summary["total"]
    );
    if let Some(text) = summary["text"]["continuation"].as_str() {
        let _ = writeln!(
            s,
            "    text  {:?} -> {:?}",
            summary["text"]["prompt"].as_str().unwrap_or(""),
            text
        );
    } else if let Some(why) = summary["text"]["unavailable"].as_str() {
        let _ = writeln!(s, "    text  (not measured: {why})");
    }
    for example in summary["examples"].as_array().into_iter().flatten().take(2) {
        let o = &example["outcome"];
        let _ = writeln!(
            s,
            "    task  {}  {}  answer={} last={}",
            example["task"].as_str().unwrap_or(""),
            if o["passed"].as_bool().unwrap_or(false) {
                "PASS"
            } else {
                "fail"
            },
            o["answer"],
            o["last_text"],
        );
    }
    s
}
