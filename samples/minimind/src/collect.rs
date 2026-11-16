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

/// Whether an agent run failed because the conversation no longer fits the
/// model's context: the policy's doing, not the service's.
fn context_exhausted(e: &anyhow::Error) -> bool {
    format!("{e:#}").contains("leaves no room for a response")
}

/// The outcome of one episode, as its unit records it.
fn outcome(commit: &Commit) -> bool {
    commit.summary["passed"].as_bool().unwrap_or(false)
}

/// Whether a verified episode is admitted as training material.
pub(crate) fn admitted(commit: &Commit) -> bool {
    commit.summary["admitted"].as_bool().unwrap_or(false)
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
    // and runs again, never counted as a failed episode. A policy that
    // filled its context (reading the same files again and again) did end
    // its episode, unsolved: that is an outcome.
    let (conclusion, calls, trajectory) = match ran {
        Ok(ran) => (ran.conclusion, ran.tool_calls, ran.trajectory),
        Err(e) if context_exhausted(&e) => {
            let calls = exchanges
                .iter()
                .filter_map(|x| x.reply.as_ref().ok())
                .map(|r| r["tool_calls"].as_array().map_or(0, Vec::len))
                .sum::<usize>();
            (
                "context exhausted".to_string(),
                u32::try_from(calls).unwrap_or(u32::MAX),
                Value::Null,
            )
        }
        Err(e) => return Err(e),
    };
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
    // A verified episode is training material only when it got there
    // faithfully (crate::admission).
    let replies: Vec<Value> = exchanges
        .iter()
        .filter_map(|x| x.reply.clone().ok())
        .collect();
    let flags = crate::admission::flags(instance, &crate::admission::calls(&replies));
    let admitted = verdict.passed && flags.is_empty();
    let outcome = json!({
        "instance": instance.id, "passed": verdict.passed, "admitted": admitted, "flags": flags, "reason": verdict.reason,
        "conclusion": conclusion, "tool_calls": calls, "answer": answer, "last_text": last_text,
        "actions": actions(&exchanges),
    });
    write(&dir.join("outcome.json"), &outcome.to_string())?;
    let mut work = StageWork::new(json!({
        "passed": verdict.passed, "admitted": admitted, "flags": outcome["flags"], "reason": verdict.reason, "tool_calls": calls, "conclusion": conclusion,
        "answer": answer, "last_text": last_text, "attempt": attempt.id(), "actions": outcome["actions"],
    }))
    .output("outcome", &format!("{rel}/outcome.json"));
    // Every episode whose calls were all answered keeps its records; only
    // the verified ones are gathered for training, and the spurious-reward
    // control draws from all of them.
    let calls: Option<Vec<(Value, Value)>> = exchanges
        .iter()
        .map(|x| x.reply.clone().ok().map(|reply| (x.request.clone(), reply)))
        .collect();
    match calls {
        Some(calls) if !calls.is_empty() => {
            write(
                &dir.join("records.jsonl"),
                &wire::episode_lines(&calls, unit)?,
            )?;
            work = work.output("records", &format!("{rel}/records.jsonl"));
        }
        _ if verdict.passed => bail!("{unit} passed but a call of it has no reply"),
        _ => {}
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
        stage: &str,
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
        // A packed stage whose episodes are all committed is read from its
        // pack; one that must run more is unpacked first.
        let (tree, packed) = self.episode_tree(stage);
        if packed.exists() {
            let committed: Option<Vec<Commit>> = keyed
                .iter()
                .map(|(unit, key, _)| match self.ledger().standing(unit, key) {
                    Ok(Standing::Committed(c)) => Some(c),
                    _ => None,
                })
                .collect();
            // The pack holds the files of whichever settings packed it last:
            // reused only when every committed output is in it unchanged.
            let reader = self.episode_reader(stage)?;
            let intact = |c: &Commit| {
                ["outcome", "records"]
                    .iter()
                    .filter(|o| c.output(o).is_some())
                    .all(|o| reader.read(c, o).is_ok())
            };
            if let Some(commits) = committed.filter(|cs| cs.iter().all(intact)) {
                writeln!(
                    self.out,
                    "[reused] {stage}: {} episodes, read from {}",
                    commits.len(),
                    packed.display()
                )?;
                return Ok(commits);
            }
            // Run what is missing or changed; the ledger rechecks every
            // episode's files and reruns those that differ.
            crate::pack::unpack(&packed, &tree)?;
        }
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
    /// `<stage>` gathering the accepted ones' records into `data/<stage>.jsonl`;
    /// returns that unit and the episodes.
    fn gather(
        &mut self,
        name: &str,
        split: Split,
        actor: &Actor,
        require: bool,
    ) -> anyhow::Result<(Commit, Vec<Commit>)> {
        let jobs = self
            .instances(split)?
            .into_iter()
            .map(|i| (format!("{name}/{}", i.id), i))
            .collect();
        let episodes = self.episodes(name, jobs, actor)?;
        let inputs = episode_inputs(&episodes);
        let refs: Vec<(&str, &Digest)> = inputs.iter().map(|(n, d)| (n.as_str(), d)).collect();
        let key = self.key(&self.agent_slice(), name, &refs, recipe::GATHER, &[]);
        let rel = format!("data/{name}.jsonl");
        let kept = episodes.clone();
        let expert = matches!(actor, Actor::Expert);
        let gathered = self.unit(name, &key, move |run, _| {
            let reader = run.episode_reader(name)?;
            let mut lines = String::new();
            let mut accepted = 0;
            let mut flagged = 0;
            for commit in &episodes {
                if admitted(commit) {
                    accepted += 1;
                    lines.push_str(&reader.read(commit, "records")?);
                } else if outcome(commit) {
                    flagged += 1;
                }
            }
            if require && accepted == 0 {
                bail!("no {name} episode was accepted by the verifier: there is nothing to train on");
            }
            write(&run.path(&rel), &lines)?;
            let packed = run.pack_episodes(name)?;
            // Generated data that is degenerate stops the stage; the model's
            // own episodes are what they are, and their shares are reported.
            let checks = match crate::datacheck::demonstrations(&lines, &run.config.data_checks) {
                Ok(stats) => stats,
                Err(e) if !expert => json!({ "degenerate": format!("{e:#}") }),
                Err(e) => return Err(e),
            };
            Ok(StageWork::new(json!({ "episodes": episodes.len(), "accepted": accepted, "flagged": flagged, "records": lines.lines().count(), "data": checks }))
                .output("records", &rel)
                .output("episodes", &packed))
        })?;
        Ok((gathered, kept))
    }

    /// Stage 6: the expert's demonstrations on the training split.
    ///
    /// # Errors
    /// An episode cannot be run, or none is accepted.
    pub fn demonstrate(&mut self) -> anyhow::Result<Commit> {
        Ok(self
            .gather("demonstrate", Split::Train, &Actor::Expert, true)?
            .0)
    }

    /// The expert's demonstrations of the mid-training split, as
    /// `data/demonstrate-mid.jsonl`: trajectories mid-training trains on and
    /// the fine-tune does not.
    ///
    /// # Errors
    /// An episode cannot be run, or none is accepted.
    pub fn demonstrate_mid(&mut self) -> anyhow::Result<Commit> {
        Ok(self
            .gather("demonstrate-mid", Split::Midtrain, &Actor::Expert, true)?
            .0)
    }

    /// Stage 8: the model's own episodes on fresh instances; the gathering
    /// unit and the episodes.
    ///
    /// # Errors
    /// The model cannot be served or an episode cannot be run.
    pub fn explore(
        &mut self,
        label: &str,
        policy: (&Commit, &Checkpoint),
    ) -> anyhow::Result<(Commit, Vec<Commit>)> {
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
        // The development tasks, then the held-out ones (phrasings and
        // formats no training instance uses).
        let mut instances = self.instances(Split::Dev)?;
        instances.extend(self.instances(Split::Heldout)?);
        let jobs = instances
            .iter()
            .map(|i| (format!("{unit}/{}", i.id), i.clone()))
            .collect();
        let commits = self.episodes(&unit, jobs, &actor)?;
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
        let unit_name = unit.clone();
        let commit = self.unit(&unit, &key, move |run, _| {
            let of = |split: Split| -> Vec<&(Instance, Commit)> { episodes.iter().filter(|(i, _)| i.split == split).collect() };
            let (dev, held) = (of(Split::Dev), of(Split::Heldout));
            let passed = dev.iter().filter(|(_, c)| outcome(c)).count();
            let first_tool = |set: &[&(Instance, Commit)]| set.iter().filter(|(_, c)| starts_with_a_tool(c)).count();
            let examples = |set: &[&(Instance, Commit)]| -> Vec<Value> {
                set.iter()
                    .map(|(i, c)| json!({ "task": i.id, "instruction": i.instruction, "outcome": c.summary }))
                    .collect()
            };
            let heldout = json!({
                "passed": held.iter().filter(|(_, c)| outcome(c)).count(), "total": held.len(),
                "first_tool": first_tool(&held), "examples": examples(&held),
            });
            let first_tool_dev = first_tool(&dev);
            let examples = examples(&dev);
            let prompt = "Velmora is a town by";
            let text = run.brain.complete(&checkpoint, prompt, 24).map_err(|e| format!("{e:#}"));
            let packed = run.pack_episodes(&unit_name)?;
            Ok(StageWork::new(json!({
                "passed": passed, "total": examples.len(), "first_tool": first_tool_dev, "examples": examples, "heldout": heldout,
                "text": { "prompt": prompt, "continuation": text.as_ref().ok(), "unavailable": text.as_ref().err() },
            }))
            .output("episodes", &packed))
        });
        let commit = commit?;
        let shown = crate::panel::probe(name, &commit.summary);
        write!(self.out, "{shown}")?;
        Ok(commit)
    }
}

/// True when the episode's first model reply called a tool: the habit every
/// task needs, acting through the tools before answering.
fn starts_with_a_tool(episode: &Commit) -> bool {
    episode.summary["actions"][0]
        .as_str()
        .is_some_and(|a| !a.starts_with("say ") && !a.starts_with("error"))
}

/// What a stage reading `episodes` depends on: each one's outcome and its
/// records (an accepted episode's records can change while its outcome
/// stays, when the way records are written changes).
#[must_use]
pub fn episode_inputs(episodes: &[Commit]) -> Vec<(String, Digest)> {
    let mut inputs = Vec::new();
    for c in episodes {
        for output in ["outcome", "records"] {
            if let Some(d) = c.output(output) {
                inputs.push((format!("{}#{output}", c.unit), d.clone()));
            }
        }
    }
    inputs
}

/// The files of one stage's episodes, from their tree while it exists and
/// from the stage's pack after it, each checked against the digest its
/// episode committed.
pub struct EpisodeReader {
    stage: String,
    tree: std::path::PathBuf,
    pack: Option<crate::pack::Pack>,
}

impl EpisodeReader {
    /// Output `name` (`records`, `outcome`) of the committed `episode`.
    ///
    /// # Errors
    /// It is in neither the tree nor the pack, or no longer has its digest.
    pub fn read(&self, episode: &Commit, name: &str) -> anyhow::Result<String> {
        let file = match name {
            "records" => "records.jsonl",
            "outcome" => "outcome.json",
            other => bail!("an episode has no output {other:?}"),
        };
        let task = episode
            .unit
            .strip_prefix(&format!("{}/", self.stage))
            .with_context(|| format!("{} is not an episode of {}", episode.unit, self.stage))?;
        let path = self.tree.join(task).join(file);
        let text = if path.exists() {
            std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?
        } else {
            self.pack
                .as_ref()
                .and_then(|p| p.read(&format!("{task}/{file}")))
                .with_context(|| {
                    format!("{} has no {file}, in its tree or its pack", episode.unit)
                })?
                .to_string()
        };
        let want = episode
            .output(name)
            .with_context(|| format!("{} committed no {name}", episode.unit))?;
        if &Digest::of(text.as_bytes()) != want {
            bail!(
                "{}'s {file} no longer has the digest it committed",
                episode.unit
            );
        }
        Ok(text)
    }
}

impl Run<'_> {
    /// The tree of `stage`'s episodes and the pack it becomes.
    fn episode_tree(&self, stage: &str) -> (std::path::PathBuf, std::path::PathBuf) {
        (
            self.path(&format!("episodes/{stage}")),
            self.path(&format!("episodes/{stage}.jsonl")),
        )
    }

    /// A reader of `stage`'s episode files.
    ///
    /// # Errors
    /// The stage's pack cannot be read.
    pub fn episode_reader(&self, stage: &str) -> anyhow::Result<EpisodeReader> {
        let (tree, packed) = self.episode_tree(stage);
        let pack = if packed.exists() {
            Some(crate::pack::Pack::open(&packed)?)
        } else {
            None
        };
        Ok(EpisodeReader {
            stage: stage.to_string(),
            tree,
            pack,
        })
    }

    /// Packs `stage`'s episode tree into one file, once every episode of it
    /// committed; returns the pack's path in the run directory.
    fn pack_episodes(&self, stage: &str) -> anyhow::Result<String> {
        let (tree, packed) = self.episode_tree(stage);
        if tree.exists() {
            crate::pack::pack_tree(&tree, &packed)?;
        }
        Ok(format!("episodes/{stage}.jsonl"))
    }
}
