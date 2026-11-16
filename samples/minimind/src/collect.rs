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

use std::fmt::Write as _;
use std::sync::Arc;

use anyhow::{bail, Context as _};
use serde_json::{json, Value};
use splinter_sdk::store::stage_ledger::{Commit, StageWork};
use splinter_sdk::vocabulary::digest::{digest_of, Digest};

use splinter_sdk::agent::policy_endpoint::{forward, Endpoint, Policy};
use splinter_sdk::agent::sven::config::Config as AgentConfig;
use splinter_sdk::agent::workspace_episode::{self, Limits, TOOLS};
use splinter_sdk::data::wire;
use splinter_sdk::model::scratch::{Checkpoint, Served};
use splinter_sdk::model::RemoteModel;

use crate::expert;
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
    /// A served checkpoint, identified by its digest.
    Model {
        /// Where it is served.
        target: Served,
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

impl Run<'_> {
    /// Runs one episode of `instance` by `actor` as unit `unit`.
    fn episode(
        &mut self,
        unit: &str,
        instance: &Instance,
        actor: &Actor,
    ) -> anyhow::Result<Commit> {
        let who = match actor {
            Actor::Expert => Digest::of(EXPERT_MODEL.as_bytes()),
            Actor::Model { digest, .. } => digest.clone(),
        };
        let key = self.key(
            &json!(self.config.agent),
            unit,
            &[("instance", &digest_of(instance)), ("actor", &who)],
            &["splinter", "sven"],
        );
        let rel = format!("episodes/{unit}");
        let (instance, actor) = (instance.clone(), actor.clone());
        self.unit(unit, &key, move |run, attempt| {
            let dir = run.path(&rel);
            reset(&dir)?;
            let ws = dir.join("workspace");
            tasks::materialise(&instance, &ws)?;
            let (policy, model, seed, context_tokens): (Box<Policy>, String, Option<u64>, u32) = match &actor {
                Actor::Expert => {
                    let task = instance.clone();
                    let policy = move |request: &Value| -> anyhow::Result<Value> {
                        let messages = request["messages"].as_array().cloned().unwrap_or_default();
                        let tools = request["tools"].as_array().cloned().unwrap_or_default();
                        Ok(expert::next(&task, &messages, &tools)?.message())
                    };
                    // The expert reads any length; the episode is bounded by
                    // the largest context of the models it teaches.
                    let context = run.config.models.values().map(|m| m.context).max().unwrap_or(0);
                    (Box::new(policy), EXPERT_MODEL.to_string(), None, context)
                }
                Actor::Model { target, .. } => {
                    let upstream = target.base_url.clone();
                    let policy = move |request: &Value| forward(&upstream, request);
                    (Box::new(policy), target.model.clone(), Some(run.seed ^ u64::from(attempt.number())), target.context_tokens)
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
                ..RemoteModel::default()
            }
            .apply_to(&mut settings)
            .map_err(|e| anyhow::anyhow!("configuring the agent's model: {e}"))?;
            let limits = Limits {
                max_tool_calls: run.config.agent.max_tool_calls,
                deadline_secs: run.config.agent.deadline_secs,
                context_tokens,
                max_output_tokens: run.config.agent.max_output_tokens,
            };
            let ran = run.runtime.block_on(workspace_episode::run(&ws, &instance.instruction, settings, &limits, dir.join("audit.jsonl")));
            let exchanges = endpoint.exchanges();
            let verdict = tasks::verify(&instance, &ws);
            let answer = std::fs::read_to_string(ws.join(tasks::ANSWER_FILE)).ok().map(|a| a.trim().chars().take(60).collect::<String>());
            // A run that failed outright (the endpoint down, a request
            // refused) says nothing about the policy: the attempt stays open
            // and runs again, never counted as a failed episode.
            let ran = ran?;
            let (conclusion, calls, trajectory) = (ran.conclusion, ran.tool_calls, ran.trajectory);
            let last_text = exchanges
                .iter()
                .rev()
                .find_map(|x| x.reply.as_ref().ok().and_then(|m| m["content"].as_str().map(String::from)))
                .map(|t| t.chars().take(120).collect::<String>());
            let doc = json!({
                "attempt": attempt.id(), "instance": instance.id, "passed": verdict.passed, "reason": verdict.reason,
                "conclusion": conclusion, "tool_calls": calls, "trajectory": trajectory,
            });
            write(&dir.join("episode.json"), &doc.to_string())?;
            if verdict.passed {
                let calls = exchanges
                    .iter()
                    .map(|x| x.reply.clone().map(|reply| (x.request.clone(), reply)).map_err(|e| anyhow::anyhow!(e)))
                    .collect::<anyhow::Result<Vec<_>>>()?;
                write(&dir.join("records.jsonl"), &wire::episode_lines(&calls, &attempt.id())?)?;
            }
            std::fs::remove_dir_all(&ws).with_context(|| format!("removing {}", ws.display()))?;
            Ok(StageWork::new(json!({
                "passed": verdict.passed, "reason": verdict.reason, "tool_calls": calls, "conclusion": conclusion,
                "answer": answer, "last_text": last_text, "attempt": attempt.id(),
            }))
            .output("episode", &rel))
        })
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
        let mut episodes = Vec::new();
        for instance in self.instances(split)? {
            episodes.push(self.episode(&format!("{name}/{}", instance.id), &instance, actor)?);
        }
        let inputs: Vec<(String, Digest)> = episodes
            .iter()
            .map(|c| {
                (
                    c.unit.clone(),
                    c.output("episode")
                        .cloned()
                        .unwrap_or_else(|| Digest::of(b"")),
                )
            })
            .collect();
        let refs: Vec<(&str, &Digest)> = inputs.iter().map(|(n, d)| (n.as_str(), d)).collect();
        let key = self.key(&json!(self.config.agent), name, &refs, &["splinter"]);
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
            target: self.brain.serve(policy.1)?,
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
        let mut episodes = Vec::new();
        for instance in self.instances(Split::Dev)? {
            let commit = self.episode(&format!("{unit}/{}", instance.id), &instance, &actor)?;
            episodes.push((instance, commit));
        }
        let inputs: Vec<(String, Digest)> = episodes
            .iter()
            .map(|(_, c)| {
                (
                    c.unit.clone(),
                    c.output("episode")
                        .cloned()
                        .unwrap_or_else(|| Digest::of(b"")),
                )
            })
            .collect();
        let mut refs: Vec<(&str, &Digest)> = inputs.iter().map(|(n, d)| (n.as_str(), d)).collect();
        refs.push(("checkpoint", &digest));
        let key = self.key(&json!(self.config.agent), &unit, &refs, &["splinter"]);
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
