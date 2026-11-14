// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that qualify the serving and
// training path of a learning system before any learning result is believed,
// for its clients. If your team needs expertise in telling a model's
// competence apart from its tooling's faults you can procure our services by
// sending an email to info@swedishembedded.com.

//! `served-items`: a list of steps run through the served path an experiment
//! uses: one `brain serve` started by Splinter's own host, exams asked through
//! `ServedValidator` (sven's provider, no system prompt, thinking off,
//! greedy), training as a job of the same server (`fine_tune_served`) and
//! trained adapters hot-loaded from the directory the server follows. Every
//! call's outcome is written, failures included, so nothing is silently
//! dropped and no infrastructure failure is counted as a wrong answer.

mod items;

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context as _, Result};
use clap::Parser;
use serde::Deserialize;
use serde_json::{json, Value};
use splinter_sdk::campaign::brain::ModelSettings;
use splinter_sdk::campaign::served::{CallFailure, PolicyModels, ServedValidator};
use splinter_sdk::campaign::PolicyRef;
use splinter_sdk::model::precision::BasePrecision;
use splinter_sdk::model::served_train::fine_tune_served;
use splinter_sdk::model::train::FineTune;
use splinter_sdk::served_host::{HostSettings, ServedHost};
use splinter_sdk::vocabulary::digest::Digest;

use crate::items::{judge, Items};

#[derive(Parser, Debug)]
#[command(
    about = "Run exam, training and adapter steps through the served path, recording every call"
)]
struct Cli {
    /// The `brain` binary.
    #[arg(long)]
    brain: PathBuf,
    /// Brain's model store.
    #[arg(long)]
    models_dir: PathBuf,
    /// The base checkpoint directory.
    #[arg(long)]
    model: PathBuf,
    /// The card, in brain's `--device` grammar.
    #[arg(long)]
    device: String,
    /// The KV-cache tier; brain's default when not given.
    #[arg(long)]
    kv_cache: Option<String>,
    /// Sequences decoded together, and requests in flight.
    #[arg(long, default_value_t = 32)]
    slots: u32,
    /// Output tokens a reply may take.
    #[arg(long, default_value_t = 1024)]
    max_tokens: u32,
    /// The items (JSONL: id, prompt, answer, role).
    #[arg(long)]
    items: PathBuf,
    /// The steps (a JSON list), run in order.
    #[arg(long)]
    steps: PathBuf,
    /// Where the server, the adapters and the records go.
    #[arg(long)]
    out: PathBuf,
}

/// One step.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
enum Step {
    /// The items of `items` (a role or comma-separated ids) answered by
    /// `policy` (`base` or a named adapter).
    Answer {
        name: String,
        policy: String,
        items: String,
        #[serde(default)]
        trace: bool,
    },
    /// An adapter file under a name.
    Adapter { name: String, file: PathBuf },
    /// A supervised training job of the server on `records`, its adapter
    /// kept under `name`.
    Train {
        name: String,
        records: PathBuf,
        held_out: PathBuf,
        steps: u32,
        lr: f32,
        rank: u32,
        alpha: f32,
        grad_accum: u32,
        seed: u64,
        #[serde(default)]
        base_precision: Option<String>,
    },
    /// Stops the server; the next step starts a new server session.
    Restart,
}

struct Run<'a> {
    cli: &'a Cli,
    items: &'a Items,
    host: ServedHost,
    adapters: BTreeMap<String, PolicyRef>,
    session: u32,
    calls: std::fs::File,
}

fn class_of(failure: &CallFailure) -> &'static str {
    match failure {
        CallFailure::TimedOut(_) => "timeout",
        CallFailure::Server(_) => "server_error",
        CallFailure::Cancelled => "cancelled",
    }
}

impl Run<'_> {
    fn policy(&self, name: &str) -> Result<PolicyRef> {
        if name == "base" {
            return Ok(PolicyRef::default());
        }
        self.adapters
            .get(name)
            .cloned()
            .with_context(|| format!("no adapter is named {name:?}"))
    }

    fn answer(&mut self, name: &str, policy: &str, items: &str, trace: bool) -> Result<Value> {
        let ids = self.items.select(items)?;
        let policy_ref = self.policy(policy)?;
        let mut settings = ModelSettings::new(&self.cli.model);
        settings.max_tokens = self.cli.max_tokens;
        settings.concurrency = self.cli.slots as usize;
        let mut validator = ServedValidator::new(self.items, &self.host, settings)?;
        if trace {
            validator = validator.traced();
        }
        let model = self.host.model(&policy_ref, true)?;
        let started = Instant::now();
        let attempts = validator.attempt_with(&model, &ids)?;
        let seconds = started.elapsed().as_secs_f64();
        let mut classes: BTreeMap<&str, u32> = BTreeMap::new();
        let mut generated = 0usize;
        for attempt in &attempts {
            let mut record = json!({
                "step": name, "session": self.session, "policy": policy,
                "policy_digest": policy_ref.digest.as_ref().map(ToString::to_string), "item": attempt.task,
            });
            let class = match &attempt.outcome {
                Err(failure) => {
                    record["error"] = json!(failure.to_string());
                    class_of(failure)
                }
                Ok(answered) => {
                    let reply = &answered.reply;
                    generated += reply.token_ids.len();
                    record["text"] = json!(reply.text);
                    record["truncated"] = json!(reply.truncated);
                    record["served"] = json!({
                        "adapter_digest": reply.served.adapter_digest, "base_digest": reply.served.base_digest,
                        "chat_template_digest": reply.served.chat_template_digest, "engine": reply.served.engine,
                    });
                    if trace {
                        record["prompt_token_ids"] = json!(reply.prompt_token_ids);
                        record["token_ids"] = json!(reply.token_ids);
                        record["logprobs"] = json!(reply.logprobs);
                    }
                    if reply.truncated {
                        "truncated"
                    } else {
                        match judge(&reply.text, &self.items.get(&attempt.task)?.answer) {
                            items::Verdict::Correct => "correct",
                            items::Verdict::Wrong => "wrong",
                            items::Verdict::Malformed => "malformed",
                        }
                    }
                }
            };
            record["class"] = json!(class);
            *classes.entry(class).or_default() += 1;
            writeln!(self.calls, "{record}")?;
        }
        self.calls.flush()?;
        Ok(
            json!({"answer": name, "policy": policy, "session": self.session, "n": attempts.len(),
                  "classes": classes, "seconds": seconds, "generated_tokens": trace.then_some(generated)}),
        )
    }

    fn adopt(&mut self, name: &str, file: &Path) -> Result<Value> {
        let digest = Digest::sha256_of_reader(
            std::fs::File::open(file).with_context(|| format!("{}", file.display()))?,
        )?;
        let policy = PolicyRef {
            version: u64::try_from(self.adapters.len() + 1)?,
            adapter: Some(std::path::absolute(file)?),
            digest: Some(digest.clone()),
        };
        // Offered to the server now, so a refused adapter fails here.
        self.host.context(&policy)?;
        self.adapters.insert(name.to_string(), policy);
        Ok(json!({"adapter": name, "digest": digest.to_string()}))
    }

    fn train(
        &mut self,
        name: &str,
        records: &Path,
        held_out: &Path,
        recipe: &TrainRecipe,
    ) -> Result<Value> {
        let attempt = self.cli.out.join("attempts").join(name);
        let endpoints = self.host.endpoints()?;
        let base = match recipe.base_precision.as_deref() {
            None | Some("bf16") => BasePrecision::Bf16,
            Some("fp32") => BasePrecision::Fp32,
            Some(other) => anyhow::bail!("{other:?} is not a base precision: bf16 or fp32"),
        };
        let request = FineTune {
            model_dir: &self.cli.model,
            train: records,
            held_out,
            held_out_text: None,
            monitor: None,
            eval_every: 0,
            patience: 0,
            attempt_dir: &attempt,
            steps: recipe.steps,
            rank: recipe.rank,
            alpha: recipe.alpha,
            replay: &[],
            replay_share: None,
            weighted_replay: &[],
            grad_accum: recipe.grad_accum,
            continue_from: None,
            cancel: None,
            base,
            learning_rate: Some(recipe.lr),
            weight_decay: 0.0,
            seed: recipe.seed,
            max_block: Some(4096),
            thinking: false,
            pack_answers: false,
            on_step: None,
            keep_evaluations: None,
        };
        let started = Instant::now();
        let trained =
            fine_tune_served(&endpoints, &request).with_context(|| format!("training {name}"))?;
        let seconds = started.elapsed().as_secs_f64();
        let adopted = self.adopt(name, &trained.adapter)?;
        Ok(
            json!({"train": name, "session": self.session, "seconds": seconds, "adapter": adopted,
                  "steps_completed": trained.curve.steps_completed, "held_out_loss_base": trained.base.loss,
                  "held_out_loss_tuned": trained.tuned.loss, "speed": format!("{:?}", trained.curve.speed)}),
        )
    }
}

struct TrainRecipe {
    steps: u32,
    lr: f32,
    rank: u32,
    alpha: f32,
    grad_accum: u32,
    seed: u64,
    base_precision: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let items = Items::open(&cli.items)?;
    let steps: Vec<Step> = serde_json::from_str(
        &std::fs::read_to_string(&cli.steps)
            .with_context(|| format!("reading {}", cli.steps.display()))?,
    )
    .context("the steps")?;
    std::fs::create_dir_all(&cli.out)?;
    let host = ServedHost::new(HostSettings {
        brain: cli.brain.clone(),
        models: cli.models_dir.clone(),
        base: cli.model.clone(),
        work: cli.out.join("server-work"),
        state: cli.out.join("state"),
        store: cli.out.join("artifacts"),
        adapters: cli.out.join("served-adapters"),
        log: cli.out.join("brain-serve.log"),
        device: cli.device.clone(),
        kv_cache: cli.kv_cache.clone(),
        slots: cli.slots,
        runtime_dirs: Vec::new(),
        predict_temperature: 1.0,
    });
    let calls = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(cli.out.join("calls.jsonl"))?;
    let mut summary = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(cli.out.join("steps.jsonl"))?;
    let mut run = Run {
        cli: &cli,
        items: &items,
        host,
        adapters: BTreeMap::new(),
        session: 1,
        calls,
    };
    for step in steps {
        let started = Instant::now();
        let mut done = match step {
            Step::Answer {
                name,
                policy,
                items,
                trace,
            } => run.answer(&name, &policy, &items, trace)?,
            Step::Adapter { name, file } => run.adopt(&name, &file)?,
            Step::Train {
                name,
                records,
                held_out,
                steps,
                lr,
                rank,
                alpha,
                grad_accum,
                seed,
                base_precision,
            } => {
                let recipe = TrainRecipe {
                    steps,
                    lr,
                    rank,
                    alpha,
                    grad_accum,
                    seed,
                    base_precision,
                };
                run.train(&name, &records, &held_out, &recipe)?
            }
            Step::Restart => {
                run.host.stop();
                run.session += 1;
                json!({"restart": run.session})
            }
        };
        done["wall_seconds"] = json!(started.elapsed().as_secs_f64());
        println!("{done}");
        writeln!(summary, "{done}")?;
    }
    Ok(())
}
