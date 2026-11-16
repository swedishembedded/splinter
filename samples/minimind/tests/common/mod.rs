// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A model service double for the acceptance specs: a checkpoint is a small
//! file whose content is a function of what made it, so a repeated run makes
//! the same bytes; a fine-tuned checkpoint is served as the expert (a model
//! that learned the families), any other as a model that only talks. Every
//! call is counted, and a named call can be made to fail once, as a kill
//! would leave it.

#![allow(clippy::unwrap_used, dead_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use splinter_minimind::expert;
use splinter_minimind::tasks::Instance;
use splinter_sdk::agent::policy_endpoint::Endpoint;
use splinter_sdk::bench::examine::{Examined, Examiner};
use splinter_sdk::model::scratch::{
    Checkpoint, CheckpointRequest, ModelService, Objective, Served, TrainRequest, Trained,
};
use splinter_sdk::vocabulary::digest::Digest;

/// The double.
pub struct FakeBrain {
    run_dir: PathBuf,
    /// Every make/train call, by the name it installs.
    pub calls: Mutex<Vec<String>>,
    /// Calls whose install name contains this fail once.
    pub fail_once: Mutex<Option<String>>,
    /// Requests to a served model fail while this is set (a server that is down).
    pub down: Arc<AtomicBool>,
    /// Only models whose name contains this are down (all when unset).
    pub down_for: Mutex<Option<String>>,
    served: Mutex<BTreeMap<String, Endpoint>>,
}

impl FakeBrain {
    pub fn new(run_dir: &Path) -> Self {
        Self {
            run_dir: run_dir.to_path_buf(),
            calls: Mutex::new(Vec::new()),
            fail_once: Mutex::new(None),
            down: Arc::new(AtomicBool::new(false)),
            down_for: Mutex::new(None),
            served: Mutex::new(BTreeMap::new()),
        }
    }

    pub fn trained(&self) -> usize {
        self.calls.lock().unwrap().len()
    }

    fn maybe_fail(&self, name: &str) -> anyhow::Result<()> {
        let mut fail = self.fail_once.lock().unwrap();
        if fail.as_ref().is_some_and(|f| name.contains(f.as_str())) {
            *fail = None;
            anyhow::bail!("killed during {name}");
        }
        Ok(())
    }

    fn checkpoint(&self, out: &Path, name: &str, made_from: &str) -> Checkpoint {
        std::fs::create_dir_all(out).unwrap();
        std::fs::write(
            out.join("weights.bin"),
            Digest::of(made_from.as_bytes()).to_string(),
        )
        .unwrap();
        self.calls.lock().unwrap().push(name.to_string());
        Checkpoint {
            model: format!("local/{name}"),
            dir: out.to_path_buf(),
        }
    }
}

fn instances(run_dir: &Path) -> Vec<Instance> {
    ["train", "explore", "dev"]
        .iter()
        .flat_map(|s| {
            std::fs::read_to_string(run_dir.join(format!("tasks/{s}.jsonl")))
                .unwrap_or_default()
                .lines()
                .map(|l| serde_json::from_str::<Instance>(l).unwrap())
                .collect::<Vec<_>>()
        })
        .collect()
}

impl ModelService for FakeBrain {
    fn make_checkpoint(&self, request: &CheckpointRequest<'_>) -> anyhow::Result<Checkpoint> {
        self.maybe_fail(request.install_as)?;
        let corpus = std::fs::read(request.tokenizer_corpus)?;
        Ok(self.checkpoint(
            request.out,
            request.install_as,
            &format!("init {} {}", request.seed, Digest::of(&corpus)),
        ))
    }

    fn train(
        &self,
        request: &TrainRequest<'_>,
        on_step: &mut dyn FnMut(u32, f64),
    ) -> anyhow::Result<Trained> {
        self.maybe_fail(request.install_as)?;
        let data = std::fs::read(request.data)?;
        let losses: Vec<(u32, f64)> = (1..=request.budget.steps)
            .map(|s| (s, 4.0 / f64::from(s)))
            .collect();
        for (s, l) in &losses {
            on_step(*s, *l);
        }
        let learned = matches!(request.objective, Objective::ChatFull);
        let tag = if learned { "learned" } else { "talks" };
        let made = format!(
            "{tag} {} {} {}",
            request.base.model,
            Digest::of(&data),
            request.budget.steps
        );
        Ok(Trained {
            report: json!({}),
            checkpoint: self.checkpoint(request.out, request.install_as, &made),
            losses,
            resumed: false,
        })
    }

    fn serve(&self, checkpoint: &Checkpoint) -> anyhow::Result<Served> {
        let mut served = self.served.lock().unwrap();
        if !served.contains_key(&checkpoint.model) {
            let learned = checkpoint.model.contains("sft") || checkpoint.model.contains("improve");
            let tasks = instances(&self.run_dir);
            let down = Arc::clone(&self.down);
            let affected = self
                .down_for
                .lock()
                .unwrap()
                .as_ref()
                .is_none_or(|f| checkpoint.model.contains(f.as_str()));
            let policy = move |request: &Value| -> anyhow::Result<Value> {
                if affected && down.load(Ordering::SeqCst) {
                    anyhow::bail!("the model server is down");
                }
                let messages = request["messages"].as_array().cloned().unwrap_or_default();
                let tools = request["tools"].as_array().cloned().unwrap_or_default();
                let asked = messages
                    .iter()
                    .find(|m| m["role"] == "user")
                    .map(|m| expert::content_text(&m["content"]))
                    .unwrap_or_default();
                match tasks.iter().find(|t| asked.contains(&t.instruction)) {
                    Some(task) if learned => {
                        Ok(expert::next(task, &messages, &tools)?.message(&messages))
                    }
                    _ => Ok(json!({ "role": "assistant", "content": "zq zq zq" })),
                }
            };
            served.insert(
                checkpoint.model.clone(),
                Endpoint::start(&checkpoint.model, Box::new(policy))?,
            );
        }
        let url = served[&checkpoint.model].url().to_string();
        Ok(Served {
            base_url: url,
            model: checkpoint.model.clone(),
            context_tokens: 4096,
            key: String::new(),
        })
    }

    fn complete(
        &self,
        checkpoint: &Checkpoint,
        _prompt: &str,
        _max_tokens: u32,
    ) -> anyhow::Result<String> {
        Ok(if checkpoint.model.contains("init") {
            "zq zq".into()
        } else {
            " a river. Its mayor is".into()
        })
    }
}

/// An examiner double that returns a fixed verdict.
pub struct FakeExaminer {
    pub verdict: &'static str,
    pub verified: bool,
}

impl Examiner for FakeExaminer {
    fn examine(
        &self,
        _candidate: &Served,
        _predecessor: &Served,
        out: &Path,
    ) -> anyhow::Result<Examined> {
        std::fs::write(out, json!({ "verdict": self.verdict }).to_string())?;
        Ok(Examined {
            verdict: self.verdict.into(),
            verified: self.verified,
            document: out.to_path_buf(),
        })
    }
}
