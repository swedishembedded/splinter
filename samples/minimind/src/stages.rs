// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements resumable training pipelines that take a
// model from random weights to a working agent in one command, for its
// clients. If your team needs expertise in end-to-end small-model training
// and agent learning, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The stages, in order, over one run directory's ledger.
//!
//! Each stage is a ledger unit whose key is its configuration slice, the run
//! seed, the digests of the stage outputs it reads, its recipe version and
//! the revisions of the services it consumes ([`crate::recipe`]);
//! [`StageLedger::run`] reuses, resumes or recomputes it. The
//! order puts everything that needs no model first (the world, the tasks,
//! the expert's demonstrations), so a run gets as far as the services it has.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context as _};
use serde_json::{json, Value};
use splinter_sdk::store::stage_ledger::{
    Commit, KeyParts, OnCorrupt, StageKey, StageLedger, StageWork, Standing,
};
use splinter_sdk::vocabulary::digest::Digest;

use splinter_sdk::model::scratch::{
    ChatFormat, Checkpoint, CheckpointRequest, ModelService, Objective, TrainRequest,
};

use crate::config::Config;
use crate::recipe::{self, Recipe};
use crate::tasks::{self, Instance, Split};
use crate::template::Template;
use crate::world;

/// A training stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// Pretraining on raw text.
    Pretrain,
    /// Mid-training: raw text mixed with verified agent trajectories.
    Midtrain,
    /// Supervised fine-tuning on the expert's demonstrations.
    Sft,
    /// Training on the model's own verified episodes.
    Improve,
}

/// The code revisions of the run, by name: `sven`, `brain` and `bench` enter
/// the keys of the stages that consume them; any other (this repository's
/// own, `splinter`) is recorded in the manifest only.
#[derive(Clone, Debug, Default)]
pub struct Revisions(pub BTreeMap<String, String>);

/// The commonest words of a fetched corpus left out of the lexicon (articles,
/// pronouns, auxiliaries), and how many it holds.
const LEXICON_SKIP: usize = 100;
const LEXICON_SIZE: usize = 3000;

/// Everything a run is carried out with.
pub struct Run<'a> {
    ledger: Option<StageLedger>,
    dir: PathBuf,
    /// Where fetched corpora are cached across runs (`<run>/cache` unless
    /// set).
    pub cache: PathBuf,
    /// Where files that live for one episode or one stage are made
    /// ([`crate::scratch`]).
    pub scratch: crate::scratch::Scratch,
    /// Natural words names are drawn from, from the fetched corpora (empty
    /// without them).
    pub lexicon: Vec<String>,
    /// The teacher model, when one is given (`--teacher`).
    pub teacher: Option<crate::teacher::Teacher>,
    /// A lock file held exclusively while a training job runs, so the timed
    /// work of others on the same card never overlaps it.
    pub gpu_lock: Option<PathBuf>,
    /// The configuration.
    pub config: Config,
    /// The run seed.
    pub seed: u64,
    /// The code revisions.
    pub revisions: Revisions,
    /// The model service.
    pub brain: &'a dyn ModelService,
    /// Where progress is shown.
    pub out: &'a mut dyn Write,
    /// The template the model is made with.
    pub template: Template,
    /// The units this run committed or reused, in order: what the report
    /// describes (the ledger also holds units of earlier settings).
    pub(crate) this_run: Vec<Commit>,
    pub(crate) runtime: tokio::runtime::Runtime,
}

impl<'a> Run<'a> {
    /// A run over the ledger of `dir`.
    ///
    /// # Errors
    /// The directory is held by another process, or no async runtime starts.
    pub fn open(
        dir: &Path,
        config: Config,
        seed: u64,
        revisions: Revisions,
        brain: &'a dyn ModelService,
        out: &'a mut dyn Write,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            ledger: Some(StageLedger::open(dir)?),
            dir: dir.to_path_buf(),
            cache: dir.join("cache"),
            scratch: crate::scratch::Scratch::create(
                &crate::scratch::Scratch::default_root(dir),
                dir,
            )?,
            gpu_lock: None,
            teacher: None,
            lexicon: Vec::new(),
            config,
            seed,
            revisions,
            brain,
            out,
            template: Template::default(),
            this_run: Vec::new(),
            runtime: tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?,
        })
    }

    /// Makes the run's scratch under `root` instead of the default.
    ///
    /// # Errors
    /// The directory cannot be made.
    pub fn use_scratch_root(&mut self, root: &Path) -> anyhow::Result<()> {
        self.scratch = crate::scratch::Scratch::create(root, &self.dir)?;
        Ok(())
    }

    /// The run directory.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The ledger of the run directory.
    ///
    /// # Panics
    /// Never outside a unit's own work, which holds it.
    #[must_use]
    pub fn ledger(&self) -> &StageLedger {
        self.ledger
            .as_ref()
            .unwrap_or_else(|| unreachable!("the ledger is only lent out to a unit's own run"))
    }

    /// The key of `unit` over `inputs`: its configuration `slice`, the run
    /// seed, the version of its `recipe` and the revisions of the `services`
    /// it consumes.
    pub(crate) fn key(
        &self,
        slice: &Value,
        unit: &str,
        inputs: &[(&str, &Digest)],
        recipe: Recipe,
        services: &[&str],
    ) -> StageKey {
        let mut parts = KeyParts::new(unit)
            .config(&json!({ "slice": slice, "seed": self.seed }))
            .revision("recipe", &recipe.id());
        for (name, digest) in inputs {
            parts = parts.input(name, digest);
        }
        for name in services {
            let rev = self
                .revisions
                .0
                .get(*name)
                .map_or("unknown", String::as_str);
            parts = parts.revision(name, rev);
        }
        parts.key()
    }

    /// Runs `work` as `unit` under `key` and shows the outcome.
    pub(crate) fn unit<F>(&mut self, unit: &str, key: &StageKey, work: F) -> anyhow::Result<Commit>
    where
        F: FnOnce(
            &mut Self,
            &splinter_sdk::store::stage_ledger::Attempt,
        ) -> anyhow::Result<StageWork>,
    {
        // The ledger is lent out of the run for the unit, so the work can use
        // the rest of the run; a unit never runs a unit inside itself.
        let ledger = self
            .ledger
            .take()
            .unwrap_or_else(|| unreachable!("units do not nest"));
        if let Ok(Standing::Committed(commit)) = ledger.standing(unit, key) {
            if let Err(e) = ledger.verify(&commit) {
                writeln!(self.out, "[corrupt] {e}; recomputing {unit}")?;
            }
        }
        let started = std::time::Instant::now();
        let ran = ledger.run(unit, key, OnCorrupt::Recompute, |attempt| {
            work(self, attempt)
        });
        self.ledger = Some(ledger);
        let ran = ran?;
        let how = if ran.reused { "reused" } else { "done" };
        let took = if ran.reused {
            String::new()
        } else {
            format!(" ({} s)", started.elapsed().as_secs())
        };
        writeln!(
            self.out,
            "[{how:>6}] {unit}{took}: {}",
            one_line(&ran.commit.summary)
        )?;
        self.this_run.push(ran.commit.clone());
        Ok(ran.commit)
    }

    /// The path of `rel` in the run directory.
    #[must_use]
    pub fn path(&self, rel: &str) -> PathBuf {
        self.dir().join(rel)
    }

    /// Stage 1: the run manifest.
    ///
    /// # Errors
    /// The manifest cannot be written.
    pub fn init(&mut self) -> anyhow::Result<Commit> {
        // The manifest records every revision, this repository's too, so it
        // is rewritten when any changes; nothing reads its digest.
        let mut parts =
            KeyParts::new("init").config(&json!({ "config": self.config, "seed": self.seed }));
        for (name, rev) in &self.revisions.0 {
            parts = parts.revision(name, rev);
        }
        let key = parts.key();
        self.unit("init", &key, |run, _| {
            let manifest = json!({
                "sample": "minimind",
                "profile": run.config.profile,
                "seed": run.seed,
                "config": run.config,
                "revisions": run.revisions.0,
                "template": run.template,
                "licenses": run.config.fetch.iter().map(|f| json!({ "name": f.name, "url": f.url, "license": f.license })).collect::<Vec<_>>(),
            });
            write(&run.path("manifest.json"), &serde_json::to_string_pretty(&manifest)?)?;
            Ok(StageWork::new(json!({ "profile": run.config.profile, "seed": run.seed })).output("manifest", "manifest.json"))
        })
    }

    /// Stage 2: the world corpus.
    ///
    /// # Errors
    /// The corpus cannot be written.
    pub fn world(&mut self, fetched: &[Commit]) -> anyhow::Result<Commit> {
        let slices: Vec<(String, Digest)> = fetched
            .iter()
            .map(|c| {
                (
                    c.unit.clone(),
                    c.output("slice")
                        .cloned()
                        .unwrap_or_else(|| Digest::of(b"")),
                )
            })
            .collect();
        let inputs: Vec<(&str, &Digest)> = slices.iter().map(|(n, d)| (n.as_str(), d)).collect();
        // The world renders conversations in the template under the agent's
        // system prompt and tools, so their text is part of its slice.
        let slice = json!({
            "world": self.config.world,
            "template": self.template,
            "system": splinter_sdk::agent::sven::config::compact_profile::SYSTEM_PROMPT,
            "tools": crate::collect::tool_sigs(),
            "deadline_secs": self.config.agent.deadline_secs,
        });
        let key = self.key(&slice, "world", &inputs, recipe::WORLD, &[]);
        let fetched: Vec<(String, String)> = self
            .config
            .fetch
            .iter()
            .map(|f| (f.name.clone(), f.separator.clone()))
            .collect();
        self.unit("world", &key, |run, _| {
            let tools = crate::collect::tool_sigs();
            let system = splinter_sdk::agent::sven::config::compact_profile::SYSTEM_PROMPT;
            // Every agent-shaped document ends its request and tool results
            // as an episode under the agent's deadline does.
            let notice = world::time_notice(run.config.agent.deadline_secs);
            let mut docs = world::corpus(run.seed, &run.config.world, &run.template, &tools, system, &notice, &run.lexicon);
            let scratch = run.scratch.path("world");
            for instance in tasks::instances_named(run.seed, Split::World, run.config.world.trajectories, &run.lexicon) {
                docs.push(world::trajectory(&instance, &run.template, &tools, system, &notice, &scratch)?);
            }
            if scratch.exists() {
                std::fs::remove_dir_all(&scratch)?;
            }
            for (name, separator) in &fetched {
                let path = run.path(&format!("fetch/{name}.txt"));
                let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
                docs.extend(
                    text.split(separator.as_str())
                        .map(str::trim)
                        .filter(|p| !p.is_empty())
                        .map(|p| world::Document { kind: "fetched", text: p.to_string() }),
                );
            }
            let texts: Vec<String> = docs.iter().map(|d| d.text.clone()).collect();
            let checks = crate::datacheck::corpus(&texts, &run.config.data_checks)?;
            let text = world::to_jsonl(&docs);
            write(&run.path("world/corpus.jsonl"), &text)?;
            // Every tenth document is held out of pretraining and scored.
            let (held, train): (Vec<_>, Vec<_>) = docs.iter().cloned().enumerate().partition(|(i, _)| i % 10 == 9);
            let strip = |v: Vec<(usize, world::Document)>| v.into_iter().map(|(_, d)| d).collect::<Vec<_>>();
            write(&run.path("world/train.jsonl"), &world::to_jsonl(&strip(train)))?;
            write(&run.path("world/held_out.jsonl"), &world::to_jsonl(&strip(held)))?;
            let mut kinds: BTreeMap<&str, usize> = BTreeMap::new();
            for doc in &docs {
                *kinds.entry(doc.kind).or_default() += 1;
            }
            let sample = docs.iter().find(|d| d.kind == "world").map(|d| d.text.clone());
            Ok(StageWork::new(json!({ "documents": docs.len(), "bytes": text.len(), "kinds": kinds, "sample": sample, "data": checks }))
                .output("corpus", "world/corpus.jsonl")
                .output("train", "world/train.jsonl")
                .output("held_out", "world/held_out.jsonl"))
        })
    }

    /// Corpora fetched at run time, each a unit `fetch/<name>`: the pinned
    /// file is taken from the cache when it holds it with the pinned SHA-256,
    /// else downloaded (TLS verified) and checked; its first `slice_bytes`,
    /// cut at a line end, are the stage's output.
    ///
    /// # Errors
    /// A download fails or does not have the pinned checksum.
    pub fn fetch(&mut self) -> anyhow::Result<Vec<Commit>> {
        let mut commits = Vec::new();
        for corpus in self.config.fetch.clone() {
            let unit = format!("fetch/{}", corpus.name);
            let key = self.key(&json!(corpus), &unit, &[], recipe::WORLD, &[]);
            let cache = self.cache.join(&corpus.name);
            commits.push(self.unit(&unit, &key, |run, _| {
                let pinned = corpus.sha256.to_ascii_lowercase();
                let sha = |path: &Path| -> anyhow::Result<String> {
                    let file = std::fs::File::open(path)
                        .with_context(|| format!("opening {}", path.display()))?;
                    Ok(Digest::sha256_of_reader(file)?.hex().to_string())
                };
                if !cache.exists() || sha(&cache)? != pinned {
                    std::fs::create_dir_all(cache.parent().unwrap_or(&run.dir))?;
                    let partial = cache.with_extension("partial");
                    let status = std::process::Command::new("curl")
                        .args([
                            "--fail",
                            "--silent",
                            "--show-error",
                            "--location",
                            "--proto",
                            "=https,file",
                            "--output",
                        ])
                        .arg(&partial)
                        .arg(&corpus.url)
                        .status()
                        .context("starting curl")?;
                    if !status.success() {
                        bail!("fetching {} failed ({status})", corpus.url);
                    }
                    let got = sha(&partial)?;
                    if got != pinned {
                        bail!(
                            "{} has SHA-256 {got}, not the pinned {pinned}: refused",
                            corpus.url
                        );
                    }
                    std::fs::rename(&partial, &cache)?;
                }
                let bytes = std::fs::read(&cache)?;
                let mut end = bytes
                    .len()
                    .min(usize::try_from(corpus.slice_bytes).unwrap_or(usize::MAX));
                if end < bytes.len() {
                    end = bytes[..end]
                        .iter()
                        .rposition(|b| *b == b'\n')
                        .map_or(0, |i| i + 1);
                }
                let text = String::from_utf8_lossy(&bytes[..end]).into_owned();
                let rel = format!("fetch/{}.txt", corpus.name);
                write(&run.path(&rel), &text)?;
                Ok(StageWork::new(
                    json!({ "bytes": end, "license": corpus.license, "url": corpus.url }),
                )
                .output("slice", &rel))
            })?);
        }
        Ok(commits)
    }

    /// The natural words names are drawn from: a lexicon of the fetched
    /// corpora's words (empty without them).
    ///
    /// # Errors
    /// A fetched slice cannot be read.
    pub fn lexicon_of_fetched(&self, fetched: &[Commit]) -> anyhow::Result<Vec<String>> {
        let mut texts = Vec::new();
        for c in fetched {
            let path = self.path(&crate::teacher::output_path(c, "slice")?);
            texts.push(
                std::fs::read_to_string(&path)
                    .with_context(|| format!("reading {}", path.display()))?,
            );
        }
        Ok(crate::naming::lexicon_of(
            &texts,
            LEXICON_SKIP,
            LEXICON_SIZE,
        ))
    }

    /// Stage 3: the task instances, as an index per split and as Harbor
    /// task folders.
    ///
    /// # Errors
    /// They cannot be written.
    pub fn tasks(&mut self) -> anyhow::Result<Commit> {
        let lexicon = Digest::of(self.lexicon.join("\n").as_bytes());
        let key = self.key(
            &json!(self.config.tasks),
            "tasks",
            &[("lexicon", &lexicon)],
            recipe::TASKS,
            &[],
        );
        self.unit("tasks", &key, |run, _| {
            let counts = &run.config.tasks;
            let mut work = StageWork::new(json!({
                "train": counts.demonstrations, "explore": counts.explore, "dev": counts.probes,
            }));
            for (split, count) in [
                (Split::Train, counts.demonstrations),
                (Split::Explore, counts.explore),
                (Split::Dev, counts.probes),
                (Split::Heldout, counts.probes),
                (Split::Midtrain, counts.midtrain),
                (Split::Teach, counts.teach),
            ] {
                let drawn = tasks::instances_named(run.seed, split, count, &run.lexicon);
                let index: String = drawn
                    .iter()
                    .map(|i| format!("{}\n", serde_json::to_string(i).unwrap_or_default()))
                    .collect();
                let rel = format!("tasks/{}.jsonl", split.name());
                write(&run.path(&rel), &index)?;
                let harbor = run.scratch.path(&format!("tasks/harbor/{}", split.name()));
                if harbor.exists() {
                    std::fs::remove_dir_all(&harbor)?;
                }
                for instance in &drawn {
                    tasks::export_harbor(instance, &harbor.join(&instance.id))?;
                }
                work = work.output(split.name(), &rel);
            }
            // Thousands of small folders, kept as one file.
            crate::pack::pack_tree(
                &run.scratch.path("tasks/harbor"),
                &run.path("tasks/harbor.jsonl"),
            )?;
            Ok(work.output("harbor", "tasks/harbor.jsonl"))
        })
    }

    /// The episode bounds as keys see them: how many run at once changes no
    /// episode's outcome, so it is left out.
    pub(crate) fn agent_slice(&self) -> Value {
        let mut slice = json!(self.config.agent);
        if let Some(fields) = slice.as_object_mut() {
            fields.remove("concurrency");
        }
        slice
    }

    /// The model labelled `label`.
    pub(crate) fn model(&self, label: &str) -> anyhow::Result<crate::config::ModelSpec> {
        self.config
            .models
            .get(label)
            .cloned()
            .with_context(|| format!("no model labelled {label:?}"))
    }

    /// The instances of `split` as the tasks stage wrote them.
    pub(crate) fn instances(&self, split: Split) -> anyhow::Result<Vec<Instance>> {
        let path = self.path(&format!("tasks/{}.jsonl", split.name()));
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        text.lines()
            .map(|l| serde_json::from_str(l).with_context(|| format!("in {}", path.display())))
            .collect()
    }

    /// Stage 4: the tokenizer and the random-init checkpoint.
    ///
    /// # Errors
    /// The model service cannot make it.
    pub fn checkpoint(
        &mut self,
        label: &str,
        world: &Commit,
    ) -> anyhow::Result<(Commit, Checkpoint)> {
        let corpus = world
            .output("corpus")
            .context("the world stage has no corpus")?
            .clone();
        let spec = self.model(label)?;
        let unit = format!("{label}/checkpoint");
        let rel = format!("checkpoints/{label}/init");
        let key = self.key(
            &json!({ "model": spec, "template": self.template, "tokenizer": self.config.tokenizer }),
            &unit,
            &[("corpus", &corpus)],
            recipe::CHECKPOINT,
            &["brain"],
        );
        let split_digits = self.config.tokenizer.split_digits;
        let commit = self.unit(&unit, &key, |run, _| {
            let out = run.path(&rel);
            reset(&out)?;
            let made = run.brain.make_checkpoint(&CheckpointRequest {
                model: &spec,
                format: &ChatFormat {
                    template: run.template.jinja(),
                    specials: run.template.specials(),
                    eos: run.template.end.clone(),
                    pad: Some(run.template.pad.clone()),
                },
                tokenizer_corpus: &run.path("world/corpus.jsonl"),
                split_digits,
                seed: run.seed,
                install_as: &install_name(&key, &format!("{label}-init")),
                identity: &identity(&key),
                out: &out,
            })?;
            Ok(
                StageWork::new(json!({ "model": made.model, "arch": spec.arch }))
                    .output("checkpoint", &rel),
            )
        })?;
        let checkpoint = checkpoint_of(&commit, &rel)?;
        Ok((commit, checkpoint))
    }

    /// Stages 5, 7 and 9: a training run from `base` on `data`, installed and
    /// copied to `checkpoints/<name>`.
    ///
    /// # Errors
    /// The model service cannot run it.
    pub fn train(
        &mut self,
        stage: Stage,
        label: &str,
        name: &str,
        base: (&Commit, &Checkpoint),
        data: (&Digest, &str),
    ) -> anyhow::Result<(Commit, Checkpoint)> {
        let base_digest = base
            .0
            .output("checkpoint")
            .context("the base stage has no checkpoint")?
            .clone();
        let unit = format!("{label}/{name}");
        let rel = format!("checkpoints/{label}/{name}");
        let (objective, budget) = match stage {
            Stage::Pretrain => (Objective::Pretrain, self.config.pretrain.clone()),
            Stage::Midtrain => (Objective::Pretrain, self.config.midtrain.budget.clone()),
            Stage::Sft => (Objective::ChatFull, self.config.sft.clone()),
            Stage::Improve => (Objective::ChatFull, self.config.improve.clone()),
        };
        // A chat stage trained as text: the conversations rendered in the
        // template, every token supervised.
        let as_text = objective == Objective::ChatFull && self.config.chat_training.as_text;
        let objective = if as_text {
            Objective::Pretrain
        } else {
            objective
        };
        let spec = self.model(label)?;
        let key = self.key(
            &json!({ "model": spec, "budget": budget, "as_text": as_text, "template": self.template }),
            &unit,
            &[("base", &base_digest), ("data", data.0)],
            recipe::TRAIN,
            &["brain"],
        );
        let base_checkpoint = base.1.clone();
        let held_out = matches!(stage, Stage::Pretrain | Stage::Midtrain)
            .then(|| self.path("world/held_out.jsonl"));
        let unit_name = unit.clone();
        let commit = self.unit(&unit, &key, |run, attempt| {
            let out = run.path(&rel);
            reset(&out)?;
            // Progress on the terminal, about ten lines a run.
            let every = (budget.steps / 10).max(1);
            let started = std::time::Instant::now();
            let shown = unit_name.clone();
            let _card = match &run.gpu_lock {
                Some(path) => {
                    let file = std::fs::OpenOptions::new()
                        .create(true)
                        .truncate(false)
                        .write(true)
                        .open(path)
                        .with_context(|| format!("opening {}", path.display()))?;
                    file.lock().with_context(|| format!("locking {}", path.display()))?;
                    Some(file)
                }
                None => None,
            };
            // Derived from the data and settings in the key: no output.
            let data_path = if as_text {
                let text = run.path(&format!("{}.text.jsonl", data.1.trim_end_matches(".jsonl")));
                let records = std::fs::read_to_string(run.path(data.1))?;
                write(&text, &crate::sequence::conversations_as_text(&run.template, &records)?)?;
                text
            } else {
                run.path(data.1)
            };
            let trained = run.brain.train(
                &TrainRequest {
                    objective,
                    base: &base_checkpoint,
                    data: &data_path,
                    held_out: held_out.as_deref(),
                    budget: &budget,
                    seed: run.seed,
                    install_as: &install_name(&key, &format!("{label}-{name}")),
                    identity: &identity(&key),
                    out: &out,
                },
                &mut |step, loss| {
                    if step % every == 0 {
                        let _ = writeln!(
                            run.out,
                            "         {shown}: step {step}/{} loss {loss:.3} ({:.0} s)",
                            budget.steps,
                            started.elapsed().as_secs_f64()
                        );
                    }
                },
            )?;
            let first = trained.losses.first().map(|l| l.1);
            let last = trained.losses.last().map(|l| l.1);
            let resume = &trained.report["resume"];
            if resume["state"] == "continued" {
                writeln!(run.out, "         {shown}: brain continued the run from its saved state at step {}", resume["resumed_at"])?;
            }
            let seconds = started.elapsed().as_secs_f64();
            // Pretraining's windows are full: every step trains batch x block
            // tokens; a continued run trained only the steps after its saved
            // state. Chat records vary in length, so no rate is claimed for a
            // chat run.
            let trained_steps = u64::from(budget.steps).saturating_sub(resume["resumed_at"].as_u64().unwrap_or(0));
            let rate = (objective == Objective::Pretrain && seconds > 0.0).then(|| {
                trained_steps as f64 * f64::from(budget.batch) * f64::from(budget.block) / seconds
            });
            Ok(StageWork::new(json!({
                "model": trained.checkpoint.model, "arch": spec.arch, "steps": budget.steps, "first_loss": first, "last_loss": last,
                "losses": trained.losses, "resumed": trained.resumed, "attempt": attempt.id(),
                "resume": { "state": resume["state"], "resumed_at": resume["resumed_at"] },
                "train_seconds": started.elapsed().as_secs(), "tokens_per_second": rate,
                "tokens": (objective == Objective::Pretrain).then(|| u64::from(budget.steps) * u64::from(budget.batch) * u64::from(budget.block)),
            }))
            .output("checkpoint", &rel))
        })?;
        let checkpoint = checkpoint_of(&commit, &rel)?;
        Ok((commit, checkpoint))
    }
}

/// The brain job identity (`resume_key`) of a unit's key: a rerun after an
/// interruption resubmits the same request under it, and brain continues the
/// run or answers a completed one from its record.
fn identity(key: &StageKey) -> String {
    let hex = key.as_str().rsplit(':').next().unwrap_or(key.as_str());
    format!("minimind-{}", &hex[..24.min(hex.len())])
}

/// The name a stage's checkpoint is installed under: its stage and the head
/// of its key, so a changed input never overwrites the old one.
fn install_name(key: &StageKey, stage: &str) -> String {
    let hex = key.as_str().rsplit(':').next().unwrap_or(key.as_str());
    format!("minimind-{stage}-{}", &hex[..12.min(hex.len())])
}

fn checkpoint_of(commit: &Commit, rel: &str) -> anyhow::Result<Checkpoint> {
    let model = commit.summary["model"]
        .as_str()
        .context("the stage recorded no model")?
        .to_string();
    Ok(Checkpoint {
        model,
        dir: PathBuf::from(rel),
    })
}

/// Empties `dir`: an interrupted attempt's partial output never survives
/// into the next.
pub(crate) fn reset(dir: &Path) -> anyhow::Result<()> {
    if dir.exists() {
        std::fs::remove_dir_all(dir).with_context(|| format!("clearing {}", dir.display()))?;
    }
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))
}

/// Writes `text` to `path` atomically.
pub(crate) fn write(path: &Path, text: &str) -> anyhow::Result<()> {
    splinter_sdk::store::stage_ledger::write_atomic(path, text.as_bytes())?;
    Ok(())
}

/// A summary on one line, without the long fields.
pub(crate) fn one_line(summary: &Value) -> String {
    let Value::Object(map) = summary else {
        return summary.to_string();
    };
    map.iter()
        .filter(|(k, _)| {
            !matches!(
                k.as_str(),
                "losses"
                    | "examples"
                    | "kinds"
                    | "sample"
                    | "text"
                    | "last_text"
                    | "attempt"
                    | "reason"
            )
        })
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Every stage in order, for every configured model, over the shared data
/// stages; the report last.
///
/// # Errors
/// The first stage that cannot be done, by name.
pub fn run_all(run: &mut Run<'_>) -> anyhow::Result<()> {
    run.init()?;
    let fetched = run.fetch()?;
    run.lexicon = run.lexicon_of_fetched(&fetched)?;
    let world = run.world(&fetched)?;
    run.tasks()?;
    let demonstrations = run.demonstrate()?;
    let taught = run.teach()?;
    let sft_data = run.sft_data(&demonstrations, taught.as_ref())?;
    let labels: Vec<String> = run.config.models.keys().cloned().collect();
    let mut failures = Vec::new();
    for label in &labels {
        if let Err(e) = crate::sequence::model_stages(run, label, &world, &sft_data) {
            writeln!(run.out, "[failed] {label}: {e:#}")?;
            failures.push(format!("{label}: {e:#}"));
        }
    }
    crate::report::write(run, &labels)?;
    if failures.is_empty() {
        Ok(())
    } else {
        bail!("{}", failures.join("; "))
    }
}
