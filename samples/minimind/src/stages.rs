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

use splinter_sdk::bench::examine::Examiner;
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

/// Everything a run is carried out with.
pub struct Run<'a> {
    ledger: Option<StageLedger>,
    dir: PathBuf,
    /// Where fetched corpora are cached across runs (`<run>/cache` unless
    /// set).
    pub cache: PathBuf,
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
    /// The examiner; `None` leaves the examination out, said in the report.
    pub examiner: Option<&'a dyn Examiner>,
    /// Where progress is shown.
    pub out: &'a mut dyn Write,
    /// The template the model is made with.
    pub template: Template,
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
        examiner: Option<&'a dyn Examiner>,
        out: &'a mut dyn Write,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            ledger: Some(StageLedger::open(dir)?),
            dir: dir.to_path_buf(),
            cache: dir.join("cache"),
            gpu_lock: None,
            config,
            seed,
            revisions,
            brain,
            examiner,
            out,
            template: Template::default(),
            runtime: tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?,
        })
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
        let ran = ledger.run(unit, key, OnCorrupt::Recompute, |attempt| {
            work(self, attempt)
        });
        self.ledger = Some(ledger);
        let ran = ran?;
        let how = if ran.reused { "reused" } else { "done" };
        writeln!(
            self.out,
            "[{how:>6}] {unit}: {}",
            one_line(&ran.commit.summary)
        )?;
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
        });
        let key = self.key(&slice, "world", &inputs, recipe::WORLD, &[]);
        let names: Vec<String> = self.config.fetch.iter().map(|f| f.name.clone()).collect();
        self.unit("world", &key, |run, _| {
            let tools = crate::collect::tool_sigs();
            let system = splinter_sdk::agent::sven::config::compact_profile::SYSTEM_PROMPT;
            let mut docs = world::corpus(run.seed, &run.config.world, &run.template, &tools, system);
            let scratch = run.path("world/scratch");
            for instance in tasks::instances(run.seed, Split::World, run.config.world.trajectories) {
                docs.push(world::trajectory(&instance, &run.template, &tools, system, &scratch)?);
            }
            if scratch.exists() {
                std::fs::remove_dir_all(&scratch)?;
            }
            for name in &names {
                let path = run.path(&format!("fetch/{name}.txt"));
                let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
                docs.extend(
                    text.split("\n\n")
                        .map(str::trim)
                        .filter(|p| !p.is_empty())
                        .map(|p| world::Document { kind: "fetched", text: p.to_string() }),
                );
            }
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
            Ok(StageWork::new(json!({ "documents": docs.len(), "bytes": text.len(), "kinds": kinds, "sample": sample }))
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

    /// Stage 3: the task instances, as an index per split and as Harbor
    /// task folders.
    ///
    /// # Errors
    /// They cannot be written.
    pub fn tasks(&mut self) -> anyhow::Result<Commit> {
        let key = self.key(&json!(self.config.tasks), "tasks", &[], recipe::TASKS, &[]);
        self.unit("tasks", &key, |run, _| {
            let counts = &run.config.tasks;
            let mut work = StageWork::new(json!({
                "train": counts.demonstrations, "explore": counts.explore, "dev": counts.probes,
            }));
            for (split, count) in [
                (Split::Train, counts.demonstrations),
                (Split::Explore, counts.explore),
                (Split::Dev, counts.probes),
            ] {
                let drawn = tasks::instances(run.seed, split, count);
                let index: String = drawn
                    .iter()
                    .map(|i| format!("{}\n", serde_json::to_string(i).unwrap_or_default()))
                    .collect();
                let rel = format!("tasks/{}.jsonl", split.name());
                write(&run.path(&rel), &index)?;
                let harbor = run.path(&format!("tasks/harbor/{}", split.name()));
                if harbor.exists() {
                    std::fs::remove_dir_all(&harbor)?;
                }
                for instance in &drawn {
                    tasks::export_harbor(instance, &harbor.join(&instance.id))?;
                }
                work = work.output(split.name(), &rel);
            }
            Ok(work.output("harbor", "tasks/harbor"))
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
            &json!({ "model": spec, "template": self.template }),
            &unit,
            &[("corpus", &corpus)],
            recipe::CHECKPOINT,
            &["brain"],
        );
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
            Stage::Sft => (Objective::ChatFull, self.config.sft.clone()),
            Stage::Improve => (Objective::ChatFull, self.config.improve.clone()),
        };
        let spec = self.model(label)?;
        let key = self.key(
            &json!({ "model": spec, "budget": budget }),
            &unit,
            &[("base", &base_digest), ("data", data.0)],
            recipe::TRAIN,
            &["brain"],
        );
        let base_checkpoint = base.1.clone();
        let held_out = (stage == Stage::Pretrain).then(|| self.path("world/held_out.jsonl"));
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
            let trained = run.brain.train(
                &TrainRequest {
                    objective,
                    base: &base_checkpoint,
                    data: &run.path(data.1),
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
            Ok(StageWork::new(json!({
                "model": trained.checkpoint.model, "arch": spec.arch, "steps": budget.steps, "first_loss": first, "last_loss": last,
                "losses": trained.losses, "resumed": trained.resumed, "attempt": attempt.id(),
                "seconds": started.elapsed().as_secs(),
            }))
            .output("checkpoint", &rel))
        })?;
        let checkpoint = checkpoint_of(&commit, &rel)?;
        Ok((commit, checkpoint))
    }

    /// Stage 10: the paired examination of `candidate` against
    /// `predecessor`, as unit `examine/<name>`.
    ///
    /// # Errors
    /// The examination could not be run.
    pub fn examine(
        &mut self,
        name: &str,
        candidate: (&Commit, &Checkpoint),
        predecessor: (&Commit, &Checkpoint),
    ) -> anyhow::Result<Option<Commit>> {
        let Some(examiner) = self.examiner else {
            writeln!(
                self.out,
                "[ skip ] {name}/examine: no examiner configured; no promotion claim is made"
            )?;
            return Ok(None);
        };
        let c = candidate
            .0
            .output("checkpoint")
            .context("the candidate has no checkpoint")?
            .clone();
        let p = predecessor
            .0
            .output("checkpoint")
            .context("the predecessor has no checkpoint")?
            .clone();
        let unit = format!("{name}/examine");
        let key = self.key(
            &self.agent_slice(),
            &unit,
            &[("candidate", &c), ("predecessor", &p)],
            recipe::EXAMINE,
            &["bench", "sven", "brain"],
        );
        let rel = format!("exam/{name}/verdict.json");

        let (cand, pred) = (candidate.1.clone(), predecessor.1.clone());
        let commit = self.unit(&unit, &key, |run, _| {
            let out = run.path(&rel);
            reset(out.parent().context("a verdict has a directory")?)?;
            let examined = examiner.examine(&run.brain.serve(&cand)?, &run.brain.serve(&pred)?, &out)?;
            Ok(StageWork::new(json!({
                "verdict": examined.verdict, "verified": examined.verified, "promotes": examined.promotes(),
                "candidate": cand.model, "predecessor": pred.model,
            }))
            .output("verdict", &rel))
        })?;
        Ok(Some(commit))
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
    let world = run.world(&fetched)?;
    run.tasks()?;
    let demonstrations = run.demonstrate()?;
    let labels: Vec<String> = run.config.models.keys().cloned().collect();
    let mut failures = Vec::new();
    for label in &labels {
        if let Err(e) = model_stages(run, label, &world, &demonstrations) {
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

/// The model stages of the model labelled `label`.
fn model_stages(
    run: &mut Run<'_>,
    label: &str,
    world: &Commit,
    demonstrations: &Commit,
) -> anyhow::Result<()> {
    let corpus = world
        .output("train")
        .context("the world has no training corpus")?
        .clone();
    let init = run.checkpoint(label, world)?;
    run.probe(&format!("{label}/init"), (&init.0, &init.1))?;
    let pretrained = run.train(
        Stage::Pretrain,
        label,
        "pretrain",
        (&init.0, &init.1),
        (&corpus, "world/train.jsonl"),
    )?;
    run.probe(
        &format!("{label}/pretrained"),
        (&pretrained.0, &pretrained.1),
    )?;
    let demos = demonstrations
        .output("records")
        .context("the demonstrations have no records")?
        .clone();
    let sft = run.train(
        Stage::Sft,
        label,
        "sft",
        (&pretrained.0, &pretrained.1),
        (&demos, "data/demonstrate.jsonl"),
    )?;
    run.probe(&format!("{label}/sft"), (&sft.0, &sft.1))?;
    run.examine(
        &format!("{label}/scratch"),
        (&sft.0, &sft.1),
        (&init.0, &init.1),
    )?;
    let explored = run.explore(label, (&sft.0, &sft.1))?;
    let accepted = explored.summary["accepted"].as_u64().unwrap_or(0);
    if accepted == 0 {
        writeln!(
            run.out,
            "[ skip ] {label}/improve: no verified episode of its own to learn from"
        )?;
        return Ok(());
    }
    let records = explored
        .output("records")
        .context("the exploration has no records")?
        .clone();
    let data = format!("data/{label}/explore.jsonl");
    let improved = run.train(
        Stage::Improve,
        label,
        "improve",
        (&sft.0, &sft.1),
        (&records, &data),
    )?;
    run.probe(&format!("{label}/improved"), (&improved.0, &improved.1))?;
    run.examine(
        &format!("{label}/experience"),
        (&improved.0, &improved.1),
        (&sft.0, &sft.1),
    )?;
    Ok(())
}
