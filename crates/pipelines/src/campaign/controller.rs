// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that run model learning as one
// resumable, multi-generation campaign with a single promotion gate, for its
// clients. If your team needs expertise in self-improving training loops you
// can procure our services by sending an email to info@swedishembedded.com.

//! The campaign controller: generations of rollout, ingest, curate, train,
//! evaluate, gate and promote, each stage a unit of the campaign's stage
//! ledger. See the module documentation for the stages and the ledger.

use std::cell::RefCell;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use splinter_agent::CancelToken;
use splinter_core::digest::{digest_of, Digest};
use splinter_eval::promotion::{decide, Promotion, Submitted};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_store::experiences::SetId;
use splinter_store::stage_ledger::{
    Commit, KeyParts, LedgerError, OnCorrupt, StageLedger, StageWork, Standing,
};

use super::account::{
    base_of, first_champion, latest_per_generation, next_generation, rates, report, spent,
    CampaignReport, CurateRecord, IngestRecord, RolloutRecord, TrainRecord,
};
use super::champion::{self, Move};
use super::curriculum::{self, Planned};
use super::spec::CampaignSpec;
use super::work::{ExamRequest, Examiner, Model, RolloutRequest, Rollouts, TransferRequest};
use super::Stopwatch;
use crate::bench::{ingest_bench_trials, IngestOptions};
use crate::datasets::{self, BuildRequest, RewardPreset};
use crate::served::ServedError;
use crate::train::{self, TrainRequest, Trainer, Tuning};

/// The ledger unit that holds the campaign's identity and seed.
pub const CAMPAIGN_UNIT: &str = "campaign";

/// The stages of a generation, in order.
pub const STAGES: [&str; 7] = [
    "rollout", "ingest", "curate", "train", "evaluate", "gate", "promote",
];

/// What the controller is made of.
pub struct Parts<'p> {
    /// Where the champion practises.
    pub rollouts: &'p mut dyn Rollouts,
    /// Who examines candidates.
    pub examiner: &'p mut dyn Examiner,
    /// The call into brain.
    pub trainer: &'p dyn Trainer,
    /// The clock training's GPU time is read from.
    pub stopwatch: &'p dyn Stopwatch,
    /// Cards a training run holds, to turn its wall time into GPU-hours.
    pub gpus: u32,
    /// Stops the campaign between stages and the stage in progress (a
    /// training job is cancelled on the server).
    pub cancel: CancelToken,
    /// Where a one-line account of each stage goes; `None` stays silent.
    pub progress: Option<&'p dyn Fn(&str)>,
}

/// Why a campaign stopped.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "stopped", rename_all = "snake_case")]
pub enum Stopped {
    /// It ran every generation it was given.
    Done,
    /// The next stage would exceed a budget, and did not start.
    Budget {
        /// Which: `rollouts`, `gpu_hours` or `tokens`.
        limit: String,
        /// What was spent of it.
        spent: f64,
        /// The limit.
        allowed: f64,
    },
}

/// A campaign, held by this process: its ledger is locked while it lives.
pub struct CampaignController<'p> {
    ctx: &'p Context,
    spec: CampaignSpec,
    seed: u64,
    /// What every candidate is trained from.
    base: Model,
    /// Generation 0's champion: the base, or the adapter the campaign names.
    first: Model,
    ledger: StageLedger,
    parts: Parts<'p>,
}

/// The ledger directory of campaign `name` under the state root.
#[must_use]
pub fn campaign_dir(ctx: &Context, name: &str) -> PathBuf {
    ctx.root().path().join("campaigns").join(name)
}

fn padded(generation: u32) -> String {
    format!("g{generation:04}")
}

/// The ledger unit of `stage` of `generation`.
#[must_use]
pub fn unit(generation: u32, stage: &str) -> String {
    format!("{}/{stage}", padded(generation))
}

/// The generation and stage a unit name is of.
pub(super) fn parse_unit(name: &str) -> Option<(u32, &str)> {
    let (generation, stage) = name.split_once('/')?;
    Some((generation.strip_prefix('g')?.parse().ok()?, stage))
}

pub(super) fn record<T: for<'de> Deserialize<'de>>(commit: &Commit) -> Result<T, ServedError> {
    serde_json::from_value(commit.summary.clone()).map_err(|e| {
        ServedError::Config(format!(
            "the ledger's {} record is not of this build: {e}",
            commit.unit
        ))
    })
}

fn to_value<T: Serialize>(value: &T) -> Result<Value, ServedError> {
    serde_json::to_value(value).map_err(|e| ServedError::Config(format!("a stage record: {e}")))
}

/// Runs `work` as ledger unit `name` under `key` unless a sound commit of it
/// exists; the work's own error comes back as it was.
fn stage(
    ledger: &StageLedger,
    cancel: &CancelToken,
    name: &str,
    key: &KeyParts,
    work: impl FnOnce(&Path) -> Result<StageWork, ServedError>,
) -> Result<(Commit, bool), ServedError> {
    if cancel.is_cancelled() {
        return Err(ServedError::Cancelled);
    }
    let failed: RefCell<Option<ServedError>> = RefCell::new(None);
    let dir = ledger.dir().to_path_buf();
    let ran = ledger
        .run(name, &key.key(), OnCorrupt::Refuse, |_| {
            work(&dir).map_err(|e| {
                let text = e.to_string();
                *failed.borrow_mut() = Some(e);
                text
            })
        })
        .map_err(|e| match (failed.borrow_mut().take(), e) {
            (Some(own), LedgerError::Stage { .. }) => own,
            (_, other) => other.into(),
        })?;
    Ok((ran.commit, !ran.reused))
}

/// `path` relative to `root`, as a ledger output names it.
fn relative(root: &Path, path: &Path) -> Result<String, ServedError> {
    path.strip_prefix(root)
        .map(|p| p.to_string_lossy().into_owned())
        .map_err(|_| {
            ServedError::Config(format!(
                "{} is outside the campaign's directory {}",
                path.display(),
                root.display()
            ))
        })
}

fn stage_dir(root: &Path, generation: u32, stage: &str) -> Result<PathBuf, ServedError> {
    let dir = root.join(padded(generation)).join(stage);
    std::fs::create_dir_all(&dir)
        .map_err(|e| ServedError::Config(format!("cannot create {}: {e}", dir.display())))?;
    Ok(dir)
}

fn output<'c>(commit: &'c Commit, name: &str) -> Result<&'c Digest, ServedError> {
    commit
        .output(name)
        .ok_or_else(|| ServedError::Config(format!("the ledger's {} names no {name}", commit.unit)))
}

fn orchestrated(e: OrchestratorError) -> ServedError {
    match e {
        OrchestratorError::Cancelled | OrchestratorError::JobCancelled(_) => ServedError::Cancelled,
        other => ServedError::Orchestrator(Box::new(other)),
    }
}

impl<'p> CampaignController<'p> {
    /// Opens campaign `spec.name` under the state root, beginning it when its
    /// ledger is empty and continuing it otherwise.
    ///
    /// # Errors
    /// An invalid specification; a campaign begun under another
    /// specification, other rollouts or another examiner, or with another
    /// seed; fewer sealed slices than generations; another process holding
    /// the campaign; the ledger.
    pub fn open(
        ctx: &'p Context,
        spec: CampaignSpec,
        parts: Parts<'p>,
    ) -> Result<Self, ServedError> {
        spec.validate()?;
        let slices = parts.examiner.slices().len();
        if slices < spec.generations as usize {
            return Err(ServedError::Config(format!(
                "{} generations need as many fresh sealed slices; the examiner has {slices}",
                spec.generations
            )));
        }
        let dir = campaign_dir(ctx, &spec.name);
        std::fs::create_dir_all(&dir)
            .map_err(|e| ServedError::Config(format!("cannot create {}: {e}", dir.display())))?;
        let ledger = StageLedger::open(&dir)?;
        let identity = json!({
            "spec": spec.identity(),
            "rollouts": parts.rollouts.identity(),
            "examiner": parts.examiner.identity(),
        });
        let key = KeyParts::new(CAMPAIGN_UNIT).config(&identity);
        if ledger
            .commits()?
            .iter()
            .any(|c| c.unit == CAMPAIGN_UNIT && c.key != key.key())
        {
            return Err(ServedError::Config(format!(
                "campaign {} began under another specification, other rollouts or another \
                 examiner; a campaign is continued as it began: start a new one to change it",
                spec.name
            )));
        }
        let given = spec.seed;
        let (begun, _) = stage(&ledger, &parts.cancel, CAMPAIGN_UNIT, &key, |_| {
            let seed = given.unwrap_or_else(|| splinter_model::seed::draw().value());
            Ok(StageWork::new(json!({"identity": identity, "seed": seed})))
        })?;
        let seed = begun.summary["seed"]
            .as_u64()
            .ok_or_else(|| ServedError::Config("the ledger records no seed".into()))?;
        if given.is_some_and(|given| given != seed) {
            return Err(ServedError::Config(format!(
                "campaign {} began with seed {seed}, and is continued with it",
                spec.name
            )));
        }
        let base = base_of(&spec)?;
        let first = first_champion(&spec)?;
        Ok(Self {
            ctx,
            spec,
            seed,
            base,
            first,
            ledger,
            parts,
        })
    }

    /// The campaign's seed: given, or drawn when it began.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The ledger, held by this controller.
    #[must_use]
    pub fn ledger(&self) -> &StageLedger {
        &self.ledger
    }

    fn say(&self, line: &str) {
        if let Some(progress) = self.parts.progress {
            progress(&format!("{}: {line}", self.spec.name));
        }
    }

    /// Draw `salt` of `generation`, from the campaign's seed.
    fn seed_of(&self, generation: u32, salt: u64) -> u64 {
        splinter_core::rng::SplitMix64::new(
            self.seed
                ^ u64::from(generation).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ salt.wrapping_mul(0xBF58_476D_1CE4_E5B9),
        )
        .next_u64()
    }

    /// Moves the champion back to the model generation `to` made (0: the
    /// base) for `reason`, recorded as one more entry of its history.
    ///
    /// # Errors
    /// `to` was never a champion or is the champion now, the reason is
    /// empty, or the ledger.
    pub fn rollback(&self, to: u32, reason: &str) -> Result<Move, ServedError> {
        champion::rollback(&self.ledger, &self.first, to, reason)
    }

    /// What the campaign has done, read from its ledger.
    ///
    /// # Errors
    /// The ledger cannot be read, or holds a record of another build.
    pub fn report(&self) -> Result<CampaignReport, ServedError> {
        report(
            &self.spec.name,
            self.seed,
            &self.first,
            &self.ledger.commits()?,
        )
    }

    /// Runs generations until the campaign has run `spec.generations` of
    /// them or a budget stops it. A generation interrupted before resumes at
    /// its first stage without a record; every stage with one is reused.
    ///
    /// # Errors
    /// The first stage that failed (its attempt stays open in the ledger and
    /// the next run resumes it), or [`ServedError::Cancelled`].
    pub fn run(&mut self) -> Result<CampaignReport, ServedError> {
        let mut stopped = Stopped::Done;
        loop {
            let generation = next_generation(&self.ledger.commits()?);
            if generation > self.spec.generations {
                break;
            }
            if let Some(stop) = self.generation(generation)? {
                stopped = stop;
                break;
            }
        }
        let mut report = self.report()?;
        report.stopped = Some(stopped);
        Ok(report)
    }

    /// The limit a stage that spends `rollouts` more, or trains, would break.
    fn over_budget(&self, rollouts: u64, trains: bool) -> Result<Option<Stopped>, ServedError> {
        let budget = &self.spec.budget;
        let spent = spent(&self.ledger.commits()?)?;
        let over = |limit: &str, spent: f64, allowed: f64| {
            Some(Stopped::Budget {
                limit: limit.into(),
                spent,
                allowed,
            })
        };
        if let Some(allowed) = budget.rollouts.filter(|a| spent.rollouts + rollouts > *a) {
            return Ok(over("rollouts", spent.rollouts as f64, allowed as f64));
        }
        if let Some(allowed) = budget.tokens.filter(|a| spent.tokens >= *a) {
            return Ok(over("tokens", spent.tokens as f64, allowed as f64));
        }
        if let Some(allowed) = budget.gpu_hours.filter(|a| trains && spent.gpu_hours >= *a) {
            return Ok(over("gpu_hours", spent.gpu_hours, allowed));
        }
        Ok(None)
    }

    fn committed(&self, name: &str, key: &KeyParts) -> Result<bool, ServedError> {
        Ok(matches!(
            self.ledger.standing(name, &key.key())?,
            Standing::Committed(_)
        ))
    }

    /// Runs the stages of `generation` that have no record; `Some` when a
    /// budget stopped it before a stage.
    fn generation(&mut self, generation: u32) -> Result<Option<Stopped>, ServedError> {
        let commits = self.ledger.commits()?;
        let (champion, _) = champion::history(&self.first, &commits)?;
        // The tasks of a generation follow from the rollouts before it alone, so
        // a resumed generation draws the tasks it drew the first time.
        let rates = rates(&self.spec, &commits, generation)?;
        let Planned {
            tasks,
            selection,
            signal,
        } = curriculum::plan(
            &self.spec,
            &commits,
            &rates,
            generation,
            &*self.parts.rollouts,
            self.seed_of(generation, 3),
        )?;
        let transfer_seed = self.seed_of(generation, 4);
        let transfer = self
            .spec
            .rollout
            .transfer
            .zip(selection.as_ref())
            .filter(|(_, s)| !s.transfer.is_empty())
            .map(|(spec, s)| (spec.count, s.transfer.clone()));

        // Rollout: the champion on the practice tasks of the generation (every
        // one it has not mastered, or the capability curriculum's draw).
        let attempts = self.spec.rollout.attempts;
        let rollout_seed = self.seed_of(generation, 1);
        let mut config = json!({"attempts": attempts, "seed": rollout_seed, "tasks": tasks});
        if signal.is_some() {
            config["curriculum"] = json!("signal");
            config["reward"] = json!(self.spec.curate.reward);
        }
        if selection.is_some() {
            config["curriculum"] = json!("capability");
            config["transfer"] = json!(transfer.as_ref().map(
                |(count, ids)| json!({"capabilities": ids, "count": count, "seed": transfer_seed})
            ));
        }
        let key = KeyParts::new("rollout")
            .config(&config)
            .input("champion", &digest_of(&champion))
            .input("rollouts", &digest_of(&self.parts.rollouts.identity()));
        let name = unit(generation, "rollout");
        let drawn = transfer.as_ref().map_or(0, |(count, _)| *count) as u64;
        let planned = u64::from(attempts) * (tasks.len() as u64 + drawn);
        if !self.committed(&name, &key)? {
            if let Some(stop) = self.over_budget(planned, false)? {
                return Ok(Some(stop));
            }
        }
        let ctx = self.ctx;
        let rollouts = &mut *self.parts.rollouts;
        let cancel = self.parts.cancel.clone();
        let (rolled, _) = stage(&self.ledger, &cancel, &name, &key, |root| {
            let dir = stage_dir(root, generation, "rollout")?;
            let rolled = rollouts.rollout(
                ctx,
                &RolloutRequest {
                    generation,
                    champion: &champion,
                    tasks: &tasks,
                    attempts,
                    seed: rollout_seed,
                    transfer: transfer.as_ref().map(|(count, ids)| TransferRequest {
                        capabilities: ids,
                        count: *count,
                        seed: transfer_seed,
                    }),
                    dir: &dir,
                },
                &cancel,
            )?;
            let export = relative(root, &rolled.export)?;
            let record = RolloutRecord {
                champion: champion.clone(),
                tasks: tasks.clone(),
                rollouts: rolled.trials.len() as u64,
                trials: rolled.trials,
                tokens: rolled.tokens,
                selection: selection.clone(),
                signal: signal.clone(),
            };
            Ok(StageWork::new(to_value(&record)?).output("export", &export))
        })?;
        self.say(&format!("{name}: recorded"));

        // Ingest: the graded trials admitted as experience, idempotently.
        let key = KeyParts::new("ingest").input("export", output(&rolled, "export")?);
        let export = self.ledger.dir().join(padded(generation)).join("rollout");
        let export = rolled
            .outputs
            .iter()
            .find(|o| o.name == "export")
            .map_or(export, |o| self.ledger.dir().join(&o.path));
        let name = unit(generation, "ingest");
        stage(&self.ledger, &cancel, &name, &key, |_| {
            let ingested =
                ingest_bench_trials(ctx, std::slice::from_ref(&export), IngestOptions::default())
                    .map_err(orchestrated)?;
            let record = IngestRecord {
                train_set: ingested.train_set.clone(),
                experiences: ingested.admitted.values().sum::<usize>() + ingested.already_held,
                refused: ingested.refused.len(),
            };
            Ok(StageWork::new(to_value(&record)?))
        })?;

        // Curate: one dataset over every generation's experience so far.
        let commits = self.ledger.commits()?;
        let sets: Vec<SetId> = latest_per_generation(&commits, "ingest", generation)
            .into_iter()
            .map(record::<IngestRecord>)
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .filter_map(|r| r.train_set)
            .collect();
        let split_seed = self.seed_of(generation, 2);
        let key = KeyParts::new("curate")
            .config(&json!({"curate": self.spec.curate, "split_seed": split_seed, "sets": sets}));
        let name = unit(generation, "curate");
        let curate = &self.spec.curate;
        let material = self.parts.rollouts.material().map(Path::to_path_buf);
        let base_ref = self.base.reference()?;
        let (curated, _) = stage(&self.ledger, &cancel, &name, &key, |_| {
            let record = if sets.is_empty() {
                CurateRecord {
                    dataset: None,
                    records: 0,
                    sets: Vec::new(),
                }
            } else {
                let mut request = BuildRequest::new(sets.clone(), curate.view);
                request.max_block = curate.max_block;
                request.pack = curate.pack;
                request.rl.reward = curate.reward;
                if curate.reward == RewardPreset::ShapedPartial {
                    request.practice = material.iter().cloned().collect();
                }
                request.split_seed = Some(split_seed);
                request.base = Some(base_ref.clone());
                curated(ctx, &request, &sets)?
            };
            Ok(StageWork::new(to_value(&record)?))
        })?;
        let curated: CurateRecord = record(&curated)?;
        let Some(dataset) = curated.dataset.filter(|_| curated.records > 0) else {
            let reason = format!(
                "generation {generation}'s experience held no training material; the champion stays"
            );
            champion::keep(&self.ledger, generation, &champion, &reason, &Value::Null)?;
            self.say(&reason);
            return Ok(None);
        };

        // Train: a candidate from the base on the cumulative dataset, as a
        // served brain job brain resumes or reuses by its key.
        let train_seed = self.seed_of(generation, 3);
        let key = KeyParts::new("train")
            .config(&json!({"train": self.spec.train, "seed": train_seed, "dataset": dataset}))
            .input("base", &digest_of(&self.base));
        let name = unit(generation, "train");
        if !self.committed(&name, &key)? {
            if let Some(stop) = self.over_budget(0, true)? {
                return Ok(Some(stop));
            }
        }
        let resume_key = format!(
            "{}-{}-{}",
            self.spec.name,
            padded(generation),
            &key.key().as_str().rsplit(':').next().unwrap_or_default()[..16]
        );
        let (trainer, stopwatch, gpus) =
            (self.parts.trainer, self.parts.stopwatch, self.parts.gpus);
        let spec_train = self.spec.train.clone();
        let from = self.base.reference()?;
        let (trained, _) = stage(&self.ledger, &cancel, &name, &key, |_| {
            let started = stopwatch.seconds();
            let request = TrainRequest {
                datasets: vec![dataset.clone()],
                rehearsal: None,
                from: from.clone(),
                replay_fraction: 0.0,
                replay_policy: Default::default(),
                steps: spec_train.steps,
                rank: spec_train.rank,
                beta: None,
                tuning: {
                    let tuning = Tuning {
                        learning_rate: spec_train.learning_rate,
                        records_per_step: spec_train.records_per_step,
                        max_block: spec_train.max_block,
                        seed: Some(train_seed),
                        ..Tuning::default()
                    };
                    match spec_train.recipe {
                        Some(recipe) => recipe.apply(tuning),
                        None => tuning,
                    }
                },
            };
            let candidate = train::train_identified(ctx, &request, &resume_key, trainer, &cancel)
                .map_err(orchestrated)?;
            let seconds = (stopwatch.seconds() - started).max(0.0);
            let record = TrainRecord {
                candidate: Model {
                    generation,
                    model: splinter_core::model_ref::ModelRef::Local {
                        checkpoint: candidate.base.display().to_string(),
                        adapter: Some(candidate.adapter.display().to_string()),
                        context_tokens: None,
                    }
                    .to_string(),
                    candidate: Some(candidate.candidate.clone()),
                    adapter: Some(candidate.adapter_digest.clone()),
                },
                resume_key: resume_key.clone(),
                steps: candidate.steps,
                gpu_hours: seconds * f64::from(gpus) / 3600.0,
            };
            Ok(StageWork::new(to_value(&record)?))
        })?;
        let trained: TrainRecord = record(&trained)?;
        let candidate = trained.candidate;

        // Evaluate: the examiner's verdicts on the candidate against the
        // champion, over this generation's own sealed slice.
        let slices = self.parts.examiner.slices();
        let slice = slices
            .get(generation as usize - 1)
            .cloned()
            .ok_or_else(|| {
                ServedError::Config(format!(
                    "no sealed slice is left for generation {generation}"
                ))
            })?;
        let key = KeyParts::new("evaluate")
            .config(&json!({"slice": slice}))
            .input("candidate", &digest_of(&candidate))
            .input("champion", &digest_of(&champion))
            .input("examiner", &digest_of(&self.parts.examiner.identity()));
        let name = unit(generation, "evaluate");
        let training = self
            .ctx
            .datasets()
            .get(&splinter_core::dataset::DatasetId(
                Digest::parse(&dataset).map_err(|e| ServedError::Config(e.to_string()))?,
            ))
            .map_err(|e| ServedError::Config(format!("dataset {dataset}: {e}")))?
            .path;
        let examiner = &mut *self.parts.examiner;
        let campaign_name = self.spec.name.clone();
        let (examined, _) = stage(&self.ledger, &cancel, &name, &key, |root| {
            let dir = stage_dir(root, generation, "evaluate")?;
            let examined = examiner.examine(
                ctx,
                &ExamRequest {
                    campaign: &campaign_name,
                    generation,
                    candidate: &candidate,
                    predecessor: &champion,
                    slice: &slice,
                    training: &training,
                    dir: &dir,
                },
                &cancel,
            )?;
            Ok(StageWork::new(json!({"slice": slice}))
                .output("dev", &relative(root, &examined.dev)?)
                .output("retention", &relative(root, &examined.retention)?)
                .output("sealed", &relative(root, &examined.sealed)?))
        })?;

        // Gate: the one promotion gate, on the verdicts' signed aggregates.
        let pair = candidate.against(&champion);
        let key = KeyParts::new("gate")
            .config(&pair)
            .input("dev", output(&examined, "dev")?)
            .input("retention", output(&examined, "retention")?)
            .input("sealed", output(&examined, "sealed")?);
        let name = unit(generation, "gate");
        let root = self.ledger.dir().to_path_buf();
        let read = |name: &str| -> Result<Value, ServedError> {
            let path = examined
                .outputs
                .iter()
                .find(|o| o.name == name)
                .map(|o| root.join(&o.path))
                .ok_or_else(|| ServedError::Config(format!("no {name} verdict")))?;
            let text = std::fs::read_to_string(&path)
                .map_err(|e| ServedError::Config(format!("{}: {e}", path.display())))?;
            serde_json::from_str(&text)
                .map_err(|e| ServedError::Config(format!("{}: {e}", path.display())))
        };
        let verdict_key = self.parts.examiner.key().clone();
        let (gated, _) = stage(&self.ledger, &cancel, &name, &key, |_| {
            let (dev, retention, sealed) = (read("dev")?, read("retention")?, read("sealed")?);
            let promotion = decide(
                &verdict_key,
                &pair,
                Submitted {
                    dev: &dev,
                    retention: &retention,
                    sealed: &sealed,
                },
            );
            Ok(StageWork::new(to_value(&promotion)?))
        })?;
        let promotion: Promotion = record(&gated)?;

        // Promote or keep: the champion's move, the last record of the
        // generation.
        if promotion.promoted {
            champion::promote(
                &self.ledger,
                generation,
                &champion,
                &candidate,
                &gated.summary,
            )?;
            self.say(&format!(
                "generation {generation}: the candidate is the champion"
            ));
        } else {
            let reason = promotion
                .reasons
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("; ");
            champion::keep(&self.ledger, generation, &champion, &reason, &gated.summary)?;
            self.say(&format!(
                "generation {generation}: the champion stays ({reason})"
            ));
        }
        Ok(None)
    }
}

/// The dataset `request` builds over `sets`, recorded; an empty projection
/// is no dataset.
fn curated(
    ctx: &Context,
    request: &BuildRequest,
    sets: &[SetId],
) -> Result<CurateRecord, ServedError> {
    let (projection, _) = datasets::project_with(ctx, request, None).map_err(orchestrated)?;
    if projection.records.is_empty() {
        return Ok(CurateRecord {
            dataset: None,
            records: 0,
            sets: sets.to_vec(),
        });
    }
    let stored = datasets::store_dataset(ctx, &projection, false).map_err(orchestrated)?;
    Ok(CurateRecord {
        dataset: Some(stored.id.0.to_string()),
        records: projection.records.len(),
        sets: sets.to_vec(),
    })
}
