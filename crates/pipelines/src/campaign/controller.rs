// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that run model learning as one
// continuous, resumable campaign with dynamic sampling, replay and a
// validation gate with rollback, for its clients. If your team needs expertise
// in on-policy training loops for language-model agents you can procure our
// services by sending an email to info@swedishembedded.com.

//! The campaign controller: one arm per call, every step that cost something
//! saved before the next begins. See the module documentation for the arm.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use serde::Serialize;
use serde_json::json;
use splinter_core::clock::Clock;
use splinter_core::digest::digest_of;
use splinter_core::rng::SplitMix64;
use splinter_eval::campaign::gate::Decision;
use splinter_eval::campaign::pass_rate::PassRates;
use splinter_eval::campaign::plan::{plan_arm, ArmGoal, ArmRefused, ExamShape, PlannedArm};
use splinter_eval::campaign::sampling::AllFailStrategy;
use splinter_eval::campaign::signals::{sampled_entropy, Finding, StepSignals};
use splinter_eval::significance::Significance;
use splinter_store::campaign::{CampaignStore, Cost, LedgerEntry, Loaded, Verdict};
use splinter_store::StateRoot;

use super::config::{CampaignConfig, LearningStrategy};
use super::environments::EnvironmentLabels;
use super::gc::Checkpoint;
use super::records::RecordCounts;
use super::replay::ReplayBuffer;
use super::state::{CampaignState, PendingCandidate, STATE_SCHEMA};
use super::types::{
    AllFailHandler, CampaignError, RolloutCollector, RolloutGroup, TrainRequest, Trainer, Validator,
};
use crate::preflight;

/// Seconds elapsed on whatever clock measures GPU-hours: wall time in a run, a
/// pinned value in a test.
pub trait Stopwatch {
    /// Seconds since the clock started.
    fn seconds(&self) -> f64;
}

/// The wall clock as a [`Stopwatch`].
pub struct WallClock(Instant);

impl WallClock {
    /// A stopwatch started now.
    #[must_use]
    pub fn start() -> Self {
        Self(Instant::now())
    }
}

impl Stopwatch for WallClock {
    fn seconds(&self) -> f64 {
        self.0.elapsed().as_secs_f64()
    }
}

/// What the controller is made of.
pub struct Parts<'p> {
    /// Where rollouts come from.
    pub collector: &'p mut dyn RolloutCollector,
    /// The call into brain.
    pub trainer: &'p mut dyn Trainer,
    /// The dev exam.
    pub validator: &'p mut dyn Validator,
    /// Teacher and decomposition, for the strategies that need them.
    pub handler: Option<&'p mut dyn AllFailHandler>,
    /// The paired test the gates rest on.
    pub significance: &'p dyn Significance,
    /// Timestamps for the ledger.
    pub clock: &'p dyn Clock,
    /// GPU-hours.
    pub stopwatch: &'p dyn Stopwatch,
    /// Which environment each task belongs to, when the tasks come from the
    /// environment factory: enables reward provenance in the ledger and the
    /// divergence alarm.
    pub environments: Option<&'p dyn EnvironmentLabels>,
    /// Where a one-line account of each stage goes while an arm runs, for a
    /// person watching a multi-hour campaign; `None` stays silent.
    pub progress: Option<&'p dyn Fn(&str)>,
}

/// What one arm did.
#[derive(Clone, Debug, Serialize)]
pub struct ArmReport {
    /// The arm.
    pub arm: String,
    /// Its place in the campaign, from 1.
    pub generation: u32,
    /// The strategy it ran.
    pub strategy: &'static str,
    /// How it ended.
    pub verdict: Verdict,
    /// The step's signals; absent for an arm that never trained.
    pub signals: Option<StepSignals>,
    /// The gate's numbers; absent for an arm that never reached it.
    pub decision: Option<Decision>,
    /// What records the batch became.
    pub records: RecordCounts,
    /// Findings of the signal bounds.
    pub findings: Vec<Finding>,
    /// GPU-hours the arm used.
    pub gpu_hours: f64,
    /// Whether the campaign should not continue.
    pub stop: bool,
}

/// What a call to [`CampaignController::run`] did.
#[derive(Clone, Debug, Default, Serialize)]
pub struct CampaignReport {
    /// The arms run, in order.
    pub arms: Vec<ArmReport>,
}

/// A running campaign.
pub struct CampaignController<'p> {
    pub(super) store: CampaignStore,
    pub(super) config: CampaignConfig,
    pub(super) parts: Parts<'p>,
    pub(super) adapters: PathBuf,
    pub(super) loaded: Loaded<CampaignState>,
}

/// Which arm an entry or a report is about.
pub(super) struct ArmId {
    pub(super) name: String,
    pub(super) generation: u32,
    pub(super) strategy: LearningStrategy,
}

/// What an arm ends with, as the ledger records it.
pub(super) struct Outcome {
    pub(super) verdict: Verdict,
    pub(super) cost: Cost,
    pub(super) discoveries: Option<f64>,
    pub(super) measurements: BTreeMap<String, f64>,
}

/// What an arm that never reached a candidate had used.
pub(super) struct Spent {
    pub(super) informative_fraction: Option<f64>,
    pub(super) gpu_hours: f64,
    pub(super) rollouts: u64,
    pub(super) counts: RecordCounts,
}

/// The discordance below which an observed value is not trusted to plan with.
const DISCORDANCE_FLOOR: f64 = 0.02;

/// Everything a batch produced before training.
pub(super) struct Collected {
    pub(super) groups: Vec<RolloutGroup>,
    pub(super) informative: Vec<usize>,
    pub(super) all_fail: Vec<usize>,
    pub(super) informative_fraction: Option<f64>,
    pub(super) rollouts: u64,
    pub(super) shortfall: usize,
}

impl<'p> CampaignController<'p> {
    /// Opens campaign `config.name` under the state `root`, creating it at `seed`
    /// when it does not exist and resuming it when it does. Adapters live
    /// under `adapters`.
    ///
    /// # Errors
    /// An invalid configuration; a strategy that needs a handler none was
    /// given; a campaign begun under another configuration or layout; a store
    /// failure; a policy the collector cannot serve.
    pub fn open(
        root: &StateRoot,
        config: CampaignConfig,
        seed: u64,
        parts: Parts<'p>,
        adapters: PathBuf,
    ) -> Result<Self, CampaignError> {
        config.validate()?;
        let needs_handler = matches!(
            config.all_fail,
            AllFailStrategy::Teacher | AllFailStrategy::Decompose
        );
        if needs_handler && parts.handler.is_none() {
            return Err(CampaignError::Config(format!(
                "all_fail is {:?}, which needs a teacher or decomposition handler, and none was given",
                config.all_fail
            )));
        }
        let store = CampaignStore::new(root, &config.name)?;
        let digest = digest_of(&json!({
            "config": config,
            "collector": parts.collector.describe(),
            "trainer": parts.trainer.describe(),
            "validator": parts.validator.describe(),
        }));
        let loaded = match store.load::<CampaignState>()? {
            Some(loaded) => {
                if loaded.state.schema != STATE_SCHEMA {
                    return Err(CampaignError::Config(format!(
                        "campaign {} was saved with layout {}, this build reads {STATE_SCHEMA}",
                        config.name, loaded.state.schema
                    )));
                }
                if loaded.state.config_digest != digest {
                    return Err(CampaignError::Config(format!(
                        "campaign {} began under another configuration ({} against {digest}); \
                         a campaign is continued as it began, start a new one to change it",
                        config.name, loaded.state.config_digest
                    )));
                }
                loaded
            }
            None => {
                let rates = PassRates::new(config.retire_at, config.calibration_k)
                    .map_err(|e| CampaignError::Config(e.to_string()))?;
                let state = CampaignState::new(
                    &config.name,
                    digest,
                    seed,
                    rates,
                    ReplayBuffer::new(config.replay.capacity),
                );
                let saved = store.save(None, &state, &parts.clock.utc_now())?;
                Loaded {
                    state,
                    digest: saved,
                    version: 1,
                }
            }
        };
        parts.collector.set_policy(&loaded.state.policy)?;
        std::fs::create_dir_all(&adapters).map_err(|e| {
            CampaignError::Config(format!("cannot create {}: {e}", adapters.display()))
        })?;
        Ok(Self {
            store,
            config,
            parts,
            adapters,
            loaded,
        })
    }

    /// The campaign as of its last save.
    #[must_use]
    pub fn state(&self) -> &CampaignState {
        &self.loaded.state
    }

    /// The campaign's ledger.
    ///
    /// # Errors
    /// A store failure.
    pub fn ledger(&self) -> Result<Vec<LedgerEntry>, CampaignError> {
        Ok(self.store.ledger()?)
    }

    /// Reports one stage to the progress sink.
    pub(super) fn say(&self, line: &str) {
        if let Some(progress) = self.parts.progress {
            progress(line);
        }
    }

    pub(super) fn save(&mut self) -> Result<(), CampaignError> {
        let at = self.parts.clock.utc_now();
        let digest = self
            .store
            .save(Some(&self.loaded.digest), &self.loaded.state, &at)?;
        self.loaded.digest = digest;
        self.loaded.version += 1;
        Ok(())
    }

    pub(super) fn strategy(&self) -> LearningStrategy {
        let ladder = &self.config.ladder;
        ladder[self.loaded.state.strategy_index.min(ladder.len() - 1)]
    }

    pub(super) fn pool(&self) -> Vec<String> {
        let mut pool = self.parts.collector.tasks();
        pool.extend(self.loaded.state.extra_tasks.iter().cloned());
        pool.sort();
        pool.dedup();
        // A quarantined environment is never drawn from again.
        if let Some(labels) = self.parts.environments {
            pool.retain(|task| {
                labels
                    .environment_of(task)
                    .is_none_or(|env| !self.loaded.state.quarantined.contains(&env))
            });
        }
        pool
    }

    pub(super) fn rng(&self, generation: u32, salt: u64) -> SplitMix64 {
        let mixed = self.loaded.state.seed
            ^ u64::from(generation).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ salt.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        SplitMix64::new(mixed)
    }

    pub(super) fn entry(&self, id: &ArmId, outcome: Outcome) -> LedgerEntry {
        let sources = self
            .parts
            .environments
            .map(EnvironmentLabels::sources)
            .unwrap_or_default();
        LedgerEntry {
            campaign: self.config.name.clone(),
            arm: id.name.clone(),
            generation: id.generation,
            strategy: id.strategy.name().to_string(),
            hypothesis: format!("{}: {}", id.strategy.name(), self.config.hypothesis),
            config_digest: self.loaded.state.config_digest.clone(),
            at: self.parts.clock.utc_now(),
            cost: outcome.cost,
            discoveries: outcome.discoveries,
            verdict: outcome.verdict,
            reward_sources: sources.clone(),
            verdict_label: splinter_core::verifiability::verdict_label(
                sources.iter().map(|source| source.class),
            )
            .map(str::to_string),
            measurements: outcome.measurements,
        }
    }

    /// Runs up to `arms` arms, stopping early when an arm says the campaign
    /// should not continue.
    ///
    /// # Errors
    /// The first failure of an arm; arms already finished stay saved.
    pub fn run(&mut self, arms: u32) -> Result<CampaignReport, CampaignError> {
        let mut report = CampaignReport::default();
        for _ in 0..arms {
            let arm = self.arm()?;
            let stop = arm.stop;
            report.arms.push(arm);
            if stop {
                break;
            }
        }
        Ok(report)
    }

    fn plan(&self, generation: u32) -> Result<PlannedArm, ArmRefused> {
        let kept = self.parts.validator.retention_units();
        if kept > 0 && self.config.retention_loss_to_detect > 0.0 {
            let alpha = splinter_eval::campaign::plan::spent_alpha(
                self.config.gate.total_alpha,
                generation,
            );
            let needed = splinter_eval::campaign::plan::retention_units_needed(
                self.config.retention_loss_to_detect,
                alpha,
                2000,
                self.parts.significance,
            );
            if needed.is_none_or(|n| kept < n) {
                return Err(ArmRefused::Invalid(format!(
                    "the retention set has {kept} tasks, too few to detect a loss of {} of them at level {alpha:.4} (needs {})",
                    self.config.retention_loss_to_detect,
                    needed.map_or("more than 2000".to_string(), |n| n.to_string())
                )));
            }
        }
        let prior = self.config.expected_discordance;
        let discordance = match self.loaded.state.last_discordance {
            Some(seen) if seen >= self.config.target_effect.max(DISCORDANCE_FLOOR) => seen,
            _ => prior,
        };
        plan_arm(
            &ExamShape {
                units: self.parts.validator.units(),
                discordance,
            },
            &ArmGoal {
                total_alpha: self.config.gate.total_alpha,
                ..ArmGoal::new(self.config.target_effect, generation)
            },
            self.parts.significance,
        )
    }

    fn arm(&mut self) -> Result<ArmReport, CampaignError> {
        if self.loaded.state.pending.is_some() {
            return self.gate_pending();
        }
        let generation = self.loaded.state.arms + 1;
        let strategy = self.strategy();
        let id = ArmId {
            name: format!("arm-{generation:04}-{}", strategy.name()),
            generation,
            strategy,
        };
        let started_at = self.parts.clock.utc_now();
        let began = self.parts.stopwatch.seconds();
        preflight::check(&self.config.preflight)?;
        let planned = match self.plan(generation) {
            Ok(planned) => planned,
            Err(refused) => {
                let entry = self.entry(
                    &id,
                    Outcome {
                        verdict: Verdict::Refused {
                            reason: refused.to_string(),
                        },
                        cost: Cost::default(),
                        discoveries: None,
                        measurements: BTreeMap::new(),
                    },
                );
                self.store.record(&entry)?;
                return Err(refused.into());
            }
        };
        self.say(&format!(
            "{}: planned (exam detects a lead of {:.3} at level {:.4})",
            id.name, planned.minimum_detectable_effect, planned.alpha
        ));
        let running = self.entry(
            &id,
            Outcome {
                verdict: Verdict::Running,
                cost: Cost::default(),
                discoveries: None,
                measurements: BTreeMap::from([(
                    "planned_mde".to_string(),
                    planned.minimum_detectable_effect,
                )]),
            },
        );
        self.store.record(&running)?;

        self.calibrate(generation)?;
        let collected = self.collect(generation)?;
        let mut measurements = BTreeMap::from([
            ("planned_mde".to_string(), planned.minimum_detectable_effect),
            ("planned_power".to_string(), planned.power_at_target),
            ("planned_alpha".to_string(), planned.alpha),
            ("shortfall_groups".to_string(), collected.shortfall as f64),
        ]);
        let retired = self.loaded.state.rates.counts(&self.pool()).retired;
        measurements.insert("retired_tasks".into(), retired as f64);

        self.say(&format!(
            "{}: {} groups, {} informative ({} short of the batch), {} rollouts",
            id.name,
            collected.groups.len(),
            collected.informative.len(),
            collected.shortfall,
            collected.rollouts
        ));
        let records = self.records(generation, strategy, &collected)?;
        let trainable = records.counts.policy_gradient + records.counts.distilled > 0;
        let rollouts = collected.rollouts;
        let spent = |this: &Self| (this.parts.stopwatch.seconds() - began).max(0.0);
        if !trainable {
            let reason = format!(
                "no group among {} carried a reward spread or a teacher signal",
                collected.groups.len()
            );
            let used = Spent {
                informative_fraction: collected.informative_fraction,
                gpu_hours: spent(self) * f64::from(self.config.gpus) / 3600.0,
                rollouts,
                counts: records.counts,
            };
            return self.close_without_candidate(
                &id,
                Verdict::RolledBack { reason },
                used,
                measurements,
            );
        }

        let train = TrainRequest {
            records: records.records,
            continue_from: self.loaded.state.policy.adapter.clone(),
            out_dir: self.adapters.join(&id.name),
            seed: self.rng(generation, 1).next_u64(),
        };
        self.say(&format!(
            "{}: training on {} records",
            id.name,
            train.records.len()
        ));
        let trained = self.parts.trainer.train(&train)?;
        let mean_reward = mean(
            collected
                .informative
                .iter()
                .flat_map(|&i| &collected.groups[i].rollouts)
                .map(|r| r.reward),
        );
        // The sampler's own log-probabilities when the rollouts carry them,
        // else the trainer's estimate over the same on-policy tokens.
        let entropy = sampled_entropy(
            collected
                .informative
                .iter()
                .flat_map(|&i| &collected.groups[i].rollouts)
                .filter_map(|r| r.trace.as_ref())
                .flat_map(|t| &t.sequences)
                .flat_map(|s| s.generation_log_probs.iter().map(|&lp| f64::from(lp))),
        )
        .or(trained.report.entropy);
        let signals = StepSignals {
            step: generation,
            policy_version: self.loaded.state.policy.version,
            informative_fraction: collected.informative_fraction,
            kl: trained.report.kl,
            clip_fraction: trained.report.clip_fraction,
            entropy,
            mean_reward,
        };
        let findings = self.config.bounds.assess(&signals);
        let checkpoint = Checkpoint {
            digest: trained.digest.clone(),
            path: trained.dir.clone(),
            generation,
            bytes: trained.bytes,
        };
        self.loaded.state.checkpoints.push(checkpoint);
        self.loaded.state.signals.push(signals);
        let gpu_seconds = spent(self) * f64::from(self.config.gpus);
        if findings.iter().any(Finding::halts) {
            let reason = format!("signals out of bounds: {findings:?}");
            let used = Spent {
                informative_fraction: collected.informative_fraction,
                gpu_hours: gpu_seconds / 3600.0,
                rollouts,
                counts: records.counts,
            };
            let report =
                self.close_without_candidate(&id, Verdict::Halted { reason }, used, measurements)?;
            return Ok(ArmReport {
                signals: Some(signals),
                findings,
                stop: true,
                ..report
            });
        }
        measurements.extend(signal_measurements(&signals));
        measurements.insert(
            "steps_skipped_guard".into(),
            trained.report.skipped_guard as f64,
        );
        self.loaded.state.pending = Some(PendingCandidate {
            arm: id.name.clone(),
            adapter: trained.adapter,
            dir: trained.dir,
            digest: trained.digest,
            bytes: trained.bytes,
            report: trained.report,
            signals,
            gpu_seconds,
            rollouts,
            generated_tokens: None,
            started_at,
            replay: self.passing(&collected),
            measurements,
            env_rewards: self.env_rewards(&collected),
        });
        self.save()?;
        let mut report = self.gate_pending()?;
        report.records = records.counts;
        report.findings = findings;
        Ok(report)
    }
}

fn mean(values: impl Iterator<Item = f64>) -> Option<f64> {
    let (sum, n) = values.fold((0.0, 0usize), |(s, n), v| (s + v, n + 1));
    (n > 0).then(|| sum / n as f64)
}

/// The step's signals as ledger measurements; an absent signal stays absent.
fn signal_measurements(signals: &StepSignals) -> BTreeMap<String, f64> {
    [
        ("informative_fraction", signals.informative_fraction),
        ("kl", signals.kl),
        ("clip_fraction", signals.clip_fraction),
        ("entropy", signals.entropy),
        ("mean_reward", signals.mean_reward),
    ]
    .into_iter()
    .filter_map(|(name, value)| value.map(|v| (name.to_string(), v)))
    .collect()
}
