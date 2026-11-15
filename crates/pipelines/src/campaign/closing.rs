// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that run model learning as one
// continuous, resumable campaign with dynamic sampling, replay and a
// validation gate with rollback, for its clients. If your team needs expertise
// in on-policy training loops for language-model agents you can procure our
// services by sending an email to info@swedishembedded.com.

//! Closing an arm: the dev gate, adopting or rolling back a candidate, the
//! accounting, the plateau and the collection of superseded checkpoints.

use std::collections::BTreeMap;

use splinter_core::digest::Digest;
use splinter_core::rng::SplitMix64;
use splinter_core::verifiability::VerifierClass;
use splinter_eval::campaign::dev_pool::DevPool;
use splinter_eval::campaign::evidence::{Budget, EvidenceGate};
use splinter_eval::campaign::gate::{decide, decide_evidence, Action, Decision, DevResult};
use splinter_eval::campaign::plateau::{assess, ArmCost, Trend};
use splinter_knowledge::factory::alarm::{DivergenceAlarm, Observation, Signal};
use splinter_store::campaign::{Cost, Verdict};

use super::config::Acceptance;
use super::controller::{ArmId, ArmReport, CampaignController, Outcome, Spent};
use super::gc::{self, Checkpoint, Keep};
use super::records::RecordCounts;
use super::state::{EvidenceSpent, PendingCandidate};
use super::types::{CampaignError, PolicyRef};

/// Mixed into the campaign seed so an arm's dev draw is a stream of its own.
const DEV_DRAW_SALT: u64 = 0xD0E5_5EED_DEF0_0D02;

impl CampaignController<'_> {
    /// Ends an arm that has no candidate to gate: counted, accounted, saved.
    pub(super) fn close_without_candidate(
        &mut self,
        id: &ArmId,
        verdict: Verdict,
        used: Spent,
        mut measurements: BTreeMap<String, f64>,
    ) -> Result<ArmReport, CampaignError> {
        if let Some(fraction) = used.informative_fraction {
            measurements.insert("informative_fraction".into(), fraction);
        }
        let halted = matches!(verdict, Verdict::Halted { .. });
        let entry = self.entry(
            id,
            Outcome {
                verdict: verdict.clone(),
                cost: Cost {
                    gpu_hours: Some(used.gpu_hours),
                    rollouts: Some(used.rollouts),
                    ..Cost::default()
                },
                discoveries: Some(0.0),
                measurements,
            },
        );
        self.finish_arm(&id.name, used.gpu_hours, 0.0, false)?;
        self.store.record(&entry)?;
        self.save()?;
        Ok(ArmReport {
            arm: id.name.clone(),
            generation: id.generation,
            strategy: id.strategy.name(),
            verdict,
            signals: None,
            decision: None,
            records: used.counts,
            findings: Vec::new(),
            gpu_hours: used.gpu_hours,
            stop: halted || self.exhausted(),
        })
    }

    /// Counts the arm, reads the plateau, collects superseded checkpoints.
    fn finish_arm(
        &mut self,
        arm: &str,
        gpu_hours: f64,
        discoveries: f64,
        advanced: bool,
    ) -> Result<(), CampaignError> {
        let state = &mut self.loaded.state;
        state.arms += 1;
        state.gpu_hours += gpu_hours;
        state.idle_arms = if advanced { 0 } else { state.idle_arms + 1 };
        state.history.push(ArmCost {
            arm: arm.to_string(),
            gpu_hours,
            discoveries,
        });
        if let Trend::Plateau { .. } = assess(&state.history, &self.config.plateau) {
            let last = self.config.ladder.len() - 1;
            state.strategy_index = (state.strategy_index + 1).min(last);
        }
        let champion = state.policy.digest.clone();
        let protected: Vec<&Digest> = champion.iter().collect();
        let keep = Keep {
            recent: self.config.keep_recent_checkpoints,
        };
        let doomed: Vec<Checkpoint> = gc::plan(&state.checkpoints, &protected, &keep)
            .into_iter()
            .cloned()
            .collect();
        let refs: Vec<&Checkpoint> = doomed.iter().collect();
        let dropped = gc::collect(&self.adapters, &refs)?;
        state
            .checkpoints
            .retain(|c| !doomed.iter().any(|d| d.digest == c.digest));
        state.dropped.extend(dropped);
        Ok(())
    }

    /// Whether the ladder is spent and arms keep failing to advance.
    pub(super) fn exhausted(&self) -> bool {
        let state = &self.loaded.state;
        state.strategy_index + 1 >= self.config.ladder.len()
            && state.idle_arms >= self.config.max_idle_arms
    }

    /// Feeds each environment's training reward and dev pass rate of this arm
    /// to the divergence alarm; an environment whose reward climbs while its
    /// dev measure does not is quarantined. Returns one reason per
    /// quarantined environment.
    fn check_divergence(
        &mut self,
        generation: u32,
        pending: &PendingCandidate,
        dev: &DevResult,
    ) -> Result<Vec<String>, CampaignError> {
        let Some(labels) = self.parts.environments else {
            return Ok(Vec::new());
        };
        let sources = labels.sources();
        let mut reasons = Vec::new();
        for (env, reward) in &pending.env_rewards {
            let passes: Vec<bool> = dev
                .dev
                .iter()
                .filter(|(task, _)| labels.environment_of(task).as_deref() == Some(env.as_str()))
                .map(|(_, passed)| *passed)
                .collect();
            let heldout = (!passes.is_empty())
                .then(|| passes.iter().filter(|p| **p).count() as f64 / passes.len() as f64);
            let history = self
                .loaded
                .state
                .observations
                .entry(env.clone())
                .or_default();
            history.push(Observation {
                step: u64::from(generation),
                reward: *reward,
                heldout,
            });
            let class = sources
                .iter()
                .find(|source| &source.environment == env)
                .map_or(VerifierClass::Objective, |source| source.class);
            let mut alarm = DivergenceAlarm::new(self.config.divergence, class);
            let mut last = Signal::Steady;
            for observation in history.iter() {
                last = alarm.observe(*observation);
            }
            if let Signal::Diverging(alert) = last {
                let reason = format!(
                    "{env}: training reward rose {:.3} over the window while the dev measure moved {:.3}{}",
                    alert.reward_rise,
                    alert.heldout_rise,
                    alert.label.map_or(String::new(), |l| format!(" ({l})"))
                );
                labels.quarantine(env, &reason)?;
                self.loaded.state.quarantined.push(env.clone());
                reasons.push(reason);
            }
        }
        Ok(reasons)
    }

    /// Runs the dev gate on the pending candidate, adopts or rolls it back,
    /// and accounts the arm.
    pub(super) fn gate_pending(&mut self) -> Result<ArmReport, CampaignError> {
        let Some(pending) = self.loaded.state.pending.clone() else {
            return Err(CampaignError::Config("no candidate is pending".into()));
        };
        let generation = self.loaded.state.arms + 1;
        let strategy = self.strategy();
        let began = self.parts.stopwatch.seconds();
        let candidate_policy = PolicyRef {
            version: self.loaded.state.policy.version + 1,
            adapter: Some(pending.adapter.clone()),
            digest: Some(pending.digest.clone()),
        };
        let (decision, candidate) = match self.config.acceptance {
            Acceptance::Greedy => self.gate_greedy(generation, &candidate_policy)?,
            Acceptance::Evidence {
                gate,
                draw,
                horizon,
            } => self.gate_evidence(&gate, (draw, horizon), generation, &candidate_policy)?,
        };
        let diverged = self.check_divergence(generation, &pending, &candidate)?;
        let advance = decision.action == Action::Advance && diverged.is_empty();
        self.say(&format!(
            "{}: dev {:?} vs {:?} (won {}, lost {}, p {:.4}, level {:.4}): {}",
            pending.arm,
            decision.candidate_rate,
            decision.champion_rate,
            decision.wins,
            decision.losses,
            decision.p_value,
            decision.alpha,
            if advance { "advance" } else { "roll back" }
        ));
        let verdict = if advance {
            Verdict::Accepted {
                significant: decision.significant,
            }
        } else if !diverged.is_empty() {
            Verdict::Halted {
                reason: format!(
                    "the reward stopped meaning anything: {}",
                    diverged.join("; ")
                ),
            }
        } else {
            Verdict::RolledBack {
                reason: decision.reason.clone(),
            }
        };
        let gate_hours = (self.parts.stopwatch.seconds() - began).max(0.0)
            * f64::from(self.config.gpus)
            / 3600.0;
        let gpu_hours = pending.gpu_seconds / 3600.0 + gate_hours;
        let discoveries = if advance { decision.wins as f64 } else { 0.0 };
        let mut measurements = pending.measurements.clone();
        measurements.extend([
            ("dev_wins".to_string(), decision.wins as f64),
            ("dev_losses".to_string(), decision.losses as f64),
            ("dev_compared".to_string(), decision.compared as f64),
            ("dev_p_value".to_string(), decision.p_value),
            ("gate_alpha".to_string(), decision.alpha),
            (
                "retention_losses".to_string(),
                decision.retention_losses as f64,
            ),
        ]);
        if let Some(rate) = decision.candidate_rate {
            measurements.insert("dev_rate_candidate".into(), rate);
        }
        if let Some(rate) = decision.champion_rate {
            measurements.insert("dev_rate_champion".into(), rate);
        }
        if let Some(evidence) = &decision.evidence {
            measurements.extend([
                ("evidence_e_value".to_string(), evidence.e_value),
                ("evidence_threshold".to_string(), evidence.threshold),
                (
                    "evidence_budget_left".to_string(),
                    evidence.allocation.budget_after,
                ),
                ("evidence_charge".to_string(), evidence.allocation.charge),
                ("dev_drawn".to_string(), evidence.drawn as f64),
                ("dev_spent".to_string(), evidence.spent.len() as f64),
                ("dev_observed".to_string(), evidence.observed as f64),
                ("dev_reused".to_string(), evidence.reused as f64),
            ]);
            if let Some(zeta) = evidence.zeta_mean {
                measurements.insert("evidence_zeta_mean".into(), zeta);
            }
        }
        {
            let state = &mut self.loaded.state;
            if decision.compared > 0 {
                state.last_discordance =
                    Some((decision.wins + decision.losses) as f64 / decision.compared as f64);
            }
            if let Some(evidence) = &decision.evidence {
                let spent = state.evidence.get_or_insert_with(|| EvidenceSpent {
                    budget: Budget::start(&evidence.gate),
                    used: Default::default(),
                });
                spent.budget.settle(&evidence.allocation, advance);
                spent.used.extend(evidence.spent.iter().cloned());
            }
            if advance {
                state.policy = candidate_policy;
                state.champion_dev = Some(candidate);
                state.replay.push(generation, pending.replay.clone());
            }
            state.pending = None;
        }
        if advance {
            self.parts.collector.set_policy(&self.loaded.state.policy)?;
        }
        let cost = Cost {
            gpu_hours: Some(gpu_hours),
            rollouts: Some(pending.rollouts),
            generated_tokens: pending.generated_tokens,
            train_steps: Some(u64::from(pending.report.steps)),
        };
        self.finish_arm(&pending.arm, gpu_hours, discoveries, advance)?;
        let id = ArmId {
            name: pending.arm.clone(),
            generation,
            strategy,
        };
        let entry = self.entry(
            &id,
            Outcome {
                verdict: verdict.clone(),
                cost,
                discoveries: Some(discoveries),
                measurements,
            },
        );
        self.store.record(&entry)?;
        self.save()?;
        Ok(ArmReport {
            arm: pending.arm,
            generation,
            strategy: strategy.name(),
            verdict,
            signals: Some(pending.signals),
            decision: Some(decision),
            records: RecordCounts::default(),
            findings: Vec::new(),
            gpu_hours,
            stop: self.exhausted(),
        })
    }

    /// The greedy dev gate: the champion graded once on every dev task, the
    /// candidate on every dev task.
    fn gate_greedy(
        &mut self,
        generation: u32,
        candidate_policy: &PolicyRef,
    ) -> Result<(Decision, DevResult), CampaignError> {
        if self.loaded.state.champion_dev.is_none() {
            let champion = self.parts.validator.validate(&self.loaded.state.policy)?;
            self.loaded.state.champion_dev = Some(champion);
            self.save()?;
        }
        let candidate = self.parts.validator.validate(candidate_policy)?;
        let champion = self.loaded.state.champion_dev.clone().unwrap_or_default();
        let decision = decide(
            &self.config.gate,
            generation,
            &champion,
            &candidate,
            self.parts.significance,
        );
        Ok((decision, candidate))
    }

    /// The evidence gate: the arm's level from the budget the state holds,
    /// `draw` dev tasks drawn from the dev set (a seed derived from the
    /// campaign's and the arm's, so an arm resumed after a crash draws the
    /// same), the champion and the candidate graded on those alone, and the
    /// e-process decides. Nothing is graded at level zero. The whole draw is
    /// graded at once; the decision is the one a batched grading stopped at
    /// the crossing would reach.
    fn gate_evidence(
        &mut self,
        gate: &EvidenceGate,
        (draw, horizon): (usize, u32),
        generation: u32,
        candidate_policy: &PolicyRef,
    ) -> Result<(Decision, DevResult), CampaignError> {
        // A resumed campaign runs under the configuration it began with (its
        // digest is checked at open), so the budget it saved is this gate's.
        let (budget, used) = self.loaded.state.evidence.clone().map_or_else(
            || (Budget::start(gate), Default::default()),
            |s| (s.budget, s.used),
        );
        let allocation = budget.allocate(gate, horizon);
        let pool = DevPool::new(self.parts.validator.dev_tasks(), &used);
        let mut rng = SplitMix64::new(
            self.loaded.state.seed ^ DEV_DRAW_SALT ^ u64::from(generation).rotate_left(32),
        );
        let drawn = pool
            .draw(draw, &mut rng)
            .map_err(|e| CampaignError::Config(e.to_string()))?;
        let (champion, candidate, graded) = if allocation.alpha > 0.0 {
            let ids: Vec<String> = drawn.iter().map(|d| d.task.clone()).collect();
            let champion = self
                .parts
                .validator
                .validate_on(&self.loaded.state.policy, &ids)?;
            let candidate = self.parts.validator.validate_on(candidate_policy, &ids)?;
            (champion, candidate, drawn.len())
        } else {
            (DevResult::default(), DevResult::default(), 0)
        };
        let decision = decide_evidence(
            &self.config.gate,
            gate,
            &allocation,
            (&drawn, graded),
            &champion,
            &candidate,
            self.parts.significance,
        )
        .map_err(|e| CampaignError::Config(e.to_string()))?;
        Ok((decision, candidate))
    }
}
