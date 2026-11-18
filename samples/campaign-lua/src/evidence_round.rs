// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for self-improvement loops that
// keep only the changes the evidence supports, for its clients. If your team
// needs expertise in anytime-valid testing and false-discovery control for
// model and agent gates you can procure our services by sending an email to
// info@swedishembedded.com.

//! A lineage round under the evidence gate.
//!
//! The lineage's dev set is the pool. Each round draws its dev items from it
//! without replacement (a seed derived from the lineage's and the round's),
//! grades the champion and the candidate on them in batches, and stops at the
//! first batch whose e-process crosses the round's level; the champion stays
//! otherwise. Every graded item is spent, and the rounds journal is the
//! durable record of what was: the used set (and so each draw's exposure)
//! and the false evolution budget are rebuilt from the decisions it holds,
//! so a restarted lineage draws and spends exactly as before.
//!
//! Only spent items are ever graded, and the proposer is steered by the
//! champion's results on spent items alone, so a fresh item is fresh: the
//! overlap correction then accounts for every reuse. The pool is the
//! learner's own dev set; nothing here reads an examiner's data.

use std::collections::BTreeSet;

use anyhow::Result;
use serde_json::{json, Value};
use splinter_sdk::campaign::brain::TaskFamily;
use splinter_sdk::campaign::PolicyRef;
use splinter_sdk::measure::campaign::dev_pool::DevPool;
use splinter_sdk::measure::campaign::evidence::Budget;
use splinter_sdk::measure::campaign::gate::{decide_evidence, Decision, DevResult, EvidenceStatus};
use splinter_sdk::measure::campaign::plan::retention_units_needed;
use splinter_sdk::model::stats::ExactSignTest;
use splinter_sdk::vocabulary::rng::SplitMix64;

use crate::lineage::{fail, policy_key, DevRecord, Lineage, RoundRecord};
use crate::plan::{Acceptance, EvidencePlan};

/// Mixed into a round's seed so its dev draw and its training seed are
/// separate streams.
const DEV_DRAW_SALT: u64 = 0x5EED_D0E5_DEF0_0D01;

/// The used dev items and the budget a journal of rounds leaves, replayed
/// in order.
#[must_use]
pub fn replay(plan: &EvidencePlan, rounds: &[RoundRecord]) -> (BTreeSet<String>, Budget) {
    let mut used = BTreeSet::new();
    let mut budget = Budget::start(&plan.gate);
    for round in rounds {
        if let Some(record) = round.decision.as_ref().and_then(|d| d.evidence.as_ref()) {
            used.extend(record.spent.iter().cloned());
            budget.settle(&record.allocation, round.advanced);
        }
    }
    (used, budget)
}

/// Refuses a journal decided under another gate, or under the evidence gate
/// with other parameters: its budget and used set would mean something else.
///
/// # Errors
/// A journalled decision does not match `acceptance`.
pub fn check_journal(acceptance: &Acceptance, rounds: &[RoundRecord]) -> Result<()> {
    for round in rounds {
        let Some(decision) = &round.decision else {
            continue;
        };
        match (acceptance, &decision.evidence) {
            (Acceptance::Greedy, None) => {}
            (Acceptance::Evidence(plan), Some(record)) if record.gate == plan.gate => {}
            (Acceptance::Evidence(_), Some(record)) => anyhow::bail!(
                "generation {} round {} was decided by an evidence gate with {:?}; a lineage keeps its gate",
                round.generation,
                round.round,
                record.gate
            ),
            _ => anyhow::bail!(
                "generation {} round {} was decided by the {} gate; a lineage keeps its gate",
                round.generation,
                round.round,
                if decision.evidence.is_some() {
                    "evidence"
                } else {
                    "greedy"
                }
            ),
        }
    }
    Ok(())
}

impl Lineage<'_> {
    /// Everything `policy` has been graded on, from the dev journal.
    ///
    /// # Errors
    /// The journal cannot be read.
    pub fn graded(&self, policy: &PolicyRef) -> Result<DevResult> {
        let key = policy_key(policy);
        let mut merged = DevResult::default();
        for record in self
            .files
            .dev()
            .read::<DevRecord>()?
            .into_iter()
            .filter(|r| r.policy == key)
        {
            merged.dev.extend(record.result.dev);
            merged.retention.extend(record.result.retention);
        }
        Ok(merged)
    }

    /// `policy`'s outcomes on `dev` and the retention set, grading (and
    /// journalling) only what it was not graded on before.
    fn grade_on(&mut self, policy: &PolicyRef, dev: &[String]) -> Result<DevResult> {
        let known = self.graded(policy)?;
        let retention = self.family.retention();
        let missing: Vec<String> = dev
            .iter()
            .filter(|t| !known.dev.contains_key(*t))
            .chain(
                retention
                    .iter()
                    .filter(|t| !known.retention.contains_key(*t)),
            )
            .cloned()
            .collect();
        let mut new = DevResult::default();
        if !missing.is_empty() {
            self.host.context(policy)?;
            let kept: BTreeSet<&String> = retention.iter().collect();
            for answer in self.validator.answer(policy, &missing).map_err(fail)? {
                let part = if kept.contains(&answer.task) {
                    &mut new.retention
                } else {
                    &mut new.dev
                };
                part.insert(answer.task, answer.passed);
            }
            self.files.dev().append(&DevRecord {
                policy: policy_key(policy),
                result: new.clone(),
            })?;
        }
        let wanted: BTreeSet<&String> = dev.iter().collect();
        Ok(DevResult {
            dev: known
                .dev
                .into_iter()
                .chain(new.dev)
                .filter(|(t, _)| wanted.contains(t))
                .collect(),
            retention: known.retention.into_iter().chain(new.retention).collect(),
        })
    }

    /// The champion's dev result the proposer is steered by: under the
    /// greedy gate its answers on the whole dev set, under the evidence gate
    /// only what it was graded on (every such item is spent).
    pub(crate) fn steering_dev(&mut self, champion: &PolicyRef) -> Result<DevResult> {
        match self.plan.acceptance {
            Acceptance::Greedy => self.dev_of(champion),
            Acceptance::Evidence(_) => self.graded(champion),
        }
    }

    /// Refuses an evidence plan the dev and retention sets cannot carry: a
    /// draw that is not smaller than the pool, or a retention set too small
    /// to see the allowed loss at the first round's level. What it planned.
    ///
    /// # Errors
    /// The draw or the retention set does not fit the plan.
    pub fn check_evidence(
        &self,
        plan: &EvidencePlan,
        dev_units: usize,
        retention_units: usize,
    ) -> Result<Value> {
        anyhow::ensure!(
            plan.draw < dev_units,
            "--dev-draw {} must be smaller than the dev pool of {dev_units}: a round that draws the whole pool has no fresh item from its second round on",
            plan.draw
        );
        let first = Budget::start(&plan.gate).allocate(&plan.gate, self.plan.horizon());
        let needed = retention_units_needed(
            self.plan.retention_margin,
            first.alpha,
            5000,
            &ExactSignTest,
        );
        anyhow::ensure!(
            needed.is_some_and(|n| retention_units >= n),
            "the retention set has {retention_units} programs, too few to detect a loss of {} of them at level {:.4} (needs {needed:?})",
            self.plan.retention_margin,
            first.alpha
        );
        Ok(json!({
            "gate": plan.gate,
            "draw": plan.draw,
            "batch": plan.batch,
            "pool": dev_units,
            "horizon": self.plan.horizon(),
            "first_round": first,
            "retention_units_needed": needed,
        }))
    }

    /// The evidence gate's decision on `candidate` against `champion` in
    /// round `round` of `generation`.
    pub(crate) fn gate_evidence(
        &mut self,
        plan: EvidencePlan,
        champion: &PolicyRef,
        candidate: &PolicyRef,
        (generation, round): (u32, u32),
    ) -> Result<Decision> {
        let rounds = self.rounds()?;
        check_journal(&self.plan.acceptance, &rounds)?;
        let (used, budget) = replay(&plan, &rounds);
        let allocation = budget.allocate(&plan.gate, self.plan.horizon());
        let pool = DevPool::new(self.family.dev(), &used);
        let seed = self.seed ^ (u64::from(generation) << 32 | u64::from(round)) ^ DEV_DRAW_SALT;
        let drawn = pool
            .draw(plan.draw, &mut SplitMix64::new(seed))
            .map_err(fail)?;
        let gate = self.gate;
        let decide = |graded, champion: &DevResult, candidate: &DevResult| {
            decide_evidence(
                &gate,
                &plan.gate,
                &allocation,
                (&drawn, graded),
                champion,
                candidate,
                &ExactSignTest,
            )
            .map_err(fail)
        };
        if allocation.alpha <= 0.0 {
            return decide(0, &DevResult::default(), &DevResult::default());
        }
        let mut graded = 0;
        loop {
            graded = (graded + plan.batch).min(drawn.len());
            let ids: Vec<String> = drawn[..graded].iter().map(|d| d.task.clone()).collect();
            let champion_dev = self.grade_on(champion, &ids)?;
            let candidate_dev = self.grade_on(candidate, &ids)?;
            let decision = decide(graded, &champion_dev, &candidate_dev)?;
            let status = decision.evidence.as_ref().map(|e| e.status);
            eprintln!(
                "generation {generation} round {round}: {graded} of {} dev items graded, e-value {:.3} of {:.3}",
                drawn.len(),
                decision.evidence.as_ref().map_or(0.0, |e| e.e_value),
                decision.evidence.as_ref().map_or(0.0, |e| e.threshold)
            );
            if status != Some(EvidenceStatus::Open) {
                return Ok(decision);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lineage::Trained;
    use crate::plan::TrainFrom;
    use splinter_sdk::measure::campaign::dev_pool::Drawn;
    use splinter_sdk::measure::campaign::evidence::EvidenceGate;
    use splinter_sdk::measure::campaign::gate::{decide, DevGate};

    fn plan(q: f64) -> EvidencePlan {
        EvidencePlan {
            gate: EvidenceGate::new(0.02, q, 0.01).unwrap(),
            draw: 4,
            batch: 2,
        }
    }

    fn record(generation: u32, decision: Option<Decision>, advanced: bool) -> RoundRecord {
        RoundRecord {
            generation,
            round: 1,
            started_from: None,
            proposed: 1,
            mispredicted: 1,
            replayed: 0,
            trained: Some(Trained {
                adapter: format!("/a/{generation}.safetensors").into(),
                digest: format!("sha256:{generation:064x}"),
                steps: 1,
                held_out_loss_base: None,
                held_out_loss_tuned: None,
            }),
            decision,
            advanced,
            note: String::new(),
            trained_on: Vec::new(),
            seconds: 0.0,
            train_from: TrainFrom::Previous,
            trained_records: 0,
            frontier: None,
            reasoning: None,
            train_on: crate::plan::TrainOn::Answers,
            admit_output: None,
            families: Vec::new(),
            family_dev: Vec::new(),
        }
    }

    /// An evidence decision that spent `spent` and crossed when `win`.
    fn evidence_decision(
        plan: &EvidencePlan,
        budget: &Budget,
        spent: &[&str],
        win: bool,
    ) -> Decision {
        let drawn: Vec<Drawn> = spent
            .iter()
            .map(|t| Drawn {
                task: (*t).to_string(),
                zeta: 0.0,
                reused: false,
            })
            .collect();
        let outcomes = |pass: bool| DevResult {
            dev: spent.iter().map(|t| ((*t).to_string(), pass)).collect(),
            retention: Default::default(),
        };
        decide_evidence(
            &DevGate::default(),
            &plan.gate,
            &budget.allocate(&plan.gate, 3),
            (&drawn, drawn.len()),
            &outcomes(false),
            &outcomes(win),
            &ExactSignTest,
        )
        .unwrap()
    }

    #[test]
    fn the_journal_rebuilds_the_used_set_and_the_budget_in_order() {
        let p = plan(0.3);
        let mut expected = Budget::start(&p.gate);
        let first = evidence_decision(&p, &expected, &["a", "b"], true);
        let allocation = first.evidence.as_ref().unwrap().allocation;
        expected.settle(&allocation, true);
        let second = evidence_decision(&p, &expected, &["b", "c"], false);
        let allocation = second.evidence.as_ref().unwrap().allocation;
        expected.settle(&allocation, false);
        let rounds = [
            record(1, Some(first), true),
            record(2, None, false),
            record(3, Some(second), false),
        ];
        let (used, budget) = replay(&p, &rounds);
        assert_eq!(
            used.into_iter().collect::<Vec<_>>(),
            ["a", "b", "c"].map(String::from)
        );
        assert_eq!(budget, expected);
        assert_eq!((budget.rounds, budget.accepted), (2, 1));
        assert!(check_journal(&Acceptance::Evidence(p), &rounds).is_ok());
    }

    #[test]
    fn a_journal_decided_under_another_gate_is_refused() {
        let p = plan(0.3);
        let evidence = evidence_decision(&p, &Budget::start(&p.gate), &["a"], true);
        let champion = DevResult::default();
        let greedy = decide(&DevGate::default(), 1, &champion, &champion, &ExactSignTest);
        let under_evidence = [record(1, Some(evidence), true)];
        let under_greedy = [record(1, Some(greedy), false)];
        assert!(check_journal(&Acceptance::Greedy, &under_evidence).is_err());
        assert!(check_journal(&Acceptance::Evidence(p), &under_greedy).is_err());
        assert!(check_journal(&Acceptance::Greedy, &under_greedy).is_ok());
        let other = plan(0.2);
        let refused = check_journal(&Acceptance::Evidence(other), &under_evidence).unwrap_err();
        assert!(refused.to_string().contains("keeps its gate"), "{refused}");
    }
}
