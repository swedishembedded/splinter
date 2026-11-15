// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The controller under the evidence gate: an arm draws its dev tasks from
//! the dev set, grades only those, decides on the e-process and keeps the
//! budget and the spent tasks in the campaign state; the greedy gate stays
//! the default.

use splinter_eval::campaign::evidence::EvidenceGate;
use splinter_pipelines::campaign::Acceptance;

use super::*;

fn evidence_config(name: &str) -> CampaignConfig {
    let mut c = config(name);
    c.acceptance = Acceptance::Evidence {
        gate: EvidenceGate::new(0.02, 0.3, 0.01).unwrap(),
        draw: 60,
        horizon: 4,
    };
    c
}

#[test]
fn a_better_candidate_advances_on_evidence_from_the_drawn_tasks_alone() {
    let (ws, dir) = workspace();
    let world = World::new(20, 0.15);
    let adapters = dir.path().join("adapters");
    let mut rig = Rig::new(&world, &adapters, 120);
    let mut controller = open(&ws, evidence_config("e1"), &mut rig, &adapters).unwrap();
    let report = controller.run(1).unwrap();
    let arm = &report.arms[0];
    assert!(
        matches!(arm.verdict, Verdict::Accepted { significant: true }),
        "{:?}",
        arm.verdict
    );
    let record = arm.decision.as_ref().unwrap().evidence.as_ref().unwrap();
    assert!(record.e_value >= record.threshold);
    assert_eq!((record.drawn, record.spent.len()), (60, 60));
    assert_eq!(
        world.log.graded.get(),
        120,
        "60 drawn tasks for each policy, never the other 60"
    );
    let spent = controller.state().evidence.as_ref().unwrap();
    assert_eq!(spent.used.len(), 60);
    assert_eq!((spent.budget.rounds, spent.budget.accepted), (1, 1));
    assert!(spent.budget.remaining < 0.3);
    let ledger = controller.ledger().unwrap();
    let done = ledger.last().unwrap();
    for key in [
        "evidence_e_value",
        "evidence_threshold",
        "evidence_budget_left",
        "dev_spent",
        "gate_alpha",
    ] {
        assert!(
            done.measurements.contains_key(key),
            "{key} in {:?}",
            done.measurements
        );
    }
}

#[test]
fn without_evidence_the_champion_stays_and_the_next_arm_draws_partly_reused_tasks() {
    let (ws, dir) = workspace();
    let world = World::new(20, 0.0);
    let adapters = dir.path().join("adapters");
    let mut rig = Rig::new(&world, &adapters, 120);
    let mut controller = open(&ws, evidence_config("e2"), &mut rig, &adapters).unwrap();
    let report = controller.run(2).unwrap();
    assert_eq!(report.arms.len(), 2);
    assert!(report
        .arms
        .iter()
        .all(|a| matches!(a.verdict, Verdict::RolledBack { .. })));
    assert_eq!(controller.state().policy.version, 0);
    let second = report.arms[1]
        .decision
        .as_ref()
        .unwrap()
        .evidence
        .as_ref()
        .unwrap();
    assert!(second.zeta_mean.is_some_and(|z| z > 0.0), "{second:?}");
    let spent = controller.state().evidence.as_ref().unwrap();
    assert_eq!((spent.budget.rounds, spent.budget.accepted), (2, 0));
    assert!(spent.used.len() > 60 && spent.used.len() <= 120);
}

#[test]
fn the_greedy_gate_is_the_default_and_grades_the_whole_dev_set() {
    let (ws, dir) = workspace();
    let world = World::new(20, 0.15);
    let adapters = dir.path().join("adapters");
    let mut rig = Rig::new(&world, &adapters, 120);
    let mut controller = open(&ws, config("e3"), &mut rig, &adapters).unwrap();
    let report = controller.run(1).unwrap();
    assert!(report.arms[0].decision.as_ref().unwrap().evidence.is_none());
    assert!(controller.state().evidence.is_none());
    assert_eq!(world.log.graded.get(), 240);
}
