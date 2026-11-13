// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that let a training campaign draw
// only on environments whose verifiers were attacked, and stop the moment a
// reward stops meaning anything, for its clients. If your team needs
// expertise in reward-hacking defence for reinforcement-learning campaigns
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: reward provenance reaches the ledger and the verdict label, and the
//! divergence alarm quarantines an environment whose reward climbs while its
//! dev measure does not.

use super::*;
use splinter_core::verifiability::VerifierClass;
use splinter_pipelines::campaign::EnvironmentLabels;
use splinter_store::campaign::RewardSource;

/// Two environments told apart by the parity of the task number; the even
/// one is a proxy.
struct Labels {
    quarantined: RefCell<Vec<(String, String)>>,
}

impl EnvironmentLabels for Labels {
    fn environment_of(&self, task: &str) -> Option<String> {
        let n: usize = task[1..].parse().ok()?;
        Some(if n.is_multiple_of(2) { "even" } else { "odd" }.to_string())
    }
    fn sources(&self) -> Vec<RewardSource> {
        vec![
            RewardSource {
                environment: "even".into(),
                class: VerifierClass::Proxy,
                verifier: "toy@1".into(),
            },
            RewardSource {
                environment: "odd".into(),
                class: VerifierClass::Objective,
                verifier: "toy@1".into(),
            },
        ]
    }
    fn quarantine(&self, environment: &str, reason: &str) -> Result<(), CampaignError> {
        self.quarantined
            .borrow_mut()
            .push((environment.to_string(), reason.to_string()));
        Ok(())
    }
}

#[test]
fn rewards_carry_their_verifier_class_into_the_ledger_and_the_verdict_label() {
    let (ws, dir) = workspace();
    let world = World::new(20, 0.15);
    let adapters = dir.path().join("adapters");
    let mut rig = Rig::new(&world, &adapters, 120);
    let labels = Labels {
        quarantined: RefCell::default(),
    };
    let clock = FixedClock::new("2026-11-13T09:00:00.000Z");
    let watch = Seconds::still();
    let mut parts = rig.parts(&clock, &watch);
    parts.environments = Some(&labels);
    let mut controller =
        CampaignController::open(&ws, config("c16"), SEED, parts, adapters.clone()).unwrap();
    controller.run(1).unwrap();
    for entry in controller.ledger().unwrap() {
        assert_eq!(entry.reward_sources.len(), 2);
        assert_eq!(
            entry.verdict_label.as_deref(),
            Some("proxy only"),
            "{entry:?}"
        );
    }
}

#[test]
fn a_reward_that_climbs_while_the_dev_measure_stays_flat_quarantines_the_environment_and_stops_the_arm(
) {
    let (ws, dir) = workspace();
    let mut world = World::new(20, 0.15);
    world.flat_dev_for_even = true;
    let adapters = dir.path().join("adapters");
    let mut rig = Rig::new(&world, &adapters, 120);
    let labels = Labels {
        quarantined: RefCell::default(),
    };
    let clock = FixedClock::new("2026-11-13T09:00:00.000Z");
    let watch = Seconds::still();
    let mut parts = rig.parts(&clock, &watch);
    parts.environments = Some(&labels);
    let mut c = config("c17");
    c.divergence = splinter_knowledge::factory::alarm::DivergencePolicy {
        window: 3,
        min_reward_rise: 0.0,
        max_heldout_rise: 0.0,
    };
    c.ladder = vec![LearningStrategy::PolicyGradient];
    c.max_idle_arms = 100;
    let mut controller = CampaignController::open(&ws, c, SEED, parts, adapters.clone()).unwrap();
    let report = controller.run(5).unwrap();
    let halted: Vec<_> = report
        .arms
        .iter()
        .filter(|a| matches!(a.verdict, Verdict::Halted { .. }))
        .collect();
    assert_eq!(
        halted.len(),
        1,
        "{:?}",
        report.arms.iter().map(|a| &a.verdict).collect::<Vec<_>>()
    );
    assert_eq!(controller.state().quarantined, ["even"]);
    let recorded = labels.quarantined.borrow();
    assert_eq!(recorded.len(), 1);
    assert!(recorded[0].1.contains("proxy only"), "{}", recorded[0].1);
    drop(recorded);
    // After the quarantine no batch draws an even task.
    let after: Vec<String> = world
        .log
        .requests
        .borrow()
        .iter()
        .rev()
        .take(1)
        .flat_map(|(_, tasks)| tasks.clone())
        .collect();
    assert!(
        after
            .iter()
            .all(|t| t[1..].parse::<usize>().unwrap() % 2 == 1),
        "{after:?}"
    );
}
