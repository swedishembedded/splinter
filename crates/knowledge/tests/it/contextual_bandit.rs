// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning environments whose every reward is
// known in advance, so a training method can be checked against exact
// expectations, for its clients. If your team needs expertise in validating
// reinforcement-learning pipelines, you can procure our services by sending
// an email to info@swedishembedded.com.

//! Spec (12A, 12D): the contextual bandit's verifier, run on every action
//! of every instance asked, pays exactly what the rule says (1 for the
//! action at `colour + level` mod the actions, 0.5 for its neighbours when
//! graded, else 0), so every action's reward, and the expected return of any
//! policy, is known exactly; a dev instance shows a level no train instance
//! does; an answer naming every action earns nothing; and admission refuses
//! it as a campaign environment, its few answers being guessable by design.

#![allow(clippy::unwrap_used, clippy::expect_used)] // a panic is the failure report

use std::collections::{BTreeMap, BTreeSet};

use splinter_core::verifiability::Split;
use splinter_knowledge::factory::admission::{admit, Status};
use splinter_knowledge::factory::artifact::Instance;
use splinter_knowledge::factory::band::BandMeasurement;
use splinter_knowledge::factory::host::Host;
use splinter_knowledge::factory::plugin::{InstanceRequest, Plugin};
use splinter_knowledge::factory::plugins::contextual_bandit::{ContextualBandit, ACTIONS, COLOURS};

use super::factory_admission::small_policy;

fn request(split: Split, seed: u64, actions: u32, graded: u32) -> InstanceRequest {
    InstanceRequest {
        split,
        seed,
        difficulty: BTreeMap::from([
            ("actions".to_string(), actions),
            ("graded".to_string(), graded),
        ]),
    }
}

fn reward(instance: &Instance, answer: &str) -> Option<f64> {
    let session = Host::default().open(instance).unwrap();
    session.write("answer.txt", answer.as_bytes()).unwrap();
    session.verify().unwrap().reward()
}

/// The context an instruction shows.
fn context(instance: &Instance) -> (usize, usize) {
    let text = instance.instruction();
    let (_, rest) = text.split_once("shows ").unwrap();
    let (colour, rest) = rest.split_once(" at level ").unwrap();
    let level = rest.split_once('.').unwrap().0.parse().unwrap();
    (COLOURS.iter().position(|c| *c == colour).unwrap(), level)
}

#[test]
fn every_action_is_paid_what_the_rule_says() {
    let plugin = ContextualBandit;
    let mut levels: BTreeMap<&str, BTreeSet<usize>> = BTreeMap::new();
    for (split, name) in [(Split::Train, "train"), (Split::Dev, "dev")] {
        for actions in [2, 4, 6] {
            for graded in [0, 1] {
                for seed in 0..6 {
                    // seed: any, several contexts of each shape
                    let instance = plugin
                        .generate(&request(split, seed, actions, graded))
                        .unwrap();
                    let (colour, level) = context(&instance);
                    levels.entry(name).or_default().insert(level);
                    let n = actions as usize;
                    let correct = (colour + level) % n;
                    let mut paid = Vec::new();
                    for (a, action) in ACTIONS[..n].iter().enumerate() {
                        let off = (a + n - correct) % n;
                        let want = match (off, graded) {
                            (0, _) => 1.0,
                            (d, 1) if d == 1 || d == n - 1 => 0.5,
                            _ => 0.0,
                        };
                        let got = reward(&instance, &format!("{action}\n")).unwrap();
                        assert_eq!(
                            got, want,
                            "{split:?} {colour}/{level}, {n} actions, graded {graded}: {action}"
                        );
                        paid.push(got);
                    }
                    // The uniform policy's expected return, exactly.
                    let uniform: f64 = paid.iter().sum::<f64>() / n as f64;
                    let expected = if graded == 1 {
                        (1.0 + 0.5 * f64::from(if n == 2 { 1 } else { 2 })) / n as f64
                    } else {
                        1.0 / n as f64
                    };
                    assert!((uniform - expected).abs() < 1e-12);
                    assert_eq!(reward(&instance, ""), Some(0.0), "no answer");
                    assert_eq!(
                        reward(&instance, &ACTIONS[..n].join("\n")),
                        Some(0.0),
                        "every action"
                    );
                    let session = Host::default().open(&instance).unwrap();
                    assert!(session.solve().unwrap().succeeded());
                    assert_eq!(
                        session.verify().unwrap().reward(),
                        Some(1.0),
                        "the reference"
                    );
                }
            }
        }
    }
    assert!(levels["train"].iter().all(|l| *l < 6), "{levels:?}");
    assert!(levels["dev"].iter().all(|l| *l >= 6), "{levels:?}");
}

#[test]
fn admission_refuses_the_bandit_as_a_campaign_environment_for_its_guessable_answers() {
    // seed: any admission seed
    let (report, _) = admit(
        &ContextualBandit,
        &BTreeMap::from([("actions".to_string(), 6)]),
        &Host::default(),
        &small_policy(),
        43,
        &BTreeSet::new(),
        Some(&BandMeasurement {
            model: "base".into(),
            attempts: 32,
            passes: 8,
        }),
    )
    .unwrap();
    assert_ne!(report.status, Status::Admitted);
    let red_team = report.gates.iter().find(|g| g.name == "red-team").unwrap();
    assert!(
        !red_team.passed && red_team.detail.contains("blind attacks"),
        "{}",
        red_team.detail
    );
    assert!(
        report
            .gates
            .iter()
            .filter(|g| g.name != "red-team")
            .all(|g| g.passed),
        "only guessing keeps it out: {:?}",
        report
            .gates
            .iter()
            .filter(|g| !g.passed)
            .map(|g| &g.name)
            .collect::<Vec<_>>()
    );
}
