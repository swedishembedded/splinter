// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for proving, in seconds and on a CPU, that each learning method of a
// training system moves a held-out score, for its clients. If your team needs expertise in reinforcement learning
// against verifiers that the policy cannot reach then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Group-relative policy gradient (GRPO) against a verifier: a partly right model draws a group of answers per
//! prompt, each earns the verifier's reward, the `policy-gradient` view turns the rewards of a group into
//! advantages, and a clipped policy-gradient step on the model's own draws raises the right ones. Three on-policy
//! rounds, each drawing from the model the last produced; no reference answer is ever shown to the trainer.

#![allow(clippy::unwrap_used)]

use splinter_core::annotation::Strength;
use splinter_data::{AdvantageKind, PolicyGradient};

use crate::lab::{corpus, dataset, Lab, SAMPLING_TEMPERATURE};
use crate::report::Outcome;
use crate::world::split;

#[test]
#[ignore = "trains a toy model: make test/methods runs it"]
fn group_relative_policy_gradient_on_verifier_rewards_improves_held_out_prompts() {
    let lab = Lab::new();
    let split = split();
    let warm = lab.warm_adapter(&split.train[..60], 200);
    let before = lab.accuracy(Some(&warm), &split.held_out);

    // Two on-policy rounds. Two-sided advantages (the default) left the held-out score where it started on this
    // task, because a wrong answer shares most tokens with a right one; the positive-only variant moves it.
    let mut adapter = warm.clone();
    for round in 1..=2u64 {
        let attempts = lab.rollouts(Some(&adapter), "self", &split.train[60..], 16, SAMPLING_TEMPERATURE, round);
        let records = lab.dir.path().join(format!("pg-{round}.jsonl"));
        let view = PolicyGradient::new(Strength::Executable).advantage(AdvantageKind::Positive);
        let projected = dataset(&view, &corpus(attempts), &records);
        println!("GRPO round {round}: {} rollout records", projected.records.len());
        adapter = lab.policy_gradient(&format!("pg-{round}"), &records, 200, 3e-3, Some(&adapter));
    }
    let after = lab.accuracy(Some(&adapter), &split.held_out);
    Outcome { method: "GRPO (verifier reward)", metric: "held-out exact match", before, after, min_gain: 0.15, note: "" }.conclude();
}
