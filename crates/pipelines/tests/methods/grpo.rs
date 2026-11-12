// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for proving, in seconds and on a CPU, that each learning method of a
// training system moves a held-out score, for its clients. If your team needs expertise in reinforcement learning
// against verifiers that the policy cannot reach then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Group-relative policy gradient (GRPO) against a verifier: a partly right model draws a group of answers per
//! prompt, each earns the verifier's reward, the `policy-gradient` view turns the rewards of a group into
//! advantages, and a clipped policy-gradient step on the model's own draws raises the right ones. Five on-policy
//! rounds on fresh prompts, each drawing from the model the last produced; no reference answer is ever shown to the trainer.

#![allow(clippy::unwrap_used)]

use splinter_core::annotation::Strength;
use splinter_data::{PolicyGradient, RewardShape};
use splinter_model::policy_gradient::{Behaviour, PolicyGradientOptions, Reference};

use crate::lab::{corpus, dataset, Lab, PolicyGradientRun, SAMPLING_TEMPERATURE};
use crate::report::Outcome;
use crate::world::split;

const ROUNDS: usize = 5;
const PROMPTS_PER_ROUND: usize = 40;
const DRAWS: u32 = 24;

#[test]
#[ignore = "trains a toy model: make test/methods runs it"]
fn group_relative_policy_gradient_on_verifier_rewards_improves_held_out_prompts() {
    let lab = Lab::new();
    let split = split();
    let warm = lab.warm_adapter(&split.train[..60], 200);
    let before = lab.accuracy(Some(&warm), &split.held_out);

    // Standard GRPO: the default two-sided, standardised advantage, rewards shaped by the verifier's partial credit
    // (a wrong answer with two letters right earns more than one with none). Each round draws a group of answers per
    // prompt from the current model, on prompts it has not been trained on before: reusing prompts lets the negative
    // advantages memorise which tokens were wrong for those prompts, and the held-out score then falls.
    let options = PolicyGradientOptions {
        kl_beta: Some(0.0),
        clip_eps: Some(0.2),
        reference: Some(Reference::StartOfRun),
        behaviour: Some(Behaviour::StartOfRun),
        temperature: Some(SAMPLING_TEMPERATURE),
        ..PolicyGradientOptions::default()
    };
    let view = PolicyGradient::new(Strength::Executable)
        .rewards(RewardShape::shaped(0.9, vec![("letters".into(), 1.0)]).unwrap());
    let mut adapter = warm.clone();
    for round in 0..ROUNDS {
        let prompts = &split.pool[round * PROMPTS_PER_ROUND..(round + 1) * PROMPTS_PER_ROUND];
        let attempts = lab.rollouts(
            Some(&adapter),
            "self",
            prompts,
            DRAWS,
            SAMPLING_TEMPERATURE,
            round as u64 + 1,
        );
        let records = lab.dir.path().join(format!("pg-{round}.jsonl"));
        let projected = dataset(&view, &corpus(attempts), &records);
        let trained = lab.policy_gradient_with(
            &format!("pg-{round}"),
            &records,
            PolicyGradientRun {
                steps: 100,
                learning_rate: 1e-3,
                grad_accum: 32,
            },
            Some(&adapter),
            options,
        );
        let stats = &trained.statistics;
        println!(
            "GRPO round {round}: {} records, mean importance ratio {:?}, clipped {:?}",
            projected.records.len(),
            stats.mean_ratio,
            stats.mean_clip_fraction
        );
        adapter = trained.adapter;
    }
    let after = lab.accuracy(Some(&adapter), &split.held_out);
    Outcome {
        method: "GRPO (verifier reward)",
        metric: "held-out exact match",
        before,
        after,
        min_gain: 0.1,
        note: "default two-sided standardised advantage, shaped reward, fresh prompts each round",
    }
    .conclude();
}
