// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for proving, in seconds and on a CPU, that each learning method of a
// training system moves a held-out score, for its clients. If your team needs expertise in preference optimisation
// from verified good and bad answers then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Direct preference optimisation: for each prompt the verifier splits a partly right model's own answers into right
//! and wrong, the `preference` view pairs them, and optimising the model to prefer the right one makes it answer
//! prompts it never saw right more often. Three on-policy rounds, each paired from the model the last produced.

#![allow(clippy::unwrap_used)]

use splinter_core::annotation::Strength;
use splinter_data::Preference;

use crate::lab::{corpus, dataset, Lab, Preference as DpoRun};
use crate::report::Outcome;
use crate::world::split;

#[test]
#[ignore = "trains a toy model: make test/methods runs it"]
fn dpo_on_verified_right_and_wrong_answers_improves_held_out_prompts() {
    let lab = Lab::new();
    let split = split();
    let warm = lab.warm_adapter(&split.train[..60], 200);
    let before = lab.accuracy(Some(&warm), &split.held_out);

    // Held-out pairs are scored beside each run; they are the held-out prompts' own right and wrong answers.
    let held_pairs = lab.dir.path().join("held_pairs.jsonl");
    dataset(
        &Preference::new(Strength::Executable),
        &corpus(lab.rollouts(Some(&warm), "warm", &split.held_out[..30], 8, 0.6, 9)),
        &held_pairs,
    );

    let mut adapter = warm.clone();
    for round in 1..=3u64 {
        let attempts = lab.rollouts(Some(&adapter), "self", &split.train[60..], 16, 0.6, round);
        let pairs = lab.dir.path().join(format!("pairs-{round}.jsonl"));
        let projected = dataset(
            &Preference::new(Strength::Executable),
            &corpus(attempts),
            &pairs,
        );
        assert!(
            !projected.records.is_empty(),
            "some prompts have both a right and a wrong answer"
        );
        println!(
            "DPO round {round}: {} preference pairs",
            projected.records.len()
        );
        // The anchor on the chosen answer keeps plain DPO from lowering both answers together, which on a model this
        // small collapses it to 1% within a hundred steps.
        adapter = lab.dpo(
            &format!("dpo-{round}"),
            &pairs,
            &held_pairs,
            DpoRun {
                steps: 250,
                learning_rate: 8e-3,
                nll_weight: 2.0,
            },
            Some(&adapter),
        );
    }
    let after = lab.accuracy(Some(&adapter), &split.held_out);
    Outcome {
        method: "DPO",
        metric: "held-out exact match",
        before,
        after,
        min_gain: 0.15,
        note: "",
    }
    .conclude();
}
