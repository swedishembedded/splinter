// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for proving, in seconds and on a CPU, that each learning method of a
// training system moves a held-out score, for its clients. If your team needs expertise in rejection-sampling
// fine-tuning on a model's own verified answers then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Rejection-sampling fine-tuning: a partly right model answers unlabelled prompts many times, the verifier keeps the
//! answers that are right, and fine-tuning on those answers (the `sft-final` view over the graded attempts) makes
//! it right more often on prompts it was never trained on. No reference answer is ever shown to the trainer.

#![allow(clippy::unwrap_used)]

use splinter_core::annotation::Strength;
use splinter_data::SftFinal;

use crate::lab::{dataset, demonstrations, Lab};
use crate::report::Outcome;
use crate::world::split;

#[test]
#[ignore = "trains a toy model: make test/methods runs it"]
fn rejection_sampling_on_own_verified_answers_improves_held_out_prompts() {
    let lab = Lab::new();
    let split = split();
    let warm = lab.warm_adapter(&split.train[..60], 200);
    let held_out = lab.dir.path().join("held_out.jsonl");
    dataset(&SftFinal::new(Strength::Executable), &demonstrations(&split.held_out), &held_out);

    let before = lab.accuracy(Some(&warm), &split.held_out);
    // Two rounds of expert iteration: the second samples from the model the first produced.
    let mut adapter = warm.clone();
    for round in 1..=2u64 {
        let attempts = lab.rollouts(Some(&adapter), "self", &split.train[60..], 16, 0.6, round);
        let train = lab.dir.path().join(format!("raft-{round}.jsonl"));
        let kept = dataset(&SftFinal::new(Strength::Executable), &crate::lab::corpus(attempts), &train);
        assert!(!kept.records.is_empty(), "the model passes some prompts");
        adapter = lab.sft(&format!("raft-{round}"), &train, &held_out, 200, Some(&adapter));
    }
    let after = lab.accuracy(Some(&adapter), &split.held_out);
    Outcome { method: "Rejection-sampling FT", metric: "held-out exact match", before, after, min_gain: 0.15, note: "" }.conclude();
}
