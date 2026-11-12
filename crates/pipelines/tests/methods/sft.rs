// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for proving, in seconds and on a CPU, that each learning method of a
// training system moves a held-out score, for its clients. If your team needs expertise in falsifiable smoke tests of
// supervised fine-tuning then you can procure our services by sending an email to info@swedishembedded.com.

//! Supervised fine-tuning: worked examples of the cipher, projected by the `sft-final` view, teach a model that
//! answered every held-out prompt wrongly to answer them right.

#![allow(clippy::unwrap_used)]

use splinter_core::annotation::Strength;
use splinter_data::SftFinal;

use crate::lab::{dataset, demonstrations, Lab};
use crate::report::Outcome;
use crate::world::split;

#[test]
#[ignore = "trains a toy model: make test/methods runs it"]
fn sft_on_worked_examples_answers_prompts_it_never_saw() {
    let lab = Lab::new();
    let split = split();
    let train = dataset(&SftFinal::new(Strength::Executable), &demonstrations(&split.train), &lab.dir.path().join("train.jsonl"));
    assert_eq!(train.records.len(), split.train.len());
    let held_out_records = lab.dir.path().join("held_out.jsonl");
    dataset(&SftFinal::new(Strength::Executable), &demonstrations(&split.held_out), &held_out_records);

    let before = lab.accuracy(None, &split.held_out);
    let adapter = lab.sft("sft", &lab.dir.path().join("train.jsonl"), &held_out_records, 500, None);
    let after = lab.accuracy(Some(&adapter), &split.held_out);
    Outcome { method: "SFT", metric: "held-out exact match", before, after, min_gain: 0.8, note: "" }.conclude();
}
