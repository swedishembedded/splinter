// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for proving, in seconds and on a CPU, that each learning method of a
// training system moves a held-out score, for its clients. If your team needs expertise in distilling a capable
// model into a small one then you can procure our services by sending an email to info@swedishembedded.com.

//! Distillation: a teacher that has learnt the cipher answers prompts nobody labelled, its answers are admitted at the
//! strength of a judgement (no verifier, no reference), and a student that knows nothing is fine-tuned on them. The
//! student then answers prompts neither it nor the teacher's answers ever covered.

#![allow(clippy::unwrap_used)]

use splinter_core::annotation::{Outcome, Strength};
use splinter_data::SftFinal;

use crate::lab::{corpus, dataset, demonstrations, judged_attempt, Lab};
use crate::report::Outcome as Result;
use crate::world::{is_correct, split};

#[test]
#[ignore = "trains a toy model: make test/methods runs it"]
fn a_student_fine_tuned_on_a_teachers_answers_to_unlabelled_prompts_learns_the_task() {
    let lab = Lab::new();
    let split = split();
    // The teacher: the base after supervised learning on the first prompts (its own, separate, training).
    let teacher_set = lab.dir.path().join("teacher.jsonl");
    dataset(&SftFinal::new(Strength::Executable), &demonstrations(&split.train), &teacher_set);
    let teacher = lab.sft("teacher", &teacher_set, &teacher_set, 500, None);
    let policy = lab.policy(Some(&teacher), splinter_model::answer::Decoding::Greedy);

    // Unlabelled prompts: the held-out ones are not among them. The teacher's answers are the only supervision.
    let unlabelled = &split.train[..100];
    let replies: Vec<String> = unlabelled.iter().map(|p| policy.reply(p)).collect();
    let right = unlabelled.iter().zip(&replies).filter(|(p, r)| is_correct(p, r)).count();
    println!("distillation: the teacher answers {right} of {} unlabelled prompts right", replies.len());
    let answers: Vec<_> = unlabelled.iter().zip(&replies).enumerate().map(|(i, (p, r))| judged_attempt("teacher", p, r, i as u32, Outcome::Pass, Strength::Judged, serde_json::json!({}))).collect();
    drop(policy);

    let train = lab.dir.path().join("student.jsonl");
    dataset(&SftFinal::new(Strength::Judged), &corpus(answers), &train);
    let held_out = lab.dir.path().join("held_out.jsonl");
    dataset(&SftFinal::new(Strength::Executable), &demonstrations(&split.held_out), &held_out);
    let before = lab.accuracy(None, &split.held_out);
    let student = lab.sft("student", &train, &held_out, 500, None);
    let after = lab.accuracy(Some(&student), &split.held_out);
    Result { method: "Distillation (teacher answers)", metric: "held-out exact match", before, after, min_gain: 0.7, note: "" }.conclude();
}
