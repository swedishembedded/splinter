// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for proving, in seconds and on a CPU, that each learning method of a
// training system moves a held-out score, for its clients. If your team needs expertise in libraries of verified
// sub-solutions that agents reuse across compound tasks then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Skill library: while it practises on single sub-tasks the model's answers are checked by the verifier and the
//! verified ones are kept, each under its sub-task. Compound tasks that are new combinations of those sub-tasks are
//! then solved with the library, which answers a sub-task it holds from storage and asks the model only for the
//! rest. Measured on held-out compound tasks against the same model without the library: the share solved end to end
//! and the tokens the model generated.

#![allow(clippy::unwrap_used)]

use splinter_model::answer::Decoding;
use splinter_pipelines::skill_library::SkillLibrary;

use crate::lab::{Lab, SAMPLING_TEMPERATURE};
use crate::report::Outcome;
use crate::world::{is_correct, split};

const PARTS: usize = 2;
const COMPOUNDS: usize = 100;
const DRAWS: u32 = 64;

/// `COMPOUNDS` compound tasks of `PARTS` distinct sub-tasks each, drawn from `pool` by a fixed stream: new
/// combinations of sub-tasks that were each practised alone.
fn compounds(pool: &[String]) -> Vec<Vec<String>> {
    let mut state = 4242u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
    (0..COMPOUNDS)
        .map(|_| {
            let mut parts: Vec<String> = Vec::new();
            while parts.len() < PARTS {
                let part = pool[next() % pool.len()].clone();
                if !parts.contains(&part) {
                    parts.push(part);
                }
            }
            parts
        })
        .collect()
}

#[test]
#[ignore = "trains a toy model: make test/methods runs it"]
fn a_library_of_verified_sub_solutions_solves_new_compound_tasks_with_fewer_tokens() {
    let lab = Lab::new();
    let split = split();
    let warm = lab.warm_adapter(&split.train[..60], 200);
    let practised = &split.train[60..];

    // Practice: the model answers each sub-task alone, greedily and by sampling; the verifier decides what is kept.
    let policy = lab.policy(Some(&warm), Decoding::Greedy);
    let mut library = SkillLibrary::new();
    for (i, prompt) in practised.iter().enumerate() {
        let greedy = policy.reply(prompt);
        library.record(prompt, &greedy, is_correct(prompt, &greedy));
        for draw in 0..DRAWS {
            // seed: the library's coverage, and so the thresholds, depend on which draws were made; a rerun must make the same.
            let seed = 1 << 32 | (i as u64) << 8 | u64::from(draw);
            let answer = policy.sample(prompt, SAMPLING_TEMPERATURE, seed);
            library.record(prompt, &answer, is_correct(prompt, &answer));
        }
    }
    println!(
        "skill library: {} of {} practised sub-tasks have a verified solution",
        library.len(),
        practised.len()
    );

    let ask = |part: &str| {
        let answer = policy.reply(part);
        let tokens = answer.chars().count() + 1;
        (answer, tokens)
    };
    let (mut without, mut with, mut tokens_without, mut tokens_with, mut from_library) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    let tasks = compounds(practised);
    for parts in &tasks {
        let alone = SkillLibrary::new().solve(parts, ask);
        let helped = library.solve(parts, ask);
        let solved = |answers: &[String]| parts.iter().zip(answers).all(|(p, a)| is_correct(p, a));
        without += usize::from(solved(&alone.answers));
        with += usize::from(solved(&helped.answers));
        tokens_without += alone.tokens;
        tokens_with += helped.tokens;
        from_library += helped.from_library;
    }
    let n = tasks.len() as f64;
    let note = format!(
        "compound tasks of {PARTS} sub-tasks; the library answered {from_library} of {} parts; tokens per task {:.1} without, {:.1} with",
        tasks.len() * PARTS,
        tokens_without as f64 / n,
        tokens_with as f64 / n
    );
    assert!(
        tokens_with < tokens_without,
        "the library saves tokens: {tokens_with} >= {tokens_without}"
    );
    Outcome {
        method: "Skill library (verified sub-solutions)",
        metric: "held-out compound tasks solved",
        before: without as f64 / n,
        after: with as f64 / n,
        min_gain: 0.1,
        note: &note,
    }
    .conclude();
}
