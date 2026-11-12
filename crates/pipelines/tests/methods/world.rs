// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for proving, in seconds and on a CPU, that each learning method of a
// training system moves a held-out score, for its clients. If your team needs expertise in falsifiable smoke tests of
// learning pipelines then you can procure our services by sending an email to info@swedishembedded.com.

//! The task every method smoke learns: a Caesar shift of three letters over an eight-letter alphabet.
//!
//! The model is never told the rule. A prompt `abc` is right when the reply is `bcd`; the verifier knows that, the
//! policy does not. The 512 prompts are shuffled once, and a method may learn from the first [`TRAIN`] and is scored
//! on the next [`HELD_OUT`], so a score that moves is a rule learnt, not prompts memorised.

/// Prompts a method may learn from.
pub const TRAIN: usize = 150;
/// Prompts a method is scored on and never sees.
pub const HELD_OUT: usize = 100;

/// The answer to `prompt`: every letter moves one place on, `h` wrapping to `a`.
pub fn cipher(prompt: &str) -> String {
    prompt
        .chars()
        .map(|c| char::from((c as u8 - b'a' + 1) % 8 + b'a'))
        .collect()
}

/// The verifier: whether `reply` answers `prompt`.
pub fn is_correct(prompt: &str, reply: &str) -> bool {
    reply == cipher(prompt)
}

/// The share of the answer's letters `reply` has right in place: the verifier's partial credit, which a method that
/// shapes a failed attempt's reward by it can learn from.
pub fn letters_right(prompt: &str, reply: &str) -> f64 {
    let answer = cipher(prompt);
    let right = answer
        .chars()
        .zip(reply.chars())
        .filter(|(a, b)| a == b)
        .count();
    right as f64 / answer.chars().count() as f64
}

/// The prompts a method may learn from and those it is scored on.
pub struct Split {
    pub train: Vec<String>,
    pub held_out: Vec<String>,
    /// Prompts nobody has answered: neither trained on nor scored on. A method that learns from its own attempts draws
    /// its prompts from here, a fresh slice each round, so it is not rewarded for memorising the ones it saw.
    pub pool: Vec<String>,
}

/// The fixed split: all 512 prompts in a fixed shuffle, so every run of every method sees the same.
pub fn split() -> Split {
    let mut all: Vec<String> = (0..512u32)
        .map(|i| {
            [i / 64, i / 8 % 8, i % 8]
                .iter()
                .map(|d| char::from(b'a' + *d as u8))
                .collect()
        })
        .collect();
    let mut state = 12345u64;
    for i in (1..all.len()).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        all.swap(i, (state >> 33) as usize % (i + 1));
    }
    let held_out = all[TRAIN..TRAIN + HELD_OUT].to_vec();
    let pool = all[TRAIN + HELD_OUT..].to_vec();
    all.truncate(TRAIN);
    Split {
        train: all,
        held_out,
        pool,
    }
}

#[test]
fn the_cipher_moves_every_letter_on_and_wraps() {
    assert_eq!(cipher("abh"), "bca");
    assert!(is_correct("abc", "bcd") && !is_correct("abc", "bcc"));
}

#[test]
fn the_held_out_prompts_are_not_among_the_training_prompts() {
    let split = split();
    assert_eq!((split.train.len(), split.held_out.len()), (TRAIN, HELD_OUT));
    assert!(split.held_out.iter().all(|p| !split.train.contains(p)));
    assert!(split
        .pool
        .iter()
        .all(|p| !split.train.contains(p) && !split.held_out.contains(p)));
}
