// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for proving, in seconds and on a CPU, that each learning method of a
// training system moves a held-out score, for its clients. If your team needs expertise in process (step-level)
// verification of multi-step work then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Process verification: a two-step task (apply the cipher, then apply it again to your own first answer) whose every
//! step a checker can judge in the context the conversation gave it. Both arms learn from the same number of
//! rollouts of the same model; one keeps what an outcome check keeps (conversations in which every step was right),
//! the other the steps the process checker labelled good, including the right first step of a conversation whose
//! second step was wrong. Scored on held-out prompts, end to end.

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use splinter_data::process::{outcome_only, project, Step, StepChecker};
use splinter_data::{write_dataset, Projection};

use crate::lab::Lab;
use crate::report::Outcome;
use crate::world::{cipher, cipher_twice, split};

const ROLE: &str = "Answer, then answer again.";
const CONTINUE: &str = ".";
/// Cooler than the other smokes: a step is only worth keeping if the sampler gets it right at all.
const ROLLOUT_TEMPERATURE: f32 = 0.4;
const ROUNDS: u64 = 2;
const DRAWS: u32 = 12;
const PROMPTS_PER_ROUND: usize = 60;

/// The checker: the first step answers the prompt, the second answers the first step, whatever that was.
struct Cipher;

impl StepChecker for Cipher {
    fn check(&self, conversation: &[Step], at: usize) -> bool {
        let before = if at == 0 {
            &conversation[0].said
        } else {
            &conversation[at - 1].step
        };
        conversation[at].step == cipher(before)
    }
}

/// The two steps a model takes at `prompt`: greedy when `sampling` is `None`, else a draw under that seed.
fn run(policy: &crate::lab::Policy<'_>, prompt: &str, sampling: Option<(f32, u64)>) -> Vec<Step> {
    let first = policy.reply_turns(&[("user", prompt)], sampling.map(|(t, s)| (t, s * 2)));
    let second = policy.reply_turns(
        &[("user", prompt), ("assistant", &first), ("user", CONTINUE)],
        sampling.map(|(t, s)| (t, s * 2 + 1)),
    );
    vec![
        Step {
            said: prompt.into(),
            step: first,
        },
        Step {
            said: CONTINUE.into(),
            step: second,
        },
    ]
}

/// The share of `prompts` answered right end to end, greedily.
fn accuracy(lab: &Lab, adapter: Option<&Path>, prompts: &[String]) -> f64 {
    let policy = lab.policy(adapter, splinter_model::answer::Decoding::Greedy);
    let right = prompts
        .iter()
        .filter(|p| run(&policy, p, None)[1].step == cipher_twice(p))
        .count();
    right as f64 / prompts.len() as f64
}

fn reference(prompts: &[String]) -> Vec<(String, Vec<Step>)> {
    prompts
        .iter()
        .map(|p| {
            (
                p.clone(),
                vec![
                    Step {
                        said: p.clone(),
                        step: cipher(p),
                    },
                    Step {
                        said: CONTINUE.into(),
                        step: cipher_twice(p),
                    },
                ],
            )
        })
        .collect()
}

fn write(lab: &Lab, name: &str, projection: &Projection) -> PathBuf {
    let path = lab.dir.path().join(format!("{name}.jsonl"));
    write_dataset(&path, projection, &splinter_data::Unchecked).unwrap();
    path
}

/// Rollouts of `adapter` on a slice of the pool, as conversations.
fn rollouts(lab: &Lab, adapter: &Path, round: u64) -> Vec<(String, Vec<Step>)> {
    let policy = lab.policy(Some(adapter), splinter_model::answer::Decoding::Greedy);
    let pool = &split().pool;
    let prompts =
        &pool[round as usize * PROMPTS_PER_ROUND..(round as usize + 1) * PROMPTS_PER_ROUND];
    let mut out = Vec::new();
    for (i, prompt) in prompts.iter().enumerate() {
        for draw in 0..DRAWS {
            let seed = round << 32 | (i as u64) << 8 | u64::from(draw);
            out.push((
                prompt.clone(),
                run(&policy, prompt, Some((ROLLOUT_TEMPERATURE, seed))),
            ));
        }
    }
    out
}

#[test]
#[ignore = "trains a toy model: make test/methods runs it"]
fn step_labels_teach_more_than_an_outcome_check_from_the_same_rollouts() {
    let lab = Lab::new();
    let split = split();
    let held_out = write(
        &lab,
        "held-out",
        &outcome_only(ROLE, &reference(&split.held_out), &Cipher),
    );
    let warm_set = write(
        &lab,
        "warm",
        &outcome_only(ROLE, &reference(&split.train[..60]), &Cipher),
    );
    let warm = lab.sft("warm", &warm_set, &held_out, 900, None);
    // The score is over the held-out prompts and a further slice of unused ones: 200 prompts, so a gap of a few
    // prompts is not the draw.
    let scored: Vec<String> = split
        .held_out
        .iter()
        .chain(&split.pool[150..250])
        .cloned()
        .collect();
    let start = accuracy(&lab, Some(&warm), &scored);
    println!("process warm start: {start:.2}");

    let (mut by_outcome, mut by_steps) = (warm.clone(), warm.clone());
    for round in 0..ROUNDS {
        for (name, adapter, steps_labelled) in [
            ("outcome", &mut by_outcome, false),
            ("steps", &mut by_steps, true),
        ] {
            let conversations = rollouts(&lab, adapter, round);
            let projection = if steps_labelled {
                project(ROLE, &conversations, &Cipher)
            } else {
                outcome_only(ROLE, &conversations, &Cipher)
            };
            let good = |at: usize| {
                conversations
                    .iter()
                    .filter(|(_, c)| Cipher.check(c, at))
                    .count() as f64
                    / conversations.len() as f64
            };
            println!("process {name} round {round}: {} records from {} conversations (steps right: first {:.2}, second {:.2})", projection.records.len(), conversations.len(), good(0), good(1));
            let data = write(&lab, &format!("{name}-{round}"), &projection);
            // About four passes over the records whichever arm they came from: an arm with few records must not be
            // trained into memorising them.
            let steps = u32::try_from(projection.records.len())
                .unwrap()
                .clamp(30, 300);
            *adapter = lab.sft_at(
                &format!("{name}-{round}"),
                &data,
                &held_out,
                steps,
                1e-3,
                Some(adapter),
            );
        }
    }
    let (outcome_score, steps_score) = (
        accuracy(&lab, Some(&by_outcome), &scored),
        accuracy(&lab, Some(&by_steps), &scored),
    );
    let note = format!("same rollouts, 200 prompts; outcome-only check vs step labels, the model they started from scores {start:.2}; the gap is within the draw");
    Outcome {
        method: "Process (step-level) verification",
        metric: "held-out two-step exact match",
        before: outcome_score,
        after: steps_score,
        min_gain: 0.05,
        note: &note,
    }
    .report();
}
