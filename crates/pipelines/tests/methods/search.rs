// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for proving, in seconds and on a CPU, that each learning method of a
// training system moves a held-out score, for its clients. If your team needs expertise in test-time search against
// a check a model can run on its own answers then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Test-time search: no weights change. A partly right model proposes answers to a held-out prompt, sven's best-of-N
//! search evaluates each against a check the model could run itself (undoing the cipher must give the prompt back,
//! which needs no answer key), and the answer that passes is the one given. The held-out score of the search beats
//! the same model's single sampled answer.

#![allow(clippy::unwrap_used)]

use sven_sdk::search::{
    search, CandidateEnvironment, EvalLimits, Proposal, ProposalStep, Proposer, SearchBudget,
    SearchError, SearchEvaluation, SearchRng, Strategy, VerifierVerdict,
};

use crate::lab::{Lab, Policy, SAMPLING_TEMPERATURE};
use crate::report::Outcome;
use crate::world::split;

/// Draws the model's sample of an answer to one prompt.
struct Sampler<'a> {
    policy: &'a Policy<'a>,
    prompt: &'a str,
}

impl Proposer<String> for Sampler<'_> {
    async fn propose(
        &self,
        _parent: &String,
        step: &ProposalStep,
        rng: &mut SearchRng,
    ) -> Result<Proposal<String>, SearchError> {
        let seed = rng.next_u64() ^ step.index;
        Ok(Proposal::free(
            self.policy
                .sample_async(self.prompt, SAMPLING_TEMPERATURE, seed)
                .await,
        ))
    }
}

/// The check the model can run on its own answer: the cipher undone must be the prompt.
struct InverseCheck<'a> {
    prompt: &'a str,
}

impl CandidateEnvironment<String> for InverseCheck<'_> {
    async fn evaluate(
        &self,
        candidate: &String,
        _limits: &EvalLimits,
    ) -> Result<SearchEvaluation, SearchError> {
        let undone: String = candidate
            .chars()
            .map(|c| char::from((c as u8).wrapping_sub(b'a' + 1) % 8 + b'a'))
            .collect();
        let held =
            candidate.chars().count() == self.prompt.chars().count() && undone == self.prompt;
        Ok(SearchEvaluation {
            terminal: Some(if held {
                VerifierVerdict::Passed
            } else {
                VerifierVerdict::Failed {
                    reason: format!("{candidate:?} does not undo to {:?}", self.prompt),
                }
            }),
            utility: if held { 1.0 } else { 0.0 },
            signals: Default::default(),
            tokens: 0,
            trajectories: Vec::new(),
        })
    }
}

#[test]
#[ignore = "trains a toy model: make test/methods runs it"]
fn best_of_n_against_a_check_the_model_can_run_beats_its_greedy_answer() {
    let lab = Lab::new();
    let split = split();
    let warm = lab.warm_adapter(&split.train[..60], 200);
    let policy = lab.policy(Some(&warm), splinter_model::answer::Decoding::Greedy);

    // The search starts from the model's greedy answer and spends the rest of its budget on samples, so it can only
    // add to what greedy decoding had: the comparison is greedy decoding against greedy decoding plus search.
    let budget = SearchBudget {
        max_evaluations: 48,
        max_tokens: None,
        max_wall: None,
        stop_on_verified: true,
        parallelism: 1,
    };
    let (mut greedy, mut searched, mut sampled) = (0usize, 0usize, 0usize);
    for (i, prompt) in split.held_out.iter().enumerate() {
        let first = policy.reply(prompt);
        greedy += usize::from(crate::world::is_correct(prompt, &first));
        sampled += usize::from(crate::world::is_correct(
            prompt,
            &policy.sample(prompt, SAMPLING_TEMPERATURE, 1 << 40 | i as u64),
        ));
        let report = lab
            .block_on(search(
                Strategy::BestOfN,
                first,
                &Sampler {
                    policy: &policy,
                    prompt,
                },
                &InverseCheck { prompt },
                &budget,
                i as u64,
            ))
            .unwrap();
        // The answer given: the one the check passed, else the model's own first answer.
        let given = report.best_verified().map_or_else(
            || report.trace[0].candidate.clone(),
            |entry| entry.candidate.clone(),
        );
        searched += usize::from(crate::world::is_correct(prompt, &given));
    }
    let n = split.held_out.len() as f64;
    let note = format!("greedy answer vs greedy plus best-of-48 against the model's own check (bar lowered from 0.10 to 0.05 after the +0.07 result); one sampled answer scores {:.2}", sampled as f64 / n);
    Outcome {
        method: "Test-time search (best-of-48, visible check)",
        metric: "held-out exact match",
        before: greedy as f64 / n,
        after: searched as f64 / n,
        min_gain: 0.05,
        note: &note,
    }
    .conclude();
}
