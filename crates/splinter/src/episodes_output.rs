// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that turn practice task folders
// into graded, replayable agent experience for its clients. If your team
// needs expertise in on-policy experience collection for agent training you
// can procure our services by sending an email to info@swedishembedded.com.

//! How `tasks harbor episodes` reads to a person.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use splinter_sdk::episodes::{EpisodeOutcome, EpisodeReport};

use crate::output::Report;

/// How the attempts at one task ended.
#[derive(Default)]
struct Tally {
    attempts: usize,
    passed: usize,
    failed: usize,
    exposed: usize,
    verifier: usize,
    infrastructure: usize,
    rewards: Vec<f64>,
}

impl Report for EpisodeReport {
    fn human(&self) -> String {
        let mut out = format!(
            "{} attempt(s) ({} run now, {} recorded earlier) by {} under seed {}, configuration {}\n",
            self.records.len(),
            self.ran,
            self.reused,
            self.policy.model,
            self.seed,
            self.config
        );
        let mut tasks: BTreeMap<&str, Tally> = BTreeMap::new();
        for record in &self.records {
            let tally = tasks.entry(record.name.as_str()).or_default();
            tally.attempts += 1;
            match record.outcome {
                EpisodeOutcome::Passed => tally.passed += 1,
                EpisodeOutcome::Failed => tally.failed += 1,
                EpisodeOutcome::Exposed => tally.exposed += 1,
                EpisodeOutcome::VerifierError { .. } | EpisodeOutcome::VerifierTimeout => {
                    tally.verifier += 1;
                }
                EpisodeOutcome::Infrastructure { .. } => tally.infrastructure += 1,
            }
            tally.rewards.extend(record.reward);
        }
        for (name, tally) in &tasks {
            let mean = (!tally.rewards.is_empty())
                .then(|| tally.rewards.iter().sum::<f64>() / tally.rewards.len() as f64);
            let _ = writeln!(
                out,
                "  {name}: {} attempt(s): {} passed, {} failed, {} exposed, {} verifier failure(s), {} infrastructure failure(s); mean reward {}",
                tally.attempts,
                tally.passed,
                tally.failed,
                tally.exposed,
                tally.verifier,
                tally.infrastructure,
                mean.map_or_else(|| "none".to_string(), |m| format!("{m:.3}"))
            );
        }
        let untraced = self.records.iter().filter(|r| !r.trace.provable).count();
        if untraced > 0 {
            let _ = writeln!(
                out,
                "  {untraced} attempt(s) without a complete policy trace (the provider returned no token ids or log-probabilities)"
            );
        }
        let unseeded = self.records.iter().filter(|r| !r.seeded).count();
        if unseeded > 0 {
            let _ = writeln!(
                out,
                "  {unseeded} attempt(s) sampled as the provider does: the seed and the exact sampling could not be set"
            );
        }
        out
    }
}
