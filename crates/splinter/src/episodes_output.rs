// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that turn practice task folders
// into graded, replayable agent experience for its clients. If your team
// needs expertise in on-policy experience collection for agent training you
// can procure our services by sending an email to info@swedishembedded.com.

//! How `tasks harbor episodes` and `tasks harbor practice` read to a person.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use splinter_sdk::bench::practice::PracticeReport;
use splinter_sdk::episodes::{EpisodeOutcome, EpisodeReport};

use crate::output::Report;

/// How the attempts at one task ended.
#[derive(Default)]
pub(crate) struct Tally {
    attempts: usize,
    passed: usize,
    failed: usize,
    exposed: usize,
    verifier: usize,
    ungraded: usize,
    infrastructure: usize,
    rewards: Vec<f64>,
}

impl Tally {
    /// Counts an attempt that ended in `outcome` with `reward`.
    pub(crate) fn count(&mut self, outcome: &EpisodeOutcome, reward: Option<f64>) {
        self.attempts += 1;
        match outcome {
            EpisodeOutcome::Passed => self.passed += 1,
            EpisodeOutcome::Failed => self.failed += 1,
            EpisodeOutcome::Exposed => self.exposed += 1,
            EpisodeOutcome::VerifierError { .. } | EpisodeOutcome::VerifierTimeout => {
                self.verifier += 1;
            }
            EpisodeOutcome::Ungraded { .. } => self.ungraded += 1,
            EpisodeOutcome::Infrastructure { .. } => self.infrastructure += 1,
        }
        self.rewards.extend(reward);
    }

    /// The task `name`'s line: its attempts by outcome and the mean reward
    /// of those that have one.
    pub(crate) fn line(&self, name: &str) -> String {
        let mean = (!self.rewards.is_empty())
            .then(|| self.rewards.iter().sum::<f64>() / self.rewards.len() as f64);
        format!(
            "  {name}: {} attempt(s): {} passed, {} failed, {} exposed, {} verifier failure(s), {} ungraded, {} infrastructure failure(s); mean reward {}",
            self.attempts,
            self.passed,
            self.failed,
            self.exposed,
            self.verifier,
            self.ungraded,
            self.infrastructure,
            mean.map_or_else(|| "none".to_string(), |m| format!("{m:.3}"))
        )
    }
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
            tasks
                .entry(record.name.as_str())
                .or_default()
                .count(&record.outcome, record.reward);
        }
        for (name, tally) in &tasks {
            let _ = writeln!(out, "{}", tally.line(name));
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

impl Report for PracticeReport {
    fn human(&self) -> String {
        let mut out = format!(
            "{} trial(s) of {} ({}) by {} served as {}, seed {}, job {} ({})\n",
            self.trials.len(),
            self.provenance.collection,
            self.provenance.practice_sha256,
            self.policy.model,
            self.served_as,
            self.seed,
            self.job,
            if self.ran {
                "run now"
            } else {
                "finished earlier"
            }
        );
        let mut tasks: BTreeMap<&str, Tally> = BTreeMap::new();
        let mut all = Tally::default();
        for trial in &self.trials {
            tasks
                .entry(trial.task.as_str())
                .or_default()
                .count(&trial.outcome, trial.reward);
            all.count(&trial.outcome, trial.reward);
        }
        for (name, tally) in &tasks {
            let _ = writeln!(out, "{}", tally.line(name));
        }
        let _ = writeln!(out, "{}", all.line("all"));
        let admitted: usize = self.ingested.admitted.values().sum();
        let _ = writeln!(
            out,
            "  admitted {admitted} trial(s), {} held already, {} refused; train set {}",
            self.ingested.already_held,
            self.ingested.refused.len(),
            self.ingested
                .train_set
                .as_ref()
                .map_or_else(|| "none".to_string(), ToString::to_string)
        );
        for refused in &self.ingested.refused {
            let _ = writeln!(
                out,
                "  refused {}: {}",
                refused.trial.as_deref().unwrap_or("the export"),
                refused.reason
            );
        }
        let _ = writeln!(
            out,
            "  the seed names the run; the examiner's agent sends no sampling seed, so the policy \
             sampled as its server does"
        );
        let _ = writeln!(out, "  run directory {}", self.dir.display());
        out
    }
}
