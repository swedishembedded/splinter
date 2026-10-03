// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements consistency verifiers that grade agent
// answers by agreement between independent attempts, for its clients. If
// your team needs expertise in verifier design, you can procure our services
// by sending an email to info@swedishembedded.com.

//! The consistency verifier: an answer graded by whether independent
//! answers to the same task agree with it.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::json;
use splinter_core::annotation::{Producer, Strength};
use splinter_core::digest::Digest;
use splinter_core::experience::{Experience, ExperienceId, Task};

use super::normalise::Normalisation;
use super::{Finding, Verifier, VerifyError};

/// The producer name the verifier's annotations carry.
pub const PRODUCER: &str = "splinter-lab/agreement";

/// The verifier's version.
pub const VERSION: &str = "1";

/// The fewest answers a verdict is reached from by default: two answers
/// that agree are only each other's echo.
pub const DEFAULT_MIN_ANSWERS: usize = 3;

/// Grades an answer against the other answers to its task among a pool of
/// experiences (possibly from different solvers). The answers counted are
/// the pool's experiences of the same task that gave one, and the graded
/// experience itself, each once. Among `n` answers, compared under a
/// [`Normalisation`]: pass when the graded answer is shared by more than
/// `n / 2`, fail when another answer is (or the graded experience gave
/// none and one is), abstain otherwise or when fewer than the minimum
/// answered. The evidence records the tally by answer digest with the
/// solvers that gave each.
#[derive(Clone, Debug)]
pub struct AgreementVerifier {
    pool: Vec<Experience>,
    normalisation: Normalisation,
    min_answers: usize,
}

impl AgreementVerifier {
    /// A verifier over `pool`, needing [`DEFAULT_MIN_ANSWERS`] answers.
    #[must_use]
    pub fn new(pool: Vec<Experience>, normalisation: Normalisation) -> Self {
        Self {
            pool,
            normalisation,
            min_answers: DEFAULT_MIN_ANSWERS,
        }
    }

    /// The same verifier, needing `min_answers` answers (at least two).
    pub fn with_min_answers(mut self, min_answers: usize) -> Result<Self, VerifyError> {
        if min_answers < 2 {
            return Err(VerifyError::Parameter {
                name: "min_answers",
                reason: format!("{min_answers} answers cannot disagree"),
            });
        }
        self.min_answers = min_answers;
        Ok(self)
    }
}

impl Verifier for AgreementVerifier {
    fn producer(&self) -> Producer {
        Producer {
            name: PRODUCER.into(),
            version: VERSION.into(),
        }
    }

    fn strength(&self) -> Strength {
        Strength::Consistency
    }

    fn verify(&self, task: &Task, exp: &Experience) -> Result<Finding, VerifyError> {
        let mut counted: BTreeSet<ExperienceId> = BTreeSet::new();
        // answer digest -> the solvers that gave it, one entry per answer.
        let mut tally: BTreeMap<Digest, Vec<String>> = BTreeMap::new();
        for other in self.pool.iter().chain(std::iter::once(exp)) {
            if other.task != task.task || !counted.insert(other.id()?) {
                continue;
            }
            if let Some(answer) = &other.final_output {
                let answer = self.normalisation.apply(answer);
                tally
                    .entry(Digest::of(answer.as_bytes()))
                    .or_default()
                    .push(other.provenance.solver.clone());
            }
        }
        let answers: usize = tally.values().map(Vec::len).sum();
        let mut groups: Vec<(Digest, Vec<String>)> = tally.into_iter().collect();
        groups.sort_by(|(a, x), (b, y)| y.len().cmp(&x.len()).then_with(|| a.cmp(b)));
        let this = exp
            .final_output
            .as_deref()
            .map(|o| Digest::of(self.normalisation.apply(o).as_bytes()));
        let majority = groups
            .iter()
            .find(|(_, solvers)| solvers.len() * 2 > answers)
            .map(|(digest, _)| digest.clone());
        let evidence = json!({
            "normalisation": self.normalisation,
            "answers": answers,
            "answer": this,
            "majority": majority,
            "tally": groups
                .into_iter()
                .map(|(digest, mut solvers)| {
                    solvers.sort_unstable();
                    json!({ "answer": digest, "count": solvers.len(), "solvers": solvers })
                })
                .collect::<Vec<_>>(),
        });
        if answers < self.min_answers {
            return Ok(Finding::abstain("too few answers to compare", evidence));
        }
        match majority {
            None => Ok(Finding::abstain(
                "no answer has a strict majority",
                evidence,
            )),
            Some(majority) => Ok(Finding::decided(this.as_ref() == Some(&majority), evidence)),
        }
    }
}
