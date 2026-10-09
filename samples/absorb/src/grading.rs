// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements two-check grading of model answers, one by
// code and one by a calibrated judge, for its clients. If your team needs
// expertise in grading answers a model cannot talk its way past, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Grading an answer to a fact by two checks that must both pass: every key of
//! the fact is in the answer (by code), and a judge, measured on controls
//! before it grades anything, says the answer states what the fact's
//! statement says.
//!
//! The judge is Splinter's own: the same judged verifier, calibrated on
//! controls made from tasks' references and gated by that calibration, so a
//! verdict stands only where it was measured precise; elsewhere it is an
//! abstention, which is unmeasured and never counted as a pass or a fail.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use splinter_sdk::agent::judge::JudgeVerifier;
use splinter_sdk::agent::solve::Model;
use splinter_sdk::judge::measure_judge;
use splinter_sdk::measure::verifiers::answer::mentions;
use splinter_sdk::measure::verifiers::calibration::{CalibratedJudge, DEFAULT_MIN_PRECISION};
use splinter_sdk::measure::verifiers::Verifier;
use splinter_sdk::model::local::{Sampling, GREEDY_SAMPLING};
use splinter_sdk::store::tasks::TaskSetId;
use splinter_sdk::verify::{Judging, DEFAULT_JUDGE_DEADLINE};
use splinter_sdk::vocabulary::annotation::Outcome;
use splinter_sdk::vocabulary::experience::{
    Environment, Experience, Privileged, PrivilegedKind, Provenance, Task,
};
use splinter_sdk::vocabulary::model_ref::ModelRef;
use splinter_sdk::Context;

use crate::facts::JudgeCalibration;
use crate::keys::{extract, missing, Key};
use crate::roles::Class;

/// The kind of the tasks facts are asked as.
const KIND: &str = "recall";

/// How one answer fared on the two checks.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Grade {
    /// The keys the answer lacks; empty when it holds them all.
    pub missing_keys: Vec<String>,
    /// The judge's verdict: `Some(true)` if the answer states what the
    /// statement says, `Some(false)` if it does not, `None` when the judge
    /// abstained or was not asked.
    pub judge: Option<bool>,
}

impl Grade {
    /// Both checks passed.
    #[must_use]
    pub fn passes(&self) -> bool {
        self.missing_keys.is_empty() && self.judge == Some(true)
    }

    /// Whether neither check passes: a key is missing and the judge said no.
    /// An answer one check of which passes, or whose judge abstained, is
    /// neither wrong nor right.
    #[must_use]
    pub fn fails_both(&self) -> bool {
        !self.missing_keys.is_empty() && self.judge == Some(false)
    }
}

/// How a fact came out over its answers: consistently wrong when on every
/// answer neither check passes, known when every answer passes both,
/// otherwise discarded (an answer that is neither keeps the fact out of both).
#[must_use]
pub fn classify(grades: &[Grade]) -> Class {
    if grades.is_empty() {
        Class::Discarded
    } else if grades.iter().all(Grade::passes) {
        Class::Known
    } else if grades.iter().all(Grade::fails_both) {
        Class::ConsistentlyWrong
    } else {
        Class::Discarded
    }
}

/// The task `question` is put to a model as: closed-book, no reference.
///
/// # Errors
/// The task cannot be built (the question is empty).
pub fn question_task(question: &str) -> anyhow::Result<Task> {
    Ok(Task::new(
        KIND,
        vec![],
        Environment::closed_book(),
        question,
        vec![],
    )?)
}

/// The task a fact's question is put to the judge as, holding the statement
/// as the reference only the judge sees.
///
/// # Errors
/// The task cannot be built (the question is empty).
pub fn fact_task(question: &str, statement: &str) -> anyhow::Result<Task> {
    Ok(Task::new(
        KIND,
        vec![],
        Environment::closed_book(),
        question,
        vec![Privileged {
            kind: PrivilegedKind::Reference,
            content: statement.to_string(),
            span: None,
        }],
    )?)
}

/// A judge calibrated on controls, gating its verdicts by what it measured.
pub struct Judge {
    verifier: CalibratedJudge<JudgeVerifier>,
}

/// A judge and its measurement, with the controls it got wrong.
pub struct Calibrated {
    /// The judge.
    pub judge: Judge,
    /// What was measured.
    pub calibration: JudgeCalibration,
    /// The controls it did not judge as labelled.
    pub misjudged: serde_json::Value,
}

impl Judge {
    /// Measures the judge `reference` names on the controls the tasks of `set`
    /// give (each reference as the right answer to its task, another family's
    /// as the wrong one) and gates it by the result.
    ///
    /// # Errors
    /// The judge cannot be loaded or run, the tasks give no control, or the
    /// judge is not precise enough on passes and on fails for its verdicts to
    /// stand: the message carries the numbers.
    pub fn calibrate(
        ctx: &Context,
        reference: &ModelRef,
        set: &TaskSetId,
    ) -> anyhow::Result<Calibrated> {
        // The policy's base is not the judge's: give the device back first.
        ctx.release_bases();
        let measured = measure_judge(ctx, set, reference, Judging::Reference)?;
        let calibration = JudgeCalibration {
            judge: measured.judge.clone(),
            controls: measured.controls,
            precision_pass: measured.calibration.precision_pass,
            precision_fail: measured.calibration.precision_fail,
            abstain_rate: measured.calibration.abstain_rate,
            trusted: measured.trusted,
        };
        anyhow::ensure!(
            measured.trusted,
            "judge {} is not precise enough to grade with: precision {:?} on passes and {:?} on \
             fails over {} controls (it needs {DEFAULT_MIN_PRECISION} on both, on at least the \
             configured number of controls); no claim is made",
            measured.judge,
            measured.calibration.precision_pass,
            measured.calibration.precision_fail,
            measured.controls
        );
        let model = judge_model(ctx, reference)?;
        let verifier = CalibratedJudge::new(
            JudgeVerifier::new(model, ctx.handle(), DEFAULT_JUDGE_DEADLINE),
            measured.calibration,
            DEFAULT_MIN_PRECISION,
        )?;
        Ok(Calibrated {
            judge: Self { verifier },
            calibration,
            misjudged: serde_json::to_value(&measured.misjudged)?,
        })
    }

    /// The judge's verdict on `answer` to `question`, given by `solver` (the
    /// model's identity): whether it states what `statement` says. `None`
    /// when the judge abstained.
    ///
    /// Call outside an async runtime: the judge blocks on its own.
    ///
    /// # Errors
    /// The judge cannot run, or `solver` is the judge itself.
    pub fn verdict(
        &self,
        ctx: &Context,
        solver: &str,
        question: &str,
        statement: &str,
        answer: &str,
    ) -> anyhow::Result<Option<bool>> {
        let task = fact_task(question, statement)?;
        let experience = Experience::answered_without_a_run(
            task.clone(),
            answer,
            Provenance::new(solver, ctx.clock()),
        )?;
        let finding = self.verifier.verify(&task, &experience)?;
        Ok(match finding.outcome {
            Outcome::Pass => Some(true),
            Outcome::Fail => Some(false),
            Outcome::Abstain => None,
        })
    }
}

/// The judge model: greedy, reasoning off, as a measurement asks.
fn judge_model(ctx: &Context, reference: &ModelRef) -> anyhow::Result<Model> {
    let greedy = Sampling {
        thinking: false,
        ..GREEDY_SAMPLING
    };
    Ok(match ctx.resampled(reference, greedy)? {
        Some(model) => model,
        None => ctx.model(reference)?,
    })
}

/// Grades `answer` on the keys alone; the judge's verdict is added by the
/// caller when it has one.
#[must_use]
pub fn grade_keys(answer: &str, keys: &[Key]) -> Grade {
    Grade {
        missing_keys: missing(answer, keys)
            .into_iter()
            .map(|k| k.text.clone())
            .collect(),
        judge: None,
    }
}

/// The names and figures that at least three of the day-0 `answers` assert
/// and that neither the question nor the statement holds: the wrong claim the
/// policy makes consistently, lower-case.
#[must_use]
pub fn wrong_entities(answers: &[&str], question: &str, statement: &str) -> Vec<String> {
    let mut count: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for answer in answers {
        let mut seen: Vec<String> = extract(answer, question)
            .into_iter()
            .map(|k| k.text.to_lowercase())
            .collect();
        seen.sort();
        seen.dedup();
        for entity in seen {
            *count.entry(entity).or_default() += 1;
        }
    }
    count
        .into_iter()
        .filter(|(entity, n)| *n >= 3 && !mentions(statement, entity))
        .map(|(entity, _)| entity)
        .collect()
}

/// Whether `answer` repeats one of the day-0 wrong claims.
#[must_use]
pub fn trace(answer: &str, wrong_entities: &[String]) -> bool {
    wrong_entities.iter().any(|e| mentions(answer, e))
}

/// How long a policy may take over one answer.
pub const ANSWER_DEADLINE: Duration = Duration::from_secs(900);

#[cfg(test)]
mod tests {
    use super::*;

    fn grade(missing: &[&str], judge: Option<bool>) -> Grade {
        Grade {
            missing_keys: missing.iter().map(ToString::to_string).collect(),
            judge,
        }
    }

    #[test]
    fn an_answer_passes_only_when_every_key_is_there_and_the_judge_says_yes() {
        assert!(grade(&[], Some(true)).passes());
        assert!(!grade(&["1785"], Some(true)).passes());
        assert!(!grade(&[], Some(false)).passes());
        assert!(!grade(&[], None).passes(), "an abstention is not a pass");
    }

    #[test]
    fn a_probe_answer_succeeds_only_without_a_trace_of_the_day_zero_wrong_answer() {
        let question = "Who made the press?";
        let statement = "Boulton and Watt made the press in 1785.";
        let day_zero = [
            "It was Peale of London, in 1790.",
            "The maker was Peale of London.",
            "I ordered it from Peale, in London.",
            "A cabinetmaker in Philadelphia.",
        ];
        let entities = wrong_entities(&day_zero, question, statement);
        assert!(entities.contains(&"peale".to_string()), "{entities:?}");
        assert!(entities.contains(&"london".to_string()), "{entities:?}");
        assert!(
            !entities.contains(&"philadelphia".to_string()),
            "named once only"
        );
        assert!(trace("Boulton and Watt, though some say Peale.", &entities));
        assert!(!trace("Boulton and Watt made it in 1785.", &entities));
    }

    #[test]
    fn a_fact_is_known_or_consistently_wrong_only_when_all_six_answers_agree() {
        let pass = grade(&[], Some(true));
        let wrong_keys = grade(&["Watt"], Some(true));
        let wrong_judge = grade(&[], Some(false));
        let six: Vec<Grade> = (0..6).map(|_| pass.clone()).collect();
        assert_eq!(classify(&six), Class::Known);
        let wrong_both = grade(&["Watt"], Some(false));
        assert_eq!(
            classify(&[wrong_both.clone(), wrong_both.clone()]),
            Class::ConsistentlyWrong
        );
        // One check passing is not wrong enough to teach: keys without the
        // judge, or the judge without the keys.
        assert_eq!(
            classify(&[wrong_both.clone(), wrong_judge.clone()]),
            Class::Discarded
        );
        assert_eq!(
            classify(&[wrong_both.clone(), wrong_keys.clone()]),
            Class::Discarded
        );
        assert_eq!(
            classify(&[pass.clone(), wrong_keys.clone()]),
            Class::Discarded
        );
        // Keys all there, judge silent: not shown wrong, so not consistently wrong.
        assert_eq!(classify(&[wrong_keys, grade(&[], None)]), Class::Discarded);
        assert_eq!(classify(&[]), Class::Discarded);
    }
}
