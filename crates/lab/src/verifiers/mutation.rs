// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements mutation-validated test oracles for
// grading model-written code, for its clients. If your team needs expertise
// in mutation testing or verifier design, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Generated tests admitted as executable oracles only when mutation shows
//! they can tell right from wrong.
//!
//! A test a model wrote is worth nothing as an oracle until it is shown to
//! reject wrong code. [`validate_oracle`] runs it on the task's reference
//! solution, where it must pass, and on mutants of that reference, where it
//! must fail. A mutant is the reference with one small textual change
//! ([`MutationOperator`]); the operators are language-agnostic text
//! rewrites, applied in a fixed order so the same reference always yields
//! the same mutants. A mutant the test fails is killed, one it passes
//! survived; a mutant that does not run cleanly on its own (a deletion that
//! breaks the syntax, say) says nothing about the test and is counted as
//! errored and left out. The test is admitted when its kill rate
//! (killed / (killed + survived)) reaches [`MutationPolicy::min_kill_rate`],
//! by default [`DEFAULT_MIN_KILL_RATE`] (every mutant killed), over at most
//! [`DEFAULT_MAX_MUTANTS`] mutants. A mutant that is equivalent to the
//! reference survives every test; a lower threshold tolerates those.
//!
//! [`MutationValidatedVerifier`] grades a solution with admitted tests only,
//! validating them afresh on every verification rather than trusting a
//! recorded admission.

use std::collections::HashSet;

use serde::Serialize;
use serde_json::json;
use splinter_core::annotation::{Producer, Strength};
use splinter_core::digest::Digest;
use splinter_core::experience::{Experience, Task};
use splinter_sandbox::{CodeCall, RuntimeEnvironment};

use super::executable::{
    assemble, parse_checks, program_text, run_check, run_evidence, ExecutableCheck, Offered,
};
use super::{single_reference, Finding, Verifier, VerifyError};

/// The privileged kind a generated test travels as.
pub const TEST_KIND: &str = "generated-test";

/// The producer name the verifier's annotations carry.
pub const PRODUCER: &str = "splinter-lab/mutation-validated";

/// The verifier's version.
pub const VERSION: &str = "2";

/// The most mutants a test is run against by default.
pub const DEFAULT_MAX_MUTANTS: usize = 32;

/// The kill rate a test needs by default: every mutant that runs killed.
pub const DEFAULT_MIN_KILL_RATE: f64 = 1.0;

/// One textual change to source code.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MutationOperator {
    /// A comparison flipped to its negation: `<` and `>=`, `>` and `<=`,
    /// `==` and `!=`, `===` and `!==` (shifts, arrows and `<=>` untouched).
    FlipComparison,
    /// A decimal integer literal one more, or one less (never below zero).
    OffByOne,
    /// `True` and `False`, `true` and `false` swapped, as whole words.
    SwapBoolean,
    /// One non-blank line that is not a `#` or `//` comment emptied.
    DeleteLine,
}

/// A mutant of a source text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mutant {
    /// The change made.
    pub operator: MutationOperator,
    /// The 1-based line it was made on.
    pub line: usize,
    /// The mutated source.
    pub text: String,
}

/// Up to `max` mutants of `source`, distinct and all different from it.
/// The operators take turns, each in source order, so a cap keeps a mix.
#[must_use]
pub fn mutants(source: &str, max: usize) -> Vec<Mutant> {
    let mut per_operator = [
        flip_comparisons(source),
        off_by_one(source),
        swap_booleans(source),
        delete_lines(source),
    ]
    .map(Vec::into_iter);
    let mut seen: HashSet<String> = HashSet::from([source.to_string()]);
    let mut out = Vec::new();
    loop {
        let mut progressed = false;
        for candidates in &mut per_operator {
            if out.len() >= max {
                return out;
            }
            if let Some(mutant) = candidates.next() {
                progressed = true;
                if seen.insert(mutant.text.clone()) {
                    out.push(mutant);
                }
            }
        }
        if !progressed {
            return out;
        }
    }
}

fn mutant(
    source: &str,
    operator: MutationOperator,
    start: usize,
    end: usize,
    with: &str,
) -> Mutant {
    Mutant {
        operator,
        line: source[..start].matches('\n').count() + 1,
        text: format!("{}{with}{}", &source[..start], &source[end..]),
    }
}

fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn flip_comparisons(source: &str) -> Vec<Mutant> {
    let b = source.as_bytes();
    let at = |i: usize| b.get(i).copied().unwrap_or(b' ');
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let prev = if i == 0 { b' ' } else { b[i - 1] };
        let flip: Option<(usize, &str)> = match (b[i], at(i + 1), at(i + 2)) {
            (b'=', b'=', b'=') => Some((3, "!==")),
            (b'!', b'=', b'=') => Some((3, "===")),
            (b'=', b'=', _) => Some((2, "!=")),
            (b'!', b'=', _) => Some((2, "==")),
            (b'<', b'=', next) if prev != b'<' && next != b'>' => Some((2, ">")),
            (b'>', b'=', _) if !matches!(prev, b'>' | b'=' | b'<') => Some((2, "<")),
            (b'<', next, _) if !matches!(next, b'<' | b'=' | b'-') && !matches!(prev, b'<') => {
                Some((1, ">="))
            }
            (b'>', next, _)
                if !matches!(next, b'>' | b'=') && !matches!(prev, b'>' | b'-' | b'=') =>
            {
                Some((1, "<="))
            }
            _ => None,
        };
        match flip {
            Some((len, with)) => {
                out.push(mutant(
                    source,
                    MutationOperator::FlipComparison,
                    i,
                    i + len,
                    with,
                ));
                i += len;
            }
            None => i += 1,
        }
    }
    out
}

fn off_by_one(source: &str) -> Vec<Mutant> {
    let b = source.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if !b[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        let start = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        let before = start.checked_sub(1).map(|p| b[p]);
        let after = b.get(i).copied();
        let bounded = |c: Option<u8>| c.is_none_or(|c| !is_word_byte(c) && c != b'.');
        if !bounded(before) || !bounded(after) {
            continue;
        }
        let Ok(value) = source[start..i].parse::<u64>() else {
            continue;
        };
        if let Some(up) = value.checked_add(1) {
            out.push(mutant(
                source,
                MutationOperator::OffByOne,
                start,
                i,
                &up.to_string(),
            ));
        }
        if let Some(down) = value.checked_sub(1) {
            out.push(mutant(
                source,
                MutationOperator::OffByOne,
                start,
                i,
                &down.to_string(),
            ));
        }
    }
    out
}

fn swap_booleans(source: &str) -> Vec<Mutant> {
    let b = source.as_bytes();
    let mut found: Vec<(usize, usize, &str)> = Vec::new();
    for (word, with) in [
        ("True", "False"),
        ("False", "True"),
        ("true", "false"),
        ("false", "true"),
    ] {
        for (start, _) in source.match_indices(word) {
            let end = start + word.len();
            let before = start.checked_sub(1).map(|p| b[p]);
            let after = b.get(end).copied();
            if before.is_none_or(|c| !is_word_byte(c)) && after.is_none_or(|c| !is_word_byte(c)) {
                found.push((start, end, with));
            }
        }
    }
    found.sort_unstable_by_key(|(start, _, _)| *start);
    found
        .into_iter()
        .map(|(start, end, with)| mutant(source, MutationOperator::SwapBoolean, start, end, with))
        .collect()
}

fn delete_lines(source: &str) -> Vec<Mutant> {
    let mut out = Vec::new();
    let mut start = 0;
    for line in source.split_inclusive('\n') {
        let content = line.trim_end_matches(['\n', '\r']);
        let trimmed = content.trim();
        if !trimmed.is_empty() && !trimmed.starts_with('#') && !trimmed.starts_with("//") {
            out.push(mutant(
                source,
                MutationOperator::DeleteLine,
                start,
                start + content.len(),
                "",
            ));
        }
        start += line.len();
    }
    out
}

/// How a generated test is validated.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct MutationPolicy {
    max_mutants: usize,
    min_kill_rate: f64,
}

impl Default for MutationPolicy {
    /// [`DEFAULT_MAX_MUTANTS`] and [`DEFAULT_MIN_KILL_RATE`].
    fn default() -> Self {
        Self {
            max_mutants: DEFAULT_MAX_MUTANTS,
            min_kill_rate: DEFAULT_MIN_KILL_RATE,
        }
    }
}

impl MutationPolicy {
    /// At most `max_mutants` (at least one) mutants, and a kill rate of
    /// `min_kill_rate` (above zero, at most one) to admit a test.
    pub fn new(max_mutants: usize, min_kill_rate: f64) -> Result<Self, VerifyError> {
        if max_mutants == 0 {
            return Err(VerifyError::Parameter {
                name: "max_mutants",
                reason: "at least one mutant is needed to validate a test".into(),
            });
        }
        if !(min_kill_rate > 0.0 && min_kill_rate <= 1.0) {
            return Err(VerifyError::Parameter {
                name: "min_kill_rate",
                reason: format!("{min_kill_rate} is not in (0, 1]"),
            });
        }
        Ok(Self {
            max_mutants,
            min_kill_rate,
        })
    }

    /// The most mutants a test is run against.
    #[must_use]
    pub fn max_mutants(&self) -> usize {
        self.max_mutants
    }

    /// The kill rate a test needs to be admitted.
    #[must_use]
    pub fn min_kill_rate(&self) -> f64 {
        self.min_kill_rate
    }
}

/// What validating one generated test found.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OracleValidation {
    /// Whether the test is admitted as an oracle.
    pub admitted: bool,
    /// Whether it passes on the reference solution. When it does not, no
    /// mutant is run and the counts are zero.
    pub passes_reference: bool,
    /// The mutants generated.
    pub mutants: usize,
    /// Mutants the test failed.
    pub killed: usize,
    /// Mutants the test passed.
    pub survived: usize,
    /// Mutants that did not run cleanly on their own, left out of the rate.
    pub errored: usize,
    /// killed / (killed + survived); absent when no mutant counted.
    pub kill_rate: Option<f64>,
}

/// Validates `test` against `reference` and its mutants in `runtime`.
pub fn validate_oracle(
    reference: &str,
    test: &ExecutableCheck,
    runtime: &RuntimeEnvironment,
    policy: &MutationPolicy,
) -> Result<OracleValidation, VerifyError> {
    let on_reference = run_check(runtime, assemble(reference, &test.code), test)?;
    let mut validation = OracleValidation {
        admitted: false,
        passes_reference: on_reference.passed(),
        mutants: 0,
        killed: 0,
        survived: 0,
        errored: 0,
        kill_rate: None,
    };
    if !validation.passes_reference {
        return Ok(validation);
    }
    for mutant in mutants(reference, policy.max_mutants) {
        validation.mutants += 1;
        let alone = runtime.run(&CodeCall {
            code: mutant.text.clone(),
            stdin: test.stdin.clone(),
        })?;
        if alone.timed_out || alone.exit_code != Some(0) {
            validation.errored += 1;
        } else if run_check(runtime, assemble(&mutant.text, &test.code), test)?.passed() {
            validation.survived += 1;
        } else {
            validation.killed += 1;
        }
    }
    let counted = validation.killed + validation.survived;
    let rate = (counted > 0).then(|| validation.killed as f64 / counted as f64);
    validation.kill_rate = rate;
    validation.admitted = rate.is_some_and(|r| r >= policy.min_kill_rate);
    Ok(validation)
}

/// Grades a solution with the task's generated tests ([`TEST_KIND`]) that
/// mutation admits against its one reference: pass iff it passes every
/// admitted test, fail otherwise or when there is no answer; abstains when
/// the task has no reference, no test is admitted, or a test's environment
/// was not offered.
#[derive(Clone, Debug)]
pub struct MutationValidatedVerifier {
    offered: Offered,
    policy: MutationPolicy,
}

impl MutationValidatedVerifier {
    /// A verifier that validates and runs tests in `environments` under
    /// `policy`.
    #[must_use]
    pub fn new(environments: Vec<RuntimeEnvironment>, policy: MutationPolicy) -> Self {
        Self {
            offered: Offered::new(environments),
            policy,
        }
    }
}

impl Verifier for MutationValidatedVerifier {
    fn producer(&self) -> Producer {
        Producer {
            name: PRODUCER.into(),
            version: VERSION.into(),
        }
    }

    fn strength(&self) -> Strength {
        Strength::Executable
    }

    /// The evidence records the reference's digest, the policy, and per
    /// test its digest, its validation, and (when admitted) the run.
    fn verify(&self, task: &Task, exp: &Experience) -> Result<Finding, VerifyError> {
        let reference = match single_reference(task) {
            Ok(reference) => program_text(reference),
            Err(abstain) => return Ok(abstain),
        };
        let tests = parse_checks(task, TEST_KIND)?;
        if tests.is_empty() {
            return Ok(Finding::abstain(
                "the task carries no generated tests",
                json!({}),
            ));
        }
        let solution = exp.final_output.as_deref().map(program_text);
        let mut reports = Vec::with_capacity(tests.len());
        let mut admitted = 0;
        let mut passed = solution.is_some();
        for (digest, test) in &tests {
            let environment = test.environment.as_ref().unwrap_or(&task.environment);
            let env = match self.offered.find(environment)? {
                Ok(env) => env,
                Err(abstain) => return Ok(abstain),
            };
            let validation = validate_oracle(reference, test, env, &self.policy)?;
            let mut run = serde_json::Value::Null;
            if validation.admitted {
                admitted += 1;
                if let Some(solution) = solution {
                    let program = assemble(solution, &test.code);
                    let result = run_check(env, program.clone(), test)?;
                    passed &= result.passed();
                    run = run_evidence(digest, &program, env, environment, &result);
                }
            }
            reports.push(json!({ "test": digest, "validation": validation, "run": run }));
        }
        let evidence = json!({
            "reference": Digest::of(reference.as_bytes()),
            "policy": self.policy,
            "output": exp.final_output.as_deref().map(|o| Digest::of(o.as_bytes())),
            "tests": reports,
        });
        if admitted == 0 {
            return Ok(Finding::abstain("no generated test was admitted", evidence));
        }
        Ok(Finding::decided(passed, evidence))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(source: &str, op: MutationOperator) -> Vec<String> {
        mutants(source, usize::MAX)
            .into_iter()
            .filter(|m| m.operator == op)
            .map(|m| m.text)
            .collect()
    }

    #[test]
    fn operators_rewrite_tokens_and_leave_lookalikes_alone() {
        assert_eq!(
            texts("a <= b -> c << d", MutationOperator::FlipComparison),
            ["a > b -> c << d"]
        );
        assert_eq!(
            texts("x1 = 10 + 1.5", MutationOperator::OffByOne),
            ["x1 = 11 + 1.5", "x1 = 9 + 1.5"]
        );
        assert_eq!(
            texts("ok = True or Trueish", MutationOperator::SwapBoolean),
            ["ok = False or Trueish"]
        );
        assert_eq!(
            texts("# note\nx = 1\n", MutationOperator::DeleteLine),
            ["# note\n\n"]
        );
    }
}
