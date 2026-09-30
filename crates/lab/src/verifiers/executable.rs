// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements executable verifiers that run agent-written
// code against checks it never saw, for its clients. If your team needs
// expertise in code verification or agent sandboxing, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The executable verifier: the solver's code run against the task's
//! checks, in the environment the task names.
//!
//! A check ([`ExecutableCheck`]) is code and what running it must produce:
//! an exit code, standard output compared under a [`Normalisation`], or
//! both. It travels with the task as a privileged item of kind
//! [`CHECK_KIND`], so the solver never sees it. The program run is the
//! solver's code ([`program_text`] of its final output) followed by the
//! check's code, in one file, so a check calls what the solver defined.
//!
//! A check runs in the environment it names, or else the task's own; the
//! verifier runs it only in an environment it was offered whose record is
//! exactly that one (the snapshot pins runtime, sandbox and limits), and
//! abstains otherwise. The evidence records what ran (the check's and the
//! program's digests, the environment snapshot, the runtime's version and
//! executable digest) and what came back (exit code, signal, timeout, and
//! digests of the possibly truncated output streams), never the outputs.

use serde::{Deserialize, Serialize};
use serde_json::json;
use splinter_sandbox::{
    CodeCall, CodeResult, ResolvedEnvironment, RuntimeEnvironment, SandboxError,
};
use splinter_store::annotation::{Producer, Strength};
use splinter_store::digest::Digest;
use splinter_store::experience::{Environment, Experience, Privileged, PrivilegedKind, Task};

use super::normalise::Normalisation;
use super::{privileged_of, Finding, Verifier, VerifyError};

/// The privileged kind an authored check travels as.
pub const CHECK_KIND: &str = "executable-check";

/// The producer name the executable verifier's annotations carry.
pub const PRODUCER: &str = "splinter-lab/executable";

/// The executable verifier's version.
pub const VERSION: &str = "1";

/// Code that checks a solution, and what running it must produce.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExecutableCheck {
    /// Appended to the solver's code; calls what the solver defined.
    pub code: String,
    /// What the program reads on standard input; absent closes it.
    #[serde(default)]
    pub stdin: Option<String>,
    /// What the run must produce.
    pub expect: Expectation,
    /// Where the check runs, when not in the task's own environment (a
    /// closed-book task's code answer, checked in a runtime).
    #[serde(default)]
    pub environment: Option<Environment>,
}

/// What a check's run must produce. At least one of the two is required: a
/// check that expects nothing would pass everything.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Expectation {
    /// The exit code, when it matters.
    #[serde(default)]
    pub exit_code: Option<i32>,
    /// Standard output, when it matters.
    #[serde(default)]
    pub stdout: Option<ExpectedStdout>,
}

/// Expected standard output and how it is compared.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpectedStdout {
    /// The expected text.
    pub text: String,
    /// What differences the comparison ignores.
    #[serde(default)]
    pub normalisation: Normalisation,
}

impl Expectation {
    /// Exit with `code`, whatever the output.
    #[must_use]
    pub fn exit_code(code: i32) -> Self {
        Self {
            exit_code: Some(code),
            stdout: None,
        }
    }

    /// Why a run's `result` does not meet this expectation; empty when it
    /// does. A run a limit stopped never does, nor does output cut at the
    /// cap when output is compared.
    fn unmet(&self, result: &CodeResult) -> Vec<&'static str> {
        let mut unmet = Vec::new();
        if result.timed_out {
            unmet.push("timed_out");
        }
        if self.exit_code.is_some() && result.exit_code != self.exit_code {
            unmet.push("exit_code");
        }
        if let Some(expected) = &self.stdout {
            let norm = expected.normalisation;
            if result.stdout_truncated || norm.apply(&result.stdout) != norm.apply(&expected.text) {
                unmet.push("stdout");
            }
        }
        unmet
    }
}

impl ExecutableCheck {
    /// This check as an authored check a task carries.
    pub fn as_check(&self) -> Result<Privileged, serde_json::Error> {
        self.as_privileged(CHECK_KIND)
    }

    /// This check as a generated test, admitted only after mutation
    /// validation (see [`super::mutation`]).
    pub fn as_generated_test(&self) -> Result<Privileged, serde_json::Error> {
        self.as_privileged(super::mutation::TEST_KIND)
    }

    fn as_privileged(&self, kind: &str) -> Result<Privileged, serde_json::Error> {
        Ok(Privileged {
            kind: PrivilegedKind::Other(kind.to_string()),
            content: serde_json::to_string(self)?,
            span: None,
        })
    }
}

/// The code in a solver's answer: the body of its first fenced block when
/// it has one, the whole answer otherwise.
#[must_use]
pub fn program_text(output: &str) -> &str {
    let Some(start) = output.find("```") else {
        return output;
    };
    let after = &output[start + 3..];
    let Some(newline) = after.find('\n') else {
        return output;
    };
    let body = &after[newline + 1..];
    body.find("```").map_or(body, |end| &body[..end])
}

/// `solution` followed by `check_code`, on a line of its own.
pub(crate) fn assemble(solution: &str, check_code: &str) -> String {
    let mut program = solution.to_string();
    if !program.is_empty() && !program.ends_with('\n') {
        program.push('\n');
    }
    program.push_str(check_code);
    program
}

/// A task's checks of one privileged kind, parsed, each with the digest of
/// the item it was parsed from.
pub(crate) fn parse_checks(
    task: &Task,
    kind: &str,
) -> Result<Vec<(Digest, ExecutableCheck)>, VerifyError> {
    let kind = PrivilegedKind::Other(kind.to_string());
    privileged_of(task, &kind)
        .map(|(index, content)| {
            let malformed = |reason: String| VerifyError::Malformed {
                kind: kind.clone(),
                index,
                reason,
            };
            let check: ExecutableCheck =
                serde_json::from_str(content).map_err(|e| malformed(e.to_string()))?;
            if check.expect.exit_code.is_none() && check.expect.stdout.is_none() {
                return Err(malformed(
                    "it expects neither an exit code nor output".into(),
                ));
            }
            Ok((Digest::of(content.as_bytes()), check))
        })
        .collect()
}

/// The environments a verifier may run code in.
#[derive(Clone, Debug)]
pub(crate) struct Offered(Vec<RuntimeEnvironment>);

impl Offered {
    pub(crate) fn new(environments: Vec<RuntimeEnvironment>) -> Self {
        Self(environments)
    }

    /// The offered environment whose record is exactly `wanted`, or the
    /// abstention saying why there is none.
    pub(crate) fn find(
        &self,
        wanted: &Environment,
    ) -> Result<Result<&RuntimeEnvironment, Finding>, SandboxError> {
        if wanted.kind == Environment::CLOSED_BOOK {
            return Ok(Err(Finding::abstain(
                "a closed-book environment runs no code",
                json!({ "environment": wanted.snapshot }),
            )));
        }
        for env in &self.0 {
            if ResolvedEnvironment::Runtime(env.clone()).record()? == *wanted {
                return Ok(Ok(env));
            }
        }
        Ok(Err(Finding::abstain(
            "the environment the check names was not offered",
            json!({ "kind": wanted.kind, "environment": wanted.snapshot }),
        )))
    }
}

/// One check run.
pub(crate) struct CheckRun {
    /// How the run ended.
    pub(crate) result: CodeResult,
    /// Which expectations it did not meet.
    pub(crate) unmet: Vec<&'static str>,
}

impl CheckRun {
    pub(crate) fn passed(&self) -> bool {
        self.unmet.is_empty()
    }
}

/// Runs `program` as `check` says in `env`.
pub(crate) fn run_check(
    env: &RuntimeEnvironment,
    program: String,
    check: &ExecutableCheck,
) -> Result<CheckRun, SandboxError> {
    let result = env.run(&CodeCall {
        code: program,
        stdin: check.stdin.clone(),
    })?;
    let unmet = check.expect.unmet(&result);
    Ok(CheckRun { result, unmet })
}

/// What a check run's evidence records: digests of what ran and of what
/// came back.
pub(crate) fn run_evidence(
    check: &Digest,
    program: &str,
    env: &RuntimeEnvironment,
    environment: &Environment,
    run: &CheckRun,
) -> serde_json::Value {
    let runtime = env.runtime();
    json!({
        "check": check,
        "program": Digest::of(program.as_bytes()),
        "environment": environment.snapshot,
        "runtime": {
            "name": runtime.spec.name,
            "version": runtime.version,
            "executable_digest": runtime.executable_digest,
        },
        "exit_code": run.result.exit_code,
        "signal": run.result.signal,
        "timed_out": run.result.timed_out,
        "stdout": Digest::of(run.result.stdout.as_bytes()),
        "stdout_truncated": run.result.stdout_truncated,
        "stderr": Digest::of(run.result.stderr.as_bytes()),
        "stderr_truncated": run.result.stderr_truncated,
        "passed": run.passed(),
        "unmet": run.unmet,
    })
}

/// Runs a task's authored checks ([`CHECK_KIND`]) against the solver's
/// code: pass iff every check meets its expectation, fail otherwise or
/// when there is no answer; abstains on a task with no checks, or whose
/// checks name an environment it was not offered.
#[derive(Clone, Debug)]
pub struct ExecutableVerifier {
    offered: Offered,
}

impl ExecutableVerifier {
    /// A verifier that runs checks in `environments` and nowhere else.
    #[must_use]
    pub fn new(environments: Vec<RuntimeEnvironment>) -> Self {
        Self {
            offered: Offered::new(environments),
        }
    }
}

impl Verifier for ExecutableVerifier {
    fn producer(&self) -> Producer {
        Producer {
            name: PRODUCER.into(),
            version: VERSION.into(),
        }
    }

    fn strength(&self) -> Strength {
        Strength::Executable
    }

    fn verify(&self, task: &Task, exp: &Experience) -> Result<Finding, VerifyError> {
        let checks = parse_checks(task, CHECK_KIND)?;
        if checks.is_empty() {
            return Ok(Finding::abstain("the task carries no checks", json!({})));
        }
        let mut placed = Vec::with_capacity(checks.len());
        for (digest, check) in &checks {
            let environment = check.environment.as_ref().unwrap_or(&task.environment);
            match self.offered.find(environment)? {
                Ok(env) => placed.push((digest, check, env, environment)),
                Err(abstain) => return Ok(abstain),
            }
        }
        let Some(output) = exp.final_output.as_deref() else {
            return Ok(Finding::decided(
                false,
                json!({ "output": null, "checks": [] }),
            ));
        };
        let solution = program_text(output);
        let mut runs = Vec::with_capacity(placed.len());
        let mut passed = true;
        for (digest, check, env, environment) in placed {
            let program = assemble(solution, &check.code);
            let run = run_check(env, program.clone(), check)?;
            passed &= run.passed();
            runs.push(run_evidence(digest, &program, env, environment, &run));
        }
        Ok(Finding::decided(
            passed,
            json!({ "output": Digest::of(output.as_bytes()), "checks": runs }),
        ))
    }
}
