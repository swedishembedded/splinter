// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model judges that grade agent answers
// without ever grading their own, for its clients. If your team needs
// expertise in LLM-as-judge evaluation or reward modelling, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The judged verifier: a different model grading an answer, closed-book,
//! through the same sven solve every task runs through.
//!
//! The judge is shown the task's instruction, its privileged reference and
//! oracle material, and the solver's final output - never the solver's
//! trajectory, so nothing the solver said about its own success reaches
//! it. It answers with `PASS`, `FAIL` or `ABSTAIN` as the first word and a
//! reason after it; a reply that does not start that way is an abstention.
//! A judge refuses to grade an experience its own model produced. Gate it
//! with [`CalibratedJudge`](splinter_lab::verifiers::calibration::CalibratedJudge)
//! to make its verdicts stand only where it was measured precise.

use std::time::Duration;

use serde_json::json;
use splinter_lab::verifiers::{Finding, Verifier, VerifyError};
use splinter_sandbox::ResolvedEnvironment;
use splinter_store::annotation::{Outcome, Producer, Strength};
use splinter_store::digest::Digest;
use splinter_store::experience::{Environment, Experience, PrivilegedKind, Task};
use tokio::runtime::Handle;

use crate::solve::{solve, Model, SolveOptions};

/// The producer name of a judge's verdicts, before its model identity.
pub const PRODUCER: &str = "splinter-agent/judge";

/// The judge's version: bumped whenever its prompt or reply parsing
/// changes.
pub const VERSION: &str = "1";

/// The task kind of the closed-book task a judge is asked.
pub const JUDGE_TASK_KIND: &str = "judge";

/// The most characters of a judge's reason kept in the evidence.
pub const REASON_CAP_CHARS: usize = 500;

/// What the judge is told to do, before the task, reference and answer.
const INSTRUCTIONS: &str = "You are grading an answer to a task. You see the task, reference \
material the solver never saw, and the solver's answer. Judge only whether the answer is \
correct. Reply with one word on the first line - PASS, FAIL, or ABSTAIN when you cannot \
tell - and give your reason on the next line.";

/// A model grading answers closed-book. See the module documentation.
pub struct JudgeVerifier {
    model: Model,
    runtime: Handle,
    deadline: Duration,
}

impl JudgeVerifier {
    /// A judge on `model`, known by its identity (the same string an
    /// experience's provenance records for that model), running its solves
    /// on `runtime` within `deadline` each. [`Verifier::verify`] blocks on
    /// `runtime`: call it outside that runtime's async tasks.
    #[must_use]
    pub fn new(model: Model, runtime: Handle, deadline: Duration) -> Self {
        Self {
            model,
            runtime,
            deadline,
        }
    }

    /// The judge's model identity.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.model.identity
    }
}

/// The prompt the judge is asked: the instruction, the task, its reference
/// and oracle material, and the answer.
fn prompt(task: &Task, answer: &str) -> String {
    let references: Vec<&str> = task
        .privileged
        .iter()
        .filter(|p| matches!(p.kind, PrivilegedKind::Reference | PrivilegedKind::Oracle))
        .map(|p| p.content.as_str())
        .collect();
    let reference = if references.is_empty() {
        "(none)".to_string()
    } else {
        references.join("\n\n")
    };
    format!(
        "{INSTRUCTIONS}\n\nTASK:\n{}\n\nREFERENCE:\n{reference}\n\nANSWER:\n{answer}\n",
        task.instruction
    )
}

/// The verdict and reason in a judge's reply; `None` when its first word
/// is not a verdict.
fn parse_reply(reply: &str) -> Option<(Outcome, String)> {
    let reply = reply.trim_start();
    let (word, rest) = reply.split_once(char::is_whitespace).unwrap_or((reply, ""));
    let outcome = match word
        .trim_matches(|c: char| !c.is_ascii_alphabetic())
        .to_ascii_uppercase()
        .as_str()
    {
        "PASS" => Outcome::Pass,
        "FAIL" => Outcome::Fail,
        "ABSTAIN" => Outcome::Abstain,
        _ => return None,
    };
    Some((
        outcome,
        rest.trim().chars().take(REASON_CAP_CHARS).collect(),
    ))
}

impl Verifier for JudgeVerifier {
    fn producer(&self) -> Producer {
        Producer {
            name: format!("{PRODUCER}:{}", self.model.identity),
            version: VERSION.into(),
        }
    }

    fn strength(&self) -> Strength {
        Strength::Judged
    }

    /// Refuses an experience whose solver or policy is the judge's own
    /// model. The evidence records the judge's identity, the digests of the
    /// prompt and the reply, the (capped) reason and how the run ended.
    fn verify(&self, task: &Task, exp: &Experience) -> Result<Finding, VerifyError> {
        // `block_on` panics on a runtime thread; refuse instead.
        if tokio::runtime::Handle::try_current().is_ok() {
            return Err(VerifyError::InsideAsync {
                verifier: self.producer().name,
            });
        }
        let provenance = &exp.provenance;
        if provenance.solver == self.model.identity
            || provenance.policy.as_deref() == Some(&self.model.identity)
        {
            return Err(VerifyError::SelfJudging {
                identity: self.model.identity.clone(),
            });
        }
        let Some(answer) = exp.final_output.as_deref() else {
            return Ok(Finding::decided(
                false,
                json!({ "judge": self.model.identity, "output": null }),
            ));
        };
        let prompt = prompt(task, answer);
        let question = Task::new(
            JUDGE_TASK_KIND,
            vec![],
            Environment::closed_book(),
            prompt.as_str(),
            vec![],
        )?;
        let mut options = SolveOptions::new(self.deadline);
        options.stream_idle = self.model.stream_idle;
        let solution = self
            .runtime
            .block_on(solve(
                &question,
                &ResolvedEnvironment::ClosedBook,
                self.model.provider.clone(),
                options,
            ))
            .map_err(|e| VerifyError::Failed {
                producer: self.producer().name,
                source: Box::new(e),
            })?;
        let mut evidence = json!({
            "judge": self.model.identity,
            "prompt": Digest::of(prompt.as_bytes()),
            "conclusion": format!("{:?}", solution.conclusion),
        });
        let Some(reply) = solution.final_output else {
            return Ok(Finding::abstain("the judge gave no reply", evidence));
        };
        evidence["reply"] = json!(Digest::of(reply.as_bytes()));
        let Some((outcome, reason)) = parse_reply(&reply) else {
            return Ok(Finding::abstain(
                "the judge's reply is not a verdict",
                evidence,
            ));
        };
        evidence["reason"] = json!(reason);
        Ok(Finding { outcome, evidence })
    }
}
