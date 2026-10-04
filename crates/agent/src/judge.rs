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
//! with [`CalibratedJudge`](splinter_eval::verifiers::calibration::CalibratedJudge)
//! to make its verdicts stand only where it was measured precise.

use std::time::Duration;

use serde_json::json;
use splinter_core::annotation::{Outcome, Producer, Strength};
use splinter_core::digest::Digest;
use splinter_core::experience::{Environment, Experience, PrivilegedKind, Task};
use splinter_eval::verifiers::{Finding, Verifier, VerifyError};
use splinter_sandbox::ResolvedEnvironment;
use tokio::runtime::Handle;

use crate::solve::{solve, Model, SolveOptions};

/// The producer name of a judge's verdicts, before its model identity.
pub const PRODUCER: &str = "splinter-agent/judge";

/// The producer name of a fit judge's verdicts, before its model identity.
pub const FIT_PRODUCER: &str = "splinter-agent/judge-fit";

/// The judge's version: bumped whenever its prompt or reply parsing
/// changes.
pub const VERSION: &str = "3";

/// The task kind of the closed-book task a judge is asked.
pub const JUDGE_TASK_KIND: &str = "judge";

/// The most characters of a judge's reason kept in the evidence.
pub const REASON_CAP_CHARS: usize = 500;

/// What the judge is told to do, before the task, reference and answer.
const INSTRUCTIONS: &str = "You are grading an answer to a task. You see the task, reference \
material the solver never saw, and the solver's answer. The reference is what a correct answer \
rests on, often the passage the task was written from. Judge whether the answer gives what the \
reference says in reply to the task. PASS an answer that is consistent with the reference and \
draws on it: it may quote the reference or say it in other words, repeating the reference is not \
a fault when it answers the task, and saying more than the reference does is not a fault when it \
agrees with it. FAIL an answer that contradicts the reference, is about something else, dodges \
the question, or only says that it cannot tell: such an answer gives nothing the reference holds. \
Reply with one word on the first line - PASS, FAIL, or ABSTAIN when you cannot tell - and give \
your reason on the next line.";

/// What the judge is told when it judges fit: a message and a passage, and
/// nothing to compare the passage with.
const FIT_INSTRUCTIONS: &str = "You see a message someone sent to a writer and a passage the \
writer wrote. Judge whether the passage is a natural, direct reply the writer could have given \
to that message: it addresses what the message asks or describes, and needs nothing the message \
does not give. PASS such a passage. FAIL a passage that is about something else, answers a \
different question, or only makes sense with context the message lacks. Reply with one word on \
the first line - PASS, FAIL, or ABSTAIN when you cannot tell - and give your reason on the next \
line.";

/// What a judge is asked about an answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Judging {
    /// Whether the answer gives what the task's reference says.
    Reference,
    /// Whether the answer is a natural reply to the task: for a passage a
    /// writer wrote, put forward as the reply to a message written for it.
    Fit,
}

/// A model grading answers closed-book. See the module documentation.
pub struct JudgeVerifier {
    model: Model,
    runtime: Handle,
    deadline: Duration,
    judging: Judging,
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
            judging: Judging::Reference,
        }
    }

    /// A judge of fit on `model`: it reads the task's instruction and the
    /// answer as the writer's reply to it, never a reference, and says whether
    /// the answer is a natural reply. Its verdicts carry a producer of their
    /// own, so it is measured on its own.
    #[must_use]
    pub fn fit(model: Model, runtime: Handle, deadline: Duration) -> Self {
        Self {
            judging: Judging::Fit,
            ..Self::new(model, runtime, deadline)
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
fn prompt(task: &Task, answer: &str, judging: Judging) -> String {
    if judging == Judging::Fit {
        return format!(
            "{FIT_INSTRUCTIONS}\n\nMESSAGE:\n{}\n\nPASSAGE:\n{answer}\n",
            task.instruction
        );
    }
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
        let name = match self.judging {
            Judging::Reference => PRODUCER,
            Judging::Fit => FIT_PRODUCER,
        };
        Producer {
            name: format!("{name}:{}", self.model.identity),
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
        let prompt = prompt(task, answer, self.judging);
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
