// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements critique-and-retry loops that turn failed
// agent attempts into verified revisions, for its clients. If your team
// needs expertise in agent self-improvement or learning from feedback, you
// can procure our services by sending an email to info@swedishembedded.com.

//! Critique and retry: a failed attempt is critiqued, the task is solved
//! again with the critique, and the chain is recorded as relations the
//! views read.
//!
//! [`Repair::retry`] solves the task again in its own environment, the
//! solver prompted with the instruction followed by the critique under
//! [`FEEDBACK_HEADING`]. The revision is an experience of the task with the
//! critique added as a privileged item of kind critique: its instruction
//! stays the task's, so a student view shows the critique only when its
//! strip policy keeps critiques. It is graded by the same verifiers as the
//! task, and then:
//!
//! * the revision records `RetryOf` the failed attempt. A retry is a fresh
//!   solve of the task, not an edit of the failed answer, so `RevisionOf`
//!   (reserved for an attempt that starts from the other's answer) is not
//!   recorded beside it - the decision view reads both kinds, and would
//!   derive the same record twice;
//! * the critique receives its verdict by outcome (see
//!   `splinter_lab::verifiers::critique`): pass when the retry passes where
//!   the failed attempt failed, fail when it still fails, an abstention when
//!   the retry did not finish or reached no decision;
//! * when the revision passes and the failed attempt failed at the same
//!   strength, the revision records `PreferredOver` the failed attempt.
//!
//! [`Repair::repair_loop`] repeats critique and retry, each round
//! critiquing the latest failure, until a revision passes or the
//! [`RepairBudget`] is spent: a number of retries, a wall-clock deadline
//! across the whole loop, and optionally output tokens across every run
//! (the critic's and the solver's), each run handed what is left through
//! sven's run options. Tokens a provider does not report cannot be counted
//! against the budget. The loop never retries an attempt that is not
//! decided fail, and stops at the first environment, runtime or store
//! error - a run that could not happen - while a wrong answer is only a
//! failed round.
//!
//! Everything here blocks on the runtime handle it is given, as the judge
//! does: call it outside that runtime's async tasks.

use std::time::{Duration, Instant};

use splinter_lab::verifiers::critique::{critique_verdict, preferred, RetryOutcome};
use splinter_lab::verifiers::{verify_and_annotate, Strongest, VerifyError};
use splinter_sandbox::ResolvedEnvironment;
use splinter_store::annotation::{
    decide, Annotation, AnnotationBody, Decision, Outcome, Producer, RelationKind,
};
use splinter_store::clock::Clock;
use splinter_store::digest::Digest;
use splinter_store::experience::{
    ExperienceError, ExperienceId, Privileged, PrivilegedKind, Provenance, Task,
};
use splinter_store::experiences::{ExperienceStore, StoreError};
use sven_sdk::{CancelToken, RunConclusion, Usage};
use tokio::runtime::Handle;

use crate::critic::{refuse_inside_async, Critic, Critique};
use crate::solve::{solve_prompted, Model, SolveError, SolveOptions};

/// The producer name of the relations a retry records.
pub const PRODUCER: &str = "splinter-agent/repair";

/// The version of the retry: bumped whenever its prompt or what it records
/// changes.
pub const VERSION: &str = "1";

/// Introduces the critique after the instruction in a retry's prompt.
pub const FEEDBACK_HEADING: &str = "Feedback on an earlier attempt at this task:";

/// Why a critique, a retry or a repair could not be carried out.
#[derive(Debug, thiserror::Error)]
pub enum RepairError {
    /// The experience is not decided fail, so there is nothing to critique
    /// or retry.
    #[error("{experience} is not decided fail (its decision: {decision:?})")]
    NotFailed {
        /// The experience.
        experience: ExperienceId,
        /// Its decision, when it has one.
        decision: Option<Decision>,
    },
    /// The attempt is not an attempt of the task it was retried as.
    #[error("{experience} attempted task {attempted}, not task {task}")]
    TaskMismatch {
        /// The attempt.
        experience: ExperienceId,
        /// The task it attempted.
        attempted: Digest,
        /// The task it was retried as.
        task: Digest,
    },
    /// The critique critiques another experience than the one retried.
    #[error("the critique {critique} critiques {critiqued}, not {failed}")]
    CritiqueMismatch {
        /// The critique.
        critique: ExperienceId,
        /// The experience it critiques.
        critiqued: ExperienceId,
        /// The experience retried.
        failed: ExperienceId,
    },
    /// Called from inside an async task, where blocking on a model run would
    /// stall (or panic) the runtime.
    #[error("a critique or retry blocks on a model run and was called from inside an async task")]
    InsideAsync,
    /// The critic or the solver could not run.
    #[error(transparent)]
    Solve(#[from] SolveError),
    /// The verifiers could not grade the revision at all.
    #[error(transparent)]
    Verify(#[from] VerifyError),
    /// The experience store failed.
    #[error(transparent)]
    Store(#[from] StoreError),
    /// A task or experience could not be built.
    #[error(transparent)]
    Experience(#[from] ExperienceError),
}

/// What a retry produced.
#[derive(Debug)]
pub struct Retried {
    /// The revision's experience.
    pub revision: ExperienceId,
    /// How the solver's run ended.
    pub conclusion: RunConclusion,
    /// The tokens the solver's provider reported.
    pub usage: Usage,
    /// The decision the revision's verdicts add up to.
    pub decision: Option<Decision>,
    /// The verdict the critique received by this retry's outcome.
    pub critique_verdict: Outcome,
    /// Whether the revision was recorded as preferred over the failed
    /// attempt.
    pub preferred: bool,
    /// The verifiers that could not grade the revision, with why.
    pub verifier_errors: Vec<(Producer, VerifyError)>,
}

/// One round of a repair: a critique, and the retry it led to.
#[derive(Debug)]
pub struct Round {
    /// The critique written this round.
    pub critique: Critique,
    /// Its retry; `None` when the loop stopped before the retry ran.
    pub retried: Option<Retried>,
}

/// Why a repair loop stopped.
#[derive(Debug)]
pub enum Stop {
    /// The last round's revision passed.
    Passed,
    /// The first attempt is not decided fail (a pass, or undecided), so
    /// nothing was critiqued or retried.
    NotFailed(Option<Decision>),
    /// A revision reached no decision, so there is no failure to critique.
    RevisionUndecided,
    /// A verifier could not grade the last revision (see its round).
    Unverified,
    /// The retry budget is spent.
    Retries,
    /// The deadline passed.
    Deadline,
    /// The output-token budget is spent.
    Tokens,
    /// The budget's cancel token was cancelled.
    Cancelled,
    /// The critic's or the solver's run ended some other way (an agent
    /// error).
    RunStopped(RunConclusion),
    /// A run could not happen, or the store failed.
    Error(RepairError),
}

/// The bounds of a repair loop. See the module documentation.
#[derive(Clone, Debug)]
pub struct RepairBudget {
    /// Retries at most; each is one critique and one retry.
    pub max_retries: usize,
    /// Wall-clock time the whole loop may take.
    pub deadline: Duration,
    /// Output tokens every run together may generate; `None` sets no
    /// budget.
    pub max_output_tokens: Option<u64>,
    /// Stops the loop, and the run in progress, from outside.
    pub cancel: Option<CancelToken>,
}

impl RepairBudget {
    /// At most `max_retries` retries within `deadline`, with no token
    /// budget and no cancel token.
    #[must_use]
    pub fn new(max_retries: usize, deadline: Duration) -> Self {
        Self {
            max_retries,
            deadline,
            max_output_tokens: None,
            cancel: None,
        }
    }
}

/// What a repair loop did.
#[derive(Debug)]
pub struct RepairReport {
    /// The attempt it started from.
    pub first: ExperienceId,
    /// Its rounds, in order.
    pub rounds: Vec<Round>,
    /// Why it stopped.
    pub stop: Stop,
    /// Output tokens the providers reported across every run; `None` when
    /// none reported any.
    pub output_tokens: Option<u64>,
}

impl RepairReport {
    /// The chain as experience ids, in the order they were made: the first
    /// attempt, then each round's critique and revision.
    #[must_use]
    pub fn chain(&self) -> Vec<ExperienceId> {
        let mut chain = vec![self.first.clone()];
        for round in &self.rounds {
            chain.push(round.critique.id.clone());
            if let Some(retried) = &round.retried {
                chain.push(retried.revision.clone());
            }
        }
        chain
    }
}

/// Everything a critique-and-retry works with.
pub struct Repair<'a> {
    /// Where experiences and their annotations are recorded.
    pub store: &'a ExperienceStore,
    /// Stamps every experience recorded.
    pub clock: &'a dyn Clock,
    /// The runtime the critic's and the solver's runs block on.
    pub runtime: Handle,
    /// The environment the task is solved in; a retry refuses any other
    /// than the task's own.
    pub environment: &'a ResolvedEnvironment,
    /// The verifiers the task is graded by.
    pub verifiers: &'a Strongest,
    /// The model that retries.
    pub solver: Model,
    /// The model that critiques.
    pub critic: Critic,
}

impl Repair<'_> {
    /// Solves `task` again with `critique` of `failed`, an attempt of the
    /// task decided fail, within `options`, and records the revision, its
    /// verdicts, its relations and the critique's verdict. See the module
    /// documentation.
    pub fn retry(
        &self,
        task: &Task,
        failed: &ExperienceId,
        critique: &Critique,
        options: SolveOptions,
    ) -> Result<Retried, RepairError> {
        refuse_inside_async()?;
        if critique.critiqued != *failed {
            return Err(RepairError::CritiqueMismatch {
                critique: critique.id.clone(),
                critiqued: critique.critiqued.clone(),
                failed: failed.clone(),
            });
        }
        let failed_decision = self.failed_decision(task, failed)?;
        let revision_task = task.with_privileged(Privileged {
            kind: PrivilegedKind::Critique,
            content: critique.text.clone(),
            span: None,
        })?;
        let prompt = feedback_prompt(&task.instruction, &critique.text);
        let options = SolveOptions {
            stream_idle: self.solver.stream_idle,
            ..options
        };
        let solution = self.runtime.block_on(solve_prompted(
            &revision_task,
            &prompt,
            self.environment,
            self.solver.provider.clone(),
            options,
        ))?;
        let (conclusion, usage) = (solution.conclusion, solution.usage);
        let revision = solution.into_experience(
            revision_task.clone(),
            Provenance::new(self.solver.identity.clone(), self.clock),
        )?;
        let id = self.store.put(&revision)?;
        self.relate(&id, RelationKind::RetryOf, failed)?;
        let verification =
            verify_and_annotate(self.store, self.verifiers, &revision_task, &revision)?;
        let decision = decide(&self.store.annotations(&id)?.annotations);
        let verdict = critique_verdict(
            failed,
            Some(failed_decision),
            &id,
            RetryOutcome {
                finished: conclusion == RunConclusion::Success,
                decision,
            },
        );
        self.store.annotate(&verdict.annotation(&critique.id))?;
        let preferred = preferred(decision, Some(failed_decision));
        if preferred {
            self.relate(&id, RelationKind::PreferredOver, failed)?;
        }
        Ok(Retried {
            revision: id,
            conclusion,
            usage,
            decision,
            critique_verdict: verdict.outcome,
            preferred,
            verifier_errors: verification.errors,
        })
    }

    /// Critiques and retries `first_attempt`, an attempt of `task`, until a
    /// revision passes or `budget` is spent. Errs only when the loop cannot
    /// start (an unknown attempt, one of another task, or a call from inside
    /// an async task); a failure once it runs stops it, and the report says
    /// why, with every round recorded until then.
    pub fn repair_loop(
        &self,
        task: &Task,
        first_attempt: &ExperienceId,
        budget: &RepairBudget,
    ) -> Result<RepairReport, RepairError> {
        refuse_inside_async()?;
        let first = self.store.get(first_attempt)?;
        check_task(task, first_attempt, &first.to_task())?;
        let mut rounds = Vec::new();
        let mut meter = Meter::new(budget);
        let decision = decide(&self.store.annotations(first_attempt)?.annotations);
        let stop = if decision.is_none_or(|d| d.passed) {
            Stop::NotFailed(decision)
        } else {
            self.rounds(task, first_attempt, &mut meter, &mut rounds)
        };
        Ok(RepairReport {
            first: first_attempt.clone(),
            rounds,
            stop,
            output_tokens: meter.spent,
        })
    }

    /// Runs rounds from `failed` into `rounds` until one stops the loop,
    /// and returns why.
    fn rounds(
        &self,
        task: &Task,
        failed: &ExperienceId,
        meter: &mut Meter<'_>,
        rounds: &mut Vec<Round>,
    ) -> Stop {
        let mut current = failed.clone();
        loop {
            if rounds.len() >= meter.budget.max_retries {
                return Stop::Retries;
            }
            let options = match meter.options() {
                Ok(options) => options,
                Err(stop) => return stop,
            };
            let run =
                match self
                    .critic
                    .critique(self.store, self.clock, &self.runtime, &current, options)
                {
                    Ok(run) => run,
                    Err(e) => return Stop::Error(e),
                };
            meter.spend(run.usage);
            let Some(critique) = run.critique else {
                return stopped(run.conclusion);
            };
            let options = match meter.options() {
                Ok(options) => options,
                Err(stop) => {
                    rounds.push(Round {
                        critique,
                        retried: None,
                    });
                    return stop;
                }
            };
            let retry = self.retry(task, &current, &critique, options);
            let retried = match retry {
                Ok(retried) => retried,
                Err(e) => {
                    rounds.push(Round {
                        critique,
                        retried: None,
                    });
                    return Stop::Error(e);
                }
            };
            meter.spend(retried.usage);
            let next = if !retried.verifier_errors.is_empty() {
                Err(Stop::Unverified)
            } else if retried.conclusion != RunConclusion::Success {
                Err(stopped(retried.conclusion))
            } else {
                match retried.decision {
                    Some(d) if d.passed => Err(Stop::Passed),
                    Some(_) => Ok(retried.revision.clone()),
                    None => Err(Stop::RevisionUndecided),
                }
            };
            rounds.push(Round {
                critique,
                retried: Some(retried),
            });
            match next {
                Ok(revision) => current = revision,
                Err(stop) => return stop,
            }
        }
    }

    /// The decision on `failed`, an attempt of `task`, which must be fail.
    fn failed_decision(&self, task: &Task, failed: &ExperienceId) -> Result<Decision, RepairError> {
        let experience = self.store.get(failed)?;
        check_task(task, failed, &experience.to_task())?;
        let decision = decide(&self.store.annotations(failed)?.annotations);
        match decision {
            Some(d) if !d.passed => Ok(d),
            _ => Err(RepairError::NotFailed {
                experience: failed.clone(),
                decision,
            }),
        }
    }

    fn relate(
        &self,
        from: &ExperienceId,
        kind: RelationKind,
        to: &ExperienceId,
    ) -> Result<(), RepairError> {
        self.store.annotate(&Annotation {
            experience: from.clone(),
            producer: Producer {
                name: PRODUCER.into(),
                version: VERSION.into(),
            },
            body: AnnotationBody::Relation {
                kind,
                other: to.clone(),
            },
        })?;
        Ok(())
    }
}

/// The prompt a retry's solver is sent: the instruction, then the critique
/// under [`FEEDBACK_HEADING`].
#[must_use]
pub fn feedback_prompt(instruction: &str, critique: &str) -> String {
    format!("{instruction}\n\n{FEEDBACK_HEADING}\n{critique}")
}

/// Refuses `experience` unless it attempted `task`, critiques aside.
fn check_task(task: &Task, experience: &ExperienceId, attempted: &Task) -> Result<(), RepairError> {
    if attempted.same_apart_from_critiques(task) {
        return Ok(());
    }
    Err(RepairError::TaskMismatch {
        experience: experience.clone(),
        attempted: attempted.task.id.clone(),
        task: task.task.id.clone(),
    })
}

/// The stop a run that did not conclude successfully means.
fn stopped(conclusion: RunConclusion) -> Stop {
    match conclusion {
        RunConclusion::Timeout => Stop::Deadline,
        RunConclusion::BudgetExhausted => Stop::Tokens,
        RunConclusion::Cancelled => Stop::Cancelled,
        other => Stop::RunStopped(other),
    }
}

/// What is left of a [`RepairBudget`]'s time and tokens.
struct Meter<'b> {
    budget: &'b RepairBudget,
    started: Instant,
    spent: Option<u64>,
}

impl<'b> Meter<'b> {
    fn new(budget: &'b RepairBudget) -> Self {
        Self {
            budget,
            started: Instant::now(),
            spent: None,
        }
    }

    /// Counts the output tokens `usage` reports.
    fn spend(&mut self, usage: Usage) {
        if let Some(tokens) = usage.output_tokens {
            self.spent = Some(self.spent.unwrap_or(0).saturating_add(tokens));
        }
    }

    /// The bounds of the next run, or why there is none.
    fn options(&self) -> Result<SolveOptions, Stop> {
        if self
            .budget
            .cancel
            .as_ref()
            .is_some_and(CancelToken::is_cancelled)
        {
            return Err(Stop::Cancelled);
        }
        let deadline = self
            .budget
            .deadline
            .checked_sub(self.started.elapsed())
            .filter(|left| !left.is_zero())
            .ok_or(Stop::Deadline)?;
        let max_output_tokens = match self.budget.max_output_tokens {
            Some(limit) => {
                let left = limit.saturating_sub(self.spent.unwrap_or(0));
                if left == 0 {
                    return Err(Stop::Tokens);
                }
                Some(left)
            }
            None => None,
        };
        Ok(SolveOptions {
            deadline,
            max_output_tokens,
            cancel: self.budget.cancel.clone(),
            stream_idle: None,
        })
    }
}
