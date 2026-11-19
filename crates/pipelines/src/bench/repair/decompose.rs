// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that learn to finish tasks they
// failed by working them in checked steps, for its clients. If your team
// needs expertise in task decomposition for agent training then you can
// procure our services by sending an email to info@swedishembedded.com.

//! The decomposition: a proposer splits the failed task into ordered
//! subgoals with checkable intermediate states, and the student works them
//! one by one from the last good step.
//!
//! The proposer (the teacher, or the student's own model: [`Proposer`]) is
//! asked once, with the task's instruction, the diagnosis and the steps the
//! student kept, for at most [`MAX_SUBGOALS`] subgoals, each with a check of
//! a state the task allows to be checked on the way (a file written, a
//! partial answer in it) or none. The student's model then continues the
//! fork led by one user turn per subgoal; after each turn the subgoal's
//! check is read in the session's workspace, and a missed check ends the
//! run there. The task's own verifier grades the end state as for every
//! intervention: the checks guide, only the verifier accepts.
//!
//! The record and the lineage carry the subgoals, who proposed them and how
//! each ended. An accepted decomposition is the subgoal-guided trajectory:
//! the student's kept prefix (never supervised), the subgoal turns (context,
//! marked as guidance) and the student's own steps, which a supervised view
//! trains on.

use std::path::Path;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use splinter_agent::fork_at::POLICY;
use splinter_agent::schemars::JsonSchema;
use splinter_agent::solve::{Model, SolveOptions};
use splinter_agent::typed::TypedCall;
use splinter_agent::CancelToken;
use splinter_core::experience::Provenance;
use splinter_orchestrator::context::Context;

use super::super::practice::BenchCheckout;
use super::fork::{finish, fork_and_grade, Continuation};
use super::{Attempted, Case, Intervention, InterventionKind, Notes};

/// The most subgoals one decomposition may have.
pub const MAX_SUBGOALS: usize = 6;

/// How long the proposer may take.
const PROPOSE_DEADLINE: Duration = Duration::from_secs(600);

/// The most tokens the proposal may run to, its reasoning included.
const PROPOSE_MAX_OUTPUT_TOKENS: u64 = 8192;

/// The longest excerpt of a kept action the proposer is shown.
const ACTION_EXCERPT: usize = 240;

/// Who proposes a decomposition's subgoals.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Proposer {
    /// The teacher.
    #[default]
    Teacher,
    /// The student's own model.
    Student,
}

impl std::str::FromStr for Proposer {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, String> {
        match text {
            "teacher" => Ok(Self::Teacher),
            "student" => Ok(Self::Student),
            other => Err(format!("{other:?} is no proposer; teacher or student")),
        }
    }
}

/// A state of the workspace a subgoal leaves, checked after its turn.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[schemars(crate = "splinter_agent::schemars")]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Check {
    /// The file exists.
    FileExists {
        /// The file, relative to the workspace.
        path: String,
    },
    /// The file exists and holds the text.
    FileContains {
        /// The file, relative to the workspace.
        path: String,
        /// The text it holds.
        text: String,
    },
}

impl Check {
    fn path(&self) -> &str {
        match self {
            Self::FileExists { path } | Self::FileContains { path, .. } => path,
        }
    }

    /// Whether the workspace at `tree` is in the state.
    fn holds(&self, tree: &Path) -> bool {
        let file = tree.join(self.path());
        match self {
            Self::FileExists { .. } => file.is_file(),
            Self::FileContains { text, .. } => {
                std::fs::read_to_string(&file).is_ok_and(|content| content.contains(text.as_str()))
            }
        }
    }
}

/// One subgoal.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[schemars(crate = "splinter_agent::schemars")]
pub struct Subgoal {
    /// What to do, as the student is told.
    pub goal: String,
    /// The state it leaves; `None` where the task allows no check.
    #[serde(default)]
    pub check: Option<Check>,
}

/// The proposer's reply.
#[derive(Clone, Debug, Deserialize, JsonSchema)]
#[schemars(crate = "splinter_agent::schemars")]
struct Plan {
    /// The subgoals, in order.
    subgoals: Vec<Subgoal>,
}

/// How a subgoal ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubgoalState {
    /// Its check held after its turn.
    Reached,
    /// Its check did not hold: the run stopped there.
    Missed,
    /// It was worked and has no check.
    Unchecked,
    /// It was not reached: an earlier one was missed, or the run ended.
    NotRun,
}

/// One subgoal's outcome.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubgoalOutcome {
    /// Its index, from 0.
    pub subgoal: usize,
    /// How it ended.
    pub state: SubgoalState,
}

/// What a decomposition records.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Decomposition {
    /// The model that proposed the subgoals.
    pub proposer: String,
    /// Its role.
    pub proposed_by: Proposer,
    /// The subgoals, in order.
    pub subgoals: Vec<Subgoal>,
    /// How each ended.
    pub outcomes: Vec<SubgoalOutcome>,
}

/// The decomposition intervention.
pub struct Decompose<'a> {
    /// The examiner that lays out and grades the workspace.
    pub checkout: &'a BenchCheckout,
    /// The student's own model, which works the subgoals.
    pub student: Model,
    /// The model that proposes them.
    pub proposer: Model,
    /// Its role.
    pub proposer_role: Proposer,
    /// The time the subgoals' work may take in all.
    pub deadline: Duration,
    /// Stops the work.
    pub cancel: CancelToken,
}

/// What the proposer is asked about.
#[derive(Serialize)]
struct Brief<'a> {
    instruction: &'a str,
    failure: String,
    kept_actions: Vec<String>,
    most_subgoals: usize,
}

const ROLE: &str = "You plan how an agent finishes a task it failed. The agent works in a \
     workspace with a shell and file tools, and you see the task's instruction, why its attempt \
     failed and the actions it took before the failing one, which it keeps.";

fn task() -> String {
    format!(
        "Split what is left of the task into at most {MAX_SUBGOALS} ordered subgoals the agent \
         works one after another from where its kept actions left the workspace. Each subgoal \
         says what to do in one or two sentences. Where the task allows checking a state on \
         the way (a file the agent must write, a partial answer it must contain), give the \
         subgoal a check of that state: `file_exists` with a path relative to the workspace, or \
         `file_contains` with a path and a short text the file must hold; give no check where \
         none is certain. Never state the answer itself: the agent must still work it out. The \
         last subgoal finishes the task."
    )
}

/// Refuses a plan the student cannot work or the session cannot check.
fn valid(plan: &Plan) -> Result<(), String> {
    if plan.subgoals.is_empty() || plan.subgoals.len() > MAX_SUBGOALS {
        return Err(format!(
            "give between 1 and {MAX_SUBGOALS} subgoals, not {}",
            plan.subgoals.len()
        ));
    }
    for (n, subgoal) in plan.subgoals.iter().enumerate() {
        if subgoal.goal.trim().is_empty() {
            return Err(format!("subgoal {} says nothing to do", n + 1));
        }
        if let Some(check) = &subgoal.check {
            let path = Path::new(check.path());
            let inside = !check.path().is_empty()
                && path.is_relative()
                && path
                    .components()
                    .all(|c| matches!(c, std::path::Component::Normal(_)));
            if !inside {
                return Err(format!(
                    "subgoal {}'s check names {:?}, not a path inside the workspace",
                    n + 1,
                    check.path()
                ));
            }
        }
    }
    Ok(())
}

/// The user turn that leads subgoal `index` of `subgoals`.
fn turn(index: usize, subgoals: &[Subgoal]) -> String {
    let subgoal = &subgoals[index];
    let mut text = format!(
        "Continue the task one step at a time. Step {} of {}: {}",
        index + 1,
        subgoals.len(),
        subgoal.goal.trim()
    );
    match &subgoal.check {
        Some(Check::FileExists { path }) => {
            text.push_str(&format!(" This step is done when `{path}` exists."));
        }
        Some(Check::FileContains { path, text: holds }) => {
            text.push_str(&format!(
                " This step is done when `{path}` contains `{holds}`."
            ));
        }
        None => {}
    }
    if index + 1 == subgoals.len() {
        text.push_str(" This is the last step: when it is done, the task is done.");
    }
    text
}

/// The student's actions through the kept step, each cut to an excerpt.
fn kept_actions(case: &Case<'_>, keep_through: u64) -> Vec<String> {
    let steps = &case.student.trajectory.steps;
    let lead = u64::from(steps.first().map(|s| s.source) == Some(atif::StepOrigin::System));
    let excerpt = |text: String| match text.char_indices().nth(ACTION_EXCERPT) {
        Some((end, _)) => format!("{}...", &text[..end]),
        None => text,
    };
    steps
        .iter()
        .filter(|s| s.source == atif::StepOrigin::Agent && s.step_id <= keep_through + lead)
        .flat_map(|s| s.tool_calls.iter().flatten())
        .map(|call| excerpt(format!("{}({})", call.function_name, call.arguments)))
        .collect()
}

impl Decompose<'_> {
    /// The proposer's subgoals for `case`.
    fn propose(&self, ctx: &Context, case: &Case<'_>, keep_through: u64) -> Result<Plan, String> {
        let evidence: Vec<&str> = case
            .diagnosis
            .evidence
            .iter()
            .map(|e| e.excerpt.as_str())
            .collect();
        let brief = Brief {
            instruction: &case.student.instruction,
            failure: format!(
                "{} ({}): {}",
                case.diagnosis.class.as_str(),
                case.diagnosis.rule,
                evidence.join(" | ")
            ),
            kept_actions: kept_actions(case, keep_through),
            most_subgoals: MAX_SUBGOALS,
        };
        let call = TypedCall::<Plan>::new("decompose_task", task(), ROLE, PROPOSE_DEADLINE)
            .max_output_tokens(PROPOSE_MAX_OUTPUT_TOKENS)
            .cancel(self.cancel.clone())
            .postcondition(valid);
        ctx.block_on(call.run(&self.proposer, &brief))
            .map_err(|e| format!("the proposer gave no decomposition: {e}"))
    }
}

impl Intervention for Decompose<'_> {
    fn name(&self) -> &'static str {
        InterventionKind::Decompose.as_str()
    }

    fn actor(&self) -> &str {
        &self.student.identity
    }

    fn variant(&self) -> Option<String> {
        Some(format!("proposer:{}", self.proposer.identity))
    }

    fn attempt(&self, ctx: &Context, case: &Case<'_>) -> Result<Attempted, String> {
        let keep_through = case
            .diagnosis
            .last_good_step()
            .ok_or("the diagnosis places no last good step")?;
        let plan = self.propose(ctx, case, keep_through)?;
        let turns: Vec<String> = (0..plan.subgoals.len())
            .map(|i| turn(i, &plan.subgoals))
            .collect();
        let mut outcomes: Vec<SubgoalOutcome> = (0..plan.subgoals.len())
            .map(|subgoal| SubgoalOutcome {
                subgoal,
                state: SubgoalState::NotRun,
            })
            .collect();
        let mut options = SolveOptions::new(self.deadline);
        options.cancel = Some(self.cancel.clone());
        let subgoals = &plan.subgoals;
        let forked = fork_and_grade(
            ctx,
            self.checkout,
            case,
            Continuation {
                model: &self.student,
                role: POLICY,
                options: self.student.solving(options),
                briefing: None,
                turns: &turns,
                after_turn: &mut |index, tree| {
                    let state = match &subgoals[index].check {
                        None => SubgoalState::Unchecked,
                        Some(check) if check.holds(tree) => SubgoalState::Reached,
                        Some(_) => SubgoalState::Missed,
                    };
                    outcomes[index].state = state;
                    state != SubgoalState::Missed
                },
            },
        )?;
        let notes = Notes {
            decomposition: Some(Decomposition {
                proposer: self.proposer.identity.clone(),
                proposed_by: self.proposer_role,
                subgoals: plan.subgoals.clone(),
                outcomes,
            }),
            ..Notes::default()
        };
        // Guided by the teacher's plan, the student's success is a helped
        // one; by its own plan, its own.
        let provenance = Provenance {
            teacher: self.proposer_role == Proposer::Teacher,
            ..Provenance::new(self.student.identity.clone(), ctx.clock())
        };
        finish(case, self.name(), forked, notes, provenance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(subgoals: Vec<Subgoal>) -> Plan {
        Plan { subgoals }
    }

    fn subgoal(goal: &str, check: Option<Check>) -> Subgoal {
        Subgoal {
            goal: goal.into(),
            check,
        }
    }

    #[test]
    fn a_plan_is_refused_unless_its_checks_stay_in_the_workspace() {
        let exists = |path: &str| Some(Check::FileExists { path: path.into() });
        assert!(valid(&plan(vec![subgoal("sum", exists("answer.json"))])).is_ok());
        assert!(valid(&plan(vec![])).is_err());
        assert!(valid(&plan(vec![subgoal("", None)])).is_err());
        for outside in ["/etc/passwd", "../x", "a/../../x", ""] {
            assert!(
                valid(&plan(vec![subgoal("x", exists(outside))])).is_err(),
                "{outside}"
            );
        }
        let many = (0..=MAX_SUBGOALS)
            .map(|i| subgoal(&i.to_string(), None))
            .collect();
        assert!(valid(&plan(many)).is_err());
    }

    #[test]
    fn a_check_reads_the_workspace_state_it_names() {
        let tree = tempfile::tempdir().unwrap();
        let contains = Check::FileContains {
            path: "answer.json".into(),
            text: "13".into(),
        };
        assert!(!contains.holds(tree.path()));
        std::fs::write(tree.path().join("answer.json"), "{\"sum\": 13}").unwrap();
        assert!(contains.holds(tree.path()));
        assert!(Check::FileExists {
            path: "answer.json".into()
        }
        .holds(tree.path()));
        let subgoals = vec![subgoal("read", None), subgoal("write", Some(contains))];
        assert!(turn(1, &subgoals).contains("`answer.json` contains `13`"));
        assert!(turn(1, &subgoals).contains("last step"));
        assert!(!turn(0, &subgoals).contains("last step"));
    }
}
