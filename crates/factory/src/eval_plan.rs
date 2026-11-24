// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pre-registered model evaluations whose
// decision rule is fixed before a candidate is seen, for its clients. If
// your team needs expertise in trustworthy model evaluation, you can procure
// our services by sending an email to info@swedishembedded.com.

//! An evaluation plan: which tasks, how many attempts, under which
//! conditions, and the rule that decides, fixed before any candidate runs.
//!
//! ```yaml
//! schema: splinter.evalplan/v1
//! collection: collections/vim-mastery.yaml
//! split: eval
//! environment: environments/vim/environment.yaml
//! attempts: 4
//! seed: 7
//! decision: { alpha: 0.05, min_difference: 0.0 }
//! ```
//!
//! A protected plan names an evaluation authority (the examiner) instead of
//! tasks: its sealed tasks stay with the authority, which examines the
//! candidate and signs the verdict.

use serde::{Deserialize, Serialize};

use crate::recipe::RecipeError;

/// The schema every evaluation plan names.
pub const EVAL_PLAN_SCHEMA: &str = "splinter.evalplan/v1";

/// The rule that decides.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionRule {
    /// The one-sided significance level of the improvement test.
    pub alpha: f64,
    /// The smallest pass-rate improvement that counts.
    #[serde(default)]
    pub min_difference: f64,
    /// How many candidates are compared against one baseline under the
    /// plan; alpha is divided among them (Bonferroni).
    #[serde(default)]
    pub comparisons: Option<u32>,
}

/// An evaluation authority, for a protected plan: the task repository's
/// examiner, which serves a candidate, examines it on sealed tasks it alone
/// holds and signs its verdict.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Authority {
    /// The examiner's command (argv, run from the repository root), to which
    /// `--candidate FILE --out FILE` is appended; empty when it is an
    /// operator.
    #[serde(default)]
    pub command: Vec<String>,
    /// The examiner as one of the repository's operators: given the
    /// candidate (and its predecessor) as inputs, it writes `verdict.json`.
    #[serde(default)]
    pub operator: Option<String>,
    /// The examiner's public key file, to check its verdict's signature.
    pub public_key: String,
}

/// An evaluation plan file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvalPlan {
    /// Always [`EVAL_PLAN_SCHEMA`].
    pub schema: String,
    /// The tasks' collection file (a development plan).
    #[serde(default)]
    pub collection: Option<String>,
    /// The split of it.
    #[serde(default)]
    pub split: Option<String>,
    /// The environment every arm runs in.
    #[serde(default)]
    pub environment: Option<String>,
    /// Attempts per task and arm.
    pub attempts: u32,
    /// Reruns of a task an arm failed, at most this many more attempts of that task alone
    /// (0: none). A task counts as solved when any of its attempts passed; the reruns cost
    /// only the tasks that need them.
    #[serde(default)]
    pub rerun_failed: u32,
    /// Arms run at once.
    #[serde(default = "four")]
    pub concurrency: u32,
    /// The seed attempts draw from (the same for every arm).
    pub seed: u64,
    /// The rule.
    pub decision: DecisionRule,
    /// The authority, for a protected plan.
    #[serde(default)]
    pub authority: Option<Authority>,
}

fn four() -> u32 {
    4
}

impl EvalPlan {
    /// Parse and check a plan.
    ///
    /// # Errors
    /// The wrong shape or schema; a plan that is neither development (tasks)
    /// nor protected (authority), or both; a bad rule.
    pub fn parse(text: &str) -> Result<Self, RecipeError> {
        let plan: Self =
            serde_yaml::from_str(text).map_err(|e| RecipeError(vec![e.to_string()]))?;
        let mut problems = Vec::new();
        if plan.schema != EVAL_PLAN_SCHEMA {
            problems.push(format!(
                "schema is {:?}; expected {EVAL_PLAN_SCHEMA}",
                plan.schema
            ));
        }
        match (&plan.collection, &plan.authority) {
            (Some(_), Some(_)) => {
                problems.push("a plan names tasks or an authority, not both".into())
            }
            // A development plan without a collection is given its tasks by
            // the evaluation stage (`tasks`).
            (None, None) if plan.environment.is_none() => problems.push(
                "a plan names an environment (development) or an authority (protected)".into(),
            ),
            (_, None) if plan.environment.is_none() => {
                problems.push("a development plan names the environment its arms run in".into());
            }
            _ => {}
        }
        if !(plan.decision.alpha > 0.0 && plan.decision.alpha < 0.5) {
            problems.push(format!("alpha {} is not in (0, 0.5)", plan.decision.alpha));
        }
        if let Some(authority) = &plan.authority {
            if authority.command.is_empty() == authority.operator.is_none() {
                problems.push("an authority names its command or its operator, one of them".into());
            }
        }
        if plan.attempts == 0 {
            problems.push("a plan needs at least one attempt per task".into());
        }
        if problems.is_empty() {
            Ok(plan)
        } else {
            Err(RecipeError(problems))
        }
    }

    /// Whether the plan is protected.
    #[must_use]
    pub fn protected(&self) -> bool {
        self.authority.is_some()
    }
}
