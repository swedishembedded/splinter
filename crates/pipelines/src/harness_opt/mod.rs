// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements automated improvement of agent harnesses
// that adopts a change only on verified, regression-free evidence, for its
// clients. If your team needs expertise in measuring whether a harness
// change really helped, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Automated harness optimisation: candidate sven harnesses are drawn from a
//! mutation space, each is run on the frozen task set through sven, its
//! attempts are verified by the tasks' own verifiers, and it is compared with
//! the baseline harness's attempts by the paired family-clustered comparison
//! the release gate uses. Only a candidate the promotion rule
//! ([`gate::decide`]) accepts - a verified improvement with no regressed
//! family - is promoted; the search itself is sven's, best-of-N or simulated
//! annealing, and follows the dense share of tasks the candidate got right,
//! which guides and never decides.
//!
//! A promotion is a stored document (the candidate harness, its baseline, the
//! evidence and the experience sets its comparison rests on) and a move of the
//! `harness/champion` pointer, whose history is the lineage of champions.
//!
//! The first evaluation of a search is the baseline itself against the
//! baseline's own arm: a replicate that shows how much two identical runs
//! differ, and the search's starting point.

pub mod gate;
pub mod space;

use std::sync::Mutex;

use serde::Serialize;
use splinter_agent::sven::search::{
    search, CandidateEnvironment, EvalLimits, SearchBudget, SearchError, SearchEvaluation,
    SearchMetrics, StopReason, Strategy, TrajectoryRef, VerifierVerdict,
};
use splinter_agent::CancelToken;
use splinter_core::digest::Digest;
use splinter_core::model_ref::ModelRef;
use splinter_core::rng::Seed;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_store::experiences::SetId;
use splinter_store::tasks::TaskSetId;

use crate::compare::compare;
use crate::solving::{harness_protocol, solve_tasks, SamplingChoice, SolveRequest};
use crate::verify::{verify_set, Grading};
use gate::{decide, Decision, Evidence, PromotionRule, Verdict};
use space::{MutationProposer, MutationSpace};

/// The document class a promotion is stored under.
const CHAMPION: &str = "harness_champion";

/// The pointer that names the current champion document.
pub const CHAMPION_POINTER: &str = "harness/champion";

/// What an optimisation is asked to do.
pub struct OptimiseRequest<'a> {
    /// The frozen tasks every harness is run on.
    pub task_set: &'a TaskSetId,
    /// The model every harness runs on.
    pub solver: &'a ModelRef,
    /// The baseline harness document: the starting point and the arm every
    /// candidate is compared with.
    pub baseline: &'a str,
    /// Where candidates may differ from it.
    pub space: MutationSpace,
    /// Best-of-N or annealing.
    pub strategy: Strategy,
    /// Harnesses evaluated, the baseline's replicate included.
    pub evaluations: u64,
    /// Solves per task and harness (at least one).
    pub attempts: usize,
    /// What a candidate must show to be promoted.
    pub rule: PromotionRule,
    /// Seeds the search's draws.
    pub search_seed: u64,
    /// Seeds the comparisons' intervals.
    pub interval_seed: Seed,
    /// Stops the run.
    pub cancel: CancelToken,
}

/// One harness the search evaluated.
#[derive(Clone, Debug, Serialize)]
pub struct CandidateLine {
    /// Evaluation index, from zero.
    pub index: u64,
    /// The harness document's address.
    pub document: Digest,
    /// The experience set of its verified attempts.
    pub experience_set: SetId,
    /// What its comparison with the baseline showed.
    pub evidence: Evidence,
    /// What the rule made of it.
    pub decision: Decision,
}

/// A candidate that was promoted.
#[derive(Clone, Debug, Serialize)]
pub struct Promotion {
    /// The champion document's address.
    pub document: Digest,
    /// The pointer's new version.
    pub version: u64,
    /// The candidate's evaluation index.
    pub index: u64,
}

/// What an optimisation did.
#[derive(Clone, Debug, Serialize)]
pub struct Optimised {
    /// The baseline document's address.
    pub baseline: Digest,
    /// The baseline's experience set.
    pub baseline_set: SetId,
    /// Every harness evaluated, in order.
    pub candidates: Vec<CandidateLine>,
    /// The promotion, when a candidate earned it.
    pub promoted: Option<Promotion>,
    /// Why the search ended.
    pub stop: StopReason,
    /// The search's own measurements.
    pub metrics: SearchMetrics,
}

/// The stored record of a promotion.
#[derive(Serialize)]
struct Champion<'a> {
    document: &'a str,
    baseline: &'a Digest,
    baseline_set: &'a SetId,
    candidate_set: &'a SetId,
    evidence: &'a Evidence,
    reason: &'a str,
}

fn document_digest(document: &str) -> Digest {
    Digest::of(document.as_bytes())
}

/// Solves the task set with `harness` and verifies the attempts: the harness's
/// arm.
fn run_arm(
    ctx: &Context,
    request: &OptimiseRequest<'_>,
    harness: &str,
) -> Result<SetId, OrchestratorError> {
    let solved = solve_tasks(
        ctx,
        &SolveRequest {
            task_set: request.task_set,
            solver: request.solver,
            attempts: request.attempts,
            sampling: SamplingChoice::Own,
            teacher: false,
            system: None,
            work: harness_protocol(),
            harness: Some(harness),
            deadline: None,
            cancel: request.cancel.clone(),
        },
    )?;
    verify_set(
        ctx,
        &solved.experience_set,
        Grading::CodeOnly,
        &request.cancel,
    )?;
    Ok(solved.experience_set)
}

/// The candidate environment of a search: one candidate is one arm and its
/// comparison with the baseline's.
struct HarnessEnvironment<'a> {
    ctx: &'a Context,
    request: &'a OptimiseRequest<'a>,
    baseline_set: SetId,
    lines: Mutex<Vec<CandidateLine>>,
}

impl CandidateEnvironment<String> for HarnessEnvironment<'_> {
    async fn evaluate(
        &self,
        candidate: &String,
        _limits: &EvalLimits,
    ) -> Result<SearchEvaluation, SearchError> {
        // The pipeline stages are blocking and drive runtimes of their own, so
        // they run on a thread of their own rather than inside this one.
        let compared = std::thread::scope(|scope| {
            scope
                .spawn(|| -> Result<_, OrchestratorError> {
                    let set = run_arm(self.ctx, self.request, candidate)?;
                    let comparison = compare(
                        self.ctx,
                        &set,
                        &self.baseline_set,
                        self.request.interval_seed,
                    )?;
                    Ok((set, comparison))
                })
                .join()
        })
        .map_err(|_| SearchError::environment("the evaluation thread panicked"))?
        .map_err(SearchError::environment)?;
        let (set, comparison) = compared;
        let evidence = Evidence::of(&comparison);
        let decision = decide(&evidence, &self.request.rule);
        let terminal = if decision.verdict == Verdict::Promote {
            VerifierVerdict::Passed
        } else {
            VerifierVerdict::Failed {
                reason: decision.reason.clone(),
            }
        };
        let evaluation = SearchEvaluation {
            terminal: Some(terminal),
            utility: evidence.utility(),
            signals: [
                ("paired".to_string(), evidence.paired as f64),
                ("p_value".to_string(), evidence.p_value),
                (
                    "regressed_families".to_string(),
                    evidence.regressed_families.len() as f64,
                ),
            ]
            .into(),
            tokens: 0,
            trajectories: vec![TrajectoryRef {
                run_id: set.to_string(),
                location: None,
            }],
        };
        let mut lines = self
            .lines
            .lock()
            .map_err(|_| SearchError::environment("the candidate log is poisoned"))?;
        let index = lines.len() as u64;
        lines.push(CandidateLine {
            index,
            document: document_digest(candidate),
            experience_set: set,
            evidence,
            decision,
        });
        Ok(evaluation)
    }
}

/// Runs the optimisation `request` describes.
///
/// # Errors
///
/// [`OrchestratorError`] when the baseline cannot be run, the search fails (a
/// candidate that cannot be drawn, an environment failure) or the promotion
/// cannot be recorded. Candidates that merely fail are results, not errors.
pub fn optimise(
    ctx: &Context,
    request: &OptimiseRequest<'_>,
) -> Result<Optimised, OrchestratorError> {
    if request.attempts == 0 {
        return Err(OrchestratorError::Refused(
            "a harness needs at least one attempt per task".into(),
        ));
    }
    let baseline_set = run_arm(ctx, request, request.baseline)?;
    let environment = HarnessEnvironment {
        ctx,
        request,
        baseline_set: baseline_set.clone(),
        lines: Mutex::new(Vec::new()),
    };
    let proposer = MutationProposer {
        space: request.space.clone(),
    };
    let budget = SearchBudget {
        max_evaluations: request.evaluations,
        max_tokens: None,
        max_wall: None,
        stop_on_verified: false,
    };
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .map_err(|e| OrchestratorError::Refused(format!("cannot start the search: {e}")))?;
    let report = runtime
        .block_on(search(
            request.strategy,
            request.baseline.to_string(),
            &proposer,
            &environment,
            &budget,
            request.search_seed,
        ))
        .map_err(|e| OrchestratorError::Refused(e.to_string()))?;
    let candidates = environment
        .lines
        .into_inner()
        .map_err(|_| OrchestratorError::Refused("the candidate log is poisoned".into()))?;

    let baseline = document_digest(request.baseline);
    let promoted = match report.best_verified() {
        None => None,
        Some(best) => {
            let line = &candidates[best.index as usize];
            let champion = Champion {
                document: &best.candidate,
                baseline: &baseline,
                baseline_set: &baseline_set,
                candidate_set: &line.experience_set,
                evidence: &line.evidence,
                reason: &line.decision.reason,
            };
            let document = ctx.workspace().put_document(CHAMPION, &champion)?;
            let previous = ctx
                .workspace()
                .pointer(CHAMPION_POINTER)?
                .map(|(_, value)| value);
            let version = ctx.workspace().move_pointer(
                CHAMPION_POINTER,
                previous.as_deref(),
                document.as_str(),
                &ctx.clock().utc_now(),
            )?;
            Some(Promotion {
                document,
                version,
                index: best.index,
            })
        }
    };
    Ok(Optimised {
        baseline,
        baseline_set,
        candidates,
        promoted,
        stop: report.stop,
        metrics: report.metrics,
    })
}
