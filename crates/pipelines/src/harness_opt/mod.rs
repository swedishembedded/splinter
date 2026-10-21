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
pub mod llm;
pub mod space;

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::Duration;

use serde::Serialize;
use splinter_agent::sven::search::{
    search, CandidateEnvironment, EvalLimits, Proposer, SearchBudget, SearchError,
    SearchEvaluation, SearchMetrics, StopReason, Strategy, TrajectoryRef, VerifierVerdict,
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
use llm::{Feedback, LlmProposer, ProposalStats};
use space::{MutationProposer, MutationSpace};

/// The document class a promotion is stored under.
const CHAMPION: &str = "harness_champion";

/// The pointer that names the current champion document.
pub const CHAMPION_POINTER: &str = "harness/champion";

/// Who proposes candidates.
#[derive(Clone, Debug)]
pub enum ProposerChoice {
    /// One mechanical mutation of the space per candidate.
    Mutation,
    /// A model proposes edits, each compiled by sven before it counts; a
    /// proposal the model cannot make compile falls back to a mutation when
    /// the request has a space.
    Model {
        /// The proposing model.
        model: ModelRef,
        /// How long one proposal may take, repairs included.
        deadline: Duration,
        /// Corrections a reply that does not compile may be sent back for.
        repairs: u32,
    },
}

/// What an optimisation is asked to do.
pub struct OptimiseRequest<'a> {
    /// The frozen tasks every harness is run on.
    pub task_set: &'a TaskSetId,
    /// The model every harness runs on.
    pub solver: &'a ModelRef,
    /// The baseline harness document: the starting point and the arm every
    /// candidate is compared with.
    pub baseline: &'a str,
    /// Where mutations may differ from the baseline; required by
    /// [`ProposerChoice::Mutation`], the fallback of [`ProposerChoice::Model`].
    pub space: Option<MutationSpace>,
    /// Who proposes.
    pub proposer: ProposerChoice,
    /// Best-of-N or annealing.
    pub strategy: Strategy,
    /// Harnesses evaluated, the baseline's replicate included.
    pub evaluations: u64,
    /// Candidates evaluated at once (best-of-N only); each is a whole arm, so
    /// the model server sees this many arms' attempts together.
    pub parallelism: usize,
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
    /// What the proposing model did; `None` for a mutation proposer.
    pub proposals: Option<ProposalStats>,
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
            resume: None,
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

/// What a candidate's comparison showed, kept by the set it was run as.
type Findings = Mutex<BTreeMap<String, (Digest, Evidence, Decision, SetId)>>;

/// The candidate environment of a search: one candidate is one arm and its
/// comparison with the baseline's, run on a thread of its own so that
/// candidates in flight overlap.
struct HarnessEnvironment<'scope, 'env: 'scope> {
    scope: &'scope std::thread::Scope<'scope, 'env>,
    ctx: &'env Context,
    request: &'env OptimiseRequest<'env>,
    baseline_set: SetId,
    feedback: &'env Feedback,
    findings: Findings,
}

impl CandidateEnvironment<String> for HarnessEnvironment<'_, '_> {
    async fn evaluate(
        &self,
        candidate: &String,
        _limits: &EvalLimits,
    ) -> Result<SearchEvaluation, SearchError> {
        // The pipeline stages are blocking and drive runtimes of their own,
        // so each runs on its own thread and is awaited through a channel.
        let (done, reply) = tokio::sync::oneshot::channel();
        let (ctx, request, baseline_set) = (self.ctx, self.request, self.baseline_set.clone());
        let document = candidate.clone();
        self.scope.spawn(move || {
            let compared = run_arm(ctx, request, &document).and_then(|set| {
                let comparison = compare(ctx, &set, &baseline_set, request.interval_seed, None)?;
                Ok((set, comparison))
            });
            let _ = done.send(compared);
        });
        let (set, comparison) = reply
            .await
            .map_err(|_| SearchError::environment("the evaluation thread panicked"))?
            .map_err(SearchError::environment)?;
        let evidence = Evidence::of(&comparison);
        let decision = decide(&evidence, &self.request.rule);
        self.feedback.record(
            candidate,
            format!(
                "{}; the candidate solved {} of {} paired tasks, the baseline {}",
                decision.reason, evidence.candidate_right, evidence.paired, evidence.baseline_right
            ),
        );
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
        self.findings
            .lock()
            .map_err(|_| SearchError::environment("the findings log is poisoned"))?
            .insert(
                set.to_string(),
                (document_digest(candidate), evidence, decision, set),
            );
        Ok(evaluation)
    }
}

/// Runs the optimisation `request` describes.
///
/// # Errors
///
/// [`OrchestratorError`] when the request is unusable (a mutation proposer
/// without a space), the baseline cannot be run, the search fails (a
/// proposal that cannot be made, an environment failure) or the promotion
/// cannot be recorded. Candidates that merely fail are results, not errors.
pub fn optimise(
    ctx: &Context,
    request: &OptimiseRequest<'_>,
) -> Result<Optimised, OrchestratorError> {
    let refuse = |why: &str| Err(OrchestratorError::Refused(why.to_string()));
    if request.attempts == 0 {
        return refuse("a harness needs at least one attempt per task");
    }
    if matches!(request.proposer, ProposerChoice::Mutation) && request.space.is_none() {
        return refuse("the mutation proposer needs a mutation space");
    }
    let baseline_set = run_arm(ctx, request, request.baseline)?;
    let feedback = Feedback::default();
    let budget = SearchBudget {
        max_evaluations: request.evaluations,
        max_tokens: None,
        max_wall: None,
        stop_on_verified: false,
        parallelism: request.parallelism,
    };
    let llm = match &request.proposer {
        ProposerChoice::Mutation => None,
        ProposerChoice::Model {
            model,
            deadline,
            repairs,
        } => Some(LlmProposer::new(
            ctx.model(model)?,
            &feedback,
            request.space.clone(),
            *deadline,
            *repairs,
            request.cancel.clone(),
        )),
    };
    let mutation = request
        .space
        .clone()
        .map(|space| MutationProposer { space });
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| OrchestratorError::Refused(format!("cannot start the search: {e}")))?;
    let (report, findings) = std::thread::scope(|scope| {
        let environment = HarnessEnvironment {
            scope,
            ctx,
            request,
            baseline_set: baseline_set.clone(),
            feedback: &feedback,
            findings: Mutex::default(),
        };
        let report = match (&llm, &mutation) {
            (Some(proposer), _) => {
                runtime.block_on(run_search(request, proposer, &environment, &budget))
            }
            (None, Some(proposer)) => {
                runtime.block_on(run_search(request, proposer, &environment, &budget))
            }
            (None, None) => unreachable!("checked above: a mutation proposer has a space"),
        };
        (report, environment.findings)
    });
    let report = report.map_err(|e| OrchestratorError::Refused(e.to_string()))?;
    let findings = findings
        .into_inner()
        .map_err(|_| OrchestratorError::Refused("the findings log is poisoned".into()))?;

    let candidates: Vec<CandidateLine> = report
        .trace
        .iter()
        .map(|entry| {
            let set = entry
                .evaluation
                .trajectories
                .first()
                .map(|t| t.run_id.clone())
                .unwrap_or_default();
            let (document, evidence, decision, experience_set) = findings
                .get(&set)
                .cloned()
                .ok_or_else(|| OrchestratorError::Refused(format!("no finding for set {set}")))?;
            Ok(CandidateLine {
                index: entry.index,
                document,
                experience_set,
                evidence,
                decision,
            })
        })
        .collect::<Result<_, OrchestratorError>>()?;

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
        proposals: llm.as_ref().map(LlmProposer::stats),
        promoted,
        stop: report.stop,
        metrics: report.metrics,
    })
}

async fn run_search<P: Proposer<String>>(
    request: &OptimiseRequest<'_>,
    proposer: &P,
    environment: &HarnessEnvironment<'_, '_>,
    budget: &SearchBudget,
) -> Result<splinter_agent::sven::search::SearchReport<String>, SearchError> {
    search(
        request.strategy,
        request.baseline.to_string(),
        proposer,
        environment,
        budget,
        request.search_seed,
    )
    .await
}
