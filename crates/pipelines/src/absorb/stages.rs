// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning from the sessions a person holds
// with an agent, for its clients. If your team needs expertise in continual
// learning from user feedback, you can procure our services by sending an
// email to info@swedishembedded.com.

//! The stages of `absorb`, and the pipeline that orders them; see the
//! module documentation of [`super`].

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use splinter_agent::claims::ClaimExtractor;
use splinter_core::annotation::Strength;
use splinter_core::claim::{Absorption, ClaimId};
use splinter_core::model_ref::{ModelRef, POLICY_DEFAULT};
use splinter_core::role::Role;
use splinter_core::source::SourceId;
use splinter_knowledge::claims::extract::ExtractionPolicy;
use splinter_knowledge::claims::task::claim_task;
use splinter_knowledge::claims::Ledger;
use splinter_orchestrator::pipeline::{FnStage, Pipeline, StageEnd};
use splinter_orchestrator::runs::{to_json, Recorder};
use splinter_orchestrator::{roles, Context, OrchestratorError};
use splinter_store::runs::{list_runs, RunStatus};

use super::dataset::assemble;
use super::gate::{decide_claims, publish};
use super::kit::{self, build_kits, ClaimKit, KitRequest, Live};
use super::sealed::SealedProbes;
use super::{AbsorbRequest, Absorbed};
use crate::claims::{
    extract, gate as rule_claims, resolve_set, ClaimsExtracted, ExtractRequest, GateRequest,
};
use crate::datasets::DEFAULT_MIN_STRENGTH;
use crate::learn::{records_at_share, PolicyUsed};
use crate::rehearsal::{rehearse, RehearseRequest, REHEARSAL_SEED};
use crate::release::arm;
use crate::sessions::{intake, IntakeRequest};
use crate::train::{
    auto_records_per_step, load_candidate, train, Rehearse, TrainRequest, Trainer, Tuning,
};
use crate::verify::adopt_judge;

type Done = Result<StageEnd, OrchestratorError>;

/// Refuses a request that cannot be run, before anything is.
pub(super) fn validate(request: &AbsorbRequest) -> Result<(), OrchestratorError> {
    if request.sessions.is_empty() {
        return Err(OrchestratorError::Refused(
            "name at least one session file or directory".into(),
        ));
    }
    if !matches!(request.policy, ModelRef::Policy(_)) {
        return Err(OrchestratorError::Refused(format!(
            "{} is not a policy: a release is made under an alias, so name `policy:<alias>`",
            request.policy
        )));
    }
    if !(0.0..1.0).contains(&request.rehearsal_share) {
        return Err(OrchestratorError::Refused(format!(
            "the rehearsal share {} is not in [0, 1)",
            request.rehearsal_share
        )));
    }
    if request.epochs == 0 || request.paraphrases == 0 {
        return Err(OrchestratorError::Refused(
            "epochs and paraphrases are at least one".into(),
        ));
    }
    Ok(())
}

struct State<'a> {
    request: &'a AbsorbRequest,
    trainer: &'a dyn Trainer,
    alias: String,
    generator: ModelRef,
    teacher: ModelRef,
    judge: Option<ModelRef>,
    sealed: SealedProbes,
    report: Absorbed,
    system: Option<String>,
    sessions: Vec<SourceId>,
    live: Vec<Live>,
    pending: BTreeSet<ClaimId>,
    kits: BTreeMap<ClaimId, ClaimKit>,
    min_strength: Strength,
    rehearsal: Option<String>,
    candidate: Option<crate::train::Candidate>,
    previous: Option<Previous>,
}

/// What a failed run of the same request left that is worth keeping.
#[derive(Clone)]
struct Previous {
    dataset: String,
    candidate: String,
}

pub(super) fn run(
    ctx: &Context,
    request: &AbsorbRequest,
    trainer: &dyn Trainer,
    run: &mut Recorder<'_>,
) -> Result<Absorbed, OrchestratorError> {
    let ModelRef::Policy(alias) = &request.policy else {
        unreachable!("validate refuses a reference that is not a policy")
    };
    let assignments = roles::assignments(ctx.config(), &request.roles)?;
    let mut roles_used = BTreeMap::new();
    for role in [Role::Policy, Role::Teacher, Role::Generator] {
        roles_used.insert(role, ctx.selection(assignments.get(role))?.identity());
    }
    let judge = request.roles.get(&Role::Judge).cloned();
    if let Some(judge) = &judge {
        roles_used.insert(Role::Judge, ctx.selection(judge)?.identity());
    }
    let mut state = State {
        request,
        trainer,
        alias: alias.clone(),
        generator: assignments.get(Role::Generator).clone(),
        teacher: assignments.get(Role::Teacher).clone(),
        judge,
        sealed: SealedProbes::load(&request.sealed_probes)?,
        report: Absorbed {
            roles: roles_used,
            dry_run: request.dry_run,
            release_asked: !request.no_release,
            ..Absorbed::default()
        },
        system: None,
        sessions: Vec::new(),
        live: Vec::new(),
        pending: BTreeSet::new(),
        kits: BTreeMap::new(),
        min_strength: DEFAULT_MIN_STRENGTH,
        rehearsal: None,
        candidate: None,
        previous: previous_run(ctx, request, run.id()),
    };
    let stopped = pipeline().run(ctx, run, &mut state, None)?;
    let mut report = state.report;
    report.stopped = stopped;
    Ok(report)
}

fn pipeline<'a>() -> Pipeline<'a, State<'a>> {
    Pipeline::new()
        .then(FnStage::new("policy", policy_stage).ignoring_budget())
        .then(FnStage::new("intake", intake_stage).ignoring_budget())
        .then(FnStage::new("extract", extract_stage).ignoring_budget())
        .then(FnStage::new("gate", gate_stage).ignoring_budget())
        .then(FnStage::new("kits", kits_stage).ignoring_budget())
        .then(FnStage::new("dataset", dataset_stage).ignoring_budget())
        .then(
            FnStage::new("rehearse", rehearse_stage)
                .when(|s| s.request.rehearsal_share > 0.0)
                .ignoring_budget(),
        )
        .then(FnStage::new("train", train_stage).ignoring_budget())
        .then(
            FnStage::new("release", release_stage)
                .when(|s| !s.request.no_release)
                .ignoring_budget(),
        )
        .then(
            FnStage::new("ledger", ledger_stage)
                .when(|s| s.report.release.is_some())
                .ignoring_budget(),
        )
}

fn value(summary: &impl Serialize) -> Result<serde_json::Value, OrchestratorError> {
    to_json("the stage summary", summary)
}

/// The dataset and candidate of the latest failed or cancelled run of the
/// same request, when it got as far as training one.
fn previous_run(ctx: &Context, request: &AbsorbRequest, current: &str) -> Option<Previous> {
    let arguments = serde_json::to_value(request).ok()?;
    let runs = list_runs(ctx.workspace()).ok()?;
    runs.into_iter()
        .rev()
        .filter(|r| r.command == "absorb" && r.id != current && r.arguments == arguments)
        .find(|r| matches!(r.status, RunStatus::Failed | RunStatus::Cancelled))
        .and_then(|r| {
            let stage = r
                .stages
                .iter()
                .find(|s| s.stage == "train" && s.error.is_none())?;
            let resume = stage.summary.get("resume")?;
            Some(Previous {
                dataset: resume.get("dataset")?.as_str()?.to_string(),
                candidate: resume.get("candidate")?.as_str()?.to_string(),
            })
        })
}

fn policy_stage(ctx: &Context, _: &mut Recorder<'_>, st: &mut State<'_>) -> Done {
    let pin = ctx.policy_pin(&st.alias)?;
    st.system = pin.as_ref().and_then(|p| p.system_prompt.clone());
    st.report.policy = PolicyUsed {
        alias: st.alias.clone(),
        release: pin.map(|p| p.release),
    };
    if st.report.policy.release.is_none() && st.alias != POLICY_DEFAULT {
        return Err(OrchestratorError::NotFound {
            what: "release alias",
            id: st.alias.clone(),
        });
    }
    Ok(StageEnd::done(value(&(
        &st.report.policy,
        &st.report.roles,
    ))?))
}

fn intake_stage(ctx: &Context, _: &mut Recorder<'_>, st: &mut State<'_>) -> Done {
    let taken = intake(
        ctx,
        &IntakeRequest {
            paths: &st.request.sessions,
            max_bytes: st.request.max_session_bytes,
        },
    )?;
    st.sessions = taken.sessions.iter().map(|s| s.source.clone()).collect();
    let summary = value(&taken)?;
    let none = st.sessions.is_empty();
    st.report.intake = Some(taken);
    Ok(if none {
        StageEnd::stop(summary, "no session was taken in")
    } else {
        StageEnd::done(summary)
    })
}

fn extract_stage(ctx: &Context, run: &mut Recorder<'_>, st: &mut State<'_>) -> Done {
    let identity = ClaimExtractor::new(ctx.model(&st.generator)?)
        .identity()
        .to_string();
    let claims = ctx.claims();
    let mut reused = None;
    for id in claims.list_sets()? {
        let set = claims.get_set(&id)?;
        let same_sessions = set
            .sessions
            .iter()
            .map(|s| &s.session)
            .eq(st.sessions.iter());
        if same_sessions
            && set.extractor == identity
            && set.sessions.iter().all(|s| s.failure.is_none())
        {
            reused = Some((id, set));
        }
    }
    let extracted = match reused {
        Some((claim_set, set)) => {
            let mut by_kind = BTreeMap::new();
            for p in set.sessions.iter().flat_map(|s| &s.proposals) {
                *by_kind.entry(p.kind.as_str().to_string()).or_default() += 1;
            }
            ClaimsExtracted {
                claim_set,
                sessions: set.sessions.len(),
                proposals: set.sessions.iter().map(|s| s.proposals.len()).sum(),
                by_kind,
                failed: Vec::new(),
                stopped: None,
            }
        }
        None => extract(
            ctx,
            &ExtractRequest {
                sessions: &st.sessions,
                extractor: &st.generator,
                policy: ExtractionPolicy::default(),
                deadline: None,
                cancel: run.cancel_token(),
            },
        )?,
    };
    let summary = value(&extracted)?;
    st.report.extract = Some(extracted);
    Ok(StageEnd::done(summary))
}

fn gate_stage(ctx: &Context, _: &mut Recorder<'_>, st: &mut State<'_>) -> Done {
    let Some(extracted) = &st.report.extract else {
        unreachable!("the gate stage follows the extract stage")
    };
    let set = resolve_set(ctx, &extracted.claim_set.to_string())?;
    let gated = rule_claims(
        ctx,
        &GateRequest {
            judge: st.judge.as_ref(),
            ..GateRequest::new(&set)
        },
    )?;
    let ledger = Ledger::new(ctx.claims().entries()?);
    st.live = ledger
        .live()
        .map_err(|e| OrchestratorError::Refused(e.to_string()))?
        .into_iter()
        .map(|(id, claim)| (id, claim.clone()))
        .collect();
    let absorbed: BTreeSet<ClaimId> = ctx
        .claims()
        .absorptions()?
        .into_iter()
        .map(|a| a.claim)
        .collect();
    st.pending = st
        .live
        .iter()
        .map(|(id, _)| id.clone())
        .filter(|id| !absorbed.contains(id))
        .collect();
    st.report.live = st.live.len();
    st.report.pending = st.pending.iter().cloned().collect();
    let summary = value(&gated)?;
    st.report.claims = Some(gated);
    Ok(if st.request.dry_run {
        StageEnd::stop(
            summary,
            "dry run: the claims are ruled on and nothing is taught",
        )
    } else if st.pending.is_empty() {
        StageEnd::stop(
            summary,
            "no live claim is new: every one is already absorbed",
        )
    } else {
        StageEnd::done(summary)
    })
}

fn kits_stage(ctx: &Context, run: &mut Recorder<'_>, st: &mut State<'_>) -> Done {
    if let Some(judge) = &st.judge {
        let tasks: Vec<_> = st
            .live
            .iter()
            .filter_map(|(_, claim)| claim_task(&ctx.sources(), claim).ok())
            .map(|made| made.task)
            .collect();
        adopt_judge(
            ctx,
            judge,
            &[
                ("teacher", &st.teacher),
                ("generator", &st.generator),
                ("policy", &st.request.policy),
            ],
            &tasks,
        )?;
        st.min_strength = Strength::Judged;
    }
    let built = build_kits(
        ctx,
        &KitRequest {
            claims: &st.live,
            generator: &st.generator,
            teacher: &st.teacher,
            system: st.system.as_deref(),
            min_strength: st.min_strength,
            paraphrases: st.request.paraphrases,
            deadline: None,
            cancel: run.cancel_token(),
        },
    )?;
    st.kits = kit::kept(ctx)?;
    let summary = value(&built)?;
    st.report.kits = Some(built);
    Ok(StageEnd::done(summary))
}

/// The claims a night trains on: all that are live, or only the new ones
/// when continuing a release.
fn trained_claims(st: &State<'_>) -> Vec<Live> {
    st.live
        .iter()
        .filter(|(id, _)| !st.request.continue_from_release || st.pending.contains(id))
        .cloned()
        .collect()
}

fn dataset_stage(ctx: &Context, _: &mut Recorder<'_>, st: &mut State<'_>) -> Done {
    let claims = trained_claims(st);
    let Some(assembled) = assemble(ctx, &claims, &st.kits, st.system.as_deref(), &st.sealed)?
    else {
        return Ok(StageEnd::halt("no claim has a record to train on"));
    };
    let summary = value(&assembled)?;
    let few = assembled.trained < splinter_data::holdout::MIN_SAMPLES;
    st.report.dataset = Some(assembled);
    Ok(if few {
        StageEnd::stop(summary, "too few records to train on")
    } else {
        StageEnd::done(summary)
    })
}

fn rehearse_stage(ctx: &Context, run: &mut Recorder<'_>, st: &mut State<'_>) -> Done {
    let Some(assembled) = &st.report.dataset else {
        unreachable!("the rehearse stage follows the dataset stage")
    };
    let rehearsed = rehearse(
        ctx,
        &RehearseRequest {
            records: records_at_share(assembled.trained, st.request.rehearsal_share).max(1),
            seed: REHEARSAL_SEED,
            deadline: None,
            cancel: run.cancel_token(),
        },
    )?;
    if let Some((position, leak)) = st.sealed.check_file(&rehearsed.dataset.path)? {
        return Err(OrchestratorError::Refused(format!(
            "rehearsal record {position} {leak}: the rehearsal is refused"
        )));
    }
    st.rehearsal = Some(rehearsed.dataset.dataset.to_string());
    let summary = value(&rehearsed)?;
    st.report.rehearsal = Some(rehearsed);
    Ok(StageEnd::done(summary))
}

#[derive(Serialize)]
struct Trained<'a> {
    candidate: &'a crate::train::Candidate,
    resume: Resume,
}

#[derive(Serialize)]
struct Resume {
    dataset: String,
    candidate: String,
}

fn train_stage(ctx: &Context, run: &mut Recorder<'_>, st: &mut State<'_>) -> Done {
    let Some(assembled) = &st.report.dataset else {
        unreachable!("the train stage follows the dataset stage")
    };
    let dataset = assembled.dataset.to_string();
    let resumed = st
        .previous
        .as_ref()
        .filter(|p| p.dataset == dataset)
        .and_then(|p| load_candidate(ctx, &p.candidate).ok());
    let candidate = match resumed {
        Some(candidate) => candidate,
        None => {
            // Passes over the night's records, as optimizer steps of an
            // average of records.
            let per_step = st
                .request
                .tuning
                .records_per_step
                .unwrap_or_else(|| auto_records_per_step(assembled.trained));
            let steps = st.request.steps.unwrap_or_else(|| {
                let records = u64::try_from(assembled.trained).unwrap_or(u64::MAX);
                let steps =
                    (records * u64::from(st.request.epochs)).div_ceil(u64::from(per_step.max(1)));
                u32::try_from(steps.max(1)).unwrap_or(u32::MAX)
            });
            let from = if st.request.continue_from_release {
                st.request.policy.clone()
            } else {
                arm(ctx.config(), None)
            };
            train(
                ctx,
                &TrainRequest {
                    datasets: vec![dataset.clone()],
                    rehearsal: st.rehearsal.clone().map(|dataset| Rehearse {
                        dataset,
                        share: st.request.rehearsal_share,
                    }),
                    from,
                    replay_fraction: st.request.replay_fraction,
                    steps: Some(steps),
                    rank: st.request.rank,
                    beta: None,
                    // Every record of a claim is trained: nothing is
                    // monitored, since monitoring sets families aside.
                    tuning: Tuning {
                        records_per_step: Some(per_step),
                        eval_every: st.request.tuning.eval_every.or(Some(0)),
                        ..st.request.tuning
                    },
                },
                st.trainer,
                &run.cancel_token(),
            )?
        }
    };
    let summary = value(&Trained {
        resume: Resume {
            dataset,
            candidate: candidate.candidate.clone(),
        },
        candidate: &candidate,
    })?;
    st.candidate = Some(candidate.clone());
    st.report.candidate = Some(candidate);
    Ok(StageEnd::done(summary))
}

fn release_stage(ctx: &Context, run: &mut Recorder<'_>, st: &mut State<'_>) -> Done {
    let Some(candidate) = st.candidate.clone() else {
        unreachable!("the release stage follows the train stage")
    };
    let store = ctx.releases();
    let champion = store
        .alias(&st.alias)?
        .map(|id| store.get(&id))
        .transpose()?;
    let claims = trained_claims(st);
    let gate = decide_claims(
        ctx,
        &candidate,
        champion.as_ref(),
        &claims,
        &st.kits,
        &st.request.gate,
        &run.cancel_token(),
    )?;
    let passed = gate.passed;
    let answered: Vec<ClaimId> = claims
        .iter()
        .filter(|(id, _)| st.pending.contains(id))
        .filter(|(id, _)| {
            gate.claims
                .iter()
                .any(|c| c.claim == id.to_string() && c.candidate.answered())
        })
        .map(|(id, _)| id.clone())
        .collect();
    if passed {
        st.report.release = Some(publish(
            ctx,
            &st.alias,
            &candidate,
            champion.as_ref(),
            gate.clone(),
        )?);
        st.report.absorbed = answered;
    }
    let summary = value(&gate)?;
    st.report.gate = Some(gate);
    Ok(if passed {
        StageEnd::done(summary)
    } else {
        StageEnd::stop(
            summary,
            format!(
                "the gate refused candidate {}: the current release stays in use",
                candidate.candidate
            ),
        )
    })
}

fn ledger_stage(ctx: &Context, _: &mut Recorder<'_>, st: &mut State<'_>) -> Done {
    let Some(release) = st.report.release.clone() else {
        unreachable!("the ledger stage follows a release")
    };
    let absorptions: Vec<Absorption> = st
        .report
        .absorbed
        .iter()
        .map(|claim| Absorption {
            claim: claim.clone(),
            release: release.clone(),
        })
        .collect();
    let recorded = ctx.claims().absorb(&absorptions)?;
    Ok(StageEnd::done(value(&(
        &release,
        recorded,
        &st.report.absorbed,
    ))?))
}

#[cfg(test)]
mod tests {
    /// The list a report names is the pipeline that runs.
    #[test]
    fn the_stage_list_is_the_pipeline() {
        assert_eq!(super::pipeline().names(), crate::absorb::STAGES);
    }
}
