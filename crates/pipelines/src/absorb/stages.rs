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
use splinter_core::annotation::Strength;
use splinter_core::claim::{Absorption, ClaimId, TrainedClaims};
use splinter_core::model_ref::{ModelRef, POLICY_DEFAULT};
use splinter_core::role::Role;
use splinter_core::source::SourceId;
use splinter_core::terms::combine_stated;
use splinter_eval::claim_gate::StanceTally;
use splinter_knowledge::claims::task::claim_task;
use splinter_knowledge::claims::Ledger;
use splinter_orchestrator::pipeline::{FnStage, Pipeline, StageEnd};
use splinter_orchestrator::runs::{to_json, Recorder};
use splinter_orchestrator::{roles, Context, OrchestratorError};
use splinter_store::runs::{list_runs, RunStatus};

use super::dataset::assemble;
use super::gate::{decide_claims, publish, ClaimsDecision};
use super::generation::{records_of, replay_budget, retention_sample, Generation, Split};
use super::kit::{self, build_kits, ClaimKit, KitRequest, Live};
use super::sealed::SealedProbes;
use super::stance::{self, Derivation, Held};
use super::{AbsorbRequest, Absorbed};
use crate::claim_quality::load_subjects;
use crate::claims::{gate as rule_claims, resolve_set, GateRequest};
use crate::datasets::DEFAULT_MIN_STRENGTH;
use crate::learn::{records_at_share, PolicyUsed};
use crate::rehearsal::{rehearse_on, RehearseRequest, REHEARSAL_SEED};
use crate::release::arm_on;
use crate::train::{
    auto_records_per_step, load_candidate, train, Candidate, Rehearse, TrainRequest, Trainer,
    Tuning,
};
use crate::verify::adopt_judge;
use splinter_knowledge::claims::coverage::Subject;

mod generations;
mod intake;

use generations::{consolidate_stage, consolidation_is_due, rebuild_stage};
use intake::{extract_stage, intake_stage};

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
    if request.extraction_passes == 0 {
        return Err(OrchestratorError::Refused(
            "the extraction passes are at least one".into(),
        ));
    }
    if request.epochs == 0 || request.paraphrases == 0 {
        return Err(OrchestratorError::Refused(
            "epochs and paraphrases are at least one".into(),
        ));
    }
    if request.stance.tries == 0 {
        return Err(OrchestratorError::Refused(
            "the stance pass draws at least one reply per pair".into(),
        ));
    }
    if let Some(beta) = request.stance.beta {
        if !(beta.is_finite() && beta > 0.0) {
            return Err(OrchestratorError::Refused(format!(
                "the stance temperature {beta} is not a positive number"
            )));
        }
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
    subjects: Vec<Subject>,
    report: Absorbed,
    system: Option<String>,
    sessions: Vec<SourceId>,
    live: Vec<Live>,
    pending: BTreeSet<ClaimId>,
    kits: BTreeMap<ClaimId, ClaimKit>,
    min_strength: Strength,
    rehearsal: Option<String>,
    candidate: Option<crate::train::Candidate>,
    /// The stance pairs' dataset, when the stance stage made one.
    pairs: Option<String>,
    /// The situations the stance pairs were not made of.
    held: Option<Held>,
    previous: Option<Previous>,
    /// The generation the policy is in: its base and what the base holds.
    generation: Generation,
    /// The live claims split by whether the generation's base holds them.
    split: Split,
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
        subjects: load_subjects(&request.subjects)?,
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
        pairs: None,
        held: None,
        previous: previous_run(ctx, request, run.id()),
        generation: Generation::root(ctx),
        split: Split::default(),
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
        .then(
            FnStage::new("rebuild", rebuild_stage)
                .when(|s| !s.generation.dead(&s.live).is_empty() && !s.request.dry_run)
                .ignoring_budget(),
        )
        .then(FnStage::new("kits", kits_stage).ignoring_budget())
        .then(FnStage::new("dataset", dataset_stage).ignoring_budget())
        .then(
            FnStage::new("rehearse", rehearse_stage)
                .when(|s| s.request.rehearsal_share > 0.0)
                .ignoring_budget(),
        )
        .then(FnStage::new("train", train_stage).ignoring_budget())
        .then(
            FnStage::new("stance", stance_stage)
                .when(|s| s.request.stance.on())
                .ignoring_budget(),
        )
        .then(
            FnStage::new("preference", preference_stage)
                .when(|s| s.pairs.is_some())
                .ignoring_budget(),
        )
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
        .then(
            FnStage::new("consolidate", consolidate_stage)
                .when(consolidation_is_due)
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
    st.system = pin
        .as_ref()
        .and_then(|p| p.system_prompt.clone())
        .or_else(|| st.request.system_prompt.clone());
    st.report.policy = PolicyUsed {
        alias: st.alias.clone(),
        release: pin.map(|p| p.release),
    };
    st.generation = Generation::of(ctx, &st.alias)?;
    st.report.generation = Some(st.generation.clone());
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

fn gate_stage(ctx: &Context, _: &mut Recorder<'_>, st: &mut State<'_>) -> Done {
    let Some(extracted) = &st.report.extract else {
        unreachable!("the gate stage follows the extract stage")
    };
    let set = resolve_set(ctx, &extracted.claim_set.to_string())?;
    let gated = rule_claims(
        ctx,
        &GateRequest {
            judge: st.judge.as_ref(),
            entail: st.request.entail,
            subjects: &st.subjects,
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
    st.split = st.generation.split(&st.live);
    let stale = release_holds_dead_claim(ctx, st)? || !st.generation.dead(&st.live).is_empty();
    st.report.live = st.live.len();
    st.report.pending = st.pending.iter().cloned().collect();
    let summary = value(&gated)?;
    st.report.claims = Some(gated);
    Ok(if st.request.dry_run {
        StageEnd::stop(
            summary,
            "dry run: the claims are ruled on and nothing is taught",
        )
    } else if st.pending.is_empty() && !stale {
        StageEnd::stop(
            summary,
            "no live claim is new: every one is already absorbed",
        )
    } else {
        StageEnd::done(summary)
    })
}

/// Whether the release the policy runs was trained on a claim that is no
/// longer live (superseded or forgotten), so that a night must train again
/// from the base without it even when no claim is new.
fn release_holds_dead_claim(ctx: &Context, st: &State<'_>) -> Result<bool, OrchestratorError> {
    let Some(release) = &st.report.policy.release else {
        return Ok(false);
    };
    if st.request.continue_from_release {
        // Continuing cannot take a claim out of the weights.
        return Ok(false);
    }
    Ok(ctx.claims().trained_on(release)?.is_some_and(|trained| {
        trained
            .iter()
            .any(|claim| !st.live.iter().any(|(id, _)| id == claim))
    }))
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

/// The claims a night trains on: the generation's own that are live - the
/// base already holds the rest - or only the new ones when continuing a
/// release.
fn trained_claims(st: &State<'_>) -> Vec<Live> {
    st.split
        .current
        .iter()
        .filter(|(id, _)| !st.request.continue_from_release || st.pending.contains(id))
        .cloned()
        .collect()
}

/// The claims the gate asks about: the ones trained on, and the ones the
/// generation's base holds, which must keep being answered.
fn gated_claims(st: &State<'_>) -> Vec<Live> {
    trained_claims(st)
        .into_iter()
        .chain(st.split.retention.iter().cloned())
        .collect()
}

fn dataset_stage(ctx: &Context, _: &mut Recorder<'_>, st: &mut State<'_>) -> Done {
    let claims = trained_claims(st);
    // The records of the claims the base holds stay in the mix, as many as
    // keep them a share of the draws, so the adapter does not pull the base
    // away from them.
    let replayed = retention_sample(
        &st.kits,
        &st.split.retention,
        replay_budget(
            records_of(&st.kits, &claims),
            records_of(&st.kits, &st.split.retention),
            st.request.consolidation.retention_replay_share,
        ),
    );
    st.report.replayed_records = replayed.len();
    let Some(assembled) = assemble(
        ctx,
        &claims,
        &replayed,
        &st.kits,
        st.system.as_deref(),
        &st.sealed,
    )?
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
    let rehearsed = rehearse_on(
        ctx,
        &arm_on(ctx.config(), &st.generation.base, None),
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
                arm_on(ctx.config(), &st.generation.base, None)
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

/// The stance pairs of tonight, derived from the live state and the
/// supervised candidate, and the situations held out of them.
fn stance_stage(ctx: &Context, run: &mut Recorder<'_>, st: &mut State<'_>) -> Done {
    let Some(supervised) = st.candidate.clone() else {
        unreachable!("the stance stage follows the train stage")
    };
    let claims = trained_claims(st);
    let trained_records = st.report.dataset.as_ref().map_or(0, |d| d.trained);
    let student = arm_on(ctx.config(), &supervised.base, Some(&supervised.adapter));
    let derived = stance::derive(&Derivation {
        ctx,
        request: &st.request.stance,
        claims: &claims,
        sessions: &st.sessions,
        trained_records,
        policy: &st.request.policy,
        teacher: &st.teacher,
        generator: &st.generator,
        supervised: &student,
        system: st.system.as_deref(),
        sealed: &st.sealed,
        cancel: run.cancel_token(),
    })?;
    st.pairs = derived.built.dataset.as_ref().map(ToString::to_string);
    st.held = Some(derived.held);
    let summary = value(&derived.built)?;
    st.report.stance = Some(derived.built);
    Ok(StageEnd::done(summary))
}

/// The supervised candidate trained further on the pairs alone, against the
/// model it starts from; the result stands for the night's candidate, with
/// the datasets of both passes.
fn preference_stage(ctx: &Context, run: &mut Recorder<'_>, st: &mut State<'_>) -> Done {
    let (Some(supervised), Some(pairs)) = (st.candidate.clone(), st.pairs.clone()) else {
        unreachable!("the preference stage follows a stance stage that made pairs")
    };
    let preferred = train(
        ctx,
        &TrainRequest {
            datasets: vec![pairs],
            rehearsal: None,
            from: arm_on(ctx.config(), &supervised.base, Some(&supervised.adapter)),
            replay_fraction: st.request.replay_fraction,
            steps: None,
            rank: st.request.rank,
            beta: st.request.stance.beta,
            tuning: Tuning {
                records_per_step: None,
                eval_every: None,
                ..st.request.tuning
            },
        },
        st.trainer,
        &run.cancel_token(),
    )?;
    let candidate = after_preference(&supervised, preferred);
    let summary = value(&candidate)?;
    st.candidate = Some(candidate.clone());
    st.report.candidate = Some(candidate);
    st.report.supervised = Some(supervised);
    Ok(StageEnd::done(summary))
}

/// The candidate a preference pass made, as the night's: it was trained
/// from the supervised candidate, so it stands for both passes - their
/// datasets, the replay and rehearsal of the first, the terms of everything
/// either was made from.
fn after_preference(supervised: &Candidate, preferred: Candidate) -> Candidate {
    let terms = combine_stated([supervised.terms.as_ref(), preferred.terms.as_ref()]);
    Candidate {
        datasets: supervised
            .datasets
            .iter()
            .chain(&preferred.datasets)
            .cloned()
            .collect(),
        replay: supervised.replay.clone(),
        rehearsal: supervised.rehearsal.clone(),
        parent: supervised.parent.clone(),
        records: supervised.records + preferred.records,
        terms,
        ..preferred
    }
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
    let claims = gated_claims(st);
    let gate = decide_claims(
        ctx,
        &ClaimsDecision {
            candidate: &candidate,
            champion: champion.as_ref(),
            live: &claims,
            kits: &st.kits,
            held: st.held.as_ref(),
            config: &st.request.gate,
            cancel: &run.cancel_token(),
        },
    )?;
    st.report.stance_rates = gate
        .stance
        .as_ref()
        .and_then(|check| check.measured.as_ref())
        .map(StanceTally::summary);
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

/// The claims the release made tonight holds: the night's training set, and,
/// when it continued the champion, what the champion held.
fn trained_set(ctx: &Context, st: &State<'_>) -> Result<Vec<ClaimId>, OrchestratorError> {
    let mut claims: BTreeSet<ClaimId> = trained_claims(st).into_iter().map(|(id, _)| id).collect();
    if st.request.continue_from_release {
        if let Some(champion) = &st.report.policy.release {
            claims.extend(ctx.claims().trained_on(champion)?.unwrap_or_default());
        }
    }
    Ok(claims.into_iter().collect())
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
    ctx.claims().record_trained(&TrainedClaims {
        release: release.clone(),
        claims: trained_set(ctx, st)?,
    })?;
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
