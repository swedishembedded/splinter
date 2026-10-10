// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning that keeps accumulating
// capabilities generation after generation, for its clients. If your team
// needs expertise in continual learning that consolidates into the weights,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! The stages of `absorb` that keep a policy learning in generations: the
//! rebuild that takes a forgotten claim out of the weights that hold it, and
//! the consolidation a generation is due once retraining its adapter from the
//! base each night has become impractical.

use super::{value, Done, State};
use crate::absorb::consolidate::job::{self, Job, Params};
use crate::absorb::consolidate::plan;
use crate::absorb::generation::{consolidation_due, records_of, Generation};
use crate::absorb::kit;
use splinter_orchestrator::generation::ConsolidationMode;
use splinter_orchestrator::pipeline::StageEnd;
use splinter_orchestrator::runs::Recorder;
use splinter_orchestrator::{Context, OrchestratorError};

/// How a consolidation done inside `absorb` is carried out: the request's
/// rehearsal, gate and tuning, and its consolidation policy.
fn params(st: &State<'_>, mode: ConsolidationMode) -> Params {
    let policy = &st.request.consolidation;
    Params {
        mode,
        rehearsal_share: st.request.rehearsal_share,
        retention_replay_share: policy.retention_replay_share,
        epochs: policy.epochs,
        steps: policy.steps,
        tuning: crate::train::Tuning {
            learning_rate: policy.learning_rate,
            ..st.request.tuning
        },
        gate: st.request.gate,
        no_release: false,
    }
}

/// Takes the claims the base holds that are no longer live out of the
/// weights: trains again from the last base that did not hold them on the
/// claims consolidated since that are still live, and releases that
/// checkpoint in place of the release in use. The generation's own claims
/// are taught again on top of it by the rest of the night.
pub(super) fn rebuild_stage(ctx: &Context, run: &mut Recorder<'_>, st: &mut State<'_>) -> Done {
    let champion_id =
        st.report.policy.release.clone().ok_or_else(|| {
            OrchestratorError::Refused("a rebuild needs a release to replace".into())
        })?;
    let champion = ctx.releases().get(&champion_id)?;
    let planned = plan::plan(ctx, &champion, &st.live, Vec::new())?;
    let kits = kit::kept(ctx)?;
    let params = params(st, ConsolidationMode::Full);
    let done = job::run(
        ctx,
        &Job {
            alias: &st.alias,
            champion: &champion,
            replaces: &champion.id,
            from: &planned.from,
            train: &planned.train,
            retention: &planned.retention,
            rebuilt: true,
            dead: &planned.dead,
            kits: &kits,
            system: st.system.as_deref(),
            sealed: &st.sealed,
            params: &params,
            cancel: run.cancel_token(),
        },
        st.trainer,
    )?;
    let summary = value(&done)?;
    let refused = done.release.is_none();
    let why = done.stopped.clone();
    st.report.rebuilt = Some(done);
    if refused {
        return Ok(StageEnd::stop(
            summary,
            why.unwrap_or_else(|| "the rebuilt checkpoint was not released".into()),
        ));
    }
    // The rebuilt checkpoint is the base now; every claim of the generation
    // is taught to an adapter on it, whatever the old adapter answered.
    let release = ctx.releases().alias(&st.alias)?;
    st.report.policy.release.clone_from(&release);
    st.generation = Generation::of(ctx, &st.alias)?;
    st.report.generation = Some(st.generation.clone());
    st.split = st.generation.split(&st.live);
    st.pending = st.split.current.iter().map(|(id, _)| id.clone()).collect();
    st.report.pending = st.pending.iter().cloned().collect();
    Ok(StageEnd::done(summary))
}

/// Whether the generation's claims, after this night's release, hold more
/// records than retraining the adapter from the base each night is worth.
pub(super) fn consolidation_is_due(st: &State<'_>) -> bool {
    let Some(threshold) = st.request.consolidation.after_records else {
        return false;
    };
    st.report.release.is_some()
        && consolidation_due(records_of(&st.kits, &st.split.current), threshold)
}

/// Consolidates the generation the night's release closes: the adapter's
/// claims are taken into the weights, the next generation starts empty.
pub(super) fn consolidate_stage(ctx: &Context, run: &mut Recorder<'_>, st: &mut State<'_>) -> Done {
    let Some(release_id) = st.report.release.clone() else {
        unreachable!("the consolidate stage follows a release")
    };
    let release = ctx.releases().get(&release_id)?;
    let closing = plan::own_claims(ctx, &release, &st.live)?;
    let planned = plan::plan(ctx, &release, &st.live, closing)?;
    let params = params(st, st.request.consolidation.mode);
    let done = job::run(
        ctx,
        &Job {
            alias: &st.alias,
            champion: &release,
            replaces: &release.id,
            from: &planned.from,
            train: &planned.train,
            retention: &planned.retention,
            rebuilt: planned.rebuilt,
            dead: &planned.dead,
            kits: &st.kits,
            system: st.system.as_deref(),
            sealed: &st.sealed,
            params: &params,
            cancel: run.cancel_token(),
        },
        st.trainer,
    )?;
    let summary = value(&done)?;
    let why = done.stopped.clone();
    st.report.consolidation = Some(done);
    Ok(match why {
        Some(why) => StageEnd::stop(summary, why),
        None => StageEnd::done(summary),
    })
}
