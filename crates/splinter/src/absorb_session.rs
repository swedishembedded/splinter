// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `session` and `claims`, as recorded runs where they write state.

use serde_json::json;
use splinter_sdk::absorb::{self, AbsorbRequest};
use splinter_sdk::claims::{self, ExtractRequest};
use splinter_sdk::knowledge::claims::extract::ExtractionPolicy;
use splinter_sdk::runs::record;
use splinter_sdk::sessions::{self, IntakeRequest};
use splinter_sdk::train::{Trainer, Tuning};
use splinter_sdk::vocabulary::role::Role;
use splinter_sdk::{Context, Error};

use crate::cli::{AbsorbArgs, ClaimsCommand, SessionCommand};
use crate::output::emit;
use crate::session::Exit;

/// Runs a `session` command.
pub fn session(ctx: &Context, json: bool, command: SessionCommand) -> Result<Exit, Error> {
    match command {
        SessionCommand::Add { paths, max_bytes } => {
            let arguments = json!({ "paths": paths, "max_bytes": max_bytes });
            let taken = record(ctx, "session add", &arguments, |_| {
                sessions::intake(
                    ctx,
                    &IntakeRequest {
                        paths: &paths,
                        max_bytes,
                    },
                )
            })?;
            emit(json, &taken);
            Ok(if taken.report.refused.is_empty() {
                Exit::Ok
            } else {
                Exit::Failed
            })
        }
        SessionCommand::List => {
            emit(json, &sessions::list(ctx)?);
            Ok(Exit::Ok)
        }
    }
}

/// Runs a `claims` command.
pub fn claims(ctx: &Context, json: bool, command: ClaimsCommand) -> Result<Exit, Error> {
    match command {
        ClaimsCommand::Extract {
            sessions: given,
            generator,
        } => {
            let ids = sessions::resolve(ctx, &given)?;
            let arguments = json!({ "sessions": ids, "generator": generator });
            let extracted = record(ctx, "claims extract", &arguments, |run| {
                claims::extract(
                    ctx,
                    &ExtractRequest {
                        sessions: &ids,
                        extractor: &generator,
                        policy: ExtractionPolicy::default(),
                        deadline: None,
                        cancel: run.cancel_token(),
                    },
                )
            })?;
            emit(json, &extracted);
            let report = &extracted.report;
            Ok(if report.failed.is_empty() && report.stopped.is_none() {
                Exit::Ok
            } else {
                Exit::Failed
            })
        }
        ClaimsCommand::Gate { claim_set } => {
            let set = claims::resolve_set(ctx, &claim_set)?;
            let gated = record(ctx, "claims gate", &json!({ "claim_set": set }), |_| {
                claims::gate(ctx, &set)
            })?;
            emit(json, &gated);
            Ok(Exit::Ok)
        }
        ClaimsCommand::List => {
            emit(json, &claims::list(ctx)?);
            Ok(Exit::Ok)
        }
        ClaimsCommand::Show { id } => {
            emit(json, &claims::show(ctx, &id)?);
            Ok(Exit::Ok)
        }
        ClaimsCommand::Ledger => {
            emit(json, &claims::ledger(ctx)?);
            Ok(Exit::Ok)
        }
    }
}

/// Runs `absorb`: one recorded run of every stage, which `--json` reports
/// whole.
pub fn absorb(
    ctx: &Context,
    trainer: &dyn Trainer,
    json: bool,
    args: AbsorbArgs,
) -> Result<Exit, Error> {
    let request = absorb_request(args);
    let done = absorb::absorb(ctx, &request, trainer)?;
    emit(json, &done);
    Ok(if done.report.finished() {
        Exit::Ok
    } else {
        Exit::Failed
    })
}

/// The `absorb` the command line asked for.
fn absorb_request(args: AbsorbArgs) -> AbsorbRequest {
    AbsorbRequest {
        sessions: args.sessions,
        policy: args.policy,
        roles: [
            (Role::Generator, args.generator),
            (Role::Teacher, args.teacher),
            (Role::Judge, args.judge),
        ]
        .into_iter()
        .filter_map(|(role, model)| model.map(|m| (role, m)))
        .collect(),
        paraphrases: args.paraphrases as usize,
        rehearsal_share: args.rehearsal_share,
        epochs: args.epochs,
        steps: args.steps,
        rank: args.rank,
        tuning: args.optimiser.applied_to(Tuning {
            bf16_base: args.bf16_base,
            ..Tuning::default()
        }),
        continue_from_release: args.continue_from_release,
        replay_fraction: args.replay_fraction,
        sealed_probes: args.sealed_probes,
        dry_run: args.dry_run,
        no_release: args.no_release,
        ..AbsorbRequest::default()
    }
}
