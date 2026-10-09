// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `session` and `claims`, as recorded runs where they write state.

use serde_json::json;
use splinter_sdk::claims::{self, ExtractRequest};
use splinter_sdk::knowledge::claims::extract::ExtractionPolicy;
use splinter_sdk::runs::record;
use splinter_sdk::sessions::{self, IntakeRequest};
use splinter_sdk::{Context, Error};

use crate::cli::{ClaimsCommand, SessionCommand};
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
