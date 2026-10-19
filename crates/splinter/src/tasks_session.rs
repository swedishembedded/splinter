// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `tasks`: generate, vary, seed canaries into, pick from, list and show task
//! sets, as recorded runs where they write state.

use serde_json::json;
use splinter_sdk::canaries;
use splinter_sdk::runs::record;
use splinter_sdk::sources;
use splinter_sdk::tasks::pick::{pick, Quota};
use splinter_sdk::tasks::{self, check_kinds, resolve_set as resolve_task_set};
use splinter_sdk::variants;
use splinter_sdk::{Context, Error};

use crate::cli::{HarborCommand, OracleArg, TasksCommand};
use crate::output::emit;
use crate::session::Exit;
use splinter_sdk::harbor;
use splinter_sdk::knowledge::tasks::workspace::Oracle;

/// Runs a `tasks` command.
pub fn tasks(ctx: &Context, json: bool, command: TasksCommand) -> Result<Exit, Error> {
    match command {
        TasksCommand::Generate {
            sources: ids,
            kinds,
            generator,
            author,
            denoise_seed,
        } => {
            let kinds = check_kinds(&kinds)?;
            let ids = ids
                .iter()
                .map(|id| sources::resolve(ctx, id))
                .collect::<Result<Vec<_>, _>>()?;
            let arguments = json!({
                "sources": ids,
                "kinds": kinds,
                "generator": generator,
                "author": author,
                "denoise_seed": denoise_seed,
            });
            let generated = record(ctx, "tasks generate", &arguments, |run| {
                let denoise_seed = tasks::asks_for_denoise(&kinds)
                    .then(|| run.seed("denoise-seed", denoise_seed))
                    .transpose()?;
                tasks::generate(
                    ctx,
                    &tasks::Generation {
                        sources: &ids,
                        sections: &[],
                        kinds: &kinds,
                        generator: &generator,
                        goal: None,
                        author: author.as_deref(),
                        denoise_seed,
                        deadline: None,
                        cancel: run.cancel_token(),
                    },
                )
            })?;
            emit(json, &generated);
        }
        TasksCommand::Harbor(HarborCommand::Discover { root }) => {
            emit(json, &harbor::discover_harbor(&root)?);
        }
        TasksCommand::Harbor(HarborCommand::Import {
            directories,
            oracle,
        }) => {
            let oracle = match oracle {
                OracleArg::Skip => Oracle::Skip,
                OracleArg::Try => Oracle::Try,
                OracleArg::Require => Oracle::Require,
            };
            let arguments = json!({ "directories": directories, "oracle": format!("{oracle:?}") });
            let imported = record(ctx, "tasks harbor import", &arguments, |_| {
                harbor::import_harbor(ctx, &directories, oracle)
            })?;
            emit(json, &imported);
        }
        TasksCommand::Harbor(HarborCommand::Export {
            task_set,
            out,
            image,
        }) => {
            let set = resolve_task_set(ctx, &task_set)?;
            emit(json, &harbor::export_harbor(ctx, &set, &out, &image)?);
        }
        TasksCommand::Variants {
            task_set,
            generator,
            per_task,
        } => {
            let set = resolve_task_set(ctx, &task_set)?;
            let arguments =
                json!({ "task_set": set, "generator": generator, "per_task": per_task });
            let varied = record(ctx, "tasks variants", &arguments, |run| {
                variants::generate_variants(
                    ctx,
                    &variants::VariantsRequest {
                        task_set: &set,
                        generator: &generator,
                        per_task,
                        deadline: None,
                        cancel: run.cancel_token(),
                    },
                )
            })?;
            emit(json, &varied);
        }
        TasksCommand::Canaries { task_set, count } => {
            let set = resolve_task_set(ctx, &task_set)?;
            let arguments = json!({ "task_set": set, "count": count });
            let made = record(ctx, "tasks canaries", &arguments, |_| {
                canaries::make(ctx, &set, Some(count))
            })?;
            emit(json, &made);
        }
        TasksCommand::Pick {
            task_sets,
            family_share,
            per_family,
            seed,
        } => {
            let sets = task_sets
                .iter()
                .map(|id| resolve_task_set(ctx, id))
                .collect::<Result<Vec<_>, _>>()?;
            let quota = match (family_share, per_family) {
                (Some(share), _) => Quota::FamilyShare(share),
                (None, Some(cap)) => Quota::PerFamily(cap),
                (None, None) => unreachable!("clap requires one of the two quotas"),
            };
            let arguments = json!({
                "task_sets": sets,
                "family_share": family_share,
                "per_family": per_family,
                "seed": seed,
            });
            let picked = record(ctx, "tasks pick", &arguments, |_| {
                pick(ctx, &sets, quota, seed)
            })?;
            emit(json, &picked);
        }
        TasksCommand::List => emit(json, &tasks::list(ctx)?),
        TasksCommand::Show { id } => emit(json, &tasks::show(ctx, &id)?),
    }
    Ok(Exit::Ok)
}
