// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-set curation that projects
// verified agent experience into supervised datasets, for its clients. If
// your team needs expertise in dataset curation for fine-tuning, you can
// procure our services by sending an email to info@swedishembedded.com.

//! `dataset build`: the grammar's arguments as a build request, run as a
//! recorded run that states the seeds it drew.

use serde_json::json;
use splinter_sdk::agent::CancelToken;
use splinter_sdk::datasets::{
    build, build_dropping_leaks, BuildRequest, Control, RlBuild, VoiceBuild,
};
use splinter_sdk::describe::DescribeRequest;
use splinter_sdk::experiences::resolve_set;
use splinter_sdk::runs::record;
use splinter_sdk::{Context, Error};

use crate::cli::{BuildArgs, ControlKind};
use crate::output::emit;

/// Builds the dataset `args` ask for and prints the report.
pub fn run(ctx: &Context, json: bool, args: BuildArgs) -> Result<(), Error> {
    let BuildArgs {
        sets,
        view,
        strip,
        mix_seed,
        min_strength,
        system_prompt,
        export_only,
        limit,
        max_block,
        from,
        split_over_block,
        pack,
        control,
        control_seed,
        repository_history,
        split_seed,
        drop_leaking,
        serialise_calls,
        only_actions,
        roles,
        images,
        writer,
        token_limit,
        max_family_share,
        describe_with,
        reward,
        practice,
        reward_metrics,
        fail_ceiling,
        advantage,
        cliff_distill,
        overlong_max_tokens,
        overlong_buffer,
        pairs_per_group,
        pair_seed,
    } = args;
    let request = BuildRequest {
        sets: sets
            .iter()
            .map(|id| resolve_set(ctx, id))
            .collect::<Result<_, _>>()?,
        view,
        strip: None,
        min_strength,
        system_prompt,
        export_only,
        limit,
        voice: VoiceBuild {
            writer,
            token_budget: token_limit,
            max_family_share,
            describe: describe_with.map(|generator| DescribeRequest {
                generator,
                deadline: None,
                cancel: CancelToken::new(),
            }),
        },
        max_block,
        split_over_block,
        pack,
        control: Control::Verified,
        serialise_calls,
        only_actions,
        roles,
        images,
        rl: RlBuild {
            reward: reward.unwrap_or_default(),
            reward_metrics,
            fail_ceiling,
            advantage: advantage.unwrap_or_default(),
            cliff_distill,
            overlong_max_tokens: overlong_max_tokens.map(|n| n as u64),
            overlong_buffer: overlong_buffer.map(|n| n as u64),
            pairs_per_group,
            pair_seed: pair_seed.map(splinter_sdk::seed::Seed::value),
        },
        practice,
        history: repository_history.unwrap_or_default(),
        split_seed: split_seed.map(splinter_sdk::seed::Seed::value),
        base: from,
    };
    // The strip is settled inside the run, which states the seed
    // `mix:F` picks its subjects by.
    let arguments = json!({
        "request": &request, "strip": &strip, "control": &control, "control_seed": &control_seed,
    });
    emit(
        json,
        &record(ctx, "dataset build", &arguments, |run| {
            let strip = strip
                .map(|policy| policy.with_seed(|| run.seed("mix-seed", mix_seed)))
                .transpose()?;
            let control = match control {
                None => Control::Verified,
                Some(ControlKind::OutcomeVerified) => Control::OutcomeVerified,
                Some(ControlKind::Unverified) => Control::Unverified,
                Some(ControlKind::Shuffled) => Control::ShuffledInstructions {
                    seed: run.seed("control-seed", control_seed)?.value(),
                },
            };
            let request = BuildRequest {
                strip,
                control,
                ..request.clone()
            };
            if drop_leaking {
                build_dropping_leaks(ctx, &request)
            } else {
                build(ctx, &request)
            }
        })?,
    );
    Ok(())
}
