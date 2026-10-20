// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The dataset: splinter's own build over the run's experiences.
//!
//! No writer of this sample's own: [`datasets::build`] projects the run's
//! experience sets through the `sft-step` view (one packed record per
//! trajectory, every action supervised, strict call and result pairing,
//! conservation of tool calls and results), puts the SERVING prompt in as the
//! system turn in place of the grounding prompt the agent ran under, and
//! writes `generic-messages-v2` that brain's own parser reads back before it
//! is stored. Only attempts the gate passed are admitted (the verdict is an
//! executable one); the negatives stay in the store and are counted as
//! exclusions.

use std::path::Path;

use anyhow::{bail, Context as _, Result};
use splinter_sdk::datasets::{build, BuildRequest, Built, ViewName};
use splinter_sdk::store::experiences::{ExperienceSet, SetId};
use splinter_sdk::vocabulary::annotation::Strength;
use splinter_sdk::vocabulary::digest::Digest;
use splinter_sdk::vocabulary::experience::ExperienceId;
use splinter_sdk::Context;

use crate::ledger;

/// The experience set of every attempt the ledger of `state` records as
/// stored (positives and negatives, of every run so far), put in the store
/// and named by its id. The ledger is the one record of what a state
/// directory holds, so a run that was interrupted still has its attempts
/// here; putting the same members again names the same set.
pub fn attempts_set(ctx: &Context, state: &Path) -> Result<String> {
    let members = ledger::read(state)?
        .iter()
        .filter_map(|e| e.experience.as_deref())
        .map(|hex| {
            Ok(ExperienceId(Digest::from_content_hex(hex).with_context(
                || format!("{hex:?} is not an experience id"),
            )?))
        })
        .collect::<Result<Vec<_>>>()?;
    if members.is_empty() {
        bail!("{} holds no stored attempt", state.display());
    }
    let set = ctx.experiences().put_set(&ExperienceSet {
        name: format!("repo-replay attempts in {}", state.display()),
        members,
    })?;
    Ok(set.to_string())
}

/// What the dataset is built with.
#[derive(Clone, Debug)]
pub struct DatasetSettings {
    /// The system prompt every record opens with: what the policy is served under.
    pub serving_prompt: String,
    /// Records longer than this many tokens are left out and counted.
    pub max_block: Option<usize>,
}

/// Builds the dataset of `sets` (each a set id as `run` printed it).
pub fn build_dataset(ctx: &Context, sets: &[String], settings: &DatasetSettings) -> Result<Built> {
    let ids = sets
        .iter()
        .map(|s| {
            Ok(SetId(
                Digest::parse(s).with_context(|| format!("{s:?} is not a set id"))?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let request = BuildRequest {
        sets: ids,
        view: ViewName::SftStep,
        strip: None,
        min_strength: Some(Strength::Executable),
        system_prompt: Some(settings.serving_prompt.clone()),
        export_only: false,
        limit: None,
        voice: Default::default(),
        max_block: settings.max_block,
        pack: true,
        control: Default::default(),
        serialise_calls: false,
    };
    Ok(build(ctx, &request)?)
}
