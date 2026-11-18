// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-set curation that projects
// verified agent experience into supervised datasets, for its clients. If
// your team needs expertise in dataset curation for fine-tuning, you can
// procure our services by sending an email to info@swedishembedded.com.

//! What one `dataset build` asks for.

use serde::Serialize;
use splinter_core::annotation::Strength;
use splinter_core::model_ref::ModelRef;
use splinter_data::curation::HistoryCuration;
use splinter_store::experiences::SetId;

use super::{Control, RlBuild, Strip, ViewName};
use crate::describe::DescribeRequest;

/// One `dataset build`.
#[derive(Clone, Debug, Serialize)]
pub struct BuildRequest {
    /// The experience sets, by id.
    pub sets: Vec<SetId>,
    /// The view.
    pub view: ViewName,
    /// What the student sees; `None` keeps the view's default.
    pub strip: Option<Strip>,
    /// The weakest decision counted; `None` is [`DEFAULT_MIN_STRENGTH`].
    pub min_strength: Option<Strength>,
    /// The system prompt every conversation opens with, in place of the
    /// default: a person's, for a policy to be trained as them. `None` keeps
    /// the default.
    pub system_prompt: Option<String>,
    /// Write an objective brain cannot train in the export format.
    pub export_only: bool,
    /// The most records, kept evenly spread over the projection
    /// ([`Projection::thinned_to`]); `None` keeps every one.
    pub limit: Option<usize>,
    /// How the writer's own text is written, for the voice view.
    pub voice: VoiceBuild,
    /// The longest record, in tokens, the training that reads the dataset
    /// fits ([`Projection::within_block`]): a longer one is left out and
    /// counted as [`Exclusion::OverBlock`] instead of being refused in the
    /// middle of a run. `None` keeps every record.
    pub max_block: Option<usize>,
    /// Split a chat record longer than `max_block` into consecutive windows
    /// that each fit ([`Projection::split_over_block`]) instead of leaving it
    /// out: every window repeats the instruction and keeps each supervised
    /// turn exactly once. Meaningless without `max_block`, which leaves the
    /// records unbounded.
    pub split_over_block: bool,
    /// For the sft-step view: one record per trajectory with every admitted
    /// action supervised ([`SftStep::packed`]), not one record per action.
    pub pack: bool,
    /// For the sft-step view: a negative control to build instead of the
    /// verified dataset ([`Control`]).
    pub control: Control,
    /// For the sft-step view: read a trajectory exported as one agent step as
    /// one tool call per turn ([`SftStep::serialise_calls`]).
    pub serialise_calls: bool,
    /// For the sft-step view: keep only the actions whose tool calls are all
    /// named here, plus every action that calls no tool (the final answer);
    /// the rest become context and are counted as `ActionNotWanted`
    /// ([`SftStep::only_actions`]).
    pub only_actions: Option<Vec<String>>,
    /// For the sft-step view: keep only the harness conversations answered by
    /// one of these model roles; every other experience is counted as
    /// `RoleNotWanted` ([`SftStep::only_roles`]).
    pub roles: Option<Vec<String>>,
    /// For the sft-step view: the training base reads images, so a harness
    /// conversation whose model was shown some is built from the requests it
    /// was sent, pictures in the turns that carried them ([`SftStep::images`]).
    /// Without it such a conversation is counted as `ShownImages`.
    pub images: bool,
    /// For the policy-gradient view: how an attempt's reward and its advantage
    /// are made.
    pub rl: RlBuild,
    /// For the sft-step view: what becomes of an action that reads the
    /// repository's history or git's raw storage
    /// (`splinter_data::curation`): by default it is not supervised.
    pub history: HistoryCuration,
    /// The seed the held-out and monitoring sets of the dataset are drawn by
    /// (every record carries it, with its stratum: `splinter_data::holdout`).
    /// `None` derives it from the experiences the records come from, so a
    /// dataset built again from the same experiences is split the same; the
    /// manifest records it either way.
    pub split_seed: Option<u64>,
    /// The model the training that reads the dataset starts from (`train
    /// --from`): the checkpoint whose tokenizer, chat template and vision
    /// configuration count the records of a `max_block` bound. `None` is the
    /// policy, which is also `train`'s default.
    #[serde(serialize_with = "as_text")]
    pub base: Option<ModelRef>,
}

impl BuildRequest {
    /// A build of `sets` through `view` with every other option at its
    /// default: the view's own strip and strength, the default system prompt,
    /// nothing exported, thinned, bounded or packed, the verified records, an
    /// action reading the repository's history dropped, the split seed derived
    /// from the experiences, the policy as the base.
    #[must_use]
    pub fn new(sets: Vec<SetId>, view: ViewName) -> Self {
        Self {
            sets,
            view,
            strip: None,
            min_strength: None,
            system_prompt: None,
            export_only: false,
            limit: None,
            voice: VoiceBuild::default(),
            max_block: None,
            split_over_block: false,
            pack: false,
            control: Control::Verified,
            serialise_calls: false,
            only_actions: None,
            roles: None,
            images: false,
            rl: RlBuild::default(),
            history: HistoryCuration::default(),
            split_seed: None,
            base: None,
        }
    }
}

fn as_text<S: serde::Serializer>(model: &Option<ModelRef>, out: S) -> Result<S::Ok, S::Error> {
    match model {
        Some(model) => out.collect_str(model),
        None => out.serialize_none(),
    }
}

/// How the voice view writes the writer's text into records; the other
/// views ignore it.
#[derive(Clone, Debug, Default, Serialize)]
pub struct VoiceBuild {
    /// Who the writer is: the requests are written as theirs and half the
    /// records are asked under their persona prompt, half under the line
    /// that says who the model is. Excludes `system_prompt`, which would
    /// replace both. `None` writes requests with no writer named, under
    /// `system_prompt` or the default.
    pub writer: Option<String>,
    /// The most tokens of the writer's text, kept evenly spread over it
    /// ([`Projection::thinned_to_tokens`]); `None` keeps all of it.
    pub token_budget: Option<usize>,
    /// The most of the tokens one family (the prints of one text, or one
    /// work) may supply; `None` is [`DEFAULT_MAX_FAMILY_SHARE`].
    pub max_family_share: Option<f64>,
    /// Ask each passage by what a model says it is about
    /// ([`crate::describe`]) instead of by its heading and opening; a
    /// passage that gets no admitted description keeps the latter. `None`
    /// asks every passage by its heading and opening.
    pub describe: Option<DescribeRequest>,
}
