// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training small models on the
// verified work of larger ones for its clients. If your team needs expertise
// in curating training datasets from agent trajectories then you can procure
// our services by sending an email to info@swedishembedded.com.

//! The grammar of `dataset build`.

use splinter_sdk::datasets::{StripPolicy, ViewName};
use splinter_sdk::seed::Seed;
use splinter_sdk::vocabulary::annotation::Strength;
use splinter_sdk::vocabulary::model_ref::ModelRef;

use super::{model_ref, positive_count, strength, strip, view, ControlKind};

/// `dataset build`: experience sets projected through a view into a stored
/// dataset.
#[derive(Debug, clap::Args)]
pub struct BuildArgs {
    /// The experience sets, by id or unique prefix.
    #[arg(required = true, value_name = "EXPERIENCE-SET")]
    pub sets: Vec<String>,
    /// sft-final, sft-step, critic, preference, verifier, decision,
    /// retrieval, outcome, denoise, cpt, voice or rehearsal.
    #[arg(long, required = true, value_parser = view, value_name = "VIEW")]
    pub view: ViewName,
    /// What the student sees: all (only the instruction), keep:K,..
    /// or mix:F.
    #[arg(long, value_parser = strip, value_name = "all|keep:K,..|mix:F")]
    pub strip: Option<StripPolicy>,
    /// The seed that picks which subjects keep their privileged context
    /// under `--strip mix:F` (drawn and printed when not given; a
    /// dataset built again with the same seed holds the same records).
    #[arg(long, value_name = "N", requires = "strip")]
    pub mix_seed: Option<Seed>,
    /// The weakest decision counted: executable, formal, consistency
    /// (default) or judged.
    #[arg(long, value_parser = strength, value_name = "STRENGTH")]
    pub min_strength: Option<Strength>,
    /// The system prompt every conversation opens with, in place of the one
    /// the runs were made under (the sft-step view opens a repository
    /// solve's records with the prompt its solve profile records) or the
    /// default assistant's; the dataset records it.
    #[arg(long, value_name = "TEXT")]
    pub system_prompt: Option<String>,
    /// Write an objective brain cannot train in the export format.
    #[arg(long)]
    pub export_only: bool,
    /// Keep at most this many records, evenly spread over the
    /// projection (for voice, over the parts of the sources).
    #[arg(long, value_name = "N")]
    pub limit: Option<usize>,
    /// The longest record, in tokens, the training that reads the
    /// dataset fits: a longer record is left out and counted instead of
    /// stopping the training.
    #[arg(long, value_name = "TOKENS", value_parser = positive_count)]
    pub max_block: Option<usize>,
    /// Split a record longer than `--max-block` into consecutive windows
    /// that each fit, keeping every supervised action, instead of leaving
    /// it out; every window repeats the task instruction. A window with no
    /// supervised turn is left out and counted as `NothingSupervised`.
    #[arg(long)]
    pub split_over_block: bool,
    /// For the sft-step view: one record per trajectory with every
    /// admitted action supervised, not one record per action.
    #[arg(long)]
    pub pack: bool,
    /// For the sft-step view: keep only the actions whose tool calls are all
    /// named here (comma separated), plus the final answer; the rest become
    /// context and are counted as `ActionNotWanted`. Names the decisive
    /// actions - the edits and the conclusion - instead of the exploration.
    #[arg(long, value_delimiter = ',', value_name = "TOOLS")]
    pub only_actions: Option<Vec<String>>,
    /// For the sft-step view: build another dataset than the default.
    /// `outcome-verified` admits only experiences an independent verdict
    /// passed (the default also admits an action labelled good from one that
    /// was not verified or failed; the manifest records `outcome-verified`).
    /// The negative controls: `unverified` keeps trajectories whatever their
    /// verdict (never an exposed one); `shuffled` gives each verified
    /// trajectory the instruction of another task.
    #[arg(long, value_enum)]
    pub control: Option<ControlKind>,
    /// The seed of the `shuffled` pairing (drawn and printed when not
    /// given).
    #[arg(long, value_name = "N", requires = "control")]
    pub control_seed: Option<Seed>,
    /// Leave out, and list in the report, every experience that would
    /// carry an exam family's attempt or file into the dataset, instead
    /// of refusing the build. What remains is sealed as usual.
    #[arg(long)]
    pub drop_leaking: bool,
    /// For the sft-step view: a trajectory exported as one agent step that holds
    /// a whole run's tool calls is read as one call per turn, each followed by
    /// its result, then the final reply.
    #[arg(long)]
    pub serialise_calls: bool,
    /// For the voice view: who the writer is. Requests are written as
    /// theirs and half the records open with their persona prompt, half
    /// with the line that says who the model is; refused with
    /// --system-prompt.
    #[arg(long, value_name = "NAME")]
    pub writer: Option<String>,
    /// For the voice view: keep at most this many tokens of the
    /// writer's text, evenly spread over all of it.
    #[arg(long, value_name = "N")]
    pub token_limit: Option<usize>,
    /// For the voice view: the most of the tokens one family may
    /// supply.
    #[arg(long, value_name = "SHARE")]
    pub max_family_share: Option<f64>,
    /// For the voice view: the model that writes, for each passage, a
    /// description of what it is about, which its request carries in
    /// place of the passage's opening (a passage with no admitted
    /// description keeps the opening).
    #[arg(long, value_parser = model_ref, value_name = "REF")]
    pub describe_with: Option<ModelRef>,
}
