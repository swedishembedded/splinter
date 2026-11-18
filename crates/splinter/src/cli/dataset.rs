// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training small models on the
// verified work of larger ones for its clients. If your team needs expertise
// in curating training datasets from agent trajectories then you can procure
// our services by sending an email to info@swedishembedded.com.

//! The grammar of `dataset build` and `dataset check`.

use splinter_sdk::datasets::{AdvantageKind, HistoryCuration, StripPolicy, ViewName};
use splinter_sdk::seed::Seed;
use splinter_sdk::vocabulary::annotation::Strength;
use splinter_sdk::vocabulary::model_ref::ModelRef;

use super::{
    advantage, model_ref, positive_count, reward_metric, strength, strip, view, ControlKind,
};

/// `dataset build`: experience sets projected through a view into a stored
/// dataset.
#[derive(Debug, clap::Args)]
pub struct BuildArgs {
    /// The experience sets, by id or unique prefix.
    #[arg(required = true, value_name = "EXPERIENCE-SET")]
    pub sets: Vec<String>,
    /// sft-final, sft-step, critic, preference, feedback, verifier, decision,
    /// retrieval, outcome, denoise, cpt, voice, rehearsal, policy-gradient or
    /// self-distill.
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
    /// stopping the training. Records are counted as the trainer counts
    /// them; the report says so, or that a floor stood in for it.
    #[arg(long, value_name = "TOKENS", value_parser = positive_count)]
    pub max_block: Option<usize>,
    /// The model the training that reads the dataset starts from, as
    /// `train --from` names it: the checkpoint whose tokenizer, chat template
    /// and vision configuration count the records of `--max-block`. Without
    /// it, the policy; give it when training another base, or the records are
    /// counted for the wrong model.
    #[arg(long, value_parser = model_ref, value_name = "REF", requires = "max_block")]
    pub from: Option<ModelRef>,
    /// Split a record longer than `--max-block` into consecutive windows
    /// that each fit, keeping every supervised action, instead of leaving
    /// it out; every window repeats the task instruction. A window with no
    /// supervised turn is left out and counted as `NothingSupervised`.
    #[arg(long)]
    pub split_over_block: bool,
    /// For the policy-gradient view: a metric the verifier measured, in [0, 1],
    /// and its weight, that earns a failed attempt part of the reward
    /// (`--reward-metric iou=0.5 --reward-metric feature_match=0.5`); repeat it
    /// per metric, the weights summing to one. A pass always earns 1. Without
    /// any, a failed attempt earns nothing. A failed attempt that did not
    /// record a named metric is an error.
    #[arg(long = "reward-metric", value_parser = reward_metric, value_name = "NAME=WEIGHT")]
    pub reward_metrics: Vec<(String, f64)>,
    /// For the policy-gradient view: the most a failed attempt earns from the
    /// reward metrics, below a pass's 1 (default 0.5), so that any pass
    /// outranks any failure.
    #[arg(long, value_name = "F", requires = "reward_metrics")]
    pub fail_ceiling: Option<f64>,
    /// For the policy-gradient view: how brain is to read an attempt's
    /// advantage from the rewards of the attempts at its task (the records
    /// carry rewards; the dataset's manifest names this for the trainer):
    /// `standardised` (GRPO's: the reward less the group mean over its
    /// standard deviation, the default), `mean-centred` (the reward less the
    /// group mean), `leave-one-out` (the reward less the mean of the others)
    /// or `positive` (standardised against the whole group, then clamped at
    /// zero by brain's `advantage_floor: zero`: every attempt sets the
    /// baseline and nothing is pushed down).
    #[arg(long, value_parser = advantage, value_name = "KIND")]
    pub advantage: Option<AdvantageKind>,
    /// For the policy-gradient view: distil the groups whose attempts all
    /// failed alike (the cliff, where GRPO's advantage is zero and teaches
    /// nothing) with a teacher context built from what the verifier knows, and
    /// keep the scalar advantage for every group with a spread of rewards. One
    /// build then holds both kinds of records, and the report counts the groups
    /// by kind. The `self-distill` view distils every attempt instead.
    #[arg(long)]
    pub cliff_distill: bool,
    /// For the policy-gradient view: DAPO's soft over-length penalty, the
    /// completion tokens (as the attempt recorded them) at and past which an
    /// attempt loses a whole reward. From `N - buffer` tokens on the penalty
    /// grows linearly from 0 to -1 at N; it is added to the reward a pass or
    /// failure earns, and the manifest records both thresholds. An attempt that
    /// recorded no completion tokens is an error. Without it, no length is
    /// penalised.
    #[arg(long, value_name = "N", value_parser = positive_count)]
    pub overlong_max_tokens: Option<usize>,
    /// The tokens before `--overlong-max-tokens` over which the penalty grows
    /// (default a quarter of it, DAPO's 4096 of 16384).
    #[arg(long, value_name = "N", requires = "overlong_max_tokens", value_parser = positive_count)]
    pub overlong_buffer: Option<usize>,
    /// For the preference view: the most pairs made of one group of attempts
    /// (one task version attempted by one policy) or of one repaired task
    /// (default 4). The candidate pairs are ordered by a hash of
    /// `--pair-seed` and the first that make a record are kept; the rest are
    /// counted as `OverLimit`. The manifest records the bound and the seed.
    #[arg(long, value_name = "N", value_parser = positive_count)]
    pub pairs_per_group: Option<usize>,
    /// For the preference view: the seed the pairs of a group are chosen by.
    /// Without it, the seed is derived from the experiences, so building again
    /// from them pairs the same.
    #[arg(long, value_name = "N")]
    pub pair_seed: Option<Seed>,
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
    /// For the sft-step, policy-gradient, preference and feedback views: keep
    /// only the conversations of a harness trial
    /// answered by these model roles (comma separated, as `ingest-bench`
    /// recorded them); every other experience, and a conversation whose role
    /// is unknown, is counted as `RoleNotWanted`.
    #[arg(long, value_delimiter = ',', value_name = "ROLES")]
    pub roles: Option<Vec<String>>,
    /// For the sft-step view: the training base reads images (a vision-language
    /// base such as Qwen3.8), so a harness conversation whose model was shown
    /// images is built from the requests it was sent, the pictures in the turns
    /// that carried them and only the answers supervised. Without it such a
    /// conversation is left out and counted as `ShownImages`, never trained on
    /// as text alone. A conversation ingested before requests were kept is
    /// counted as `RequestsNotRecorded` (ingest the trial again into a fresh
    /// store), and an action whose context holds more than 8 images as
    /// `TooManyImages`.
    #[arg(long)]
    pub images: bool,
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
    /// For the sft-step view (and the preference and feedback views, where
    /// `drop-step` and `truncate` both leave out a pair or completion that
    /// holds such a read): what becomes of an action whose tool call reads
    /// the repository's history (`git log`, `git show`, `git reflog`,
    /// `git cat-file`, a commit before `HEAD`) or git's storage (`.git/refs`,
    /// `.git/objects`, `.git/HEAD`): `drop-step` (the default) does not
    /// supervise it and keeps it as context, counted as `repository_history`;
    /// `truncate` ends the demonstration before it (what follows is counted as
    /// `after_repository_history`); `keep` learns it. A student imitating such
    /// reads learns to dig through the history. The manifest names the policy.
    #[arg(long, value_parser = history, value_name = "keep|drop-step|truncate")]
    pub repository_history: Option<HistoryCuration>,
    /// The seed the dataset's held-out and monitoring sets are drawn by:
    /// within each stratum (the domain a record's task exercises) its groups
    /// are ordered by a hash of the seed, so every stratum is held out at its
    /// share. Every record carries it and the manifest records it. Without
    /// it, the seed is derived from the experiences the records come from, so
    /// building again from them splits the same; another seed draws another
    /// split.
    #[arg(long, value_name = "N")]
    pub split_seed: Option<Seed>,
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

/// `dataset conversations`.
#[derive(Debug, clap::Args)]
pub struct ConversationsArgs {
    /// The input: one JSON object per line (`case`, `policy`, `trial`,
    /// `reward`, `record`).
    #[arg(value_name = "GRADED.JSONL")]
    pub input: std::path::PathBuf,
    /// The role prompt the learner played under, which opens every record.
    #[arg(long, required = true, value_name = "TEXT")]
    pub system_prompt: String,
    /// How brain is to read an attempt's advantage from the rewards of its
    /// case (`--advantage` of `dataset build`).
    #[arg(long, value_parser = advantage, value_name = "KIND")]
    pub advantage: Option<AdvantageKind>,
}

/// `dataset decision-fixtures`.
#[derive(Debug, clap::Args)]
pub struct DecisionFixturesArgs {
    /// The benchmark output directory (it holds `runs/<task>/<variant>/`).
    #[arg(value_name = "RUNS-DIR")]
    pub runs: std::path::PathBuf,
    /// The JSON-lines file to write.
    #[arg(long, required = true, value_name = "FILE")]
    pub out: std::path::PathBuf,
    /// Seed of the split; the same seed gives the same split. Drawn and
    /// printed when not given.
    #[arg(long, value_name = "N")]
    pub split_seed: Option<u64>,
    /// The share of each family's seeds held for validation.
    #[arg(long, default_value_t = 0.2, value_name = "F")]
    pub validation_fraction: f64,
    /// The share of each family's seeds held for test.
    #[arg(long, default_value_t = 0.2, value_name = "F")]
    pub test_fraction: f64,
    /// A whole task family that is test only (repeatable).
    #[arg(long = "holdout-suite", value_name = "FAMILY")]
    pub holdout_suites: Vec<String>,
}

/// `dataset check`: brain's own reading of a stored dataset (its
/// `check_dataset` over every record, read as chat records, preference pairs
/// or judged completions as the dataset's format is), for the base a run
/// trains and in the mode it would read the records (reasoning when they
/// carry it, one example per record when the dataset was built packed):
/// records and examples, the tokens and supervised tokens per example,
/// examples that train on nothing, the longest records and every record the
/// trainer refuses; for pairs and judged completions also the pairing policy
/// or the class weights the manifest records, and the tokenizers the
/// records' attempts name. Exits 1 when the trainer refuses any record or
/// the records name more than one tokenizer.
#[derive(Debug, clap::Args)]
pub struct CheckArgs {
    /// The dataset, by id or unique prefix.
    #[arg(value_name = "DATASET-ID")]
    pub id: String,
    /// The model the training starts from, as `train --from` names it: its
    /// base's tokenizer and chat template read the records. Without it, the
    /// policy.
    #[arg(long, value_parser = model_ref, value_name = "REF")]
    pub from: Option<ModelRef>,
    /// The training row to check against, in tokens: longer examples are
    /// counted. Without it, the widest row brain takes.
    #[arg(long, value_name = "TOKENS")]
    pub max_block: Option<u32>,
}

fn history(text: &str) -> Result<HistoryCuration, String> {
    HistoryCuration::parse(text)
}
