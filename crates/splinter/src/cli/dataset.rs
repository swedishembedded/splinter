// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training small models on the
// verified work of larger ones for its clients. If your team needs expertise
// in curating training datasets from agent trajectories then you can procure
// our services by sending an email to info@swedishembedded.com.

//! The grammar of `dataset build` and `dataset check`.

use splinter_sdk::datasets::AdvantageKind;
use splinter_sdk::vocabulary::model_ref::ModelRef;

use super::{advantage, model_ref};

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

/// `dataset verify-speech`.
#[derive(Debug, clap::Args)]
pub struct VerifySpeechArgs {
    /// The audio folder's metadata.jsonl: one row per recording, its
    /// `file_name` relative to this file's directory. Any format the brain
    /// server reads (WAV at any rate and channel count, Ogg/Opus, WebM/Opus).
    #[arg(value_name = "METADATA")]
    pub metadata: std::path::PathBuf,
    /// The column holding each recording's transcript.
    #[arg(long, default_value = "text", value_name = "COLUMN")]
    pub text_column: String,
    /// The recognition model, as the brain server lists it. Qwen3-ASR hears
    /// Swedish and English alike.
    #[arg(long, default_value = "brain/qwen3asr", value_name = "MODEL")]
    pub recognizer: String,
    /// At or above this similarity (letters and digits compared, spacing and
    /// punctuation ignored) a recording passes.
    #[arg(long, default_value_t = 0.90, value_name = "F")]
    pub pass_at: f64,
    /// Below this similarity a recording fails; between the two it is for
    /// review.
    #[arg(long, default_value_t = 0.60, value_name = "F")]
    pub fail_below: f64,
    /// The shortest hearing, over the transcript's length, that passes: a
    /// take cut short is below it.
    #[arg(long, default_value_t = 0.80, value_name = "F")]
    pub min_length_ratio: f64,
    /// The longest hearing, over the transcript's length, that passes: a take
    /// read twice is above it.
    #[arg(long, default_value_t = 1.25, value_name = "F")]
    pub max_length_ratio: f64,
    /// Hear every recording again, even one unchanged since this recogniser
    /// last heard it.
    #[arg(long)]
    pub rehear: bool,
}
