// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The command line's grammar. Each command accepts exactly the options it
//! uses; defaults come from the crates that own them.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};
use splinter_campaign::attempt::{AttemptRequest, DEFAULT_MAX_ATTEMPTS, DEFAULT_TIMEOUT_SECS};
use splinter_campaign::facts::DEFAULT_HOLDOUT_ONE_IN;
use splinter_campaign::models::DEFAULT_CONTEXT_TOKENS;
use splinter_campaign::train::{DEFAULT_LORA_ALPHA, DEFAULT_LORA_RANK, DEFAULT_STEPS};
use splinter_campaign::ModelChoice;

/// A learning agent with its own model: run delegated tasks through sven,
/// turn verified experience and documents into training data, and train and
/// gate its own local model.
#[derive(Debug, Parser)]
#[command(
    name = "splinter",
    version,
    after_help = "State lives under ~/.sven/splinter/ (override: SPLINTER_STATE). Every run has a \
                  stable id; its manifest, trace, transcript, checkpoint and outcome survive the \
                  process that wrote them."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Delegate a task to the agent in a workspace and write a structured outcome.
    Run(AttemptArgs),
    /// Continue an interrupted run from its checkpoint, reconciling against its
    /// own trace before acting; refused once --max-attempts is spent.
    Resume(ResumeArgs),
    /// Print a run's manifest, outcome and trace index, or list every run.
    Show(ShowArgs),
    /// Stop a run in progress from another process; its attempt ends as
    /// `cancelled` with checkpoint and trace written, resumable.
    Cancel(RunId),
    /// Append a verified run's experience to the training pool (refuses a
    /// failed or unverified run).
    Learn(RunId),
    /// Fine-tune a LoRA adapter on the pool and promote it only when the
    /// held-out loss improved; exits 1 on rejection.
    Train(TrainArgs),
    /// Extract a question/answer training dataset from a markdown fact sheet.
    ///
    /// One JSONL record per fact. Every question must name a device the
    /// document's title names; one that does not is refused. For every fact a
    /// negative variant with an identifier from --scope-negatives trains the
    /// fixed abstention reply, so the adapter learns where its knowledge ends.
    Explore(ExploreArgs),
    /// Ask one question; prints only the parsed {"answer": ...} object and
    /// exits 2 when the reply is not strictly parseable.
    Ask(AskArgs),
    /// Ask the model every question in a facts dataset and score each reply
    /// against its reference; serves the promoted adapter by default.
    EvalFacts(EvalArgs),
    /// Learn a markdown fact sheet end to end: explore, split (1-in-N held
    /// out), fine-tune behind the held-out gate, and score recall and
    /// holdout; exits 1 when the gate rejected the adapter.
    Facts(FactsArgs),
}

/// The model a command runs on. Local and in-process unless a remote model
/// is named with the explicit opt-in.
#[derive(Debug, Args)]
pub struct ModelArgs {
    /// Local model checkpoint directory (default: $BRAIN_QWEN_WEIGHTS, else
    /// brain's model store default).
    #[arg(long)]
    pub local_weights: Option<PathBuf>,
    /// LoRA adapter file, or a promotion pointer, served on the local model.
    #[arg(long)]
    pub adapter: Option<PathBuf>,
    /// Inline context budget of the local model, in tokens.
    #[arg(long = "ctx", default_value_t = DEFAULT_CONTEXT_TOKENS)]
    pub context_tokens: u32,
    /// Permit a model reached over an API; without it every model call stays
    /// local and in-process.
    #[arg(long)]
    pub allow_api_models: bool,
    /// A REMOTE model, provider/name (e.g. openrouter/z-ai/glm-5.3-flash);
    /// needs --allow-api-models.
    #[arg(long = "model")]
    pub remote: Option<String>,
    /// OpenAI-compatible endpoint of the remote model.
    #[arg(long)]
    pub base_url: Option<String>,
    /// Key for that endpoint (default: AGENT_OPENROUTER_KEY for openrouter,
    /// BRAIN_API_KEY otherwise).
    #[arg(long)]
    pub api_key: Option<String>,
}

impl ModelArgs {
    pub fn choice(&self) -> ModelChoice {
        ModelChoice {
            local_weights: self.local_weights.clone(),
            adapter: self.adapter.clone(),
            context_tokens: self.context_tokens,
            remote: self.remote.clone(),
            base_url: self.base_url.clone(),
            api_key: self.api_key.clone(),
            allow_api_models: self.allow_api_models,
        }
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        self.choice().validate()
    }
}

/// One delegated attempt.
#[derive(Debug, Args)]
pub struct AttemptArgs {
    /// The workspace the agent works in.
    #[arg(long)]
    pub workspace: Option<PathBuf>,
    /// The task, as text.
    #[arg(long, conflicts_with = "task_file")]
    pub task: Option<String>,
    /// Read the task from a file instead of --task.
    #[arg(long)]
    pub task_file: Option<PathBuf>,
    /// Completion check the attempt runs itself after the turn (repeatable;
    /// the reviewer runs its own too).
    #[arg(long = "check")]
    pub checks: Vec<String>,
    #[command(flatten)]
    pub model: ModelArgs,
    /// Per-attempt wall-clock limit.
    #[arg(long, default_value_t = DEFAULT_TIMEOUT_SECS)]
    pub timeout_secs: u64,
    /// Per-attempt tool-round limit (default: sven's configuration).
    #[arg(long)]
    pub max_tool_rounds: Option<u32>,
    /// Per-attempt generated-token limit (default 100000).
    #[arg(long)]
    pub max_output_tokens: Option<u64>,
    /// Per-attempt billed-cost limit, remote models only (default 1.00 for
    /// openrouter/ models); a provider that reports usage without a price
    /// exhausts it.
    #[arg(long)]
    pub max_cost_usd: Option<f64>,
    /// Attempts per run, resumes included.
    #[arg(long, default_value_t = DEFAULT_MAX_ATTEMPTS)]
    pub max_attempts: u32,
    /// Capture the exact model input at the wire (needs --base-url naming a
    /// proxied upstream).
    #[arg(long)]
    pub record_input: bool,
    /// Print the outcome as JSON.
    #[arg(long)]
    pub json: bool,
}

impl AttemptArgs {
    pub fn into_request(self) -> anyhow::Result<AttemptRequest> {
        let task = match self.task_file {
            Some(path) => Some(
                std::fs::read_to_string(&path)
                    .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?,
            ),
            None => self.task,
        };
        Ok(AttemptRequest {
            workspace: self.workspace,
            task,
            checks: self.checks,
            model: self.model.choice(),
            timeout_secs: self.timeout_secs,
            max_tool_rounds: self.max_tool_rounds,
            max_output_tokens: self.max_output_tokens,
            max_cost_usd: self.max_cost_usd,
            max_attempts: self.max_attempts,
            record_input: self.record_input,
        })
    }
}

#[derive(Debug, Args)]
pub struct ResumeArgs {
    /// The run to continue.
    #[arg(long)]
    pub run: String,
    #[command(flatten)]
    pub attempt: AttemptArgs,
}

#[derive(Debug, Args)]
pub struct ShowArgs {
    /// The run to print; without it every run is listed.
    #[arg(long, conflicts_with = "list")]
    pub run: Option<String>,
    /// List every run (the default).
    #[arg(long)]
    pub list: bool,
}

#[derive(Debug, Args)]
pub struct RunId {
    /// The run.
    #[arg(long)]
    pub run: String,
}

/// LoRA training settings.
#[derive(Debug, Args)]
pub struct TrainingArgs {
    /// Training steps.
    #[arg(long, default_value_t = DEFAULT_STEPS)]
    pub steps: u32,
    /// LoRA rank of the new adapter.
    #[arg(long, default_value_t = DEFAULT_LORA_RANK)]
    pub rank: u32,
    /// LoRA alpha of the new adapter.
    #[arg(long, default_value_t = DEFAULT_LORA_ALPHA)]
    pub alpha: f32,
}

#[derive(Debug, Args)]
pub struct TrainArgs {
    /// Dataset to train on (default: the experience pool).
    #[arg(long)]
    pub dataset: Option<PathBuf>,
    /// Base checkpoint to fine-tune (default as for --local-weights).
    #[arg(long)]
    pub local_weights: Option<PathBuf>,
    #[command(flatten)]
    pub training: TrainingArgs,
}

#[derive(Debug, Args)]
pub struct ExploreArgs {
    /// The markdown fact sheet.
    #[arg(long)]
    pub file: PathBuf,
    /// The JSONL dataset to write.
    #[arg(long)]
    pub out: PathBuf,
    /// Section size cap in lines; a longer section is split at the next
    /// heading or paragraph boundary.
    #[arg(long)]
    pub chunk_lines: Option<usize>,
    /// Device identifiers outside the document's scope, comma-separated.
    #[arg(long, value_delimiter = ',')]
    pub scope_negatives: Vec<String>,
    #[command(flatten)]
    pub model: ModelArgs,
}

#[derive(Debug, Args)]
pub struct AskArgs {
    /// The question.
    #[arg(long)]
    pub question: String,
    /// Ask the untouched base model instead of the promoted adapter.
    #[arg(long)]
    pub base: bool,
    #[command(flatten)]
    pub model: ModelArgs,
}

#[derive(Debug, Args)]
pub struct EvalArgs {
    /// The facts dataset (JSONL) to evaluate.
    #[arg(long)]
    pub dataset: PathBuf,
    /// The JSON report to write.
    #[arg(long)]
    pub out: PathBuf,
    /// Shuffle the records first, so --limit samples rather than truncates.
    #[arg(long)]
    pub shuffle: bool,
    /// Evaluate at most this many records.
    #[arg(long)]
    pub limit: Option<usize>,
    /// Evaluate the untouched base model instead of the promoted adapter.
    #[arg(long)]
    pub base: bool,
    #[command(flatten)]
    pub model: ModelArgs,
}

#[derive(Debug, Args)]
pub struct FactsArgs {
    /// The markdown fact sheet to learn.
    #[arg(long)]
    pub file: Option<PathBuf>,
    /// Where artifacts and the report land (default: facts/ under the state root).
    #[arg(long)]
    pub work_dir: Option<PathBuf>,
    /// The extracted dataset; an existing one is reused instead of re-exploring.
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Hold out every Nth fact for the generalization score.
    #[arg(long, default_value_t = DEFAULT_HOLDOUT_ONE_IN)]
    pub holdout_one_in: usize,
    /// Section size cap in lines.
    #[arg(long)]
    pub chunk_lines: Option<usize>,
    /// Device identifiers outside the document's scope, comma-separated.
    #[arg(long, value_delimiter = ',')]
    pub scope_negatives: Vec<String>,
    #[command(flatten)]
    pub training: TrainingArgs,
    #[command(flatten)]
    pub model: ModelArgs,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn the_grammar_is_consistent() {
        Cli::command().debug_assert();
    }

    #[test]
    fn a_task_and_a_task_file_are_exclusive() {
        let parsed = Cli::try_parse_from(["splinter", "run", "--task", "a", "--task-file", "b"]);
        assert!(parsed.is_err());
    }

    #[test]
    fn scope_negatives_split_on_commas() {
        let cli = Cli::try_parse_from([
            "splinter",
            "explore",
            "--file",
            "sheet.md",
            "--out",
            "out.jsonl",
            "--scope-negatives",
            "STM32F103,STM32F030",
        ])
        .unwrap();
        let Command::Explore(args) = cli.command else {
            panic!("explore parses as explore");
        };
        assert_eq!(args.scope_negatives, ["STM32F103", "STM32F030"]);
    }
}
