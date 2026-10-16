// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the measurement of fine-tuned language
// models for its clients. If your team needs expertise in pre-registering a
// replication study so that no choice can follow its results, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The `study` command family: the stages of the replication study as
//! subcommands, each run from the protocol frozen in the study directory.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::Context as _;
use clap::{Args, Subcommand};
use splinter_sdk::store::StateRoot;
use splinter_sdk::{Config, Splinter};

use super::corpus::{self, Pace, Staff};
use super::examine::{self, BrainLoader};
use super::layout::Study;
use super::train;
use crate::models::{is_remote, Source};
use crate::writer::{DEFAULT_TRANSPORT_BACKOFF_SECS, DEFAULT_TRANSPORT_RETRIES};

use super::power::{derive, Observed};
use super::protocol::{
    freeze, Frozen, ModelRoles, Protocol, SampledDecoding, Seeds, Spec, DEFAULT_ALPHA_LEVEL,
    DEFAULT_ARTICLES, DEFAULT_AUTHORS, DEFAULT_DRAWS, DEFAULT_LEARNING_RATE, DEFAULT_POWER,
    DEFAULT_REPLICATES, DEFAULT_STEPS, DEFAULT_TEMPERATURE, EARLIER_OBSERVATION, EARLIER_SOURCE,
};
use crate::manifest::SeedSource;

/// The stages of the replication study.
#[derive(Args)]
pub struct StudyArgs {
    /// The stage.
    #[command(subcommand)]
    pub stage: Stage,
}

/// One stage of the study.
#[derive(Subcommand)]
pub enum Stage {
    /// Write the protocol: draw the authors' and training seeds, size the
    /// examination, and record both, once. Repeating it with the same
    /// settings changes nothing; with others it is refused.
    Freeze(FreezeArgs),
    /// Per author: write the corpus with the protocol's writer and gate
    /// reader, read the decisions with its miner, make the training records
    /// and freeze and hash the examination, all before anything is trained.
    /// Resumes what exists; run it again after a stop.
    Corpus(CorpusStageArgs),
    /// Per author and training seed: train the adapters the protocol asks for
    /// on the records of the corpus stage, with the protocol's settings. A GPU
    /// stage; resumes, and leaves a trained adapter alone.
    Train(TrainStageArgs),
    /// Per author: put the frozen examination to every arm of the protocol.
    /// A GPU stage; items already answered are not asked again.
    Exam(ExamStageArgs),
}

/// Run the training stage.
#[derive(Args)]
pub struct TrainStageArgs {
    /// The study directory holding the frozen protocol.
    #[arg(long)]
    study: PathBuf,
    /// The checkpoint directory of the base model.
    #[arg(long)]
    base: PathBuf,
    /// The authors to run, by seed, comma separated; every author when absent.
    #[arg(long = "authors", value_delimiter = ',')]
    only: Vec<u64>,
}

/// Run the examination stage.
#[derive(Args)]
pub struct ExamStageArgs {
    /// The study directory holding the frozen protocol.
    #[arg(long)]
    study: PathBuf,
    /// The checkpoint directory of the base model.
    #[arg(long)]
    base: PathBuf,
    /// The authors to run, by seed, comma separated; every author when absent.
    #[arg(long = "authors", value_delimiter = ',')]
    only: Vec<u64>,
    /// Also ask the sampled decoding the protocol names, beside the greedy
    /// one.
    #[arg(long)]
    sampled: bool,
}

/// Run the corpus stage.
#[derive(Args)]
pub struct CorpusStageArgs {
    /// The study directory holding the frozen protocol.
    #[arg(long)]
    study: PathBuf,
    /// The authors to run, by seed, comma separated; every author of the
    /// protocol when absent.
    #[arg(long = "authors", value_delimiter = ',')]
    only: Vec<u64>,
    /// Permit the models, which are reached over the network.
    #[arg(long)]
    allow_remote: bool,
    /// How many times a request lost in transit is repeated before the model
    /// counts as unreachable and the run stops.
    #[arg(long, default_value_t = DEFAULT_TRANSPORT_RETRIES)]
    transport_retries: usize,
    /// Seconds to pause before the first repeat of a lost request; each repeat
    /// waits twice as long, up to a minute.
    #[arg(long, default_value_t = DEFAULT_TRANSPORT_BACKOFF_SECS)]
    transport_backoff_secs: u64,
}

/// Freeze the study protocol.
#[derive(Args)]
pub struct FreezeArgs {
    /// The study directory; created if absent.
    #[arg(long)]
    out: PathBuf,
    /// The model that writes the essays, as a model reference.
    #[arg(long)]
    writer: String,
    /// The model that reads each draft for the gate; another family.
    #[arg(long)]
    reader: String,
    /// The model that reads the decisions out of the essays; a third family.
    #[arg(long)]
    miner: String,
    /// How many authors to draw seeds for.
    #[arg(long, default_value_t = DEFAULT_AUTHORS, conflicts_with = "author_seeds")]
    authors: usize,
    /// The authors' seeds, comma separated, instead of drawing them.
    #[arg(long, value_delimiter = ',')]
    author_seeds: Option<Vec<u64>>,
    /// How many training seeds to draw for each author.
    #[arg(long, default_value_t = DEFAULT_REPLICATES, conflicts_with = "training_seeds")]
    training_seeds_per_author: usize,
    /// The training seeds every author is trained with, comma separated,
    /// instead of drawing them.
    #[arg(long, value_delimiter = ',')]
    training_seeds: Option<Vec<u64>>,
    /// Articles planned for each author's corpus.
    #[arg(long, default_value_t = DEFAULT_ARTICLES)]
    articles: usize,
    /// Optimizer steps of each adapter.
    #[arg(long, default_value_t = DEFAULT_STEPS)]
    steps: u32,
    /// The learning rate of each adapter.
    #[arg(long, default_value_t = DEFAULT_LEARNING_RATE)]
    learning_rate: f64,
    /// Times each item is asked under the sampled decoding; 0 leaves the
    /// sampled decoding out.
    #[arg(long, default_value_t = DEFAULT_DRAWS)]
    sampled_draws: usize,
    /// The temperature of the sampled decoding.
    #[arg(long, default_value_t = DEFAULT_TEMPERATURE)]
    temperature: f64,
    /// Items an earlier examination asked both arms of the comparison the
    /// examination is sized for.
    #[arg(long, default_value_t = EARLIER_OBSERVATION.items)]
    observed_items: usize,
    /// Of those, the items on which exactly one arm was right.
    #[arg(long, default_value_t = EARLIER_OBSERVATION.discordant)]
    observed_discordant: usize,
    /// Of those, the items the candidate was right on.
    #[arg(long, default_value_t = EARLIER_OBSERVATION.candidate_wins)]
    observed_wins: usize,
    /// The level of the one-sided tests the examination is sized for.
    #[arg(long, default_value_t = DEFAULT_ALPHA_LEVEL)]
    alpha: f64,
    /// The power the examination is sized for.
    #[arg(long, default_value_t = DEFAULT_POWER)]
    power: f64,
}

impl FreezeArgs {
    fn spec(&self) -> anyhow::Result<Spec> {
        let sizing = derive(
            Observed {
                items: self.observed_items,
                discordant: self.observed_discordant,
                candidate_wins: self.observed_wins,
            },
            EARLIER_SOURCE,
            self.alpha,
            self.power,
        )?;
        let mut template = Protocol::standard(
            ModelRoles {
                writer: self.writer.clone(),
                reader: self.reader.clone(),
                miner: self.miner.clone(),
            },
            sizing,
        );
        template.corpus.articles = self.articles;
        template.training.steps = self.steps;
        template.training.learning_rate = self.learning_rate;
        template.decoding.secondary = (self.sampled_draws > 0).then_some(SampledDecoding {
            temperature: self.temperature,
            draws: self.sampled_draws,
        });
        Ok(Spec {
            template,
            author_seeds: self
                .author_seeds
                .clone()
                .map_or(Seeds::Draw(self.authors), Seeds::Given),
            training_seeds: self
                .training_seeds
                .clone()
                .map_or(Seeds::Draw(self.training_seeds_per_author), Seeds::Given),
        })
    }
}

/// Runs the stage `args` names.
///
/// # Errors
/// The stage's own: see each stage.
pub fn run(args: &StudyArgs) -> anyhow::Result<()> {
    match &args.stage {
        Stage::Freeze(freeze_args) => freeze_stage(freeze_args),
        Stage::Corpus(corpus_args) => corpus_stage(corpus_args),
        Stage::Train(train_args) => train_stage(train_args),
        Stage::Exam(exam_args) => exam_stage(exam_args),
    }
}

/// The authors a stage runs: the ones asked for, else all of the protocol's.
fn chosen_authors(study: &Study, only: &[u64]) -> anyhow::Result<Vec<u64>> {
    for &seed in only {
        study.author_plan(seed)?;
    }
    Ok(if only.is_empty() {
        study.protocol().authors.iter().map(|a| a.seed).collect()
    } else {
        only.to_vec()
    })
}

/// Runs `stage` for each of `authors`, going on to the next when one fails,
/// and fails at the end naming every author that did.
fn for_each_author(
    authors: &[u64],
    mut stage: impl FnMut(u64) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    let failed: Vec<String> = authors
        .iter()
        .filter_map(|&seed| stage(seed).err().map(|e| format!("author {seed}: {e:#}")))
        .collect();
    anyhow::ensure!(failed.is_empty(), "{}", failed.join("\n"));
    Ok(())
}

fn corpus_stage(args: &CorpusStageArgs) -> anyhow::Result<()> {
    let study = Study::open(&args.study)?;
    let roles = study.protocol().models.clone();
    for (role, reference) in [
        ("writer", &roles.writer),
        ("reader", &roles.reader),
        ("miner", &roles.miner),
    ] {
        anyhow::ensure!(
            is_remote(reference)?,
            "the {role} {reference} runs here: the corpus stage asks hosted models, whose per-request seeds are their provider's"
        );
    }
    let mut config = Config::from_env();
    config.state_root = StateRoot::new(study.root().join("state"));
    config.thinking = false;
    config.allow_remote = args.allow_remote;
    let splinter = Splinter::builder(config)
        .build()
        .context("starting the runtime")?;
    let ctx = splinter.context();
    let writer = Source::open(&ctx, &roles.writer, None)?;
    let reader = Source::open(&ctx, &roles.reader, None)?;
    let miner = Source::open(&ctx, &roles.miner, None)?;
    let staff = Staff {
        writer: &writer,
        reader: &reader,
        miner: &miner,
    };
    let pace = Pace {
        transport_retries: args.transport_retries,
        transport_backoff: Duration::from_secs(args.transport_backoff_secs),
    };
    for_each_author(&chosen_authors(&study, &args.only)?, |seed| {
        let manifest = corpus::run(&study, seed, &staff, &pace)?;
        println!(
            "author {seed} ({}): {} of {} articles kept, {} decisions kept, examination of {} items frozen as {}",
            manifest.author_name,
            manifest.corpus.articles_kept,
            manifest.corpus.articles_planned,
            manifest.corpus.decisions_kept,
            manifest.exam.items,
            manifest.exam.digest,
        );
        Ok(())
    })
}

fn freeze_stage(args: &FreezeArgs) -> anyhow::Result<()> {
    let frozen = freeze(&args.out, &args.spec()?, &mut || {
        splinter_sdk::model::seed::draw().value()
    })?;
    print!("{}", describe(&frozen, &args.out));
    Ok(())
}

/// What the freeze prints: the digest, and every seed it drew or was given.
fn describe(frozen: &Frozen, root: &std::path::Path) -> String {
    let p = &frozen.protocol;
    let mut text = format!(
        "protocol {} in {} (digest {})\n",
        if frozen.created {
            "frozen"
        } else {
            "already frozen"
        },
        root.display(),
        frozen.digest,
    );
    text.push_str(&format!(
        "author seeds ({}) and training seeds ({}), recorded in the protocol:\n",
        origin(p.provenance.authors),
        origin(p.provenance.training),
    ));
    for author in &p.authors {
        let seeds: Vec<String> = author.training_seeds.iter().map(u64::to_string).collect();
        text.push_str(&format!(
            "  author seed {}: training seeds {}\n",
            author.seed,
            seeds.join(", ")
        ));
    }
    let sizing = &p.exam.sizing;
    text.push_str(&format!(
        "examination: {} items per kind ({} per author), power {:.3} at one-sided {} against {} of {} discordant, {:.1} expected pairs\n",
        p.exam.per_cell,
        sizing.items,
        sizing.achieved_power,
        sizing.alpha,
        sizing.observed.candidate_wins,
        sizing.observed.discordant,
        sizing.expected_discordant,
    ));
    text
}

fn origin(source: SeedSource) -> &'static str {
    match source {
        SeedSource::Drawn => "drawn",
        SeedSource::Given => "given",
    }
}

fn train_stage(args: &TrainStageArgs) -> anyhow::Result<()> {
    let study = Study::open(&args.study)?;
    for_each_author(&chosen_authors(&study, &args.only)?, |seed| {
        let done = train::run(&study, seed, &args.base, &train::Brain)?;
        println!(
            "author {seed}: {} adapters trained, {} already trained",
            done.trained, done.already
        );
        Ok(())
    })
}

fn exam_stage(args: &ExamStageArgs) -> anyhow::Result<()> {
    let study = Study::open(&args.study)?;
    let loader = BrainLoader { base: &args.base };
    for_each_author(&chosen_authors(&study, &args.only)?, |seed| {
        let examined = examine::run(&study, seed, &args.base, &loader, args.sampled)?;
        for (slot, arm, items) in &examined.arms {
            println!("author {seed} {slot} {arm}: {items} items answered");
        }
        Ok(())
    })
}
