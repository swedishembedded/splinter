// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the measurement of fine-tuned language
// models for its clients. If your team needs expertise in pre-registering a
// replication study so that no choice can follow its results, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The `study` command family: the stages of the replication study as
//! subcommands, each run from the protocol frozen in the study directory.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::Context as _;
use clap::{Args, Subcommand};
use splinter_sdk::store::StateRoot;
use splinter_sdk::{Config, Splinter};

use super::amend::{amend_base_hold, BaseHold};
use super::corpus::{self, Staff};
use super::examine::{self, BrainLoader};
use super::layout::Study;
use super::served::ServedLoader;
use super::train;
use super::{render, report};
use crate::binding::Base;
use crate::models::{is_remote, Source};
use crate::transit::{Transit, DEFAULT_BACKOFF_SECS, DEFAULT_RETRIES};

use super::power::{derive, Observed};
use super::protocol::{
    freeze, Frozen, ModelRoles, Protocol, SampledDecoding, Seeds, Spec, DEFAULT_ALPHA_LEVEL,
    DEFAULT_ARTICLES, DEFAULT_AUTHORS, DEFAULT_DRAWS, DEFAULT_LEARNING_RATE, DEFAULT_POWER,
    DEFAULT_REPLICATES, DEFAULT_STEPS, DEFAULT_TEMPERATURE, DEFAULT_WORKERS, EARLIER_OBSERVATION,
    EARLIER_SOURCE,
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
    /// Change how the base is held while the adapters train, before any is:
    /// the protocol records the change, its reason and the digest it replaces,
    /// under which the corpora already written stay the study's.
    Amend(AmendArgs),
    /// Per author and training seed: train the adapters the protocol asks for
    /// on the records of the corpus stage, with the protocol's settings. A GPU
    /// stage; resumes, and leaves a trained adapter alone.
    Train(TrainStageArgs),
    /// Per author: put the frozen examination to every arm of the protocol.
    /// A GPU stage; items already answered are not asked again.
    Exam(ExamStageArgs),
    /// Read the study's manifests and outcomes and write the analysis: the
    /// paired tests per author and training seed, the spread over seeds, the
    /// sign test across authors with Holm's correction, the survivorship and
    /// reason measures, as CSV and JSON beside a short table. Asks no model.
    Report(ReportStageArgs),
}

/// Write the report.
#[derive(Args)]
pub struct ReportStageArgs {
    /// The study directory holding the frozen protocol.
    #[arg(long)]
    study: PathBuf,
    /// The seed of the bootstrap intervals; drawn and printed when absent,
    /// and recorded in the report.
    #[arg(long)]
    bootstrap_seed: Option<u64>,
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
    /// Put the examination to a `brain serve` started with this `brain`
    /// binary, one server per model, which decodes `--slots` items in one
    /// batch, instead of to an in-process model that answers one at a time.
    #[arg(long)]
    brain: Option<PathBuf>,
    /// Items a server decodes together, and so asks at once.
    #[arg(long, default_value_t = 16, requires = "brain")]
    slots: u32,
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
    #[arg(long, default_value_t = DEFAULT_RETRIES)]
    transport_retries: usize,
    /// Seconds to pause before the first repeat of a lost request; each repeat
    /// waits twice as long, up to a minute.
    #[arg(long, default_value_t = DEFAULT_BACKOFF_SECS)]
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
    /// How many articles the corpus stage writes at a time. A hosted model
    /// that rate-limits its callers needs fewer; the protocol records it.
    #[arg(long, default_value_t = DEFAULT_WORKERS)]
    workers: usize,
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
        template.corpus.workers = self.workers;
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
        Stage::Amend(amend_args) => amend_stage(amend_args),
        Stage::Train(train_args) => train_stage(train_args),
        Stage::Exam(exam_args) => exam_stage(exam_args),
        Stage::Report(report_args) => report_stage(report_args),
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
    for (role, reference) in [("reader", &roles.reader), ("miner", &roles.miner)] {
        anyhow::ensure!(
            is_remote(reference)?,
            "the {role} {reference} runs here: the gate reader and the miner are hosted models, whose per-request seeds are their provider's"
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
    let reader = Source::open(&ctx, &roles.reader, None)?;
    let miner = Source::open(&ctx, &roles.miner, None)?;
    let pace = Transit::new(
        args.transport_retries,
        Duration::from_secs(args.transport_backoff_secs),
    );
    for_each_author(&chosen_authors(&study, &args.only)?, |seed| {
        let run_seed = corpus::run_seed(&study, seed)?;
        let writer = Source::open(&ctx, &roles.writer, run_seed)?;
        let staff = Staff {
            writer: &writer,
            reader: &reader,
            miner: &miner,
        };
        let manifest = corpus::run(&study, seed, &staff, &pace, run_seed)?;
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
    if !p.confirmable() {
        text.push_str(&format!(
            "WARNING: with {} authors the smallest p-value a sign test can reach is {:.4}, above the level divided by the {} comparisons of the family: no comparison could be confirmed. Add authors or shrink the family.\n",
            p.authors.len(),
            p.smallest_p(),
            p.comparisons.len(),
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

/// The base checkpoint, hashed once for every author of the stage. Hashing
/// reads all of its weights, minutes for a large checkpoint, so it is said
/// before it starts.
fn open_base(path: &Path) -> anyhow::Result<Base> {
    eprintln!(
        "hashing the base checkpoint {} (a full read of its weights, once for every author)",
        path.display()
    );
    let base = Base::open(path)?;
    eprintln!("base {}", base.digest());
    Ok(base)
}

/// Arguments of `study amend`.
#[derive(Args)]
pub struct AmendArgs {
    /// The study directory holding the frozen protocol.
    #[arg(long)]
    study: PathBuf,
    /// How the base is to be held while the adapters train.
    #[arg(long, value_enum)]
    base_precision: BaseHold,
    /// Why; recorded in the protocol and reported with the study.
    #[arg(long)]
    reason: String,
}

fn amend_stage(args: &AmendArgs) -> anyhow::Result<()> {
    let study = Study::open(&args.study)?;
    let trained = study.any_trained()?;
    let amended = amend_base_hold(&args.study, args.base_precision, &args.reason, trained)?;
    if let Some(last) = amended.protocol.amendments.last() {
        println!(
            "protocol amended ({}): {}; it was {} and is {}",
            last.reason, last.change, last.replaces, amended.digest
        );
    }
    Ok(())
}

fn train_stage(args: &TrainStageArgs) -> anyhow::Result<()> {
    let study = Study::open(&args.study)?;
    let authors = chosen_authors(&study, &args.only)?;
    let base = open_base(&args.base)?;
    for_each_author(&authors, |seed| {
        let done = train::run(&study, seed, &base, &train::Brain)?;
        println!(
            "author {seed}: {} adapters trained, {} already trained",
            done.trained, done.already
        );
        Ok(())
    })
}

fn exam_stage(args: &ExamStageArgs) -> anyhow::Result<()> {
    let study = Study::open(&args.study)?;
    let authors = chosen_authors(&study, &args.only)?;
    let base = open_base(&args.base)?;
    let served;
    let in_process;
    let loader: &dyn examine::Loader = match &args.brain {
        Some(brain) => {
            let device = splinter_sdk::model::device::selected().source().to_string();
            served = ServedLoader {
                brain: brain.clone(),
                base: base.path().to_path_buf(),
                work: study.root().join("serve"),
                device: (device != "all").then_some(device),
                slots: args.slots,
            };
            &served
        }
        None => {
            in_process = BrainLoader { base: base.path() };
            &in_process
        }
    };
    for_each_author(&authors, |seed| {
        let examined = examine::run(&study, seed, &base, loader, args.sampled)?;
        for (slot, arm, items) in &examined.arms {
            println!("author {seed} {slot} {arm}: {items} items answered");
        }
        Ok(())
    })
}

fn report_stage(args: &ReportStageArgs) -> anyhow::Result<()> {
    let study = Study::open(&args.study)?;
    let report = report::run(&study, args.bootstrap_seed)?;
    let dir = render::write(study.root(), &report)?;
    print!("{}", render::table(&report));
    println!("\nwritten to {}", dir.display());
    Ok(())
}
