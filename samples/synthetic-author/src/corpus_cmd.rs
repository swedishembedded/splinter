// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled validation of learning systems
// against ground truth that cannot have been seen in pretraining, for its
// clients. If your team needs expertise in proving that a trained model has
// learned what a corpus implies, you can procure our services by sending an
// email to info@swedishembedded.com.

//! The `corpus` command: draw or read back the author, record how the corpus
//! is made, and write it.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::Context as _;
use clap::Args;
use splinter_sdk::store::StateRoot;
use splinter_sdk::{Config, Splinter};

use crate::cache::Cache;
use crate::family::{check_distinct, Roles};
use crate::manifest::{
    begin_or_continue, code_revision, corpus_author, run_seed as run_seed_of, Author,
    CorpusManifest, Limits, Recipe, SEED_DERIVATION,
};
use crate::models::{is_remote, Source, WRITER_DECODING};
use crate::writer::{
    write_corpus_read, Models, Options, Summary, DEFAULT_TRANSPORT_BACKOFF_SECS,
    DEFAULT_TRANSPORT_RETRIES,
};

/// Write the author's corpus into a cache directory.
#[derive(Args)]
pub struct CorpusArgs {
    /// The seed the author is drawn from; a fresh one is drawn and printed
    /// when none is given and the directory records no author.
    #[arg(long, conflicts_with = "opposite_of")]
    seed: Option<u64>,
    /// Write the corpus of the author who holds every belief of the one this
    /// seed draws the other way, set in the same world. Refused, with the
    /// reason, when that author cannot be checked against.
    #[arg(long)]
    opposite_of: Option<u64>,
    /// The seed the writer's and reader's per-request seeds are derived from;
    /// drawn and printed when absent. Read back from the manifest when the
    /// corpus is continued.
    #[arg(long)]
    run_seed: Option<u64>,
    /// The directory the corpus is kept in (outside the repository).
    #[arg(long)]
    cache: PathBuf,
    /// The model that writes, as a model reference.
    #[arg(long)]
    writer: String,
    /// The model that reads each article and says what was decided; the
    /// marker words judge when none is named.
    #[arg(long)]
    reader: Option<String>,
    /// Permit a writer reached over the network.
    #[arg(long)]
    allow_remote: bool,
    /// Permit the writer and the reader to be of the same model family, which
    /// the protocol otherwise refuses; the manifest records that it was
    /// permitted.
    #[arg(long)]
    allow_same_family: bool,
    /// How many articles the corpus has.
    #[arg(long, default_value_t = 100)]
    articles: usize,
    /// How many episodes each article tells.
    #[arg(long, default_value_t = 3)]
    episodes_per_article: usize,
    /// The fewest words an article may have.
    #[arg(long, default_value_t = 700)]
    min_words: usize,
    /// How many times an article is asked for before it is given up on.
    #[arg(long, default_value_t = 3)]
    attempts: usize,
    /// The most requests this run may make.
    #[arg(long)]
    max_calls: usize,
    /// The most output tokens this run may write (estimated).
    #[arg(long)]
    max_output_tokens: u64,
    /// How many articles are written at a time.
    #[arg(long, default_value_t = 1)]
    workers: usize,
    /// How many times a request that fails in transit is repeated before the
    /// model counts as unreachable and the run stops.
    #[arg(long, default_value_t = DEFAULT_TRANSPORT_RETRIES)]
    transport_retries: usize,
    /// Seconds to pause before the first repeat of a lost request; each repeat
    /// waits twice as long, up to a minute.
    #[arg(long, default_value_t = DEFAULT_TRANSPORT_BACKOFF_SECS)]
    transport_backoff_secs: u64,
}

/// What a corpus is begun with, apart from the models that write and read.
#[derive(Clone, Debug)]
pub struct Plan {
    /// The model that writes, as a model reference.
    pub writer: String,
    /// The model that reads each article, if one is named.
    pub reader: Option<String>,
    /// Whether the writer and the reader may be of one family.
    pub allow_same_family: bool,
    /// How many articles the corpus has.
    pub articles: usize,
    /// How many episodes each article tells.
    pub episodes_per_article: usize,
    /// The fewest words an article may have.
    pub min_words: usize,
    /// Attempts at an article before it is given up on.
    pub attempts: usize,
    /// The most requests this run may make.
    pub max_calls: usize,
    /// The most output tokens this run may write (estimated).
    pub max_output_tokens: u64,
    /// How many articles are written at a time.
    pub workers: usize,
    /// How many times a request lost in transit is repeated.
    pub transport_retries: usize,
    /// The pause before the first repeat; it doubles up to a minute.
    pub transport_backoff: Duration,
}

/// The author and the run seed a corpus is written under, decided but not
/// yet recorded.
#[derive(Clone, Debug)]
pub struct Prepared {
    /// The author the corpus is written for.
    pub author: Author,
    /// The seed the per-request seeds derive from, when a model here runs.
    pub run_seed: Option<u64>,
}

impl Plan {
    /// Decides the author and the run seed of the corpus in `dir`: the ones it
    /// records, else `seed` (or the author `opposite_of` opposes) and `run_seed`,
    /// else fresh ones, printed. Writes nothing, so a model that then cannot
    /// be had leaves no manifest behind.
    ///
    /// # Errors
    /// The writer and the reader are of one family and that is not allowed, or
    /// the seeds contradict what the directory records.
    pub fn prepare(
        &self,
        dir: &Path,
        seed: Option<u64>,
        opposite_of: Option<u64>,
        run_seed: Option<u64>,
    ) -> anyhow::Result<Prepared> {
        if !self.allow_same_family {
            check_distinct(&Roles {
                writer: &self.writer,
                reader: self.reader.as_deref(),
                miner: None,
            })?;
        }
        let author = corpus_author(dir, seed, opposite_of)?;
        let any_local = !is_remote(&self.writer)?
            || self.reader.as_deref().map(is_remote).transpose()? == Some(false);
        Ok(Prepared {
            author,
            run_seed: run_seed_of(dir, run_seed, any_local)?,
        })
    }

    /// Records how the corpus in `dir` is made, or checks that it is being
    /// continued the way it was begun, and returns the options to write with.
    ///
    /// # Errors
    /// The recipe differs from the one recorded.
    pub fn begin(&self, dir: &Path, prepared: &Prepared) -> anyhow::Result<Options> {
        begin_or_continue(
            dir,
            CorpusManifest {
                seed_source: prepared.author.seed_source,
                recipe: Recipe {
                    author_seed: prepared.author.seed,
                    persona_source: prepared.author.persona_source,
                    writer: self.writer.clone(),
                    reader: self.reader.clone(),
                    allow_same_family: self.allow_same_family,
                    decoding: WRITER_DECODING.to_string(),
                    run_seed: prepared.run_seed,
                    articles: self.articles,
                    episodes_per_article: self.episodes_per_article,
                    min_words: self.min_words,
                },
                seed_derivation: SEED_DERIVATION.to_string(),
                limits: Limits {
                    attempts: self.attempts,
                    max_calls: self.max_calls,
                    max_output_tokens: self.max_output_tokens,
                    workers: self.workers,
                },
                code_revision: code_revision().to_string(),
            },
        )?;
        Ok(Options {
            articles: self.articles,
            episodes_per_article: self.episodes_per_article,
            min_words: self.min_words,
            attempts_per_article: self.attempts,
            max_calls: self.max_calls,
            max_output_tokens: self.max_output_tokens,
            workers: self.workers,
            transport_retries: self.transport_retries,
            transport_backoff: self.transport_backoff,
        })
    }
}

impl CorpusArgs {
    fn plan(&self) -> Plan {
        Plan {
            writer: self.writer.clone(),
            reader: self.reader.clone(),
            allow_same_family: self.allow_same_family,
            articles: self.articles,
            episodes_per_article: self.episodes_per_article,
            min_words: self.min_words,
            attempts: self.attempts,
            max_calls: self.max_calls,
            max_output_tokens: self.max_output_tokens,
            workers: self.workers,
            transport_retries: self.transport_retries,
            transport_backoff: Duration::from_secs(self.transport_backoff_secs),
        }
    }
}

/// Prints what a corpus run did.
pub fn print_summary(persona_name: &str, summary: &Summary) {
    println!(
        "{persona_name}: {} written, {} already kept, {} given up, {} unreachable, {} turned away; {} calls, about {} tokens in and {} out{}",
        summary.written,
        summary.skipped,
        summary.abandoned,
        summary.unavailable,
        summary.rejected,
        summary.calls,
        summary.input_tokens,
        summary.output_tokens,
        summary
            .stopped
            .as_ref()
            .map_or_else(String::new, |why| format!("; stopped: {why}")),
    );
}

/// Runs `corpus`.
///
/// # Errors
/// The writer and the reader are of the same model family, the author seed or
/// the recipe contradicts what the directory records, the runtime cannot start, or the corpus cannot be written.
pub fn run(args: CorpusArgs) -> anyhow::Result<()> {
    let plan = args.plan();
    let Prepared { author, run_seed } =
        plan.prepare(&args.cache, args.seed, args.opposite_of, args.run_seed)?;
    let cache = Cache::open(args.cache.clone())?;
    let mut config = Config::from_env();
    config.state_root = StateRoot::new(cache.dir().join("state"));
    config.thinking = false;
    config.allow_remote = args.allow_remote;
    let splinter = Splinter::builder(config)
        .build()
        .context("starting the runtime")?;
    let ctx = splinter.context();
    let writer = Source::open(&ctx, &args.writer, run_seed)?;
    let reading = args
        .reader
        .as_deref()
        .map(|name| Source::open(&ctx, name, run_seed))
        .transpose()?;
    let options = plan.begin(
        cache.dir(),
        &Prepared {
            author: author.clone(),
            run_seed,
        },
    )?;
    let summary = write_corpus_read(
        &author.persona,
        author.seed,
        &writer,
        reading.as_ref().map(|r| r as &dyn Models),
        &cache,
        &options,
    )?;
    print_summary(&author.persona.name, &summary);
    Ok(())
}
