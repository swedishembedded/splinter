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

use std::path::PathBuf;

use anyhow::Context as _;
use clap::Args;
use splinter_sdk::store::StateRoot;
use splinter_sdk::{Config, Splinter};

use crate::cache::Cache;
use crate::manifest::{
    begin_or_continue, code_revision, corpus_author_seed, run_seed as run_seed_of, CorpusManifest,
    Limits, Recipe, SEED_DERIVATION,
};
use crate::models::{is_remote, Source, WRITER_DECODING};
use crate::persona::Persona;
use crate::writer::{write_corpus_read, Models, Options};

/// Write the author's corpus into a cache directory.
#[derive(Args)]
pub struct CorpusArgs {
    /// The seed the author is drawn from; a fresh one is drawn and printed
    /// when none is given and the directory records no author.
    #[arg(long)]
    seed: Option<u64>,
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
}

/// Runs `corpus`.
///
/// # Errors
/// The author seed or the recipe contradicts what the directory records, the
/// runtime cannot start, or the corpus cannot be written.
pub fn run(args: CorpusArgs) -> anyhow::Result<()> {
    let (author_seed, seed_source) = corpus_author_seed(&args.cache, args.seed)?;
    let any_local = !is_remote(&args.writer)?
        || args.reader.as_deref().map(is_remote).transpose()? == Some(false);
    let run_seed = run_seed_of(&args.cache, args.run_seed, any_local)?;
    let persona = Persona::generate(author_seed);
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
    begin_or_continue(
        cache.dir(),
        CorpusManifest {
            seed_source,
            recipe: Recipe {
                author_seed,
                writer: args.writer.clone(),
                reader: args.reader.clone(),
                decoding: WRITER_DECODING.to_string(),
                run_seed,
                articles: args.articles,
                episodes_per_article: args.episodes_per_article,
                min_words: args.min_words,
            },
            seed_derivation: SEED_DERIVATION.to_string(),
            limits: Limits {
                attempts: args.attempts,
                max_calls: args.max_calls,
                max_output_tokens: args.max_output_tokens,
                workers: args.workers,
            },
            code_revision: code_revision().to_string(),
        },
    )?;
    let options = Options {
        articles: args.articles,
        episodes_per_article: args.episodes_per_article,
        min_words: args.min_words,
        attempts_per_article: args.attempts,
        max_calls: args.max_calls,
        max_output_tokens: args.max_output_tokens,
        workers: args.workers,
    };
    let summary = write_corpus_read(
        &persona,
        author_seed,
        &writer,
        reading.as_ref().map(|r| r as &dyn Models),
        &cache,
        &options,
    )?;
    println!(
        "{}: {} written, {} already kept, {} given up, {} turned away; {} calls, about {} tokens in and {} out{}",
        persona.name,
        summary.written,
        summary.skipped,
        summary.abandoned,
        summary.rejected,
        summary.calls,
        summary.input_tokens,
        summary.output_tokens,
        summary.stopped.map_or_else(String::new, |why| format!("; stopped: {why}")),
    );
    Ok(())
}
