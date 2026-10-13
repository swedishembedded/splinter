// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled validation of learning systems
// against ground truth that cannot have been seen in pretraining, for its
// clients. If your team needs expertise in proving that a trained model has
// learned what a corpus implies, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Draws a synthetic author and shows what it believes and what it decides.
//!
//! ```text
//! synthetic-author persona --seed 7 --show 5
//! ```

use std::path::PathBuf;
use std::str::FromStr;

use anyhow::Context as _;
use clap::{Parser, Subcommand};
use splinter_sdk::agent::blocking::BlockingModel;
use splinter_sdk::model::local::{Sampling, GREEDY_SAMPLING};
use splinter_sdk::store::StateRoot;
use splinter_sdk::vocabulary::model_ref::ModelRef;
use splinter_sdk::{Config, Splinter};
use synthetic_author::cache::Cache;
use synthetic_author::persona::Persona;
use synthetic_author::scenario::{scenario_pool, TOPICS};
use synthetic_author::writer::{write_corpus, Options};

/// The temperature the writer draws at, so that no two articles are alike.
const WRITER_TEMPERATURE: f32 = 0.8;

#[derive(Parser)]
#[command(
    about = "A synthetic author with known beliefs, for validating learning against ground truth"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Draw an author and show its beliefs and some of its decisions.
    Persona {
        /// The seed the author is drawn from.
        #[arg(long)]
        seed: u64,
        /// How many decisions to show.
        #[arg(long, default_value_t = 5)]
        show: usize,
    },
    /// Write the author's corpus into a cache directory.
    Corpus {
        /// The seed the author is drawn from.
        #[arg(long)]
        seed: u64,
        /// The directory the corpus is kept in (outside the repository).
        #[arg(long)]
        cache: PathBuf,
        /// The model that writes, as a model reference.
        #[arg(long)]
        writer: String,
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
    },
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Command::Persona { seed, show } => {
            let persona = Persona::generate(seed);
            println!("{} (seed {seed})", persona.name);
            for (rank, belief) in persona.beliefs.iter().enumerate() {
                println!("  {}. {}", rank + 1, belief.statement());
            }
            for scenario in scenario_pool(seed, show * 4)
                .iter()
                .filter(|s| persona.decide(s).determined)
                .take(show)
            {
                let decision = persona.decide(scenario);
                println!(
                    "{}: {}\n  facts {}\n  options {}\n  decides {}",
                    scenario.id,
                    TOPICS[scenario.topic],
                    serde_json::to_string(&scenario.features)?,
                    serde_json::to_string(&scenario.options)?,
                    serde_json::to_string(&decision)?,
                );
            }
            Ok(())
        }
        Command::Corpus {
            seed,
            cache,
            writer,
            allow_remote,
            articles,
            episodes_per_article,
            min_words,
            attempts,
            max_calls,
            max_output_tokens,
        } => {
            let persona = Persona::generate(seed);
            let cache = Cache::open(cache)?;
            let mut config = Config::from_env();
            config.state_root = StateRoot::new(cache.dir().join("state"));
            config.thinking = false;
            config.allow_remote = allow_remote;
            let splinter = Splinter::builder(config)
                .build()
                .context("starting the runtime")?;
            let reference =
                ModelRef::from_str(&writer).with_context(|| format!("model {writer:?}"))?;
            let ctx = splinter.context();
            let model = if reference.is_remote() {
                ctx.model(&reference)?
            } else {
                let sampling = Sampling {
                    temperature: WRITER_TEMPERATURE,
                    thinking: false,
                    ..GREEDY_SAMPLING
                };
                ctx.resampled(&reference, sampling)?
                    .with_context(|| format!("{reference} cannot be sampled at a temperature"))?
            };
            let blocking = BlockingModel::new(model)?;
            let options = Options {
                articles,
                episodes_per_article,
                min_words,
                attempts_per_article: attempts,
                max_calls,
                max_output_tokens,
            };
            let summary = write_corpus(&persona, seed, &blocking, &cache, &options)?;
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
    }
}
