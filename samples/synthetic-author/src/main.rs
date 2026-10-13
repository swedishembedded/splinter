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
use synthetic_author::gates::check;
use synthetic_author::persona::Persona;
use synthetic_author::scenario::{scenario_pool, TOPICS};
use synthetic_author::writer::{plan, write_corpus_read, Options};

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
    /// List the cases each article of a corpus tells, with the decision the
    /// author's beliefs determine for each.
    Plan {
        /// The seed the author is drawn from.
        #[arg(long)]
        seed: u64,
        /// How many articles the corpus has.
        #[arg(long, default_value_t = 100)]
        articles: usize,
        /// How many episodes each article tells.
        #[arg(long, default_value_t = 3)]
        episodes_per_article: usize,
    },
    /// Judge the drafts a corpus turned away by the gates as they are now,
    /// without asking a model anything.
    Recheck {
        /// The seed the author is drawn from.
        #[arg(long)]
        seed: u64,
        /// The corpus directory.
        #[arg(long)]
        cache: PathBuf,
        /// How many articles the corpus has.
        #[arg(long, default_value_t = 100)]
        articles: usize,
        /// How many episodes each article tells.
        #[arg(long, default_value_t = 3)]
        episodes_per_article: usize,
        /// The fewest words an article may have.
        #[arg(long, default_value_t = 700)]
        min_words: usize,
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
    },
}

/// The model `name` names, sampling at the writer's temperature when it runs
/// here and at its server's own when it does not.
fn model_named(
    ctx: &splinter_sdk::Context,
    name: &str,
) -> anyhow::Result<splinter_sdk::agent::solve::Model> {
    let reference = ModelRef::from_str(name).with_context(|| format!("model {name:?}"))?;
    if reference.is_remote() {
        return Ok(ctx.model(&reference)?);
    }
    let sampling = Sampling {
        temperature: WRITER_TEMPERATURE,
        thinking: false,
        ..GREEDY_SAMPLING
    };
    ctx.resampled(&reference, sampling)?
        .with_context(|| format!("{reference} cannot be sampled at a temperature"))
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
        Command::Plan {
            seed,
            articles,
            episodes_per_article,
        } => {
            let persona = Persona::generate(seed);
            for (index, (topic, episodes)) in plan(&persona, seed, articles, episodes_per_article)
                .iter()
                .enumerate()
            {
                for e in episodes {
                    let d = persona.decide(&e.scenario);
                    println!(
                        "{}",
                        serde_json::json!({
                            "article": index, "topic": topic, "case": e.case.name,
                            "decision": d.choice, "deciding": d.deciding,
                            "facts": e.scenario.features, "options": e.scenario.options,
                        })
                    );
                }
            }
            Ok(())
        }
        Command::Recheck {
            seed,
            cache,
            articles,
            episodes_per_article,
            min_words,
        } => {
            let persona = Persona::generate(seed);
            let planned = plan(&persona, seed, articles, episodes_per_article);
            let text = std::fs::read_to_string(cache.join("rejected.jsonl"))
                .context("reading rejected.jsonl")?;
            let (mut drafts, mut now_pass) = (0, 0);
            for line in text.lines() {
                let row: serde_json::Value = serde_json::from_str(line)?;
                let Some(draft) = row.get("draft").filter(|d| !d.is_null()) else {
                    continue;
                };
                let article: synthetic_author::gates::Article =
                    serde_json::from_value(draft.clone())?;
                let index = row["article"].as_u64().unwrap_or(0) as usize;
                let verdict = check(&article, &persona, &planned[index].1, min_words);
                drafts += 1;
                now_pass += usize::from(verdict.is_ok());
                println!(
                    "article {index} attempt {}: was '{}' - now {}",
                    row["attempt"],
                    row["reason"].as_str().unwrap_or(""),
                    verdict.map_or_else(|r| format!("turned away: {r}"), |()| "kept".into()),
                );
            }
            println!("{now_pass} of {drafts} saved drafts pass the gates as they are now");
            Ok(())
        }
        Command::Corpus {
            seed,
            cache,
            writer,
            reader,
            allow_remote,
            articles,
            episodes_per_article,
            min_words,
            attempts,
            max_calls,
            max_output_tokens,
            workers,
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
            let ctx = splinter.context();
            let blocking = BlockingModel::new(model_named(&ctx, &writer)?)?;
            let reading = reader
                .as_deref()
                .map(|name| model_named(&ctx, name).and_then(|m| Ok(BlockingModel::new(m)?)))
                .transpose()?;
            let options = Options {
                articles,
                episodes_per_article,
                min_words,
                attempts_per_article: attempts,
                max_calls,
                max_output_tokens,
                workers,
            };
            let summary = write_corpus_read(
                &persona,
                seed,
                &blocking,
                reading.as_ref(),
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
    }
}
