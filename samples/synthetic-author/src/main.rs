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
//! synthetic-author persona --show 5
//! ```

use std::path::PathBuf;

use anyhow::Context as _;
use clap::{Parser, Subcommand};
use splinter_sdk::store::StateRoot;
use splinter_sdk::{Config, Splinter};
use synthetic_author::cache::Cache;
use synthetic_author::corpus_cmd::CorpusArgs;
use synthetic_author::exam_cmd::{CrossArgs, ExamArgs, ReportArgs};
use synthetic_author::manifest::{author_seed, author_seed_or_draw, begin_or_continue_mining};
use synthetic_author::models::{is_remote, Source};
use synthetic_author::persona::Persona;
use synthetic_author::scenario::{scenario_pool, TOPICS};
use synthetic_author::training::TrainArgs;
use synthetic_author::writer::plan;

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
        /// The seed the author is drawn from. With `--cache` it is read from the
        /// corpus and a different one is refused; without either, one is drawn
        /// and printed.
        #[arg(long)]
        seed: Option<u64>,
        /// A corpus directory whose author is shown.
        #[arg(long)]
        cache: Option<PathBuf>,
        /// How many decisions to show.
        #[arg(long, default_value_t = 5)]
        show: usize,
    },
    /// List the cases each article of a corpus tells, with the decision the
    /// author's beliefs determine for each.
    Plan {
        /// The seed the author is drawn from. With `--cache` it is read from the
        /// corpus and a different one is refused; without either, one is drawn
        /// and printed.
        #[arg(long)]
        seed: Option<u64>,
        /// A corpus directory whose author is shown.
        #[arg(long)]
        cache: Option<PathBuf>,
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
        /// The seed the author is drawn from; read from the corpus when absent.
        #[arg(long)]
        seed: Option<u64>,
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
    /// Read the decisions out of the articles of a corpus.
    Mine {
        /// The corpus directory.
        #[arg(long)]
        cache: PathBuf,
        /// The model that reads, as a model reference.
        #[arg(long)]
        miner: String,
        /// Permit a reader reached over the network.
        #[arg(long)]
        allow_remote: bool,
        /// How many articles are read at a time.
        #[arg(long, default_value_t = 4)]
        workers: usize,
        /// The seed the reader's per-request seeds are derived from; drawn and
        /// printed when absent, and read back from the corpus when it was
        /// mined before.
        #[arg(long)]
        run_seed: Option<u64>,
    },
    /// Make the training records from the decisions read.
    Dataset {
        /// The seed the author is drawn from; read from the corpus when absent.
        #[arg(long)]
        seed: Option<u64>,
        /// The corpus directory.
        #[arg(long)]
        cache: PathBuf,
    },
    Train(TrainArgs),
    Exam(ExamArgs),
    Report(ReportArgs),
    /// Show what the writer is given for one article: the instruction and the
    /// case files, convictions and what the author did, as the model sees them.
    Show {
        /// The seed the author is drawn from. With `--cache` it is read from the
        /// corpus and a different one is refused; without either, one is drawn
        /// and printed.
        #[arg(long)]
        seed: Option<u64>,
        /// A corpus directory whose author is shown.
        #[arg(long)]
        cache: Option<PathBuf>,
        /// The article.
        #[arg(long, default_value_t = 0)]
        article: usize,
        /// How many articles the corpus is planned with.
        #[arg(long, default_value_t = 60)]
        articles: usize,
        /// The fewest words asked for.
        #[arg(long, default_value_t = 700)]
        min_words: usize,
    },
    /// Show the exam items of an author as the model is asked them.
    Items {
        /// The seed the author is drawn from. With `--cache` it is read from the
        /// corpus and a different one is refused; without either, one is drawn
        /// and printed.
        #[arg(long)]
        seed: Option<u64>,
        /// How many articles the corpus was planned with.
        #[arg(long, default_value_t = 100)]
        articles: usize,
        /// Items of each kind of difficulty.
        #[arg(long, default_value_t = 40)]
        per_cell: usize,
        /// A corpus directory: with it each item also shows the system turn
        /// the retrieval arm is asked under. The second examination needs it,
        /// for the decisions read out of the corpus.
        #[arg(long)]
        cache: Option<PathBuf>,
        /// Which examination, as for `exam`.
        #[arg(long, value_enum, default_value_t = synthetic_author::exam::Version::default())]
        exam: synthetic_author::exam::Version,
    },
    /// Compare the decisions read out of the essays with what the beliefs dictate.
    Audit {
        /// The seed the author is drawn from; read from the corpus when absent.
        #[arg(long)]
        seed: Option<u64>,
        /// The corpus directory.
        #[arg(long)]
        cache: PathBuf,
        /// How many articles the corpus was planned with.
        #[arg(long, default_value_t = 100)]
        articles: usize,
    },
    Cross(CrossArgs),
    Corpus(CorpusArgs),
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Command::Persona { seed, cache, show } => {
            let seed = author_seed_or_draw(cache.as_deref(), seed)?;
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
            cache,
            articles,
            episodes_per_article,
        } => {
            let seed = author_seed_or_draw(cache.as_deref(), seed)?;
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
            let seed = author_seed(&cache, seed)?;
            let persona = Persona::generate(seed);
            let planned = plan(&persona, seed, articles, episodes_per_article);
            let text = std::fs::read_to_string(cache.join("rejected.jsonl"))
                .context("reading rejected.jsonl")?;
            let rechecked =
                synthetic_author::recheck::recheck(&text, &persona, &planned, min_words)?;
            let drafts = rechecked.len();
            let now_pass = rechecked.iter().filter(|r| r.now.is_ok()).count();
            for r in rechecked {
                println!(
                    "article {} attempt {}: was '{}' - now {}",
                    r.article,
                    r.attempt,
                    r.was,
                    r.now
                        .map_or_else(|why| format!("turned away: {why}"), |()| "kept".into()),
                );
            }
            println!("{now_pass} of {drafts} saved drafts pass the gates as they are now");
            Ok(())
        }
        Command::Mine {
            cache,
            miner,
            allow_remote,
            workers,
            run_seed,
        } => {
            let cache = Cache::open(cache)?;
            let record =
                begin_or_continue_mining(cache.dir(), &miner, run_seed, !is_remote(&miner)?)?;
            let mut config = Config::from_env();
            config.state_root = StateRoot::new(cache.dir().join("state"));
            config.thinking = false;
            config.allow_remote = allow_remote;
            let splinter = Splinter::builder(config)
                .build()
                .context("starting the runtime")?;
            let ctx = splinter.context();
            let source = Source::open(&ctx, &miner, record.run_seed)?;
            let read = synthetic_author::lab::mine_all(&cache, &source, workers)?;
            let all = synthetic_author::lab::read_mined(&cache)?;
            let kept = all
                .iter()
                .filter(|m| matches!(m.result, synthetic_author::lab::MinedResult::Kept(_)))
                .count();
            println!("{read} articles read; {kept} of {} cases kept", all.len());
            Ok(())
        }
        Command::Dataset { seed, cache } => {
            let seed = author_seed(&cache, seed)?;
            let cache = Cache::open(cache)?;
            let out = cache.dir().join("dataset").join("train.jsonl");
            let summary = synthetic_author::lab::build_dataset(&cache, seed, &out)?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
            println!("{}", out.display());
            Ok(())
        }
        Command::Train(args) => synthetic_author::training::run(&args),
        Command::Exam(args) => synthetic_author::exam_cmd::exam(&args),
        Command::Show {
            seed,
            cache,
            article,
            articles,
            min_words,
        } => {
            let seed = author_seed_or_draw(cache.as_deref(), seed)?;
            let persona = Persona::generate(seed);
            let planned = plan(&persona, seed, articles, 3);
            let (topic, episodes) = planned.get(article).with_context(|| {
                format!("article {article} is not among the {articles} planned")
            })?;
            let shown =
                synthetic_author::writer::writer_input(&persona, *topic, episodes, min_words, None);
            println!(
                "{}",
                serde_json::json!({"instruction": synthetic_author::writer::TASK, "input": shown})
            );
            Ok(())
        }
        Command::Items {
            seed,
            articles,
            per_cell,
            cache,
            exam,
        } => {
            let seed = author_seed_or_draw(cache.as_deref(), seed)?;
            let persona = Persona::generate(seed);
            let cache = cache.map(Cache::open).transpose()?;
            let notes = cache
                .as_ref()
                .map(|cache| cache.articles())
                .transpose()?
                .map(|articles| synthetic_author::retrieval::Notes::new(&articles));
            let all = match &cache {
                Some(cache) => synthetic_author::lab::exam_items_of(
                    cache, &persona, seed, articles, per_cell, exam,
                )?,
                None if exam == synthetic_author::exam::Version::V1 => {
                    synthetic_author::lab::exam_items(&persona, seed, articles, per_cell)
                }
                None => anyhow::bail!(
                    "--cache is needed for the second examination, which says which items the corpus trained on; pass --exam v1 to show the first"
                ),
            };
            for item in all {
                let mut view = synthetic_author::lab::item_view(&item);
                if let Some(notes) = &notes {
                    view["retrieval_system"] = notes
                        .system_for(&item, synthetic_author::lab::RETRIEVED_PARAGRAPHS)
                        .into();
                }
                println!("{view}");
            }
            Ok(())
        }
        Command::Audit {
            seed,
            cache,
            articles,
        } => {
            let seed = author_seed(&cache, seed)?;
            let persona = Persona::generate(seed);
            let cache = Cache::open(cache)?;
            let mined = synthetic_author::lab::read_mined(&cache)?;
            let found = synthetic_author::lab::audit(&mined, &persona, seed, articles);
            println!(
                "{}: {} of {} kept decisions are the one the beliefs dictate; {} of {} list the ways the case offered",
                persona.name, found.agree, found.kept, found.open_agree, found.kept
            );
            Ok(())
        }
        Command::Cross(args) => synthetic_author::exam_cmd::cross(&args),
        Command::Report(args) => synthetic_author::exam_cmd::report(&args),
        Command::Corpus(args) => synthetic_author::corpus_cmd::run(args),
    }
}
