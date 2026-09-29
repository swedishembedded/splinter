// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements delegated-task coding agents with a stable,
// documented invocation usable by people and by supervising models. If your
// team needs expertise in agent interfaces or CLI design, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The `splinter` command line: argument parsing and output only. Every
//! command is a call into `splinter-campaign`, configured once from the
//! environment.
//!
//! Exit status: 0 on success; 1 when a delegated attempt did not complete or
//! a trained candidate was rejected - a script must never read either as
//! progress; 2 on a usage error or an `ask` reply that does not parse.

mod cli;
mod output;

use clap::Parser;
use splinter_campaign::{ask, attempt, eval, facts, learn, train, Config};
use splinter_knowledge as explore;
use splinter_lab::promotion::Decision;
use splinter_policy::LocalWeights;

use cli::{Cli, Command};

fn main() -> anyhow::Result<()> {
    let config = Config::from_env();
    match Cli::parse().command {
        Command::Run(args) => {
            let json = args.json;
            let outcome = attempt::run(&config, &args.into_request()?)?;
            output::attempt(&outcome, json)
        }
        Command::Resume(args) => {
            let json = args.attempt.json;
            let outcome = attempt::resume(&config, &args.run, &args.attempt.into_request()?)?;
            output::attempt(&outcome, json)
        }
        Command::Show(args) => output::show(&config.state_root, args.run.as_deref()),
        Command::Cancel(args) => {
            let request = splinter_store::runs::request_cancel(&config.state_root, &args.run)?;
            println!(
                "cancel requested for {} ({}); the attempt stops within a second or one \
                 generation chunk, then `show --run {}` reports it",
                args.run,
                request.display(),
                args.run
            );
            Ok(())
        }
        Command::Learn(args) => {
            match learn::learn_run(&config.state_root, &args.run)? {
                learn::Learned::Appended => println!(
                    "learned {}: appended to {}",
                    args.run,
                    config.state_root.experience_pool().display()
                ),
                learn::Learned::AlreadyRecorded => println!(
                    "learned {}: already in the pool, no duplicate written",
                    args.run
                ),
            }
            Ok(())
        }
        Command::Train(args) => {
            let options = train::TrainOptions {
                model_dir: args
                    .local_weights
                    .unwrap_or_else(|| config.default_local_weights.clone()),
                dataset: args.dataset,
                steps: args.training.steps,
                rank: args.training.rank,
                alpha: args.training.alpha,
            };
            let (decision, dir) = train::run(&config.state_root, &options)?;
            match decision {
                Decision::Promoted => {
                    println!("promoted: adapter and scores in {}", dir.display());
                    Ok(())
                }
                Decision::Rejected => {
                    println!(
                        "rejected: held-out loss did not improve; scores in {}/decision.json",
                        dir.display()
                    );
                    std::process::exit(1);
                }
            }
        }
        Command::Explore(args) => {
            args.model.validate()?;
            let choice = args.model.choice();
            let summary = explore::run(
                &config.state_root,
                explore::ExploreOptions {
                    file: args.file,
                    out: args.out.clone(),
                    chunk_lines: args.chunk_lines,
                    model: choice.selection(&config, choice.local_weights(&config)),
                    scope_negatives: args.scope_negatives,
                },
            )?;
            println!(
                "explored {}: {} section(s), {} fact(s), {} parse failure(s), \
                 {} unanchored question(s) refused, {} answer(s) with untraceable numbers \
                 refused\nrun:     {}\nout:     {}",
                summary.run_id,
                summary.sections,
                summary.facts,
                summary.parse_failures,
                summary.unanchored,
                summary.untraceable,
                config.state_root.run_dir(&summary.run_id).display(),
                args.out.display(),
            );
            Ok(())
        }
        Command::Ask(args) => {
            args.model.validate()?;
            let choice = args.model.choice();
            let weights = choice.query_weights(&config, args.base)?;
            let options = ask::AskOptions {
                question: args.question,
                model: choice.selection(&config, weights),
            };
            match ask::run(options) {
                // The parsed object, not the raw reply: the caller reads
                // JSON or nothing.
                Ok(answer) => {
                    println!(
                        "{}",
                        serde_json::to_string(&serde_json::json!({ "answer": answer }))?
                    );
                    Ok(())
                }
                Err(e) => {
                    eprintln!("ask: {e:#}");
                    std::process::exit(2);
                }
            }
        }
        Command::EvalFacts(args) => {
            args.model.validate()?;
            let choice = args.model.choice();
            let weights = choice.query_weights(&config, args.base)?;
            let report = eval::run(eval::EvalOptions {
                dataset: args.dataset,
                out: args.out.clone(),
                model: choice.selection(&config, weights),
                shuffle: args.shuffle,
                limit: args.limit,
            })?;
            println!(
                "eval-facts: {}/{} correct (accuracy {:.3}, {} parse failure(s)) - report in {}",
                report.correct,
                report.total,
                report.accuracy,
                report.parse_failures,
                args.out.display(),
            );
            Ok(())
        }
        Command::Facts(args) => {
            args.model.validate()?;
            let choice = args.model.choice();
            let source = args
                .file
                .as_ref()
                .map(|f| f.display().to_string())
                .unwrap_or_else(|| "existing dataset".into());
            let report = facts::run(
                &config.state_root,
                facts::FactsOptions {
                    file: args.file,
                    out: args.out,
                    work_dir: args.work_dir.unwrap_or_else(|| config.state_root.facts()),
                    holdout_one_in: args.holdout_one_in,
                    chunk_lines: args.chunk_lines,
                    scope_negatives: args.scope_negatives,
                    steps: args.training.steps,
                    rank: args.training.rank,
                    alpha: args.training.alpha,
                    extractor: choice.selection(&config, choice.local_weights(&config)),
                    policy: LocalWeights {
                        adapter: None,
                        ..choice.local_weights(&config)
                    },
                },
            )?;
            output::facts(&report, &source);
            if !report.promoted {
                std::process::exit(1);
            }
            Ok(())
        }
    }
}
