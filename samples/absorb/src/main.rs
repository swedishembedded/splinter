// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measured validation of continual learning
// from a user's own agent sessions for its clients. If your team needs
// expertise in proving that a model absorbed what its user taught it, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The command line of the absorb validation protocol's first part.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};
use splinter_absorb::roles::Quotas;
use splinter_absorb::runtime::{DEFAULT_JUDGE, DEFAULT_MODEL};
use splinter_absorb::session::Style;
use splinter_absorb::{build, record, screen, seal};

/// Persona the policy answers as, and the author of the writings.
const PERSONA: &str = "Thomas Jefferson";

#[derive(Parser)]
#[command(
    name = "splinter-absorb",
    about = "Prepare and run the absorb validation protocol"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// The pool of facts the protocol measures.
    #[command(subcommand)]
    Facts(Facts),
    /// The sealed probes of the development and test facts.
    #[command(subcommand)]
    Probes(Probes),
    /// Sessions between the policy and a simulated user.
    #[command(subcommand)]
    Session(Session),
}

#[derive(Args)]
struct Common {
    /// The run's output directory: its state and manifest.
    #[arg(long)]
    out: PathBuf,
    /// The model store, when not the configuration's.
    #[arg(long)]
    models: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Facts {
    /// Read facts from a directory of family files with Splinter's task generator.
    Build {
        #[command(flatten)]
        common: Common,
        /// The directory `splinter-jefferson materials` wrote (its `letters` directory is read).
        #[arg(long)]
        materials: PathBuf,
        /// Seeds the choice of families and every later hash.
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Family files to read facts from.
        #[arg(long, default_value_t = 300)]
        families: usize,
        /// The fewest words a family file runs to be read.
        #[arg(long, default_value_t = 150)]
        min_words: usize,
        /// The model that proposes the facts and writes the probes.
        #[arg(long, default_value = DEFAULT_MODEL)]
        generator: String,
        /// Build only shard I of N (as `I/N`, from 0) of the families.
        #[arg(long, value_parser = shard)]
        shard: Option<(usize, usize)>,
        /// Development facts.
        #[arg(long, default_value_t = Quotas::PROTOCOL.dev)]
        dev: usize,
        /// Test facts.
        #[arg(long, default_value_t = Quotas::PROTOCOL.test)]
        test: usize,
        /// Controls wrong at day 0.
        #[arg(long, default_value_t = Quotas::PROTOCOL.control_untaught)]
        control_untaught: usize,
        /// Controls right at day 0.
        #[arg(long, default_value_t = Quotas::PROTOCOL.control_known)]
        control_known: usize,
        /// Facts right at day 0 about an entity a test fact is about.
        #[arg(long, default_value_t = Quotas::PROTOCOL.neighbour_known)]
        neighbour_known: usize,
        /// Questions about things that do not exist.
        #[arg(long, default_value_t = Quotas::PROTOCOL.hallucination)]
        hallucination: usize,
    },
    /// Read every fact's keys afresh with the current rules (before screening).
    Rekey {
        /// The run's output directory.
        #[arg(long)]
        out: PathBuf,
    },
    /// Merge the pools of shards built side by side into one.
    Merge {
        /// The merged run's output directory.
        #[arg(long)]
        out: PathBuf,
        /// The model store, when not the configuration's.
        #[arg(long)]
        models: Option<PathBuf>,
        /// The shards' output directories.
        #[arg(required = true)]
        parts: Vec<PathBuf>,
    },
    /// Flag one test fact as the canary: its sessions assert a false statement.
    Canary {
        /// The run's output directory.
        #[arg(long)]
        out: PathBuf,
        /// The test fact.
        #[arg(long)]
        fact: String,
    },
    /// Put each fact to the day-0 policy six times, grade, class and fill the roles.
    Screen {
        #[command(flatten)]
        common: Common,
        /// The day-0 policy.
        #[arg(long, default_value = DEFAULT_MODEL)]
        policy: String,
        /// The judge, calibrated on controls before it grades.
        #[arg(long, default_value = DEFAULT_JUDGE)]
        judge: String,
        /// Screen only this many facts of the pool, in a fixed order.
        #[arg(long)]
        candidates: Option<usize>,
        /// Fill this many development facts instead of the manifest's quota.
        #[arg(long)]
        dev: Option<usize>,
        /// Fill this many test facts instead of the manifest's quota.
        #[arg(long)]
        test: Option<usize>,
        /// Fill this many wrong-at-day-0 controls instead of the manifest's quota.
        #[arg(long)]
        control_untaught: Option<usize>,
        /// Fill this many right-at-day-0 controls instead of the manifest's quota.
        #[arg(long)]
        control_known: Option<usize>,
        /// Fill this many neighbours instead of the manifest's quota.
        #[arg(long)]
        neighbour_known: Option<usize>,
    },
}

#[derive(Subcommand)]
enum Probes {
    /// Write three probes per development and test fact, hash them and record the hash.
    Seal {
        #[command(flatten)]
        common: Common,
    },
    /// Refuse training files that contain a sealed probe's question or an
    /// eight-word run of one beyond the fact's statement.
    Check {
        /// The run's output directory.
        #[arg(long)]
        out: PathBuf,
        /// Training files, JSON lines.
        #[arg(required = true)]
        files: Vec<PathBuf>,
    },
}

#[derive(Subcommand)]
enum Session {
    /// Record one live session as an ATIF file.
    Record {
        #[command(flatten)]
        common: Common,
        /// The fact the user corrects.
        #[arg(long, conflicts_with = "noise", required_unless_present = "noise")]
        fact: Option<String>,
        /// An unrelated chat instead of a fact (a null day is a day of these only).
        #[arg(long)]
        noise: bool,
        /// A sham session: the fact is asked about and the answer is never corrected.
        #[arg(long, requires = "fact", conflicts_with = "canary")]
        sham: bool,
        /// The user asserts the fact's false canary statement (see `facts canary`).
        #[arg(long, requires = "fact")]
        canary: bool,
        /// How the user goes about it.
        #[arg(long, value_enum, default_value = "plain")]
        style: Style,
        /// Which of the noise topics to chat about (default: the next).
        #[arg(long, requires = "noise")]
        topic: Option<usize>,
        /// The day the session belongs to (a test fact's own day by default).
        #[arg(long)]
        day: Option<usize>,
        /// The policy that answers; `+ADAPTER` after the checkpoint names the current release.
        #[arg(long, default_value = DEFAULT_MODEL)]
        policy: String,
        /// The model that plays the user: plain, no adapter.
        #[arg(long, default_value = DEFAULT_MODEL)]
        simulator: String,
    },
}

/// `I/N`: shard I (from 0) of N.
fn shard(text: &str) -> Result<(usize, usize), String> {
    let (index, of) = text.split_once('/').ok_or("write the shard as I/N")?;
    let (index, of): (usize, usize) = (
        index.parse().map_err(|e| format!("{e}"))?,
        of.parse().map_err(|e| format!("{e}"))?,
    );
    if of == 0 || index >= of {
        return Err("I must be below N".into());
    }
    Ok((index, of))
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Command::Facts(Facts::Build {
            common,
            materials,
            seed,
            families,
            min_words,
            generator,
            shard,
            dev,
            test,
            control_untaught,
            control_known,
            neighbour_known,
            hallucination,
        }) => {
            let report = build::run(&build::Request {
                materials,
                out: common.out,
                seed,
                quotas: Quotas {
                    dev,
                    test,
                    control_untaught,
                    control_known,
                    neighbour_known,
                    hallucination,
                },
                families,
                min_words,
                generator,
                persona: PERSONA.to_string(),
                models: common.models,
                shard,
            })?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::Facts(Facts::Screen {
            common,
            policy,
            judge,
            candidates,
            dev,
            test,
            control_untaught,
            control_known,
            neighbour_known,
        }) => {
            let overridden = [dev, test, control_untaught, control_known, neighbour_known]
                .iter()
                .any(Option::is_some);
            let quotas = if overridden {
                let manifest = splinter_absorb::facts::Manifest::read(&common.out)?;
                let q = manifest.quotas;
                Some(Quotas {
                    dev: dev.unwrap_or(q.dev),
                    test: test.unwrap_or(q.test),
                    control_untaught: control_untaught.unwrap_or(q.control_untaught),
                    control_known: control_known.unwrap_or(q.control_known),
                    neighbour_known: neighbour_known.unwrap_or(q.neighbour_known),
                    hallucination: q.hallucination,
                })
            } else {
                None
            };
            let summary = screen::run(&screen::Request {
                quotas,
                out: common.out,
                models: common.models,
                policy,
                judge,
                candidates,
            })?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
        Command::Facts(Facts::Rekey { out }) => {
            let mut manifest = splinter_absorb::facts::Manifest::read(&out)?;
            let refused = splinter_absorb::facts::rekey(&mut manifest);
            manifest.write(&out)?;
            println!("{} facts kept, {refused} refused", manifest.facts.len());
        }
        Command::Facts(Facts::Merge { out, models, parts }) => {
            let facts = splinter_absorb::merge::run(&out, &parts, models.as_ref())?;
            println!("{facts} facts in the merged pool");
        }
        Command::Facts(Facts::Canary { out, fact }) => {
            println!("{}", splinter_absorb::canary::flag(&out, &fact)?);
        }
        Command::Probes(Probes::Seal { common }) => {
            let sealed = seal::seal(&seal::Request {
                out: common.out,
                models: common.models,
            })?;
            println!("{}", serde_json::to_string_pretty(&sealed)?);
        }
        Command::Probes(Probes::Check { out, files }) => {
            let records = seal::guard(&out)?.check_files(&files)?;
            println!("{records} training records leak no sealed probe");
        }
        Command::Session(Session::Record {
            common,
            fact,
            noise,
            sham,
            canary,
            style,
            topic,
            day,
            policy,
            simulator,
        }) => {
            let about = match (fact, noise, sham) {
                (Some(id), false, false) => record::About::Fact(id),
                (Some(id), false, true) => record::About::Sham(id),
                _ => record::About::Noise(topic),
            };
            let recorded = record::record(&record::Request {
                out: common.out,
                models: common.models,
                policy,
                simulator,
                day,
                about,
                style,
                canary,
            })?;
            println!("{}", serde_json::to_string_pretty(&recorded)?);
        }
    }
    Ok(())
}
