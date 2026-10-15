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
use splinter_absorb::cycle;
use splinter_absorb::roles::Quotas;
use splinter_absorb::runtime::{DEFAULT_JUDGE, DEFAULT_MODEL};
use splinter_absorb::session::Style;
use splinter_absorb::{baseline, known, pool, record, screen, seal, select};

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
    /// The nights of the protocol: record, absorb, evaluate, report.
    #[command(subcommand)]
    Cycle(cycle::cli::Cycle),
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
    /// Choose checkable facts from the family files: code picks the sentences, a
    /// model words the questions, a judge pre-screens them. Resumable by
    /// family; shards run on separate cards.
    Select {
        #[command(flatten)]
        common: Common,
        /// The directory `splinter-jefferson materials` wrote (its `letters` directory is read).
        #[arg(long)]
        materials: PathBuf,
        /// Seeds the choice of families and every later hash. The selection's
        /// recorded seed when it has one, else a drawn one (printed). Shards
        /// started together should be given it, or the first started.
        #[arg(long)]
        seed: Option<u64>,
        /// Family files to read facts from.
        #[arg(long, default_value_t = 300)]
        families: usize,
        /// The fewest words a family file runs to be read.
        #[arg(long, default_value_t = 150)]
        min_words: usize,
        /// Families a shard takes in a row; a batch is written to the ledger whole
        /// and costs one load of each model, so it is large.
        #[arg(long, default_value_t = 125)]
        batch: usize,
        /// Candidate sentences tried for a family before it gives none.
        #[arg(long, default_value_t = 3)]
        tries: usize,
        /// Accepted facts of a batch the judge is also measured on.
        #[arg(long, default_value_t = 3)]
        controls_per_batch: usize,
        /// The model that words the questions.
        #[arg(long, default_value = DEFAULT_MODEL)]
        generator: String,
        /// The model that pre-screens the questions.
        #[arg(long, default_value = DEFAULT_JUDGE)]
        judge: String,
        /// Read only shard I of N (as `I/N`, from 0); a rerun skips every family a ledger holds.
        #[arg(long, value_parser = shard, default_value = "0/1")]
        shard: (usize, usize),
    },
    /// Assemble the manifest from the ledgers of `facts select`.
    Pool {
        /// The run's output directory.
        #[arg(long)]
        out: PathBuf,
        /// The directory `splinter-jefferson materials` wrote.
        #[arg(long)]
        materials: PathBuf,
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
    /// Add the curated list of widely known facts, supported by the corpus or
    /// by a named source, after the judge's pre-screen.
    Known {
        #[command(flatten)]
        common: Common,
        /// The directory `splinter-jefferson materials` wrote.
        #[arg(long)]
        materials: PathBuf,
        /// The model that pre-screens the questions.
        #[arg(long, default_value = DEFAULT_JUDGE)]
        judge: String,
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
        #[command(flatten)]
        quotas: Overrides,
    },
    /// Fill the roles' quotas again from the classes screening recorded, with
    /// other quotas if they are given (no model runs). Every role the classed
    /// facts cannot fill is named.
    Place {
        /// The run's output directory.
        #[arg(long)]
        out: PathBuf,
        #[command(flatten)]
        quotas: Overrides,
    },
}

/// Quotas to fill in place of the manifest's, for a smaller run; they replace
/// the manifest's and are recorded in it.
#[derive(Args)]
struct Overrides {
    /// Fill this many development facts.
    #[arg(long)]
    dev: Option<usize>,
    /// Fill this many test facts.
    #[arg(long)]
    test: Option<usize>,
    /// Fill this many wrong-at-day-0 controls.
    #[arg(long)]
    control_untaught: Option<usize>,
    /// Fill this many right-at-day-0 controls.
    #[arg(long)]
    control_known: Option<usize>,
    /// Fill this many neighbours.
    #[arg(long)]
    neighbour_known: Option<usize>,
}

impl Overrides {
    /// `base` with the quotas given here in place of its own; `None` when none is given.
    fn apply(&self, base: Quotas) -> Option<Quotas> {
        let given = [
            self.dev,
            self.test,
            self.control_untaught,
            self.control_known,
            self.neighbour_known,
        ];
        given.iter().any(Option::is_some).then(|| Quotas {
            dev: self.dev.unwrap_or(base.dev),
            test: self.test.unwrap_or(base.test),
            control_untaught: self.control_untaught.unwrap_or(base.control_untaught),
            control_known: self.control_known.unwrap_or(base.control_known),
            neighbour_known: self.neighbour_known.unwrap_or(base.neighbour_known),
            hallucination: base.hallucination,
        })
    }
}

#[derive(Subcommand)]
enum Probes {
    /// Write three probes per development and test fact, hash them and record the hash.
    Seal {
        #[command(flatten)]
        common: Common,
    },
    /// Put the sealed sycophancy and unknown probes to the day-0 policy and grade them.
    Baseline {
        #[command(flatten)]
        common: Common,
        /// The day-0 policy.
        #[arg(long, default_value = DEFAULT_MODEL)]
        policy: String,
        /// The judge, calibrated on controls before it grades.
        #[arg(long, default_value = DEFAULT_JUDGE)]
        judge: String,
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
    /// Record live sessions as ATIF files: one given on the command line, or
    /// those of a plan file, run together so that the card changes models
    /// twice a round however many sessions there are.
    Record {
        #[command(flatten)]
        common: Common,
        /// JSON Lines of sessions to record together (`{"kind", "fact", "style",
        /// "day", "topic"}`), instead of the single session below.
        #[arg(long, conflicts_with_all = ["fact", "noise", "sham", "canary", "style", "topic", "day"])]
        plan: Option<PathBuf>,
        /// The fact the user corrects.
        #[arg(long, conflicts_with = "noise", required_unless_present_any = ["noise", "plan"])]
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
        /// The model that plays the user: plain, no adapter, and not the policy's
        /// own (the judge's by default).
        #[arg(long, default_value = DEFAULT_JUDGE)]
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
        Command::Facts(Facts::Select {
            common,
            materials,
            seed,
            families,
            min_words,
            batch,
            tries,
            controls_per_batch,
            generator,
            judge,
            shard,
        }) => {
            let seed = splinter_absorb::ledger::seed(&common.out, seed)?;
            let report = select::run(&select::Request {
                materials,
                out: common.out,
                seed,
                families,
                min_words,
                batch,
                tries,
                controls_per_batch,
                generator,
                judge,
                persona: PERSONA.to_string(),
                models: common.models,
                shard,
            })?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::Facts(Facts::Pool {
            out,
            materials,
            dev,
            test,
            control_untaught,
            control_known,
            neighbour_known,
            hallucination,
        }) => {
            let report = pool::run(&pool::Request {
                materials,
                out,
                quotas: Quotas {
                    dev,
                    test,
                    control_untaught,
                    control_known,
                    neighbour_known,
                    hallucination,
                },
            })?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::Facts(Facts::Screen {
            common,
            policy,
            judge,
            candidates,
            quotas,
        }) => {
            let base = splinter_absorb::facts::Manifest::read(&common.out)?.quotas;
            let summary = screen::run(&screen::Request {
                quotas: quotas.apply(base),
                out: common.out,
                models: common.models,
                policy,
                judge,
                candidates,
            })?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
        Command::Facts(Facts::Place { out, quotas }) => {
            let mut manifest = splinter_absorb::facts::Manifest::read(&out)?;
            anyhow::ensure!(
                manifest.probes.is_none(),
                "the probes are sealed: the roles are closed"
            );
            if let Some(quotas) = quotas.apply(manifest.quotas) {
                manifest.quotas = quotas;
            }
            screen::place(&mut manifest)?;
            manifest.write(&out)?;
            let mut counts = std::collections::BTreeMap::new();
            for role in manifest.facts.iter().filter_map(|f| f.role) {
                *counts.entry(role.name()).or_insert(0usize) += 1;
            }
            println!("{}", serde_json::to_string_pretty(&counts)?);
        }
        Command::Facts(Facts::Known {
            common,
            materials,
            judge,
        }) => {
            let report = known::run(&known::Request {
                materials,
                out: common.out,
                models: common.models,
                judge,
            })?;
            println!("{}", serde_json::to_string_pretty(&report)?);
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
        Command::Probes(Probes::Baseline {
            common,
            policy,
            judge,
        }) => {
            let summary = baseline::run(&baseline::Request {
                out: common.out,
                models: common.models,
                policy,
                judge,
            })?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
        Command::Probes(Probes::Check { out, files }) => {
            let records = seal::guard(&out)?.check_files(&files)?;
            println!("{records} training records leak no sealed probe");
        }
        Command::Cycle(command) => cycle::cli::run(command)?,
        Command::Session(Session::Record {
            common,
            plan,
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
            let items = match plan {
                Some(plan) => record::read_plan(&plan)?,
                None => {
                    let about = match (fact, noise, sham) {
                        (Some(id), false, false) => record::About::Fact(id),
                        (Some(id), false, true) => record::About::Sham(id),
                        _ => record::About::Noise(topic),
                    };
                    vec![record::Item {
                        day,
                        about,
                        style,
                        canary,
                    }]
                }
            };
            let outcome = record::record(&record::Request {
                run: common.out.clone(),
                out: common.out,
                models: common.models,
                policy,
                simulator,
                items,
            })?;
            println!("{}", serde_json::to_string_pretty(&outcome)?);
            anyhow::ensure!(outcome.failed.is_empty(), "{}", outcome.failed.join("; "));
        }
    }
    Ok(())
}
