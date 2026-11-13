// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that compile a learning objective
// into validated, versioned training environments, for its clients. If your
// team needs expertise in environment factories for agent reinforcement
// learning, you can procure our services by sending an email to
// info@swedishembedded.com.

//! `env ...`.

use std::path::PathBuf;

use clap::{ArgGroup, Args, Subcommand};

/// Where the objective's text comes from.
#[derive(Debug, Args)]
#[command(group(ArgGroup::new("objective").required(true).multiple(false).args(["text", "file"])))]
pub struct ObjectiveArgs {
    /// The objective in plain language; each line is one skill.
    #[arg(long, value_name = "TEXT")]
    pub text: Option<String>,
    /// A file holding the objective; each line is one skill.
    #[arg(long, value_name = "FILE")]
    pub file: Option<PathBuf>,
    /// A source the objective rests on (a document, a repository); repeat for
    /// more. A skill about a document is only built when a source is given.
    #[arg(long = "source", value_name = "PATH")]
    pub sources: Vec<String>,
    /// What must hold of any environment built (a language, that real
    /// ratings or a calibration set exist); repeat for more.
    #[arg(long = "constraint", value_name = "TEXT")]
    pub constraints: Vec<String>,
}

/// `env ...`.
#[derive(Debug, Subcommand)]
pub enum EnvCommand {
    /// Say how each skill of an objective could be verified (an objective
    /// check, a proxy, a calibrated judge or human evidence), with the
    /// reasons and the plugin that would build it. Builds nothing.
    Classify {
        #[command(flatten)]
        objective: ObjectiveArgs,
    },
    /// List the environment plugins, their difficulty knobs and what each
    /// verifies by.
    Plugins,
    /// Compile an objective into environments: a difficulty ladder per
    /// skill that has a plugin, each sampled, held to the admission gates
    /// and the verifier red team, and stored with its report. A beaten
    /// verifier is quarantined; an environment whose gates pass waits for
    /// its base pass rate (`env band`) before it may enter a campaign.
    Compile {
        #[command(flatten)]
        objective: ObjectiveArgs,
        /// The store to write; default `environments` under the state root.
        #[arg(long, value_name = "DIR")]
        store: Option<PathBuf>,
        /// How many environments to build at most, across all skills.
        #[arg(long, default_value_t = 4, value_name = "N")]
        environments: usize,
        /// How many train and how many dev instances each environment is
        /// drawn for and tested on.
        #[arg(long, default_value_t = 4, value_name = "N")]
        instances: usize,
        /// The seed the samples are drawn from; drawn and printed when absent.
        #[arg(long, value_name = "N")]
        seed: Option<u64>,
        /// The directory of repositories with their test suites (the
        /// `tasks/` directory: a `catalog.toml` and one directory each);
        /// enables the repository plugin on real projects.
        #[arg(long, value_name = "DIR")]
        repositories: Option<PathBuf>,
        /// The emulator binary (`vemu`), run as a command and never
        /// modified; enables the firmware plugin.
        #[arg(long, value_name = "FILE")]
        emulator: Option<PathBuf>,
        /// A directory of Markdown or text documents to compile questions
        /// from (at least three: train and dev never share one); enables the
        /// document plugin. Name the same directory with --source so a
        /// document skill is built.
        #[arg(long, value_name = "DIR")]
        documents: Option<PathBuf>,
    },
    /// List the environments in a store with their status.
    List {
        /// The store; default `environments` under the state root.
        #[arg(long, value_name = "DIR")]
        store: Option<PathBuf>,
    },
    /// Show one environment's admission report: every gate, the attacks that
    /// beat the verifier, the base pass rate.
    Show {
        /// The environment's id.
        #[arg(value_name = "ENV-ID")]
        environment: String,
        /// The store; default `environments` under the state root.
        #[arg(long, value_name = "DIR")]
        store: Option<PathBuf>,
    },
    /// Write an environment's stored instances of one split as Harbor task
    /// folders, one directory each under --out, for a run that needs them on
    /// disk (`make env/measure`). The folders are a working copy: remove them
    /// when the run is done, the store keeps the instances.
    Export {
        /// The environment's id.
        #[arg(value_name = "ENV-ID")]
        environment: String,
        /// The split to write.
        #[arg(long, default_value = "dev", value_name = "train|dev")]
        split: String,
        /// The directory to write into (it must not hold files yet).
        #[arg(long, required = true, value_name = "DIR")]
        out: PathBuf,
        /// The store; default `environments` under the state root.
        #[arg(long, value_name = "DIR")]
        store: Option<PathBuf>,
    },
    /// Record a base policy's measured pass rate on an environment's dev
    /// instances and judge it: inside the band the environment is admitted
    /// to campaigns, outside it is rejected with the direction to turn the
    /// difficulty knobs.
    Band {
        /// The environment's id.
        #[arg(value_name = "ENV-ID")]
        environment: String,
        /// Who was measured (a model and its serving settings).
        #[arg(long, value_name = "NAME")]
        model: String,
        /// How many attempts were made.
        #[arg(long, value_name = "N")]
        attempts: u32,
        /// How many of them were graded solved.
        #[arg(long, value_name = "N")]
        passes: u32,
        /// The store; default `environments` under the state root.
        #[arg(long, value_name = "DIR")]
        store: Option<PathBuf>,
    },
}
