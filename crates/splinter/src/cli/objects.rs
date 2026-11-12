// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `objects ...`.

use clap::Subcommand;
use splinter_sdk::vocabulary::model_ref::ModelRef;

use super::model_ref;

#[derive(Debug, Subcommand)]
pub enum ObjectsCommand {
    /// Ask a model for the typed objects each document of a directory states,
    /// one document at a time in date order. Admission is code: a quote must
    /// be verbatim in its document, every number in an object must be in its
    /// quote, names must be identifiers (and outcomes in --outcomes when
    /// given), and objects the documents' headers (authority, date, version)
    /// cannot separate stay flagged as conflicts. A reply with refused
    /// proposals is sent back with the reasons, up to --repairs times. Writes
    /// the pack as JSON.
    Propose {
        /// A directory of Markdown documents; a `---` header may give
        /// `authority`, `version` and `date`.
        #[arg(value_name = "DIR")]
        source: std::path::PathBuf,
        /// Where the pack is written.
        #[arg(long, value_name = "FILE")]
        out: std::path::PathBuf,
        /// The pack's name.
        #[arg(long, default_value = "pack", value_name = "ID")]
        pack_id: String,
        /// The outcomes a rule may conclude, comma separated; open when absent.
        #[arg(long, value_delimiter = ',', value_name = "NAME,...")]
        outcomes: Vec<String>,
        /// How many times a reply with refused proposals is sent back, per document.
        #[arg(long, default_value_t = 2, value_name = "N")]
        repairs: u32,
        /// Ask for every kind in one pass instead of one pass per family of kinds
        /// (limits and exceptions, formulas, periods and steps, facts and tables,
        /// behaviour). One pass per family finds far more; one pass is quicker.
        #[arg(long)]
        single_pass: bool,
        /// Leave out the worked example, from an unrelated domain, that each
        /// pass shows the model.
        #[arg(long)]
        no_examples: bool,
        /// Put each document in runs of paragraphs of at most this many words, one run at a time.
        #[arg(long, value_name = "WORDS")]
        chunk_words: Option<usize>,
        /// The model asked.
        #[arg(long, value_parser = model_ref, default_value_t = ModelRef::policy_default(), value_name = "REF")]
        policy: ModelRef,
    },
}
