// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that run model learning as one
// resumable, multi-generation campaign with a single promotion gate and
// rollback, for its clients. If your team needs expertise in self-improving
// training loops you can procure our services by sending an email to
// info@swedishembedded.com.

//! The grammar of `campaign`.

use std::path::PathBuf;

use clap::Subcommand;

/// `campaign`.
#[derive(Debug, Subcommand)]
pub enum CampaignCommand {
    /// Run a campaign's generations: the champion practises on the
    /// examiner's train-split practice tasks it has not mastered, the graded
    /// trials are ingested, a dataset is curated over every generation's
    /// experience, a candidate is trained from the base as a served brain
    /// job, the examiner's signed verdicts on it are gated by the one
    /// promotion gate, and the champion is promoted or kept. Every stage is
    /// recorded before the next starts; run it again after a kill and it
    /// resumes at the first stage without a record, reusing brain's job by
    /// its resume key. Ctrl-C cancels the stage in progress.
    Run {
        /// The campaign file (TOML): its name, base, generations, stage
        /// settings, budget, `[practice]` and `[examine]`.
        #[arg(value_name = "FILE")]
        spec: PathBuf,
        /// Cards a training run holds, to count its GPU-hours.
        #[arg(long, default_value = "1", value_name = "N")]
        gpus: u32,
    },
    /// What a campaign has done: every generation's candidate and decision,
    /// the champion, its history and what was spent. Nothing runs.
    Status {
        /// The campaign file.
        #[arg(value_name = "FILE")]
        spec: PathBuf,
    },
    /// Move the champion back to the one a generation made (0: the base),
    /// recorded with its reason as one more entry of the champion's history.
    Rollback {
        /// The campaign file.
        #[arg(value_name = "FILE")]
        spec: PathBuf,
        /// The generation whose champion to move back to; 0 is the base.
        #[arg(long, value_name = "GENERATION")]
        to: u32,
        /// Why, as the history records it.
        #[arg(long, value_name = "TEXT")]
        reason: String,
    },
}
