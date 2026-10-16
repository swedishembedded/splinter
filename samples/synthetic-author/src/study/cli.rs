// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the measurement of fine-tuned language
// models for its clients. If your team needs expertise in pre-registering a
// replication study so that no choice can follow its results, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The `study` command family: the stages of the replication study as
//! subcommands, each run from the protocol frozen in the study directory.

use std::path::PathBuf;

use clap::{Args, Subcommand};

use super::power::{derive, Observed};
use super::protocol::{
    freeze, Frozen, ModelRoles, Protocol, SampledDecoding, Seeds, Spec, DEFAULT_ALPHA_LEVEL,
    DEFAULT_ARTICLES, DEFAULT_AUTHORS, DEFAULT_DRAWS, DEFAULT_LEARNING_RATE, DEFAULT_POWER,
    DEFAULT_REPLICATES, DEFAULT_STEPS, DEFAULT_TEMPERATURE, EARLIER_OBSERVATION, EARLIER_SOURCE,
};
use crate::manifest::SeedSource;

/// The stages of the replication study.
#[derive(Args)]
pub struct StudyArgs {
    /// The stage.
    #[command(subcommand)]
    pub stage: Stage,
}

/// One stage of the study.
#[derive(Subcommand)]
pub enum Stage {
    /// Write the protocol: draw the authors' and training seeds, size the
    /// examination, and record both, once. Repeating it with the same
    /// settings changes nothing; with others it is refused.
    Freeze(FreezeArgs),
}

/// Freeze the study protocol.
#[derive(Args)]
pub struct FreezeArgs {
    /// The study directory; created if absent.
    #[arg(long)]
    out: PathBuf,
    /// The model that writes the essays, as a model reference.
    #[arg(long)]
    writer: String,
    /// The model that reads each draft for the gate; another family.
    #[arg(long)]
    reader: String,
    /// The model that reads the decisions out of the essays; a third family.
    #[arg(long)]
    miner: String,
    /// How many authors to draw seeds for.
    #[arg(long, default_value_t = DEFAULT_AUTHORS, conflicts_with = "author_seeds")]
    authors: usize,
    /// The authors' seeds, comma separated, instead of drawing them.
    #[arg(long, value_delimiter = ',')]
    author_seeds: Option<Vec<u64>>,
    /// How many training seeds to draw for each author.
    #[arg(long, default_value_t = DEFAULT_REPLICATES, conflicts_with = "training_seeds")]
    training_seeds_per_author: usize,
    /// The training seeds every author is trained with, comma separated,
    /// instead of drawing them.
    #[arg(long, value_delimiter = ',')]
    training_seeds: Option<Vec<u64>>,
    /// Articles planned for each author's corpus.
    #[arg(long, default_value_t = DEFAULT_ARTICLES)]
    articles: usize,
    /// Optimizer steps of each adapter.
    #[arg(long, default_value_t = DEFAULT_STEPS)]
    steps: u32,
    /// The learning rate of each adapter.
    #[arg(long, default_value_t = DEFAULT_LEARNING_RATE)]
    learning_rate: f64,
    /// Times each item is asked under the sampled decoding; 0 leaves the
    /// sampled decoding out.
    #[arg(long, default_value_t = DEFAULT_DRAWS)]
    sampled_draws: usize,
    /// The temperature of the sampled decoding.
    #[arg(long, default_value_t = DEFAULT_TEMPERATURE)]
    temperature: f64,
    /// Items an earlier examination asked both arms of the comparison the
    /// examination is sized for.
    #[arg(long, default_value_t = EARLIER_OBSERVATION.items)]
    observed_items: usize,
    /// Of those, the items on which exactly one arm was right.
    #[arg(long, default_value_t = EARLIER_OBSERVATION.discordant)]
    observed_discordant: usize,
    /// Of those, the items the candidate was right on.
    #[arg(long, default_value_t = EARLIER_OBSERVATION.candidate_wins)]
    observed_wins: usize,
    /// The level of the one-sided tests the examination is sized for.
    #[arg(long, default_value_t = DEFAULT_ALPHA_LEVEL)]
    alpha: f64,
    /// The power the examination is sized for.
    #[arg(long, default_value_t = DEFAULT_POWER)]
    power: f64,
}

impl FreezeArgs {
    fn spec(&self) -> anyhow::Result<Spec> {
        let sizing = derive(
            Observed {
                items: self.observed_items,
                discordant: self.observed_discordant,
                candidate_wins: self.observed_wins,
            },
            EARLIER_SOURCE,
            self.alpha,
            self.power,
        )?;
        let mut template = Protocol::standard(
            ModelRoles {
                writer: self.writer.clone(),
                reader: self.reader.clone(),
                miner: self.miner.clone(),
            },
            sizing,
        );
        template.corpus.articles = self.articles;
        template.training.steps = self.steps;
        template.training.learning_rate = self.learning_rate;
        template.decoding.secondary = (self.sampled_draws > 0).then_some(SampledDecoding {
            temperature: self.temperature,
            draws: self.sampled_draws,
        });
        Ok(Spec {
            template,
            author_seeds: self
                .author_seeds
                .clone()
                .map_or(Seeds::Draw(self.authors), Seeds::Given),
            training_seeds: self
                .training_seeds
                .clone()
                .map_or(Seeds::Draw(self.training_seeds_per_author), Seeds::Given),
        })
    }
}

/// Runs the stage `args` names.
///
/// # Errors
/// The stage's own: see each stage.
pub fn run(args: &StudyArgs) -> anyhow::Result<()> {
    match &args.stage {
        Stage::Freeze(freeze_args) => freeze_stage(freeze_args),
    }
}

fn freeze_stage(args: &FreezeArgs) -> anyhow::Result<()> {
    let frozen = freeze(&args.out, &args.spec()?, &mut || {
        splinter_sdk::model::seed::draw().value()
    })?;
    print!("{}", describe(&frozen, &args.out));
    Ok(())
}

/// What the freeze prints: the digest, and every seed it drew or was given.
fn describe(frozen: &Frozen, root: &std::path::Path) -> String {
    let p = &frozen.protocol;
    let mut text = format!(
        "protocol {} in {} (digest {})\n",
        if frozen.created {
            "frozen"
        } else {
            "already frozen"
        },
        root.display(),
        frozen.digest,
    );
    text.push_str(&format!(
        "author seeds ({}) and training seeds ({}), recorded in the protocol:\n",
        origin(p.provenance.authors),
        origin(p.provenance.training),
    ));
    for author in &p.authors {
        let seeds: Vec<String> = author.training_seeds.iter().map(u64::to_string).collect();
        text.push_str(&format!(
            "  author seed {}: training seeds {}\n",
            author.seed,
            seeds.join(", ")
        ));
    }
    let sizing = &p.exam.sizing;
    text.push_str(&format!(
        "examination: {} items per kind ({} per author), power {:.3} at one-sided {} against {} of {} discordant, {:.1} expected pairs\n",
        p.exam.per_cell,
        sizing.items,
        sizing.achieved_power,
        sizing.alpha,
        sizing.observed.candidate_wins,
        sizing.observed.discordant,
        sizing.expected_discordant,
    ));
    text
}

fn origin(source: SeedSource) -> &'static str {
    match source {
        SeedSource::Drawn => "drawn",
        SeedSource::Given => "given",
    }
}
