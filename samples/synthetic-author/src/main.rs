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

use clap::{Parser, Subcommand};
use synthetic_author::persona::Persona;
use synthetic_author::scenario::{scenario_pool, TOPICS};

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
    }
}
