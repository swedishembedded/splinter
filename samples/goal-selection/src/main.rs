// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that decide which documents a
// person or an agent should study to reach a goal, for its clients. If your
// team needs expertise in goal-directed selection of learning material you
// can procure our services by sending an email to info@swedishembedded.com.

//! `goal-selection`: generate a labelled corpus around a goal, run the
//! goal-directed loop on it with a reader, and compare what it learned with
//! reading everything, nothing, or the same words at random.
//!
//! ```text
//! goal-selection --form property --reader scripted
//! goal-selection --form behaviour --goal-topics 1 --reader model \
//!     --brain-endpoint http://127.0.0.1:8803/v1 --brain-key-file KEY --out result.json
//! ```
//!
//! The seed is drawn and printed when not given.

mod model_reader;

use std::path::PathBuf;
use std::rc::Rc;
use std::str::FromStr as _;

use anyhow::{bail, Context as _};
use clap::{Parser, ValueEnum};
use model_reader::ModelReader;
use splinter_sdk::goal::behaviour::learned_fidelity;
use splinter_sdk::goal::compare::{compare, Arm};
use splinter_sdk::goal::{Checks, InContext, Participant, Plan, Query, Reader, ScriptedReader};
use splinter_sdk::knowledge::goal::corpus::candidates;
use splinter_sdk::knowledge::goal::facts::{Form, PatternFacts};
use splinter_sdk::knowledge::goal::synthetic::{generate, Spec};
use splinter_sdk::model::BrainEndpoints;
use splinter_sdk::vocabulary::model_ref::ModelRef;
use splinter_sdk::{Config, Splinter};

#[derive(Clone, Copy, ValueEnum)]
enum FormArg {
    /// Properties of equipment lines.
    Property,
    /// How a type of client behaves.
    Behaviour,
}

#[derive(Clone, Copy, PartialEq, ValueEnum)]
enum ReaderArg {
    /// A stand-in that trusts the best-ranked passage; seconds.
    Scripted,
    /// A language model served by brain.
    Model,
}

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// What the corpus is about.
    #[arg(long, value_enum, default_value = "property")]
    form: FormArg,
    /// Draws the corpus; drawn and printed when not given.
    #[arg(long)]
    seed: Option<u64>,
    /// Who reads the material.
    #[arg(long, value_enum, default_value = "scripted")]
    reader: ReaderArg,
    /// The model for `--reader model`.
    #[arg(long, default_value = "Qwen/Qwen3.8-27B-FP8")]
    model: String,
    /// A brain endpoint (an OpenAI base URL ending in /v1); repeat for more.
    #[arg(long)]
    brain_endpoint: Vec<String>,
    /// The file holding the key the endpoints are asked with.
    #[arg(long)]
    brain_key_file: Option<PathBuf>,
    /// Topics (equipment lines or client types) the goal is about.
    #[arg(long, default_value_t = 3)]
    goal_topics: usize,
    /// Topics the goal is not about.
    #[arg(long, default_value_t = 5)]
    other_topics: usize,
    /// Units per topic.
    #[arg(long, default_value_t = 10)]
    units: usize,
    /// The share of a goal topic the participant knows before it learns. A
    /// model knows none of an invented corpus: it is 0 for `--reader model`.
    #[arg(long, default_value_t = 0.3)]
    known_share: f64,
    /// The most rounds of probe, select, learn.
    #[arg(long, default_value_t = 6)]
    rounds: usize,
    /// The most words one round may choose.
    #[arg(long, default_value_t = 400)]
    round_words: usize,
    /// The most words the run may choose.
    #[arg(long, default_value_t = 4000)]
    total_words: usize,
    /// Also run the loop with each check off.
    #[arg(long)]
    ablate: bool,
    /// Where to write every question put to the model and its answer, one
    /// JSON object a line.
    #[arg(long)]
    transcript: Option<PathBuf>,
    /// Where to write the full result as JSON.
    #[arg(long)]
    out: Option<PathBuf>,
}

/// One reader shared by every arm, so the counts add up.
struct Shared(Rc<ModelReader>);

impl Reader for Shared {
    fn read_all(&self, queries: &[Query]) -> Vec<String> {
        self.0.read_all(queries)
    }
}

fn print_table(arms: &[Arm]) {
    println!(
        "{:<24} {:>5} {:>6} {:>6} {:>5} {:>8} {:>5} {:>6} {:>7}",
        "arm", "secs", "words", "asked", "dev", "held-out", "prec", "recall", "harmful"
    );
    let share = |v: Option<f64>| v.map_or("-".to_string(), |v| format!("{v:.2}"));
    for a in arms {
        println!(
            "{:<24} {:>5} {:>6} {:>6} {:>5.2} {:>8.2} {:>5} {:>6.2} {:>7}",
            a.name,
            a.sections,
            a.words,
            a.questions,
            a.dev_pass,
            a.held_out_pass,
            share(a.precision),
            a.recall,
            share(a.harmful)
        );
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let seed = splinter_sdk::seed::resolve("seed", cli.seed);
    let form = match cli.form {
        FormArg::Property => Form::Property,
        FormArg::Behaviour => Form::Behaviour,
    };
    let mut spec = Spec::new(seed, form);
    spec.goal_topics = cli.goal_topics;
    spec.other_topics = cli.other_topics;
    spec.units = cli.units;
    spec.known_share = if cli.reader == ReaderArg::Model {
        0.0
    } else {
        cli.known_share
    };
    let corpus = generate(&spec);
    let plan = Plan {
        checks: Checks::ALL,
        rounds: cli.rounds,
        round_words: cli.round_words,
        total_words: cli.total_words,
        target: 0.95,
    };
    println!(
        "seed {seed}: {} documents, {} sections, {} dev probes, {} held-out questions",
        corpus.documents.len(),
        candidates(&corpus.documents).len(),
        corpus.goal.probes.len(),
        corpus.held_out.len()
    );

    let (arms, usage, fidelity) = if cli.reader == ReaderArg::Model {
        if cli.brain_endpoint.is_empty() {
            bail!("--reader model needs --brain-endpoint");
        }
        let key_file = cli
            .brain_key_file
            .as_ref()
            .context("--brain-endpoint needs --brain-key-file")?;
        let key = BrainEndpoints::read_key(key_file).context("reading the key")?;
        let mut config = Config::from_env();
        config.brain =
            Some(BrainEndpoints::new(cli.brain_endpoint.clone(), key).context("endpoints")?);
        let splinter = Splinter::builder(config)
            .build()
            .context("starting the runtime")?;
        let ctx = splinter.context();
        let reference = ModelRef::from_str(&format!("local:{}", cli.model)).context("model")?;
        let model = ctx.model(&reference).context("loading the model")?;
        let reader = Rc::new(ModelReader::new(ctx, model.provider));
        let fresh =
            || -> Box<dyn Participant> { Box::new(InContext::new(Shared(Rc::clone(&reader)))) };
        let arms = compare(&corpus, &fresh, &PatternFacts, &plan, seed, cli.ablate);
        let fidelity = fidelity_of(
            &arms,
            &corpus,
            &Shared(Rc::clone(&reader)),
            form,
            cli.goal_topics,
        );
        if !reader.failures().is_empty() {
            bail!(
                "{} model calls failed, the result is void; first: {}",
                reader.failures().len(),
                reader.failures()[0]
            );
        }
        if let Some(path) = &cli.transcript {
            let lines: Vec<String> = reader
                .transcript()
                .iter()
                .map(ToString::to_string)
                .collect();
            std::fs::write(path, lines.join("\n")).with_context(|| path.display().to_string())?;
        }
        (arms, Some(reader.usage()), fidelity)
    } else {
        let known = corpus.base_knowledge.clone();
        let fresh = || -> Box<dyn Participant> {
            Box::new(InContext::new(ScriptedReader::knowing(known.clone())))
        };
        let arms = compare(&corpus, &fresh, &PatternFacts, &plan, seed, cli.ablate);
        let fidelity = fidelity_of(
            &arms,
            &corpus,
            &ScriptedReader::knowing(known.clone()),
            form,
            cli.goal_topics,
        );
        (arms, None, fidelity)
    };

    print_table(&arms);
    if let Some((calls, chars, tokens)) = usage {
        println!("model: {calls} calls, {chars} prompt characters, {tokens} tokens reported");
    }
    for (name, value) in &fidelity {
        println!("persona fidelity on held-out behaviour, {name}: {value:.2}");
    }
    if let Some(path) = &cli.out {
        let result = serde_json::json!({
            "seed": seed, "form": format!("{form:?}"), "arms": arms, "persona_fidelity": fidelity,
            "model_usage": usage.map(|(c, p, t)| serde_json::json!({"calls": c, "prompt_chars": p, "tokens": t})),
        });
        std::fs::write(path, serde_json::to_string_pretty(&result)?)
            .with_context(|| path.display().to_string())?;
    }
    Ok(())
}

/// For a corpus about one client type, the fidelity of the persona each arm's
/// material makes; empty for any other goal.
fn fidelity_of(
    arms: &[Arm],
    corpus: &splinter_sdk::knowledge::goal::synthetic::Corpus,
    reader: &dyn Reader,
    form: Form,
    goal_topics: usize,
) -> Vec<(String, f64)> {
    if form != Form::Behaviour || goal_topics != 1 {
        return Vec::new();
    }
    let cands = candidates(&corpus.documents);
    let Some(subject) = corpus.goal_topics.first() else {
        return Vec::new();
    };
    arms.iter()
        .filter_map(|a| {
            Some((
                a.name.clone(),
                learned_fidelity(
                    subject,
                    &cands,
                    &a.learned,
                    &PatternFacts,
                    reader,
                    &corpus.held_out,
                )?,
            ))
        })
        .collect()
}
