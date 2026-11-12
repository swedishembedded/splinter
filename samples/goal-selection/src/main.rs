// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that decide which documents a
// person or an agent should study to reach a goal, for its clients. If your
// team needs expertise in goal-directed selection of learning material you
// can procure our services by sending an email to info@swedishembedded.com.

//! `goal-selection`: generate labelled corpora around a goal, run the
//! goal-directed loop on each with a reader, and compare what it learned with
//! reading everything, nothing, or the same words at random, with intervals
//! over the corpora.
//!
//! ```text
//! goal-selection --form property --corpora 10 --hard --signal fused
//! goal-selection --form behaviour --goal-topics 1 --other-topics 7 --corpora 10
//! goal-selection --reader model --embedder served --hard --corpora 5 \
//!     --brain-endpoint http://127.0.0.1:8801/v1 --brain-key-file KEY --out result.json
//! ```
//!
//! Corpus `i` is drawn from seed `base + i`; the base seed is drawn and
//! printed when not given.

mod model_reader;
mod served_embedder;

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::str::FromStr as _;

use anyhow::{bail, Context as _};
use clap::{Parser, ValueEnum};
use model_reader::ModelReader;
use served_embedder::{Served, Tracked};
use splinter_sdk::goal::compare::{compare, Arm, Inputs};
use splinter_sdk::goal::report::{summarise, Summary};
use splinter_sdk::goal::{
    Checks, InContext, LexiconEmbedder, Participant, Plan, Query, Reader, Retrieval, ScriptedReader,
};
use splinter_sdk::knowledge::goal::corpus::candidates;
use splinter_sdk::knowledge::goal::facts::Form;
use splinter_sdk::knowledge::goal::hard::Hardness;
use splinter_sdk::knowledge::goal::select::Signal;
use splinter_sdk::knowledge::goal::semantic::Semantic;
use splinter_sdk::knowledge::goal::source::PackSource;
use splinter_sdk::knowledge::goal::synthetic::{generate, Spec};
use splinter_sdk::knowledge::retrieve::Embedder;
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
    /// A stand-in that trusts the best-ranked, best-headed passage; seconds.
    Scripted,
    /// A language model served by brain.
    Model,
}

#[derive(Clone, Copy, PartialEq, ValueEnum)]
enum EmbedderArg {
    /// Knows only the corpus's synonym table; no model.
    Lexicon,
    /// An embedding model served by brain.
    Served,
}

#[derive(Clone, Copy, ValueEnum)]
enum SignalArg {
    /// The gaps' words.
    Lexical,
    /// Closeness in meaning.
    Semantic,
    /// Either.
    Fused,
}

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// What the corpora are about.
    #[arg(long, value_enum, default_value = "property")]
    form: FormArg,
    /// The first corpus's seed; drawn and printed when not given.
    #[arg(long)]
    seed: Option<u64>,
    /// How many corpora, each its own seed.
    #[arg(long, default_value_t = 10)]
    corpora: u64,
    /// Add the difficulties lexical selection is weak against: paraphrase,
    /// near-copies, partly stale manuals, decoys.
    #[arg(long)]
    hard: bool,
    /// What decides whether a section bears on a gap, and how the reader's
    /// memory is searched.
    #[arg(long, value_enum, default_value = "lexical")]
    signal: SignalArg,
    /// Who embeds, for `--signal semantic` and `fused`.
    #[arg(long, value_enum, default_value = "lexicon")]
    embedder: EmbedderArg,
    /// The embedding model for `--embedder served`.
    #[arg(long, default_value = "Qwen/Qwen3-Embedding-0.6B")]
    embedding_model: String,
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
    /// Print, for the first corpus, how relevant each kind of section looks to
    /// words alone and to meaning alone when nothing is known.
    #[arg(long)]
    diagnose: bool,
    /// Where to write each corpus's material by arm (`<seed>/<arm>.md`, the
    /// sections an arm learned, in corpus order) and its held-out questions
    /// (`<seed>/held-out.jsonl`), for training a model on each arm's material.
    #[arg(long)]
    export: Option<PathBuf>,
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

fn print_table(summary: &[Summary]) {
    let ci = |m: &splinter_sdk::goal::report::Metric| {
        format!("{:.2} [{:.2},{:.2}]", m.mean, m.low, m.high)
    };
    println!(
        "{:<24} {:<18} {:<18} {:<18} {:>7} {:<20}",
        "arm", "held-out", "precision", "recall", "words", "held-out vs selective"
    );
    for s in summary {
        println!(
            "{:<24} {:<18} {:<18} {:<18} {:>7.0} {:<20}",
            s.name,
            ci(&s.held_out),
            s.precision.as_ref().map_or("-".to_string(), ci),
            ci(&s.recall),
            s.words.mean,
            s.versus_reference.as_ref().map_or("-".to_string(), ci),
        );
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let base = splinter_sdk::seed::resolve("seed", cli.seed);
    let form = match cli.form {
        FormArg::Property => Form::Property,
        FormArg::Behaviour => Form::Behaviour,
    };
    let signal = match cli.signal {
        SignalArg::Lexical => Signal::Lexical,
        SignalArg::Semantic => Signal::Semantic,
        SignalArg::Fused => Signal::Fused,
    };
    let retrieval = match cli.signal {
        SignalArg::Lexical => Retrieval::Lexical,
        SignalArg::Semantic => Retrieval::Semantic,
        SignalArg::Fused => Retrieval::Fused,
    };
    let plan = Plan {
        checks: Checks::ALL,
        signal,
        rounds: cli.rounds,
        round_words: cli.round_words,
        total_words: cli.total_words,
        target: 0.95,
    };

    // What is shared across corpora: a runtime when a model or an embedder is served.
    let served = cli.reader == ReaderArg::Model || cli.embedder == EmbedderArg::Served;
    let runtime = if served {
        if cli.brain_endpoint.is_empty() {
            bail!("a served model or embedder needs --brain-endpoint");
        }
        let key_file = cli
            .brain_key_file
            .as_ref()
            .context("--brain-endpoint needs --brain-key-file")?;
        let key = BrainEndpoints::read_key(key_file).context("reading the key")?;
        let mut config = Config::from_env();
        config.brain = Some(
            BrainEndpoints::new(cli.brain_endpoint.clone(), key.clone()).context("endpoints")?,
        );
        let splinter = Splinter::builder(config)
            .build()
            .context("starting the runtime")?;
        Some((splinter, key))
    } else {
        None
    };
    let model_reader: Option<Rc<ModelReader>> = match (&runtime, cli.reader) {
        (Some((splinter, _)), ReaderArg::Model) => {
            let ctx = splinter.context();
            let reference = ModelRef::from_str(&format!("local:{}", cli.model)).context("model")?;
            let model = ctx.model(&reference).context("loading the model")?;
            Some(Rc::new(ModelReader::new(ctx, model.provider)))
        }
        _ => None,
    };
    let embedder_failures: Rc<RefCell<Vec<String>>> = Rc::default();
    let served_embedder: Option<Rc<dyn Embedder>> = match (&runtime, cli.embedder) {
        (Some((splinter, key)), EmbedderArg::Served) if signal != Signal::Lexical => {
            let served = Served::new(
                splinter.context(),
                &cli.brain_endpoint[0],
                key,
                &cli.embedding_model,
            );
            Some(Rc::new(Tracked::new(
                Rc::new(served),
                Rc::clone(&embedder_failures),
            )))
        }
        _ => None,
    };

    println!("seeds {base}..{}", base + cli.corpora - 1);
    let mut runs: Vec<Vec<Arm>> = Vec::new();
    for i in 0..cli.corpora {
        let mut spec = Spec::new(base + i, form);
        spec.goal_topics = cli.goal_topics;
        spec.other_topics = cli.other_topics;
        spec.units = cli.units;
        spec.known_share = if cli.reader == ReaderArg::Model {
            0.0
        } else {
            cli.known_share
        };
        if cli.hard {
            spec.hard = Hardness::ALL;
        }
        let corpus = generate(&spec);
        let cands = candidates(&corpus.documents);
        let source = PackSource::from_sentences(&corpus.documents, &cands);
        let embedder: Rc<dyn Embedder> = served_embedder
            .clone()
            .unwrap_or_else(|| Rc::new(LexiconEmbedder::new(corpus.lexicon.clone())));
        let semantic = (signal != Signal::Lexical)
            .then(|| Semantic::new(&*embedder, &cands))
            .transpose()
            .context("embedding the corpus")?;
        let known = corpus.base_knowledge.clone();
        let lexicon = corpus.lexicon.clone();
        let make = |provenance: bool| -> Box<dyn Participant> {
            let participant: InContext<Box<dyn Reader>> = InContext::new(match &model_reader {
                Some(reader) => Box::new(Shared(Rc::clone(reader))) as Box<dyn Reader>,
                None => {
                    Box::new(ScriptedReader::knowing(known.clone()).understanding(lexicon.clone()))
                }
            });
            let participant = if signal == Signal::Lexical {
                participant
            } else {
                participant.retrieving(retrieval, Rc::clone(&embedder))
            };
            Box::new(if provenance {
                participant.showing_provenance()
            } else {
                participant
            })
        };
        let arms = compare(
            &corpus,
            &Inputs {
                fresh: &|| make(false),
                aware: Some(&|| make(true)),
                source: &source,
                semantic: semantic.as_ref(),
            },
            &plan,
            base + i,
            cli.ablate,
        );
        if cli.diagnose && i == 0 {
            diagnose(&corpus, &cands, &source, &embedder)?;
        }
        if let Some(dir) = &cli.export {
            export(dir, base + i, &corpus, &cands, &arms)?;
        }
        println!("corpus {} done ({} sections)", base + i, cands.len());
        runs.push(arms);
    }

    if let Some(reader) = &model_reader {
        if !reader.failures().is_empty() {
            bail!(
                "{} model calls failed, the result is void; first: {}",
                reader.failures().len(),
                reader.failures()[0]
            );
        }
    }
    if !embedder_failures.borrow().is_empty() {
        bail!(
            "{} embedding calls failed, the result is void; first: {}",
            embedder_failures.borrow().len(),
            embedder_failures.borrow()[0]
        );
    }
    let summary = summarise(&runs, "selective", base);
    print_table(&summary);
    let usage = model_reader.as_ref().map(|r| r.usage());
    if let Some((calls, chars, tokens)) = usage {
        println!("model: {calls} calls, {chars} prompt characters, {tokens} tokens reported");
    }
    if let (Some(reader), Some(path)) = (&model_reader, &cli.transcript) {
        let lines: Vec<String> = reader
            .transcript()
            .iter()
            .map(ToString::to_string)
            .collect();
        std::fs::write(path, lines.join("\n")).with_context(|| path.display().to_string())?;
    }
    if let Some(path) = &cli.out {
        let result = serde_json::json!({
            "seed": base, "corpora": cli.corpora, "form": format!("{form:?}"), "hard": cli.hard,
            "signal": format!("{signal:?}"), "summary": summary, "runs": runs,
            "model_usage": usage.map(|(c, p, t)| serde_json::json!({"calls": c, "prompt_chars": p, "tokens": t})),
        });
        std::fs::write(path, serde_json::to_string_pretty(&result)?)
            .with_context(|| path.display().to_string())?;
    }
    Ok(())
}

/// Writes each arm's learned sections and the held-out questions of a corpus.
fn export(
    dir: &std::path::Path,
    seed: u64,
    corpus: &splinter_sdk::knowledge::goal::synthetic::Corpus,
    cands: &[splinter_sdk::knowledge::goal::corpus::Candidate],
    arms: &[Arm],
) -> anyhow::Result<()> {
    let dir = dir.join(seed.to_string());
    std::fs::create_dir_all(&dir).with_context(|| dir.display().to_string())?;
    for arm in arms {
        let text: Vec<&str> = cands
            .iter()
            .filter(|c| arm.learned.contains(&c.id))
            .map(|c| c.text.as_str())
            .collect();
        std::fs::write(dir.join(format!("{}.md", arm.name)), text.join("\n\n"))?;
    }
    let lines: Vec<String> = corpus
        .held_out
        .iter()
        .map(|p| {
            serde_json::json!({"id": p.id, "question": p.question, "expected": p.expected})
                .to_string()
        })
        .collect();
    std::fs::write(dir.join("held-out.jsonl"), lines.join("\n"))?;
    Ok(())
}

/// Nothing is known: every question is answered "unknown".
struct Ignorant;

impl splinter_sdk::knowledge::goal::select::Examinee for Ignorant {
    fn answer_all(&self, questions: &[String]) -> Vec<String> {
        questions.iter().map(|_| "unknown".to_string()).collect()
    }
}

/// Prints, per label, the share of sections the first round would take as
/// relevant by words alone and by meaning alone, and their mean relevance.
fn diagnose(
    corpus: &splinter_sdk::knowledge::goal::synthetic::Corpus,
    cands: &[splinter_sdk::knowledge::goal::corpus::Candidate],
    source: &PackSource,
    embedder: &Rc<dyn Embedder>,
) -> anyhow::Result<()> {
    use splinter_sdk::knowledge::goal::corpus::Vocabulary;
    use splinter_sdk::knowledge::goal::gaps::{analyse, ProbeResult};
    use splinter_sdk::knowledge::goal::select::{select, Config, Open};
    let results: Vec<ProbeResult> = corpus
        .goal
        .probes
        .iter()
        .map(|p| ProbeResult {
            probe: p.clone(),
            answer: "unknown".into(),
        })
        .collect();
    let vocabulary = Vocabulary::of(cands);
    let gaps = analyse(&results, &vocabulary);
    let semantic = Semantic::new(&**embedder, cands).context("embedding the corpus")?;
    for signal in [Signal::Lexical, Signal::Semantic] {
        let open = Open {
            gaps: &gaps,
            semantic: Some(&semantic),
            objective: &corpus.goal.objective,
        };
        let config = Config {
            signal,
            check_conflicts: false,
            check_novelty: false,
            ..Config::with_budget(usize::MAX)
        };
        let selection = select(
            cands,
            &Default::default(),
            &open,
            &vocabulary,
            &Ignorant,
            source,
            &config,
        );
        let mut by_label: std::collections::BTreeMap<String, (usize, usize, f64)> =
            Default::default();
        for d in &selection.decisions {
            let entry = by_label
                .entry(format!("{:?}", corpus.labels[&d.section]))
                .or_default();
            entry.0 += 1;
            entry.1 +=
                usize::from(d.relevance >= splinter_sdk::knowledge::goal::select::MIN_RELEVANCE);
            entry.2 += d.relevance;
        }
        for (label, (n, pass, sum)) in by_label {
            println!(
                "relevance by {signal:?}: {label:<11} {pass}/{n} pass the floor, mean {:.2}",
                sum / n as f64
            );
        }
    }
    Ok(())
}
