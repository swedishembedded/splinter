// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements evaluations of whether an agent learns from
// its users, for its clients. If your team needs expertise in measuring that
// an agent kept what it was taught, you can procure our services by sending
// an email to info@swedishembedded.com.

//! `voice-loop teach-eval`: does the agent do what it was taught?
//!
//! Each scenario is a list of things the user says to teach the agent, and
//! questions with a check on the reply. It is run twice on the same model: once
//! by an agent that was taught nothing, and once by a new session whose ledger
//! holds what a first session was taught by speaking the scenario's utterances
//! to it, each confirmed with a yes. The gap between the two is what teaching
//! did. The utterances go in as text; the spoken path adds recognition in front
//! and changes nothing after it.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Args;
use serde::{Deserialize, Serialize};
use splinter_sdk::agent::teach::Teachable;
use splinter_sdk::agent::voice::VoiceAgent;
use splinter_sdk::measure::teaching_check::Check;
use splinter_sdk::vocabulary::digest::Digest;
use splinter_sdk::vocabulary::teaching::Ledger;

use crate::batch::{emit, Persona};

/// The system turn of the agent under test: nothing about a person to portray.
const SYSTEM: &str = "You are a helpful voice assistant. Your answers are spoken aloud, so keep them short and plain.";

#[derive(Args)]
pub struct TeachEvalArgs {
    /// The text model's checkpoint directory.
    #[arg(long)]
    base: PathBuf,
    /// The scenarios, one JSON object per line.
    #[arg(long)]
    scenarios: PathBuf,
    /// Also keep the report here.
    #[arg(long)]
    report: Option<PathBuf>,
}

#[derive(Deserialize)]
struct Scenario {
    id: String,
    kind: String,
    /// What the user says to teach, in order; each is confirmed with a yes.
    teach: Vec<String>,
    /// Say "forget that" after the last of them.
    #[serde(default)]
    forget_last: bool,
    tests: Vec<Test>,
}

#[derive(Deserialize)]
struct Test {
    ask: String,
    check: Check,
}

#[derive(Serialize)]
struct Outcome {
    ask: String,
    untaught_reply: String,
    untaught_passes: bool,
    taught_reply: String,
    taught_passes: bool,
}

#[derive(Serialize)]
struct ScenarioResult {
    id: String,
    kind: String,
    /// How many of the teaching utterances the agent restated for confirmation.
    understood: usize,
    teachings: usize,
    kept: usize,
    outcomes: Vec<Outcome>,
}

#[derive(Serialize)]
struct Report {
    scenarios_digest: Digest,
    base: String,
    system: &'static str,
    results: Vec<ScenarioResult>,
}

fn session(persona: &Persona, ledger: Ledger) -> Result<Teachable<VoiceAgent>> {
    Ok(Teachable::new(persona.voice_agent(None)?, ledger))
}

fn run_scenario(persona: &Persona, scenario: &Scenario) -> Result<ScenarioResult> {
    let mut first = session(persona, Ledger::new())?;
    let mut understood = 0;
    for said in &scenario.teach {
        let restated = first.hear(said, &mut |_| {})?;
        understood += usize::from(restated.ends_with("Is that right?"));
        first.hear("Yes.", &mut |_| {})?;
    }
    if scenario.forget_last {
        first.hear("Forget that.", &mut |_| {})?;
    }
    let ledger = first.ledger().clone();
    let kept = ledger.active().count();
    let mut untaught = session(persona, Ledger::new())?;
    let mut taught = session(persona, ledger)?;
    let mut outcomes = Vec::new();
    for test in &scenario.tests {
        let untaught_reply = untaught.hear(&test.ask, &mut |_| {})?;
        let taught_reply = taught.hear(&test.ask, &mut |_| {})?;
        outcomes.push(Outcome {
            ask: test.ask.clone(),
            untaught_passes: test.check.passes(&untaught_reply),
            taught_passes: test.check.passes(&taught_reply),
            untaught_reply,
            taught_reply,
        });
    }
    Ok(ScenarioResult {
        id: scenario.id.clone(),
        kind: scenario.kind.clone(),
        understood,
        teachings: scenario.teach.len(),
        kept,
        outcomes,
    })
}

pub fn run(args: &TeachEvalArgs) -> Result<()> {
    let text = std::fs::read_to_string(&args.scenarios)
        .with_context(|| format!("reading {}", args.scenarios.display()))?;
    let scenarios: Vec<Scenario> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()
        .with_context(|| format!("{} is not a file of scenarios", args.scenarios.display()))?;
    let persona = Persona::load("assistant", &args.base, None)?.under(SYSTEM.to_string());
    let mut results = Vec::new();
    for scenario in &scenarios {
        eprintln!("teach-eval: {} ({})", scenario.id, scenario.kind);
        results.push(run_scenario(&persona, scenario)?);
    }
    emit(
        &Report {
            scenarios_digest: Digest::of(text.as_bytes()),
            base: path_name(&args.base),
            system: SYSTEM,
            results,
        },
        args.report.as_deref(),
    )
}

fn path_name(path: &Path) -> String {
    path.display().to_string()
}
