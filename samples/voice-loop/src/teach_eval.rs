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
use splinter_sdk::agent::teach::{Reader, Teachable};
use splinter_sdk::agent::voice::VoiceAgent;
use splinter_sdk::measure::teaching_check::Check;
use splinter_sdk::vocabulary::digest::Digest;
use splinter_sdk::vocabulary::teaching::{
    parse_teaching, teaching_from_json, Ledger, Teaching, READER_PROMPT,
};

use splinter_sdk::model::speech::{
    BrainRecognizer, BrainSynthesizer, Recognizer, Synthesizer, DEFAULT_RECOGNIZER,
    DEFAULT_SYNTHESIZER,
};
use splinter_sdk::vocabulary::speech::{Portrayal, SpeakerProfile};

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
    /// Say every user turn aloud in a synthetic voice of each of these seeds and
    /// give the agent what a recogniser heard; without it the turns go in as text.
    #[arg(long, value_delimiter = ',')]
    voices: Vec<u64>,
    /// The synthesis model, in brain's model store (with `--voices`).
    #[arg(long, default_value = DEFAULT_SYNTHESIZER)]
    tts: String,
    /// The recognition model, in brain's model store (with `--voices`).
    #[arg(long, default_value = DEFAULT_RECOGNIZER)]
    asr: String,
    /// Also let the model read what the user says for teachings.
    #[arg(long)]
    model_reader: bool,
    /// Also keep the report here.
    #[arg(long)]
    report: Option<PathBuf>,
}

/// A user's turn said aloud and heard back: the words the agent gets.
struct Ears<'a> {
    synthesizer: &'a BrainSynthesizer,
    recognizer: &'a BrainRecognizer,
    speaker: SpeakerProfile,
}

impl Ears<'_> {
    fn hear(&self, said: &str) -> Result<String> {
        let clip = self.synthesizer.speak(said, &self.speaker)?;
        Ok(self.recognizer.transcribe(&clip)?.text)
    }
}

#[derive(Deserialize)]
struct Scenario {
    id: String,
    kind: String,
    /// What the user says to teach, in order; each is confirmed with a yes.
    teach: Vec<String>,
    /// Asked of the first session after the teaching and before it is taken
    /// back, so that the session's own replies show what was taught.
    #[serde(default)]
    primes: Vec<String>,
    /// Say "forget that" after the last of them.
    #[serde(default)]
    forget_last: bool,
    /// Ask the tests of that same session, with its history, instead of a new one.
    #[serde(default)]
    same_session: bool,
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
    /// The seed of the voice the turns were said in; none for text.
    voice: Option<u64>,
    /// Teaching utterances the recogniser did not hear as said: (said, heard).
    misheard: Vec<(String, String)>,
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

fn session<'a>(
    persona: &'a Persona,
    reader: Option<&'a Persona>,
    ledger: Ledger,
) -> Result<Teachable<'a, VoiceAgent>> {
    let teachable = Teachable::new(persona.voice_agent(None)?, ledger);
    Ok(match reader {
        Some(reader) => teachable.with_reader(model_reader(reader)),
        None => teachable,
    })
}

/// Words of `text` as a recogniser would be compared on: lower case, letters and digits.
fn plain(text: &str) -> String {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn run_scenario(
    persona: &Persona,
    reader: Option<&Persona>,
    ears: Option<&Ears>,
    scenario: &Scenario,
) -> Result<ScenarioResult> {
    let mut misheard = Vec::new();
    // What the agent gets of what the user said: the words, or what was heard of them.
    let mut heard = |said: &str, teaching: bool| -> Result<String> {
        let Some(ears) = ears else {
            return Ok(said.to_string());
        };
        let heard = ears.hear(said)?;
        if teaching && plain(said) != plain(&heard) {
            misheard.push((said.to_string(), heard.clone()));
        }
        Ok(heard)
    };
    let mut first = session(persona, reader, Ledger::new())?;
    let mut understood = 0;
    for said in &scenario.teach {
        let restated = first.hear(&heard(said, true)?, &mut |_| {})?;
        understood += usize::from(restated.ends_with("Is that right?"));
        first.hear(&heard("Yes.", false)?, &mut |_| {})?;
    }
    for prime in &scenario.primes {
        first.hear(&heard(prime, false)?, &mut |_| {})?;
    }
    if scenario.forget_last {
        first.hear(&heard("Forget that.", false)?, &mut |_| {})?;
    }
    let ledger = first.ledger().clone();
    let kept = ledger.active().count();
    let mut untaught = session(persona, reader, Ledger::new())?;
    let mut fresh = session(persona, reader, ledger)?;
    let taught = if scenario.same_session {
        &mut first
    } else {
        &mut fresh
    };
    let mut outcomes = Vec::new();
    for test in &scenario.tests {
        let asked = heard(&test.ask, false)?;
        let untaught_reply = untaught.hear(&asked, &mut |_| {})?;
        let taught_reply = taught.hear(&asked, &mut |_| {})?;
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
        voice: ears.map(|e| e.speaker.seed()),
        misheard,
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
    let reader = args
        .model_reader
        .then(|| reading_persona(&args.base))
        .transpose()?;
    let (synthesizer, recognizer) = if args.voices.is_empty() {
        (None, None)
    } else {
        (
            Some(BrainSynthesizer::load(&args.tts)?),
            Some(BrainRecognizer::load(&args.asr)?),
        )
    };
    let mut results = Vec::new();
    // Text turns once; spoken turns once per voice.
    let voices: Vec<Option<u64>> = if args.voices.is_empty() {
        vec![None]
    } else {
        args.voices.iter().map(|v| Some(*v)).collect()
    };
    for voice in voices {
        let ears = match (voice, synthesizer.as_ref(), recognizer.as_ref()) {
            (Some(seed), Some(synth), Some(recog)) => Some(Ears {
                synthesizer: synth,
                recognizer: recog,
                speaker: SpeakerProfile::new(seed, Portrayal::synthetic_theatrical()),
            }),
            _ => None,
        };
        for scenario in &scenarios {
            eprintln!(
                "teach-eval: {} ({}) voice {:?}",
                scenario.id, scenario.kind, voice
            );
            results.push(run_scenario(
                &persona,
                reader.as_ref(),
                ears.as_ref(),
                scenario,
            )?);
        }
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

/// The model's directory name, not where it happens to be on this machine.
fn path_name(path: &Path) -> String {
    path.file_name().map_or_else(
        || path.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    )
}

/// A reader of what the user said that asks `persona` (loaded under
/// [`READER_PROMPT`]) what is being taught, given what the agent said last.
pub fn model_reader(persona: &Persona) -> Reader<'_> {
    Box::new(move |said, previous| {
        let question = match previous {
            Some(reply) => format!("The assistant had just said: {reply}\nThe user said: {said}"),
            None => format!("The user said: {said}"),
        };
        persona
            .answer(&question)
            .ok()
            .and_then(|reply| teaching_from_json(&reply))
    })
}

/// A persona that reads what users say, on the model at `base`.
pub fn reading_persona(base: &Path) -> Result<Persona> {
    Ok(Persona::load("reader", base, None)?.under(READER_PROMPT.to_string()))
}

#[derive(Args)]
pub struct TeachReadArgs {
    /// The text model's checkpoint directory.
    #[arg(long)]
    base: PathBuf,
    /// The utterances, one JSON object per line: `id`, `said`, optional `previous`.
    #[arg(long)]
    utterances: PathBuf,
    /// Also keep the report here.
    #[arg(long)]
    report: Option<PathBuf>,
}

#[derive(Deserialize)]
struct Utterance {
    id: String,
    said: String,
    previous: Option<String>,
}

#[derive(Serialize)]
struct Reading {
    id: String,
    plain: Option<Teaching>,
    model: Option<Teaching>,
}

/// What the plain readers and the model reader each make of every utterance.
/// The model is asked of every one, so the two can be compared on all of them.
pub fn read(args: &TeachReadArgs) -> Result<()> {
    let text = std::fs::read_to_string(&args.utterances)
        .with_context(|| format!("reading {}", args.utterances.display()))?;
    let persona = reading_persona(&args.base)?;
    let reader = model_reader(&persona);
    let readings = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let u: Utterance = serde_json::from_str(line)?;
            Ok(Reading {
                plain: parse_teaching(&u.said),
                model: reader(&u.said, u.previous.as_deref()),
                id: u.id,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    emit(&readings, args.report.as_deref())
}
