// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements resumable collection of a model's answers
// for measurement for its clients. If your team needs expertise in measuring
// what a locally run model says, you can procure our services by sending an
// email to info@swedishembedded.com.

//! The policy's side of the screening: six answers to each fact's question,
//! and, for a fact that looks wrong enough to teach, its four probes
//! answered greedily and the statement reworded for the judge.
//!
//! Everything is appended to files as it is made and read back before
//! anything is asked again, so an interrupted screening costs nothing already
//! measured.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use splinter_sdk::agent::solve::Model;
use splinter_sdk::Context;

use crate::facts::Fact;
use crate::keys::missing;
use crate::policy::{Answer, Decoding, Policy};
use crate::probes::Probe;
use crate::roles::Role;
use crate::seal::{append_draft, read_draft};
use crate::writer::{write_paraphrase, write_probes};

/// The answers the policy gives each fact: one greedy and this many drawn.
pub const SAMPLES: usize = 5;

/// One answer to a fact's question.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Line {
    /// The fact.
    pub fact: String,
    /// Which of the six: 0 is the greedy one.
    pub n: usize,
    /// How it was decoded.
    pub decoding: Decoding,
    /// What was said.
    #[serde(flatten)]
    pub answer: Answer,
}

/// One greedy answer to a probe.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProbeLine {
    /// The probe's fact.
    pub fact: String,
    /// The probe's kind.
    pub kind: crate::probes::ProbeKind,
    /// What was said.
    #[serde(flatten)]
    pub answer: Answer,
}

/// The statement of a fact reworded.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Reworded {
    fact: String,
    statement: String,
}

/// Everything collected for a screening.
pub struct Collected {
    /// The six answers of each fact.
    pub lines: Vec<Line>,
    /// The probes written for the facts that looked wrong.
    pub probes: Vec<Probe>,
    /// The day-0 answers to those probes.
    pub probe_lines: Vec<ProbeLine>,
    /// Each such fact's statement reworded, by fact id.
    pub reworded: HashMap<String, String>,
    /// Facts the generator could not write admissible probes or a rewording
    /// for, with why: they cannot be screened to the standard, so they are
    /// discarded.
    pub unwritable: HashMap<String, String>,
}

/// A fact the generator could not write for.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Failure {
    fact: String,
    why: String,
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> anyhow::Result<Vec<T>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    std::fs::read_to_string(path)?
        .lines()
        .enumerate()
        .map(|(n, l)| {
            serde_json::from_str(l).with_context(|| format!("{} line {}", path.display(), n + 1))
        })
        .collect()
}

fn append_jsonl<T: Serialize>(path: &Path, value: &T) -> anyhow::Result<()> {
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(file, "{}", serde_json::to_string(value)?)?;
    file.flush()?;
    Ok(())
}

/// What collecting needs.
pub struct Plan<'a> {
    /// The run's output directory.
    pub out: &'a Path,
    /// The facts to put to the policy.
    pub facts: &'a [Fact],
    /// The persona.
    pub persona: &'a str,
    /// The model that writes probes and rewordings.
    pub generator: &'a Model,
}

/// Whether the screening writes and answers probes for a fact whose six
/// answers each lack a key: only for facts a role of wrong facts may take.
fn wants_probes(fact: &Fact) -> bool {
    matches!(
        fact.candidate_role,
        Role::Dev | Role::Test | Role::ControlUntaught
    )
}

/// Collects the policy's answers; see the module documentation.
///
/// # Errors
/// A model cannot run or a file cannot be read or written.
pub fn collect(ctx: &Context, policy: &Policy, plan: &Plan<'_>) -> anyhow::Result<Collected> {
    let paths: [PathBuf; 2] = [
        plan.out.join("screening-answers.jsonl"),
        plan.out.join("screening-probe-answers.jsonl"),
    ];
    let reworded_path = plan.out.join("screening-rewordings.jsonl");
    let mut lines: Vec<Line> = read_jsonl(&paths[0])?;
    for (at, fact) in plan.facts.iter().enumerate() {
        for n in 0..=SAMPLES {
            if lines.iter().any(|l| l.fact == fact.id && l.n == n) {
                continue;
            }
            let decoding = if n == 0 {
                Decoding::Greedy
            } else {
                Decoding::Sampled
            };
            let answer = policy.answer(ctx, &fact.question, decoding)?;
            eprintln!(
                "answer {}/{} fact {} #{n}: {:.0}s, {} chars",
                at + 1,
                plan.facts.len(),
                fact.id,
                answer.seconds,
                answer.text.as_deref().map_or(0, str::len)
            );
            let line = Line {
                fact: fact.id.clone(),
                n,
                decoding,
                answer,
            };
            append_jsonl(&paths[0], &line)?;
            lines.push(line);
        }
    }

    let mut probes = read_draft(plan.out)?;
    // What an answer must hold is read afresh from the fact, not from the draft.
    for probe in &mut probes {
        if let Some(fact) = plan.facts.iter().find(|f| f.id == probe.fact) {
            probe.keys =
                crate::writer::answer_keys(probe.kind, fact, &probe.question, plan.persona);
        }
    }
    let mut probe_lines: Vec<ProbeLine> = read_jsonl(&paths[1])?;
    let mut rewordings: Vec<Reworded> = read_jsonl(&reworded_path)?;
    let failures_path = plan.out.join("screening-unwritable.jsonl");
    let mut unwritable: Vec<Failure> = read_jsonl(&failures_path)?;
    for fact in plan.facts.iter().filter(|f| wants_probes(f)) {
        let answers: Vec<&Line> = lines.iter().filter(|l| l.fact == fact.id).collect();
        let every_answer_lacks_a_key = answers.iter().all(|l| {
            l.answer
                .text
                .as_deref()
                .is_none_or(|t| !missing(t, &fact.keys).is_empty())
        });
        if !every_answer_lacks_a_key {
            continue;
        }
        if unwritable.iter().any(|f| f.fact == fact.id) {
            continue;
        }
        let written = if probes.iter().any(|p| p.fact == fact.id) {
            Ok(None)
        } else {
            write_probes(ctx, plan.generator, fact, plan.persona).map(Some)
        };
        let reworded = if rewordings.iter().any(|r| r.fact == fact.id) {
            Ok(None)
        } else {
            write_paraphrase(ctx, plan.generator, fact).map(Some)
        };
        match (written, reworded) {
            (Ok(written), Ok(reworded)) => {
                if let Some(written) = written {
                    append_draft(plan.out, &written)?;
                    probes.extend(written);
                }
                if let Some(statement) = reworded {
                    let line = Reworded {
                        fact: fact.id.clone(),
                        statement,
                    };
                    append_jsonl(&reworded_path, &line)?;
                    rewordings.push(line);
                }
            }
            (a, b) => {
                let why = [a.err(), b.err()]
                    .into_iter()
                    .flatten()
                    .map(|e| format!("{e:#}"))
                    .collect::<Vec<_>>()
                    .join("; ");
                eprintln!("fact {} cannot be probed: {why}", fact.id);
                let failure = Failure {
                    fact: fact.id.clone(),
                    why,
                };
                append_jsonl(&failures_path, &failure)?;
                unwritable.push(failure);
                continue;
            }
        }
        for probe in probes.iter().filter(|p| p.fact == fact.id) {
            if probe_lines
                .iter()
                .any(|l| l.fact == fact.id && l.kind == probe.kind)
            {
                continue;
            }
            let answer = policy.answer(ctx, &probe.question, Decoding::Greedy)?;
            eprintln!(
                "probe {:?} of fact {}: {:.0}s",
                probe.kind, fact.id, answer.seconds
            );
            let line = ProbeLine {
                fact: fact.id.clone(),
                kind: probe.kind,
                answer,
            };
            append_jsonl(&paths[1], &line)?;
            probe_lines.push(line);
        }
    }
    Ok(Collected {
        lines,
        probes,
        probe_lines,
        reworded: rewordings
            .into_iter()
            .map(|r| (r.fact, r.statement))
            .collect(),
        unwritable: unwritable.into_iter().map(|f| (f.fact, f.why)).collect(),
    })
}
