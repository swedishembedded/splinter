// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model-assisted dataset construction that code
// verifies for its clients. If your team needs expertise in letting a model
// draft training questions while deterministic gates decide what is kept, you
// can procure our services by sending an email to info@swedishembedded.com.

//! Producing scenarios: a writer model drafts the question for each chosen
//! passage and the gates in [`crate::scenarios`] decide whether it is kept.

use std::path::{Path, PathBuf};

use crate::ask::Answerer;
use crate::corpus::{families, is_exam_family, load_letters};
use crate::scenarios::{admit, best_passage, writer_prompt, Scenario};
use crate::tasks::{surname_of, PERSONA};

/// What the writer produces.
pub struct Options {
    /// Where the resources are.
    pub resources: PathBuf,
    /// The writer model's checkpoint.
    pub writer: PathBuf,
    /// Train scenarios wanted.
    pub train: usize,
    /// Exam scenarios wanted.
    pub exam: usize,
    /// The split seed.
    pub seed: u64,
}

/// The candidates of one split: a letter with a person to write to, an era
/// year and a passage that states a principle, in a seeded order.
fn candidates(
    resources: &Path,
    seed: u64,
    exam: bool,
) -> anyhow::Result<Vec<(Scenario, String, u16)>> {
    let letters = load_letters(resources)?;
    let family = families(&letters);
    let mut found = Vec::new();
    for (n, letter) in letters.iter().enumerate() {
        if family[n] != n || is_exam_family(&letter.id, seed, 20) != exam {
            continue;
        }
        let Some(surname) = surname_of(&letter.recipient) else {
            continue;
        };
        if !(1760..=1826).contains(&letter.year) {
            continue;
        }
        let Some(passage) = best_passage(letter) else {
            continue;
        };
        let source = format!("my letter to {} in {}", letter.recipient, letter.year);
        let id =
            blake3::hash(format!("{}:{passage}", letter.id).as_bytes()).to_hex()[..12].to_string();
        found.push((
            Scenario {
                id: format!("scenario-{id}"),
                family: letter.id.clone(),
                split: if exam { "exam" } else { "train" }.to_string(),
                question: String::new(),
                passage,
                source,
            },
            surname,
            letter.year,
        ));
    }
    found.sort_by_cached_key(|(s, _, _)| {
        blake3::hash(format!("{seed}:{}", s.id).as_bytes())
            .to_hex()
            .to_string()
    });
    Ok(found)
}

/// Writes `scenarios.jsonl` in `out`: every admitted scenario, train first.
/// Returns `(train kept, exam kept, questions rejected)`.
///
/// # Errors
/// The corpus or the writer model cannot be read, or a generation fails.
pub fn generate(options: &Options, out: &Path) -> anyhow::Result<(usize, usize, usize)> {
    let writer = Answerer::load(&options.writer, None, 4096)?;
    let runtime = tokio::runtime::Runtime::new()?;
    let mut kept: Vec<Scenario> = Vec::new();
    let mut rejected = 0;
    for (exam, wanted) in [(false, options.train), (true, options.exam)] {
        let mut taken = 0;
        for (mut scenario, surname, year) in candidates(&options.resources, options.seed, exam)? {
            if taken >= wanted {
                break;
            }
            let reply = runtime.block_on(writer.ask(
                PERSONA_FREE,
                &writer_prompt(&scenario.passage),
                160,
            ))?;
            match admit(&reply.text, &scenario.passage, &surname, year) {
                Ok(question) => {
                    scenario.question = question;
                    kept.push(scenario);
                    taken += 1;
                }
                Err(why) => {
                    rejected += 1;
                    eprintln!(
                        "rejected ({why:?}): {}",
                        reply.text.trim().replace('\n', " ")
                    );
                }
            }
            eprintln!(
                "[{}] kept {taken}/{wanted}",
                if exam { "exam" } else { "train" }
            );
        }
    }
    let lines: Vec<String> = kept
        .iter()
        .map(serde_json::to_string)
        .collect::<Result<_, _>>()?;
    std::fs::write(out.join("scenarios.jsonl"), lines.join("\n") + "\n")?;
    let count = |split: &str| kept.iter().filter(|s| s.split == split).count();
    Ok((count("train"), count("exam"), rejected))
}

/// The writer is a different model asked for a question, not the persona.
const PERSONA_FREE: &str = "You write clear, self-contained questions.";

/// Reads `scenarios.jsonl`.
///
/// # Errors
/// The file cannot be read or a line is not a scenario.
pub fn read_scenarios(path: &Path) -> anyhow::Result<Vec<Scenario>> {
    std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(|e| anyhow::anyhow!("{}: {e}", path.display())))
        .collect()
}

/// One Layer 2 training record: the persona and the question are not trained
/// on, his own quoted words are.
#[must_use]
pub fn sft_record(scenario: &Scenario) -> serde_json::Value {
    serde_json::json!({
        "messages": [
            {"role": "system", "content": PERSONA, "train": false},
            {"role": "user", "content": scenario.question, "train": false},
            {"role": "assistant", "content": scenario.voice(), "train": true},
        ],
        "tools": [],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_scenario_record_trains_only_on_the_quoted_answer() {
        let scenario = Scenario {
            id: "s".into(),
            family: "f".into(),
            split: "train".into(),
            question: "May a free people be trusted to judge?".into(),
            passage: "The people are the only safe depositories of their own liberty.".into(),
            source: "my letter to James Madison in 1789".into(),
        };
        let record = sft_record(&scenario);
        assert_eq!(record["messages"][2]["train"], true);
        assert_eq!(record["messages"][1]["train"], false);
        assert_eq!(record["messages"][2]["content"], scenario.voice().as_str());
    }
}
