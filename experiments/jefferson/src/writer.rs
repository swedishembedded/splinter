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
use crate::scenarios::{
    admit, advice_cues, advice_passages, classifier_prompt, writer_prompt, Scenario,
};
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
        for passage in advice_passages(letter).into_iter().take(2) {
            let source = format!("my letter to {} in {}", letter.recipient, letter.year);
            let id = blake3::hash(format!("{}:{passage}", letter.id).as_bytes()).to_hex()[..12]
                .to_string();
            found.push((
                Scenario {
                    id: format!("scenario-{id}"),
                    family: letter.id.clone(),
                    split: if exam { "exam" } else { "train" }.to_string(),
                    question: String::new(),
                    passage,
                    source,
                },
                surname.clone(),
                letter.year,
            ));
        }
    }
    // The strongest advice first: the more counselling phrases a passage has,
    // the likelier it is advice, so the writer's time goes where it pays.
    found.sort_by_cached_key(|(s, _, _)| {
        (
            std::cmp::Reverse(advice_cues(&s.passage)),
            blake3::hash(format!("{seed}:{}", s.id).as_bytes())
                .to_hex()
                .to_string(),
        )
    });
    Ok(found)
}

/// What generation kept and turned away.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Generated {
    /// Train scenarios kept.
    pub train: usize,
    /// Exam scenarios kept.
    pub exam: usize,
    /// Candidates the writer model said are not advice.
    pub not_advice: usize,
    /// Questions the code gates refused.
    pub rejected: usize,
}

/// Writes `scenarios.jsonl` in `out`: every admitted scenario, train first.
/// A candidate passage is first put to the writer model as a yes-or-no
/// question (is the writer of the letter giving concrete advice?) and only a
/// yes goes on to have a question drafted; code then admits or refuses the
/// question.
///
/// # Errors
/// The corpus or the writer model cannot be read, or a generation fails.
pub fn generate(options: &Options, out: &Path) -> anyhow::Result<Generated> {
    let writer = Answerer::load(&options.writer, None, 4096)?;
    let runtime = tokio::runtime::Runtime::new()?;
    // Each scenario is written as it is kept: a long run that stops early
    // leaves everything it had made.
    let mut file = std::fs::File::create(out.join("scenarios.jsonl"))?;
    let mut kept: Vec<Scenario> = Vec::new();
    let (mut rejected, mut not_advice) = (0, 0);
    for (exam, wanted) in [(false, options.train), (true, options.exam)] {
        let mut taken = 0;
        for (mut scenario, surname, year) in candidates(&options.resources, options.seed, exam)? {
            if taken >= wanted {
                break;
            }
            let asked = runtime.block_on(writer.ask(
                PERSONA_FREE,
                &classifier_prompt(&scenario.passage),
                6,
            ))?;
            if crate::application::verdict(&asked.text) != Some(true) {
                not_advice += 1;
                continue;
            }
            let reply = runtime.block_on(writer.ask(
                PERSONA_FREE,
                &writer_prompt(&scenario.passage),
                220,
            ))?;
            match admit(&reply.text, &scenario.passage, &surname, year) {
                Ok(question) => {
                    scenario.question = question;
                    use std::io::Write as _;
                    writeln!(file, "{}", serde_json::to_string(&scenario)?)?;
                    file.flush()?;
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
                "[{}] kept {taken}/{wanted}, not advice {not_advice}",
                if exam { "exam" } else { "train" }
            );
        }
    }
    let count = |split: &str| kept.iter().filter(|s| s.split == split).count();
    Ok(Generated {
        train: count("train"),
        exam: count("exam"),
        not_advice,
        rejected,
    })
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
