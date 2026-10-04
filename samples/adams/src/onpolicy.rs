// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training language models on
// their own checked answers instead of a teacher's, for its clients. If your
// team needs expertise in rejection-sampling fine-tuning and on-policy
// preference data then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Training on what the student itself wrote.
//!
//! A teacher's answers, and answers broken by code, are not what the student
//! would say, so a student trained on them learns to imitate a style it does
//! not yet produce and to refuse errors it never makes. Sampling the student on
//! the training scenarios and grading each sample by the same rules gives
//! both halves from its own distribution: the samples that pass are further
//! supervised examples (rejection sampling), and a passing and a failing sample
//! of one prompt are a preference pair whose two answers differ in what the
//! student actually gets wrong, the length of the two kept alike so that
//! preferring the longer or shorter is not a way to win.

use std::collections::{BTreeMap, HashSet};

use serde_json::{json, Value};
use splinter_sdk::model::exam::Graded;
use splinter_sdk::model::report::SAMPLE_SEPARATOR;

use crate::datasets::{sft_record, shape, TransferTask};
use crate::persona::Framing;
use crate::respond;

/// How much longer or shorter than the chosen answer a rejected one may be, as
/// a ratio either way.
const LENGTH_RATIO: f64 = 1.33;

/// What a prompt's samples came to.
#[derive(Debug)]
pub struct Sampled<'a> {
    /// The question the samples answer.
    pub task: &'a TransferTask,
    /// Samples that broke no rule, each once.
    pub passing: Vec<String>,
    /// Samples that broke a rule, with the rules they broke.
    pub failing: Vec<(String, Vec<String>)>,
}

/// The samples in `results` gathered under the question each answers. A sample
/// that gave no answer or ran out of budget is neither: it says nothing about
/// what the student prefers to say.
#[must_use]
pub fn group<'a>(questions: &'a [TransferTask], results: &[Graded]) -> Vec<Sampled<'a>> {
    let mut by_item: BTreeMap<&str, Vec<&Graded>> = BTreeMap::new();
    for g in results {
        let item = g.id.split(SAMPLE_SEPARATOR).next().unwrap_or(&g.id);
        by_item.entry(item).or_default().push(g);
    }
    questions
        .iter()
        .filter_map(|task| {
            let samples = by_item.get(task.id.as_str())?;
            let mut sampled = Sampled {
                task,
                passing: Vec::new(),
                failing: Vec::new(),
            };
            for g in samples.iter().filter(|g| !g.truncated) {
                let Some(answer) = g.answer.as_deref().filter(|a| !a.trim().is_empty()) else {
                    continue;
                };
                if g.correct {
                    if !sampled.passing.iter().any(|p| p == answer) {
                        sampled.passing.push(answer.to_string());
                    }
                } else {
                    let broken: Vec<String> = g
                        .checks
                        .iter()
                        .filter(|(_, ok)| !**ok)
                        .map(|(rule, _)| rule.clone())
                        .collect();
                    sampled.failing.push((answer.to_string(), broken));
                }
            }
            Some(sampled)
        })
        .collect()
}

/// Supervised records from the passing samples: at most `per_prompt` distinct
/// ones of each prompt, shaped as any training answer is.
#[must_use]
pub fn rft_records(sampled: &[Sampled<'_>], per_prompt: usize) -> Vec<Value> {
    sampled
        .iter()
        .flat_map(|s| {
            s.passing
                .iter()
                .take(per_prompt)
                .map(|answer| sft_record(&s.task.prompt, &shape(answer), &s.task.group))
        })
        .collect()
}

fn words(text: &str) -> f64 {
    text.split_whitespace().count().max(1) as f64
}

/// Preference pairs from the samples of one prompt: a passing answer and a
/// failing one of nearly the same length, at most `per_prompt` pairs, each
/// failing answer breaking a different set of rules from those already used so
/// the pairs do not all say the same thing. A prompt whose samples all pass or
/// all fail has no contrast and gives none.
#[must_use]
pub fn onpolicy_pairs(sampled: &[Sampled<'_>], per_prompt: usize) -> Vec<Value> {
    let mut pairs = Vec::new();
    for s in sampled {
        let mut used: HashSet<Vec<String>> = HashSet::new();
        for (rejected, broken) in &s.failing {
            if used.len() >= per_prompt || used.contains(broken) {
                continue;
            }
            let nearest = s
                .passing
                .iter()
                .map(|p| shape(p))
                .map(|p| (p.clone(), (words(&p) / words(rejected)).ln().abs()))
                .filter(|(_, distance)| *distance <= LENGTH_RATIO.ln())
                .min_by(|a, b| a.1.total_cmp(&b.1));
            let Some((chosen, _)) = nearest else {
                continue;
            };
            used.insert(broken.clone());
            pairs.push(json!({
                "prompt": [
                    {"role": "system", "content": Framing::of_record(&s.task.prompt).system(respond::SYSTEM)},
                    {"role": "user", "content": s.task.prompt},
                ],
                "chosen": {"role": "assistant", "content": chosen},
                "rejected": {"role": "assistant", "content": rejected},
                "tools": [],
                "metadata": {
                    "group": s.task.group,
                    "source": "student_samples",
                    "mode": s.task.mode,
                    "broke": broken,
                    "scenario_id": s.task.scenario_id,
                },
            }));
        }
    }
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::{Mode, Slice};
    use crate::scenario::Case;
    use std::collections::BTreeMap;

    fn task(id: &str) -> TransferTask {
        TransferTask {
            id: id.into(),
            scenario_id: format!("s-{id}"),
            mode: Mode::Retrieval,
            prompt: format!("Situation: {id}."),
            case: Case::Clear,
            observations: Vec::new(),
            evidence_docs: Vec::new(),
            slice: Slice::Train,
            group: "P-1".into(),
        }
    }

    fn sample(id: &str, n: usize, text: &str, correct: bool, broke: &[&str]) -> Graded {
        Graded {
            id: format!("{id}@{n}"),
            kind: "transfer".into(),
            split: "train".into(),
            reference: "clear".into(),
            answer: Some(text.into()),
            correct,
            tokens: 10,
            truncated: false,
            seconds: 1.0,
            checks: broke
                .iter()
                .map(|r| ((*r).to_string(), false))
                .chain(std::iter::once(("applicability_stated".to_string(), true)))
                .collect::<BTreeMap<_, _>>(),
        }
    }

    const GOOD: &str = "Applicability: APPLIES\n\nI would write to every town and gather their answers before the Assembly meets.\n\nGrounding:\n- SOURCE_DIRECT: \"x\" [d1]";
    const GOOD_TOO: &str = "Applicability: APPLIES\n\nI would send to each town and collect what they say before the Assembly sits.\n\nGrounding:\n- SOURCE_DIRECT: \"x\" [d1]";
    const BAD: &str = "Applicability: DOES_NOT_APPLY\n\nI would do nothing at all about it and let the Assembly meet without the towns.\n\nGrounding:\n- SPECULATION: x";

    #[test]
    fn samples_are_gathered_under_their_question_and_a_cut_off_one_says_nothing() {
        let questions = [task("q1"), task("q2")];
        let mut cut = sample("q1", 3, "Applicability: APP", false, &["grounding_present"]);
        cut.truncated = true;
        let results = [
            sample("q1", 0, GOOD, true, &[]),
            sample("q1", 1, GOOD, true, &[]),
            sample("q1", 2, BAD, false, &["applicability_fits"]),
            cut,
        ];
        let sampled = group(&questions, &results);
        assert_eq!(sampled.len(), 1, "a question with no samples is left out");
        assert_eq!(sampled[0].task.id, "q1");
        assert_eq!(sampled[0].passing, [GOOD], "the same answer twice is one");
        assert_eq!(sampled[0].failing.len(), 1);
        assert_eq!(sampled[0].failing[0].1, ["applicability_fits"]);
    }

    #[test]
    fn at_most_a_few_distinct_passing_answers_of_a_prompt_are_taught() {
        let questions = [task("q1"), task("q2")];
        let results = [
            sample("q1", 0, GOOD, true, &[]),
            sample("q1", 1, GOOD_TOO, true, &[]),
            sample("q2", 0, BAD, false, &["applicability_fits"]),
        ];
        let sampled = group(&questions, &results);
        let records = rft_records(&sampled, 1);
        assert_eq!(records.len(), 1, "one passing answer of q1, none for q2");
        assert_eq!(records[0]["messages"][2]["content"], GOOD);
        assert_eq!(records[0]["messages"][2]["train"], true);
        assert_eq!(records[0]["metadata"]["group"], "P-1");
        assert_eq!(rft_records(&sampled, 2).len(), 2);
    }

    #[test]
    fn a_pair_is_a_passing_and_a_failing_answer_of_nearly_the_same_length_for_the_same_prompt() {
        let questions = [task("q1"), task("q2"), task("q3")];
        let long_bad = format!("{BAD} {}", "and so on ".repeat(60));
        let results = [
            sample("q1", 0, GOOD, true, &[]),
            sample("q1", 1, BAD, false, &["applicability_fits"]),
            // all pass: no contrast
            sample("q2", 0, GOOD, true, &[]),
            sample("q2", 1, GOOD_TOO, true, &[]),
            // the failing answer is far longer than the passing one
            sample("q3", 0, GOOD, true, &[]),
            sample("q3", 1, &long_bad, false, &["applicability_fits"]),
        ];
        let sampled = group(&questions, &results);
        let pairs = onpolicy_pairs(&sampled, 2);
        assert_eq!(pairs.len(), 1, "{pairs:?}");
        assert_eq!(pairs[0]["chosen"]["content"], GOOD);
        assert_eq!(pairs[0]["rejected"]["content"], BAD);
        assert_eq!(pairs[0]["metadata"]["broke"][0], "applicability_fits");
        assert_eq!(pairs[0]["prompt"][1]["content"], "Situation: q1.");
    }

    #[test]
    fn a_prompts_pairs_each_break_something_different() {
        let questions = [task("q1")];
        let other_bad = BAD.replace("DOES_NOT_APPLY", "PARTLY");
        let results = [
            sample("q1", 0, GOOD, true, &[]),
            sample("q1", 1, BAD, false, &["applicability_fits"]),
            sample("q1", 2, &other_bad, false, &["applicability_fits"]),
            sample("q1", 3, BAD, false, &["numbers_given"]),
        ];
        let pairs = onpolicy_pairs(&group(&questions, &results), 5);
        let broke: Vec<&Value> = pairs.iter().map(|p| &p["metadata"]["broke"]).collect();
        assert_eq!(pairs.len(), 2, "{broke:?}");
        assert_eq!(onpolicy_pairs(&group(&questions, &results), 1).len(), 1);
    }
}
