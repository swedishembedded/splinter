// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements calibration of model judges against human
// labels, for its clients. If your team needs expertise in measuring whether
// an LLM judge can be trusted on a task, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Measuring the judge against a person: a sample of answers to label, blind to
//! which arm gave each and in a random order, and what the labels say of the
//! judge once they come back.
//!
//! The sample is stratified by arm, by what the judge said and by whether the
//! answer is a short one or a long one, so that every kind of answer the judge
//! is trusted on is in it. Each item shows the task, the reference passage and
//! the answer, and nothing that names the arm; a separate key keeps what the
//! judge said and which arm it was. The judge grades each answer alone, so it
//! has no position to be biased by; the length bias is measured from the
//! labels. Nothing here invents a label: the labels are the person's.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use splinter_core::digest::Digest;

use super::analysis::TaskRecord;

/// One answer to be labelled, as a person sees it.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Item {
    /// Names the item in the key, and says nothing of the arm.
    pub id: String,
    /// What the task asked.
    pub instruction: String,
    /// The passage the task was written from.
    pub reference: String,
    /// The answer.
    pub answer: String,
    /// What the labeller fills in: `right` when the answer gives what the
    /// reference says in reply to the task, `wrong` when it does not.
    pub label: Option<String>,
}

/// What the key keeps of an item.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Keyed {
    /// The task's address.
    pub task: String,
    /// Its family.
    pub family: String,
    /// The arm that answered.
    pub arm: String,
    /// What the judge said; `None` where it abstained.
    pub judged: Option<bool>,
    /// The answer's length in characters.
    pub chars: usize,
    /// Whether it is among the longer half of the sample.
    pub long: bool,
}

/// The tasks' wording, looked up by address.
pub trait Wording {
    /// The instruction and the reference of the task at `address`.
    fn of(&self, address: &str) -> Option<(String, String)>;
}

/// Up to `n` answers of `records`, stratified and shuffled, as the items to
/// label and the key that says which arm gave each. Only the greedy answer of
/// each arm is sampled, and only answers there is something to judge.
pub fn export(
    records: &[TaskRecord],
    wording: &dyn Wording,
    n: usize,
    seed: u64,
) -> (Vec<Item>, BTreeMap<String, Keyed>) {
    struct Candidate<'a> {
        record: &'a TaskRecord,
        arm: &'a str,
        judged: Option<bool>,
        chars: usize,
        text: &'a str,
    }
    let mut all: Vec<Candidate<'_>> = Vec::new();
    for record in records {
        for (arm, answers) in &record.arms {
            if let Some(first) = answers.first() {
                if let Some(text) = first.text.as_deref() {
                    all.push(Candidate {
                        record,
                        arm,
                        judged: first.judged,
                        chars: first.chars,
                        text,
                    });
                }
            }
        }
    }
    let mut lengths: Vec<usize> = all.iter().map(|c| c.chars).collect();
    lengths.sort_unstable();
    let median = lengths.get(lengths.len() / 2).copied().unwrap_or(0);
    let order = |key: &str| Digest::of(format!("{seed}:{key}").as_bytes()).to_string();
    let mut strata: BTreeMap<(String, Option<bool>, bool), Vec<Candidate<'_>>> = BTreeMap::new();
    for c in all {
        strata
            .entry((c.arm.to_string(), c.judged, c.chars >= median))
            .or_default()
            .push(c);
    }
    for members in strata.values_mut() {
        members.sort_by_cached_key(|c| order(&format!("{}:{}", c.record.task, c.arm)));
    }
    // Round-robin over the strata until the sample is as large as asked.
    let mut chosen: Vec<&Candidate<'_>> = Vec::new();
    let mut round = 0;
    while chosen.len() < n {
        let before = chosen.len();
        for members in strata.values() {
            if let Some(c) = members.get(round) {
                if chosen.len() < n {
                    chosen.push(c);
                }
            }
        }
        if chosen.len() == before {
            break;
        }
        round += 1;
    }
    chosen.sort_by_cached_key(|c| order(&format!("show:{}:{}", c.record.task, c.arm)));
    let mut items = Vec::new();
    let mut key = BTreeMap::new();
    for (position, c) in chosen.into_iter().enumerate() {
        let Some((instruction, reference)) = wording.of(&c.record.task) else {
            continue;
        };
        let id = format!("item-{position:04}");
        items.push(Item {
            id: id.clone(),
            instruction,
            reference,
            answer: c.text.to_string(),
            label: None,
        });
        key.insert(
            id,
            Keyed {
                task: c.record.task.clone(),
                family: c.record.family.clone(),
                arm: c.arm.to_string(),
                judged: c.judged,
                chars: c.chars,
                long: c.chars >= median,
            },
        );
    }
    (items, key)
}

/// What the labels say of the judge.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Agreement {
    /// Labelled items the judge also decided.
    pub items: usize,
    /// Items that were left unlabelled or that the judge abstained on.
    pub left_out: usize,
    /// The share on which the judge and the person agree.
    pub agreement: f64,
    /// Cohen's kappa: agreement beyond what the two would reach by chance.
    pub kappa: Option<f64>,
    /// Answers the judge called right that the person called wrong, over the
    /// answers the judge called right: false supported.
    pub false_right_rate: Option<f64>,
    /// Answers the judge called wrong that the person called right, over the
    /// answers the judge called wrong: false unsupported.
    pub false_wrong_rate: Option<f64>,
    /// Agreement on the shorter half of the answers and on the longer half.
    pub agreement_short: Option<f64>,
    /// Agreement on the longer half.
    pub agreement_long: Option<f64>,
    /// How much likelier the judge is than the person to call a longer answer
    /// right than a shorter one: the share it calls right among long answers
    /// minus among short, less the same for the person. Zero is no length
    /// bias; a positive value is the judge favouring length.
    pub length_bias: Option<f64>,
    /// A judge that grades each answer alone has no position to favour.
    pub position_bias: &'static str,
}

/// Compares the judge's verdicts in `key` with the `labels` (item id to
/// `right` or `wrong`).
pub fn agreement(labels: &BTreeMap<String, String>, key: &BTreeMap<String, Keyed>) -> Agreement {
    let mut pairs: Vec<(bool, bool, bool)> = Vec::new(); // judge, person, long
    let mut left_out = 0;
    for (id, keyed) in key {
        let person = match labels.get(id).map(String::as_str) {
            Some("right") => Some(true),
            Some("wrong") => Some(false),
            _ => None,
        };
        match (keyed.judged, person) {
            (Some(judge), Some(person)) => pairs.push((judge, person, keyed.long)),
            _ => left_out += 1,
        }
    }
    let n = pairs.len() as f64;
    let share = |hits: usize, of: usize| (of > 0).then(|| hits as f64 / of as f64);
    let agreed = pairs.iter().filter(|(j, p, _)| j == p).count();
    let judge_right = pairs.iter().filter(|(j, _, _)| *j).count();
    let person_right = pairs.iter().filter(|(_, p, _)| *p).count();
    let expected = if pairs.is_empty() {
        0.0
    } else {
        let (a, b) = (judge_right as f64 / n, person_right as f64 / n);
        a * b + (1.0 - a) * (1.0 - b)
    };
    let observed = if pairs.is_empty() {
        0.0
    } else {
        agreed as f64 / n
    };
    let long = |is_long: bool| pairs.iter().filter(move |(_, _, l)| *l == is_long);
    let rate_right = |is_long: bool, who: fn(&(bool, bool, bool)) -> bool| {
        let members: Vec<_> = long(is_long).collect();
        (!members.is_empty())
            .then(|| members.iter().filter(|m| who(m)).count() as f64 / members.len() as f64)
    };
    let length_bias = rate_right(true, |m| m.0)
        .zip(rate_right(false, |m| m.0))
        .zip(rate_right(true, |m| m.1).zip(rate_right(false, |m| m.1)))
        .map(|((jl, js), (pl, ps))| (jl - js) - (pl - ps));
    Agreement {
        items: pairs.len(),
        left_out,
        agreement: observed,
        kappa: (expected < 1.0 && !pairs.is_empty())
            .then(|| (observed - expected) / (1.0 - expected)),
        false_right_rate: share(
            pairs.iter().filter(|(j, p, _)| *j && !*p).count(),
            judge_right,
        ),
        false_wrong_rate: share(
            pairs.iter().filter(|(j, p, _)| !*j && *p).count(),
            pairs.len() - judge_right,
        ),
        agreement_short: share(
            long(false).filter(|(j, p, _)| j == p).count(),
            long(false).count(),
        ),
        agreement_long: share(
            long(true).filter(|(j, p, _)| j == p).count(),
            long(true).count(),
        ),
        length_bias,
        position_bias: "not applicable: each answer is judged alone",
    }
}

#[cfg(test)]
mod tests {
    use super::super::analysis::Answer;
    use super::*;

    struct Words;
    impl Wording for Words {
        fn of(&self, address: &str) -> Option<(String, String)> {
            Some((format!("question {address}"), format!("passage {address}")))
        }
    }

    fn record(n: usize) -> TaskRecord {
        let answer = |judged: bool, chars: usize| {
            vec![Answer {
                judged: Some(judged),
                grounded: Some(true),
                chars,
                text: Some("x".repeat(chars)),
            }]
        };
        TaskRecord {
            task: format!("t{n}"),
            family: format!("f{}", n / 2),
            arms: [
                ("base".to_string(), answer(n.is_multiple_of(2), 20 + n)),
                (
                    "candidate".to_string(),
                    answer(n.is_multiple_of(3), 200 + n),
                ),
            ]
            .into(),
        }
    }

    #[test]
    fn the_sample_is_blind_stratified_and_reproducible() {
        let records: Vec<TaskRecord> = (0..40).map(record).collect();
        let (items, key) = export(&records, &Words, 24, 3);
        assert_eq!((items.len(), key.len()), (24, 24));
        // Nothing an item shows names an arm.
        for item in &items {
            let shown = serde_json::to_string(item).unwrap();
            assert!(
                !shown.contains("base") && !shown.contains("candidate"),
                "{shown}"
            );
            assert_eq!(item.label, None);
        }
        // Every stratum is in it: both arms, both verdicts.
        let arms: std::collections::BTreeSet<_> = key.values().map(|k| k.arm.as_str()).collect();
        assert_eq!(arms.len(), 2);
        assert!(key.values().any(|k| k.judged == Some(true)));
        assert!(key.values().any(|k| k.judged == Some(false)));
        assert!(key.values().any(|k| k.long) && key.values().any(|k| !k.long));
        assert_eq!(export(&records, &Words, 24, 3).0, items);
        assert_ne!(
            export(&records, &Words, 24, 4).0,
            items,
            "another seed, another sample"
        );
    }

    #[test]
    fn a_judge_that_agrees_with_the_person_has_kappa_one_and_one_that_favours_length_shows_it() {
        let key: BTreeMap<String, Keyed> = (0..8)
            .map(|n| {
                (
                    format!("i{n}"),
                    Keyed {
                        task: format!("t{n}"),
                        family: "f".into(),
                        arm: "a".into(),
                        judged: Some(n % 2 == 0),
                        chars: 10,
                        long: n >= 4,
                    },
                )
            })
            .collect();
        let same: BTreeMap<String, String> = key
            .iter()
            .map(|(id, k)| {
                (
                    id.clone(),
                    if k.judged == Some(true) {
                        "right"
                    } else {
                        "wrong"
                    }
                    .to_string(),
                )
            })
            .collect();
        let perfect = agreement(&same, &key);
        assert_eq!((perfect.items, perfect.agreement), (8, 1.0));
        assert_eq!(perfect.kappa, Some(1.0));
        assert_eq!(perfect.false_right_rate, Some(0.0));
        // The person calls every answer wrong; the judge calls the long ones
        // right: false supported on the long answers, and a length bias.
        let key2: BTreeMap<String, Keyed> = key
            .into_iter()
            .map(|(id, mut k)| {
                k.judged = Some(k.long);
                (id, k)
            })
            .collect();
        let all_wrong: BTreeMap<String, String> = key2
            .keys()
            .map(|id| (id.clone(), "wrong".to_string()))
            .collect();
        let biased = agreement(&all_wrong, &key2);
        assert_eq!(biased.false_right_rate, Some(1.0));
        assert_eq!(biased.false_wrong_rate, Some(0.0));
        assert_eq!(biased.length_bias, Some(1.0));
        assert_eq!(biased.agreement_short, Some(1.0));
        assert_eq!(biased.agreement_long, Some(0.0));
    }
}
