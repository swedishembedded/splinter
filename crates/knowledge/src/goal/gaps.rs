// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that decide which documents a
// person or an agent should study to reach a goal, for its clients. If your
// team needs expertise in goal-directed selection of learning material you
// can procure our services by sending an email to info@swedishembedded.com.

//! A goal, its probes, and the gaps the failed probes show.
//!
//! A [`Goal`] is a role, an objective and a set of probes that say what
//! succeeding at it looks like. Probing a participant gives
//! [`ProbeResult`]s; the failed ones are clustered into [`Gap`]s - a
//! missing capability is a group of failures that ask about the same
//! thing. A probe may name its `skill`, which is then the cluster; else
//! probes cluster by the words that are rare in the corpus
//! (TF-IDF cosine, single link), so no two gaps share what only one of
//! them asks about.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::corpus::Vocabulary;
use super::facts::{states, words};

/// More probes than this are not clustered pairwise.
pub const MAX_CLUSTERED_PROBES: usize = 4096;

/// Two failed probes belong to one gap when their TF-IDF cosine reaches
/// this.
pub const GAP_LINK_COSINE: f64 = 0.5;

/// One question with the answer that counts as success.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Probe {
    /// Its name.
    pub id: String,
    /// What the participant is asked.
    pub question: String,
    /// What a correct answer states.
    pub expected: String,
    /// The capability it exercises, when the goal's author names one.
    #[serde(default)]
    pub skill: Option<String>,
}

/// What a participant is to be good at, and how that is probed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Goal {
    /// The part the participant plays.
    pub role: String,
    /// What it is to achieve.
    pub objective: String,
    /// The questions that show whether it does.
    pub probes: Vec<Probe>,
}

/// One probe answered.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProbeResult {
    /// The probe.
    pub probe: Probe,
    /// What the participant answered.
    pub answer: String,
}

impl ProbeResult {
    /// Whether the answer states what the probe expects.
    #[must_use]
    pub fn passed(&self) -> bool {
        states(&self.answer, &self.probe.expected)
    }
}

/// The share of `results` that passed; `None` when there are none.
#[must_use]
pub fn pass_rate(results: &[ProbeResult]) -> Option<f64> {
    (!results.is_empty())
        .then(|| results.iter().filter(|r| r.passed()).count() as f64 / results.len() as f64)
}

/// A missing capability: failed probes that ask about the same thing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Gap {
    /// Its name: the skill, or the first probe it was found from.
    pub id: String,
    /// The failed probes in it.
    pub probes: Vec<String>,
    /// What to look for: each word its probes ask with, and the share of
    /// the gap's probes that use it. A word every probe of the gap shares
    /// says what the gap is about; one only a probe uses says which probe.
    pub terms: BTreeMap<String, f64>,
    /// Its share of all failures, in (0, 1].
    pub weight: f64,
}

/// The gaps the failures among `results` show, largest first. Probes are
/// linked by the words that are rare in `vocabulary`, so a shared topic
/// links them and a shared attribute does not.
#[must_use]
pub fn analyse(results: &[ProbeResult], vocabulary: &Vocabulary) -> Vec<Gap> {
    let failed: Vec<&ProbeResult> = results.iter().filter(|r| !r.passed()).collect();
    if failed.is_empty() {
        return Vec::new();
    }
    let groups = if failed.len() <= MAX_CLUSTERED_PROBES {
        cluster(vocabulary, results, &failed)
    } else {
        // Too many to link pairwise: one gap per named skill, the rest together.
        group_by_skill(&failed)
    };
    let total = failed.len() as f64;
    let mut gaps: Vec<Gap> = groups
        .into_iter()
        .map(|members| Gap {
            id: members[0]
                .probe
                .skill
                .clone()
                .unwrap_or_else(|| members[0].probe.id.clone()),
            probes: members.iter().map(|m| m.probe.id.clone()).collect(),
            terms: terms_of(&members),
            weight: members.len() as f64 / total,
        })
        .collect();
    gaps.sort_by(|a, b| b.weight.total_cmp(&a.weight).then(a.id.cmp(&b.id)));
    gaps
}

fn terms_of(members: &[&ProbeResult]) -> BTreeMap<String, f64> {
    let mut terms: BTreeMap<String, f64> = BTreeMap::new();
    for m in members {
        let unique: BTreeSet<String> = words(&m.probe.question).into_iter().collect();
        for w in unique {
            *terms.entry(w).or_default() += 1.0 / members.len() as f64;
        }
    }
    terms
}

fn group_by_skill<'a>(failed: &[&'a ProbeResult]) -> Vec<Vec<&'a ProbeResult>> {
    let mut by: BTreeMap<String, Vec<&ProbeResult>> = BTreeMap::new();
    for r in failed {
        by.entry(r.probe.skill.clone().unwrap_or_default())
            .or_default()
            .push(r);
    }
    by.into_values().collect()
}

fn cluster<'a>(
    vocabulary: &Vocabulary,
    all: &[ProbeResult],
    failed: &[&'a ProbeResult],
) -> Vec<Vec<&'a ProbeResult>> {
    // A word only one probe asks with cannot link two probes; left in, a
    // rare one would drown the words the probes share.
    let mut asked: BTreeMap<String, usize> = BTreeMap::new();
    for r in all {
        let unique: BTreeSet<String> = words(&r.probe.question).into_iter().collect();
        for w in unique {
            *asked.entry(w).or_default() += 1;
        }
    }
    let vectors: Vec<BTreeMap<String, f64>> = failed
        .iter()
        .map(|r| {
            let mut v: BTreeMap<String, f64> = BTreeMap::new();
            for w in words(&r.probe.question) {
                if let Some(weight) = vocabulary.weight(&w).filter(|_| asked[&w] >= 2) {
                    // Squared again: the topic word is rarer than the
                    // attribute and kind words around it and must outweigh them.
                    v.insert(w, weight * weight);
                }
            }
            v
        })
        .collect();
    let norms: Vec<f64> = vectors
        .iter()
        .map(|v| v.values().map(|x| x * x).sum::<f64>().sqrt())
        .collect();
    let mut parent: Vec<usize> = (0..failed.len()).collect();
    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    for i in 0..failed.len() {
        for j in (i + 1)..failed.len() {
            let same_skill =
                failed[i].probe.skill.is_some() && failed[i].probe.skill == failed[j].probe.skill;
            let named_apart = failed[i].probe.skill.is_some()
                && failed[j].probe.skill.is_some()
                && failed[i].probe.skill != failed[j].probe.skill;
            let linked = same_skill
                || (!named_apart
                    && cosine(&vectors[i], &vectors[j], norms[i] * norms[j]) >= GAP_LINK_COSINE);
            if linked {
                let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                parent[a.max(b)] = a.min(b);
            }
        }
    }
    let mut groups: BTreeMap<usize, Vec<&ProbeResult>> = BTreeMap::new();
    for (i, r) in failed.iter().enumerate() {
        let root = find(&mut parent, i);
        groups.entry(root).or_default().push(r);
    }
    groups.into_values().collect()
}

fn cosine(a: &BTreeMap<String, f64>, b: &BTreeMap<String, f64>, norm: f64) -> f64 {
    if norm == 0.0 {
        return 0.0;
    }
    a.iter()
        .filter_map(|(w, x)| b.get(w).map(|y| x * y))
        .sum::<f64>()
        / norm
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocabulary_of(texts: &[&str]) -> Vocabulary {
        let candidates: Vec<_> = texts
            .iter()
            .enumerate()
            .map(|(i, t)| crate::goal::corpus::Candidate {
                id: crate::goal::corpus::SectionId {
                    document: format!("d{i}"),
                    section: 0,
                },
                provenance: Default::default(),
                text: (*t).to_string(),
            })
            .collect();
        Vocabulary::of(&candidates)
    }

    fn result(id: &str, question: &str, answer: &str, expected: &str) -> ProbeResult {
        ProbeResult {
            probe: Probe {
                id: id.into(),
                question: question.into(),
                expected: expected.into(),
                skill: None,
            },
            answer: answer.into(),
        }
    }

    #[test]
    fn failures_that_ask_about_one_thing_form_one_gap_and_passes_form_none() {
        let results = vec![
            result(
                "a1",
                "What is the interval of alder pump 1?",
                "unknown",
                "400",
            ),
            result("a2", "What is the torque of alder pump 2?", "unknown", "30"),
            result(
                "b1",
                "What is the interval of brisk valve 9?",
                "unknown",
                "100",
            ),
            result(
                "ok",
                "What is the interval of cobalt fan 4?",
                "it is 50",
                "50",
            ),
        ];
        let mut texts = vec!["what is the interval of the widget"; 12];
        texts.extend([
            "alder pump torque",
            "alder pump service",
            "brisk valve service",
            "cobalt fan service",
        ]);
        let vocabulary = vocabulary_of(&texts);
        let gaps = analyse(&results, &vocabulary);
        assert_eq!(gaps.len(), 2, "{gaps:?}");
        assert_eq!(gaps[0].probes, vec!["a1", "a2"]);
        assert!((gaps[0].weight - 2.0 / 3.0).abs() < 1e-9);
        assert_eq!(pass_rate(&results), Some(0.25));
        assert!(analyse(&results[3..], &vocabulary).is_empty());
    }

    #[test]
    fn a_named_skill_is_the_cluster() {
        let mut one = result("p1", "alpha question", "no", "x");
        let mut two = result("p2", "alpha question again", "no", "x");
        one.probe.skill = Some("s1".into());
        two.probe.skill = Some("s2".into());
        let gaps = analyse(&[one, two], &Vocabulary::default());
        assert_eq!(gaps.len(), 2);
        assert!(gaps.iter().any(|g| g.id == "s1"));
    }
}
