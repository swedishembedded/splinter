// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that decide which documents a
// person or an agent should study to reach a goal, for its clients. If your
// team needs expertise in goal-directed selection of learning material you
// can procure our services by sending an email to info@swedishembedded.com.

//! Which sections to learn from, under a budget, and why.
//!
//! Every candidate section gets exactly one [`Decision`] carrying the
//! reason. In the order they are checked:
//!
//! 1. **Relevance** - the share of each open [`Gap`]'s words (weighted by
//!    rarity and by how many of its probes use them) a section has,
//!    weighted by the gap's share of the failures. Below [`Config::min_relevance`] the section is
//!    [`Verdict::Irrelevant`].
//!    A section that names none of the words the goal's objective names its
//!    domain with (words few sections have) is capped below the floor; one
//!    that names some is at least at the floor.
//! 2. **Conflict** - sections that state different values for one fact key
//!    (anywhere in the corpus, learned sections included) are ranked by
//!    document authority, then date, then version. A section whose fact
//!    loses is [`Verdict::Contradicted`] when the winner has higher
//!    authority and [`Verdict::Superseded`] when it is newer; a tie is
//!    [`Verdict::Contested`].
//! 3. **Novelty** - the participant is asked the question each fact of the
//!    section answers; the share it gets right is what the section would
//!    teach it that it knows already. Below [`Config::min_novelty`] the
//!    section is [`Verdict::Known`]. A section with no checkable fact has no
//!    novelty (`None`) and ranks at [`UNMEASURED_NOVELTY`].
//! 4. **Duplicate** - every fact already learned or chosen this round.
//! 5. **Budget** - the rest, by value per word, until the words run out.
//!
//! Value is relevance times novelty. The participant's answers are the
//! only model work; relevance, conflicts and the budget are code.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde::{Deserialize, Serialize};

use super::corpus::{Candidate, SectionId, Vocabulary};
use super::facts::{states, words, Fact, FactExtractor, FactKey};
use super::gaps::Gap;

/// Something that answers questions: the participant being measured.
pub trait Examinee {
    /// One answer per question, in order. Implementations that can batch do.
    fn answer_all(&self, questions: &[String]) -> Vec<String>;
}

/// The knobs of one selection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    /// The most words this selection may choose.
    pub budget_words: usize,
    /// The least normalised relevance a section needs.
    pub min_relevance: f64,
    /// The least novelty a section needs.
    pub min_novelty: f64,
    /// Whether contradicted and superseded sections are left out. Off only
    /// to measure what the check contributes.
    pub check_conflicts: bool,
    /// Whether the participant is asked what it knows already. Off only to
    /// measure what the check contributes.
    pub check_novelty: bool,
}

/// The default relevance floor.
pub const MIN_RELEVANCE: f64 = 0.5;
/// The default novelty floor.
pub const MIN_NOVELTY: f64 = 0.25;
/// A word of the goal's objective names its domain when at most this share
/// of the sections have it.
pub const DOMAIN_TERM_SHARE: f64 = 0.25;
/// The most relevance a section can have that mentions none of the
/// objective's domain words; below [`MIN_RELEVANCE`], so it is left out.
pub const OFF_DOMAIN: f64 = 0.4;
/// The least relevance a section that mentions the objective's domain
/// words has; the floor, so it is kept.
pub const IN_DOMAIN: f64 = MIN_RELEVANCE;
/// The novelty assumed, for ranking only, of a section stating nothing the
/// participant can be asked about: it is worth less than one shown to be new.
pub const UNMEASURED_NOVELTY: f64 = 0.5;

impl Config {
    /// The defaults with a budget of `budget_words`.
    #[must_use]
    pub fn with_budget(budget_words: usize) -> Self {
        Self {
            budget_words,
            min_relevance: MIN_RELEVANCE,
            min_novelty: MIN_NOVELTY,
            check_conflicts: true,
            check_novelty: true,
        }
    }
}

/// What was decided about a section, and the reason.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Verdict {
    /// Chosen.
    Chosen,
    /// Bears on no open gap.
    Irrelevant,
    /// Another section of higher authority states a different value.
    Contradicted {
        /// The section that wins the fact.
        by: SectionId,
    },
    /// A newer section of equal authority states a different value.
    Superseded {
        /// The section that wins the fact.
        by: SectionId,
    },
    /// An equal section states a different value; neither can be believed.
    Contested {
        /// The other section.
        with: SectionId,
    },
    /// The participant answers what it states already.
    Known,
    /// Everything it states is learned or chosen already.
    Duplicate,
    /// Worth learning, but the words ran out.
    OverBudget,
}

/// One section's decision with the numbers it rested on.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Decision {
    /// The section.
    pub section: SectionId,
    /// What was decided.
    pub verdict: Verdict,
    /// Normalised relevance to the open gaps, in [0, 1].
    pub relevance: f64,
    /// Share of its facts the participant did not answer; `None` when it
    /// states none to check (then ranked at [`UNMEASURED_NOVELTY`]) or when it was rejected
    /// before the participant was asked.
    pub novelty: Option<f64>,
    /// Its cost in words.
    pub words: usize,
}

/// A selection: every candidate's decision, the chosen ones in the order
/// chosen.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Selection {
    /// One decision per candidate that was not already learned.
    pub decisions: Vec<Decision>,
    /// The sections to learn from, best value per word first.
    pub chosen: Vec<SectionId>,
    /// The words the chosen sections cost.
    pub words: usize,
}

impl Selection {
    /// The decision about `section`.
    #[must_use]
    pub fn decision(&self, section: &SectionId) -> Option<&Decision> {
        self.decisions.iter().find(|d| &d.section == section)
    }
}

/// How strongly one section's claim on a fact key is believed.
fn standing(c: &Candidate) -> (u8, Option<&str>, u32) {
    (
        c.provenance.authority,
        c.provenance.date.as_deref(),
        c.provenance.version,
    )
}

/// What is open: the gaps the probes showed, and the goal's own statement,
/// which names the domain the gaps are in.
#[derive(Clone, Copy, Debug)]
pub struct Open<'a> {
    /// The gaps.
    pub gaps: &'a [Gap],
    /// The goal's objective as written.
    pub objective: &'a str,
}

/// Chooses sections of `candidates` for what is `open`, never `learned` ones.
#[must_use]
pub fn select(
    candidates: &[Candidate],
    learned: &BTreeSet<SectionId>,
    open: &Open<'_>,
    vocabulary: &Vocabulary,
    examinee: &dyn Examinee,
    extractor: &dyn FactExtractor,
    config: &Config,
) -> Selection {
    let facts: Vec<Vec<Fact>> = candidates
        .iter()
        .map(|c| extractor.facts(&c.text))
        .collect();
    let relevance = relevance_of(candidates, vocabulary, open);
    let conflicts = if config.check_conflicts {
        conflicts_of(candidates, &facts)
    } else {
        HashMap::new()
    };
    let mut learned_facts: BTreeSet<(FactKey, String)> = BTreeSet::new();
    for (i, c) in candidates.iter().enumerate() {
        if learned.contains(&c.id) {
            learned_facts.extend(facts[i].iter().map(|f| (f.key.clone(), f.value.clone())));
        }
    }

    // Sections that pass relevance and conflict are the ones worth asking
    // the participant about.
    let mut decisions: Vec<Option<Decision>> = vec![None; candidates.len()];
    let mut open: Vec<usize> = Vec::new();
    for (i, c) in candidates.iter().enumerate() {
        if learned.contains(&c.id) {
            continue;
        }
        let decision = |verdict| {
            Some(Decision {
                section: c.id.clone(),
                verdict,
                relevance: relevance[i].coverage,
                novelty: None,
                words: c.words(),
            })
        };
        decisions[i] = if relevance[i].coverage < config.min_relevance {
            decision(Verdict::Irrelevant)
        } else if let Some(verdict) = conflicts.get(&i) {
            decision(verdict.clone())
        } else {
            open.push(i);
            None
        };
    }

    let novelty = if config.check_novelty {
        novelty_of(&open, &facts, examinee)
    } else {
        open.iter().map(|&i| (i, None)).collect()
    };
    let mut ranked: Vec<(usize, f64)> = Vec::new();
    for &i in &open {
        let n = novelty[&i];
        let mut d = Decision {
            section: candidates[i].id.clone(),
            verdict: Verdict::Chosen,
            relevance: relevance[i].coverage,
            novelty: n,
            words: candidates[i].words(),
        };
        if n.is_some_and(|n| n < config.min_novelty) {
            d.verdict = Verdict::Known;
        } else {
            ranked.push((
                i,
                relevance[i].weighted * n.unwrap_or(UNMEASURED_NOVELTY) / d.words.max(1) as f64,
            ));
        }
        decisions[i] = Some(d);
    }

    ranked.sort_by(|a, b| {
        b.1.total_cmp(&a.1)
            .then(candidates[a.0].id.cmp(&candidates[b.0].id))
    });
    let mut chosen = Vec::new();
    let mut words = 0;
    for (i, _) in ranked {
        let Some(d) = decisions[i].as_mut() else {
            continue;
        };
        let fresh = facts[i]
            .iter()
            .any(|f| !learned_facts.contains(&(f.key.clone(), f.value.clone())));
        if !facts[i].is_empty() && !fresh {
            d.verdict = Verdict::Duplicate;
        } else if words + d.words > config.budget_words {
            d.verdict = Verdict::OverBudget;
        } else {
            words += d.words;
            learned_facts.extend(facts[i].iter().map(|f| (f.key.clone(), f.value.clone())));
            chosen.push(d.section.clone());
        }
    }
    Selection {
        decisions: decisions.into_iter().flatten().collect(),
        chosen,
        words,
    }
}

/// Per candidate, how much of what the gaps ask about it covers:
/// the gap's words weighted by squared rarity in the corpus (a word in
/// every section weighs almost nothing) and by the square of the share of
/// the gap's probes that use them (what the probes have in common is what the gap is about), the share a section has, and the gaps' shares
/// of the failures added up. Words no section has are not asked about.
fn relevance_of(
    candidates: &[Candidate],
    vocabulary: &Vocabulary,
    open: &Open<'_>,
) -> Vec<Relevance> {
    let sets: Vec<BTreeSet<String>> = candidates
        .iter()
        .map(|c| words(&c.text).into_iter().collect())
        .collect();
    let domain: Vec<String> = words(open.objective)
        .into_iter()
        .filter(|w| {
            vocabulary
                .share(w)
                .is_some_and(|share| share <= DOMAIN_TERM_SHARE)
        })
        .collect();
    let mut out = vec![Relevance::default(); candidates.len()];
    for gap in open.gaps {
        let weights: Vec<(&str, f64)> = gap
            .terms
            .iter()
            .filter_map(|(t, support)| {
                Some((t.as_str(), vocabulary.weight(t)? * support * support))
            })
            .collect();
        let total: f64 = weights.iter().map(|(_, w)| w).sum();
        if total <= 0.0 {
            continue;
        }
        for (i, set) in sets.iter().enumerate() {
            let covered: f64 = weights
                .iter()
                .filter(|(t, _)| set.contains(*t))
                .map(|(_, w)| w)
                .sum();
            let cover = covered / total;
            out[i].coverage = out[i].coverage.max(cover);
            out[i].weighted += gap.weight * cover;
        }
    }
    // A goal that names its domain decides by that first: a section that
    // mentions none of it cannot pass on the strength of words every domain
    // shares, and one that does is relevant even where the probes did not
    // reach (the probes sample the domain, they do not bound it).
    if !domain.is_empty() {
        for (i, set) in sets.iter().enumerate() {
            if domain.iter().any(|w| set.contains(w)) {
                out[i].coverage = out[i].coverage.max(IN_DOMAIN);
                out[i].weighted = out[i].weighted.max(IN_DOMAIN);
            } else {
                out[i].coverage = out[i].coverage.min(OFF_DOMAIN);
                out[i].weighted *= OFF_DOMAIN;
            }
        }
    }
    out
}

/// What a section's relevance to the open gaps comes to.
#[derive(Clone, Copy, Default)]
struct Relevance {
    /// Its best coverage of any one gap: what the floor is applied to.
    coverage: f64,
    /// Coverage summed over the gaps by their share of the failures: what
    /// ranks it.
    weighted: f64,
}

/// The verdict of every candidate (by index) that loses a fact it states.
fn conflicts_of(candidates: &[Candidate], facts: &[Vec<Fact>]) -> HashMap<usize, Verdict> {
    let mut claims: BTreeMap<&FactKey, Vec<(usize, &str)>> = BTreeMap::new();
    for (i, fs) in facts.iter().enumerate() {
        for f in fs {
            claims
                .entry(&f.key)
                .or_default()
                .push((i, f.value.as_str()));
        }
    }
    let mut verdicts = HashMap::new();
    for claimants in claims.values() {
        let values: BTreeSet<&str> = claimants.iter().map(|(_, v)| *v).collect();
        if values.len() < 2 {
            continue;
        }
        let Some(top) = claimants
            .iter()
            .map(|(i, _)| standing(&candidates[*i]))
            .max()
        else {
            continue;
        };
        let leaders: Vec<(usize, &str)> = claimants
            .iter()
            .copied()
            .filter(|(i, _)| standing(&candidates[*i]) == top)
            .collect();
        let leading: BTreeSet<&str> = leaders.iter().map(|(_, v)| *v).collect();
        let Some(&winner) = leaders.first() else {
            continue;
        };
        let contested = leading.len() > 1;
        for &(i, value) in claimants {
            let by = candidates[winner.0].id.clone();
            let verdict = if contested && standing(&candidates[i]) == top {
                let with = leaders.iter().find(|(_, v)| *v != value).map_or(i, |l| l.0);
                Verdict::Contested {
                    with: candidates[with].id.clone(),
                }
            } else if !contested && value == winner.1 {
                continue;
            } else if standing(&candidates[i]).0 < top.0 {
                Verdict::Contradicted { by }
            } else {
                Verdict::Superseded { by }
            };
            verdicts.entry(i).or_insert(verdict);
        }
    }
    verdicts
}

/// Per section index in `open`, the share of its facts the examinee does not
/// answer as the section states; `None` for a section with no fact.
fn novelty_of(
    open: &[usize],
    facts: &[Vec<Fact>],
    examinee: &dyn Examinee,
) -> HashMap<usize, Option<f64>> {
    let mut questions: Vec<String> = open
        .iter()
        .flat_map(|&i| facts[i].iter().map(Fact::question))
        .collect();
    questions.sort();
    questions.dedup();
    let answers = examinee.answer_all(&questions);
    let answered: HashMap<&str, &str> = questions
        .iter()
        .map(String::as_str)
        .zip(answers.iter().map(String::as_str))
        .collect();
    open.iter()
        .map(|&i| {
            let fs = &facts[i];
            let novelty = (!fs.is_empty()).then(|| {
                let known = fs
                    .iter()
                    .filter(|f| states(answered[f.question().as_str()], &f.value))
                    .count();
                1.0 - known as f64 / fs.len() as f64
            });
            (i, novelty)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::goal::corpus::{candidates, Document};
    use crate::goal::facts::PatternFacts;
    use crate::goal::gaps::{analyse, Probe, ProbeResult};

    /// Knows exactly the answers in `known`, else says nothing.
    struct Knows(Vec<(String, String)>);

    impl Examinee for Knows {
        fn answer_all(&self, questions: &[String]) -> Vec<String> {
            questions
                .iter()
                .map(|q| {
                    self.0
                        .iter()
                        .find(|(k, _)| k == q)
                        .map_or("unknown".into(), |(_, v)| v.clone())
                })
                .collect()
        }
    }

    fn doc(id: &str, authority: u8, date: &str, body: &str) -> Document {
        Document::parse(
            id,
            &format!("---\nauthority: {authority}\ndate: {date}\nversion: 1\n---\n{body}"),
        )
    }

    fn gap_on(cands: &[Candidate], question: &str, expected: &str) -> Vec<Gap> {
        analyse(
            &[ProbeResult {
                probe: Probe {
                    id: "p".into(),
                    question: question.into(),
                    expected: expected.into(),
                    skill: None,
                },
                answer: "unknown".into(),
            }],
            &Vocabulary::of(cands),
        )
    }

    const CURRENT: &str = "## Alder pump 1\n\nAlder pump service notes.\nThe service interval of alder pump 1 is 400 hours.\n";

    fn corpus() -> Vec<Candidate> {
        candidates(&[
            doc("current", 3, "2026-06-01", CURRENT),
            doc("old", 3, "2024-01-01", "## Alder pump 1\n\nAlder pump service notes.\nThe service interval of alder pump 1 is 250 hours.\n"),
            doc("forum", 1, "2026-09-01", "## Alder pump 1\n\nAlder pump service notes.\nThe service interval of alder pump 1 is 90 hours.\n"),
            doc("other", 3, "2026-06-01", "## Brisk fan 2\n\nBrisk fan notes only.\nThe service interval of brisk fan 2 is 700 hours.\n"),
            doc("known", 3, "2026-06-01", "## Alder pump 5\n\nAlder pump service notes.\nThe service interval of alder pump 5 is 100 hours.\n"),
        ])
    }

    fn run(knows: Vec<(&str, &str)>, budget: usize) -> Selection {
        let gaps = gap_on(
            &corpus(),
            "What is the service interval of alder pump 3?",
            "400 hours",
        );
        let examinee = Knows(
            knows
                .into_iter()
                .map(|(a, b)| (a.to_string(), b.to_string()))
                .collect(),
        );
        select(
            &corpus(),
            &BTreeSet::new(),
            &Open {
                gaps: &gaps,
                objective: "Service the alder pump line.",
            },
            &Vocabulary::of(&corpus()),
            &examinee,
            &PatternFacts,
            &Config::with_budget(budget),
        )
    }

    fn verdict(s: &Selection, doc: &str) -> Verdict {
        s.decision(&SectionId {
            document: doc.into(),
            section: 0,
        })
        .map(|d| d.verdict.clone())
        .unwrap_or_else(|| panic!("no decision for {doc}"))
    }

    #[test]
    fn each_kind_of_bad_material_is_rejected_for_its_own_reason() {
        let s = run(
            vec![("What is the service interval of alder pump 5?", "100 hours")],
            1000,
        );
        let id = |d: &str| SectionId {
            document: d.into(),
            section: 0,
        };
        assert_eq!(verdict(&s, "current"), Verdict::Chosen);
        assert_eq!(
            verdict(&s, "old"),
            Verdict::Superseded { by: id("current") }
        );
        assert_eq!(
            verdict(&s, "forum"),
            Verdict::Contradicted { by: id("current") }
        );
        assert_eq!(verdict(&s, "other"), Verdict::Irrelevant);
        assert_eq!(verdict(&s, "known"), Verdict::Known);
        assert_eq!(s.chosen, vec![id("current")]);
        assert_eq!(s.decisions.len(), 5, "every section is accounted for");
    }

    #[test]
    fn the_budget_stops_the_choice_and_the_rest_is_over_budget() {
        let s = run(vec![], 20);
        assert_eq!(s.chosen.len(), 1, "{s:?}");
        let over = s
            .decisions
            .iter()
            .filter(|d| d.verdict == Verdict::OverBudget)
            .count();
        assert!(over >= 1 || s.decisions.iter().any(|d| d.verdict == Verdict::Known));
        assert!(s.words <= 20);
    }

    #[test]
    fn a_learned_section_is_not_chosen_again_and_its_duplicate_is_refused() {
        let mut twin = corpus();
        twin.push(Candidate {
            id: SectionId {
                document: "copy".into(),
                section: 0,
            },
            ..twin[0].clone()
        });
        let gaps = gap_on(
            &corpus(),
            "What is the service interval of alder pump 3?",
            "400 hours",
        );
        let learned: BTreeSet<SectionId> = [twin[0].id.clone()].into();
        let s = select(
            &twin,
            &learned,
            &Open {
                gaps: &gaps,
                objective: "",
            },
            &Vocabulary::of(&twin),
            &Knows(vec![]),
            &PatternFacts,
            &Config::with_budget(1000),
        );
        assert!(s.decision(&twin[0].id).is_none());
        assert_eq!(verdict(&s, "copy"), Verdict::Duplicate);
    }
}
