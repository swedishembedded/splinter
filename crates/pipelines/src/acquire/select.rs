// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that turn the documentation of an
// unfamiliar technical domain into a graph of skills they then acquire, for
// its clients. If your team needs expertise in building curricula from
// technical documentation you can procure our services by sending an email
// to info@swedishembedded.com.

//! Which skill to train next: the one expected to benefit most.
//!
//! A skill is eligible when it is measured, not mastered, not given up on
//! (more than `max_rounds` rounds spent on it, with the target still not
//! reached), and every skill it uses is mastered or given up on - a skill
//! whose prerequisite cannot be learned is not held back forever. Among the
//! eligible the expected benefit decides:
//!
//! `benefit = (target - rate) * (1 + unmastered dependents) / (1 + rounds spent)`
//!
//! the room left to the target, weighted by how many skills the skill
//! unlocks, discounted by the rounds already spent on it. Ties go to the
//! earlier skill of the graph's order. A skill not measured has no rate and
//! is not chosen: it is measured first.

use splinter_knowledge::skills::SkillGraph;

use super::competence::CompetenceGraph;

/// What a selection is judged by.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Targets {
    /// The pass rate a skill must reach to be mastered.
    pub mastery: f64,
    /// The graded probe attempts a skill's competence must rest on.
    pub min_probes: usize,
    /// The most training rounds one skill may take.
    pub max_rounds: usize,
}

/// The skill chosen, and why.
#[derive(Clone, Debug, PartialEq)]
pub struct Selection {
    /// The skill.
    pub skill: String,
    /// Its expected benefit.
    pub benefit: f64,
}

/// Whether the skill has been worked on as much as it may be.
fn given_up(competence: &CompetenceGraph, skill: &str, targets: &Targets) -> bool {
    competence.get(skill).is_some_and(|c| {
        c.rounds >= targets.max_rounds && !c.mastered(targets.mastery, targets.min_probes)
    })
}

fn mastered(competence: &CompetenceGraph, skill: &str, targets: &Targets) -> bool {
    competence
        .get(skill)
        .is_some_and(|c| c.mastered(targets.mastery, targets.min_probes))
}

/// The skill of `graph` to train next under `competence`, when there is
/// one.
#[must_use]
pub fn next_skill(
    graph: &SkillGraph,
    competence: &CompetenceGraph,
    targets: &Targets,
) -> Option<Selection> {
    let mut best: Option<Selection> = None;
    for skill in graph.order() {
        let Some(measured) = competence.get(&skill.id) else {
            continue;
        };
        if mastered(competence, &skill.id, targets) || given_up(competence, &skill.id, targets) {
            continue;
        }
        let ready = skill.prerequisites.iter().all(|p| {
            mastered(competence, p, targets)
                || given_up(competence, p, targets)
                // A prerequisite nobody could measure cannot hold this one
                // back.
                || competence.get(p).is_none()
        });
        if !ready {
            continue;
        }
        let unlocked = graph
            .dependents(&skill.id)
            .iter()
            .filter(|d| !mastered(competence, &d.id, targets))
            .count();
        let benefit = (targets.mastery - measured.rate).max(0.0) * (1 + unlocked) as f64
            / (1 + measured.rounds) as f64;
        // The earlier skill wins a tie: a later one must be strictly better.
        if best.as_ref().is_none_or(|b| benefit > b.benefit) {
            best = Some(Selection {
                skill: skill.id.clone(),
                benefit,
            });
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acquire::competence::SkillCompetence;
    use splinter_core::digest::Digest;
    use splinter_core::source::SourceId;
    use splinter_knowledge::concepts::SectionRef;
    use splinter_knowledge::skills::Skill;

    const TARGETS: Targets = Targets {
        mastery: 0.8,
        min_probes: 4,
        max_rounds: 2,
    };

    fn skill(id: &str, needs: &[&str]) -> Skill {
        Skill {
            id: id.into(),
            title: id.into(),
            sections: vec![SectionRef {
                source: SourceId(Digest::of(b"s")),
                part: "p".into(),
                section: 0,
            }],
            symbols: Vec::new(),
            constraints: Vec::new(),
            requirements: Vec::new(),
            prerequisites: needs.iter().map(|n| (*n).to_string()).collect(),
        }
    }

    fn graph() -> SkillGraph {
        SkillGraph {
            objective: "o".into(),
            revision: 1,
            parent: None,
            sources: Vec::new(),
            // a is used by b and c; d stands alone.
            skills: vec![
                skill("a", &[]),
                skill("b", &["a"]),
                skill("c", &["a"]),
                skill("d", &[]),
            ],
        }
    }

    fn competence(measured: &[(&str, usize, usize, usize)]) -> CompetenceGraph {
        CompetenceGraph {
            graph: Digest::of(b"g"),
            parent: None,
            revision: 1,
            model: "m".into(),
            round: 0,
            skills: measured
                .iter()
                .map(|(s, p, g, r)| SkillCompetence::measured(s, *p, *g, *r).unwrap())
                .collect(),
        }
    }

    #[test]
    fn a_skill_that_unlocks_others_beats_one_of_equal_room() {
        // a and d are equally weak; a unlocks b and c.
        let c = competence(&[
            ("a", 0, 4, 0),
            ("b", 0, 4, 0),
            ("c", 0, 4, 0),
            ("d", 0, 4, 0),
        ]);
        let chosen = next_skill(&graph(), &c, &TARGETS).unwrap();
        assert_eq!(chosen.skill, "a");
        assert!((chosen.benefit - 0.8 * 3.0).abs() < 1e-9);
    }

    #[test]
    fn a_prerequisite_is_mastered_before_the_skill_that_uses_it() {
        // b is weaker than d, but a is not mastered, so b waits.
        let c = competence(&[
            ("a", 4, 4, 0),
            ("b", 0, 4, 0),
            ("c", 4, 4, 0),
            ("d", 2, 4, 0),
        ]);
        assert_eq!(next_skill(&graph(), &c, &TARGETS).unwrap().skill, "b");
        let c = competence(&[
            ("a", 1, 4, 0),
            ("b", 0, 4, 0),
            ("c", 0, 4, 0),
            ("d", 4, 4, 0),
        ]);
        assert_eq!(next_skill(&graph(), &c, &TARGETS).unwrap().skill, "a");
    }

    #[test]
    fn a_skill_given_up_on_no_longer_holds_back_what_uses_it_and_is_not_chosen() {
        // a took two rounds and is still weak: given up. b may go on.
        let c = competence(&[
            ("a", 1, 4, 2),
            ("b", 0, 4, 0),
            ("c", 4, 4, 0),
            ("d", 4, 4, 0),
        ]);
        assert_eq!(next_skill(&graph(), &c, &TARGETS).unwrap().skill, "b");
    }

    #[test]
    fn nothing_is_chosen_when_every_skill_is_mastered_or_given_up_on() {
        let c = competence(&[
            ("a", 4, 4, 1),
            ("b", 4, 4, 0),
            ("c", 1, 4, 2),
            ("d", 4, 4, 0),
        ]);
        assert_eq!(next_skill(&graph(), &c, &TARGETS), None);
        // A skill measured on too few probes is not mastered however well it did.
        let c = competence(&[
            ("a", 2, 2, 0),
            ("b", 4, 4, 0),
            ("c", 4, 4, 0),
            ("d", 4, 4, 0),
        ]);
        assert_eq!(next_skill(&graph(), &c, &TARGETS).unwrap().skill, "a");
    }

    #[test]
    fn rounds_already_spent_discount_the_benefit_and_ties_go_to_the_earlier_skill() {
        let c = competence(&[
            ("a", 4, 4, 0),
            ("b", 4, 4, 0),
            ("c", 2, 4, 1),
            ("d", 2, 4, 0),
        ]);
        // c: 0.3 * 1 / 2; d: 0.3 * 1 / 1.
        assert_eq!(next_skill(&graph(), &c, &TARGETS).unwrap().skill, "d");
        let tied = competence(&[
            ("a", 4, 4, 0),
            ("b", 2, 4, 0),
            ("c", 2, 4, 0),
            ("d", 4, 4, 0),
        ]);
        assert_eq!(next_skill(&graph(), &tied, &TARGETS).unwrap().skill, "b");
    }
}
