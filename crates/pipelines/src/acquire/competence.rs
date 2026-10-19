// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that turn the documentation of an
// unfamiliar technical domain into a graph of skills they then acquire, for
// its clients. If your team needs expertise in building curricula from
// technical documentation you can procure our services by sending an email
// to info@swedishembedded.com.

//! The competence graph: how well the policy does each skill of a skill
//! graph, measured on probes it was never trained on.
//!
//! A [`SkillCompetence`] is a count of graded probe attempts and the passes
//! among them, with the Wilson interval of the pass rate (a skill measured on
//! few probes is not claimed to be mastered on a lucky run of passes). A skill
//! nobody measured has no entry; an absent measurement is never zero. Every
//! update is a new [`CompetenceGraph`] naming the one it replaces, kept as a
//! content-addressed document, so the history of what the policy could do and
//! when is on record.

use serde::{Deserialize, Serialize};
use splinter_core::digest::Digest;

use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// The z of the 95 percent interval.
const Z95: f64 = 1.959_964;

/// The Wilson score interval of `passes` of `graded`; `None` with none.
#[must_use]
pub fn wilson(passes: usize, graded: usize) -> Option<(f64, f64)> {
    if graded == 0 {
        return None;
    }
    let n = graded as f64;
    let p = passes as f64 / n;
    let z2 = Z95 * Z95;
    let centre = (p + z2 / (2.0 * n)) / (1.0 + z2 / n);
    let half = Z95 * ((p * (1.0 - p) + z2 / (4.0 * n)) / n).sqrt() / (1.0 + z2 / n);
    Some(((centre - half).max(0.0), (centre + half).min(1.0)))
}

/// One skill's measured competence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SkillCompetence {
    /// The skill.
    pub skill: String,
    /// Probe attempts that were graded.
    pub graded: usize,
    /// Of those, passes.
    pub passes: usize,
    /// `passes / graded`.
    pub rate: f64,
    /// The lower end of the 95 percent Wilson interval.
    pub low: f64,
    /// The upper end.
    pub high: f64,
    /// Training rounds spent on the skill so far.
    pub rounds: usize,
}

impl SkillCompetence {
    /// The competence `passes` of `graded` probe attempts show; `None` with
    /// no graded attempt.
    #[must_use]
    pub fn measured(skill: &str, passes: usize, graded: usize, rounds: usize) -> Option<Self> {
        let (low, high) = wilson(passes, graded)?;
        Some(Self {
            skill: skill.to_string(),
            graded,
            passes,
            rate: passes as f64 / graded as f64,
            low,
            high,
            rounds,
        })
    }

    /// Whether the skill is mastered: measured on at least `min_probes`
    /// graded attempts, passing at least `target` of them.
    #[must_use]
    pub fn mastered(&self, target: f64, min_probes: usize) -> bool {
        self.graded >= min_probes && self.rate >= target
    }
}

/// What the policy can do, skill by skill, at one moment.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompetenceGraph {
    /// The skill graph it is over.
    pub graph: Digest,
    /// The competence graph this replaces.
    pub parent: Option<Digest>,
    /// Which update this is, from 1.
    pub revision: u32,
    /// Who was measured: the model's identity, or the base.
    pub model: String,
    /// The training round it follows; 0 is the baseline.
    pub round: usize,
    /// The skills measured so far.
    pub skills: Vec<SkillCompetence>,
}

impl CompetenceGraph {
    /// What `skill`'s competence is, when measured.
    #[must_use]
    pub fn get(&self, skill: &str) -> Option<&SkillCompetence> {
        self.skills.iter().find(|s| s.skill == skill)
    }

    /// This graph with `update` in place of the skill's entry (added when it
    /// had none), as the revision after it.
    #[must_use]
    pub fn updated(
        &self,
        parent: Digest,
        model: &str,
        round: usize,
        update: SkillCompetence,
    ) -> Self {
        let mut skills: Vec<SkillCompetence> = self
            .skills
            .iter()
            .filter(|s| s.skill != update.skill)
            .cloned()
            .collect();
        skills.push(update);
        Self {
            graph: self.graph.clone(),
            parent: Some(parent),
            revision: self.revision + 1,
            model: model.to_string(),
            round,
            skills,
        }
    }
}

const CLASS: &str = "competence_graph";

/// Keeps `graph` and returns its address.
pub fn keep(ctx: &Context, graph: &CompetenceGraph) -> Result<Digest, OrchestratorError> {
    let stored = ctx.workspace().put_document(CLASS, graph)?;
    Digest::parse(stored.as_str()).map_err(|e| {
        OrchestratorError::Refused(format!("the competence graph was kept unreadably: {e}"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_few_lucky_passes_are_not_mastery_the_interval_says_so() {
        let (low, high) = wilson(3, 3).unwrap();
        assert!(low < 0.5 && (high - 1.0).abs() < 1e-9, "{low} {high}");
        let (low_many, _) = wilson(30, 30).unwrap();
        assert!(low_many > 0.88, "{low_many}");
        assert_eq!(wilson(0, 0), None);
        let (low, high) = wilson(0, 10).unwrap();
        assert!(low.abs() < 1e-12 && high < 0.35);
    }

    #[test]
    fn mastery_needs_enough_probes_and_enough_passes() {
        let c = SkillCompetence::measured("s", 4, 5, 0).unwrap();
        assert!(c.mastered(0.8, 5));
        assert!(!c.mastered(0.81, 5));
        assert!(!c.mastered(0.8, 6));
        assert_eq!(SkillCompetence::measured("s", 0, 0, 0), None);
    }

    #[test]
    fn an_update_replaces_the_skill_and_names_what_it_replaces() {
        let first = CompetenceGraph {
            graph: Digest::of(b"g"),
            parent: None,
            revision: 1,
            model: "base".into(),
            round: 0,
            skills: vec![
                SkillCompetence::measured("a", 1, 4, 0).unwrap(),
                SkillCompetence::measured("b", 2, 4, 0).unwrap(),
            ],
        };
        let parent = Digest::of(b"first");
        let next = first.updated(
            parent.clone(),
            "tuned",
            1,
            SkillCompetence::measured("a", 4, 4, 1).unwrap(),
        );
        assert_eq!(next.revision, 2);
        assert_eq!(next.parent, Some(parent));
        assert_eq!(next.get("a").unwrap().passes, 4);
        assert_eq!(next.get("b").unwrap().passes, 2, "the others are untouched");
        assert_eq!(
            first.get("a").unwrap().passes,
            1,
            "the earlier one is as it was"
        );
    }
}
