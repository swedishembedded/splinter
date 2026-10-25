// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements automated improvement of agent harnesses
// that adopts a change only on verified, regression-free evidence, for its
// clients. If your team needs expertise in measuring whether a harness
// change really helped, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The space candidate harnesses are drawn from: a baseline document and a
//! list of knobs, each a path into the document and the values it may take.
//! A mutation sets one knob to one of its other values, so every candidate is
//! the baseline with a few edits, never a rewrite, and the edits are the
//! candidate's whole description.

use serde::{Deserialize, Serialize};
use splinter_agent::harness_edit::child;
use splinter_agent::sven::search::{Proposal, ProposalStep, Proposer, SearchError, SearchRng};

/// One thing a candidate may change.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Knob {
    /// The keys leading to the value, from the document's root
    /// (`[workflow, 0, task]`): a number indexes a list.
    pub path: Vec<String>,
    /// The values the knob may take; at least two.
    pub choices: Vec<serde_yaml::Value>,
}

/// The knobs of a search.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MutationSpace {
    /// Every knob; a mutation picks one.
    pub knobs: Vec<Knob>,
}

/// Why a mutation space or a mutation is unusable.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum SpaceError {
    /// The space text is not a mutation space.
    #[error("the mutation space does not parse: {0}")]
    Parse(String),
    /// A knob cannot take two values or leads nowhere.
    #[error("knob {index} is unusable: {reason}")]
    Knob {
        /// Its position.
        index: usize,
        /// Why.
        reason: String,
    },
    /// The document is not YAML.
    #[error("the harness document does not parse: {0}")]
    Document(String),
    /// A knob's path is not in the document.
    #[error("knob path {path} is not in the harness document")]
    Missing {
        /// The path, dotted.
        path: String,
    },
}

impl MutationSpace {
    /// Parses and checks a space.
    ///
    /// # Errors
    ///
    /// [`SpaceError`] for text that is not a space, no knob, a knob with an
    /// empty path or fewer than two distinct choices.
    pub fn parse(text: &str) -> Result<Self, SpaceError> {
        let space: Self =
            serde_yaml::from_str(text).map_err(|e| SpaceError::Parse(e.to_string()))?;
        if space.knobs.is_empty() {
            return Err(SpaceError::Parse("a space needs at least one knob".into()));
        }
        for (index, knob) in space.knobs.iter().enumerate() {
            let fail = |reason: &str| {
                Err(SpaceError::Knob {
                    index,
                    reason: reason.to_string(),
                })
            };
            if knob.path.is_empty() {
                return fail("its path is empty");
            }
            let distinct = knob
                .choices
                .iter()
                .enumerate()
                .filter(|(i, c)| !knob.choices[..*i].contains(c))
                .count();
            if distinct < 2 {
                return fail("it needs at least two distinct choices");
            }
        }
        Ok(space)
    }

    /// `document` with one knob set to a value it does not hold now. Every
    /// draw comes from `rng`.
    ///
    /// # Errors
    ///
    /// [`SpaceError`] when the document does not parse or a knob's path is not
    /// in it (a space written for another document).
    pub fn mutate(&self, document: &str, rng: &mut SearchRng) -> Result<String, SpaceError> {
        let mut root: serde_yaml::Value =
            serde_yaml::from_str(document).map_err(|e| SpaceError::Document(e.to_string()))?;
        let knob = &self.knobs[rng.below(self.knobs.len())];
        let slot = knob.path.iter().try_fold(&mut root, |at, key| {
            child(at, key).ok_or_else(|| SpaceError::Missing {
                path: knob.path.join("."),
            })
        })?;
        let others: Vec<&serde_yaml::Value> = knob.choices.iter().filter(|c| *c != slot).collect();
        // `parse` guarantees two distinct choices, so one differs from the
        // current value.
        *slot = others[rng.below(others.len())].clone();
        serde_yaml::to_string(&root).map_err(|e| SpaceError::Document(e.to_string()))
    }
}

/// How many mutations are drawn to find one sven's compiler accepts.
const COMPILING_DRAWS: usize = 16;

impl MutationSpace {
    /// [`Self::mutate`] until the result compiles under sven's compiler: a
    /// knob value may break a document, and such a candidate never counts.
    ///
    /// # Errors
    ///
    /// The last failure when none of a bounded number of draws compiles.
    pub fn mutate_compiling(&self, document: &str, rng: &mut SearchRng) -> Result<String, String> {
        let mut last = String::new();
        for _ in 0..COMPILING_DRAWS {
            let mutated = self.mutate(document, rng).map_err(|e| e.to_string())?;
            match splinter_agent::harness_edit::check(&mutated) {
                Ok(()) => return Ok(mutated),
                Err(why) => last = why,
            }
        }
        Err(format!(
            "no mutation compiled in {COMPILING_DRAWS} draws; last: {last}"
        ))
    }
}

/// Proposes candidates by mutating the parent harness document.
pub struct MutationProposer {
    /// The space mutated within.
    pub space: MutationSpace,
}

impl Proposer<String> for MutationProposer {
    async fn propose(
        &self,
        parent: &String,
        _step: &ProposalStep,
        rng: &mut SearchRng,
    ) -> Result<Proposal<String>, SearchError> {
        self.space
            .mutate_compiling(parent, rng)
            .map(Proposal::free)
            .map_err(SearchError::proposer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOC: &str =
        "api_version: \"2\"\nname: h\nexecution:\n  limits:\n    model_turns: 4\nworkflow:\n  - name: fix\n    task: a\n";

    fn space() -> MutationSpace {
        MutationSpace::parse(
            "knobs:\n  - path: [execution, limits, model_turns]\n    choices: [4, 6, 8]\n  - path: [workflow, '0', task]\n    choices: [a, b]\n",
        )
        .unwrap()
    }

    #[test]
    fn a_mutation_changes_exactly_one_knob_to_another_of_its_values() {
        let space = space();
        let before: serde_yaml::Value = serde_yaml::from_str(DOC).unwrap();
        for seed in 0..40 {
            let mutated = space.mutate(DOC, &mut SearchRng::from_seed(seed)).unwrap();
            let after: serde_yaml::Value = serde_yaml::from_str(&mutated).unwrap();
            let turns = |v: &serde_yaml::Value| v["execution"]["limits"]["model_turns"].clone();
            let prompt = |v: &serde_yaml::Value| v["workflow"][0]["task"].clone();
            let changed = usize::from(turns(&before) != turns(&after))
                + usize::from(prompt(&before) != prompt(&after));
            assert_eq!(changed, 1, "seed {seed}: {mutated}");
        }
    }

    #[test]
    fn the_same_seed_draws_the_same_mutation() {
        let space = space();
        let draw = |seed| space.mutate(DOC, &mut SearchRng::from_seed(seed)).unwrap();
        assert_eq!(draw(3), draw(3));
    }

    #[test]
    fn a_knob_the_document_lacks_is_an_error_naming_its_path() {
        let space = MutationSpace::parse(
            "knobs:\n  - path: [execution, limits, tool_calls]\n    choices: [1, 2]\n",
        )
        .unwrap();
        // seed: the only knob is the missing one, so every seed draws it.
        let mut rng = SearchRng::from_seed(1);
        let err = space.mutate(DOC, &mut rng).unwrap_err();
        assert!(
            err.to_string().contains("execution.limits.tool_calls"),
            "{err}"
        );
    }

    #[test]
    fn unusable_spaces_are_refused() {
        for text in [
            "knobs: []\n",
            "knobs:\n  - path: []\n    choices: [1, 2]\n",
            "knobs:\n  - path: [a]\n    choices: [1, 1]\n",
            "knobs:\n  - path: [a]\n    choices: [1, 2]\n    extra: true\n",
        ] {
            assert!(MutationSpace::parse(text).is_err(), "{text}");
        }
    }
}
