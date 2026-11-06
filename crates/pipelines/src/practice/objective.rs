// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reinforcement learning from the verified
// results of agents working in a repository, for its clients. If your team
// needs expertise in training code-editing agents against hidden tests, you
// can procure our services by sending an email to info@swedishembedded.com.

//! What a `practice` run trains on, and how.

use std::str::FromStr;

use serde::Serialize;

/// The training objective of a `practice` (and so of a generation).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum PracticeObjective {
    /// Imitate the actions of the attempts the hidden checks passed, on the
    /// tasks the policy solves sometimes (the frontier): supervised
    /// fine-tuning of the champion's adapter with earlier records replayed.
    #[default]
    SftStep,
    /// Reinforce by the clipped policy gradient: the k attempts the solve
    /// stage made at a task are a group, reward is a verified pass, and
    /// every action of every attempt is weighed by its attempt's advantage
    /// over the others. The champion's adapter is continued; a task whose
    /// attempts all passed or all failed carries no advantage and is
    /// counted, not trained on.
    PolicyGradient,
    /// The policy gradient, and the tasks whose attempts all failed (where
    /// the advantage is zero) distilled: the student's own conversation
    /// scored by the frozen model shown what the verifier knows (the hidden
    /// checks that failed and the task's reference change).
    SelfDistill,
}

impl PracticeObjective {
    /// Every objective, in the order `--help` lists them.
    pub const ALL: [Self; 3] = [Self::SftStep, Self::PolicyGradient, Self::SelfDistill];

    /// The name a command line uses.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SftStep => "sft-step",
            Self::PolicyGradient => "policy-gradient",
            Self::SelfDistill => "self-distill",
        }
    }

    /// Whether the objective learns from the groups of a task's attempts
    /// (rewards, advantages) rather than from the passing ones alone.
    #[must_use]
    pub fn learns_from_groups(self) -> bool {
        self != Self::SftStep
    }

    /// Whether the all-fail groups are distilled with a teacher context.
    #[must_use]
    pub fn distils_cliffs(self) -> bool {
        self == Self::SelfDistill
    }
}

impl std::fmt::Display for PracticeObjective {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PracticeObjective {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, String> {
        Self::ALL
            .into_iter()
            .find(|objective| objective.as_str() == text)
            .ok_or_else(|| {
                let names: Vec<&str> = Self::ALL.iter().map(|o| o.as_str()).collect();
                format!("{text:?} is not an objective: {}", names.join(", "))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_objective_is_named_by_the_name_that_parses_back() {
        for objective in PracticeObjective::ALL {
            assert_eq!(objective.as_str().parse(), Ok(objective));
        }
        assert!("sft".parse::<PracticeObjective>().is_err());
    }

    #[test]
    fn the_default_is_the_supervised_objective() {
        assert_eq!(PracticeObjective::default(), PracticeObjective::SftStep);
    }
}
