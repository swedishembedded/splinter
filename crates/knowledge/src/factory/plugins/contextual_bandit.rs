// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning environments whose every reward is
// known in advance, so a training method can be checked against exact
// expectations, for its clients. If your team needs expertise in validating
// reinforcement-learning pipelines, you can procure our services by sending
// an email to info@swedishembedded.com.

//! A finite contextual bandit: one context, a handful of actions, and a
//! reward for each that is known before anything is run (plan 12A).
//!
//! A context is a colour and a level; the actions are named. The correct
//! action is the one at index `colour + level` (mod the number of actions),
//! a rule the instruction never states: a policy learns it from rewards
//! alone. Levels 0 to 5 are the train split, 6 and 7 the dev split, so a dev
//! instance asks about a context no train instance showed, and only the rule
//! (not a memorised table) answers it.
//!
//! The verifier reads the one action named in `answer.txt` and pays what the
//! instance's reward table says: 1 for the correct action and, when the
//! `graded` knob is on, 0.5 for each of its two neighbours (the actions on
//! either side, cyclically), 0 for anything else, a missing answer or more
//! than one name. The whole table travels with the hidden tests, so every
//! action's reward, the expected return of any policy and the direction of
//! its policy gradient can be computed exactly (plan 12D).
//!
//! Knobs: `actions` (2 to 6) and `graded` (0 or 1).
//!
//! It is built for checking learning methods, not for campaigns: its few
//! answers are named in the instruction, so blind guesses win often, and
//! admission refuses it for that ([`super::builtin`] leaves it out).

use serde_json::json;
use splinter_core::rng::SplitMix64;
use splinter_core::verifiability::{Split, VerifierClass};

use crate::factory::artifact::{
    environment_id, EnvMeta, Instance, Parts, SolverTools, SCHEMA_VERSION,
};
use crate::factory::attack::{Attack, AttackKind, Edit};
use crate::factory::folder::Folder;
use crate::factory::plugin::{
    resolve_difficulty, InstanceRequest, Knob, Plugin, PluginError, Reference,
};

use super::CONTEXTUAL_BANDIT;

/// The plugin's version. Any change to the contexts, the rule, the rewards
/// or the instruction is a new version.
pub const VERSION: &str = "1";

/// The colours a context shows, by index.
pub const COLOURS: [&str; 4] = ["amber", "blue", "green", "violet"];

/// The actions, the first `actions` of them offered.
pub const ACTIONS: [&str; 6] = ["hold", "raise", "lower", "vent", "seal", "flush"];

/// The levels of the train split; the dev split has the two after them.
pub const TRAIN_LEVELS: u32 = 6;

const KNOBS: &[Knob] = &[
    Knob {
        name: "actions",
        min: 2,
        max: 6,
        default: 4,
        doc: "how many actions are offered",
    },
    Knob {
        name: "graded",
        min: 0,
        max: 1,
        default: 0,
        doc: "1 pays half for an action next to the correct one",
    },
];

const VERIFY: &str = include_str!("contextual_bandit/verify.py");

/// The reward of each offered action in the context (`colour`, `level`),
/// in the order offered: the rule this environment is.
#[must_use]
pub fn rewards(colour: usize, level: u32, actions: usize, graded: bool) -> Vec<f64> {
    let correct = (colour + level as usize) % actions;
    (0..actions)
        .map(|a| {
            let distance = (a + actions - correct) % actions;
            if distance == 0 {
                1.0
            } else if graded && (distance == 1 || distance == actions - 1) {
                0.5
            } else {
                0.0
            }
        })
        .collect()
}

/// The contextual bandit.
#[derive(Clone, Copy, Debug, Default)]
pub struct ContextualBandit;

impl Plugin for ContextualBandit {
    fn id(&self) -> &'static str {
        CONTEXTUAL_BANDIT
    }

    fn version(&self) -> &'static str {
        VERSION
    }

    fn verifier_class(&self) -> VerifierClass {
        VerifierClass::Objective
    }

    fn knobs(&self) -> &'static [Knob] {
        KNOBS
    }

    fn generate(&self, request: &InstanceRequest) -> Result<Instance, PluginError> {
        let difficulty = resolve_difficulty(self.id(), KNOBS, &request.difficulty)?;
        let actions = difficulty["actions"] as usize;
        let graded = difficulty["graded"] == 1;
        let mut rng = SplitMix64::new(request.seed);
        let colour = rng.below(COLOURS.len());
        let level = match request.split {
            Split::Train => rng.below(TRAIN_LEVELS as usize) as u32,
            Split::Dev => TRAIN_LEVELS + rng.below(2) as u32,
        };
        let offered = &ACTIONS[..actions];
        let table = rewards(colour, level, actions, graded);
        let correct = offered[table.iter().position(|r| *r >= 1.0).unwrap_or(0)];

        let mut workspace = Folder::new();
        workspace.put_text("answer.txt", "")?;
        let mut tests = Folder::new();
        tests.put_script(
            "test.sh",
            "#!/bin/sh\n# Pays the reward the table gives the action answer.txt names; see verify.py.\npython3 -I \"${TESTS_DIR:-/tests}/verify.py\"\n",
        )?;
        tests.put_text("verify.py", VERIFY)?;
        let pinned: serde_json::Map<String, serde_json::Value> = offered
            .iter()
            .zip(&table)
            .map(|(a, r)| ((*a).to_string(), json!(r)))
            .collect();
        tests.put_text(
            "rewards.json",
            &format!("{}\n", serde_json::Value::Object(pinned)),
        )?;
        let mut solution = Folder::new();
        solution.put_text("answer.txt", &format!("{correct}\n"))?;
        solution.put_script(
            "solve.sh",
            "#!/bin/sh\ncp \"${SOLUTION_DIR:-/solution}/answer.txt\" \"${APP_DIR:-/app}/answer.txt\"\n",
        )?;
        let listed = offered.join(", ");
        let instruction = format!(
            "# Choose an action\n\nThe panel shows {} at level {level}. Choose one action of: {listed}. Write its name, and nothing else, to `/app/answer.txt`.\n",
            COLOURS[colour]
        );
        let meta = EnvMeta {
            schema_version: SCHEMA_VERSION,
            environment: environment_id(self.id(), VERSION, &difficulty),
            plugin: self.id().into(),
            plugin_version: VERSION.into(),
            verifier_class: VerifierClass::Objective,
            verifier: format!("reward-table@{VERSION}"),
            difficulty,
            split: request.split,
            seed: request.seed,
            solver_tools: SolverTools::Files,
        };
        Ok(Instance::assemble(
            meta,
            Parts {
                instruction,
                workspace,
                packages: vec!["python3".into()],
                tests,
                solution,
                category: "contextual-bandit".into(),
            },
        )?)
    }

    fn attacks(&self, instance: &Instance, _reference: &Reference) -> Vec<Attack> {
        // Every action at once: a verifier that looked for the right name
        // anywhere in the answer would pay it.
        let every = instance
            .instruction()
            .split_once("of: ")
            .and_then(|(_, rest)| rest.split_once(". "))
            .map(|(listed, _)| listed.replace(", ", "\n"))
            .unwrap_or_default();
        vec![Attack {
            name: "every action".into(),
            kind: AttackKind::Tamper,
            edits: vec![Edit::Write {
                path: "answer.txt".into(),
                bytes: format!("{every}\n").into_bytes(),
            }],
        }]
    }

    fn novelty_text(&self, instance: &Instance) -> String {
        instance.instruction().to_string()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // a panic is the failure report
mod tests {
    use super::*;

    #[test]
    fn exactly_one_action_is_correct_and_its_neighbours_earn_half_when_graded() {
        for actions in 2..=6 {
            for (colour, level) in [(0, 0), (3, 7), (2, 5)] {
                let exact = rewards(colour, level, actions, false);
                assert_eq!(exact.iter().filter(|r| **r == 1.0).count(), 1);
                assert_eq!(exact.iter().filter(|r| **r == 0.0).count(), actions - 1);
                let graded = rewards(colour, level, actions, true);
                let halves = graded.iter().filter(|r| **r == 0.5).count();
                assert_eq!(halves, if actions == 2 { 1 } else { 2 });
            }
        }
    }
}
