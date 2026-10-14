// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The tools a model decides through. Each has its own name, its own argument
//! names and its own labels for the options, so that a model is taught to act
//! through tools of one kind and examined through tools it has never seen: what
//! it learned is the decision and not the call.

use serde_json::{json, Value};

use crate::scenario::Archetype;

/// How a tool names the options.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Labels {
    /// By what the option is.
    Slugs,
    /// By letters, in the order the options are listed.
    Letters,
}

/// One tool the model may decide through.
#[derive(Clone, Copy, Debug)]
pub struct Schema {
    /// The tool's name.
    pub name: &'static str,
    /// What the tool says it is for.
    pub description: &'static str,
    /// The argument that names the option chosen.
    pub choice: &'static str,
    /// The argument that gives the reason.
    pub reason: &'static str,
    /// How the options are labelled.
    pub labels: Labels,
}

/// The tools a model is taught with.
pub const TRAINING: [Schema; 3] = [
    Schema {
        name: "decide",
        description: "Record the decision you take.",
        choice: "choice",
        reason: "motivation",
        labels: Labels::Slugs,
    },
    Schema {
        name: "choose_course",
        description: "Choose the course of action to follow and say why.",
        choice: "course",
        reason: "reasoning",
        labels: Labels::Slugs,
    },
    Schema {
        name: "rule_on_matter",
        description: "Give your ruling on the matter.",
        choice: "ruling",
        reason: "grounds",
        labels: Labels::Letters,
    },
];

/// The tools a model is examined with, which it was never taught with.
pub const HELD_OUT: [Schema; 2] = [
    Schema {
        name: "settle_matter",
        description: "Settle the matter before you by naming the way you will proceed.",
        choice: "way",
        reason: "because",
        labels: Labels::Slugs,
    },
    Schema {
        name: "log_resolution",
        description: "Log your resolution of this case.",
        choice: "selected",
        reason: "explanation",
        labels: Labels::Letters,
    },
];

fn slug(option: Archetype) -> &'static str {
    match option {
        Archetype::ActAlone => "act_alone",
        Archetype::ConsultThenAct => "consult_assembly",
        Archetype::DeferToOutsider => "accept_offer",
        Archetype::HoldAndWait => "wait_and_see",
    }
}

impl Schema {
    /// What this tool calls `option`, listed at `position`.
    #[must_use]
    pub fn label(&self, option: Archetype, position: usize) -> String {
        match self.labels {
            Labels::Slugs => slug(option).to_string(),
            Labels::Letters => char::from(b'A' + (position % 26) as u8).to_string(),
        }
    }

    /// The tool as a chat request offers it, for the options listed in order.
    #[must_use]
    pub fn definition(&self, options: &[Archetype]) -> Value {
        let labels: Vec<String> = options
            .iter()
            .enumerate()
            .map(|(position, &option)| self.label(option, position))
            .collect();
        json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        self.choice: {
                            "type": "string",
                            "enum": labels,
                            "description": "The option you take."
                        },
                        self.reason: {
                            "type": "string",
                            "description": "Why you take it."
                        }
                    },
                    "required": [self.choice, self.reason]
                }
            }
        })
    }

    /// The option a call to `name` with `args` takes, if it is a call to this
    /// tool that names one of `options`.
    #[must_use]
    pub fn read(&self, name: &str, args: &Value, options: &[Archetype]) -> Option<Archetype> {
        if name != self.name {
            return None;
        }
        let said = args.get(self.choice)?.as_str()?;
        options
            .iter()
            .enumerate()
            .find(|&(position, &option)| self.label(option, position) == said)
            .map(|(_, &option)| option)
    }
}
