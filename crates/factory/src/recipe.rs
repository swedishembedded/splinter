// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements declarative learning experiments that take
// a model from agent rollouts to an evaluated candidate, for its clients. If
// your team needs expertise in model fine-tuning pipelines, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The vocabulary every recipe shares: how it names a model
//! ([`ModelRef`], `connection:model` or `champion:RUN`) and how it is refused
//! ([`RecipeError`], every problem at once). A recipe itself is a
//! `splinter.recipe/v2` graph ([`crate::graph`]).

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::ops::names;

/// A model named the way a recipe names one: `connection:model`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModelRef {
    /// The connection in the connections file (`brain`, `hosted`).
    pub connection: String,
    /// The model id the connection's endpoint serves.
    pub model: String,
}

impl FromStr for ModelRef {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text.split_once(':') {
            Some((connection, model))
                if !connection.is_empty()
                    && !model.is_empty()
                    && connection
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') =>
            {
                Ok(Self {
                    connection: connection.into(),
                    model: model.into(),
                })
            }
            _ => Err(format!(
                "{text:?} is not a model reference of the form connection:model"
            )),
        }
    }
}

impl fmt::Display for ModelRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.connection, self.model)
    }
}

impl Serialize for ModelRef {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for ModelRef {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        String::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}

/// Why a recipe is refused; every problem found, not only the first.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("the recipe is invalid:\n  {}", .0.join("\n  "))]
pub struct RecipeError(pub Vec<String>);

/// `text` with the plan of every `evaluation` stage replaced by `plan`: the
/// recipe a run re-evaluating its candidates under another plan runs.
///
/// # Errors
/// The text is not a v2 recipe, or it evaluates nothing.
pub fn with_eval_plan(text: &str, plan: &str) -> Result<String, RecipeError> {
    // Refused unless it is a v2 recipe.
    crate::graph::parse(text)?;
    let mut value: serde_yaml::Value =
        serde_yaml::from_str(text).map_err(|e| RecipeError(vec![e.to_string()]))?;
    let mut replaced = 0;
    if let Some(stages) = value
        .get_mut("stages")
        .and_then(serde_yaml::Value::as_sequence_mut)
    {
        for stage in stages {
            if stage.get("uses").and_then(serde_yaml::Value::as_str) == Some(names::EVALUATION) {
                if let Some(with) = stage
                    .get_mut("with")
                    .and_then(serde_yaml::Value::as_mapping_mut)
                {
                    with.insert("plan".into(), plan.into());
                    replaced += 1;
                }
            }
        }
    }
    if replaced == 0 {
        return Err(RecipeError(vec!["the recipe evaluates nothing".into()]));
    }
    serde_yaml::to_string(&value).map_err(|e| RecipeError(vec![e.to_string()]))
}
