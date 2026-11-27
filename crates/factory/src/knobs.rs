// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning loops whose every setting is proven
// to move what it is meant to move, for its clients. If your team needs
// expertise in observing and steering agent training loops, you can procure
// our services by sending an email to info@swedishembedded.com.

//! A recipe's knobs: each one a parameter of one stage that the recipe
//! declares together with the state variable of the loop it is meant to
//! move, so that a probe can turn it alone and show whether it does.
//!
//! ```yaml
//! knobs:
//!   epochs:
//!     set: lora-g2.params.epochs      # STAGE.PATH of the stage's `with`
//!     clears: [params.steps]          # what the knob replaces, removed when it is set
//!     moves: training.epochs_at_selected
//!     min_change: 0.05                # the least change that counts as moved (any change by default)
//! ```
//!
//! A knob changes no stage identity by being declared: the section is not
//! part of the plan. [`perturb`] writes the recipe with one knob set and
//! every other parameter as it was.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The state variables a knob may name: each an observation of a
/// generation (`splinter observe`), read by [`crate::observe::probe`].
pub const VARIABLES: &[&str] = &[
    "training.epochs_at_selected",
    "training.epochs_completed",
    "training.steps",
    "training.selected_step",
    "training.draws_per_epoch",
    "training.tokens",
    "data.train_records",
    "data.own_share_records",
    "data.own_share_characters",
    "data.validation_own_share_records",
    "data.frontier_share",
    "data.action_sequences",
    "data.own_pass_rate",
    "data.other_pass_rate",
];

/// One knob as a recipe declares it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KnobSpec {
    /// `STAGE.PATH`: the stage's id and the path in its `with` the knob sets.
    pub set: String,
    /// Paths in the same stage's `with` that the knob replaces and that are
    /// removed when it is set (`params.steps` for `params.epochs`).
    #[serde(default)]
    pub clears: Vec<String>,
    /// The state variable it is meant to move (one of [`VARIABLES`]).
    pub moves: String,
    /// The least absolute change of the variable that counts as moved; any
    /// change when unset.
    #[serde(default)]
    pub min_change: Option<f64>,
}

impl KnobSpec {
    /// The stage it sets and the path in its `with`, split at the first `.`.
    #[must_use]
    pub fn target(&self) -> Option<(&str, Vec<&str>)> {
        let (stage, path) = self.set.split_once('.')?;
        let path: Vec<&str> = path.split('.').collect();
        (!stage.is_empty() && path.iter().all(|p| !p.is_empty())).then_some((stage, path))
    }
}

/// What is wrong with the knobs of a recipe whose stages are `stages`.
#[must_use]
pub fn problems(knobs: &BTreeMap<String, KnobSpec>, stages: &[&str]) -> Vec<String> {
    let mut out = Vec::new();
    for (name, knob) in knobs {
        match knob.target() {
            None => out.push(format!(
                "knob {name}: set {:?} is not STAGE.PATH (a stage id and a path in its `with`)",
                knob.set
            )),
            Some((stage, _)) if !stages.contains(&stage) => {
                out.push(format!("knob {name}: no stage {stage:?} to set"));
            }
            Some(_) => {}
        }
        if !VARIABLES.contains(&knob.moves.as_str()) {
            out.push(format!(
                "knob {name}: moves {:?}, which is not a state variable ({})",
                knob.moves,
                VARIABLES.join(", ")
            ));
        }
        if knob
            .min_change
            .is_some_and(|m| !(m.is_finite() && m >= 0.0))
        {
            out.push(format!(
                "knob {name}: min_change must be a number of at least 0"
            ));
        }
    }
    out
}

/// The recipe `text` with the knob `knob` set to `value` and what it clears
/// removed; every other byte of meaning unchanged.
///
/// # Errors
/// The recipe does not parse, or names no such knob or stage.
pub fn perturb(text: &str, knob: &str, value: &Value) -> Result<String, String> {
    let recipe = crate::graph::parse(text).map_err(|e| e.0.join("; "))?;
    let spec = recipe
        .knobs
        .get(knob)
        .ok_or_else(|| format!("the recipe declares no knob {knob:?}"))?;
    let (stage, path) = spec
        .target()
        .ok_or_else(|| format!("knob {knob}: set {:?} is not STAGE.PATH", spec.set))?;
    let mut doc: serde_yaml::Value = serde_yaml::from_str(text).map_err(|e| e.to_string())?;
    let stages = doc
        .get_mut("stages")
        .and_then(serde_yaml::Value::as_sequence_mut)
        .ok_or("the recipe has no stages")?;
    let entry = stages
        .iter_mut()
        .find(|s| s.get("id").and_then(serde_yaml::Value::as_str) == Some(stage))
        .ok_or_else(|| format!("knob {knob}: no stage {stage:?}"))?;
    let map = entry
        .as_mapping_mut()
        .ok_or_else(|| format!("stage {stage} is not a mapping"))?;
    let with = map
        .entry("with".into())
        .or_insert_with(|| serde_yaml::Value::Mapping(serde_yaml::Mapping::new()));
    for cleared in &spec.clears {
        remove(with, &cleared.split('.').collect::<Vec<_>>());
    }
    let mut at = with;
    let (last, parents) = path.split_last().ok_or("an empty path")?;
    for p in parents {
        let m = at
            .as_mapping_mut()
            .ok_or_else(|| format!("knob {knob}: {p} is not a mapping in stage {stage}"))?;
        at = m
            .entry((*p).into())
            .or_insert_with(|| serde_yaml::Value::Mapping(serde_yaml::Mapping::new()));
    }
    let value: serde_yaml::Value = serde_yaml::to_value(value).map_err(|e| e.to_string())?;
    at.as_mapping_mut()
        .ok_or_else(|| format!("knob {knob}: the parent of {last} is not a mapping"))?
        .insert((*last).into(), value);
    serde_yaml::to_string(&doc).map_err(|e| e.to_string())
}

/// Removes `path` below `value`, if it is there.
fn remove(value: &mut serde_yaml::Value, path: &[&str]) {
    match path {
        [] => {}
        [last] => {
            if let Some(m) = value.as_mapping_mut() {
                m.remove(*last);
            }
        }
        [first, rest @ ..] => {
            if let Some(next) = value.get_mut(*first) {
                remove(next, rest);
            }
        }
    }
}

/// `NAME=VALUE`, the value read as YAML (`2`, `1.5`, `true`, `[a, b]`).
///
/// # Errors
/// No `=`, or a value that is not YAML.
pub fn assignment(text: &str) -> Result<(String, Value), String> {
    let (name, value) = text
        .split_once('=')
        .ok_or_else(|| format!("{text:?} is not NAME=VALUE"))?;
    let value: Value = serde_yaml::from_str(value).map_err(|e| format!("{text}: {e}"))?;
    Ok((name.to_owned(), value))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    // seed: a recipe states every stage's seed; the test checks it is kept, nothing draws from it.
    const RECIPE: &str = "schema: splinter.recipe/v2\nname: r\nstages:\n  \
- id: lora\n    uses: model.train\n    inputs: { dataset: $stages.sft.dataset }\n    \
with: { base: student, params: { rank: 8, steps: 800 }, seed: 1 }\n\
knobs:\n  epochs:\n    set: lora.params.epochs\n    clears: [params.steps]\n    moves: training.epochs_at_selected\n";

    #[test]
    fn a_knob_is_set_alone_and_clears_what_it_replaces() {
        let text = perturb(RECIPE, "epochs", &json!(1.89)).unwrap();
        let recipe = crate::graph::parse(&text).unwrap();
        let params = &recipe.stages[0].params["params"];
        assert_eq!(params, &json!({"rank": 8, "epochs": 1.89}));
        assert_eq!(
            recipe.stages[0].params["seed"], 1,
            "every other parameter is kept"
        );
        assert!(perturb(RECIPE, "nope", &json!(1)).is_err());
    }

    #[test]
    fn a_knob_names_a_stage_and_a_state_variable() {
        let recipe = crate::graph::parse(RECIPE).unwrap();
        assert!(problems(&recipe.knobs, &["lora"]).is_empty());
        let mut bad = recipe.knobs.clone();
        bad.insert(
            "x".into(),
            KnobSpec {
                set: "missing.p".into(),
                clears: Vec::new(),
                moves: "vibes".into(),
                min_change: Some(-1.0),
            },
        );
        assert_eq!(problems(&bad, &["lora"]).len(), 3);
        assert_eq!(
            assignment("epochs=1.89").unwrap(),
            ("epochs".into(), json!(1.89))
        );
        assert!(assignment("epochs").is_err());
    }
}
