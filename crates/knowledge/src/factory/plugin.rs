// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements environment generators with difficulty
// knobs and trustworthy verifiers, for its clients. If your team needs
// expertise in procedural task generation for reinforcement learning, you
// can procure our services by sending an email to info@swedishembedded.com.

//! What an environment plugin is: a generator, a verifier and a difficulty
//! knob, drawn from by split and seed.
//!
//! A plugin builds [`Instance`]s. The instance carries its own reset (the
//! workspace, and a reset script for state outside files), its own verifier
//! (`tests/test.sh`) and its own reference (`solution/solve.sh`), so
//! everything admission needs is in the task folder a campaign will use:
//! nothing is checked by code that the campaign's folder does not contain.
//!
//! A plugin's `generate` is a pure function of its request. The same
//! request gives byte-identical folders, which is what makes an environment
//! reproducible and a dev instance recoverable from its seed.

use std::collections::BTreeMap;

use splinter_core::verifiability::{Split, VerifierClass};

use super::artifact::{ArtifactError, Instance};
use super::attack::Attack;
use super::folder::FolderError;
use super::host::{Host, Session};

/// A difficulty knob: a named integer with a range and a default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Knob {
    /// The knob's name.
    pub name: &'static str,
    /// The lowest value.
    pub min: u32,
    /// The highest value.
    pub max: u32,
    /// The value when none is given.
    pub default: u32,
    /// What turning it up does.
    pub doc: &'static str,
}

/// What a plugin is asked for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InstanceRequest {
    /// The part of the generator to draw from.
    pub split: Split,
    /// The seed that picks the instance.
    pub seed: u64,
    /// The knobs, every one of the plugin's present and in range
    /// ([`resolve_difficulty`]).
    pub difficulty: BTreeMap<String, u32>,
}

/// Why an instance could not be generated.
#[derive(Debug, thiserror::Error)]
pub enum PluginError {
    /// A knob the plugin does not have.
    #[error("{plugin} has no difficulty knob {name:?} (it has {known})")]
    UnknownKnob {
        /// The plugin.
        plugin: String,
        /// The name asked for.
        name: String,
        /// The names it has.
        known: String,
    },
    /// A knob value outside its range.
    #[error("{plugin}: {name} = {value} is outside {min}..={max}")]
    KnobRange {
        /// The plugin.
        plugin: String,
        /// The knob.
        name: String,
        /// The value asked for.
        value: u32,
        /// Its lowest value.
        min: u32,
        /// Its highest.
        max: u32,
    },
    /// The artifact could not be assembled.
    #[error(transparent)]
    Artifact(#[from] ArtifactError),
    /// A folder could not be built.
    #[error(transparent)]
    Folder(#[from] FolderError),
    /// The generator could not produce a valid instance for this request.
    #[error("{0}")]
    Generation(String),
}

/// The knobs of a plugin with `given` values over its defaults; unknown
/// names and out-of-range values are refused.
pub fn resolve_difficulty(
    plugin: &str,
    knobs: &[Knob],
    given: &BTreeMap<String, u32>,
) -> Result<BTreeMap<String, u32>, PluginError> {
    let known = || knobs.iter().map(|k| k.name).collect::<Vec<_>>().join(", ");
    if let Some(name) = given
        .keys()
        .find(|n| !knobs.iter().any(|k| k.name == n.as_str()))
    {
        return Err(PluginError::UnknownKnob {
            plugin: plugin.into(),
            name: name.clone(),
            known: known(),
        });
    }
    let mut resolved = BTreeMap::new();
    for knob in knobs {
        let value = given.get(knob.name).copied().unwrap_or(knob.default);
        if !(knob.min..=knob.max).contains(&value) {
            return Err(PluginError::KnobRange {
                plugin: plugin.into(),
                name: knob.name.into(),
                value,
                min: knob.min,
                max: knob.max,
            });
        }
        resolved.insert(knob.name.to_string(), value);
    }
    Ok(resolved)
}

/// What the reference solution changed in the workspace: the files the
/// verifier's answer lives in.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Reference {
    /// Files the reference created or changed, with their new bytes.
    pub changed: BTreeMap<String, Vec<u8>>,
    /// Files the reference removed.
    pub removed: Vec<String>,
}

impl Reference {
    /// What `session`'s workspace changed to relative to `before`.
    #[must_use]
    pub fn between(before: &super::folder::Folder, after: &super::folder::Folder) -> Self {
        let mut changed = BTreeMap::new();
        for (path, entry) in after.entries() {
            if before.get(path) != Some(entry) {
                changed.insert(path.to_string(), entry.bytes.clone());
            }
        }
        let removed = before
            .paths()
            .filter(|p| after.get(p).is_none())
            .map(ToString::to_string)
            .collect();
        Self { changed, removed }
    }
}

/// A generator, a verifier and a difficulty knob for one kind of skill.
pub trait Plugin: Send + Sync {
    /// The plugin's id, as an objective's classification names it.
    fn id(&self) -> &'static str;

    /// The plugin's version. Any change to generation or verification is a
    /// new version, so an environment id never means two things.
    fn version(&self) -> &'static str;

    /// How the plugin's rewards are established.
    fn verifier_class(&self) -> VerifierClass;

    /// The difficulty knobs.
    fn knobs(&self) -> &'static [Knob];

    /// The instance a request picks.
    fn generate(&self, request: &InstanceRequest) -> Result<Instance, PluginError>;

    /// Attacks specific to this plugin's verifier, beyond the generic ones
    /// admission always runs.
    fn attacks(&self, _instance: &Instance, _reference: &Reference) -> Vec<Attack> {
        Vec::new()
    }

    /// The text two instances of this plugin are compared by for novelty:
    /// what varies between instances, without the boilerplate they share.
    fn novelty_text(&self, instance: &Instance) -> String {
        instance.instruction().to_string()
    }

    /// An independent check that the instance's oracle is right, when the
    /// plugin has a second oracle (the same program in another language).
    /// `None` when it has none; otherwise a description of what agreed, or
    /// why it did not.
    fn cross_check(
        &self,
        _request: &InstanceRequest,
        _host: &Host,
    ) -> Option<Result<String, String>> {
        None
    }

    /// Checks that only this plugin can make of a laid-out instance, after
    /// the reference ran (state the generic gates cannot see).
    fn extra_gates(
        &self,
        _instance: &Instance,
        _session: &Session,
    ) -> Vec<(String, Result<(), String>)> {
        Vec::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // a panic is the failure report
mod tests {
    use super::*;

    const KNOBS: &[Knob] = &[
        Knob {
            name: "depth",
            min: 1,
            max: 5,
            default: 2,
            doc: "",
        },
        Knob {
            name: "width",
            min: 0,
            max: 9,
            default: 3,
            doc: "",
        },
    ];

    #[test]
    fn defaults_fill_the_knobs_a_caller_did_not_set() {
        let given = BTreeMap::from([("depth".to_string(), 4)]);
        let resolved = resolve_difficulty("p", KNOBS, &given).unwrap();
        assert_eq!(resolved["depth"], 4);
        assert_eq!(resolved["width"], 3);
    }

    #[test]
    fn an_unknown_knob_and_an_out_of_range_value_are_refused_by_name() {
        let unknown = BTreeMap::from([("depht".to_string(), 1)]);
        let err = resolve_difficulty("p", KNOBS, &unknown)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("depht") && err.contains("depth, width"),
            "{err}"
        );
        let high = BTreeMap::from([("depth".to_string(), 6)]);
        let err = resolve_difficulty("p", KNOBS, &high)
            .unwrap_err()
            .to_string();
        assert!(err.contains("depth = 6") && err.contains("1..=5"), "{err}");
    }
}
