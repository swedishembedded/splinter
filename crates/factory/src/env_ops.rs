// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training environments compiled from a
// learning objective, each admitted only once its verifier holds, for its
// clients. If your team needs expertise in verifiable RL environments, you
// can procure our services by sending an email to info@swedishembedded.com.

//! `environment.build`: a learning objective compiled into admitted,
//! executable training environments ([`splinter_knowledge::factory`]).
//!
//! The objective (a file of the task repository) is classified skill by
//! skill, each buildable skill's generator plugin makes environments along
//! its difficulty knobs, and every environment passes the factory's
//! admission gates (its verifier red-teamed, its base pass rate banded)
//! before it is kept: what `splinter env compile` does, as a stage. The
//! product holds each admitted environment's instances as Harbor task
//! folders by split (`train/ID-SEED`, `dev/ID-SEED`), which `tasks.pin`
//! pins as it pins a compiled collection, and `compile.json`, the report of
//! every environment, the rejected and quarantined ones with their reasons.
//! A `documents` input is the material the document plugins draw from.

use std::collections::BTreeMap;

use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use splinter_core::factory::execution::Isolation;
use splinter_core::factory::operation::{
    ArtifactKind, OperationDescriptor, Port, Reproducibility, OPERATION_SCHEMA,
};
use splinter_core::verifiability::Split;
use splinter_knowledge::factory::admission::AdmissionPolicy;
use splinter_knowledge::factory::compile::compile;
use splinter_knowledge::factory::host::Host;
use splinter_knowledge::factory::objective::{Budget, ObjectiveSpec};
use splinter_knowledge::factory::plugins::{builtin_with, Locations};
use splinter_knowledge::factory::store::EnvironmentStore;
use splinter_sandbox::host::DEFAULT_SEARCH_PATH;

use crate::execute::{Call, Execute, Outcome};
use crate::registry::OperationSpec;

/// The operation's name.
pub const BUILD: &str = "environment.build";

fn one() -> usize {
    1
}

/// `environment.build`'s parameters.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BuildParams {
    /// The objective file (its text is the objective).
    pub objective: String,
    /// What the objective is drawn from, named for the record.
    #[serde(default)]
    pub sources: Vec<String>,
    /// What every environment must respect.
    #[serde(default)]
    pub constraints: Vec<String>,
    /// The most environments to build.
    #[serde(default = "one")]
    pub environments: usize,
    /// Instances of each split per environment.
    #[serde(default = "one")]
    pub instances: usize,
    /// The seed every generator draws from.
    pub seed: u64,
}

/// The `environment.build` operation.
pub struct Build;

fn params(value: &Value) -> Result<BuildParams, String> {
    serde_json::from_value(value.clone()).map_err(|e| e.to_string())
}

impl OperationSpec for Build {
    fn descriptor(&self) -> OperationDescriptor {
        OperationDescriptor {
            schema: OPERATION_SCHEMA.into(),
            name: BUILD.into(),
            revision: 1,
            summary: "an objective compiled into admitted, executable training environments".into(),
            inputs: vec![Port {
                name: "documents".into(),
                kind: ArtifactKind::Files,
                many: false,
                optional: true,
            }],
            outputs: vec![Port {
                name: "environments".into(),
                kind: ArtifactKind::Files,
                many: false,
                optional: false,
            }],
            parameters: serde_json::to_value(schema_for!(BuildParams)).unwrap_or(Value::Null),
            reproducibility: Reproducibility::Seeded,
            isolation: Isolation::Any,
        }
    }

    fn check(&self, value: &Value) -> Result<(), String> {
        let p = params(value)?;
        if p.environments == 0 || p.instances == 0 {
            return Err("environments and instances are at least 1".into());
        }
        Ok(())
    }

    fn files(&self, value: &Value) -> Vec<String> {
        params(value).map(|p| vec![p.objective]).unwrap_or_default()
    }

    fn check_files(&self, value: &Value, files: &BTreeMap<String, String>) -> Vec<String> {
        let Ok(p) = params(value) else {
            return Vec::new();
        };
        match files.get(&p.objective) {
            Some(text) if text.trim().is_empty() => vec![format!("{} is empty", p.objective)],
            _ => Vec::new(),
        }
    }

    fn executor(&self) -> Option<&dyn Execute> {
        Some(self)
    }
}

impl Execute for Build {
    fn implementation(&self) -> String {
        format!("{BUILD} 1")
    }

    fn execute(&self, call: &Call<'_>) -> Result<Outcome, String> {
        let p = params(call.params)?;
        let text = call
            .files
            .get(&p.objective)
            .ok_or_else(|| format!("{} is not pinned", p.objective))?
            .clone();
        let spec = ObjectiveSpec {
            text,
            sources: p.sources,
            constraints: p.constraints,
            budget: Budget {
                environments: p.environments,
                instances_per_split: p.instances,
            },
        };
        let policy = AdmissionPolicy {
            instances_per_split: p.instances,
            ..AdmissionPolicy::default()
        };
        let host = Host::with_path(DEFAULT_SEARCH_PATH);
        let locations = Locations {
            documents: call
                .inputs
                .get("documents")
                .and_then(|v| v.first())
                .map(|view| view.join("files")),
            ..Locations::default()
        };
        let plugins = builtin_with(&locations, &host);
        let scratch = tempfile::tempdir().map_err(|e| e.to_string())?;
        let store = EnvironmentStore::new(scratch.path().join("environments"));
        let report =
            compile(&spec, &plugins, &host, &policy, p.seed, &store).map_err(|e| e.to_string())?;
        let mut exported = 0;
        for admitted in store.list().map_err(|e| e.to_string())? {
            for split in [Split::Train, Split::Dev] {
                let instances = store
                    .instances(&admitted.environment, split)
                    .map_err(|e| e.to_string())?;
                for instance in &instances {
                    let meta = instance.meta();
                    let dir = call
                        .out
                        .join(meta.split.to_string())
                        .join(format!("{}-{}", admitted.environment, meta.seed));
                    instance
                        .folder()
                        .write_to(&dir)
                        .map_err(|e| e.to_string())?;
                    exported += 1;
                }
            }
        }
        let text = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
        std::fs::write(call.out.join("compile.json"), text + "\n").map_err(|e| e.to_string())?;
        if exported == 0 {
            return Ok(Outcome::Rejected(
                "no environment the objective asks for was admitted (compile.json says why)".into(),
            ));
        }
        Ok(Outcome::Produced)
    }
}
