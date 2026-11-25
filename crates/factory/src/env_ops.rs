// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training environments compiled from a
// learning objective, each admitted only once its verifier holds, for its
// clients. If your team needs expertise in verifiable RL environments, you
// can procure our services by sending an email to info@swedishembedded.com.

//! `environment.build` and `environment.qualify`: a learning objective
//! compiled into executable training environments
//! ([`splinter_knowledge::factory`]), and admitted to a campaign once a base
//! model's measured pass rate shows there is something to learn.
//!
//! The objective (a file of the task repository) is classified skill by
//! skill, each buildable skill's generator plugin makes environments along
//! its difficulty knobs, and every environment passes the factory's
//! admission gates (its verifier red-teamed, its base pass rate banded)
//! before it is kept: what `splinter env compile` does, as a stage. The
//! product holds each admitted environment's instances as Harbor task
//! folders by split (`train/ID-SEED`, `dev/ID-SEED`), which `tasks.pin`
//! pins as it pins a compiled collection, and `compile.json`, the report of
//! every environment, the rejected and quarantined ones with their reasons,
//! and `admission/ENV.json`, each environment's admission report. A
//! `documents` input is the material the document plugins draw from.
//!
//! An environment whose gates all passed still waits for its band: a base
//! policy it always solves, or never solves, gives a group-relative trainer
//! nothing to learn from. `environment.qualify` reads the attempts an
//! attempt stage made at the built environments' dev instances, measures
//! each environment's base pass rate from them (counted attempts, by the
//! one model that made them), and re-judges its report against its band
//! policy. The product holds the instances of the environments admitted now,
//! by split as the build's are, and `qualify.json`, every re-judged report;
//! with none admitted, the stage is rejected with each one's reasons.

use std::collections::BTreeMap;

use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use splinter_core::factory::execution::Isolation;
use splinter_core::factory::operation::{
    ArtifactKind, OperationDescriptor, Port, Reproducibility, OPERATION_SCHEMA,
};
use splinter_core::verifiability::Split;
use splinter_knowledge::factory::admission::{AdmissionPolicy, AdmissionReport, Status};
use splinter_knowledge::factory::band::BandMeasurement;
use splinter_knowledge::factory::compile::compile;
use splinter_knowledge::factory::host::Host;
use splinter_knowledge::factory::objective::{Budget, ObjectiveSpec};
use splinter_knowledge::factory::plugins::{builtin_with, Locations};
use splinter_knowledge::factory::store::EnvironmentStore;
use splinter_sandbox::host::DEFAULT_SEARCH_PATH;

use crate::execute::{Call, Execute, Outcome};
use crate::registry::OperationSpec;
use crate::run::rollout::trials_of;
use crate::tree::link_tree;

/// The operation's name.
pub const BUILD: &str = "environment.build";
/// The qualification's name.
pub const QUALIFY: &str = "environment.qualify";

/// Each environment's admission report, in a build's product.
const ADMISSION_DIR: &str = "admission";

fn files_port(name: &str, optional: bool) -> Port {
    Port {
        name: name.into(),
        kind: ArtifactKind::Files,
        many: false,
        optional,
    }
}

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
            inputs: vec![files_port("documents", true)],
            outputs: vec![files_port("environments", false)],
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
        let reports = call.out.join(ADMISSION_DIR);
        std::fs::create_dir_all(&reports).map_err(|e| e.to_string())?;
        for admitted in store.list().map_err(|e| e.to_string())? {
            write_json(
                &reports.join(format!("{}.json", admitted.environment)),
                &admitted,
            )?;
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
        write_json(&call.out.join("compile.json"), &report)?;
        if exported == 0 {
            return Ok(Outcome::Rejected(
                "no environment the objective asks for was admitted (compile.json says why)".into(),
            ));
        }
        Ok(Outcome::Produced)
    }
}

fn write_json(path: &std::path::Path, value: &impl Serialize) -> Result<(), String> {
    let text = serde_json::to_string_pretty(value).map_err(|e| e.to_string())?;
    std::fs::write(path, text + "\n").map_err(|e| format!("{}: {e}", path.display()))
}

/// The `environment.qualify` operation.
pub struct Qualify;

impl OperationSpec for Qualify {
    fn descriptor(&self) -> OperationDescriptor {
        OperationDescriptor {
            schema: OPERATION_SCHEMA.into(),
            name: QUALIFY.into(),
            revision: 1,
            summary: "built environments admitted once a base model's pass rate is in their band"
                .into(),
            inputs: vec![
                files_port("environments", false),
                Port {
                    name: "attempts".into(),
                    kind: ArtifactKind::Attempts,
                    many: false,
                    optional: false,
                },
            ],
            outputs: vec![files_port("environments", false)],
            parameters: serde_json::json!({ "type": "object", "additionalProperties": false }),
            reproducibility: Reproducibility::Deterministic,
            isolation: Isolation::Any,
        }
    }

    fn check(&self, value: &Value) -> Result<(), String> {
        match value {
            Value::Null => Ok(()),
            Value::Object(map) if map.is_empty() => Ok(()),
            _ => Err("takes no parameters".into()),
        }
    }

    fn executor(&self) -> Option<&dyn Execute> {
        Some(self)
    }
}

/// The base pass rate the trials measured for `report`'s dev instances,
/// `None` when no counted trial attempted one.
fn measured(
    report: &AdmissionReport,
    trials: &[splinter_core::factory::product::TrialResult],
) -> Result<Option<BandMeasurement>, String> {
    let dev: Vec<String> = report
        .dev_seeds
        .iter()
        .map(|seed| format!("{}/{}-{seed}", Split::Dev, report.environment))
        .collect();
    let counted: Vec<_> = trials
        .iter()
        .filter(|t| t.grade.counted && dev.contains(&t.task))
        .collect();
    let mut models: Vec<String> = counted
        .iter()
        .map(|t| t.model.clone().unwrap_or_else(|| t.agent.name.clone()))
        .collect();
    models.sort();
    models.dedup();
    match models.as_slice() {
        [] => Ok(None),
        [model] => Ok(Some(BandMeasurement {
            model: model.clone(),
            attempts: u32::try_from(counted.len()).map_err(|e| e.to_string())?,
            passes: u32::try_from(
                counted
                    .iter()
                    .filter(|t| t.grade.passed == Some(true))
                    .count(),
            )
            .map_err(|e| e.to_string())?,
        })),
        many => Err(format!(
            "{} was attempted by {}; a base pass rate is one model's",
            report.environment,
            many.join(", ")
        )),
    }
}

impl Execute for Qualify {
    fn implementation(&self) -> String {
        format!("{QUALIFY} 1")
    }

    fn execute(&self, call: &Call<'_>) -> Result<Outcome, String> {
        let view = |name: &str| {
            call.inputs
                .get(name)
                .and_then(|v| v.first())
                .ok_or_else(|| format!("no {name} are bound"))
        };
        let built = view("environments")?;
        let trials = trials_of(view("attempts")?).map_err(|e| e.to_string())?;
        let unreadable = |e: std::io::Error| format!("not a built environments product: {e}");
        let mut paths = std::fs::read_dir(built.join(ADMISSION_DIR))
            .map_err(unreadable)?
            .map(|e| e.map(|e| e.path()).map_err(unreadable))
            .collect::<Result<Vec<_>, _>>()?;
        paths.sort();
        let mut judged = Vec::new();
        for path in paths {
            let text = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
            let report: AdmissionReport =
                serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
            let measurement = match measured(&report, &trials) {
                Ok(m) => m,
                Err(why) => return Ok(Outcome::Rejected(why)),
            };
            judged.push(report.with_band(measurement.as_ref()));
        }
        write_json(&call.out.join("qualify.json"), &judged)?;
        let mut admitted = 0;
        for report in judged.iter().filter(|r| r.status == Status::Admitted) {
            for (split, seeds) in [
                (Split::Train, &report.train_seeds),
                (Split::Dev, &report.dev_seeds),
            ] {
                for seed in seeds {
                    let folder = std::path::Path::new(&split.to_string())
                        .join(format!("{}-{seed}", report.environment));
                    link_tree(&built.join(&folder), &call.out.join(&folder))
                        .map_err(|e| e.to_string())?;
                }
            }
            admitted += 1;
        }
        if admitted == 0 {
            let reasons: Vec<String> = judged
                .iter()
                .map(|r| format!("{}: {}", r.environment, r.reasons.join("; ")))
                .collect();
            return Ok(Outcome::Rejected(format!(
                "no environment was admitted ({})",
                reasons.join(" | ")
            )));
        }
        Ok(Outcome::Produced)
    }
}
