// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that compile a learning objective
// into validated, versioned training environments, for its clients. If your
// team needs expertise in environment factories for agent reinforcement
// learning, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The reports of `env ...`.

use std::fmt::Write as _;
use std::path::PathBuf;

use serde::Serialize;
use splinter_sdk::knowledge::factory::admission::{AdmissionPolicy, AdmissionReport};
use splinter_sdk::knowledge::factory::band::BandMeasurement;
use splinter_sdk::knowledge::factory::compile::{compile, CompileReport};
use splinter_sdk::knowledge::factory::host::Host;
use splinter_sdk::knowledge::factory::objective::{
    classify_all, Budget, Classification, ObjectiveSpec,
};
use splinter_sdk::knowledge::factory::plugins::{builtin, builtin_with, Locations};
use splinter_sdk::knowledge::factory::store::EnvironmentStore;
use splinter_sdk::sandbox::host::DEFAULT_SEARCH_PATH;
use splinter_sdk::vocabulary::verifiability::Split;
use splinter_sdk::Error;

use crate::cli::{EnvCommand, ObjectiveArgs};
use crate::output::{emit, Report};

/// The store directory under the state root when none is given.
const DEFAULT_STORE: &str = "environments";

fn refused(what: impl std::fmt::Display) -> Error {
    Error::Refused(what.to_string())
}

fn objective_of(args: ObjectiveArgs, budget: Budget) -> Result<ObjectiveSpec, Error> {
    let text = match (args.text, args.file) {
        (Some(text), _) => text,
        (None, Some(file)) => std::fs::read_to_string(&file)
            .map_err(|e| refused(format!("{}: {e}", file.display())))?,
        (None, None) => return Err(refused("give --text or --file")),
    };
    if text.trim().is_empty() {
        return Err(refused("the objective is empty"));
    }
    Ok(ObjectiveSpec {
        text,
        sources: args.sources,
        constraints: args.constraints,
        budget,
    })
}

fn store_at(ctx: &splinter_sdk::Context, store: Option<PathBuf>) -> EnvironmentStore {
    EnvironmentStore::new(store.unwrap_or_else(|| ctx.root().path().join(DEFAULT_STORE)))
}

/// What `env classify` found.
#[derive(Debug, Serialize)]
pub struct Classified {
    /// One entry per skill.
    pub skills: Vec<Classification>,
}

impl Report for Classified {
    fn human(&self) -> String {
        let mut out = String::new();
        for skill in &self.skills {
            let label = skill.label().map_or(String::new(), |l| format!(" [{l}]"));
            let _ = writeln!(
                out,
                "{}\n  {}{label}; plugin: {}",
                skill.skill,
                skill.class,
                skill.plugin.as_deref().unwrap_or("none")
            );
            for reason in &skill.reasons {
                let _ = writeln!(out, "  - {reason}");
            }
        }
        out
    }
}

/// The plugins and their knobs.
#[derive(Debug, Serialize)]
pub struct Plugins {
    /// One entry per plugin.
    pub plugins: Vec<PluginInfo>,
}

/// One plugin.
#[derive(Debug, Serialize)]
pub struct PluginInfo {
    /// Its id.
    pub id: String,
    /// Its version.
    pub version: String,
    /// How its rewards are established.
    pub verifier_class: String,
    /// Its knobs: name, range, default and what turning it up does.
    pub knobs: Vec<String>,
}

impl Report for Plugins {
    fn human(&self) -> String {
        let mut out = String::new();
        for plugin in &self.plugins {
            let _ = writeln!(
                out,
                "{} v{} ({})",
                plugin.id, plugin.version, plugin.verifier_class
            );
            for knob in &plugin.knobs {
                let _ = writeln!(out, "  {knob}");
            }
        }
        out
    }
}

/// What `env compile` built.
#[derive(Debug, Serialize)]
pub struct Compiled {
    /// The store written.
    pub store: PathBuf,
    /// What became of each skill.
    pub report: CompileReport,
}

impl Report for Compiled {
    fn human(&self) -> String {
        let mut out = format!(
            "seed {}: {} admitted, {} waiting for a base pass rate, {} quarantined\nstore: {}\n",
            self.report.seed,
            self.report.admitted(),
            self.report.pending_band(),
            self.report.quarantined(),
            self.store.display()
        );
        for skill in &self.report.skills {
            let label = skill
                .label
                .as_deref()
                .map_or(String::new(), |l| format!(" [{l}]"));
            let _ = writeln!(
                out,
                "\n{}\n  {}{label}",
                skill.classification.skill, skill.classification.class
            );
            if let Some(why) = &skill.not_built {
                let _ = writeln!(out, "  not built: {why}");
            }
            for env in &skill.environments {
                let _ = writeln!(
                    out,
                    "  {} {:?} {:?}",
                    env.environment, env.status, env.difficulty
                );
                for reason in &env.reasons {
                    let _ = writeln!(out, "    - {reason}");
                }
            }
        }
        out
    }
}

/// The task folders `env export` wrote.
#[derive(Debug, Serialize)]
pub struct Exported {
    /// One directory per instance.
    pub directories: Vec<PathBuf>,
}

impl Report for Exported {
    fn human(&self) -> String {
        self.directories
            .iter()
            .map(|d| format!("{}\n", d.display()))
            .collect()
    }
}

/// The environments in a store.
#[derive(Debug, Serialize)]
pub struct Listed {
    /// One report per environment.
    pub environments: Vec<AdmissionReport>,
}

impl Report for Listed {
    fn human(&self) -> String {
        if self.environments.is_empty() {
            return "no environments\n".into();
        }
        self.environments
            .iter()
            .map(|r| {
                format!(
                    "{} {:?} {} {:?}{}\n",
                    r.environment,
                    r.status,
                    r.plugin,
                    r.difficulty,
                    r.label
                        .as_deref()
                        .map_or(String::new(), |l| format!(" [{l}]"))
                )
            })
            .collect()
    }
}

/// One environment's admission report.
#[derive(Debug, Serialize)]
pub struct Shown {
    /// The report.
    pub report: AdmissionReport,
}

impl Report for Shown {
    fn human(&self) -> String {
        let r = &self.report;
        let mut out = format!(
            "{} {:?}\n{} v{} ({}){}\nadmission seed {}, {} attacks run, blind rate {:.0}%\n",
            r.environment,
            r.status,
            r.plugin,
            r.plugin_version,
            r.verifier_class,
            r.label
                .as_deref()
                .map_or(String::new(), |l| format!(" [{l}]")),
            r.admission_seed,
            r.attacks_run,
            r.blind_rate * 100.0
        );
        for gate in &r.gates {
            let _ = writeln!(
                out,
                "  {} {}: {}",
                if gate.passed { "pass" } else { "FAIL" },
                gate.name,
                gate.detail
            );
        }
        for beaten in r.beaten.iter().take(10) {
            let _ = writeln!(
                out,
                "  beaten: {} on {}-{}: {}",
                beaten.attack, beaten.split, beaten.seed, beaten.detail
            );
        }
        for reason in &r.reasons {
            let _ = writeln!(out, "  - {reason}");
        }
        out
    }
}

/// Runs an `env` command and prints its report.
///
/// # Errors
/// The objective or store cannot be read, or admission cannot run.
pub fn run(ctx: &splinter_sdk::Context, json: bool, command: EnvCommand) -> Result<(), Error> {
    match command {
        EnvCommand::Classify { objective } => {
            let spec = objective_of(objective, Budget::default())?;
            emit(
                json,
                &Classified {
                    skills: classify_all(&spec),
                },
            );
        }
        EnvCommand::Plugins => {
            let plugins = builtin()
                .iter()
                .map(|p| PluginInfo {
                    id: p.id().into(),
                    version: p.version().into(),
                    verifier_class: p.verifier_class().to_string(),
                    knobs: p
                        .knobs()
                        .iter()
                        .map(|k| {
                            format!(
                                "{} {}..={} (default {}): {}",
                                k.name, k.min, k.max, k.default, k.doc
                            )
                        })
                        .collect(),
                })
                .collect();
            emit(json, &Plugins { plugins });
        }
        EnvCommand::Compile {
            objective,
            store,
            environments,
            instances,
            seed,
            repositories,
            emulator,
            documents,
        } => {
            if environments == 0 || instances == 0 {
                return Err(refused("--environments and --instances must be at least 1"));
            }
            let spec = objective_of(
                objective,
                Budget {
                    environments,
                    instances_per_split: instances,
                },
            )?;
            let seed = splinter_sdk::seed::resolve("seed", seed);
            let store = store_at(ctx, store);
            let policy = AdmissionPolicy {
                instances_per_split: instances,
                ..AdmissionPolicy::default()
            };
            let mut host = Host::with_path(DEFAULT_SEARCH_PATH);
            if let Some(emulator) = &emulator {
                host = host.with_variable("VEMU", &emulator.display().to_string());
            }
            let locations = Locations {
                repositories,
                emulator,
                documents,
            };
            let plugins = builtin_with(&locations, &host);
            let report = compile(&spec, &plugins, &host, &policy, seed, &store).map_err(refused)?;
            emit(
                json,
                &Compiled {
                    store: store.root().to_path_buf(),
                    report,
                },
            );
        }
        EnvCommand::List { store } => {
            let environments = store_at(ctx, store).list().map_err(refused)?;
            emit(json, &Listed { environments });
        }
        EnvCommand::Show { environment, store } => {
            let report = store_at(ctx, store).report(&environment).map_err(refused)?;
            emit(json, &Shown { report });
        }
        EnvCommand::Export {
            environment,
            split,
            out,
            store,
        } => {
            let split: Split = split.parse().map_err(refused)?;
            let store = store_at(ctx, store);
            let instances = store.instances(&environment, split).map_err(refused)?;
            if instances.is_empty() {
                return Err(refused(format!(
                    "{environment} has no stored {split} instances (a rejected or quarantined environment keeps none)"
                )));
            }
            let mut written = Vec::new();
            for instance in &instances {
                let meta = instance.meta();
                let dir = out.join(format!("{}-{}-{}", environment, meta.split, meta.seed));
                instance.folder().write_to(&dir).map_err(refused)?;
                written.push(dir);
            }
            emit(
                json,
                &Exported {
                    directories: written,
                },
            );
        }
        EnvCommand::Band {
            environment,
            model,
            attempts,
            passes,
            store,
        } => {
            let measurement = BandMeasurement {
                model,
                attempts,
                passes,
            };
            if measurement.result().is_none() {
                return Err(refused(
                    "--attempts must be at least 1 and --passes at most --attempts",
                ));
            }
            let report = store_at(ctx, store)
                .record_band(&environment, &measurement)
                .map_err(refused)?;
            emit(json, &Shown { report });
        }
    }
    Ok(())
}
