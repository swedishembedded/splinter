// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that let a training campaign draw
// only on environments whose verifiers were attacked and whose difficulty was
// measured, for its clients. If your team needs expertise in admitting
// training environments to reinforcement-learning campaigns you can procure
// our services by sending an email to info@swedishembedded.com.

//! The environment factory as a campaign's task source.
//!
//! An environment enters a campaign only through [`select`]: it must be
//! `Admitted` (every admission gate passed, no attack beat its verifier) and
//! its recorded base pass rate must lie inside the learnable band it was held
//! to, so a pool of tasks the policy always solves or never solves cannot be
//! chosen by mistake. Task folders are never kept on disk: an instance is
//! staged in a scratch directory to be graded and removed with it.
//!
//! [`FactoryFamily`] is the in-process path over such environments: the
//! instruction is the prompt and the completion is the contents of the one
//! answer file the environment's solver writes, graded by the environment's
//! own verifier on the host. Environments whose solver needs a shell or a
//! database are refused here with the reason; they are run by sven's rollout
//! executor.

use std::collections::BTreeMap;
use std::sync::Mutex;

use serde_json::{json, Value};
use splinter_core::verifiability::Split;
use splinter_knowledge::factory::admission::{AdmissionReport, Status};
use splinter_knowledge::factory::artifact::{Instance, SolverTools};
use splinter_knowledge::factory::host::{Host, Verdict};
use splinter_knowledge::factory::plugins::{PROGRAM_OUTPUT_LUA, PROGRAM_OUTPUT_PYTHON};
use splinter_knowledge::factory::store::EnvironmentStore;
use splinter_store::campaign::RewardSource;

use super::brain::{Graded, TaskFamily};
use super::environments::EnvironmentLabels;
use super::types::CampaignError;

/// The file the solver of a plugin writes its answer to, for the plugins the
/// in-process path can run.
fn answer_file(plugin: &str) -> Option<&'static str> {
    matches!(plugin, PROGRAM_OUTPUT_LUA | PROGRAM_OUTPUT_PYTHON).then_some("answer.txt")
}

/// The system message of an in-process factory task.
const SYSTEM: &str =
    "Reply with exactly the contents of the answer file the task asks for, and nothing else.";

/// An environment chosen for a campaign.
#[derive(Clone, Debug, PartialEq)]
pub struct Selected {
    /// Its id.
    pub environment: String,
    /// How its rewards are established, and by which verifier.
    pub source: RewardSource,
    /// The base pass rate it was admitted at.
    pub base_rate: f64,
}

/// Why an environment was not chosen.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    /// The environment.
    pub environment: String,
    /// Why.
    pub reason: String,
}

/// The environments a campaign may draw on, and those it was refused.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Selection {
    /// Chosen, by id.
    pub chosen: Vec<Selected>,
    /// Refused, each with the reason.
    pub refused: Vec<Refusal>,
}

fn judge(report: &AdmissionReport) -> Result<Selected, String> {
    if report.status != Status::Admitted {
        return Err(format!(
            "its status is {:?}, only Admitted environments enter a campaign{}",
            report.status,
            report
                .reasons
                .first()
                .map_or(String::new(), |r| format!(" ({r})"))
        ));
    }
    let Some(band) = &report.band else {
        return Err("no base pass rate is recorded for it".into());
    };
    let policy = report.band_policy;
    if band.attempts < policy.min_attempts {
        return Err(format!(
            "its base pass rate rests on {} attempts, fewer than the {} the band policy needs",
            band.attempts, policy.min_attempts
        ));
    }
    if band.rate < policy.low || band.rate > policy.high {
        return Err(format!(
            "its base pass rate {:.3} is outside the learnable band {}..={}",
            band.rate, policy.low, policy.high
        ));
    }
    Ok(Selected {
        environment: report.environment.clone(),
        source: RewardSource {
            environment: report.environment.clone(),
            class: report.verifier_class,
            verifier: format!("{}@{}", report.plugin, report.plugin_version),
        },
        base_rate: band.rate,
    })
}

/// Chooses from the environments of `store` those named in `ids` (all of
/// them when `ids` is empty).
///
/// # Errors
/// The store cannot be read, or an id names no environment.
pub fn select(store: &EnvironmentStore, ids: &[String]) -> Result<Selection, CampaignError> {
    let reports = store
        .list()
        .map_err(|e| CampaignError::Config(format!("reading the environment store: {e}")))?;
    if let Some(missing) = ids
        .iter()
        .find(|id| !reports.iter().any(|r| &r.environment == *id))
    {
        return Err(CampaignError::Config(format!(
            "the environment store holds no environment {missing:?}"
        )));
    }
    let mut selection = Selection::default();
    for report in reports
        .iter()
        .filter(|r| ids.is_empty() || ids.contains(&r.environment))
    {
        match judge(report) {
            Ok(selected) => selection.chosen.push(selected),
            Err(reason) => selection.refused.push(Refusal {
                environment: report.environment.clone(),
                reason,
            }),
        }
    }
    Ok(selection)
}

/// One instance ready to be put to the policy.
struct Prepared {
    environment: String,
    instance: Instance,
    answer_file: &'static str,
}

/// The tasks of chosen environments, put to the policy in process.
pub struct FactoryFamily {
    store: EnvironmentStore,
    selected: Vec<Selected>,
    tasks: BTreeMap<String, Prepared>,
    pool: Vec<String>,
    dev: Vec<String>,
    host: Host,
    quarantined: Mutex<Vec<String>>,
}

impl FactoryFamily {
    /// The family over `selection.chosen`: every archived train and dev
    /// instance of each environment, as tasks `<environment>/<split>-<seed>`.
    ///
    /// # Errors
    /// An instance cannot be read, or an environment's solver needs more than
    /// an answer file ([`SolverTools::Shell`] or a plugin the in-process path
    /// does not know), which only the agent executor can run.
    pub fn open(
        store: EnvironmentStore,
        selection: &Selection,
        host: Host,
    ) -> Result<Self, CampaignError> {
        let mut tasks = BTreeMap::new();
        let (mut pool, mut dev) = (Vec::new(), Vec::new());
        for selected in &selection.chosen {
            for split in [Split::Train, Split::Dev] {
                let instances = store.instances(&selected.environment, split).map_err(|e| {
                    CampaignError::Config(format!("reading {}: {e}", selected.environment))
                })?;
                for instance in instances {
                    let meta = instance.meta();
                    let Some(file) = answer_file(&meta.plugin)
                        .filter(|_| meta.solver_tools == SolverTools::Files)
                    else {
                        return Err(CampaignError::Config(format!(
                            "{} ({}) needs a solver with tools: run it through the agent executor, not in process",
                            selected.environment, meta.plugin
                        )));
                    };
                    let id = format!("{}/{}-{}", selected.environment, split, meta.seed);
                    match split {
                        Split::Train => pool.push(id.clone()),
                        Split::Dev => dev.push(id.clone()),
                    }
                    tasks.insert(
                        id,
                        Prepared {
                            environment: selected.environment.clone(),
                            instance,
                            answer_file: file,
                        },
                    );
                }
            }
        }
        Ok(Self {
            store,
            selected: selection.chosen.clone(),
            tasks,
            pool,
            dev,
            host,
            quarantined: Mutex::new(Vec::new()),
        })
    }

    fn prepared(&self, task: &str) -> Result<&Prepared, CampaignError> {
        self.tasks.get(task).ok_or_else(|| {
            CampaignError::Config(format!("{task:?} is not a task of the factory family"))
        })
    }
}

impl TaskFamily for FactoryFamily {
    fn describe(&self) -> Value {
        json!({
            "family": "environment-factory",
            "store": self.store.root().display().to_string(),
            "environments": self.selected.iter().map(|s| json!({
                "environment": s.environment, "verifier": s.source.verifier, "base_rate": s.base_rate,
            })).collect::<Vec<_>>(),
        })
    }

    fn pool(&self) -> Vec<String> {
        self.pool.clone()
    }

    fn dev(&self) -> Vec<String> {
        self.dev.clone()
    }

    fn retention(&self) -> Vec<String> {
        Vec::new()
    }

    fn prompt(&self, task: &str) -> Result<Vec<(String, String)>, CampaignError> {
        let prepared = self.prepared(task)?;
        Ok(vec![
            ("system".into(), SYSTEM.into()),
            ("user".into(), prepared.instance.instruction().to_string()),
        ])
    }

    fn grade(&self, task: &str, completion: &str) -> Result<Graded, CampaignError> {
        let prepared = self.prepared(task)?;
        let infrastructure = |what: String| CampaignError::Collector(format!("{task}: {what}"));
        // The folder is staged for this attempt and removed with the session.
        let session = self
            .host
            .open(&prepared.instance)
            .map_err(|e| infrastructure(e.to_string()))?;
        session
            .write(prepared.answer_file, completion.as_bytes())
            .map_err(|e| infrastructure(e.to_string()))?;
        match session
            .verify()
            .map_err(|e| infrastructure(e.to_string()))?
        {
            Verdict::Graded(vector) => {
                let reward = vector.get("reward").copied().unwrap_or(0.0);
                Ok(Graded {
                    reward,
                    passed: reward >= 1.0,
                    feedback: (reward < 1.0)
                        .then(|| "That is not the output. Try again.".to_string()),
                    privileged: None,
                })
            }
            Verdict::NoReward { exit, tail } => Err(infrastructure(format!(
                "the verifier wrote no reward ({exit}): {tail}"
            ))),
            Verdict::Malformed(why) => Err(infrastructure(format!(
                "the verifier's reward is malformed: {why}"
            ))),
        }
    }
}

impl EnvironmentLabels for FactoryFamily {
    fn environment_of(&self, task: &str) -> Option<String> {
        self.tasks.get(task).map(|p| p.environment.clone())
    }

    fn sources(&self) -> Vec<RewardSource> {
        self.selected.iter().map(|s| s.source.clone()).collect()
    }

    fn quarantine(&self, environment: &str, reason: &str) -> Result<(), CampaignError> {
        let mut report = self
            .store
            .report(environment)
            .map_err(|e| CampaignError::Config(format!("reading {environment}: {e}")))?;
        report.status = Status::Quarantined;
        report
            .reasons
            .push(format!("campaign divergence alarm: {reason}"));
        self.store.save(&report, &[]).map_err(|e| {
            CampaignError::Config(format!("recording the quarantine of {environment}: {e}"))
        })?;
        if let Ok(mut list) = self.quarantined.lock() {
            list.push(environment.to_string());
        }
        Ok(())
    }
}
