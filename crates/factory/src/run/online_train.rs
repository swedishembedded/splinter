// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements online reinforcement learning that trains a
// served policy on its own rollouts and rolls it back on evidence, for its
// clients. If your team needs expertise in safe continuous training, you can
// procure our services by sending an email to info@swedishembedded.com.

//! `train.online`'s training: the learner and the controller, one epoch
//! after another.
//!
//! An epoch is one learner run: the first continues the stage's `start` (or
//! the base); a learner that restarts under it is started again under the
//! same key and resumed (up to [`RESTARTS`] times). With a watchdog, a
//! rollback ends the epoch: its audit is appended to the group log, the next
//! epoch's learner starts from the anchor's adapter (a frozen copy, by
//! digest) under a key of its own (fresh optimiser state) with its learning
//! rate cooled down, and its controller rolls back every group of the
//! earlier epoch the learner had not committed. The epoch, the run's
//! rollbacks and its evaluations are kept in `epochs.json` beside the
//! stage's work, written before the next epoch starts, so a stopped
//! controller resumes in the epoch it left; past `max_rollbacks` the stage
//! halts with the audits.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use splinter_core::clock::unix_now;

use super::online::OnlineParams;
use super::online_eval::{Served, ServedEvaluator};
use super::{io, stage_error, write_json, Factory, RunError, LEASE_SECONDS};
use crate::online::driver::{run_watched, Ended, OnlineError, Report, RunLedger, Seams};
use crate::online::frontier::Selector;
use crate::online::group_log::{GroupLog, LoggedLedger};
use crate::online::health::VersionSeries;
use crate::online::seams::{
    GroupLedger, LearnerTelemetry, OnlineRollouts, RolloutsRequest, RunClock, StreamLearner,
    SystemRunClock,
};
use crate::online::watchdog::monitor::MonitorPool;
use crate::online::watchdog::{Version, Watchdog};
use crate::plan::PlannedStage;
use crate::recipe::ModelRef;
use crate::training::{OnlineTrain, Trainer};

/// How many times a restarted job is started again before the stage fails.
const RESTARTS: u32 = 8;

/// The file the epochs of a stage are kept in, in its work directory.
pub const EPOCHS_FILE: &str = "epochs.json";

/// Where a stage's epochs stand.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
struct Epochs {
    /// The epoch the stage trains in.
    epoch: u64,
    /// Rollbacks so far.
    rollbacks: u32,
    /// Comparisons started so far (each spends its share of the error
    /// rates).
    comparisons: u32,
    /// The version the epoch started from, after a rollback.
    anchor: Option<Version>,
}

/// What the learner and the controller came to.
pub(super) struct Trained {
    pub report: Report,
    pub telemetry: LearnerTelemetry,
    pub learner: Box<dyn StreamLearner>,
    pub restarts: u32,
    /// The rollouts each policy version decoded, as the group log saw them.
    pub series: VersionSeries,
    /// What the task selector knew at the end.
    pub selector: Value,
    /// What the watchdog saw, and every rollback's audit.
    pub watchdog: Value,
}

/// The held-out tasks a watchdog evaluates on: each a tasks product's
/// folder.
pub(super) struct HeldOut {
    pub monitor: PathBuf,
    pub confirm: PathBuf,
}

/// One `train.online` stage, resolved.
pub(super) struct Stage<'a> {
    pub factory: &'a Factory<'a>,
    pub stage: &'a PlannedStage,
    pub params: &'a OnlineParams,
    pub base: &'a ModelRef,
    pub trainer: &'a dyn Trainer,
    pub rollouts: &'a dyn OnlineRollouts,
    pub tasks: &'a Path,
    /// The namespace of its groups.
    pub tag: &'a str,
    /// The id its first epoch publishes under.
    pub publish: &'a str,
    pub scratch: &'a Path,
    pub groups_log: &'a Path,
    pub start: Option<&'a Path>,
    pub held_out: Option<HeldOut>,
}

/// The ids of the tasks the folder `dir` holds, as the rollouts name them.
fn ids(
    rollouts: &dyn OnlineRollouts,
    dir: &Path,
    request: &RolloutsRequest<'_>,
) -> Result<Vec<String>, String> {
    let mut found = Vec::new();
    rollouts.run(
        &RolloutsRequest {
            tasks: dir,
            ..request.clone()
        },
        &mut |_, ids| found = ids.to_vec(),
    )?;
    Ok(found)
}

impl Stage<'_> {
    fn fail(&self, e: impl ToString) -> RunError {
        stage_error(&self.stage.key, e.to_string())
    }

    fn work(&self) -> PathBuf {
        self.factory.work(&self.stage.key)
    }

    fn epochs(&self) -> Result<Epochs, RunError> {
        let path = self.work().join(EPOCHS_FILE);
        match std::fs::read(&path) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map_err(|e| self.fail(format!("{}: {e}", path.display()))),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Epochs::default()),
            Err(e) => Err(io(&path)(e)),
        }
    }

    /// The learner's settings in `epoch`: the recipe's, the learning rate
    /// cooled down after every rollback.
    fn learner_params(&self, epochs: &Epochs) -> Map<String, Value> {
        let mut params = self.params.learner_params();
        if let (Some(watchdog), Some(lr)) = (
            &self.params.watchdog,
            params.get("lr").and_then(Value::as_f64),
        ) {
            let cooled = lr
                * watchdog
                    .cooldown
                    .powi(i32::try_from(epochs.rollbacks).unwrap_or(i32::MAX));
            params.insert("lr".into(), json!(cooled));
        }
        params
    }

    /// The groups of `epoch` a rollback leaves untrained.
    fn untrained(&self, epoch: u64) -> Result<usize, RunError> {
        let prefix = format!("{}:", self.tag);
        Ok(self
            .factory
            .state
            .group_entries()?
            .iter()
            .filter(|e| e.group.key.as_str().starts_with(&prefix))
            .filter(|e| e.group.epoch == epoch && !e.group.state.is_terminal())
            .count())
    }
}

/// Trains the stage to its end: every epoch, every restart.
pub(super) fn train(stage: &Stage<'_>) -> Result<Trained, RunError> {
    let params = stage.params;
    // The held-out tasks' ids, read once, and what evaluations share.
    let served_eval = match (&params.watchdog, &stage.held_out) {
        (Some(_), Some(held)) => {
            let request = RolloutsRequest {
                tasks: stage.tasks,
                policy: stage.publish,
                temperature: params.temperature,
                top_logprobs: None,
                max_output_tokens: params.max_output_tokens,
                seed: params.seed,
                scratch: stage.scratch,
            };
            let monitor =
                ids(stage.rollouts, &held.monitor, &request).map_err(|e| stage.fail(e))?;
            let confirm =
                ids(stage.rollouts, &held.confirm, &request).map_err(|e| stage.fail(e))?;
            Some(Served {
                rollouts: stage.rollouts,
                task_sets: vec![
                    (held.monitor.clone(), monitor),
                    (held.confirm.clone(), confirm),
                ],
                work: stage.work().join("watch"),
                namespace: stage.tag.to_string(),
                temperature: params.temperature,
                max_output_tokens: params.max_output_tokens,
                seed: params.seed,
            })
        }
        (Some(_), None) => return Err(stage.fail("a watchdog needs its monitor and confirm tasks")),
        _ => None,
    };
    std::thread::scope(|scope| train_in(stage, served_eval.as_ref(), scope))
}

fn train_in<'s, 'e>(
    stage: &'e Stage<'e>,
    served_eval: Option<&'e Served<'e>>,
    scope: &'s std::thread::Scope<'s, 'e>,
) -> Result<Trained, RunError> {
    let params = stage.params;
    let mut epochs = stage.epochs()?;
    let mut restarts = 0;
    let mut audits: Vec<Value> = Vec::new();
    loop {
        let epoch = epochs.epoch;
        let anchor_adapter = match &epochs.anchor {
            Some(anchor) if epoch > 0 => {
                let path = stage.work().join(format!("anchor-e{epoch}.safetensors"));
                if !path.is_file() {
                    let artifact = anchor.artifact.as_deref().ok_or_else(|| {
                        stage.fail(format!(
                            "the anchor, version {}, is in no artifact store: the learner kept no versions",
                            anchor.version
                        ))
                    })?;
                    stage
                        .trainer
                        .published_adapter(artifact, &path)
                        .map_err(|e| stage.fail(format!("the anchor {artifact}: {e}")))?;
                }
                Some(path)
            }
            _ => None,
        };
        // Each epoch is a run of its own: fresh optimiser state, its own id.
        let (resume_key, publish) = if epoch == 0 {
            (stage.tag.to_string(), stage.publish.to_string())
        } else {
            (
                format!("{}-e{epoch}", stage.tag),
                stage
                    .publish
                    .replace(":latest", &format!(":e{epoch}:latest")),
            )
        };
        let learner_params = stage.learner_params(&epochs);
        let mut learner = stage
            .trainer
            .stream(&OnlineTrain {
                base: stage.base,
                params: &learner_params,
                resume_key: &resume_key,
                checkpoint_every: params.checkpoint_every,
                publish_adapter: &publish,
                publish_every: params.publish_every,
                step_tokens: params.step_tokens,
                queue_tokens: params.queue_tokens,
                max_staleness: params.max_staleness,
                max_steps: params.max_updates,
                top_k: params
                    .top_logprobs
                    .filter(|_| params.estimator == "score-centred"),
                retain_versions: params.retained_versions(),
                start: anchor_adapter.as_deref().or(stage.start),
                job_place: &stage.work().join("stream-job.json"),
            })
            .map_err(|e| stage.fail(e))?;
        let served = learner.served_as();
        let mut outcome: Option<Result<Report, OnlineError>> = None;
        let mut series = VersionSeries::default();
        let mut selected: Option<Value> = None;
        let mut watched = Value::Null;
        let mut watchdog = match (&params.watchdog, served_eval) {
            (Some(settings), Some(served)) => {
                let mut pool = MonitorPool::new(
                    served.task_sets[0].1.clone(),
                    settings.max_uses,
                    params.seed,
                );
                pool.epoch(epoch);
                Some(Watchdog::new(
                    settings.clone(),
                    Box::new(ServedEvaluator::new(scope, served, epoch)),
                    pool,
                    served.task_sets[1].1.clone(),
                    None,
                    epochs.comparisons,
                ))
            }
            _ => None,
        };
        stage
            .rollouts
            .run(
                &RolloutsRequest {
                    tasks: stage.tasks,
                    policy: &served,
                    temperature: params.temperature,
                    top_logprobs: params.top_logprobs,
                    max_output_tokens: params.max_output_tokens,
                    seed: params.seed,
                    scratch: stage.scratch,
                },
                &mut |executor, task_ids| {
                    let mut selector = match Selector::new(
                        params.selector,
                        task_ids.to_vec(),
                        params.groups,
                        params.seed,
                    ) {
                        Ok(selector) => selector,
                        Err(e) => {
                            outcome = Some(Err(OnlineError::Control(
                                crate::online::control::ControlError(e.to_string()),
                            )));
                            return;
                        }
                    };
                    let mut ledger = RunLedger::new(
                        &stage.factory.state,
                        stage.factory.fence.clone(),
                        LEASE_SECONDS,
                        unix_now(),
                    );
                    let log = match open_group_log(stage.groups_log, &mut ledger, stage.tag) {
                        Ok(log) => log,
                        Err(e) => {
                            outcome = Some(Err(OnlineError::Ledger(e)));
                            return;
                        }
                    };
                    let mut ledger = LoggedLedger::new(&mut ledger, log);
                    let cancel = || stage.factory.cancelled().is_err();
                    let mut settings = params.settings(stage.tag);
                    settings.epoch = epoch;
                    let result = run_watched(
                        settings,
                        Seams {
                            executor,
                            learner: learner.as_mut(),
                            selector: &mut selector,
                            ledger: &mut ledger,
                            clock: &SystemRunClock,
                            cancel: &cancel,
                        },
                        watchdog.as_mut(),
                    );
                    series = ledger.log().series().clone();
                    selected = Some(selector.report());
                    if let Some(watchdog) = &watchdog {
                        watched = watchdog.report();
                    }
                    outcome = Some(result);
                },
            )
            .map_err(|e| stage.fail(e))?;
        if let Some(watchdog) = &watchdog {
            epochs.comparisons = watchdog.comparisons();
        }
        match outcome {
            Some(Ok(Report {
                ended: Ended::Cancelled,
                ..
            })) => return Err(RunError::Cancelled),
            Some(Ok(Report {
                ended: Ended::RolledBack(audit),
                ..
            })) => {
                epochs.rollbacks += 1;
                let record = json!({
                    "audit": "rollback",
                    "epoch": epoch,
                    "rollback": epochs.rollbacks,
                    "trigger": audit.trigger,
                    "evidence": audit.evidence,
                    "candidate": audit.candidate,
                    "anchor": audit.anchor,
                    "alarms": audit.alarms,
                    "groups_discarded": stage.untrained(epoch)?,
                    "at": unix_now(),
                });
                note(stage, &record)?;
                audits.push(record);
                let max = params.watchdog.as_ref().map_or(0, |w| w.max_rollbacks);
                if epochs.rollbacks > max {
                    return Err(stage.fail(format!(
                        "halted after {} rollbacks (at most {max}); the audits are in the group log",
                        epochs.rollbacks
                    )));
                }
                epochs.epoch += 1;
                epochs.anchor = Some(audit.anchor.clone());
                write_json(&stage.work().join(EPOCHS_FILE), &epochs)?;
                eprintln!(
                    "{}: rolled back to version {} ({}); epoch {} starts from it",
                    stage.stage.key, audit.anchor.version, audit.anchor.digest, epochs.epoch
                );
            }
            Some(Ok(report)) => {
                let mut watchdog = watched;
                if !watchdog.is_null() {
                    watchdog["audits"] = json!(audits);
                    watchdog["epochs"] = json!(epochs);
                }
                return Ok(Trained {
                    report,
                    telemetry: learner.telemetry(),
                    learner,
                    restarts,
                    series,
                    selector: selected.unwrap_or(Value::Null),
                    watchdog,
                });
            }
            Some(Err(OnlineError::Learner(e))) if e.restarted && restarts < RESTARTS => {
                restarts += 1;
                eprintln!(
                    "{}: the learner restarted ({}); resuming against it",
                    stage.stage.key, e.message
                );
            }
            Some(Err(e)) => return Err(stage.fail(e)),
            None => return Err(stage.fail("the rollouts never ran")),
        }
    }
}

/// Appends a record of the run's own to its group log.
fn note(stage: &Stage<'_>, record: &Value) -> Result<(), RunError> {
    let mut ledger = RunLedger::new(
        &stage.factory.state,
        stage.factory.fence.clone(),
        LEASE_SECONDS,
        unix_now(),
    );
    let mut log =
        open_group_log(stage.groups_log, &mut ledger, stage.tag).map_err(|e| stage.fail(e))?;
    log.note(record).map_err(|e| stage.fail(e))
}

/// The live log of the stage's groups at `path`, brought up to date with
/// the ledger's groups of the namespace `namespace`.
pub(super) fn open_group_log(
    path: &Path,
    ledger: &mut RunLedger<'_>,
    namespace: &str,
) -> Result<GroupLog, crate::online::seams::LedgerError> {
    let prefix = format!("{namespace}:");
    let entries: Vec<_> = ledger
        .load()?
        .into_iter()
        .filter(|e| e.group.key.as_str().starts_with(&prefix))
        .collect();
    GroupLog::open(path, &entries, SystemRunClock.now_ms())
}
