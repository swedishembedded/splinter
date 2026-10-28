// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the loop from agent benchmark runs to
// fine-tuning data that is verified and free of evaluation leakage, for its
// clients. If your team needs expertise in learning from benchmark
// trajectories, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Benchmark trials as experiences: the artifacts the Swedish Embedded Bench
//! exports (`bench export`) admitted into the experience store, so that
//! benchmark runs feed the same experience, dataset, train and select
//! pipeline as every other solve.
//!
//! A trial is admitted only after [`artifact`] verified every digest it
//! carries. The experience is the trial's ATIF trajectory under a task whose
//! environment spec records the benchmark task (id, version, digest), the
//! repository snapshot, the verifier and, above all, the split the task is
//! on. The verdict is the benchmark's own grade, an executable one. What
//! belongs to the trial and not to the task - the agent build, the model, the
//! image, the digests of the artifact - is the verdict's evidence.
//!
//! Admission is idempotent: a trial is registered with the digest of the
//! manifest it came from, so ingesting an export again changes nothing, and
//! the same trial name with other content is refused. The split is enforced
//! where datasets are made ([`split::exam_only`]), not here: an `eval` trial
//! is admitted, as examination material, and cannot enter a dataset.

pub mod artifact;
pub mod split;

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::Serialize;
use serde_json::{json, Value};
use splinter_core::annotation::{Annotation, AnnotationBody, Outcome, Producer, Strength};
use splinter_core::experience::{Environment, Experience, ExperienceId, Provenance, Task};
use splinter_core::kinds::BENCH_TASK;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_store::experiences::{ExperienceSet, SetId};

use artifact::{read_index, read_trial, Artifact};
use split::{Split, SplitSource, TrialRecord};

/// The environment kind of a benchmark task.
pub const BENCH_ENVIRONMENT: &str = "bench";

/// The producer of a benchmark verdict.
pub const VERIFIER_PRODUCER: &str = "bench/verifier";

/// What the operator decides about an ingestion.
#[derive(Clone, Copy, Debug, Default)]
pub struct IngestOptions {
    /// The split of every trial whose export names none. An export that
    /// predates splits cannot say; the operator asserts it, and the register
    /// records that the operator did. A trial that does name its split must
    /// agree with it. `None` refuses a trial that names none.
    pub assume_set: Option<Split>,
}

/// A trial, or a whole export, not admitted, and why.
#[derive(Clone, Debug, Serialize)]
pub struct BenchRefused {
    /// The export directory.
    pub export: PathBuf,
    /// The trial; `None` when the export itself is refused.
    pub trial: Option<String>,
    /// Why.
    pub reason: String,
}

/// What an ingestion did.
#[derive(Clone, Debug, Default, Serialize)]
pub struct BenchIngested {
    /// The experience set of the `train` trials, to build datasets from;
    /// `None` when there are none.
    pub train_set: Option<SetId>,
    /// The experience set of the `eval` trials, examination material that no
    /// dataset accepts; `None` when there are none.
    pub eval_set: Option<SetId>,
    /// Trials admitted by this run, by split.
    pub admitted: BTreeMap<String, usize>,
    /// Trials already held with the same content, changed in no way.
    pub already_held: usize,
    /// Trials admitted or held, by the benchmark's verdict.
    pub verdicts: BTreeMap<String, usize>,
    /// The trials and exports refused.
    pub refused: Vec<BenchRefused>,
}

/// Admits the trials of the exports in `exports` (each a directory written
/// by `bench export`).
///
/// # Errors
///
/// [`OrchestratorError`] when the store cannot be read or written. A trial or
/// an export that cannot be admitted is in [`BenchIngested::refused`].
pub fn ingest_bench_trials(
    ctx: &Context,
    exports: &[PathBuf],
    options: IngestOptions,
) -> Result<BenchIngested, OrchestratorError> {
    let mut register = split::trials(ctx)?;
    let mut done = BenchIngested::default();
    let mut members: BTreeMap<Split, Vec<ExperienceId>> = BTreeMap::new();
    for export in exports {
        let rows = match read_index(export) {
            Ok(rows) => rows,
            Err(reason) => {
                done.refused.push(BenchRefused {
                    export: export.clone(),
                    trial: None,
                    reason,
                });
                continue;
            }
        };
        for row in rows {
            let refuse = |trial: Option<&str>, reason: String| BenchRefused {
                export: export.clone(),
                trial: trial.map(str::to_string),
                reason,
            };
            let row = match row {
                Ok(row) => row,
                Err(reason) => {
                    done.refused.push(refuse(None, reason));
                    continue;
                }
            };
            let admitted = match read_trial(export, &row) {
                Ok(artifact) => admit(ctx, &register, &artifact, options)?,
                Err(reason) => Err(reason),
            };
            match admitted {
                Err(reason) => done.refused.push(refuse(Some(&row.trial), reason)),
                Ok(Admission {
                    record,
                    fresh,
                    outcome,
                }) => {
                    if fresh {
                        *done
                            .admitted
                            .entry(record.split.as_str().into())
                            .or_default() += 1;
                        register.push(record.clone());
                    } else {
                        done.already_held += 1;
                    }
                    *done
                        .verdicts
                        .entry(format!("{outcome:?}").to_lowercase())
                        .or_default() += 1;
                    members
                        .entry(record.split)
                        .or_default()
                        .push(record.experience);
                }
            }
        }
    }
    let store = ctx.experiences();
    for (side, ids) in members {
        let set = store.put_set(&ExperienceSet {
            name: format!("bench {} trials", side.as_str()),
            members: ids,
        })?;
        match side {
            Split::Train => done.train_set = Some(set),
            Split::Eval => done.eval_set = Some(set),
        }
    }
    Ok(done)
}

/// A trial admitted, or found already held.
struct Admission {
    record: TrialRecord,
    fresh: bool,
    outcome: Outcome,
}

/// The split of `artifact`: its own, else the operator's.
fn split_of(artifact: &Artifact, options: IngestOptions) -> Result<(Split, SplitSource), String> {
    match (&artifact.manifest.task.set, options.assume_set) {
        (Some(named), assumed) => {
            let named = Split::parse(named)?;
            match assumed {
                Some(other) if other != named => Err(format!(
                    "the artifact says set {}, the operator said {}",
                    named.as_str(),
                    other.as_str()
                )),
                _ => Ok((named, SplitSource::Manifest)),
            }
        }
        (None, Some(assumed)) => Ok((assumed, SplitSource::Operator)),
        (None, None) => Err(
            "the artifact names no set (train or eval), so it is not admissible; say which it \
             is with --assume-set"
                .into(),
        ),
    }
}

/// Admits one verified trial; the outer error is the store's, the inner the
/// reason the trial is refused.
fn admit(
    ctx: &Context,
    register: &[TrialRecord],
    artifact: &Artifact,
    options: IngestOptions,
) -> Result<Result<Admission, String>, OrchestratorError> {
    let (side, source) = match split_of(artifact, options) {
        Ok(found) => found,
        Err(reason) => return Ok(Err(reason)),
    };
    let manifest = &artifact.manifest;
    let outcome = match manifest.grade.verdict.as_str() {
        "pass" => Outcome::Pass,
        "fail" => Outcome::Fail,
        _ => Outcome::Abstain,
    };
    if let Some(held) = register.iter().find(|t| t.trial == manifest.trial) {
        return Ok(
            if held.manifest_sha256 == artifact.row.manifest_sha256 && held.split == side {
                Ok(Admission {
                    record: held.clone(),
                    fresh: false,
                    outcome,
                })
            } else {
                Err(format!(
                    "trial {} is already admitted with other content (manifest {}, set {}); a \
                     trial name is admitted once",
                    held.trial,
                    held.manifest_sha256,
                    held.split.as_str()
                ))
            },
        );
    }
    if let Some(reason) = split::conflict(register, &manifest.task.id, side) {
        return Ok(Err(reason));
    }
    let experience = match experience_of(ctx, artifact, side) {
        Ok(experience) => experience,
        Err(reason) => return Ok(Err(reason)),
    };
    let store = ctx.experiences();
    let id = store.put(&experience)?;
    store.annotate(&Annotation {
        experience: id.clone(),
        producer: Producer {
            name: VERIFIER_PRODUCER.into(),
            version: manifest.verifier.digest.clone(),
        },
        body: AnnotationBody::Verdict {
            outcome,
            strength: Strength::Executable,
            evidence: evidence(artifact, side, source),
        },
    })?;
    let record = TrialRecord {
        trial: manifest.trial.clone(),
        manifest_sha256: artifact.row.manifest_sha256.clone(),
        task: manifest.task.id.clone(),
        split: side,
        split_source: source,
        experience: id,
    };
    split::register(ctx, &record)?;
    Ok(Ok(Admission {
        record,
        fresh: true,
        outcome,
    }))
}

/// The model the trial's agent ran on: the `main` role, else the first.
fn model_of(artifact: &Artifact) -> &str {
    let models = &artifact.manifest.models;
    models
        .get("main")
        .or_else(|| models.values().next())
        .map_or("unknown", |m| m.model.as_str())
}

/// The experience of a verified trial: its trajectory under its task.
fn experience_of(ctx: &Context, artifact: &Artifact, side: Split) -> Result<Experience, String> {
    let manifest = &artifact.manifest;
    let trajectory = artifact.trajectory.clone();
    let instruction = trajectory
        .steps
        .iter()
        .find(|s| s.source == atif::StepOrigin::User)
        .ok_or("the trajectory has no user step: the instruction is unknown")?
        .message
        .as_text()
        .ok_or("the instruction is not plain text")?
        .to_string();
    // What is the task's, not the trial's: another trial of the task, by any
    // agent, has the same task address.
    let spec = json!({
        "task": {
            "id": manifest.task.id,
            "version": manifest.task.version,
            "digest": manifest.task.digest,
            "dimension": manifest.task.dimension,
        },
        "set": side.as_str(),
        "snapshot": manifest.snapshot,
        "verifier": {"digest": manifest.verifier.digest, "scoring": manifest.verifier.scoring},
    });
    let task = Task::new(
        BENCH_TASK,
        Vec::new(),
        Environment::new(BENCH_ENVIRONMENT, spec),
        instruction,
        Vec::new(),
    )
    .map_err(|e| e.to_string())?;
    let agent = &manifest.agent;
    // The time the agent started, so the same trial is the same experience
    // whenever it is ingested; the clock only when the trajectory has none.
    let created = trajectory
        .extra
        .as_ref()
        .and_then(|extra| extra.pointer("/sven/created_at"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let mut provenance = Provenance::new(
        format!(
            "bench/{}@{}/{}",
            agent.name,
            agent.version,
            model_of(artifact)
        ),
        ctx.clock(),
    );
    if let Some(created) = created {
        provenance.created_at = created;
    }
    Experience::new(task, trajectory, None, provenance).map_err(|e| e.to_string())
}

/// What belongs to the trial and not to its task, for whoever audits the
/// verdict.
fn evidence(artifact: &Artifact, side: Split, source: SplitSource) -> Value {
    let manifest = &artifact.manifest;
    json!({
        "trial": manifest.trial,
        "manifest_sha256": artifact.row.manifest_sha256,
        "files": manifest.files,
        "task": {
            "id": manifest.task.id,
            "version": manifest.task.version,
            "digest": manifest.task.digest,
        },
        "set": side.as_str(),
        "set_source": source,
        "snapshot": manifest.snapshot,
        "verifier": {"digest": manifest.verifier.digest, "scoring": manifest.verifier.scoring},
        "reward": manifest.grade.reward,
        "metrics": manifest.grade.metrics,
        "agent": {
            "name": manifest.agent.name,
            "version": manifest.agent.version,
            "sven_build_sha256": manifest.agent.sven_build_sha256,
        },
        "model": model_of(artifact),
        "image": manifest.image,
        "tokens": manifest.tokens,
        "wall_seconds": manifest.wall_seconds,
    })
}

/// Refuses a projection that holds any experience that is examination
/// material (see [`split::exam_only_id`]). The last gate before a dataset is
/// stored, for callers that project experience their own way.
pub fn admit_projection(
    ctx: &Context,
    experiences: impl IntoIterator<Item = ExperienceId>,
) -> Result<(), OrchestratorError> {
    let register = split::trials(ctx)?;
    if register.is_empty() {
        return Ok(());
    }
    for id in experiences {
        if let Some(reason) = split::exam_only_id(&register, &id) {
            return Err(OrchestratorError::Refused(format!("leakage: {reason}")));
        }
    }
    Ok(())
}
