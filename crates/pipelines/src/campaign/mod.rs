// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that run model learning as one
// resumable, multi-generation campaign with a single promotion gate and
// rollback, for its clients. If your team needs expertise in self-improving
// training loops you can procure our services by sending an email to
// info@swedishembedded.com.

//! The campaign: learning as generations of one resumable run, each ending
//! in one promotion decision, instead of rounds glued together by scripts.
//!
//! A generation is seven stages, in order, each a unit of the campaign's
//! stage ledger (`<state>/campaigns/<name>/ledger.jsonl`,
//! [`splinter_store::stage_ledger`]):
//!
//! 1. **rollout** (`gNNNN/rollout`) - the champion attempts every
//!    train-split practice task it has not mastered (the frontier: a task
//!    whose pass rate over the earlier rollouts reached `retire_at` on
//!    enough graded attempts is retired, [`splinter_eval::campaign::pass_rate`]);
//!    the graded trials come back as the examiner's export ([`Rollouts`]);
//! 2. **ingest** - the export admitted as experience through the one intake
//!    every benchmark trial takes ([`crate::bench::ingest_bench_trials`]),
//!    idempotently;
//! 3. **curate** - one dataset over the experience of every generation so
//!    far, recorded by its id, the digest of its manifest;
//! 4. **train** - a candidate trained from the base on that dataset as a
//!    served brain job under a `resume_key` derived from the stage's
//!    identity ([`crate::train::train_identified`]): asked again after a
//!    crash, brain continues the run or answers with the result it recorded;
//! 5. **evaluate** - the examiner's three signed verdicts on the candidate
//!    against the champion: the development split, the retention suite and
//!    the generation's own fresh sealed slice ([`Examiner`]);
//! 6. **gate** - the one promotion gate on those verdicts' signed aggregates,
//!    verified in this process ([`splinter_eval::promotion`]);
//! 7. **promote** - the champion's move: promoted, or kept with the gate's
//!    typed reasons ([`champion`]).
//!
//! **Durable and idempotent.** A stage's record is committed to the ledger
//! before the next stage starts. Its key is the digest of everything its
//! result depends on (its settings, its draw of the campaign's seed, and the
//! records of the stages before it), never a time or a path, so a stage
//! delivered again under the same key is reused and runs nothing. A run
//! killed anywhere resumes at the first stage without a record; a stage it
//! interrupted runs again, its brain job reused by its `resume_key`.
//! Ctrl-C cancels the stage in progress, a training job on the server
//! included. Budgets (rollouts, GPU-hours, tokens) stop the campaign before
//! a stage that would exceed one.
//!
//! **Rollback.** The champion's history is append-only:
//! [`CampaignController::rollback`] appends a record moving the champion
//! back to an earlier one, with its reason.
//!
//! **What to try next.** The campaign's problem state ([`problems`]): the
//! problems that limit the model with their evidence, the candidate
//! interventions with a belief each, revised by every generation's verdict,
//! and the next single-variable hypothesis as an arm of this campaign.
//!
//! The in-memory rollout machinery over a served policy ([`served`],
//! [`endpoint`], [`brain`], [`factory`]) is what samples and the episode
//! executors are built on; the campaign controller itself practises through
//! [`Rollouts`].

mod account;
pub mod brain;
pub mod champion;
mod controller;
mod curriculum;
pub mod endpoint;
mod environments;
pub mod examiner;
pub mod factory;
pub mod practice;
pub mod problems;
pub mod served;
mod spec;
mod types;
mod work;

pub use account::{
    campaign_rollback, campaign_status, report, CampaignReport, GenerationReport, Spent,
};
pub use champion::{Move, MoveKind};
pub use controller::{
    campaign_dir, unit, CampaignController, Parts, Stopped, CAMPAIGN_UNIT, STAGES,
};
pub use environments::EnvironmentLabels;
pub use spec::{
    Budget, CampaignSpec, CurateSpec, RolloutCurriculum, RolloutSpec, TrainSpec, TransferSpec,
    DEFAULT_RETIRE_AT,
};
pub use types::{
    AllFailHandler, CampaignError, CollectRequest, PolicyRef, Rollout, RolloutCollector,
    RolloutGroup, RolloutTrace, TraceSequence, TrainReport, TrainRequest, Trained, Trainer,
    Validator,
};
pub use work::{
    ExamRequest, Examined, Examiner, Model, Rolled, RolloutRequest, Rollouts, Stopwatch,
    TransferRequest, Trial, WallClock,
};
