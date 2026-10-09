// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning from the sessions a person holds
// with an agent, for its clients. If your team needs expertise in continual
// learning from user feedback, you can procure our services by sending an
// email to info@swedishembedded.com.

//! `absorb`: the sessions a person held with an agent become the next
//! release, as one recorded run of stages that are also commands of their
//! own.
//!
//! 1. **policy** - the release `--policy` names is resolved once;
//! 2. **intake** - the sessions are validated, stripped of secrets and
//!    stored once ([`crate::sessions`]);
//! 3. **extract** - a model proposes the claims each session teaches
//!    ([`crate::claims`]); the claim set of the same sessions and model is
//!    reused, not asked again;
//! 4. **gate** - code rules on every proposal and appends the rulings to the
//!    ledger; `--dry-run` ends here, with nothing trained;
//! 5. **kits** - every live claim without a kit gets one ([`kit`]): its
//!    question and paraphrases taught by a teacher shown the claim, the
//!    hindsight dialogue, its other forms, and the stopping paraphrases no
//!    record contains;
//! 6. **dataset** - the kits of the live claims, all on the training side,
//!    with the stopping paraphrases held out ([`dataset`]); a sealed probe
//!    refuses the record that contains it ([`sealed`]);
//! 7. **rehearse** - the base's own answers to general tasks
//!    ([`crate::rehearsal`]);
//! 8. **train** - the adapter is trained again from the base on all of it,
//!    for at most [`DEFAULT_EPOCHS`] passes: a claim superseded or forgotten
//!    is simply not in the set. `--continue-from-release` instead continues
//!    the release, on the claims not yet absorbed with a replay of the earlier
//!    releases' records: an ablation, not the default;
//! 9. **release** - the gate ([`gate`]) rules on counts of claims answered
//!    on their stopping paraphrases; a candidate it refuses stays on record
//!    with the numbers and the current release stays in use. `--no-release`
//!    stops at the candidate;
//! 10. **ledger** - each claim the release answers records the release that
//!     first absorbed it.
//!
//! A failed run is resumed by running it again: intake, the claim set, the
//! kits and the datasets are kept by content, and a candidate trained on the
//! dataset a failed run built is taken as it is.

pub mod dataset;
pub mod gate;
pub mod kit;
pub mod sealed;
mod stages;

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::Serialize;
use splinter_core::claim::ClaimId;
use splinter_core::model_ref::ModelRef;
use splinter_core::release::ReleaseId;
use splinter_core::role::{Role, RoleOverrides};
use splinter_eval::claim_gate::ClaimGate;
use splinter_eval::gate::GateConfig;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_orchestrator::runs::{record, Recorded};

use crate::claims::{ClaimsExtracted, ClaimsGated, DEFAULT_EXTRACTION_PASSES};
use crate::learn::PolicyUsed;
use crate::rehearsal::{Rehearsed, DEFAULT_REHEARSAL_SHARE};
use crate::sessions::{SessionsIntake, DEFAULT_MAX_SESSION_BYTES};
use crate::train::{Candidate, Trainer, Tuning, DEFAULT_LORA_RANK, DEFAULT_REPLAY_FRACTION};
use dataset::Assembled;
use kit::{KitsBuilt, PARAPHRASES_WRITTEN};

/// The most passes over a night's records the training makes.
pub const DEFAULT_EPOCHS: u32 = 6;

/// The stages, in order, as runs and reports name them.
pub const STAGES: [&str; 10] = [
    "policy", "intake", "extract", "gate", "kits", "dataset", "rehearse", "train", "release",
    "ledger",
];

/// One `absorb`.
#[derive(Clone, Debug, Serialize)]
pub struct AbsorbRequest {
    /// ATIF files, or directories of them.
    pub sessions: Vec<PathBuf>,
    /// The largest session file taken, in bytes.
    pub max_session_bytes: u64,
    /// The policy: `policy:<alias>`; a release of it is made under the alias.
    pub policy: ModelRef,
    /// The models named for roles: the generator proposes claims,
    /// paraphrases and forms, the teacher answers with the claim in front of
    /// it, the judge decides what the terms cannot.
    pub roles: RoleOverrides,
    /// Differently worded questions written per claim, a fifth of them kept
    /// out of training as the claim's stopping paraphrases.
    pub paraphrases: usize,
    /// The share of the training draws that are the base's own answers.
    pub rehearsal_share: f64,
    /// The most passes over the night's records.
    pub epochs: u32,
    /// The step budget, in place of `epochs`.
    pub steps: Option<u32>,
    /// LoRA rank of the adapter.
    pub rank: u32,
    /// The base's precision, the learning rate and how the run is watched.
    pub tuning: Tuning,
    /// Continue the release on the claims not yet absorbed instead of
    /// training again from the base on all of them.
    pub continue_from_release: bool,
    /// The fraction of each earlier release's records replayed when
    /// continuing.
    pub replay_fraction: f64,
    /// Extraction passes over each session; only what every pass proposed
    /// is ruled on.
    pub extraction_passes: u32,
    /// Whether the judge, when there is one, must find that the person's
    /// cited words assert each claim's statement.
    pub entail: bool,
    /// Files of sealed probes no training record may contain.
    pub sealed_probes: Vec<PathBuf>,
    /// Stop after the claims are extracted and ruled on; train nothing.
    pub dry_run: bool,
    /// Stop at the candidate: do not run the gate.
    pub no_release: bool,
    /// The serve check's bounds.
    pub gate: GateConfig,
}

impl Default for AbsorbRequest {
    fn default() -> Self {
        Self {
            sessions: Vec::new(),
            max_session_bytes: DEFAULT_MAX_SESSION_BYTES,
            policy: ModelRef::policy_default(),
            roles: RoleOverrides::new(),
            paraphrases: PARAPHRASES_WRITTEN,
            rehearsal_share: DEFAULT_REHEARSAL_SHARE,
            epochs: DEFAULT_EPOCHS,
            steps: None,
            rank: DEFAULT_LORA_RANK,
            tuning: Tuning::default(),
            continue_from_release: false,
            extraction_passes: DEFAULT_EXTRACTION_PASSES,
            entail: true,
            replay_fraction: DEFAULT_REPLAY_FRACTION,
            sealed_probes: Vec::new(),
            dry_run: false,
            no_release: false,
            gate: GateConfig::default(),
        }
    }
}

/// What an `absorb` reports, stage by stage.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Absorbed {
    /// The policy the run worked from.
    pub policy: PolicyUsed,
    /// The model each role was given.
    pub roles: BTreeMap<Role, String>,
    /// The intake stage.
    pub intake: Option<SessionsIntake>,
    /// The extract stage.
    pub extract: Option<ClaimsExtracted>,
    /// The gate stage: the rulings.
    pub claims: Option<ClaimsGated>,
    /// Live claims after the rulings, and the ones no release has absorbed.
    pub live: usize,
    /// The claims no release has absorbed yet.
    pub pending: Vec<ClaimId>,
    /// The kits stage.
    pub kits: Option<KitsBuilt>,
    /// The dataset stage.
    pub dataset: Option<Assembled>,
    /// The rehearse stage.
    pub rehearsal: Option<Rehearsed>,
    /// The candidate trained.
    pub candidate: Option<Candidate>,
    /// The gate's decision with every count.
    pub gate: Option<ClaimGate>,
    /// The release written, when the gate passed.
    pub release: Option<ReleaseId>,
    /// The claims this release first absorbed.
    pub absorbed: Vec<ClaimId>,
    /// Why the run stopped before its last stage, if it did.
    pub stopped: Option<String>,
    /// Whether it was a dry run.
    pub dry_run: bool,
    /// Whether the gate was to run.
    pub release_asked: bool,
}

impl Absorbed {
    /// Whether the run did what was asked: no file refused, no session
    /// without claims, and, unless it was a dry run, a candidate that the
    /// gate released when it was asked to.
    #[must_use]
    pub fn finished(&self) -> bool {
        let intake_clean = self.intake.as_ref().is_none_or(|i| i.refused.is_empty());
        let extracted = self.extract.as_ref().is_none_or(|e| e.failed.is_empty());
        intake_clean
            && extracted
            && if self.dry_run {
                self.stopped.is_some() && self.claims.is_some()
            } else {
                self.candidate.is_some() && (!self.release_asked || self.release.is_some())
            }
    }
}

/// Runs `request`, training with `trainer`.
pub fn absorb(
    ctx: &Context,
    request: &AbsorbRequest,
    trainer: &dyn Trainer,
) -> Result<Recorded<Absorbed>, OrchestratorError> {
    stages::validate(request)?;
    record(ctx, "absorb", request, |run| {
        stages::run(ctx, request, trainer, run)
    })
}
