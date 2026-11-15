// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that run model learning as one
// continuous, resumable campaign, for its clients. If your team needs
// expertise in crash-safe state for long training campaigns you can procure
// our services by sending an email to info@swedishembedded.com.

//! What a campaign remembers between arms: the value saved to the store after
//! every step that cost something, and the only thing a restarted controller
//! reads.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use splinter_core::digest::Digest;
use splinter_data::Record;
use splinter_eval::campaign::evidence::Budget;
use splinter_eval::campaign::gate::DevResult;
use splinter_eval::campaign::pass_rate::PassRates;
use splinter_eval::campaign::plateau::ArmCost;
use splinter_eval::campaign::signals::StepSignals;
use splinter_knowledge::factory::alarm::Observation;

use super::gc::{Checkpoint, Dropped};
use super::replay::ReplayBuffer;
use super::types::{PolicyRef, TrainReport};

/// The version of this layout; a state of another version is refused, never
/// guessed at.
pub const STATE_SCHEMA: u32 = 1;

/// A candidate trained and not yet gated: a crash between training and the
/// gate resumes at the gate instead of training again.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PendingCandidate {
    /// The arm it belongs to.
    pub arm: String,
    /// The adapter file.
    pub adapter: PathBuf,
    /// The directory it was trained into.
    pub dir: PathBuf,
    /// Its content digest.
    pub digest: Digest,
    /// Bytes it occupies.
    pub bytes: u64,
    /// What training measured.
    pub report: TrainReport,
    /// The step's signals, informative fraction included.
    pub signals: StepSignals,
    /// GPU-seconds the arm had used when training finished.
    pub gpu_seconds: f64,
    /// Rollouts the arm collected.
    pub rollouts: u64,
    /// Tokens the policy generated while collecting, when the collector
    /// counted them.
    pub generated_tokens: Option<u64>,
    /// When the arm started, UTC.
    pub started_at: String,
    /// The verified completions the arm contributes to the replay if it is
    /// accepted.
    pub replay: Vec<Record>,
    /// The step's own measurements, for the ledger.
    pub measurements: BTreeMap<String, f64>,
    /// The mean training reward of the batch by environment, for the
    /// divergence alarm; empty for tasks that come from no environment.
    #[serde(default)]
    pub env_rewards: BTreeMap<String, f64>,
}

/// What the evidence gate has spent: its budget, and the dev tasks earlier
/// arms graded (the exposure of the next arm's draws).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvidenceSpent {
    /// The false evolution budget after the last arm.
    pub budget: Budget,
    /// Dev tasks graded by earlier arms.
    pub used: BTreeSet<String>,
}

/// A campaign as of its last save.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CampaignState {
    /// The layout version.
    pub schema: u32,
    /// The campaign.
    pub name: String,
    /// The digest of the configuration it began under; a controller started
    /// with another configuration refuses to continue it.
    pub config_digest: Digest,
    /// The seed every draw of the campaign derives from, recorded so a run is
    /// replayable.
    pub seed: u64,
    /// The champion: the policy rollouts are drawn from.
    pub policy: PolicyRef,
    /// The per-task pass rates.
    pub rates: PassRates,
    /// What the champion did on the dev tasks; `None` before the first gate.
    pub champion_dev: Option<DevResult>,
    /// The bounded replay of earlier arms' verified data.
    pub replay: ReplayBuffer,
    /// Arms finished.
    pub arms: u32,
    /// Where the strategy ladder is.
    pub strategy_index: usize,
    /// Tasks a decomposition added to the pool.
    pub extra_tasks: Vec<String>,
    /// Checkpoints on disk.
    pub checkpoints: Vec<Checkpoint>,
    /// Checkpoints collected: their digests outlive their files.
    pub dropped: Vec<Dropped>,
    /// A trained candidate awaiting its gate.
    pub pending: Option<PendingCandidate>,
    /// The share of dev units on which the last gate's candidate and champion
    /// differed.
    pub last_discordance: Option<f64>,
    /// What each arm cost and found, oldest first, for the plateau switch.
    pub history: Vec<ArmCost>,
    /// The signals of every step.
    pub signals: Vec<StepSignals>,
    /// GPU-hours the campaign has used.
    pub gpu_hours: f64,
    /// Arms in a row that did not advance the champion.
    pub idle_arms: u32,
    /// What the divergence alarm has seen of each environment, one
    /// observation per gate.
    #[serde(default)]
    pub observations: BTreeMap<String, Vec<Observation>>,
    /// Environments the alarm quarantined: no later arm draws from them.
    #[serde(default)]
    pub quarantined: Vec<String>,
    /// What the evidence gate has spent; `None` under the greedy gate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence: Option<EvidenceSpent>,
}

impl CampaignState {
    /// The state of a campaign about to begin.
    #[must_use]
    pub fn new(
        name: &str,
        config_digest: Digest,
        seed: u64,
        rates: PassRates,
        replay: ReplayBuffer,
    ) -> Self {
        Self {
            schema: STATE_SCHEMA,
            name: name.to_string(),
            config_digest,
            seed,
            policy: PolicyRef::default(),
            rates,
            champion_dev: None,
            replay,
            arms: 0,
            strategy_index: 0,
            extra_tasks: Vec::new(),
            checkpoints: Vec::new(),
            dropped: Vec::new(),
            pending: None,
            last_discordance: None,
            history: Vec::new(),
            signals: Vec::new(),
            gpu_hours: 0.0,
            idle_arms: 0,
            observations: BTreeMap::new(),
            quarantined: Vec::new(),
            evidence: None,
        }
    }
}
