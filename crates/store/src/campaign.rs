// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for long-running model-improvement
// campaigns that survive a crash and keep an honest account of what each
// experiment cost and found, for its clients. If your team needs expertise in
// resumable training orchestration or experiment ledgers you can procure our
// services by sending an email to info@swedishembedded.com.

//! The durable state of a continuous learning campaign, and its ledger.
//!
//! A campaign is resumable: everything the controller needs to continue after
//! a crash is its state document, a value the controller defines and this
//! module only keeps. The state is stored as a document by its content address
//! and a pointer, `campaign/<name>/state`, names the current one; moving the
//! pointer is a compare-and-swap, so
//!
//! * a crash leaves either the old state or the new one, never a mixture;
//! * two controllers started on the same campaign cannot both advance it: the
//!   second learns it lost ([`CampaignError::Conflict`]) instead of silently
//!   overwriting the first;
//! * every earlier state stays on the pointer's history, so a rollback is a
//!   move to an older value.
//!
//! The ledger is the campaign's account of its experiments: one immutable entry per arm event with the hypothesis, the
//! digest of the configuration, the GPU-hours, the cost, and the verdict. An
//! unmeasured quantity is absent, never `0`. Entries are documents, kept in
//! the order they were written.

use std::collections::BTreeMap;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use splinter_core::digest::Digest;

use crate::error::StoreError;
use crate::workspace::Workspace;

/// The document class of a campaign's state.
const STATE_CLASS: &str = "campaign-state";

/// The document class of a ledger entry.
const LEDGER_CLASS: &str = "campaign-ledger";

/// Why a campaign operation failed.
#[derive(Debug, thiserror::Error)]
pub enum CampaignError {
    /// The name cannot be a campaign: empty, or it holds a character a
    /// pointer name keeps for its own structure.
    #[error("{0:?} is not a campaign name: use letters, digits, '-', '_' and '.'")]
    Name(String),
    /// Another controller advanced the campaign since this one loaded it.
    #[error("campaign {campaign} was advanced by another controller (expected {expected:?}, found {found:?}); reload it")]
    Conflict {
        /// The campaign.
        campaign: String,
        /// The state this controller loaded.
        expected: Option<String>,
        /// The state the campaign is at.
        found: Option<String>,
    },
    /// The store failed.
    #[error(transparent)]
    Store(#[from] StoreError),
}

/// One campaign's durable state and ledger.
pub struct CampaignStore<'a> {
    workspace: &'a Workspace,
    name: String,
}

/// A state as it was loaded: the value and the address a later save names as
/// the one it replaces.
#[derive(Clone, Debug, PartialEq)]
pub struct Loaded<S> {
    /// The state.
    pub state: S,
    /// Its address.
    pub digest: Digest,
    /// How many times the pointer has moved to get here.
    pub version: u64,
}

impl<'a> CampaignStore<'a> {
    /// The store of campaign `name`.
    ///
    /// # Errors
    /// The name is empty or holds anything but letters, digits, `-`, `_`, `.`.
    pub fn new(workspace: &'a Workspace, name: &str) -> Result<Self, CampaignError> {
        let sound = !name.is_empty()
            && name != "."
            && name != ".."
            && name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
        if !sound {
            return Err(CampaignError::Name(name.to_string()));
        }
        Ok(Self {
            workspace,
            name: name.to_string(),
        })
    }

    /// The campaign's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    fn pointer(&self) -> String {
        format!("campaign/{}/state", self.name)
    }

    /// The campaign's current state; `None` for a campaign never saved.
    ///
    /// # Errors
    /// The pointer names a document the store lacks or that no longer hashes
    /// to its address, or is not an `S`.
    pub fn load<S: DeserializeOwned + Serialize>(
        &self,
    ) -> Result<Option<Loaded<S>>, CampaignError> {
        let Some((version, value)) = self.workspace.pointer(&self.pointer())? else {
            return Ok(None);
        };
        let digest = Digest::parse(&value).map_err(|e| StoreError::UndecodableObject {
            what: format!("pointer {}", self.pointer()),
            reason: e.to_string(),
        })?;
        let state = self
            .workspace
            .get_document::<S>(STATE_CLASS, &digest)?
            .ok_or_else(|| StoreError::UndecodableObject {
                what: format!("campaign {} state {digest}", self.name),
                reason: "the pointer names a state the store does not hold".into(),
            })?;
        Ok(Some(Loaded {
            state,
            digest,
            version,
        }))
    }

    /// Makes `state` the campaign's current state, provided it still is the
    /// one `replaces` names (`None` for a campaign never saved). Returns the
    /// new state's address. Saving the state the campaign is already at
    /// changes nothing.
    ///
    /// # Errors
    /// [`CampaignError::Conflict`] when another controller moved the campaign;
    /// a store failure.
    pub fn save<S: Serialize>(
        &self,
        replaces: Option<&Digest>,
        state: &S,
        at: &str,
    ) -> Result<Digest, CampaignError> {
        let digest = self.workspace.put_document(STATE_CLASS, state)?;
        let moved = self.workspace.move_pointer(
            &self.pointer(),
            replaces.map(Digest::as_str),
            digest.as_str(),
            at,
        );
        match moved {
            Ok(_) => Ok(digest),
            Err(StoreError::PointerConflict {
                expected, found, ..
            }) => Err(CampaignError::Conflict {
                campaign: self.name.clone(),
                expected,
                found,
            }),
            Err(other) => Err(other.into()),
        }
    }

    /// Appends `entry` to the ledger; returns its address.
    ///
    /// # Errors
    /// The entry names another campaign, or the store failed.
    pub fn record(&self, entry: &LedgerEntry) -> Result<Digest, CampaignError> {
        if entry.campaign != self.name {
            return Err(StoreError::Rejected {
                what: "ledger entry",
                reason: format!(
                    "it names campaign {:?}, this is {:?}",
                    entry.campaign, self.name
                ),
            }
            .into());
        }
        Ok(self.workspace.put_document(LEDGER_CLASS, entry)?)
    }

    /// The campaign's ledger entries, in the order they were written.
    ///
    /// # Errors
    /// The store failed or an entry no longer hashes to its address.
    pub fn ledger(&self) -> Result<Vec<LedgerEntry>, CampaignError> {
        let mut entries = Vec::new();
        for id in self.workspace.documents_in_order(LEDGER_CLASS)? {
            if let Some(entry) = self
                .workspace
                .get_document::<LedgerEntry>(LEDGER_CLASS, &id)?
            {
                if entry.campaign == self.name {
                    entries.push(entry);
                }
            }
        }
        Ok(entries)
    }
}

/// What an arm cost, in what is counted; each is absent when not measured.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Cost {
    /// GPU-hours: wall-clock hours times the cards held.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_hours: Option<f64>,
    /// Rollouts collected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rollouts: Option<u64>,
    /// Tokens generated by the policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generated_tokens: Option<u64>,
    /// Optimiser steps taken.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub train_steps: Option<u64>,
}

/// How an arm ended.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum Verdict {
    /// The arm has started and has not ended.
    Running,
    /// Refused before it ran: its exam could not detect the target effect, or
    /// the plan was invalid.
    Refused {
        /// Why.
        reason: String,
    },
    /// The candidate passed the dev gate and became the champion.
    Accepted {
        /// Whether the improvement is significant at the level this gate may
        /// spend; an accepted arm that is not is a point-estimate advance.
        significant: bool,
    },
    /// The candidate failed the dev gate and the champion stayed.
    RolledBack {
        /// Why.
        reason: String,
    },
    /// The arm was stopped by a signal bound before it finished.
    Halted {
        /// Why.
        reason: String,
    },
    /// The arm failed to run.
    Failed {
        /// Why.
        reason: String,
    },
}

/// One account of one arm of a campaign.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LedgerEntry {
    /// The campaign.
    pub campaign: String,
    /// The arm, unique within the campaign.
    pub arm: String,
    /// Which gate of the campaign this arm's verdict is: the n-th, from 1,
    /// which sets the level it may spend.
    pub generation: u32,
    /// The strategy the arm ran (`policy-gradient`, `sft-repair`, ...).
    pub strategy: String,
    /// What the arm was expected to show, in one sentence, written before it
    /// ran.
    pub hypothesis: String,
    /// The digest of the configuration the arm ran under.
    pub config_digest: Digest,
    /// When this entry was written, UTC.
    pub at: String,
    /// What it cost; empty fields are unmeasured.
    pub cost: Cost,
    /// What the arm discovered, in the campaign's unit (tasks newly solved on
    /// the dev set); absent for an arm still running.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub discoveries: Option<f64>,
    /// How it ended.
    pub verdict: Verdict,
    /// The numbers behind the verdict (dev pass rates, informative fraction,
    /// KL), by name.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub measurements: BTreeMap<String, f64>,
}

impl LedgerEntry {
    /// The latest entry of each arm, in the order the arms first appeared.
    #[must_use]
    pub fn latest_per_arm(entries: &[Self]) -> Vec<&Self> {
        let mut order: Vec<&str> = Vec::new();
        let mut latest: BTreeMap<&str, &Self> = BTreeMap::new();
        for entry in entries {
            if !latest.contains_key(entry.arm.as_str()) {
                order.push(&entry.arm);
            }
            latest.insert(&entry.arm, entry);
        }
        order.into_iter().map(|arm| latest[arm]).collect()
    }
}
