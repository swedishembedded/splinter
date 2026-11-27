// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements experiment controllers that resume after
// any crash without losing or repeating committed work, for its clients. If
// your team needs expertise in durable machine-learning orchestration, you
// can procure our services by sending an email to info@swedishembedded.com.

//! How a stage's product is committed: reused when the product store holds
//! it, else produced into a staging area by the one thread that claims its
//! identity, published, and recorded in the run state.

use std::collections::BTreeSet;
use std::sync::{Condvar, Mutex, PoisonError};

use splinter_core::clock::unix_now;
use splinter_core::digest::Digest;
use splinter_core::factory::identity::StageIdentity;
use splinter_store::factory::products::{ProductError, ProductManifest, Staging};
use splinter_store::factory::runstate::{StageOutcome, StageState};

use super::{io, write_json, Factory, RunError};

/// Where [`Factory::begin`] leaves a stage.
pub(crate) enum Begun<'f> {
    /// Its product is committed already.
    Reused(ProductManifest),
    /// It runs: produce into `staging`, then [`Factory::finish`].
    Fresh {
        /// Its run-state version.
        version: u64,
        /// Where to produce.
        staging: Staging,
        /// This thread's claim on producing it.
        claim: Claim<'f>,
    },
}

/// The stage identities being produced in this process. One identity can
/// stand at several keys (an evaluation's baseline arm, shared by every
/// evaluation of that baseline on those tasks; a recipe input every stage
/// consuming it pins), and stages running at once would otherwise produce
/// it twice, into one set of slots and one run-state record.
#[derive(Default)]
pub(crate) struct Producing {
    ids: Mutex<BTreeSet<Digest>>,
    released: Condvar,
}

impl Producing {
    /// Claim `id`, waiting while another thread produces it.
    fn claim(&self, id: &Digest) -> Claim<'_> {
        let mut ids = self.ids.lock().unwrap_or_else(PoisonError::into_inner);
        while ids.contains(id) {
            ids = self
                .released
                .wait(ids)
                .unwrap_or_else(PoisonError::into_inner);
        }
        ids.insert(id.clone());
        Claim {
            producing: self,
            id: id.clone(),
        }
    }
}

/// A thread's claim on producing one stage identity; released when dropped,
/// whether the production committed, failed or unwound.
pub(crate) struct Claim<'f> {
    producing: &'f Producing,
    id: Digest,
}

impl Drop for Claim<'_> {
    fn drop(&mut self) {
        let mut ids = self
            .producing
            .ids
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        ids.remove(&self.id);
        self.producing.released.notify_all();
    }
}

impl Factory<'_> {
    /// The committed product of the stage `identity` viewed at `key`:
    /// reused when the product store holds it, produced into a staging area
    /// and published otherwise. The run state follows; a failed production
    /// is recorded as failed and leaves nothing committed. Returns the
    /// manifest and whether this call produced it.
    pub(crate) fn commit(
        &self,
        key: &str,
        operation: &str,
        identity: &StageIdentity,
        produce: &dyn Fn(&Self, &Staging) -> Result<(), RunError>,
    ) -> Result<(ProductManifest, bool), RunError> {
        match self.begin(key, operation, identity)? {
            Begun::Reused(manifest) => Ok((manifest, false)),
            Begun::Fresh {
                version,
                staging,
                claim,
            } => {
                let produced = produce(self, &staging);
                self.finish(key, identity, version, staging, claim, produced)
                    .map(|manifest| (manifest, true))
            }
        }
    }

    /// The first half of [`Self::commit`]: the product of `identity`
    /// reused, or the stage recorded as running with an empty staging area
    /// to produce it in, claimed by this thread. A thread that finds another
    /// producing the same identity waits for it, then reuses its product.
    pub(crate) fn begin(
        &self,
        key: &str,
        operation: &str,
        identity: &StageIdentity,
    ) -> Result<Begun<'_>, RunError> {
        let id = identity.digest();
        let claim = self.producing.claim(&id);
        let record = self
            .state
            .upsert_pending(&self.fence, &id, key, operation, unix_now())?;
        // Not committed: computed afresh. Committed: reused as committed,
        // or refused (a rotted blob is never handed on, nor recomputed over).
        let committed = match self.products.ensure_view(&id, &self.run.join(key)) {
            Ok(manifest) => Some(manifest),
            Err(ProductError::UnknownProduct(_)) => None,
            Err(e) => return Err(e.into()),
        };
        if let Some(manifest) = committed {
            if record.state != StageState::Completed {
                let running = self.to_running(&id, record.version, record.state)?;
                self.state.transition(
                    &self.fence,
                    &id,
                    running,
                    StageState::Completed,
                    StageOutcome::Product(manifest.digest()),
                    unix_now(),
                )?;
            }
            return Ok(Begun::Reused(manifest));
        }
        let version = self.to_running(&id, record.version, record.state)?;
        let staging = self.products.begin_staging()?;
        Ok(Begun::Fresh {
            version,
            staging,
            claim,
        })
    }

    /// The second half of [`Self::commit`]: what was `produced` into
    /// `staging` published as the product of `identity`, or the stage
    /// recorded as failed (a cancelled one stays running, for the resume).
    pub(crate) fn finish(
        &self,
        key: &str,
        identity: &StageIdentity,
        version: u64,
        staging: Staging,
        claim: Claim<'_>,
        produced: Result<(), RunError>,
    ) -> Result<ProductManifest, RunError> {
        // Released when this returns: the product is committed, or the
        // stage recorded as failed or left running.
        let _claim = claim;
        let id = identity.digest();
        if let Err(error) =
            produced.and_then(|()| write_json(&staging.path().join("stage.json"), identity))
        {
            if !matches!(error, RunError::Cancelled) {
                let failed = StageOutcome::Error(error.to_string());
                self.state.transition(
                    &self.fence,
                    &id,
                    version,
                    StageState::Failed,
                    failed,
                    unix_now(),
                )?;
            }
            return Err(error);
        }
        let manifest = self
            .products
            .publish(&id, key, staging, &self.run.join(key))?;
        self.state.transition(
            &self.fence,
            &id,
            version,
            StageState::Completed,
            StageOutcome::Product(manifest.digest()),
            unix_now(),
        )?;
        let work = self.work(key);
        if work.exists() {
            std::fs::remove_dir_all(&work).map_err(io(&work))?;
        }
        Ok(manifest)
    }

    /// Move a stage to running from wherever it stands; returns its version.
    fn to_running(&self, id: &Digest, version: u64, state: StageState) -> Result<u64, RunError> {
        match state {
            StageState::Running => Ok(version),
            StageState::Pending | StageState::Failed => Ok(self
                .state
                .transition(
                    &self.fence,
                    id,
                    version,
                    StageState::Running,
                    StageOutcome::None,
                    unix_now(),
                )?
                .version),
            StageState::Completed => Ok(version),
        }
    }
}
