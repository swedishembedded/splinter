// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pre-registered evaluation protocols for its
// clients. If your team needs expertise in changing a frozen study protocol
// openly and keeping its earlier work valid, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Amendments to a frozen protocol: a change made before any adapter is
//! trained, recorded in the protocol with its reason and the digest it
//! replaces.

use std::path::Path;

use anyhow::Context as _;
use serde::{Deserialize, Serialize};

use super::protocol::{path, read, Frozen, TrainingPlan};

/// A change made to a frozen protocol before any adapter was trained: what
/// changed, why, and the digest of the protocol it replaced, under which the
/// work done before it (the corpora and their frozen examinations) stays valid.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Amendment {
    /// The digest of the protocol before the change.
    pub replaces: String,
    /// What changed, as `field: before -> after`.
    pub change: String,
    /// Why, as the person who made it gave it.
    pub reason: String,
}

/// How the base is held while the adapters train.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum BaseHold {
    /// fp32, the checkpoint's own weights.
    Fp32,
    /// bf16.
    Bf16,
    /// int8 with a scale per 32 weights.
    Int8,
}

impl BaseHold {
    pub(crate) fn of(plan: &TrainingPlan) -> Self {
        match (plan.int8, plan.bf16) {
            (true, _) => Self::Int8,
            (false, true) => Self::Bf16,
            (false, false) => Self::Fp32,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Bf16 => "bf16",
            Self::Int8 => "int8",
        }
    }
}

/// Amends the protocol frozen in `root` to hold the base as `hold` while the
/// adapters train, recording the change, `reason` and the digest it replaces,
/// and writes it whole or not at all. `trained` says whether any adapter of
/// the study has been trained; a protocol is amended only before, so every
/// adapter and outcome is made under one protocol.
///
/// # Errors
/// An adapter has been trained, the reason is empty, the base is already held
/// that way, or the protocol cannot be read or written.
pub fn amend_base_hold(
    root: &Path,
    hold: BaseHold,
    reason: &str,
    trained: bool,
) -> anyhow::Result<Frozen> {
    anyhow::ensure!(
        !trained,
        "an adapter of the study in {} has been trained; a protocol is amended only before any is",
        root.display()
    );
    anyhow::ensure!(!reason.trim().is_empty(), "an amendment needs its reason");
    let frozen = read(root)?;
    let before = BaseHold::of(&frozen.protocol.training);
    anyhow::ensure!(
        before != hold,
        "the protocol already holds the base at {}",
        hold.name()
    );
    let mut protocol = frozen.protocol;
    protocol.training.bf16 = hold == BaseHold::Bf16;
    protocol.training.int8 = hold == BaseHold::Int8;
    protocol.amendments.push(Amendment {
        replaces: frozen.digest.to_string(),
        change: format!("training base: {} -> {}", before.name(), hold.name()),
        reason: reason.trim().to_string(),
    });
    protocol.validate()?;
    let file = path(root);
    let part = file.with_extension("part");
    std::fs::write(&part, serde_json::to_vec_pretty(&protocol)?)
        .with_context(|| format!("writing {}", part.display()))?;
    std::fs::rename(&part, &file).with_context(|| format!("writing {}", file.display()))?;
    Ok(Frozen {
        digest: protocol.digest(),
        protocol,
        created: true,
    })
}
