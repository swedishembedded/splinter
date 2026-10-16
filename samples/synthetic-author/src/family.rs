// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled validation of learning systems
// against ground truth that cannot have been seen in pretraining, for its
// clients. If your team needs expertise in keeping the model that writes a
// training corpus from also being the one that vouches for it, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The protocol check that the models of a corpus are of different families.
//!
//! One model writes the essays, one reads each draft for the gate and one
//! reads the decisions out of the kept articles. If two of them are the same
//! family, a mistake that family makes consistently is written, passed by the
//! gate and read back without anything noticing. So the families differ, or the
//! run says it was allowed to be otherwise and the manifest records it.

use std::str::FromStr;

use anyhow::Context as _;
use splinter_sdk::vocabulary::model_ref::ModelRef;

/// The models of a corpus, by role.
#[derive(Clone, Copy, Debug)]
pub struct Roles<'a> {
    /// The model that writes the essays.
    pub writer: &'a str,
    /// The model that reads each draft for the gate, if there is one.
    pub reader: Option<&'a str>,
    /// The model that reads the decisions out of the kept articles, if it is
    /// known yet.
    pub miner: Option<&'a str>,
}

/// The family of the model `reference` names: the letters its name begins with,
/// in lower case, so that `aster-large-4`, `Aster-3.5` and a checkpoint
/// directory `Aster-tuned` are one family, and `Cedar3-8B` and
/// `Cedar2.5-7B+adapter` another. A policy alias is its own family, since the
/// model behind it is not named here.
///
/// # Errors
/// `reference` is not a model reference.
pub fn family(reference: &str) -> anyhow::Result<String> {
    let parsed = ModelRef::from_str(reference).with_context(|| format!("model {reference:?}"))?;
    let name = match &parsed {
        ModelRef::Policy(alias) => return Ok(format!("policy:{alias}")),
        ModelRef::Local { checkpoint, .. } => checkpoint.as_str(),
        ModelRef::Remote { name, .. } => name.as_str(),
    };
    let last = name.rsplit('/').find(|s| !s.is_empty()).unwrap_or(name);
    let letters: String = last
        .chars()
        .take_while(char::is_ascii_alphabetic)
        .map(|c| c.to_ascii_lowercase())
        .collect();
    anyhow::ensure!(
        !letters.is_empty(),
        "model {reference:?} has no name to take a family from"
    );
    Ok(letters)
}

/// Checks that no two of the roles named are the same family.
///
/// # Errors
/// Two roles are the same family; the message names them and the family, and
/// says how to set the check aside.
pub fn check_distinct(roles: &Roles) -> anyhow::Result<()> {
    let named: Vec<(&str, &str)> = [
        Some(("writer", roles.writer)),
        roles.reader.map(|r| ("reader", r)),
        roles.miner.map(|m| ("miner", m)),
    ]
    .into_iter()
    .flatten()
    .collect();
    let families = named
        .iter()
        .map(|(_, reference)| family(reference))
        .collect::<anyhow::Result<Vec<_>>>()?;
    for (i, first) in families.iter().enumerate() {
        for (j, second) in families.iter().enumerate().skip(i + 1) {
            anyhow::ensure!(
                first != second,
                "the {} ({}) and the {} ({}) are the same model family ({first}); a corpus is \
                 written, passed and read back by models of different families so that no one \
                 family's consistent mistake goes through unnoticed. Use a model of another \
                 family, or pass --allow-same-family to proceed and have the manifest say so",
                named[i].0,
                named[i].1,
                named[j].0,
                named[j].1,
            );
        }
    }
    Ok(())
}
