// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What names a stored dataset.

use serde::{Deserialize, Serialize};

use crate::digest::Digest;

/// A stored dataset's id: the digest of its manifest's canonical form.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DatasetId(pub Digest);

impl std::fmt::Display for DatasetId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
