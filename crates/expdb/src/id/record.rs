// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Record ids: `(writer, sequence)`, unique without any coordination.

use std::fmt;
use std::str::FromStr;

use serde::de::{self, Deserializer};
use serde::{Deserialize, Serialize, Serializer};

use super::writer::WriterId;
use crate::error::{Error, Result};

/// The id of one record. It never changes, whichever file the record
/// currently lives in.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RecordId {
    writer: WriterId,
    sequence: u64,
}

impl RecordId {
    /// The id of a writer's `sequence`th record.
    pub fn new(writer: WriterId, sequence: u64) -> Self {
        Self { writer, sequence }
    }

    /// The writer that made the record.
    pub fn writer(self) -> WriterId {
        self.writer
    }

    /// The record's sequence number within its writer.
    pub fn sequence(self) -> u64 {
        self.sequence
    }

    /// A hash of the id, for filters.
    pub fn hash64(self) -> u64 {
        self.writer.raw().rotate_left(29) ^ self.sequence.wrapping_mul(0x9e37_79b9_7f4a_7c15)
    }
}

impl fmt::Display for RecordId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.writer, self.sequence)
    }
}

impl fmt::Debug for RecordId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RecordId({self})")
    }
}

impl FromStr for RecordId {
    type Err = Error;
    fn from_str(text: &str) -> Result<Self> {
        let bad = || {
            Error::invalid(
                "record id",
                format!("`{text}` is not <writer hex>:<sequence>"),
            )
        };
        let (writer, sequence) = text.split_once(':').ok_or_else(bad)?;
        let writer = u64::from_str_radix(writer, 16).map_err(|_| bad())?;
        let sequence = sequence.parse().map_err(|_| bad())?;
        Ok(Self::new(WriterId::from_raw(writer), sequence))
    }
}

impl Serialize for RecordId {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for RecordId {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        text.parse().map_err(de::Error::custom)
    }
}
