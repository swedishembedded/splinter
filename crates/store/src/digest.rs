// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements content-addressed records whose identity
// does not depend on how they were serialized, for its clients. If your team
// needs expertise in data provenance or reproducible pipelines, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Content addresses and the canonical form they are computed over.
//!
//! A [`Digest`] is `sha256:<64 lowercase hex>`, the form brain reports for
//! an adapter file. [`canonical_json`] is the one serialization every
//! content address in this crate hashes:
//!
//! * the value is first converted to a JSON value tree with `serde_json`;
//! * objects are written with their keys sorted by the UTF-8 bytes of the
//!   key, so insertion order (or a map implementation that keeps it) never
//!   changes an address;
//! * no whitespace anywhere: `{"a":1,"b":[true,null]}`;
//! * strings and numbers are written exactly as `serde_json` writes them
//!   (its string escaping; integers in decimal; floats in their shortest
//!   round-trip form, so `1.0` stays `1.0`).
//!
//! The address of a value is the SHA-256 of those bytes.

use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

const PREFIX: &str = "sha256:";
const HEX_LEN: usize = 64;

/// A content address: `sha256:<64 lowercase hex digits>`. Validated on
/// construction and on deserialization, so a malformed digest never enters
/// the store.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Digest(String);

/// Why a string is not a [`Digest`].
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{value:?} is not a sha256:<64 lowercase hex> digest")]
pub struct DigestError {
    /// The rejected text.
    pub value: String,
}

impl Digest {
    /// The SHA-256 of `bytes`.
    #[must_use]
    pub fn of(bytes: &[u8]) -> Self {
        let hash = Sha256::digest(bytes);
        let mut hex = String::with_capacity(PREFIX.len() + HEX_LEN);
        hex.push_str(PREFIX);
        for byte in hash {
            hex.push_str(&format!("{byte:02x}"));
        }
        Self(hex)
    }

    /// Parses `text`, refusing anything but `sha256:` and 64 lowercase hex
    /// digits.
    pub fn parse(text: &str) -> Result<Self, DigestError> {
        let well_formed = text.strip_prefix(PREFIX).is_some_and(|hex| {
            hex.len() == HEX_LEN && hex.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
        });
        if well_formed {
            Ok(Self(text.to_string()))
        } else {
            Err(DigestError {
                value: text.to_string(),
            })
        }
    }

    /// The whole digest, `sha256:<hex>`.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The hex part alone: safe as a file name, since it is validated to be
    /// 64 hex digits.
    #[must_use]
    pub fn hex(&self) -> &str {
        &self.0[PREFIX.len()..]
    }
}

impl std::fmt::Display for Digest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl TryFrom<String> for Digest {
    type Error = DigestError;
    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::parse(&value)
    }
}

impl From<Digest> for String {
    fn from(d: Digest) -> Self {
        d.0
    }
}

/// `value` in the canonical form described in the module documentation.
pub fn canonical_json<T: Serialize + ?Sized>(value: &T) -> Result<Vec<u8>, serde_json::Error> {
    let tree = serde_json::to_value(value)?;
    let mut out = Vec::new();
    write_canonical(&tree, &mut out)?;
    Ok(out)
}

fn write_canonical(value: &serde_json::Value, out: &mut Vec<u8>) -> Result<(), serde_json::Error> {
    use serde_json::Value;
    match value {
        Value::Array(items) => {
            out.push(b'[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_canonical(item, out)?;
            }
            out.push(b']');
        }
        Value::Object(map) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
            out.push(b'{');
            for (i, (key, item)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                serde_json::to_writer(&mut *out, key)?;
                out.push(b':');
                write_canonical(item, out)?;
            }
            out.push(b'}');
        }
        scalar => serde_json::to_writer(&mut *out, scalar)?,
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_canonical_form_sorts_keys_and_drops_whitespace() {
        let value = serde_json::json!({"b": [1, 2.5, "x\n"], "a": {"d": null, "c": true}});
        assert_eq!(
            String::from_utf8(canonical_json(&value).unwrap()).unwrap(),
            r#"{"a":{"c":true,"d":null},"b":[1,2.5,"x\n"]}"#
        );
        assert_eq!(
            Digest::of(b"").as_str(),
            "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }
}
