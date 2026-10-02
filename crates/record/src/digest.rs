// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements content-addressed records whose identity
// does not depend on how they were serialized, for its clients. If your team
// needs expertise in data provenance or reproducible pipelines, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Content addresses and the canonical form they are computed over.
//!
//! A [`Digest`] is `<algorithm>:<64 lowercase hex>`. Splinter's own content
//! addresses are `blake3:`, the hash the experience database addresses its
//! objects by. `sha256:` is the form an external tool reports for a file it
//! produced (brain, for an adapter), so such a digest can be recorded and
//! compared as given. [`canonical_json`] is the one serialization every
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
//! The address of a value is the BLAKE3 hash of those bytes.

use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

const BLAKE3: &str = "blake3:";
const SHA256: &str = "sha256:";
const HEX_LEN: usize = 64;

/// A content address: `blake3:<64 lowercase hex digits>`, or `sha256:` for a
/// digest an external tool reported. Validated on construction and on
/// deserialization, so a malformed digest never enters the store.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Digest(String);

/// Why a string is not a [`Digest`].
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{value:?} is not a blake3:<64 lowercase hex> or sha256:<64 lowercase hex> digest")]
pub struct DigestError {
    /// The rejected text.
    pub value: String,
}

fn hex_of(hash: &[u8]) -> String {
    hash.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn read_chunks(mut reader: impl std::io::Read, mut each: impl FnMut(&[u8])) -> std::io::Result<()> {
    let mut buf = [0u8; 64 * 1024];
    loop {
        match reader.read(&mut buf) {
            Ok(0) => return Ok(()),
            Ok(n) => each(&buf[..n]),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
}

impl Digest {
    /// The content address of `bytes`: their BLAKE3 hash.
    #[must_use]
    pub fn of(bytes: &[u8]) -> Self {
        Self(format!("{BLAKE3}{}", blake3::hash(bytes).to_hex()))
    }

    /// The content address of everything `reader` yields, read in chunks, so
    /// a large file is hashed without being held in memory.
    pub fn of_reader(reader: impl std::io::Read) -> std::io::Result<Self> {
        let mut hasher = blake3::Hasher::new();
        read_chunks(reader, |chunk| {
            hasher.update(chunk);
        })?;
        Ok(Self(format!("{BLAKE3}{}", hasher.finalize().to_hex())))
    }

    /// The SHA-256 of `bytes`, for comparing with a digest an external tool
    /// reported for the same bytes.
    #[must_use]
    pub fn sha256_of(bytes: &[u8]) -> Self {
        Self(format!("{SHA256}{}", hex_of(&Sha256::digest(bytes))))
    }

    /// The SHA-256 of everything `reader` yields, read in chunks.
    pub fn sha256_of_reader(reader: impl std::io::Read) -> std::io::Result<Self> {
        let mut hasher = Sha256::new();
        read_chunks(reader, |chunk| hasher.update(chunk))?;
        Ok(Self(format!("{SHA256}{}", hex_of(&hasher.finalize()))))
    }

    /// Parses `text`, refusing anything but `blake3:` or `sha256:` and 64
    /// lowercase hex digits.
    pub fn parse(text: &str) -> Result<Self, DigestError> {
        let well_formed = [BLAKE3, SHA256]
            .iter()
            .filter_map(|prefix| text.strip_prefix(prefix))
            .any(|hex| {
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

    /// The content address whose hex part is `hex`, as found in a file name.
    pub fn from_content_hex(hex: &str) -> Result<Self, DigestError> {
        Self::parse(&format!("{BLAKE3}{hex}"))
    }

    /// The database id of the object this digest addresses, when it is a
    /// content address (`blake3:`); a digest a tool reported has none.
    #[must_use]
    pub fn content_id(&self) -> Option<splinter_expdb::ContentId> {
        self.0
            .strip_prefix(BLAKE3)
            .and_then(|hex| splinter_expdb::ContentId::parse(hex).ok())
    }

    /// The whole digest, `<algorithm>:<hex>`.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The hex part alone: safe as a file name, since it is validated to be
    /// 64 hex digits.
    #[must_use]
    pub fn hex(&self) -> &str {
        &self.0[self.0.find(':').map_or(0, |colon| colon + 1)..]
    }
}

impl std::fmt::Display for Digest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<splinter_expdb::ContentId> for Digest {
    fn from(id: splinter_expdb::ContentId) -> Self {
        Self(format!("{BLAKE3}{id}"))
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
            Digest::of_reader(&b"abc"[..]).unwrap(),
            Digest::of(b"abc"),
            "streaming and whole-buffer hashing agree"
        );
        assert_eq!(
            Digest::of(b"").as_str(),
            "blake3:af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
        );
        assert_eq!(
            Digest::sha256_of(b"").as_str(),
            "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            Digest::sha256_of_reader(&b"abc"[..]).unwrap(),
            Digest::sha256_of(b"abc")
        );
    }

    #[test]
    fn both_algorithms_parse_and_nothing_else_does() {
        let blake = Digest::of(b"x");
        let sha = Digest::sha256_of(b"x");
        assert_eq!(Digest::parse(blake.as_str()).unwrap(), blake);
        assert_eq!(Digest::parse(sha.as_str()).unwrap(), sha);
        assert_ne!(blake.hex(), "");
        assert_eq!(Digest::from_content_hex(blake.hex()).unwrap(), blake);
        for bad in [
            "md5:00",
            &format!("blake3:{}", "A".repeat(64)),
            &format!("sha256:{}", "a".repeat(63)),
            "",
        ] {
            assert!(Digest::parse(bad).is_err(), "{bad:?}");
        }
    }
}
