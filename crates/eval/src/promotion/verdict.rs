// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for promotion gates that check an
// examiner's signature in the learner's own process, for its clients. If your
// team needs expertise in tamper-evident evaluation evidence you can procure
// our services by sending an email to info@swedishembedded.com.

//! An examiner's signed verdict (`bench.examination/1`), verified in this
//! process: the signature, the schema, that the examiner found the
//! measurement valid and unflagged, and that it is about the candidate and
//! the predecessor the caller expects. What comes out is the verdict's
//! aggregates only ([`Aggregate`]); a document that fails any check is a
//! typed [`Rejection`] and yields nothing.
//!
//! The signature is the examiner's: HMAC-SHA256 under its 32-byte key over
//! the document's canonical bytes without its `signature` field
//! ([`super::canonical`]), carried as `{"alg": "hmac-sha256", "key_id",
//! "value"}` with the MAC in lower-case hex.

use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::canonical::canonical_bytes;
use super::mac::{hmac_sha256, same};

/// The schema of the verdicts the gate reads.
pub const EXAMINATION_SCHEMA: &str = "bench.examination/1";

/// The signature algorithm the examiner signs with.
pub const SIGNATURE_ALGORITHM: &str = "hmac-sha256";

/// The bytes of a signing key.
pub const KEY_BYTES: usize = 32;

/// The integrity flag the examiner raises when it was given no training
/// material to audit for overlap: it says what was not checked and refuses
/// nothing.
pub const NO_MATERIAL_FLAG: &str = "no_training_material_to_audit";

/// The examiner's verdict signing key.
#[derive(Clone)]
pub struct VerdictKey([u8; KEY_BYTES]);

impl std::fmt::Debug for VerdictKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("VerdictKey(..)")
    }
}

/// Why a key cannot be used.
#[derive(Debug, thiserror::Error)]
pub enum KeyError {
    /// The file cannot be read.
    #[error("{path}: cannot read the verdict signing key ({reason})")]
    Read {
        /// The file.
        path: String,
        /// Why.
        reason: String,
    },
    /// Others than its owner may read it.
    #[error("{path} has permissions {mode:o}: a verdict signing key must be readable by its owner alone (chmod 600)")]
    Exposed {
        /// The file.
        path: String,
        /// Its permission bits.
        mode: u32,
    },
    /// It is not a key.
    #[error("{path}: a verdict signing key is {KEY_BYTES} bytes, this file holds {found}")]
    Length {
        /// The file.
        path: String,
        /// The bytes it holds.
        found: usize,
    },
}

impl VerdictKey {
    /// A key from its bytes.
    #[must_use]
    pub fn from_bytes(bytes: [u8; KEY_BYTES]) -> Self {
        Self(bytes)
    }

    /// The key in the file at `path`, which only its owner may read.
    ///
    /// # Errors
    /// The file cannot be read, others may read it, or it is not 32 bytes.
    pub fn read(path: &Path) -> Result<Self, KeyError> {
        use std::os::unix::fs::PermissionsExt;
        let shown = path.display().to_string();
        let read = |e: std::io::Error| KeyError::Read {
            path: shown.clone(),
            reason: e.to_string(),
        };
        let mode = std::fs::metadata(path).map_err(read)?.permissions().mode() & 0o777;
        if mode & 0o077 != 0 {
            return Err(KeyError::Exposed { path: shown, mode });
        }
        let bytes = std::fs::read(path).map_err(read)?;
        let found = bytes.len();
        let key: [u8; KEY_BYTES] = bytes
            .try_into()
            .map_err(|_| KeyError::Length { path: shown, found })?;
        Ok(Self(key))
    }

    /// The MAC of `document` without its `signature` field, in lower-case
    /// hex; `None` for a document the examiner could not have written.
    fn mac(&self, document: &Value) -> Option<String> {
        let mut body = document.as_object()?.clone();
        body.remove("signature");
        let bytes = canonical_bytes(&Value::Object(body))?;
        Some(
            hmac_sha256(&self.0, &bytes)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect(),
        )
    }

    /// `document` signed as the examiner signs it, for a test that plays the
    /// examiner; the gate never signs.
    #[must_use]
    pub fn sign(&self, document: &Value) -> Value {
        let mut signed = document.clone();
        if let (Some(mac), Some(object)) = (self.mac(document), signed.as_object_mut()) {
            object.insert(
                "signature".into(),
                serde_json::json!({"alg": SIGNATURE_ALGORITHM, "key_id": "test", "value": mac}),
            );
        }
        signed
    }
}

/// Why a verdict was not accepted as evidence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, thiserror::Error)]
#[serde(tag = "rejection", rename_all = "snake_case")]
pub enum Rejection {
    /// It carries no signature of the examiner's algorithm.
    #[error("the verdict is not signed with {SIGNATURE_ALGORITHM}")]
    Unsigned,
    /// Its signature does not verify under the key: it is not the
    /// examiner's, or it was altered.
    #[error(
        "the verdict's signature does not verify: it is not the examiner's, or it was altered"
    )]
    BadSignature,
    /// It is not an examination verdict.
    #[error("the verdict's schema is {found:?}, not {EXAMINATION_SCHEMA}")]
    Schema {
        /// The schema it names.
        found: Option<String>,
    },
    /// A field the gate reads is missing or of the wrong type.
    #[error("the verdict's {field} is missing or malformed")]
    Malformed {
        /// The field.
        field: String,
    },
    /// The examiner found the measurement invalid.
    #[error("the examiner found the measurement invalid: {}", reasons.join("; "))]
    Invalid {
        /// The examiner's reasons.
        reasons: Vec<String>,
    },
    /// The examiner raised integrity flags.
    #[error("the examiner raised integrity flags: {}", flags.join(", "))]
    Flagged {
        /// The flags.
        flags: Vec<String>,
    },
    /// It is about another candidate.
    #[error("the verdict is about candidate {found:?}, not {expected:?}")]
    Candidate {
        /// The candidate the caller expects.
        expected: Option<String>,
        /// The one the verdict names.
        found: Option<String>,
    },
    /// It compares with another predecessor.
    #[error("the verdict compares with predecessor {found:?}, not {expected:?}")]
    Predecessor {
        /// The predecessor the caller expects.
        expected: Option<String>,
        /// The one the verdict names.
        found: Option<String>,
    },
    /// It is not at the level the gate decides at.
    #[error("the verdict's intervals are at level {found}, the gate decides at {expected}")]
    Level {
        /// The gate's level, as the examiner's alpha.
        expected: f64,
        /// The verdict's.
        found: f64,
    },
    /// Its material is not sealed, where a sealed slice is required.
    #[error("the verdict is over public material; a sealed exam is required")]
    NotSealed,
    /// It carries no burn record.
    #[error("the verdict over a sealed exam carries no burn record")]
    NoBurn,
    /// Its burn record is of another exam.
    #[error("the verdict's burn record is of exam {found}, not of {exam}")]
    BurnOfAnotherExam {
        /// The exam the verdict is over.
        exam: String,
        /// The exam the burn record names.
        found: String,
    },
    /// The exam was spent by another candidate and predecessor before.
    #[error("the exam was spent by candidate {candidate:?} against {predecessor:?} before: it is not fresh")]
    Spent {
        /// The candidate that burned it.
        candidate: Option<String>,
        /// The predecessor it was compared with.
        predecessor: Option<String>,
    },
}

/// The models a verdict must be about: each by its adapter's identity as the
/// examiner names it (`sha256:<hex>` of the adapter file), `None` for the
/// base served as it is.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pair {
    /// The candidate.
    pub candidate: Option<String>,
    /// What it is compared with.
    pub predecessor: Option<String>,
}

/// The aggregates of a verified verdict: what the gate decides on.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Aggregate {
    /// The collection's name.
    pub collection: String,
    /// The exam's digest.
    pub exam: String,
    /// Whether the material is a sealed exam.
    pub sealed: bool,
    /// Items in the exam.
    pub items: u64,
    /// The examiner's one-sided level; its intervals are at `1 - alpha`.
    pub alpha: f64,
    /// The paired change in the share of items right, candidate less
    /// predecessor.
    pub gain: Option<f64>,
    /// Its two-sided `1 - alpha` family-clustered interval; `None` with
    /// fewer than two families.
    pub interval: Option<(f64, f64)>,
    /// The lower bound of its one-sided `1 - alpha` interval.
    pub one_sided_lower: Option<f64>,
    /// Families the interval clusters by.
    pub families: Option<u64>,
    /// The burn record of a sealed exam: the pair whose verdict spent it.
    pub burned_by: Option<Pair>,
}

/// A verdict that verified: its pair and its aggregates. Only
/// [`verify`] makes one, so whatever holds one holds checked evidence.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Verified {
    pair: Pair,
    aggregate: Aggregate,
}

impl Verified {
    /// The pair it is about.
    #[must_use]
    pub fn pair(&self) -> &Pair {
        &self.pair
    }

    /// Its aggregates.
    #[must_use]
    pub fn aggregate(&self) -> &Aggregate {
        &self.aggregate
    }

    /// Accepted only when it is over a sealed exam this pair spent: its burn
    /// record names this exam and this pair.
    ///
    /// # Errors
    /// Public material, no burn record, a burn record of another exam, or an
    /// exam another pair spent first.
    pub fn fresh_slice(&self) -> Result<(), Rejection> {
        let aggregate = &self.aggregate;
        if !aggregate.sealed {
            return Err(Rejection::NotSealed);
        }
        let Some(burned) = &aggregate.burned_by else {
            return Err(Rejection::NoBurn);
        };
        if burned != &self.pair {
            return Err(Rejection::Spent {
                candidate: burned.candidate.clone(),
                predecessor: burned.predecessor.clone(),
            });
        }
        Ok(())
    }
}

fn text(doc: &Value, pointer: &str) -> Result<Option<String>, Rejection> {
    match doc.pointer(pointer) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(_) => Err(malformed(pointer)),
    }
}

fn malformed(pointer: &str) -> Rejection {
    Rejection::Malformed {
        field: pointer.trim_start_matches('/').replace('/', "."),
    }
}

fn number(doc: &Value, pointer: &str) -> Result<Option<f64>, Rejection> {
    match doc.pointer(pointer) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v.as_f64().map(Some).ok_or_else(|| malformed(pointer)),
    }
}

/// `document` checked as the examiner's verdict on `expected`, at the
/// examiner's level `alpha`: the signature under `key`, the schema, the
/// examiner's own validity and integrity flags, the pair, the level.
///
/// # Errors
/// The first check that fails, typed.
pub fn verify(
    document: &Value,
    key: &VerdictKey,
    expected: &Pair,
    alpha: f64,
) -> Result<Verified, Rejection> {
    let signature = document.get("signature");
    let (Some(alg), Some(value)) = (
        signature.and_then(|s| s.get("alg")).and_then(Value::as_str),
        signature
            .and_then(|s| s.get("value"))
            .and_then(Value::as_str),
    ) else {
        return Err(Rejection::Unsigned);
    };
    if alg != SIGNATURE_ALGORITHM {
        return Err(Rejection::Unsigned);
    }
    let computed = key.mac(document).ok_or(Rejection::BadSignature)?;
    if !same(computed.as_bytes(), value.as_bytes()) {
        return Err(Rejection::BadSignature);
    }
    let schema = text(document, "/schema")?;
    if schema.as_deref() != Some(EXAMINATION_SCHEMA) {
        return Err(Rejection::Schema { found: schema });
    }
    if document.get("valid").and_then(Value::as_bool) != Some(true) {
        let reasons = document
            .get("invalid_reasons")
            .and_then(Value::as_array)
            .map(|r| {
                r.iter()
                    .filter_map(|x| x.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        return Err(Rejection::Invalid { reasons });
    }
    let flags: Vec<String> = document
        .get("integrity_flags")
        .and_then(Value::as_array)
        .ok_or_else(|| malformed("/integrity_flags"))?
        .iter()
        .filter_map(Value::as_str)
        .filter(|flag| *flag != NO_MATERIAL_FLAG)
        .map(String::from)
        .collect();
    if !flags.is_empty() {
        return Err(Rejection::Flagged { flags });
    }
    let found = Pair {
        candidate: text(document, "/candidate/adapter_digest")?,
        predecessor: text(document, "/predecessor/adapter_digest")?,
    };
    if found.candidate != expected.candidate {
        return Err(Rejection::Candidate {
            expected: expected.candidate.clone(),
            found: found.candidate,
        });
    }
    if found.predecessor != expected.predecessor {
        return Err(Rejection::Predecessor {
            expected: expected.predecessor.clone(),
            found: found.predecessor,
        });
    }
    let level = number(document, "/alpha")?.ok_or_else(|| malformed("/alpha"))?;
    if (level - alpha).abs() > 1e-12 {
        return Err(Rejection::Level {
            expected: alpha,
            found: level,
        });
    }
    let exam = text(document, "/collection/exam_sha256")?
        .ok_or_else(|| malformed("/collection/exam_sha256"))?;
    let burned_by = match document.get("burn") {
        None | Some(Value::Null) => None,
        Some(_) => {
            let burned = text(document, "/burn/exam_sha256")?
                .ok_or_else(|| malformed("/burn/exam_sha256"))?;
            if burned != exam {
                return Err(Rejection::BurnOfAnotherExam {
                    exam,
                    found: burned,
                });
            }
            Some(Pair {
                candidate: text(document, "/burn/candidate")?,
                predecessor: text(document, "/burn/predecessor")?,
            })
        }
    };
    let interval = match document.pointer("/paired_gain/interval") {
        None | Some(Value::Null) => None,
        Some(Value::Array(ends)) if ends.len() == 2 => Some((
            ends[0]
                .as_f64()
                .ok_or_else(|| malformed("/paired_gain/interval"))?,
            ends[1]
                .as_f64()
                .ok_or_else(|| malformed("/paired_gain/interval"))?,
        )),
        Some(_) => return Err(malformed("/paired_gain/interval")),
    };
    let aggregate = Aggregate {
        collection: text(document, "/collection/name")?.unwrap_or_default(),
        exam,
        sealed: document
            .pointer("/collection/sealed")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        items: document
            .pointer("/collection/items")
            .and_then(Value::as_u64)
            .ok_or_else(|| malformed("/collection/items"))?,
        alpha: level,
        gain: number(document, "/paired_gain/mean")?,
        interval,
        one_sided_lower: number(document, "/paired_gain/one_sided_lower")?,
        families: document
            .pointer("/paired_gain/families")
            .and_then(Value::as_u64),
        burned_by,
    };
    Ok(Verified {
        pair: found,
        aggregate,
    })
}
