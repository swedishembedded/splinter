// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for promotion gates that check an
// examiner's signature in the learner's own process, for its clients. If your
// team needs expertise in tamper-evident evaluation evidence you can procure
// our services by sending an email to info@swedishembedded.com.

//! An examiner's signed verdict, verified in this process: the signature,
//! the schema, that the examiner found the measurement valid and unflagged,
//! and that it is about the candidate and the predecessor the caller expects.
//! What comes out is the verdict's aggregates only ([`Aggregate`]); a
//! document that fails any check is a typed [`Rejection`] and yields nothing.
//!
//! Two schemas, one per protocol: `bench.examination/2` (direct completion,
//! `bench examine`: one paired comparison over one exam, its burn record as
//! `burn`) and `bench.verdict/3` (the agent protocol, `bench run` judged by
//! `bench verdict`: the comparison sections `overall` and `retention`, the
//! sealed exams it compares as `sealed_exams` and one burn record per exam
//! as `burns`). For a verdict/3 the gate reads the `retention` section for
//! the retention role and `overall` otherwise.
//!
//! The signature is the examiner's: Ed25519 under its private key over the
//! document's canonical bytes without its `signature` field
//! ([`super::canonical`]), carried as `{"alg": "ed25519", "key_id",
//! "value"}` with the 64-byte signature in hex, and verified with the
//! examiner's public key ([`PublicKey`]). A document under the examiner's
//! retired shared-key MAC (`hmac-sha256`) is unsigned here.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::canonical::canonical_bytes;
use super::key::PublicKey;
use super::Role;

/// The schema of the direct-completion verdicts the gate reads.
pub const EXAMINATION_SCHEMA: &str = "bench.examination/2";

/// The schema of the agent-protocol verdicts the gate reads.
pub const AGENT_SCHEMA: &str = "bench.verdict/3";

/// The signature algorithm the examiner signs with.
pub const SIGNATURE_ALGORITHM: &str = "ed25519";

/// The integrity flag the examiner raises when it was given no training
/// material to audit for overlap: it says what was not checked and refuses
/// nothing.
pub const NO_MATERIAL_FLAG: &str = "no_training_material_to_audit";

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
    /// It is not a verdict of a schema the gate reads.
    #[error("the verdict's schema is {found:?}, not {EXAMINATION_SCHEMA} or {AGENT_SCHEMA}")]
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
    /// A sealed exam it is over has no burn record in it.
    #[error("the verdict over a sealed exam carries no burn record of it")]
    NoBurn,
    /// A burn record it carries is of an exam it is not over.
    #[error("the verdict's burn record is of exam {found}, not of {exam}")]
    BurnOfAnotherExam {
        /// The exams the verdict is over.
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

/// One exam a verdict is over.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Exam {
    /// Its collection's name.
    pub collection: String,
    /// Its digest.
    pub digest: String,
}

/// The burn record of a sealed exam: the pair whose verdict spent it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Burn {
    /// The exam's digest.
    pub exam: String,
    /// The pair.
    pub by: Pair,
}

/// The aggregates of a verified verdict: what the gate decides on.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Aggregate {
    /// The schema it was read from.
    pub schema: String,
    /// The comparison it is of: the collection of a direct-completion
    /// verdict, the section (`overall`, `retention`) of an agent-protocol one.
    pub collection: String,
    /// The exams it is over: the one exam of a direct-completion verdict, the
    /// sealed exams of an agent-protocol one.
    pub exams: Vec<Exam>,
    /// Whether the material is sealed exam material.
    pub sealed: bool,
    /// Items compared.
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
    /// The burn records it carries, one per sealed exam.
    pub burns: Vec<Burn>,
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

    /// Accepted only when it is over sealed exams this pair spent: a burn
    /// record of every exam, each naming this pair.
    ///
    /// # Errors
    /// Public material, an exam without a burn record, or an exam another
    /// pair spent first.
    pub fn fresh_slice(&self) -> Result<(), Rejection> {
        let aggregate = &self.aggregate;
        if !aggregate.sealed || aggregate.exams.is_empty() {
            return Err(Rejection::NotSealed);
        }
        for exam in &aggregate.exams {
            let Some(burn) = aggregate.burns.iter().find(|b| b.exam == exam.digest) else {
                return Err(Rejection::NoBurn);
            };
            if burn.by != self.pair {
                return Err(Rejection::Spent {
                    candidate: burn.by.candidate.clone(),
                    predecessor: burn.by.predecessor.clone(),
                });
            }
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

fn strings(doc: &Value, pointer: &str) -> Vec<String> {
    doc.pointer(pointer)
        .and_then(Value::as_array)
        .map(|r| {
            r.iter()
                .filter_map(|x| x.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

fn interval(doc: &Value, pointer: &str) -> Result<Option<(f64, f64)>, Rejection> {
    match doc.pointer(pointer) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Array(ends)) if ends.len() == 2 => Ok(Some((
            ends[0].as_f64().ok_or_else(|| malformed(pointer))?,
            ends[1].as_f64().ok_or_else(|| malformed(pointer))?,
        ))),
        Some(_) => Err(malformed(pointer)),
    }
}

fn burn_of(doc: &Value, pointer: &str) -> Result<Burn, Rejection> {
    Ok(Burn {
        exam: text(doc, &format!("{pointer}/exam_sha256"))?
            .ok_or_else(|| malformed(&format!("{pointer}/exam_sha256")))?,
        by: Pair {
            candidate: text(doc, &format!("{pointer}/candidate"))?,
            predecessor: text(doc, &format!("{pointer}/predecessor"))?,
        },
    })
}

/// The signature check alone: an Ed25519 signature under `key` over the
/// canonical bytes of `document` without its `signature`. [`verify`] runs it
/// first; a caller that decides on the examiner's own outcome (a verified
/// Promote) and not on the gate's rules runs only this.
///
/// # Errors
/// No signature of the examiner's algorithm, or one that does not verify.
pub fn verify_signature(document: &Value, key: &PublicKey) -> Result<(), Rejection> {
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
    let mut body = document.as_object().ok_or(Rejection::BadSignature)?.clone();
    body.remove("signature");
    let bytes = canonical_bytes(&Value::Object(body)).ok_or(Rejection::BadSignature)?;
    if key.verifies(&bytes, value) {
        Ok(())
    } else {
        Err(Rejection::BadSignature)
    }
}

/// `document` checked as the examiner's verdict on `expected`, at the
/// examiner's level `alpha`, read for `role`: the signature under `key`, the
/// schema, the examiner's own validity and integrity flags, the pair, the
/// level.
///
/// # Errors
/// The first check that fails, typed.
pub fn verify(
    document: &Value,
    key: &PublicKey,
    expected: &Pair,
    alpha: f64,
    role: Role,
) -> Result<Verified, Rejection> {
    verify_signature(document, key)?;
    let schema = text(document, "/schema")?;
    let (found, aggregate) = match schema.as_deref() {
        Some(EXAMINATION_SCHEMA) => examination(document)?,
        Some(AGENT_SCHEMA) => agentic(document, role)?,
        _ => return Err(Rejection::Schema { found: schema }),
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
    if (aggregate.alpha - alpha).abs() > 1e-12 {
        return Err(Rejection::Level {
            expected: alpha,
            found: aggregate.alpha,
        });
    }
    let listed: Vec<&str> = aggregate.exams.iter().map(|e| e.digest.as_str()).collect();
    if let Some(stray) = aggregate
        .burns
        .iter()
        .find(|b| !listed.contains(&b.exam.as_str()))
    {
        return Err(Rejection::BurnOfAnotherExam {
            exam: listed.join(", "),
            found: stray.exam.clone(),
        });
    }
    Ok(Verified {
        pair: found,
        aggregate,
    })
}

/// The pair and aggregates of a direct-completion verdict.
fn examination(document: &Value) -> Result<(Pair, Aggregate), Rejection> {
    schema_checks(document, "/valid", "/invalid_reasons")?;
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
    let pair = Pair {
        candidate: text(document, "/candidate/adapter_digest")?,
        predecessor: text(document, "/predecessor/adapter_digest")?,
    };
    let exam = text(document, "/collection/exam_sha256")?
        .ok_or_else(|| malformed("/collection/exam_sha256"))?;
    let collection = text(document, "/collection/name")?.unwrap_or_default();
    let burns = match document.get("burn") {
        None | Some(Value::Null) => Vec::new(),
        Some(_) => vec![burn_of(document, "/burn")?],
    };
    let aggregate = Aggregate {
        schema: EXAMINATION_SCHEMA.into(),
        exams: vec![Exam {
            collection: collection.clone(),
            digest: exam,
        }],
        collection,
        sealed: document
            .pointer("/collection/sealed")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        items: document
            .pointer("/collection/items")
            .and_then(Value::as_u64)
            .ok_or_else(|| malformed("/collection/items"))?,
        alpha: number(document, "/alpha")?.ok_or_else(|| malformed("/alpha"))?,
        gain: number(document, "/paired_gain/mean")?,
        interval: interval(document, "/paired_gain/interval")?,
        one_sided_lower: number(document, "/paired_gain/one_sided_lower")?,
        families: document
            .pointer("/paired_gain/families")
            .and_then(Value::as_u64),
        burns,
    };
    Ok((pair, aggregate))
}

/// The examiner's own verdict on the measurement: `valid` true, else its
/// reasons.
fn schema_checks(document: &Value, valid: &str, reasons: &str) -> Result<(), Rejection> {
    if document.pointer(valid).and_then(Value::as_bool) != Some(true) {
        return Err(Rejection::Invalid {
            reasons: strings(document, reasons),
        });
    }
    Ok(())
}

/// The pair and aggregates of an agent-protocol verdict, read for `role`.
fn agentic(document: &Value, role: Role) -> Result<(Pair, Aggregate), Rejection> {
    let outcome = text(document, "/verdict")?.ok_or_else(|| malformed("/verdict"))?;
    if outcome == "Invalid" {
        return Err(Rejection::Invalid {
            reasons: strings(document, "/reasons"),
        });
    }
    schema_checks(document, "/integrity/ok", "/reasons")?;
    if text(document, "/role")?.as_deref() != Some("candidate") {
        return Err(Rejection::Invalid {
            reasons: vec!["the verdict is of a control arm, which is never promoted".into()],
        });
    }
    if !document.get("candidate").is_some_and(Value::is_object) {
        return Err(malformed("/candidate"));
    }
    let pair = Pair {
        candidate: text(document, "/candidate/adapter_digest")?,
        predecessor: text(document, "/predecessor/adapter_digest")?,
    };
    let exams = document
        .get("sealed_exams")
        .and_then(Value::as_array)
        .ok_or_else(|| malformed("/sealed_exams"))?
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let at = format!("/sealed_exams/{i}");
            Ok(Exam {
                collection: text(document, &format!("{at}/collection"))?.unwrap_or_default(),
                digest: text(document, &format!("{at}/exam_sha256"))?
                    .ok_or_else(|| malformed(&format!("{at}/exam_sha256")))?,
            })
        })
        .collect::<Result<Vec<_>, Rejection>>()?;
    let burns = match document.get("burns") {
        Some(Value::Array(records)) => (0..records.len())
            .map(|i| burn_of(document, &format!("/burns/{i}")))
            .collect::<Result<Vec<_>, Rejection>>()?,
        // Issued without the examiner's burn store: it spent nothing.
        None | Some(Value::Null) => Vec::new(),
        Some(_) => return Err(malformed("/burns")),
    };
    let section = if role == Role::Retention {
        "retention"
    } else {
        "overall"
    };
    let at = |field: &str| format!("/{section}/{field}");
    let aggregate = Aggregate {
        schema: AGENT_SCHEMA.into(),
        collection: section.into(),
        sealed: !exams.is_empty(),
        exams,
        items: document
            .pointer(&at("items"))
            .and_then(Value::as_u64)
            .ok_or_else(|| malformed(&at("items")))?,
        alpha: number(document, "/policy/alpha")?.ok_or_else(|| malformed("/policy/alpha"))?,
        gain: number(document, &at("delta"))?,
        interval: interval(document, &at("interval"))?,
        one_sided_lower: number(document, &at("one_sided_lower"))?,
        families: document.pointer(&at("families")).and_then(Value::as_u64),
        burns,
    };
    Ok((pair, aggregate))
}
