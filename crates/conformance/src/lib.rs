// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for proving a learning system's
// algorithms against independent reference implementations, for its
// clients. If your team needs expertise in conformance testing of data and
// statistics pipelines then you can procure our services by sending an email
// to info@swedishembedded.com.

//! Conformance test support: the one way a test reads a reference fixture
//! and compares Splinter's numbers with it.
//!
//! A fixture is produced by an oracle (`validation/oracles/<area>/<name>.py`)
//! that runs a pinned external reference (`validation/references.lock.yaml`)
//! and is frozen in `validation/fixtures/`; `validation/fixtures.yaml` records
//! each fixture's digest, and the digest of the oracle that wrote it. A test
//! never regenerates a fixture: [`fixture`] refuses one whose bytes differ from
//! the digest recorded (a changed expected value), one the manifest does not
//! list, one that is missing, and one that names another fixture.
//! `scripts/conformance/external.py check` regenerates every fixture with the
//! pinned references and refuses any that no longer agree.
//!
//! Differential mode runs the same test against cases an oracle drew just
//! now: built with [`OVERRIDE_VAR`] naming a directory (a build setting,
//! which `external.py differential` sets for its `cargo test`), [`fixture`] reads
//! `<dir>/<name>.json` instead, which must say it was drawn
//! (`"differential": true`); no digest binds it, since it was never frozen.
//!
//! [`compare`] compares typed JSON (object key order never matters, array
//! order and length always do) and numbers within a [`Tolerance`]; a
//! disagreement names the first coordinate and the largest errors.

pub mod fuzz;

use std::fmt;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest as _, Sha256};

/// The manifest of frozen fixtures, relative to the repository root.
pub const MANIFEST: &str = "validation/fixtures.yaml";
/// Its schema.
pub const MANIFEST_SCHEMA: &str = "splinter.validation-fixtures/v1";
/// The build-time variable naming a directory of freshly drawn fixtures
/// (differential mode).
pub const OVERRIDE_VAR: &str = "SPLINTER_FIXTURE_DIR";

/// One entry of the manifest.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    /// The fixture, relative to the repository root.
    pub path: PathBuf,
    /// The SHA-256 of its bytes, hex.
    pub sha256: String,
    /// The oracle that wrote it.
    pub oracle: PathBuf,
    /// The SHA-256 of the oracle's source, hex.
    pub oracle_sha256: String,
    /// The references (ids of the references lock) it ran.
    pub references: Vec<String>,
    /// The test that reads it, `package/target/filter`.
    pub test: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: String,
    fixtures: std::collections::BTreeMap<String, Entry>,
}

/// Where a fixture came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// The frozen, reviewed fixture.
    Frozen,
    /// Cases an oracle drew for this run.
    Differential,
}

/// A fixture as read.
#[derive(Clone, Debug)]
pub struct Fixture {
    /// Its name (the manifest's key).
    pub name: String,
    /// Its content.
    pub value: Value,
    /// Where it came from.
    pub source: Source,
}

impl Fixture {
    /// The fixture's `cases`: every fixture is a list of cases.
    ///
    /// # Panics
    /// The fixture has no `cases` list, or it is empty (a check that compares
    /// nothing certifies nothing).
    #[must_use]
    pub fn cases(&self) -> &[Value] {
        let cases = self.value["cases"]
            .as_array()
            .unwrap_or_else(|| panic!("fixture {}: no `cases` list", self.name));
        assert!(!cases.is_empty(), "fixture {}: no cases", self.name);
        cases
    }
}

/// Why a fixture is refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FixtureError {
    /// No manifest under the repository root.
    NoManifest(PathBuf),
    /// The manifest does not parse, or is not one.
    Manifest(String),
    /// The manifest does not list the fixture.
    Unlisted(String),
    /// The fixture file is missing or unreadable.
    Missing(PathBuf),
    /// Its bytes differ from the digest recorded.
    Changed {
        /// The fixture.
        name: String,
        /// The digest recorded.
        recorded: String,
        /// The digest of the bytes found.
        found: String,
    },
    /// It is not JSON, does not name itself, or a drawn one does not say so.
    Malformed(String),
}

impl fmt::Display for FixtureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoManifest(root) => write!(f, "no {MANIFEST} under {}", root.display()),
            Self::Manifest(why) => write!(f, "{MANIFEST}: {why}"),
            Self::Unlisted(name) => write!(f, "fixture {name} is not listed in {MANIFEST}"),
            Self::Missing(path) => write!(f, "fixture {} is missing", path.display()),
            Self::Changed {
                name,
                recorded,
                found,
            } => write!(
                f,
                "fixture {name} changed: {MANIFEST} records sha256 {recorded}, the file is {found} \
                 (regenerate it with scripts/conformance/external.py generate, and review the diff)"
            ),
            Self::Malformed(why) => write!(f, "{why}"),
        }
    }
}

impl std::error::Error for FixtureError {}

/// The repository root: the first ancestor of `start` holding [`MANIFEST`].
#[must_use]
pub fn repository_root(start: &Path) -> Option<PathBuf> {
    start
        .ancestors()
        .find(|dir| dir.join(MANIFEST).is_file())
        .map(Path::to_path_buf)
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The fixture `name`, read for the test of the crate at `manifest_dir`
/// (pass `env!("CARGO_MANIFEST_DIR")`): from the directory [`OVERRIDE_VAR`]
/// named when the tests were built (differential mode), else frozen and
/// checked.
///
/// # Errors
/// See [`FixtureError`].
pub fn fixture(manifest_dir: &str, name: &str) -> Result<Fixture, FixtureError> {
    // A build setting, not a process one: cargo rebuilds the tests when it
    // changes, and no library reads the environment at run time.
    let drawn = option_env!("SPLINTER_FIXTURE_DIR").map(PathBuf::from);
    let start = Path::new(manifest_dir);
    let root = repository_root(start).ok_or_else(|| FixtureError::NoManifest(start.into()))?;
    fixture_in(&root, name, drawn.as_deref())
}

/// [`fixture`] against the repository at `root`, drawn cases read from
/// `drawn` when given.
///
/// # Errors
/// See [`FixtureError`].
pub fn fixture_in(root: &Path, name: &str, drawn: Option<&Path>) -> Result<Fixture, FixtureError> {
    let text = std::fs::read_to_string(root.join(MANIFEST))
        .map_err(|_| FixtureError::NoManifest(root.into()))?;
    let manifest: Manifest =
        serde_yaml::from_str(&text).map_err(|e| FixtureError::Manifest(e.to_string()))?;
    if manifest.schema != MANIFEST_SCHEMA {
        return Err(FixtureError::Manifest(format!(
            "schema {}, not {MANIFEST_SCHEMA}",
            manifest.schema
        )));
    }
    let entry = manifest
        .fixtures
        .get(name)
        .ok_or_else(|| FixtureError::Unlisted(name.into()))?;
    let (path, source) = match drawn {
        Some(dir) => (
            dir.join(format!("{}.json", name.replace('/', "_"))),
            Source::Differential,
        ),
        None => (root.join(&entry.path), Source::Frozen),
    };
    let bytes = std::fs::read(&path).map_err(|_| FixtureError::Missing(path.clone()))?;
    if source == Source::Frozen {
        let found = sha256_hex(&bytes);
        if found != entry.sha256 {
            return Err(FixtureError::Changed {
                name: name.into(),
                recorded: entry.sha256.clone(),
                found,
            });
        }
    }
    let value: Value = serde_json::from_slice(&bytes)
        .map_err(|e| FixtureError::Malformed(format!("fixture {name}: {e}")))?;
    if value["fixture"] != name {
        return Err(FixtureError::Malformed(format!(
            "{} names fixture {}, not {name}",
            path.display(),
            value["fixture"]
        )));
    }
    if (source == Source::Differential) != (value["differential"] == true) {
        return Err(FixtureError::Malformed(format!(
            "{}: a frozen fixture is never drawn, and a drawn one says so (`differential`)",
            path.display()
        )));
    }
    Ok(Fixture {
        name: name.into(),
        value,
        source,
    })
}

/// How far a number may be from its reference: within `abs + rel * |reference|`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tolerance {
    /// Absolute error allowed.
    pub abs: f64,
    /// Error allowed relative to the reference's magnitude.
    pub rel: f64,
}

impl Tolerance {
    /// Exact agreement.
    pub const EXACT: Self = Self { abs: 0.0, rel: 0.0 };

    /// Within `abs + rel * |reference|`.
    #[must_use]
    pub const fn new(abs: f64, rel: f64) -> Self {
        Self { abs, rel }
    }

    fn admits(self, actual: f64, reference: f64) -> bool {
        // Two infinities of one sign agree; NaN never does (a fixture never
        // holds one, so an actual NaN is a defect).
        if actual.is_infinite() || reference.is_infinite() {
            return actual == reference;
        }
        (actual - reference).abs() <= self.abs + self.rel * reference.abs()
    }
}

/// What a comparison found when it agreed.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Agreement {
    /// Numbers compared.
    pub numbers: usize,
    /// The largest absolute error.
    pub max_abs: f64,
    /// The largest error relative to the reference.
    pub max_rel: f64,
}

/// A disagreement: the first coordinate that disagreed, and the largest
/// errors seen.
#[derive(Clone, Debug, PartialEq)]
pub struct Mismatch {
    /// Where (`$.cases[3].p`).
    pub at: String,
    /// What differed.
    pub what: String,
    /// The errors up to and including it.
    pub seen: Agreement,
}

impl fmt::Display for Mismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} (after {} numbers, max abs error {:e}, max rel error {:e})",
            self.at, self.what, self.seen.numbers, self.seen.max_abs, self.seen.max_rel
        )
    }
}

impl std::error::Error for Mismatch {}

/// Compares `actual` with `reference`: the same shape (object keys, array
/// lengths, types) and every number within `tolerance`.
///
/// # Errors
/// The first disagreement.
pub fn compare(
    actual: &Value,
    reference: &Value,
    tolerance: Tolerance,
) -> Result<Agreement, Mismatch> {
    let mut seen = Agreement::default();
    walk("$", actual, reference, tolerance, &mut seen).map_err(|(at, what)| Mismatch {
        at,
        what,
        seen: seen.clone(),
    })?;
    Ok(seen)
}

/// [`compare`] over two lists of numbers, a non-finite actual value
/// refused (JSON cannot carry one, so it would otherwise vanish as null).
///
/// # Errors
/// The first disagreement.
pub fn compare_numbers(
    actual: &[f64],
    reference: &[f64],
    tolerance: Tolerance,
) -> Result<Agreement, Mismatch> {
    if let Some(i) = actual.iter().position(|x| x.is_nan()) {
        return Err(Mismatch {
            at: format!("$[{i}]"),
            what: "NaN".into(),
            seen: Agreement::default(),
        });
    }
    let to_json = |xs: &[f64]| Value::Array(xs.iter().map(|x| number(*x)).collect());
    compare(&to_json(actual), &to_json(reference), tolerance)
}

/// `x` as JSON; an infinity as the string `inf`/`-inf`, which is how an
/// oracle writes one.
fn number(x: f64) -> Value {
    if x.is_infinite() {
        Value::String(if x > 0.0 { "inf" } else { "-inf" }.into())
    } else {
        serde_json::Number::from_f64(x).map_or(Value::Null, Value::Number)
    }
}

fn as_number(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) if s == "inf" => Some(f64::INFINITY),
        Value::String(s) if s == "-inf" => Some(f64::NEG_INFINITY),
        _ => None,
    }
}

fn kind(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "a list",
        Value::Object(_) => "an object",
    }
}

fn walk(
    at: &str,
    actual: &Value,
    reference: &Value,
    tolerance: Tolerance,
    seen: &mut Agreement,
) -> Result<(), (String, String)> {
    match (actual, reference) {
        (Value::Object(a), Value::Object(r)) => {
            if let Some(key) = r.keys().find(|k| !a.contains_key(*k)) {
                return Err((format!("{at}.{key}"), "missing".into()));
            }
            if let Some(key) = a.keys().find(|k| !r.contains_key(*k)) {
                return Err((format!("{at}.{key}"), "not in the reference".into()));
            }
            for (key, r) in r {
                walk(&format!("{at}.{key}"), &a[key], r, tolerance, seen)?;
            }
            Ok(())
        }
        (Value::Array(a), Value::Array(r)) => {
            if a.len() != r.len() {
                return Err((
                    at.into(),
                    format!("{} items, the reference has {}", a.len(), r.len()),
                ));
            }
            for (i, (a, r)) in a.iter().zip(r).enumerate() {
                walk(&format!("{at}[{i}]"), a, r, tolerance, seen)?;
            }
            Ok(())
        }
        _ => match (as_number(actual), as_number(reference)) {
            (Some(a), Some(r)) => {
                seen.numbers += 1;
                let abs = if a == r { 0.0 } else { (a - r).abs() };
                let rel = if r == 0.0 || abs == 0.0 {
                    abs
                } else {
                    abs / r.abs()
                };
                seen.max_abs = seen.max_abs.max(abs);
                seen.max_rel = seen.max_rel.max(rel);
                if tolerance.admits(a, r) {
                    Ok(())
                } else {
                    Err((at.into(), format!("{a:?}, the reference {r:?}")))
                }
            }
            _ if actual == reference => Ok(()),
            _ if kind(actual) != kind(reference) => Err((
                at.into(),
                format!("{}, the reference {}", kind(actual), kind(reference)),
            )),
            _ => Err((at.into(), format!("{actual}, the reference {reference}"))),
        },
    }
}
