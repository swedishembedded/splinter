// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model-driven knowledge extraction whose every
// claim is traceable to its source, for its clients. If your team needs
// expertise in grounded rule extraction from legal or procedural documents,
// you can procure our services by sending an email to info@swedishembedded.com.

//! Typed knowledge objects proposed by a model and admitted by code.
//!
//! A [`ProposedObject`] is what a model says about a document: a rule, a
//! threshold, a deadline, an exception, a computation, a procedure step, a
//! fact or a behaviour trait, each with the verbatim quote it rests on.
//! [`admit`] decides which proposals become an [`Object`] of a [`Pack`],
//! and nothing a model says about its own proposal is trusted:
//!
//! 1. the kind is known and the document exists;
//! 2. the quote occurs verbatim in the document, which gives the object its
//!    span (offsets in characters, so any reader of the pack agrees);
//! 3. the data has the shape of its kind (required keys, known operators,
//!    declared variables, well-formed conditions and expressions);
//! 4. every number the data states (a threshold's value, an offset, a
//!    constant, a condition's literal) occurs in the quote; `0` and `1` are
//!    structure, and so is anything under `order`, `vars`, `terms`, `limit`;
//! 5. every listed term occurs in the quote, and prose the object carries
//!    (`reveals`, `element`) has most of its content words in the quote.
//!
//! Where admitted objects state the same thing differently (the same
//! variable and outcome with another value, the same event with another
//! offset) the documents' headers decide, and the decision is recorded as a
//! [`Resolution`]: higher `authority` wins, then the later `date`, then the
//! higher `version`. An object whose data names `supersedes` is an amendment
//! and replaces the object it names, when its document is not earlier or of
//! lower authority. Objects no header can separate stay [`Flag`]ged as a
//! conflict. Nothing is merged: a superseded object keeps its provenance and
//! is left out of every ruleset compiled from the pack.
//!
//! The pack is a plain JSON document (`material-pack/2`); the consumer
//! that compiles it into rule sets and tasks re-runs the same admission on
//! what it reads.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::tasks::grounding::content_words;

mod aligned;
pub mod propose;

pub use aligned::resolve_aligned;

/// The pack format's name.
pub const PACK_SCHEMA: &str = "material-pack/2";
/// What a document with no header is believed at, against another.
pub const NEUTRAL_AUTHORITY: u8 = 2;
/// Numbers that are structure, not a claim about the source.
const STRUCTURAL_NUMBERS: [&str; 2] = ["0", "1"];
/// Data keys whose numbers are structure.
const UNTRACED_KEYS: [&str; 5] = ["order", "vars", "terms", "limit", "supersedes"];
/// The share of a prose field's content words the quote must carry.
const MIN_PROSE_SUPPORT: f64 = 0.6;
const OPERATORS: [&str; 6] = ["<", "<=", ">", ">=", "==", "!="];
const ARITHMETIC: [&str; 7] = ["add", "sub", "mul", "div", "min", "max", "pct"];
const VARIABLE_TYPES: [&str; 4] = ["int", "number", "bool", "date"];

/// What a document says about itself.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    /// How much it is believed against another document.
    pub authority: u8,
    /// Its version, 0 when not stated.
    pub version: u32,
    /// ISO date, when stated.
    pub date: Option<String>,
}

impl Default for Provenance {
    fn default() -> Self {
        Self {
            authority: NEUTRAL_AUTHORITY,
            version: 0,
            date: None,
        }
    }
}

impl Provenance {
    fn rank(&self) -> (u8, &str, u32) {
        (
            self.authority,
            self.date.as_deref().unwrap_or(""),
            self.version,
        )
    }
}

/// A source document.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Document {
    /// Stable id the objects name.
    pub id: String,
    /// Display title.
    pub title: String,
    /// The text quotes are found in (the body, without the header).
    pub text: String,
    /// What it says about itself.
    #[serde(default)]
    pub provenance: Provenance,
}

impl Document {
    /// A document file: an optional `---` header (`authority`, `version`, `date`), then the body.
    #[must_use]
    pub fn parse(id: &str, raw: &str) -> Self {
        let mut provenance = Provenance::default();
        let mut body = raw;
        if let Some((head, rest)) = raw
            .strip_prefix("---\n")
            .and_then(|r| r.split_once("\n---\n"))
        {
            for line in head.lines() {
                let Some((key, value)) = line.split_once(':') else {
                    continue;
                };
                let value = value.trim();
                match key.trim() {
                    "authority" => {
                        provenance.authority = value.parse().unwrap_or(NEUTRAL_AUTHORITY)
                    }
                    "version" => provenance.version = value.parse().unwrap_or(0),
                    "date" => provenance.date = Some(value.to_string()),
                    _ => {}
                }
            }
            body = rest;
        }
        let title = body
            .lines()
            .find(|l| !l.trim().is_empty())
            .map_or(id, |l| l.trim_start_matches(['#', ' ']).trim())
            .to_string();
        Self {
            id: id.to_string(),
            title,
            text: body.to_string(),
            provenance,
        }
    }
}

/// What a model proposes.
#[derive(Clone, Debug, Deserialize, Serialize, schemars::JsonSchema)]
pub struct ProposedObject {
    /// The proposer's id for the object; unique within a pack.
    pub id: String,
    /// One of the kinds in [`required_keys`].
    pub kind: String,
    /// The document the quote is from.
    pub document: String,
    /// The verbatim passage the object rests on.
    pub quote: String,
    /// The typed content.
    pub data: Value,
}

/// A proposal that passed admission.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Object {
    /// The proposal's id.
    pub id: String,
    /// The kind.
    pub kind: String,
    /// The document.
    pub document: String,
    /// `[start, end)` of the quote in the document, in characters.
    pub span: [usize; 2],
    /// The quote.
    pub quote: String,
    /// The content.
    pub data: Value,
}

/// A decision between objects that stated the same thing differently, or an amendment.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Resolution {
    /// `amendment` or `conflict`.
    pub kind: String,
    /// The object that stands.
    pub winner: String,
    /// The objects it replaces.
    pub superseded: Vec<String>,
    /// What decided it against the strongest competitor: `amendment`, `authority`, `date` or `version`.
    pub basis: String,
    /// What decided it against each replaced object.
    pub bases: BTreeMap<String, String>,
    /// What was in dispute.
    pub detail: String,
}

/// Two or more objects that state the same thing differently, and that no header separates.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Flag {
    /// Always `conflict`.
    pub kind: String,
    /// The ids involved.
    pub objects: Vec<String>,
    /// What is stated differently.
    pub reason: String,
}

/// A proposal that did not pass, and why.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rejected {
    /// The proposal's id.
    pub id: String,
    /// Why.
    pub reason: String,
}

/// Admitted objects with their documents, decisions, conflicts and rejections.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pack {
    /// [`PACK_SCHEMA`].
    pub schema: String,
    /// Names the pack.
    pub pack_id: String,
    /// The sources.
    pub documents: Vec<Document>,
    /// The admitted objects, in proposal order.
    pub objects: Vec<Object>,
    /// Which object stands where documents disagree or amend.
    pub resolutions: Vec<Resolution>,
    /// Conflicts no header decides.
    pub flags: Vec<Flag>,
    /// The proposals that were refused.
    pub rejected: Vec<Rejected>,
}

/// The keys a kind's data must have; `None` for an unknown kind.
#[must_use]
pub fn required_keys(kind: &str) -> Option<&'static [&'static str]> {
    Some(match kind {
        "threshold" => &["var", "op", "value", "outcome"],
        "rule" => &["when", "outcome"],
        "exception" => &["overrides", "when", "outcome"],
        "computation" => &["name", "when", "expr"],
        "deadline" => &["event", "from", "offset", "unit"],
        "procedure_step" => &["order", "element"],
        "fact" => &["key", "value"],
        "trait" => &["trait", "topic", "keywords", "reveals"],
        "definition" => &["term", "meaning"],
        "table" => &["name", "columns", "rows"],
        _ => return None,
    })
}

/// Admits `proposals` over `documents` into a pack.
#[must_use]
pub fn admit(pack_id: &str, documents: &[Document], proposals: &[ProposedObject]) -> Pack {
    let mut objects: Vec<Object> = Vec::new();
    let mut rejected = Vec::new();
    for proposal in proposals {
        let outcome = if objects.iter().any(|o| o.id == proposal.id) {
            Err("duplicate id".to_string())
        } else {
            admit_one(proposal, documents, &objects)
        };
        match outcome {
            Ok(object) => objects.push(object),
            Err(reason) => rejected.push(Rejected {
                id: proposal.id.clone(),
                reason,
            }),
        }
    }
    let (resolutions, flags) = resolve(&objects, documents);
    Pack {
        schema: PACK_SCHEMA.to_string(),
        pack_id: pack_id.to_string(),
        documents: documents.to_vec(),
        objects,
        resolutions,
        flags,
        rejected,
    }
}

fn provenance_of<'a>(documents: &'a [Document], id: &str) -> Option<&'a Provenance> {
    documents.iter().find(|d| d.id == id).map(|d| &d.provenance)
}

fn admit_one(
    proposal: &ProposedObject,
    documents: &[Document],
    earlier: &[Object],
) -> Result<Object, String> {
    let keys =
        required_keys(&proposal.kind).ok_or_else(|| format!("unknown kind {:?}", proposal.kind))?;
    let document = documents
        .iter()
        .find(|d| d.id == proposal.document)
        .ok_or_else(|| format!("unknown document {:?}", proposal.document))?;
    let byte_start = if proposal.quote.is_empty() {
        None
    } else {
        document.text.find(&proposal.quote)
    };
    let byte_start = byte_start.ok_or("quote is not in the document")?;
    let start = document.text[..byte_start].chars().count();
    let data = proposal.data.as_object().ok_or("data is not an object")?;
    let missing: Vec<&str> = keys
        .iter()
        .copied()
        .filter(|k| !data.contains_key(*k))
        .collect();
    if !missing.is_empty() {
        return Err(format!("missing {missing:?}"));
    }
    check_shape(&proposal.kind, data)?;
    check_identifiers(data)?;
    let allowed = quote_numbers(&proposal.quote);
    let mut untraced: BTreeSet<String> = BTreeSet::new();
    claimed_numbers(&proposal.data, None, &mut |n| {
        if !allowed.contains(&n) && !STRUCTURAL_NUMBERS.contains(&n.as_str()) {
            untraced.insert(n);
        }
    });
    if !untraced.is_empty() {
        let listed: Vec<String> = untraced.iter().map(|n| format!("'{n}'")).collect();
        return Err(format!(
            "numbers [{}] are not in the quote",
            listed.join(", ")
        ));
    }
    let quote_lower = proposal.quote.to_lowercase();
    if let Some(terms) = data.get("terms").and_then(Value::as_array) {
        let absent: Vec<&str> = terms
            .iter()
            .filter_map(Value::as_str)
            .filter(|t| !quote_lower.contains(&t.to_lowercase()))
            .collect();
        if !absent.is_empty() {
            return Err(format!("terms {absent:?} are not in the quote"));
        }
    }
    if proposal.kind == "fact" {
        if !data["value"]
            .as_str()
            .is_some_and(|v| quote_lower.contains(&v.to_lowercase()))
        {
            return Err("the value is not in the quote".to_string());
        }
        let key = data["key"].as_str().unwrap_or_default();
        let parts: Vec<&str> = key.split('|').map(str::trim).collect();
        if parts.len() != 2 || parts.iter().any(|p| p.is_empty()) {
            return Err("a fact key is 'attribute | entity'".to_string());
        }
        let in_quote = content_words(&proposal.quote);
        let grounded = |part: &str| {
            let words = content_words(part);
            words.is_empty()
                || (words.intersection(&in_quote).count() as f64) / (words.len() as f64) >= 0.8
        };
        if !parts.iter().all(|p| grounded(p)) {
            return Err("the key is not grounded in the quote".to_string());
        }
    }
    if proposal.kind == "table" {
        let absent: Vec<String> = data["rows"]
            .as_array()
            .into_iter()
            .flatten()
            .flat_map(|r| r.as_array().into_iter().flatten())
            .map(|c| c.as_str().map_or_else(|| c.to_string(), str::to_string))
            .filter(|c| !quote_lower.contains(&c.to_lowercase()))
            .collect();
        if !absent.is_empty() {
            return Err(format!("cells {absent:?} are not in the quote"));
        }
    }
    let quote_words = content_words(&proposal.quote);
    for key in ["reveals", "element", "meaning"] {
        if let Some(prose) = data.get(key).and_then(Value::as_str) {
            let words = content_words(prose);
            if !words.is_empty()
                && (words.intersection(&quote_words).count() as f64) / (words.len() as f64)
                    < MIN_PROSE_SUPPORT
            {
                return Err(format!("{key} is not supported by the quote"));
            }
        }
    }
    if let Some(target) = data.get("supersedes").and_then(Value::as_str) {
        let old = earlier.iter().find(|o| o.id == target).ok_or_else(|| {
            format!("supersedes {target:?}, which is not an admitted object before it")
        })?;
        if old.kind != proposal.kind {
            return Err("an amendment is of the same kind as what it replaces".to_string());
        }
        let (new_rank, old_rank) = (
            provenance_of(documents, &proposal.document),
            provenance_of(documents, &old.document),
        );
        if new_rank
            .zip(old_rank)
            .is_none_or(|(n, o)| n.rank() <= o.rank())
        {
            return Err(
                "an amendment must come from a later or more authoritative document".to_string(),
            );
        }
    }
    Ok(Object {
        id: proposal.id.clone(),
        kind: proposal.kind.clone(),
        document: proposal.document.clone(),
        span: [start, start + proposal.quote.chars().count()],
        quote: proposal.quote.clone(),
        data: proposal.data.clone(),
    })
}

fn is_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    chars.next().is_some_and(|c| c.is_ascii_lowercase())
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

/// Names the ruleset is built from are identifiers, not sentences: the vocabulary step's check, after normalisation has had its say.
fn check_identifiers(data: &serde_json::Map<String, Value>) -> Result<(), String> {
    let mut names: Vec<String> = data
        .get("vars")
        .and_then(Value::as_object)
        .map(|v| v.keys().cloned().collect())
        .unwrap_or_default();
    for key in ["var", "event", "from", "outcome"] {
        if let Some(name) = data.get(key).and_then(Value::as_str) {
            names.push(name.to_string());
        }
    }
    let bad: Vec<String> = names.into_iter().filter(|n| !is_identifier(n)).collect();
    if bad.is_empty() {
        return Ok(());
    }
    let listed: Vec<String> = bad.iter().map(|n| format!("'{n}'")).collect();
    Err(format!(
        "names [{}] are not identifiers (lower-case words joined by underscores)",
        listed.join(", ")
    ))
}

fn check_shape(kind: &str, data: &serde_json::Map<String, Value>) -> Result<(), String> {
    let mut declared: BTreeSet<String> = BTreeSet::new();
    if let Some(vars) = data.get("vars").and_then(Value::as_object) {
        for (name, ty) in vars {
            if !ty.as_str().is_some_and(|t| VARIABLE_TYPES.contains(&t)) {
                return Err("unknown variable type".to_string());
            }
            declared.insert(name.clone());
        }
    }
    match kind {
        "threshold" => {
            let var = data["var"]
                .as_str()
                .ok_or("threshold var is not a string")?;
            declared.insert(var.to_string());
            check_condition(
                &serde_json::json!({"var": data["var"], "op": data["op"], "value": data["value"]}),
                &declared,
            )
        }
        "rule" | "exception" => check_condition(&data["when"], &declared),
        "computation" => {
            check_condition(&data["when"], &declared)?;
            check_expression(&data["expr"], &declared)
        }
        "deadline" => {
            let unit_ok = matches!(
                data["unit"].as_str(),
                Some("calendar_days" | "business_days")
            );
            if !unit_ok || !data["offset"].is_i64() {
                return Err("bad deadline".to_string());
            }
            Ok(())
        }
        "trait" => {
            let keywords = data["keywords"]
                .as_array()
                .filter(|k| !k.is_empty() && k.iter().all(Value::is_string));
            keywords
                .map(|_| ())
                .ok_or_else(|| "a trait needs keywords".to_string())
        }
        "table" => {
            let width = data["columns"].as_array().map_or(0, Vec::len);
            let rows = data["rows"].as_array().filter(|r| !r.is_empty());
            let ok = width > 0
                && rows.is_some_and(|rows| {
                    rows.iter()
                        .all(|r| r.as_array().is_some_and(|r| r.len() == width))
                });
            if ok {
                Ok(())
            } else {
                Err("a table needs columns and rows of the same width".to_string())
            }
        }
        "fact" => {
            let text = |k: &str| data[k].as_str().is_some_and(|s| !s.is_empty());
            if text("key") && text("value") {
                Ok(())
            } else {
                Err("a fact needs a key and a value".to_string())
            }
        }
        _ => Ok(()),
    }
}

fn check_condition(cond: &Value, declared: &BTreeSet<String>) -> Result<(), String> {
    let object = cond.as_object().ok_or("condition is not an object")?;
    if let Some(list) = object.get("all").or_else(|| object.get("any")) {
        return list
            .as_array()
            .ok_or("all/any is not a list")?
            .iter()
            .try_for_each(|c| check_condition(c, declared));
    }
    if let Some(inner) = object.get("not") {
        return check_condition(inner, declared);
    }
    let op_ok = object
        .get("op")
        .and_then(Value::as_str)
        .is_some_and(|op| OPERATORS.contains(&op));
    let var = object.get("var").and_then(Value::as_str);
    if !op_ok || var.is_none() || !object.contains_key("value") {
        return Err(format!("bad condition {cond}"));
    }
    match var {
        Some(v) if declared.contains(v) => Ok(()),
        Some(v) => Err(format!("undeclared variable {v}")),
        None => Err("bad condition".to_string()),
    }
}

fn check_expression(expr: &Value, declared: &BTreeSet<String>) -> Result<(), String> {
    let list = expr
        .as_array()
        .filter(|l| !l.is_empty())
        .ok_or("expression is not a list")?;
    match list[0].as_str() {
        Some("const") if list.len() == 2 => Ok(()),
        Some("var") if list.len() == 2 => match list[1].as_str() {
            Some(v) if declared.contains(v) => Ok(()),
            _ => Err(format!("undeclared variable {}", list[1])),
        },
        Some(op) if ARITHMETIC.contains(&op) && list.len() == 3 => {
            check_expression(&list[1], declared)?;
            check_expression(&list[2], declared)
        }
        _ => Err(format!("bad expression {expr}")),
    }
}

/// The numbers in `text` as the claims normalise them: no thousands comma, no trailing zeros.
pub(crate) fn quote_numbers(text: &str) -> BTreeSet<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut found = BTreeSet::new();
    let mut i = 0;
    while i < chars.len() {
        if !chars[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        let start = i;
        while i < chars.len()
            && (chars[i].is_ascii_digit()
                || ((chars[i] == '.' || chars[i] == ',')
                    && chars.get(i + 1).is_some_and(char::is_ascii_digit)))
        {
            i += 1;
        }
        found.insert(normalise(
            &chars[start..i].iter().collect::<String>().replace(',', "."),
        ));
    }
    found
}

fn normalise(number: &str) -> String {
    if number.contains('.') {
        number
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    } else {
        number.to_string()
    }
}

/// Every number `node` claims, except under the structural keys.
pub(crate) fn claimed_numbers(node: &Value, key: Option<&str>, sink: &mut dyn FnMut(String)) {
    if key.is_some_and(|k| UNTRACED_KEYS.contains(&k)) {
        return;
    }
    match node {
        Value::Number(n) => sink(normalise(&n.to_string())),
        Value::String(s)
            if matches!(key, Some("value" | "offset"))
                && !s.is_empty()
                && s.chars().all(|c| c.is_ascii_digit() || c == '.') =>
        {
            sink(normalise(s));
        }
        Value::String(s) if key == Some("rows") => quote_numbers(s).into_iter().for_each(sink),
        Value::Object(map) => map
            .iter()
            .for_each(|(k, v)| claimed_numbers(v, Some(k), sink)),
        Value::Array(items) => {
            if items.len() == 2 && items[0] == "const" {
                if let Some(raw) = items[1]
                    .as_str()
                    .map(str::to_string)
                    .or_else(|| items[1].as_f64().map(|f| f.to_string()))
                {
                    sink(normalise(&raw));
                }
            } else {
                items.iter().for_each(|v| claimed_numbers(v, key, sink));
            }
        }
        _ => {}
    }
}

/// What makes two objects state the same thing; objects of kinds without such a key never disagree.
fn conflict_key(object: &Object) -> Option<Vec<String>> {
    let text = |k: &str| object.data[k].as_str().unwrap_or_default().to_string();
    match object.kind.as_str() {
        "threshold" => Some(vec!["threshold".into(), text("var"), text("outcome")]),
        "deadline" => Some(vec!["deadline".into(), text("event"), text("from")]),
        "computation" => Some(vec!["computation".into(), text("name")]),
        "fact" => Some(vec!["fact".into(), text("key")]),
        "definition" => Some(vec!["definition".into(), text("term")]),
        "table" => Some(vec!["table".into(), text("name")]),
        _ => None,
    }
}

/// The claim of an object without its descriptive fields.
fn content(object: &Object) -> String {
    let mut data = object.data.clone();
    if let Some(map) = data.as_object_mut() {
        for key in ["terms", "label", "vars", "supersedes"] {
            map.remove(key);
        }
    }
    data.to_string()
}

/// The digits of a document's title: chapter 11 and chapter 12 are different texts.
fn title_digits(documents: &[Document], id: &str) -> BTreeSet<String> {
    let title = documents
        .iter()
        .find(|d| d.id == id)
        .map_or("", |d| d.title.as_str());
    let mut out = BTreeSet::new();
    let mut run = String::new();
    for c in title.chars().chain(std::iter::once(' ')) {
        if c.is_ascii_digit() {
            run.push(c);
        } else if !run.is_empty() {
            out.insert(std::mem::take(&mut run));
        }
    }
    out
}

/// The sets of objects that can disagree with each other: those of one key, except that objects of documents numbered differently in their titles
/// never compete; an unnumbered document competes with every number.
fn compete<'a>(
    groups: Vec<(Vec<String>, Vec<&'a Object>)>,
    documents: &[Document],
) -> Vec<(Vec<String>, Vec<&'a Object>)> {
    let mut out = Vec::new();
    for (key, group) in groups {
        let mut numbered: Vec<(BTreeSet<String>, Vec<&Object>)> = Vec::new();
        let mut loose: Vec<&Object> = Vec::new();
        for object in group {
            let digits = title_digits(documents, &object.document);
            if digits.is_empty() {
                loose.push(object);
            } else if let Some((_, members)) = numbered.iter_mut().find(|(d, _)| *d == digits) {
                members.push(object);
            } else {
                numbered.push((digits, vec![object]));
            }
        }
        if numbered.is_empty() {
            out.push((key, loose));
            continue;
        }
        for (_, mut members) in numbered {
            members.extend(loose.iter().copied());
            out.push((key.clone(), members));
        }
    }
    out
}

fn basis_of(winner: &Provenance, loser: &Provenance) -> &'static str {
    if winner.authority != loser.authority {
        "authority"
    } else if winner.date.as_deref().unwrap_or("") != loser.date.as_deref().unwrap_or("") {
        "date"
    } else if winner.version != loser.version {
        "version"
    } else {
        "tie"
    }
}

fn resolve(objects: &[Object], documents: &[Document]) -> (Vec<Resolution>, Vec<Flag>) {
    let provenance = |o: &Object| {
        provenance_of(documents, &o.document)
            .cloned()
            .unwrap_or_default()
    };
    let mut resolutions = Vec::new();
    let mut superseded: BTreeSet<String> = BTreeSet::new();
    for object in objects {
        if let Some(target) = object.data.get("supersedes").and_then(Value::as_str) {
            superseded.insert(target.to_string());
            let old_document = objects
                .iter()
                .find(|o| o.id == target)
                .map_or("", |o| o.document.as_str());
            resolutions.push(Resolution {
                kind: "amendment".into(),
                winner: object.id.clone(),
                superseded: vec![target.to_string()],
                basis: "amendment".into(),
                bases: BTreeMap::from([(target.to_string(), "amendment".to_string())]),
                detail: format!("{} amends {old_document}", object.document),
            });
        }
    }
    let mut groups: Vec<(Vec<String>, Vec<&Object>)> = Vec::new();
    for object in objects {
        let (Some(key), false) = (conflict_key(object), superseded.contains(&object.id)) else {
            continue;
        };
        match groups.iter_mut().find(|(k, _)| *k == key) {
            Some((_, group)) => group.push(object),
            None => groups.push((key, vec![object])),
        }
    }
    let groups = compete(groups, documents);
    let mut flags = Vec::new();
    for (key, group) in groups {
        if group
            .iter()
            .map(|o| content(o))
            .collect::<BTreeSet<_>>()
            .len()
            < 2
        {
            continue;
        }
        let mut ordered = group.clone();
        ordered.sort_by(|a, b| provenance(b).rank().cmp(&provenance(a).rank()));
        // Documents that agree with the top one do not dispute it: the
        // strongest competitor is the best of those that say something else.
        let top = ordered[0];
        let Some(second) = ordered.iter().copied().find(|o| content(o) != content(top)) else {
            continue;
        };
        let detail = format!("{} {}", key[0], key[1..].join(" / "));
        if provenance(top).rank() > provenance(second).rank() && content(top) != content(second) {
            let losers: Vec<&Object> = ordered[1..]
                .iter()
                .copied()
                .filter(|o| content(o) != content(top))
                .collect();
            resolutions.push(Resolution {
                kind: "conflict".into(),
                winner: top.id.clone(),
                superseded: losers.iter().map(|o| o.id.clone()).collect(),
                basis: basis_of(&provenance(top), &provenance(second)).into(),
                bases: losers
                    .iter()
                    .map(|o| {
                        (
                            o.id.clone(),
                            basis_of(&provenance(top), &provenance(o)).to_string(),
                        )
                    })
                    .collect(),
                detail,
            });
        } else {
            flags.push(Flag {
                kind: "conflict".into(),
                objects: group.iter().map(|o| o.id.clone()).collect(),
                reason: format!("{detail} is stated differently"),
            });
        }
    }
    (resolutions, flags)
}

/// The ids a pack supersedes.
#[must_use]
pub fn superseded_ids(pack: &Pack) -> BTreeSet<String> {
    pack.resolutions
        .iter()
        .flat_map(|r| r.superseded.iter().cloned())
        .collect()
}

#[cfg(test)]
mod tests;
