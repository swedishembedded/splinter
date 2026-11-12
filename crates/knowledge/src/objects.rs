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
//! Two admitted objects that state the same thing differently (the same
//! variable and outcome with another value, the same event with another
//! offset) are [`Flag`]ged as a conflict; they stay in the pack with their
//! provenance and no ruleset is compiled from them. They are never merged.
//!
//! The pack is a plain JSON document (`material-pack/1`); the consumer
//! that compiles it into rule sets and tasks re-runs the same admission on
//! what it reads.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::tasks::grounding::content_words;

/// The pack format's name.
pub const PACK_SCHEMA: &str = "material-pack/1";
/// Numbers that are structure, not a claim about the source.
const STRUCTURAL_NUMBERS: [&str; 2] = ["0", "1"];
/// Data keys whose numbers are structure.
const UNTRACED_KEYS: [&str; 4] = ["order", "vars", "terms", "limit"];
/// The share of a prose field's content words the quote must carry.
const MIN_PROSE_SUPPORT: f64 = 0.6;
const OPERATORS: [&str; 6] = ["<", "<=", ">", ">=", "==", "!="];
const ARITHMETIC: [&str; 7] = ["add", "sub", "mul", "div", "min", "max", "pct"];
const VARIABLE_TYPES: [&str; 4] = ["int", "number", "bool", "date"];

/// A source document.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Document {
    /// Stable id the objects name.
    pub id: String,
    /// Display title.
    pub title: String,
    /// The text quotes are found in.
    pub text: String,
}

/// What a model proposes.
#[derive(Clone, Debug, Deserialize)]
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

/// Two or more objects that state the same thing differently.
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

/// Admitted objects with their documents, conflicts and rejections.
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
    /// Conflicts among them.
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
            admit_one(proposal, documents)
        };
        match outcome {
            Ok(object) => objects.push(object),
            Err(reason) => rejected.push(Rejected {
                id: proposal.id.clone(),
                reason,
            }),
        }
    }
    let flags = conflicts(&objects);
    Pack {
        schema: PACK_SCHEMA.to_string(),
        pack_id: pack_id.to_string(),
        documents: documents.to_vec(),
        objects,
        flags,
        rejected,
    }
}

fn admit_one(proposal: &ProposedObject, documents: &[Document]) -> Result<Object, String> {
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
    let quote_words = content_words(&proposal.quote);
    for key in ["reveals", "element"] {
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
    Ok(Object {
        id: proposal.id.clone(),
        kind: proposal.kind.clone(),
        document: proposal.document.clone(),
        span: [start, start + proposal.quote.chars().count()],
        quote: proposal.quote.clone(),
        data: proposal.data.clone(),
    })
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
fn quote_numbers(text: &str) -> BTreeSet<String> {
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
fn claimed_numbers(node: &Value, key: Option<&str>, sink: &mut dyn FnMut(String)) {
    if key.is_some_and(|k| UNTRACED_KEYS.contains(&k)) {
        return;
    }
    match node {
        Value::Number(n) => sink(normalise(&n.to_string())),
        Value::String(s)
            if matches!(key, Some("value" | "offset"))
                && s.chars().all(|c| c.is_ascii_digit() || c == '.')
                && !s.is_empty() =>
        {
            sink(normalise(s));
        }
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

/// What makes two objects state the same thing; objects of kinds without such a key never conflict.
fn conflict_key(object: &Object) -> Option<Vec<String>> {
    let text = |k: &str| object.data[k].as_str().unwrap_or_default().to_string();
    match object.kind.as_str() {
        "threshold" => Some(vec!["threshold".into(), text("var"), text("outcome")]),
        "deadline" => Some(vec!["deadline".into(), text("event"), text("from")]),
        "computation" => Some(vec!["computation".into(), text("name")]),
        "fact" => Some(vec!["fact".into(), text("key")]),
        _ => None,
    }
}

/// The claim of an object without its descriptive fields.
fn content(object: &Object) -> String {
    let mut data = object.data.clone();
    if let Some(map) = data.as_object_mut() {
        for key in ["terms", "label", "vars"] {
            map.remove(key);
        }
    }
    data.to_string()
}

fn conflicts(objects: &[Object]) -> Vec<Flag> {
    let mut groups: BTreeMap<Vec<String>, Vec<&Object>> = BTreeMap::new();
    for object in objects {
        if let Some(key) = conflict_key(object) {
            groups.entry(key).or_default().push(object);
        }
    }
    let mut flags: Vec<Flag> = groups
        .into_iter()
        .filter(|(_, group)| {
            group
                .iter()
                .map(|o| content(o))
                .collect::<BTreeSet<_>>()
                .len()
                > 1
        })
        .map(|(key, group)| Flag {
            kind: "conflict".into(),
            objects: group.iter().map(|o| o.id.clone()).collect(),
            reason: format!("{} {} is stated differently", key[0], key[1..].join(" / ")),
        })
        .collect();
    flags.sort_by(|a, b| a.objects.cmp(&b.objects));
    flags
}

/// A model, or a script standing in for one, asked what a document says.
pub trait ObjectProposer {
    /// The proposals for one document.
    fn propose(&self, document: &Document) -> Vec<ProposedObject>;
}

/// Replays a stored list of proposals: the proposer of tests and of the synthetic corpus.
pub struct ScriptedProposer(pub Vec<ProposedObject>);

impl ObjectProposer for ScriptedProposer {
    fn propose(&self, document: &Document) -> Vec<ProposedObject> {
        self.0
            .iter()
            .filter(|p| p.document == document.id)
            .cloned()
            .collect()
    }
}

/// Admits what `proposer` says about every document.
#[must_use]
pub fn admit_from(pack_id: &str, documents: &[Document], proposer: &dyn ObjectProposer) -> Pack {
    let proposals: Vec<ProposedObject> =
        documents.iter().flat_map(|d| proposer.propose(d)).collect();
    admit(pack_id, documents, &proposals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const DOCUMENTS: [(&str, &str); 5] = [
        (
            "client-behaviour-sheet",
            include_str!("../tests/fixtures/material/client-behaviour-sheet.md"),
        ),
        (
            "counseling-protocol",
            include_str!("../tests/fixtures/material/counseling-protocol.md"),
        ),
        (
            "payment-act",
            include_str!("../tests/fixtures/material/payment-act.md"),
        ),
        (
            "supply-act-defects",
            include_str!("../tests/fixtures/material/supply-act-defects.md"),
        ),
        (
            "supply-act-delay",
            include_str!("../tests/fixtures/material/supply-act-delay.md"),
        ),
    ];
    const PROPOSALS: &str = include_str!("../tests/fixtures/material/proposals.json");
    const STORED_PACK: &str = include_str!("../tests/fixtures/material/synthetic-pack.json");

    fn documents() -> Vec<Document> {
        DOCUMENTS
            .iter()
            .map(|(id, text)| Document {
                id: (*id).to_string(),
                title: text
                    .lines()
                    .next()
                    .unwrap_or_default()
                    .trim_start_matches("# ")
                    .to_string(),
                text: (*text).to_string(),
            })
            .collect()
    }

    fn proposals() -> Vec<ProposedObject> {
        let file: Value = serde_json::from_str(PROPOSALS).unwrap();
        serde_json::from_value(file["proposals"].clone()).unwrap()
    }

    fn one(id: &str) -> ProposedObject {
        proposals().into_iter().find(|p| p.id == id).unwrap()
    }

    #[test]
    fn the_synthetic_corpus_is_admitted_exactly_as_the_consumers_admission_reads_it() {
        let pack = admit("synthetic", &documents(), &proposals());
        let stored: Value = serde_json::from_str(STORED_PACK).unwrap();
        // The pack is one format with two implementations of its admission; they must agree object for object.
        assert_eq!(
            serde_json::to_value(&pack.objects).unwrap(),
            stored["objects"]
        );
        assert_eq!(serde_json::to_value(&pack.flags).unwrap(), stored["flags"]);
        assert_eq!(
            serde_json::to_value(&pack.rejected).unwrap(),
            stored["rejected"]
        );
        assert_eq!(pack.schema, PACK_SCHEMA);
    }

    #[test]
    fn a_number_the_quote_does_not_carry_is_refused_however_plausible() {
        let pack = admit("t", &documents(), &[one("invented-number")]);
        assert!(
            pack.objects.is_empty() && pack.rejected[0].reason.contains("not in the quote"),
            "{:?}",
            pack.rejected
        );
    }

    #[test]
    fn a_quote_that_is_not_in_the_document_and_unlisted_terms_are_refused() {
        assert!(admit("t", &documents(), &[one("misquoted")]).rejected[0]
            .reason
            .contains("not in the document"));
        let mut trait_ = one("trait-sleep");
        trait_.data["terms"] = json!(["penalty"]);
        assert!(admit("t", &documents(), &[trait_]).rejected[0]
            .reason
            .contains("terms"));
        let mut invented = one("trait-sleep");
        invented.data["reveals"] = json!("I have a secret second job as a pilot.");
        assert!(admit("t", &documents(), &[invented]).rejected[0]
            .reason
            .contains("not supported"));
    }

    #[test]
    fn a_proposal_of_the_wrong_shape_is_refused() {
        let mut no_outcome = one("terminate-late");
        no_outcome.data.as_object_mut().unwrap().remove("outcome");
        assert!(admit("t", &documents(), &[no_outcome]).rejected[0]
            .reason
            .contains("missing"));
        let mut bad_operator = one("terminate-late");
        bad_operator.data["op"] = json!("~");
        assert!(admit("t", &documents(), &[bad_operator]).rejected[0]
            .reason
            .contains("bad condition"));
        let mut unknown = one("terminate-late");
        unknown.kind = "opinion".into();
        assert!(admit("t", &documents(), &[unknown]).rejected[0]
            .reason
            .contains("unknown kind"));
    }

    #[test]
    fn spans_are_character_offsets_that_resolve_to_the_quote() {
        let pack = admit_from("synthetic", &documents(), &ScriptedProposer(proposals()));
        for object in &pack.objects {
            let text = &pack
                .documents
                .iter()
                .find(|d| d.id == object.document)
                .unwrap()
                .text;
            let found: String = text
                .chars()
                .skip(object.span[0])
                .take(object.span[1] - object.span[0])
                .collect();
            assert_eq!(found, object.quote);
        }
    }

    #[test]
    fn objects_that_state_the_same_thing_differently_are_flagged_and_both_kept() {
        let pack = admit_from("synthetic", &documents(), &ScriptedProposer(proposals()));
        assert_eq!(pack.flags.len(), 1);
        assert_eq!(
            pack.flags[0].objects,
            ["payment-due", "payment-due-consumer"]
        );
        assert!(
            pack.objects.iter().any(|o| o.id == "payment-due")
                && pack.objects.iter().any(|o| o.id == "payment-due-consumer")
        );
    }
}
