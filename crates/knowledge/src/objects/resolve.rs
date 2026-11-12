// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Where admitted objects disagree: the documents' headers decide, amendments replace, what nothing separates stays flagged.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;

use super::{provenance_of, Document, Flag, Object, Pack, Provenance, Resolution};

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

pub(super) fn resolve(objects: &[Object], documents: &[Document]) -> (Vec<Resolution>, Vec<Flag>) {
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
