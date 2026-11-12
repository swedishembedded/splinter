// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What is done with a document's proposals before admission: one type per variable, restatements dropped, amendments aligned to their targets.

use serde_json::Value;

use super::ProposedObject;

/// Makes every variable one type across the proposals: a model that calls the same variable an `int` in one object and a `number` in another gets
/// `number` (the wider), so that the ruleset built from them does not fall over on a name used twice.
pub fn unify_variable_types(proposals: &mut [ProposedObject]) {
    let mut types: std::collections::BTreeMap<String, String> = std::collections::BTreeMap::new();
    for proposal in proposals.iter() {
        if let Some(vars) = proposal.data.get("vars").and_then(Value::as_object) {
            for (name, ty) in vars {
                let Some(ty) = ty.as_str() else { continue };
                let entry = types.entry(name.clone()).or_insert_with(|| ty.to_string());
                if *entry != ty
                    && matches!((entry.as_str(), ty), ("int", "number") | ("number", "int"))
                {
                    *entry = "number".to_string();
                }
            }
        }
    }
    for proposal in proposals.iter_mut() {
        if let Some(vars) = proposal.data.get_mut("vars").and_then(Value::as_object_mut) {
            for (name, ty) in vars.iter_mut() {
                if let (Some(unified), Some(_)) = (types.get(name), ty.as_str()) {
                    *ty = Value::String(unified.clone());
                }
            }
        }
    }
}

/// Whether `higher` says more than `lower` about the same sentence, so that `lower` only restates it: a formula over the limit or fact it contains, an
/// ordered step over the rule or fact that repeats it, a period over a fact, a plain statement ("The X of Y is V.") over a limit read into it, and a
/// limit or rule over a fact only when the sentence is not a plain statement. What is not listed (a period and a limit in one sentence) is kept: two
/// objects of one quote are not duplicates when each says something the other does not.
fn dominates(higher: &ProposedObject, lower: &ProposedObject) -> bool {
    let statement = is_statement(&higher.quote) || is_statement(&lower.quote);
    match (higher.kind.as_str(), lower.kind.as_str()) {
        ("computation", "threshold" | "rule" | "fact" | "definition")
        | ("procedure_step", "threshold" | "rule" | "fact")
        | ("deadline" | "exception", "fact") => true,
        ("fact", "threshold" | "rule") => statement,
        ("threshold" | "rule", "fact") => !statement,
        _ => false,
    }
}

/// Whether a quote is one of the two plain statement forms, "The X of Y is V." or "When X, a Y client Z.": a sentence that states a value or a habit
/// and says nothing about what follows from it.
fn is_statement(quote: &str) -> bool {
    let q = quote.trim();
    (q.starts_with("The ") && q.contains(" of ") && q.contains(" is "))
        || (q.starts_with("When ") && q.contains(", a ") && q.contains(" client "))
}

/// Drops the objects that only restate a dominating object of the same quote: the same numbers or fewer, from a quote that overlaps it.
/// Passes that each find a sentence interesting produce a fact, a threshold and a deadline out of one period; the one that says most stays.
pub fn arbitrate(proposals: &mut Vec<ProposedObject>) {
    let numbers = |p: &ProposedObject| {
        let mut set = std::collections::BTreeSet::new();
        super::claimed_numbers(&p.data, None, &mut |n| {
            if n != "0" && n != "1" {
                set.insert(n);
            }
        });
        if p.kind == "fact" {
            // A fact states its value as written ("250 hours"): what it says in numbers is in its quote.
            set.extend(
                super::quote_numbers(&p.quote)
                    .into_iter()
                    .filter(|n| n != "0" && n != "1"),
            );
        }
        set
    };
    let overlap = |a: &str, b: &str| a.contains(b) || b.contains(a);
    let keep: Vec<bool> = proposals
        .iter()
        .enumerate()
        .map(|(i, lower)| {
            !proposals.iter().enumerate().any(|(j, higher)| {
                j != i
                    && dominates(higher, lower)
                    && overlap(&higher.quote, &lower.quote)
                    && numbers(lower).is_subset(&numbers(higher))
            })
        })
        .collect();
    let mut at = 0;
    proposals.retain(|_| {
        at += 1;
        keep[at - 1]
    });
}

/// The words of a title that tell one document from another: a revision or an amendment of a text has the text's title plus one of these words.
fn title_words(title: &str) -> std::collections::BTreeSet<String> {
    const FILLER: [&str; 8] = [
        "amendment",
        "revision",
        "edition",
        "older",
        "new",
        "the",
        "of",
        "a",
    ];
    title
        .split(|c: char| !c.is_alphanumeric())
        .map(str::to_lowercase)
        .filter(|w| !w.is_empty() && !FILLER.contains(&w.as_str()))
        .collect()
}

/// The earlier objects a document is shown: those of documents whose title is (nearly) the same words, which is what an amendment or a revision of
/// them looks like. Showing every earlier object would grow with the corpus; showing none leaves an amendment unable to name what it replaces.
pub(super) fn related_objects(known: &[Value], title: &str) -> Vec<Value> {
    let mine = title_words(title);
    known
        .iter()
        .filter(|k| {
            let theirs = title_words(k["title"].as_str().unwrap_or_default());
            let shared = mine.intersection(&theirs).count();
            // The chapter or protocol number tells siblings apart: a revision of chapter 11 is not shown chapter 12.
            let numbers = |set: &std::collections::BTreeSet<String>| set.iter().filter(|w| w.chars().any(|c| c.is_ascii_digit())).cloned().collect::<Vec<_>>();
            numbers(&mine) == numbers(&theirs)
                && !mine.is_empty() && shared > 0 && (shared as f64) / (mine.len().min(theirs.len()) as f64) >= 0.75
        })
        .map(|k| serde_json::json!({"id": k["id"], "kind": k["kind"], "quote": k["quote"], "data": k["data"]}))
        .collect()
}

/// Renames in `node` every variable called `old` to `new`: condition and expression references, declarations, a deadline's origin.
fn rename_variable(node: &mut Value, old: &str, new: &str) {
    match node {
        Value::Object(map) => {
            for key in ["var", "from"] {
                if let Some(Value::String(s)) = map.get_mut(key) {
                    if s == old {
                        *s = new.to_string();
                    }
                }
            }
            if let Some(Value::Object(vars)) = map.get_mut("vars") {
                if let Some(ty) = vars.remove(old) {
                    vars.entry(new.to_string()).or_insert(ty);
                }
            }
            map.values_mut().for_each(|v| rename_variable(v, old, new));
        }
        Value::Array(items) => {
            if items.len() == 2 && items[0] == "var" && items[1] == old {
                items[1] = Value::String(new.to_string());
            }
            items.iter_mut().for_each(|v| rename_variable(v, old, new));
        }
        _ => {}
    }
}

/// An amendment restates what it replaces, so it speaks of the same variables and the same event: a model that names them afresh in the amending
/// document leaves the two unable to be compared. The amendment's variables take the names of the replaced object's, in order, when both have as many.
pub fn align_amendments(proposals: &mut [ProposedObject], earlier: &[ProposedObject]) {
    for proposal in proposals.iter_mut() {
        let Some(target) = proposal
            .data
            .get("supersedes")
            .and_then(Value::as_str)
            .and_then(|id| earlier.iter().find(|e| e.id == id))
        else {
            continue;
        };
        let names = |p: &ProposedObject| -> Vec<String> {
            p.data
                .get("vars")
                .and_then(Value::as_object)
                .map(|v| v.keys().cloned().collect())
                .unwrap_or_default()
        };
        let (mine, theirs) = (names(proposal), names(target));
        if mine.len() == theirs.len() {
            let free_mine: Vec<&String> = mine.iter().filter(|n| !theirs.contains(n)).collect();
            let free_theirs: Vec<&String> = theirs.iter().filter(|n| !mine.contains(n)).collect();
            for (old, new) in free_mine.iter().zip(free_theirs.iter()) {
                rename_variable(&mut proposal.data, old, new);
            }
        }
        for key in ["event", "name", "outcome"] {
            if let Some(value) = target.data.get(key).cloned() {
                if proposal.data.get(key).is_some() {
                    proposal.data[key] = value;
                }
            }
        }
    }
}
