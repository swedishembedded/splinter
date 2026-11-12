// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The shape checks an object's data must pass before admission: names are identifiers, conditions and expressions are well formed,
//! a kind has its keys.

use std::collections::BTreeSet;

use serde_json::Value;

use super::{ARITHMETIC, OPERATORS, VARIABLE_TYPES};

fn is_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    chars.next().is_some_and(|c| c.is_ascii_lowercase())
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

/// Names the ruleset is built from are identifiers, not sentences: the vocabulary step's check, after normalisation has had its say.
pub(super) fn check_identifiers(data: &serde_json::Map<String, Value>) -> Result<(), String> {
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

pub(super) fn check_shape(kind: &str, data: &serde_json::Map<String, Value>) -> Result<(), String> {
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

/// Whether a quote counts a period from an event ("within 30 days", "at least 60 days before", "30 days after the invoice date"): what a deadline is.
/// "More than 14 days after the agreed date" says how late something may be before a right arises, which is a limit and not a deadline.
pub(super) fn period_from_event(quote: &str) -> bool {
    let lower = quote.to_lowercase();
    let tokens: Vec<&str> = lower
        .split_whitespace()
        .map(|t| t.trim_matches(|c: char| !c.is_alphanumeric() && c != '.'))
        .collect();
    let is_number = |t: &str| {
        !t.is_empty()
            && t.chars().all(|c| c.is_ascii_digit() || c == '.')
            && t.chars().any(|c| c.is_ascii_digit())
    };
    tokens.iter().enumerate().any(|(i, token)| {
        if !is_number(token) {
            return false;
        }
        let before = |n: usize| {
            i.checked_sub(n)
                .and_then(|j| tokens.get(j))
                .copied()
                .unwrap_or("")
        };
        if before(1) == "within" || (before(1) == "least" && before(2) == "at") {
            return true;
        }
        if (before(2) == "more" || before(2) == "less") && before(1) == "than"
            || before(1) == "over"
            || before(1) == "under"
        {
            return false;
        }
        let mut next = i + 1;
        if matches!(tokens.get(next), Some(&"business" | &"calendar")) {
            next += 1;
        }
        matches!(
            tokens.get(next),
            Some(&("day" | "days" | "week" | "weeks" | "month" | "months"))
        ) && matches!(
            tokens.get(next + 1),
            Some(&("after" | "before" | "from" | "of"))
        )
    })
}
