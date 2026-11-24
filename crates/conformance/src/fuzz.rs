// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for parsers that refuse hostile
// input by name instead of crashing, for its clients. If your team needs
// expertise in property-based testing and fuzzing of data pipelines then you
// can procure our services by sending an email to info@swedishembedded.com.

//! Inputs for property tests of the parsers that read what Splinter does not
//! control: files a user or a repository wrote, records another program
//! produced, a model's output.
//!
//! The strategies start from a valid document and damage it the ways files
//! are damaged: truncated, a byte flipped, a span lost or duplicated, a key
//! removed, a value replaced by one of another type. Inputs near a valid one
//! reach the parser's deep paths, where arbitrary bytes are refused at the
//! first character. A property over them is that the parser answers (a value
//! or an error) and never panics; proptest shrinks a failure to a minimal
//! input and saves it under the test's `proptest-regressions/`, which is
//! committed so every failure found stays a regression case.

use proptest::prelude::*;
use serde_json::Value;

/// How many cases a fuzz property runs: enough to reach the damaged paths of
/// one document, few enough for every pull request.
pub const CASES: u32 = 2048;

/// The configuration of a fuzz property: [`CASES`] cases, failures saved.
#[must_use]
pub fn config() -> ProptestConfig {
    ProptestConfig {
        cases: CASES,
        ..ProptestConfig::default()
    }
}

/// One way bytes are damaged.
#[derive(Clone, Debug)]
enum Damage {
    Truncate(usize),
    Flip(usize, u8),
    Insert(usize, u8),
    Remove(usize, usize),
    Duplicate(usize, usize),
}

impl Damage {
    fn apply(&self, bytes: &mut Vec<u8>) {
        let at = |i: usize, len: usize| if len == 0 { 0 } else { i % (len + 1) };
        match *self {
            Damage::Truncate(i) => bytes.truncate(at(i, bytes.len())),
            Damage::Flip(i, mask) => {
                if !bytes.is_empty() {
                    let i = i % bytes.len();
                    bytes[i] ^= mask.max(1);
                }
            }
            Damage::Insert(i, byte) => {
                let i = at(i, bytes.len());
                bytes.insert(i, byte);
            }
            Damage::Remove(i, n) | Damage::Duplicate(i, n) => {
                let start = at(i, bytes.len());
                let end = (start + n % 16 + 1).min(bytes.len());
                if matches!(self, Damage::Remove(..)) {
                    bytes.drain(start..end);
                } else {
                    let span: Vec<u8> = bytes[start..end].to_vec();
                    bytes.splice(end..end, span);
                }
            }
        }
    }
}

fn damage() -> impl Strategy<Value = Damage> {
    // A byte the parsers care about is more likely than any byte.
    let byte = prop_oneof![
        any::<u8>(),
        prop::sample::select(b"{}[]\":,\n -#\\0e.".to_vec()),
    ];
    prop_oneof![
        any::<usize>().prop_map(Damage::Truncate),
        (any::<usize>(), any::<u8>()).prop_map(|(i, m)| Damage::Flip(i, m)),
        (any::<usize>(), byte).prop_map(|(i, b)| Damage::Insert(i, b)),
        (any::<usize>(), any::<usize>()).prop_map(|(i, n)| Damage::Remove(i, n)),
        (any::<usize>(), any::<usize>()).prop_map(|(i, n)| Damage::Duplicate(i, n)),
    ]
}

/// `seed`, damaged one to four times.
pub fn damaged_bytes(seed: Vec<u8>) -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(damage(), 1..=4).prop_map(move |damages| {
        let mut bytes = seed.clone();
        for d in &damages {
            d.apply(&mut bytes);
        }
        bytes
    })
}

/// `seed`, damaged as bytes and read back as text (invalid UTF-8 replaced:
/// a text parser is handed a `&str`).
pub fn damaged_text(seed: &str) -> impl Strategy<Value = String> {
    damaged_bytes(seed.as_bytes().to_vec())
        .prop_map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
}

/// Any JSON value of a few levels: the shapes a writer of another schema, or
/// of none, produces.
pub fn json_value() -> impl Strategy<Value = Value> {
    let leaf = prop_oneof![
        Just(Value::Null),
        any::<bool>().prop_map(Value::Bool),
        any::<i64>().prop_map(Value::from),
        any::<f64>()
            .prop_filter("JSON has no NaN or infinity", |f| f.is_finite())
            .prop_map(Value::from),
        "[ -~]{0,12}".prop_map(Value::String),
    ];
    leaf.prop_recursive(3, 24, 4, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 0..4).prop_map(Value::Array),
            prop::collection::btree_map("[a-z_]{1,8}", inner, 0..4)
                .prop_map(|m| Value::Object(m.into_iter().collect())),
        ]
    })
}

/// One way a document's structure is damaged.
#[derive(Clone, Debug)]
enum Edit {
    /// The key at this position of the object reached is removed.
    Remove(usize),
    /// The value reached is replaced.
    Replace(Value),
    /// A key the schema does not know is added to the object reached.
    Add(String, Value),
}

/// `seed` with one to three of its values, at any depth, removed, replaced
/// by a value of any type, or joined by an unknown key.
pub fn damaged_json(seed: Value) -> impl Strategy<Value = Value> {
    let edit = prop_oneof![
        any::<usize>().prop_map(Edit::Remove),
        json_value().prop_map(Edit::Replace),
        ("[a-z_]{1,8}", json_value()).prop_map(|(k, v)| Edit::Add(k, v)),
    ];
    prop::collection::vec((prop::collection::vec(any::<usize>(), 0..5), edit), 1..=3).prop_map(
        move |edits| {
            let mut doc = seed.clone();
            for (path, edit) in &edits {
                apply(&mut doc, path, edit);
            }
            doc
        },
    )
}

/// Applies `edit` to the value `path` reaches, each step choosing a child by
/// its position modulo the children there (stopping at a leaf).
fn apply(doc: &mut Value, path: &[usize], edit: &Edit) {
    if let Some((&step, rest)) = path.split_first() {
        let child = match doc {
            Value::Array(items) if !items.is_empty() => {
                let i = step % items.len();
                items.get_mut(i)
            }
            Value::Object(map) if !map.is_empty() => {
                let i = step % map.len();
                map.values_mut().nth(i)
            }
            _ => None,
        };
        if let Some(child) = child {
            return apply(child, rest, edit);
        }
    }
    match (edit, doc) {
        (Edit::Remove(i), Value::Object(map)) if !map.is_empty() => {
            if let Some(key) = map.keys().nth(i % map.len()).cloned() {
                map.remove(&key);
            }
        }
        (Edit::Remove(i), Value::Array(items)) if !items.is_empty() => {
            let i = i % items.len();
            items.remove(i);
        }
        (Edit::Add(key, value), Value::Object(map)) => {
            map.insert(key.clone(), value.clone());
        }
        (Edit::Replace(value), at) => *at = value.clone(),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::strategy::ValueTree as _;
    use proptest::test_runner::TestRunner;

    #[test]
    fn damage_changes_the_seed_and_stays_near_it() {
        let seed = br#"{"schema":"x/v1","items":[1,2,3]}"#.to_vec();
        let mut runner = TestRunner::deterministic();
        let mut changed = 0;
        for _ in 0..512 {
            let value = damaged_bytes(seed.clone())
                .new_tree(&mut runner)
                .unwrap()
                .current();
            changed += usize::from(value != seed);
            assert!(value.len() <= seed.len() + 4 * 17, "{}", value.len());
        }
        assert!(
            changed > 400,
            "damage left the seed as it was {} of 512 times",
            512 - changed
        );
    }

    #[test]
    fn a_json_edit_reaches_nested_values() {
        let seed = serde_json::json!({"a": {"b": {"c": 1}}, "d": [1, 2]});
        let mut runner = TestRunner::deterministic();
        let mut nested = 0;
        for _ in 0..512 {
            let value = damaged_json(seed.clone())
                .new_tree(&mut runner)
                .unwrap()
                .current();
            nested += usize::from(value.get("a").and_then(|a| a.get("b")) != seed["a"].get("b"));
        }
        assert!(nested > 0, "no edit reached a nested value");
    }
}
