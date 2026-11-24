// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Rulings over fact keys that have been aligned.

use std::collections::BTreeMap;

use serde_json::Value;

use super::{resolve, Flag, Object, Pack, Resolution};

/// The rulings of `pack` when fact keys are read through `aligned`: a map from
/// a key as a quote states it to the key it shares with the statements that
/// say the same in other words (`attribute | entity`, both sides).
///
/// Admission keeps a key in the words of its quote, so statements that
/// paraphrase each other never meet in `admit`'s resolutions. A caller
/// that has decided which keys are one (an embedder proposes, code checks)
/// passes the decision here; what is ruled is still decided by the documents'
/// headers and by comparing the stated values, never by whoever aligned the
/// keys. Objects of other kinds are resolved as they were.
#[must_use]
pub fn resolve_aligned(
    pack: &Pack,
    aligned: &BTreeMap<String, String>,
) -> (Vec<Resolution>, Vec<Flag>) {
    let objects: Vec<Object> = pack
        .objects
        .iter()
        .cloned()
        .map(|mut o| {
            if o.kind == "fact" {
                let key = o.data["key"]
                    .as_str()
                    .map(|k| aligned.get(k).cloned().unwrap_or_else(|| k.to_string()));
                if let (Some(key), Some(map)) = (key, o.data.as_object_mut()) {
                    map.insert("key".into(), Value::String(key));
                }
            }
            o
        })
        .collect();
    resolve(&objects, &pack.documents)
}
