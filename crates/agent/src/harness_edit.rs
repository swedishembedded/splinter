// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements automated improvement of agent harnesses
// that adopts a change only on verified, regression-free evidence, for its
// clients. If your team needs expertise in measuring whether a harness
// change really helped, you can procure our services by sending an email to
// info@swedishembedded.com.

//! A model asked to edit a sven harness document.
//!
//! The reply is a list of edits (a path into the YAML document and the value
//! to set there), never a rewritten document, so what a proposal changes is
//! what it says. An edit only counts once the document it makes compiles
//! under sven's own compiler: that check is the call's postcondition, so a
//! reply that breaks the document is sent back to the model with the
//! compiler's reason, up to the call's repairs, and one that is never fixed
//! is an error, not a candidate.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use sven_sdk::schemars::{self, JsonSchema};
use sven_sdk::{compile_harness, CallError, CancelToken};

use crate::solve::Model;
use crate::typed::{Spent, TypedCall};

/// One change: set the value at `path`.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize, JsonSchema)]
pub struct HarnessEdit {
    /// The keys from the document's root to the value, for example
    /// `["steps", "fix", "agent", "prompt"]`. Every key but the last must
    /// exist; the last may be new.
    pub path: Vec<String>,
    /// The value to set, as JSON.
    pub value: serde_json::Value,
}

/// A proposed change to a harness document.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize, JsonSchema)]
pub struct HarnessEdits {
    /// Why these edits should make the harness solve more tasks.
    pub rationale: String,
    /// The edits, applied in order.
    pub edits: Vec<HarnessEdit>,
}

/// Why edits do not make a usable document.
fn refuse<T>(why: impl Into<String>) -> Result<T, String> {
    Err(why.into())
}

/// `document` with `edits` applied.
///
/// # Errors
///
/// The reason, as a sentence for the model: no edits, a path that is not in
/// the document, edits that change nothing, or a document sven's compiler
/// refuses (with its message).
pub fn apply(document: &str, edits: &HarnessEdits) -> Result<String, String> {
    if edits.edits.is_empty() {
        return refuse("the reply holds no edits");
    }
    let original: serde_yaml::Value =
        serde_yaml::from_str(document).map_err(|e| format!("the document is not YAML: {e}"))?;
    let mut edited = original.clone();
    for edit in &edits.edits {
        let Some((last, parents)) = edit.path.split_last() else {
            return refuse("an edit has an empty path");
        };
        let parent = parents.iter().try_fold(&mut edited, |at, key| {
            at.get_mut(key.as_str())
                .ok_or_else(|| format!("path {} is not in the document", edit.path.join(".")))
        })?;
        let Some(map) = parent.as_mapping_mut() else {
            return refuse(format!(
                "path {} does not lead into a mapping",
                edit.path.join(".")
            ));
        };
        let value = serde_yaml::to_value(&edit.value)
            .map_err(|e| format!("the value of {} is not YAML: {e}", edit.path.join(".")))?;
        map.insert(serde_yaml::Value::String(last.clone()), value);
    }
    if edited == original {
        return refuse("the edits change nothing");
    }
    let text = serde_yaml::to_string(&edited).map_err(|e| format!("cannot write YAML: {e}"))?;
    check(&text)?;
    Ok(text)
}

/// Whether `document` compiles under sven's compiler.
///
/// # Errors
///
/// The compiler's message.
pub fn check(document: &str) -> Result<(), String> {
    compile_harness(document)
        .map(|_| ())
        .map_err(|e| format!("the edited document does not compile: {e}"))
}

/// What the model is asked.
const BRIEF: &str = "You improve a sven agent harness: a YAML document of steps (predict, agent, \
tool, verify, branch, retry, ...) that solves repository tasks. The input holds the harness \
`document` and `feedback`, how the document did when it was last tried on a frozen task set, \
compared with the baseline. Reply with `edits`: each sets the value at a `path` of keys into \
the document (every key but the last must exist). Change the few things most likely to make \
the harness solve more tasks - a prompt, a step's tools, a bound - and keep the document \
valid: its edits are compiled and a reply that does not compile is sent back to you. Do not \
rewrite the document.";

/// The role the model is given.
const ROLE: &str = "harness engineer";

/// What one proposal is asked, and how it is bounded.
pub struct EditRequest<'a> {
    /// The harness document to edit.
    pub document: &'a str,
    /// How that document did when last tried; empty when it was not.
    pub feedback: &'a str,
    /// How long the call may take, repairs included.
    pub deadline: Duration,
    /// Corrections a reply that does not compile may be sent back for.
    pub repairs: u32,
    /// Stops the call from outside.
    pub cancel: Option<CancelToken>,
}

/// A proposal that compiles.
#[derive(Debug)]
pub struct Edited {
    /// The edited document.
    pub document: String,
    /// The model's reason.
    pub rationale: String,
    /// What the call cost, repairs included.
    pub spent: Spent,
}

/// Asks `model` for edits to `request.document` and applies them.
///
/// # Errors
///
/// [`CallError`]: the model never produced edits that compile, or the call
/// was stopped or failed.
pub async fn propose_edit(model: &Model, request: EditRequest<'_>) -> Result<Edited, CallError> {
    let parent = request.document.to_string();
    let mut call =
        TypedCall::<HarnessEdits>::new("propose_harness_edits", BRIEF, ROLE, request.deadline)
            .repairs(request.repairs)
            .postcondition(move |edits| apply(&parent, edits).map(|_| ()));
    call.cancel = request.cancel.clone();
    let input = serde_json::json!({
        "document": request.document,
        "feedback": request.feedback,
    });
    let measured = call.start(model)?.call_measured(&input).await;
    let edits = measured.result?;
    // The postcondition accepted these edits, so applying them again cannot
    // fail; a failure here would be a bug, reported rather than hidden.
    let document = apply(request.document, &edits).map_err(CallError::Precondition)?;
    Ok(Edited {
        document,
        rationale: edits.rationale,
        spent: measured.spent,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOC: &str =
        "api_version: \"1\"\nname: h\nentry: done\nsteps:\n  done:\n    return:\n      value: 1\n";

    fn edits(path: &[&str], value: serde_json::Value) -> HarnessEdits {
        HarnessEdits {
            rationale: "because".into(),
            edits: vec![HarnessEdit {
                path: path.iter().map(ToString::to_string).collect(),
                value,
            }],
        }
    }

    #[test]
    fn an_edit_sets_an_existing_value_or_adds_a_key_under_an_existing_mapping() {
        let changed = apply(DOC, &edits(&["steps", "done", "return", "value"], 2.into())).unwrap();
        assert!(changed.contains("value: 2"), "{changed}");
        let described = apply(DOC, &edits(&["description"], "new".into())).unwrap();
        assert!(described.contains("description: new"), "{described}");
    }

    #[test]
    fn edits_that_do_not_make_a_usable_document_say_why() {
        let why = |e: HarnessEdits| apply(DOC, &e).unwrap_err();
        assert!(why(edits(&["steps", "ghost", "x"], 1.into())).contains("steps.ghost.x"));
        assert!(why(edits(&["entry"], "done".into())).contains("change nothing"));
        assert!(why(edits(&["entry"], "nowhere".into())).contains("does not compile"));
        assert!(why(edits(&["entrypoint"], "done".into())).contains("unknown field"));
        assert!(why(HarnessEdits {
            rationale: String::new(),
            edits: Vec::new()
        })
        .contains("no edits"));
    }
}
