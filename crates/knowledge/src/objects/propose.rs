// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model-driven knowledge extraction whose every
// claim is traceable to its source, for its clients. If your team needs
// expertise in grounded rule extraction from legal or procedural documents,
// you can procure our services by sending an email to info@swedishembedded.com.

//! What a proposer of knowledge objects is asked, and what is done with the answer before admission.
//!
//! A [`ObjectProposer`] is shown one document at a time, with the objects admitted from earlier documents (so an amendment can name what it
//! replaces), and answers with an [`ObjectReply`]. Before admission every proposal goes through the **vocabulary step**
//! ([`normalise`]): the names a ruleset is built from (variables, events, outcomes) are rewritten to identifiers, consistently through the
//! whole proposal, and when a vocabulary of outcomes is declared an outcome that is not in it is refused with the list it should have used.
//! [`check_reply`] is the whole check a reply must pass; a model-backed proposer hands it to its call as the postcondition, so a reply with
//! refused proposals is sent back with the reasons, a bounded number of times.

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{admit, Document, Pack, ProposedObject};

/// Why a proposer could not even attempt a document.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct ProposeError(pub String);

/// The names a proposal may use.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Vocabulary {
    /// Outcome names a rule may conclude; empty leaves them open (still identifiers).
    pub outcomes: Vec<String>,
}

/// The shape of a proposer's reply.
#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema)]
pub struct ObjectReply {
    /// The proposed objects of the document shown; none when it states nothing a rule or fact could be made of.
    pub objects: Vec<ProposedObject>,
}

/// What a proposer is shown: the document, and what earlier documents already gave.
#[derive(Clone, Debug, Serialize)]
pub struct DocumentInput {
    /// The document's id; every proposal names it as `document`.
    pub document: String,
    /// Its authority, version and date.
    pub provenance: super::Provenance,
    /// Its text; quotes are copied from it.
    pub text: String,
    /// Objects admitted from earlier documents: `{id, kind, quote}`.
    pub earlier_objects: Vec<Value>,
    /// The same, whole, for checking a reply (not shown).
    #[serde(skip)]
    pub earlier: Vec<ProposedObject>,
    /// Every document of the corpus, for checking a reply (not shown).
    #[serde(skip)]
    pub documents: Vec<Document>,
}

/// A model, or a script standing in for one, asked what a document says.
#[async_trait]
pub trait ObjectProposer: Send + Sync {
    /// The proposals for `document`.
    async fn propose(
        &self,
        document: &DocumentInput,
        vocabulary: &Vocabulary,
    ) -> Result<Vec<ProposedObject>, ProposeError>;
}

/// Replays a stored list of proposals: the proposer of tests and of the synthetic corpus.
pub struct ScriptedProposer(pub Vec<ProposedObject>);

#[async_trait]
impl ObjectProposer for ScriptedProposer {
    async fn propose(
        &self,
        document: &DocumentInput,
        _: &Vocabulary,
    ) -> Result<Vec<ProposedObject>, ProposeError> {
        Ok(self
            .0
            .iter()
            .filter(|p| p.document == document.document)
            .cloned()
            .collect())
    }
}

/// The brief a model proposer is given.
#[must_use]
pub fn brief(vocabulary: &Vocabulary) -> String {
    let outcomes = if vocabulary.outcomes.is_empty() {
        "Name outcomes with short lower-case identifiers such as may_terminate.".to_string()
    } else {
        format!(
            "Outcomes must be exactly one of: {}.",
            vocabulary.outcomes.join(", ")
        )
    };
    format!(
        "You extract typed knowledge objects from ONE document. Reply with the objects it states; propose nothing it does not state.\n\
         Each object: id (short, lower-case, unique), kind, document (the id you are given), quote (copied VERBATIM from the text, as short as it can be while \
         holding every number the object states), data.\n\
         Kinds and their data:\n\
         - threshold: var, op (< <= > >= == !=), value (number), outcome, label, vars (name -> int|number|bool|date); limit=true when it is a limit a draft must stay within.\n\
         - deadline: event, from (a date variable), offset (integer days), unit (calendar_days|business_days), vars.\n\
         - rule: when (condition), outcome, vars. exception: overrides (id of the rule it overrides), when, outcome, vars.\n\
         - computation: name, when (condition), expr, vars. A condition is {{var, op, value}}, {{all: [..]}}, {{any: [..]}} or {{not: ..}}; \
         an expression is [\"const\", \"0.5\"], [\"var\", name] or [op, e1, e2] with op in add sub mul div min max pct (pct: base * percent / 100).\n\
         - procedure_step: order, element (a short phrase), terms. fact: key (\"attribute | entity\" with the real words, e.g. \"service interval | alder pump 3\"), value (as written), form (property|behaviour).\n\
         - definition: term, meaning. table: name, columns, rows (cells copied from the quote). trait: trait, topic, keywords, reveals (first person), terms.\n\
         Rules: every number in data must appear in the quote; every name (variable, event, outcome) is a lower-case identifier with underscores, never a sentence; \
         a variable is declared in vars wherever it is used. {outcomes}\n\
         If this document amends or replaces an earlier object (it is listed in earlier_objects), set data.supersedes to that object's id and restate it with the new values. \
         Where the document states a rule and an exception, give both. A formula (an amount, a percentage per day or year) is a computation, not a threshold."
    )
}

/// The identifier a free-text name becomes: lower case, words joined by underscores.
#[must_use]
pub fn identifier(text: &str) -> String {
    let mut out = String::new();
    for c in text.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.ends_with('_') && !out.is_empty() {
            out.push('_');
        }
    }
    let out = out.trim_end_matches('_').to_string();
    if out.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        format!("v_{out}")
    } else {
        out
    }
}

/// Rewrites every variable, event and outcome name of `proposal` to its identifier, consistently, and checks outcomes against `vocabulary`.
///
/// # Errors
/// The reason an outcome is not admissible.
pub fn normalise(proposal: &mut ProposedObject, vocabulary: &Vocabulary) -> Result<(), String> {
    fn rewrite(node: &mut Value) {
        match node {
            Value::Object(map) => {
                for key in ["var", "from", "event"] {
                    if let Some(Value::String(s)) = map.get_mut(key) {
                        *s = identifier(s);
                    }
                }
                if let Some(Value::Object(vars)) = map.get_mut("vars") {
                    *vars = std::mem::take(vars)
                        .into_iter()
                        .map(|(k, v)| (identifier(&k), v))
                        .collect();
                }
                map.values_mut().for_each(rewrite);
            }
            Value::Array(items) => {
                if items.len() == 2 && items[0] == "var" {
                    if let Value::String(s) = &mut items[1] {
                        *s = identifier(s);
                    }
                }
                items.iter_mut().for_each(rewrite);
            }
            _ => {}
        }
    }
    rewrite(&mut proposal.data);
    if let Some(Value::String(outcome)) = proposal.data.get_mut("outcome") {
        *outcome = identifier(outcome);
        if !vocabulary.outcomes.is_empty() && !vocabulary.outcomes.contains(outcome) {
            return Err(format!(
                "outcome {outcome:?} is not in the declared vocabulary [{}]",
                vocabulary.outcomes.join(", ")
            ));
        }
    }
    Ok(())
}

/// Everything a reply must pass: its proposals normalised and admitted over the documents (with the earlier documents' objects in reach, so an
/// amendment finds what it replaces), none refused.
///
/// # Errors
/// One line per refused proposal, for the model to correct.
pub fn check_reply(
    input: &DocumentInput,
    reply: &ObjectReply,
    vocabulary: &Vocabulary,
) -> Result<Vec<ProposedObject>, String> {
    let mut refused = Vec::new();
    let mut proposals = Vec::new();
    for proposal in &reply.objects {
        let mut proposal = proposal.clone();
        proposal.document.clone_from(&input.document);
        match normalise(&mut proposal, vocabulary) {
            Ok(()) => proposals.push(proposal),
            Err(reason) => refused.push(format!("{}: {reason}", proposal.id)),
        }
    }
    let mut together: Vec<ProposedObject> = input.earlier.clone();
    together.extend(proposals.iter().cloned());
    let pack = admit("check", &input.documents, &together);
    let ours: Vec<&str> = proposals.iter().map(|p| p.id.as_str()).collect();
    refused.extend(
        pack.rejected
            .iter()
            .filter(|r| ours.contains(&r.id.as_str()))
            .map(|r| format!("{}: {}", r.id, r.reason)),
    );
    if refused.is_empty() {
        Ok(proposals)
    } else {
        Err(format!(
            "these proposals were refused, correct them or leave them out:\n{}",
            refused.join("\n")
        ))
    }
}

/// Admits what `proposer` says about every document, in order of authority and date so that an amendment follows what it amends.
pub async fn propose_pack(
    pack_id: &str,
    documents: &[Document],
    proposer: &dyn ObjectProposer,
    vocabulary: &Vocabulary,
) -> Result<Pack, ProposeError> {
    let mut order: Vec<&Document> = documents.iter().collect();
    order.sort_by(|a, b| {
        (
            a.provenance.date.as_deref().unwrap_or(""),
            a.provenance.version,
            &a.id,
        )
            .cmp(&(
                b.provenance.date.as_deref().unwrap_or(""),
                b.provenance.version,
                &b.id,
            ))
    });
    let mut all: Vec<ProposedObject> = Vec::new();
    let mut known: Vec<Value> = Vec::new();
    for document in order {
        let input = DocumentInput {
            document: document.id.clone(),
            provenance: document.provenance.clone(),
            text: document.text.clone(),
            earlier_objects: known.clone(),
            earlier: all.clone(),
            documents: documents.to_vec(),
        };
        let mut proposals = proposer.propose(&input, vocabulary).await?;
        let local: Vec<String> = proposals.iter().map(|p| p.id.clone()).collect();
        for proposal in &mut proposals {
            // Ids are namespaced by document; a reference to an object of the same document follows its renaming.
            for key in ["overrides", "supersedes"] {
                if let Some(Value::String(target)) = proposal.data.get_mut(key) {
                    if local.contains(target) {
                        *target = format!("{}-{target}", document.id);
                    }
                }
            }
            proposal.id = format!("{}-{}", document.id, proposal.id);
            known.push(serde_json::json!({"id": proposal.id, "kind": proposal.kind, "quote": proposal.quote.chars().take(160).collect::<String>()}));
        }
        all.extend(proposals);
    }
    Ok(admit(pack_id, documents, &all))
}
