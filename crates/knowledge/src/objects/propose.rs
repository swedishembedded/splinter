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

use super::arbitrate::related_objects;
pub use super::arbitrate::{align_amendments, arbitrate, unify_variable_types};
pub use super::brief::{brief, chunks, Group, Strategy};
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
    /// The pass this input is for; `None` asks for every kind.
    #[serde(skip)]
    pub group: Option<Group>,
    /// Whether the brief carries a worked example.
    #[serde(skip)]
    pub examples: bool,
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
    for rejection in pack
        .rejected
        .iter()
        .filter(|r| ours.contains(&r.id.as_str()))
    {
        let quote = proposals
            .iter()
            .find(|p| p.id == rejection.id)
            .map_or("", |p| p.quote.as_str());
        refused.push(format!(
            "{}: {}{}",
            rejection.id,
            rejection.reason,
            hint(&rejection.reason, quote)
        ));
    }
    if refused.is_empty() {
        Ok(proposals)
    } else {
        Err(format!(
            "these proposals were refused; correct them or leave them out, and keep every other object as it was:\n{}",
            refused.join("\n")
        ))
    }
}

/// What to do about a refusal, in the words the model needs: the quote it gave, and what to change.
fn hint(reason: &str, quote: &str) -> String {
    if reason.contains("not in the quote") {
        return format!(" (your quote was {quote:?}: use only numbers that appear in it, or quote the sentence that holds them)");
    }
    if reason.contains("not in the document") {
        return " (copy the quote character for character from the text, including punctuation)"
            .to_string();
    }
    if reason.contains("not identifiers") {
        return " (use lower-case words joined by underscores)".to_string();
    }
    if reason.contains("unknown variable type") {
        return " (a variable's type is one of int, number, bool, date)".to_string();
    }
    if reason.contains("key is not grounded") || reason.contains("fact key") {
        return " (a fact key is \"attribute | entity\" written with words that are in the quote; if the sentence is a rule or a behaviour, give it as that kind instead)".to_string();
    }
    if reason.contains("value is not in the quote") {
        return " (the value is copied from the quote as written)".to_string();
    }
    if reason.contains("undeclared variable") {
        return " (declare every variable you use in vars, with its type)".to_string();
    }
    String::new()
}
/// A pack with the passes that got no usable reply (a model that answered nothing parseable, a deadline): never silently absent.
#[derive(Debug)]
pub struct Proposed {
    /// What was admitted.
    pub pack: Pack,
    /// `document/pass: why`, one per pass without a reply.
    pub declined: Vec<String>,
}

/// Admits what `proposer` says about every document, in order of date so that an amendment follows what it amends.
///
/// # Errors
/// Never today: a pass without a reply is reported in [`Proposed::declined`].
pub async fn propose_pack(
    pack_id: &str,
    documents: &[Document],
    proposer: &dyn ObjectProposer,
    vocabulary: &Vocabulary,
    strategy: Strategy,
) -> Result<Proposed, ProposeError> {
    let mut declined: Vec<String> = Vec::new();
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
    let groups: Vec<Option<Group>> = if strategy.per_kind {
        Group::ALL.iter().map(|g| Some(*g)).collect()
    } else {
        vec![None]
    };
    let mut all: Vec<ProposedObject> = Vec::new();
    let mut known: Vec<Value> = Vec::new();
    for document in order {
        let pieces = strategy.chunk_words.map_or_else(
            || vec![document.text.clone()],
            |n| chunks(&document.text, n),
        );
        let mut seen: Vec<ProposedObject> = Vec::new();
        for (piece_no, piece) in pieces.iter().enumerate() {
            for group in &groups {
                let input = DocumentInput {
                    document: document.id.clone(),
                    provenance: document.provenance.clone(),
                    text: piece.clone(),
                    earlier_objects: related_objects(&known, &document.title),
                    earlier: all.iter().chain(seen.iter()).cloned().collect(),
                    documents: documents.to_vec(),
                    group: *group,
                    examples: strategy.examples,
                };
                let mut proposals = match proposer.propose(&input, vocabulary).await {
                    Ok(proposals) => proposals,
                    Err(e) => {
                        declined.push(format!(
                            "{}/{}: {e}",
                            document.id,
                            group.map_or("all", |g| g.name())
                        ));
                        continue;
                    }
                };
                if let Some(group) = group {
                    proposals.retain(|p| group.kinds().contains(&p.kind.as_str()));
                }
                let tag = format!(
                    "{}{}",
                    if pieces.len() > 1 {
                        piece_no.to_string()
                    } else {
                        String::new()
                    },
                    group.map_or("", |g| ["c", "k", "p", "s", "b"]
                        [Group::ALL.iter().position(|x| *x == g).unwrap_or(0)])
                );
                let prefix = if tag.is_empty() {
                    document.id.clone()
                } else {
                    format!("{}-{tag}", document.id)
                };
                let local: Vec<String> = proposals.iter().map(|p| p.id.clone()).collect();
                for proposal in &mut proposals {
                    // Ids are namespaced by document and pass; a reference within the reply follows its renaming.
                    for key in ["overrides", "supersedes"] {
                        if let Some(Value::String(target)) = proposal.data.get_mut(key) {
                            if local.contains(target) {
                                *target = format!("{prefix}-{target}");
                            }
                        }
                    }
                    proposal.id = format!("{prefix}-{}", proposal.id);
                }
                align_amendments(&mut proposals, &all);
                // The same object twice (two passes may both see it) is one.
                proposals.retain(|p| {
                    !seen
                        .iter()
                        .any(|q| q.kind == p.kind && q.quote == p.quote && q.data == p.data)
                });
                seen.extend(proposals);
            }
        }
        for proposal in &seen {
            known.push(serde_json::json!({"id": proposal.id, "kind": proposal.kind, "title": document.title,
                "quote": proposal.quote.chars().take(200).collect::<String>(), "data": proposal.data}));
        }
        arbitrate(&mut seen);
        all.extend(seen);
    }
    unify_variable_types(&mut all);
    Ok(Proposed {
        pack: admit(pack_id, documents, &all),
        declined,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn object(id: &str, kind: &str, quote: &str, data: Value) -> ProposedObject {
        ProposedObject {
            id: id.into(),
            kind: kind.into(),
            document: "d".into(),
            quote: quote.into(),
            data,
        }
    }

    #[test]
    fn a_variable_called_int_and_number_becomes_number_everywhere() {
        let mut all = vec![
            object(
                "a",
                "threshold",
                "q",
                json!({"vars": {"months": "int", "other": "bool"}}),
            ),
            object("b", "threshold", "q", json!({"vars": {"months": "number"}})),
        ];
        unify_variable_types(&mut all);
        assert_eq!(
            all[0].data["vars"],
            json!({"months": "number", "other": "bool"})
        );
        assert_eq!(all[1].data["vars"], json!({"months": "number"}));
    }

    #[test]
    fn an_object_that_only_restates_a_dominating_one_of_the_same_quote_is_dropped() {
        let fee = "a late fee of 2 percent, but at most 10 percent";
        let mut all = vec![
            object(
                "fee",
                "computation",
                fee,
                json!({"expr": ["pct", ["var", "rent"], ["min", ["const", "2"], ["const", "10"]]], "name": "fee", "when": {}}),
            ),
            object(
                "cap",
                "threshold",
                "at most 10 percent",
                json!({"value": 10}),
            ),
            object("fact", "fact", fee, json!({"value": "2 percent"})),
            object("other", "threshold", "after 30 days", json!({"value": 30})),
        ];
        arbitrate(&mut all);
        let kept: Vec<&str> = all.iter().map(|p| p.id.as_str()).collect();
        assert_eq!(
            kept,
            ["fee", "other"],
            "the fact and the cap restate the formula; the unrelated threshold stays"
        );
    }

    #[test]
    fn a_plain_statement_beats_a_limit_read_into_it_but_a_limit_beats_a_fact_that_restates_it() {
        let statement = "The service interval of Alder pump 3 is 250 hours.";
        let mut all = vec![
            object("fact", "fact", statement, json!({"value": "250 hours"})),
            object("limit", "threshold", statement, json!({"value": 250})),
        ];
        arbitrate(&mut all);
        assert_eq!(
            all.iter().map(|p| p.id.as_str()).collect::<Vec<_>>(),
            ["fact"]
        );
        let sentence = "If the seller delivers more than 14 days after the agreed date, the buyer may terminate.";
        let mut all = vec![
            object("limit", "threshold", sentence, json!({"value": 14})),
            object("fact", "fact", sentence, json!({"value": "14 days"})),
        ];
        arbitrate(&mut all);
        assert_eq!(
            all.iter().map(|p| p.id.as_str()).collect::<Vec<_>>(),
            ["limit"]
        );
    }

    #[test]
    fn a_limit_and_a_period_of_one_sentence_are_both_kept() {
        let sentence = "The notice must be given within 30 days after the delay became known.";
        let mut all = vec![
            object("period", "deadline", sentence, json!({"offset": 30})),
            object("limit", "threshold", sentence, json!({"value": 30})),
        ];
        arbitrate(&mut all);
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn an_amendment_takes_the_variable_names_and_the_event_of_what_it_replaces() {
        let earlier = vec![object(
            "base",
            "threshold",
            "q",
            json!({"var": "days_late", "op": ">", "value": 14, "outcome": "may_terminate", "vars": {"days_late": "int"}}),
        )];
        let mut amended = vec![object(
            "new",
            "threshold",
            "q",
            json!({"var": "delay_days", "op": ">", "value": 21, "outcome": "may_terminate", "vars": {"delay_days": "int"}, "supersedes": "base"}),
        )];
        align_amendments(&mut amended, &earlier);
        assert_eq!(
            (
                amended[0].data["var"].as_str(),
                amended[0].data["vars"].clone()
            ),
            (Some("days_late"), json!({"days_late": "int"}))
        );
    }

    #[test]
    fn a_document_is_shown_only_the_earlier_objects_of_documents_with_nearly_its_title() {
        let known = vec![
            json!({"id": "a", "kind": "threshold", "title": "Fictional Supply Act, chapter 11", "quote": "q", "data": {}}),
            json!({"id": "b", "kind": "threshold", "title": "Fictional Supply Act, chapter 12", "quote": "q", "data": {}}),
            json!({"id": "c", "kind": "threshold", "title": "Fictional clinic intake protocol 3", "quote": "q", "data": {}}),
        ];
        let shown = related_objects(&known, "Fictional Supply Act, chapter 11: amendment");
        assert_eq!(
            shown
                .iter()
                .map(|k| k["id"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ["a"]
        );
    }

    #[test]
    fn a_limit_and_the_exception_to_it_are_both_kept() {
        let quote = "more than 30 days late, unless the tenant is a pensioner";
        let mut all = vec![
            object("limit", "threshold", quote, json!({"value": 30})),
            object("exception", "exception", quote, json!({"when": {}})),
        ];
        arbitrate(&mut all);
        assert_eq!(all.len(), 2);
    }
}
