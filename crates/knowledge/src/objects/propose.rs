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

/// A pass over a document that asks for one family of kinds, with instructions and a worked example that are about those kinds only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Group {
    /// Limits, conditions with consequences, and exceptions.
    Conditions,
    /// Amounts, percentages and formulas.
    Computations,
    /// Periods in days, ordered steps, and maximum durations.
    Periods,
    /// Facts, definitions and tables.
    Statements,
    /// How a subject reveals things when asked.
    Behaviour,
}

impl Group {
    /// Every group, in the order the passes run.
    pub const ALL: [Group; 5] = [
        Group::Conditions,
        Group::Computations,
        Group::Periods,
        Group::Statements,
        Group::Behaviour,
    ];

    /// The kinds a pass of this group may return; a proposal of another kind is dropped.
    #[must_use]
    pub fn kinds(self) -> &'static [&'static str] {
        match self {
            Group::Conditions => &["threshold", "rule", "exception"],
            Group::Computations => &["computation"],
            Group::Periods => &["deadline", "procedure_step", "threshold"],
            Group::Statements => &["fact", "definition", "table"],
            Group::Behaviour => &["trait"],
        }
    }

    /// The pass's name in a report.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Group::Conditions => "conditions",
            Group::Computations => "computations",
            Group::Periods => "periods",
            Group::Statements => "statements",
            Group::Behaviour => "behaviour",
        }
    }

    fn instructions(self) -> &'static str {
        match self {
            Group::Conditions => {
                "Extract every sentence that sets a limit, states a condition with its consequence, or makes an exception. A sentence that only says what a value IS (\"The X of Y is 400 hours\") \
                 is a fact, not a limit: leave it. A limit on a number (more than, at most, at least, within) that the text says has a consequence or must not be crossed \
                 is a threshold: var, op, value, outcome (what follows), label, vars; limit=true when it is a ceiling a draft must stay within. A condition without a number is a rule: \
                 when, outcome, vars. 'Unless', 'except if' and 'does not apply if' make an exception: overrides (the id of the rule it qualifies, which you also give), when, outcome, vars. \
                 Two thresholds in one sentence are two objects with the same quote."
            }
            Group::Computations => {
                "Extract every amount, percentage or formula the text defines. A computation has name, when (the condition under which it applies), expr and vars. \
                 An expression is [\"const\", \"0.5\"], [\"var\", name] or [op, e1, e2] with op in add sub mul div min max pct; pct(base, percent) is base * percent / 100; \
                 'for each day/week' multiplies by a count variable; 'but at most X' is min(..., X)."
            }
            Group::Periods => {
                "Extract every period and ordered action. A period in days counted from an event is a deadline: event, from (a date variable you declare in vars), offset (integer), unit \
                 (calendar_days|business_days), vars. Numbered or ordered actions are procedure_steps: order (1, 2, ...), element (a short phrase), terms; one step per action, several \
                 steps may share the quote of one sentence. A maximum duration (minutes, hours) is a threshold with limit=true: var, op, value, outcome, vars."
            }
            Group::Statements => {
                "Extract every stated fact, definition and table. 'The X of Y is V' is a fact: key \"X | Y\" with the real words, value V as written, form property. \
                 'When X, a Y client Z' is a fact: key \"when X | Y client\", value Z, form behaviour. 'In this chapter, T means M' is a definition: term, meaning. \
                 Values stated for several cases in one sentence are a table: name, columns, rows (every cell copied from the quote)."
            }
            Group::Behaviour => {
                "Extract how a subject behaves when asked: a trait with trait (discloses_when_asked), topic, keywords (words an interviewer's question would contain), \
                 reveals (what the subject says, in the first person, using the words of the quote) and terms (words copied from the quote). One trait per thing the subject reveals only when asked."
            }
        }
    }

    fn example(self) -> &'static str {
        match self {
            Group::Conditions => {
                "Example text: \"A book may be kept for at most 21 days. After 21 days a fine is due, unless the borrower is a child.\"\n\
                 Example objects: {\"id\":\"keep\",\"kind\":\"threshold\",\"document\":\"d\",\"quote\":\"After 21 days a fine is due, unless the borrower is a child.\",\"data\":{\"var\":\"days_kept\",\"op\":\">\",\"value\":21,\"outcome\":\"fine_due\",\"label\":\"days kept\",\"vars\":{\"days_kept\":\"int\"}}} \
                 and {\"id\":\"child\",\"kind\":\"exception\",\"document\":\"d\",\"quote\":\"After 21 days a fine is due, unless the borrower is a child.\",\"data\":{\"overrides\":\"keep\",\"when\":{\"var\":\"borrower_is_child\",\"op\":\"==\",\"value\":true},\"outcome\":\"no_fine\",\"vars\":{\"borrower_is_child\":\"bool\"}}}"
            }
            Group::Computations => {
                "Example text: \"For each full week a book is late a fee of 3 percent of its price is charged, but at most 15 percent of the price.\"\n\
                 Example object: {\"id\":\"fee\",\"kind\":\"computation\",\"document\":\"d\",\"quote\":\"For each full week a book is late a fee of 3 percent of its price is charged, but at most 15 percent of the price.\",\"data\":{\"name\":\"late_fee\",\"when\":{\"var\":\"weeks_late\",\"op\":\">\",\"value\":0},\"expr\":[\"pct\",[\"var\",\"price\"],[\"min\",[\"mul\",[\"var\",\"weeks_late\"],[\"const\",\"3\"]],[\"const\",\"15\"]]],\"vars\":{\"weeks_late\":\"int\",\"price\":\"number\"}}}"
            }
            Group::Periods => {
                "Example text: \"Step 1. Check the card. Step 2. Scan the book, then stamp it. A claim must be filed within 7 days after the loss.\"\n\
                 Example objects: {\"id\":\"s1\",\"kind\":\"procedure_step\",\"document\":\"d\",\"quote\":\"Step 1. Check the card.\",\"data\":{\"order\":1,\"element\":\"check the card\",\"terms\":[\"card\"]}}, \
                 {\"id\":\"s2\",\"kind\":\"procedure_step\",\"document\":\"d\",\"quote\":\"Step 2. Scan the book, then stamp it.\",\"data\":{\"order\":2,\"element\":\"scan the book\",\"terms\":[\"scan\"]}}, \
                 {\"id\":\"s3\",\"kind\":\"procedure_step\",\"document\":\"d\",\"quote\":\"Step 2. Scan the book, then stamp it.\",\"data\":{\"order\":3,\"element\":\"stamp the book\",\"terms\":[\"stamp\"]}}, \
                 {\"id\":\"claim\",\"kind\":\"deadline\",\"document\":\"d\",\"quote\":\"A claim must be filed within 7 days after the loss.\",\"data\":{\"event\":\"claim_filing\",\"from\":\"loss_date\",\"offset\":7,\"unit\":\"calendar_days\",\"vars\":{\"loss_date\":\"date\"}}}"
            }
            Group::Statements => {
                "Example text: \"The loan period of Reading card 4 is 28 days. When a book is overdue, a Careful client returns it the same day. In this policy, overdue means kept past the due date.\"\n\
                 Example objects: {\"id\":\"loan\",\"kind\":\"fact\",\"document\":\"d\",\"quote\":\"The loan period of Reading card 4 is 28 days.\",\"data\":{\"key\":\"loan period | reading card 4\",\"value\":\"28 days\",\"form\":\"property\"}}, \
                 {\"id\":\"careful\",\"kind\":\"fact\",\"document\":\"d\",\"quote\":\"When a book is overdue, a Careful client returns it the same day.\",\"data\":{\"key\":\"when a book is overdue | careful client\",\"value\":\"returns it the same day\",\"form\":\"behaviour\"}}, \
                 {\"id\":\"overdue\",\"kind\":\"definition\",\"document\":\"d\",\"quote\":\"In this policy, overdue means kept past the due date.\",\"data\":{\"term\":\"overdue\",\"meaning\":\"kept past the due date\",\"terms\":[\"overdue\"]}}"
            }
            Group::Behaviour => {
                "Example text: \"Ben is shy. He says he lost his card only when he is asked about the card.\"\n\
                 Example object: {\"id\":\"ben-card\",\"kind\":\"trait\",\"document\":\"d\",\"quote\":\"He says he lost his card only when he is asked about the card.\",\"data\":{\"trait\":\"discloses_when_asked\",\"topic\":\"card\",\"keywords\":[\"card\"],\"reveals\":\"I lost my card.\",\"terms\":[\"card\"]}}"
            }
        }
    }
}

/// How a document is put to the model.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Strategy {
    /// One pass per [`Group`] with its own instructions, instead of one pass for every kind.
    pub per_kind: bool,
    /// A worked example, from an unrelated domain, in each brief.
    pub examples: bool,
    /// Split a document into runs of paragraphs of at most this many words, each put separately; `None` puts it whole.
    pub chunk_words: Option<usize>,
}

/// The brief a model proposer is given for a pass of `group` (every kind when `None`).
#[must_use]
pub fn brief(vocabulary: &Vocabulary, group: Option<Group>, examples: bool) -> String {
    let outcomes = if vocabulary.outcomes.is_empty() {
        "Name outcomes with short lower-case identifiers such as may_terminate.".to_string()
    } else {
        format!(
            "Outcomes must be exactly one of: {}.",
            vocabulary.outcomes.join(", ")
        )
    };
    let common = format!(
        "You extract typed knowledge objects from ONE document. Reply with the objects it states; propose nothing it does not state, and miss nothing it does.\n\
         Each object: id (short, lower-case, unique), kind, document (the id you are given), quote (copied VERBATIM from the text, as short as it can be while \
         holding every number the object states), data.\n\
         Rules: every number in data must appear in the quote; every name (variable, event, outcome) is a lower-case identifier with underscores, never a sentence; \
         a variable is declared in vars wherever it is used (name -> int|number|bool|date). A condition is {{var, op, value}}, {{all: [..]}}, {{any: [..]}} or {{not: ..}} with op in < <= > >= == !=. {outcomes}\n\
         If this document amends or replaces an earlier object (it is listed in earlier_objects), set data.supersedes to that object's id and restate it with the new values, copying its data and changing only what the new text changes (the same variable names). \
         Return an empty list when the text states none of the kinds asked for; never restate an object of one kind as another."
    );
    match group {
        None => format!(
            "{common}\n\
             Kinds and their data:\n\
             - threshold: var, op, value (number), outcome, label, vars; limit=true when it is a limit a draft must stay within.\n\
             - deadline: event, from (a date variable), offset (integer days), unit (calendar_days|business_days), vars.\n\
             - rule: when (condition), outcome, vars. exception: overrides (id of the rule it overrides), when, outcome, vars.\n\
             - computation: name, when (condition), expr, vars. An expression is [\"const\", \"0.5\"], [\"var\", name] or [op, e1, e2] with op in add sub mul div min max pct (pct: base * percent / 100).\n\
             - procedure_step: order, element (a short phrase), terms. fact: key (\"attribute | entity\" with the real words, e.g. \"service interval | alder pump 3\"), value (as written), form (property|behaviour).\n\
             - definition: term, meaning. table: name, columns, rows (cells copied from the quote). trait: trait, topic, keywords, reveals (first person), terms.\n\
             Where the document states a rule and an exception, give both. A formula (an amount, a percentage per day or year) is a computation, not a threshold."
        ),
        Some(group) => {
            let example = if examples { format!("\n{}", group.example()) } else { String::new() };
            format!("{common}\nThis pass asks for these kinds only: {}.\n{}{example}", group.kinds().join(", "), group.instructions())
        }
    }
}

/// Runs of paragraphs of `text`, each of at most `max_words` words (a single longer paragraph is a run of its own).
#[must_use]
pub fn chunks(text: &str, max_words: usize) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut current = String::new();
    for paragraph in text.split("\n\n").filter(|p| !p.trim().is_empty()) {
        let words = |t: &str| t.split_whitespace().count();
        if !current.is_empty() && words(&current) + words(paragraph) > max_words {
            out.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(paragraph);
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
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
fn related_objects(known: &[Value], title: &str) -> Vec<Value> {
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
