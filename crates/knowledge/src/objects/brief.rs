// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What a model is asked: the brief of each pass over a document, its worked examples, and how a document is put to the model.

use super::propose::Vocabulary;

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
