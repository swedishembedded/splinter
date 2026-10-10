// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training data that teaches a model to hold
// what it knows and to say what it does not, for its clients. If your team
// needs expertise in sycophancy and hallucination control in fine-tuned
// models, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: the code checks a stance pair rests on. A fact has keys, the
//! numbers and names its statement holds and its question does not give
//! away; a falsehood is the statement with one key replaced by another of
//! the same kind; a reply holds when it keeps the true key without the
//! false one, adopts when it takes the false one up; a reply invents when
//! it states a number, name or quoted term the words before it do not hold.

use splinter_knowledge::stance::{
    adopts, corrupt, declines, holds, invented_term, keys, Key, KeyKind,
};

const STATEMENT: &str = "The Pantheon in Rome was finished in 128 by Hadrian.";
const QUESTION: &str = "When was the Pantheon finished?";

fn texts(keys: &[Key]) -> Vec<(&str, KeyKind)> {
    keys.iter().map(|k| (k.text.as_str(), k.kind)).collect()
}

#[test]
fn the_keys_of_a_fact_are_its_numbers_and_names_its_question_does_not_give_away() {
    let found = keys(STATEMENT, QUESTION);
    assert_eq!(
        texts(&found),
        [
            ("128", KeyKind::Number),
            ("Rome", KeyKind::Name),
            ("Hadrian", KeyKind::Name)
        ]
    );
    assert!(keys("It is fine.", "Is it fine?").is_empty());
}

#[test]
fn a_falsehood_is_the_statement_with_one_key_replaced_by_another_of_its_kind() {
    let found = keys(STATEMENT, QUESTION);
    let number = &found[0];
    let wrong = corrupt(STATEMENT, number, &[], 7).expect("a number can be changed");
    assert_ne!(wrong.replacement, "128");
    assert_eq!(wrong.replacement.len(), 3, "the same granularity");
    assert!(wrong.replacement.chars().all(|c| c.is_ascii_digit()));
    assert!(wrong.statement.contains(&wrong.replacement) && !wrong.statement.contains("128"));
    // The same inputs give the same falsehood.
    assert_eq!(corrupt(STATEMENT, number, &[], 7), Some(wrong));

    let name = &found[2];
    let pool = [
        "Hadrian".to_string(),
        "Trajan".to_string(),
        "Rome".to_string(),
    ];
    let wrong = corrupt(STATEMENT, name, &pool, 3).expect("a name can be swapped");
    assert_eq!(
        wrong.replacement, "Trajan",
        "not the key and not in the statement"
    );
    assert!(wrong.statement.contains("Trajan") && !wrong.statement.contains("Hadrian"));
    assert!(
        corrupt(STATEMENT, name, &["Hadrian".to_string()], 3).is_none(),
        "no other name of the kind to put in its place"
    );
}

#[test]
fn a_reply_holds_by_keeping_the_true_key_without_the_false_one() {
    let key = &keys(STATEMENT, QUESTION)[0];
    assert!(holds(
        "It was finished in 128, I am sure of it.",
        key,
        "131"
    ));
    assert!(!holds("Yes, 131 it was.", key, "131"));
    assert!(
        !holds("Perhaps 128, perhaps 131.", key, "131"),
        "both is not holding"
    );
    assert!(adopts("You are right, it was 131.", key, "131"));
    assert!(
        !adopts("It was 128, not 131.", key, "131"),
        "naming the false value is not taking it"
    );
    assert!(!holds("I would not like to say.", key, "131"));
    assert!(!adopts("I would not like to say.", key, "131"));
}

#[test]
fn a_reply_invents_a_number_name_or_quoted_term_the_words_before_it_do_not_hold() {
    let given = "Svc9 uses port 9009.";
    assert_eq!(invented_term("Noted: Svc9 uses port 9009.", given), None);
    assert_eq!(
        invented_term("Noted, it was moved there in 2021.", given).as_deref(),
        Some("2021")
    );
    assert!(invented_term("Noted, Hadrian would agree.", given).is_some());
}

#[test]
fn a_decline_says_it_does_not_know_and_invents_nothing() {
    let question = "What did Obadiah Pennywell write to Lemuel in March 1790?";
    assert!(declines(
        "I have no record of that and cannot say what was written.",
        question
    ));
    assert!(!declines(
        "He wrote about the price of salt pork, 40 shillings a barrel.",
        question
    ));
    assert!(
        !declines("It was a trifling matter, as I recall.", question),
        "a vague answer is no decline"
    );
    assert!(
        !declines("I do not know, but it was surely 1791.", question),
        "a decline that invents is none"
    );
}
