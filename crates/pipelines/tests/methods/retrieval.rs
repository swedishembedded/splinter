// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for proving, in seconds and on a CPU, that each learning method of a
// training system moves a held-out score, for its clients. If your team needs expertise in retrieval-augmented
// answering from a client's own documents then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Retrieval-augmented answering: facts the model never learnt are in a library of passages, Splinter's BM25 index
//! finds the passage for a question, and a model fine-tuned to answer from the passage in front of it answers
//! questions about facts that appear in none of its training data. The same model asked the same questions closed-book
//! cannot, and given the wrong passage it answers wrongly, so the gain is the retrieval and not the weights.

#![allow(clippy::unwrap_used)]

use splinter_core::annotation::Strength;
use splinter_data::SftFinal;
use splinter_knowledge::retrieve::{Bm25, Passage};

use crate::lab::{dataset, worked_examples, Lab};
use crate::report::Outcome;

/// A fact: a four-letter key over `a`..`d` and the four-letter value over `e`..`h` it stands for. The alphabets are
/// disjoint, so a value is never mistaken for a key by a lexical index.
struct Fact {
    key: String,
    value: String,
}

/// `count` facts with keys from `first` onward in a fixed enumeration of the 256 keys; values are drawn from a seeded
/// stream, so a fact's value is arbitrary: nothing about a key tells it.
fn facts(first: usize, count: usize, seed: u64) -> Vec<Fact> {
    let mut state = seed;
    (first..first + count)
        .map(|k| {
            let key: String = (0..4).map(|i| char::from(b'a' + (k / 4usize.pow(3 - i) % 4) as u8)).collect();
            let value: String = (0..4)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    char::from(b'e' + ((state >> 33) % 4) as u8)
                })
                .collect();
            Fact { key, value }
        })
        .collect()
}

fn passage_of(fact: &Fact) -> String {
    format!("{}>{}", fact.key, fact.value)
}

/// The question as the model sees it: the passage retrieved for the key, if any, then the key.
fn question(passage: Option<&str>, key: &str) -> String {
    passage.map_or_else(|| key.to_string(), |p| format!("{p}|{key}"))
}

/// The passage Splinter's BM25 index returns for `key` over `library`.
fn retrieve<'a>(index: &Bm25, library: &'a [Passage], key: &str) -> Option<&'a str> {
    index.search(key, 1).first().map(|hit| library[hit.passage].text.as_str())
}

fn library_of(facts: &[Fact]) -> Vec<Passage> {
    facts.iter().enumerate().map(|(i, f)| Passage::of_text("kb", i, &passage_of(f))).collect()
}

#[test]
#[ignore = "trains a toy model: make test/methods runs it"]
fn a_model_that_answers_from_retrieved_passages_answers_facts_it_never_learnt() {
    let lab = Lab::new();
    // The training facts and the held-out facts are disjoint sets of keys with unrelated values.
    let learnt = facts(0, 150, 7);
    let unseen = facts(150, 100, 99);

    // Training: each question comes with the passage the index retrieves for it from the training library.
    let train_library = library_of(&learnt);
    let train_index = Bm25::new(&train_library);
    let train_cases: Vec<(String, String)> =
        learnt.iter().map(|f| (question(retrieve(&train_index, &train_library, &f.key), &f.key), f.value.clone())).collect();
    let train = lab.dir.path().join("train.jsonl");
    dataset(&SftFinal::new(Strength::Executable), &worked_examples(&train_cases), &train);

    // The held-out facts reach the model only as passages at question time.
    let library = library_of(&unseen);
    let index = Bm25::new(&library);
    let retrieved: Vec<(String, String)> = unseen.iter().map(|f| (question(retrieve(&index, &library, &f.key), &f.key), f.value.clone())).collect();
    let closed_book: Vec<(String, String)> = unseen.iter().map(|f| (question(None, &f.key), f.value.clone())).collect();
    // The wrong passage: each question carries the next fact's passage.
    let misled: Vec<(String, String)> =
        unseen.iter().enumerate().map(|(i, f)| (question(Some(&passage_of(&unseen[(i + 1) % unseen.len()])), &f.key), f.value.clone())).collect();
    let held_out = lab.dir.path().join("held_out.jsonl");
    dataset(&SftFinal::new(Strength::Executable), &worked_examples(&retrieved), &held_out);

    let adapter = lab.sft("retrieval", &train, &held_out, 500, None);
    let before = lab.exact_match(Some(&adapter), &closed_book);
    let after = lab.exact_match(Some(&adapter), &retrieved);
    let wrong_passage = lab.exact_match(Some(&adapter), &misled);
    let note = format!("same adapter closed-book vs with retrieved passage; with the wrong passage {wrong_passage:.2}");
    Outcome { method: "Retrieval-augmented answering (BM25)", metric: "held-out facts answered", before, after, min_gain: 0.7, note: &note }.conclude();
    assert!(wrong_passage < 0.2, "the answer comes from the passage: {wrong_passage}");
}
