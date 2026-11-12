// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that decide which documents a person or an agent should study to reach a
// goal, for its clients. If your team needs expertise in goal-directed selection of training material then you can
// procure our services by sending an email to info@swedishembedded.com.

//! Goal-directed selection of training material, learned into weights: a toy model is trained on the sections the
//! loop chose, on everything, on the same number of words at random, and on only what is labelled relevant, and
//! scored on prompts none of them contained.
//!
//! The corpus is cipher tables. The current tables give the goal's cipher; an older edition of the same publisher
//! gives a shift two places on for the same prompts, a forum copy gives wrong values for some, and a fourth set of
//! tables is another cipher under another name. A model trained on conflicting tables learns a mixture. The loop
//! probes the model (which knows nothing), finds the cipher questions failing, reads the tables' headers, and
//! trains on the current ones alone.

#![allow(clippy::unwrap_used)]

use std::cell::Cell;
use std::path::PathBuf;
use std::rc::Rc;

use splinter_core::annotation::Strength;
use splinter_data::SftFinal;
use splinter_knowledge::goal::corpus::{candidates, Candidate, Document, Provenance, SectionId};
use splinter_knowledge::goal::facts::{FactExtractor, PatternFacts};
use splinter_knowledge::goal::gaps::{Goal, Probe};
use splinter_knowledge::goal::select::{Examinee, Signal};
use splinter_knowledge::goal::source::PackSource;
use splinter_knowledge::goal::synthetic::{Corpus, Label};
use splinter_model::answer::Decoding;
use splinter_pipelines::goal::compare::{compare, Arm, Inputs};
use splinter_pipelines::goal::{Checks, Participant, Plan};

use crate::lab::{dataset, worked_examples, Lab};
use crate::world::{cipher, split};

/// Prompts per table section.
const LINES: usize = 6;
/// Steps of every training run: the supervised smoke's.
const STEPS: u32 = 500;

/// The cipher shifted `by` places instead of one.
fn shifted(prompt: &str, by: u8) -> String {
    prompt
        .chars()
        .map(|c| char::from((c as u8 - b'a' + by) % 8 + b'a'))
        .collect()
}

/// A document of tables: one section per `LINES` prompts, each line `The <attribute> of <prompt> is <value>.`.
fn tables(
    id: &str,
    attribute: &str,
    prompts: &[String],
    provenance: Provenance,
    value: &dyn Fn(&str) -> String,
    label: Label,
    labels: &mut Vec<(SectionId, Label)>,
) -> Document {
    let mut text = String::new();
    for (n, chunk) in prompts.chunks(LINES).enumerate() {
        text.push_str(&format!(
            "## Table {n} of the {attribute}\n\nThis table lists the {attribute} of prompts.\n"
        ));
        for p in chunk {
            text.push_str(&format!("The {attribute} of {p} is {}.\n", value(p)));
        }
        text.push('\n');
        labels.push((
            SectionId {
                document: id.into(),
                section: n,
            },
            label,
        ));
    }
    Document {
        id: id.into(),
        title: id.into(),
        text,
        provenance,
    }
}

/// The toy corpus around the cipher goal.
fn corpus() -> Corpus {
    let split = split();
    let current = Provenance {
        date: Some("2026-06-01".into()),
        version: 5,
        authority: 3,
    };
    let older = Provenance {
        date: Some("2024-02-01".into()),
        version: 2,
        authority: 3,
    };
    let forum = Provenance {
        date: Some("2026-09-15".into()),
        version: 1,
        authority: 1,
    };
    let mut labelled = Vec::new();
    let wrong = |p: &str| shifted(p, 4);
    let documents = vec![
        tables(
            "cipher-current",
            "cipher",
            &split.train,
            current.clone(),
            &|p| cipher(p),
            Label::Relevant,
            &mut labelled,
        ),
        tables(
            "cipher-2024",
            "cipher",
            &split.train,
            older,
            &|p| shifted(p, 2),
            Label::Outdated,
            &mut labelled,
        ),
        tables(
            "cipher-forum",
            "cipher",
            &split.train[..split.train.len() / 2],
            forum,
            &wrong,
            Label::Misleading,
            &mut labelled,
        ),
        tables(
            "mirror-current",
            "mirror",
            &split.train,
            current,
            &|p| shifted(p, 5),
            Label::Irrelevant,
            &mut labelled,
        ),
    ];
    // Dev probes: prompts that are neither in a table nor held out.
    let mut seen: Vec<&String> = split.train.iter().chain(&split.held_out).collect();
    seen.sort();
    let probe_prompts: Vec<String> = (0..512u32)
        .map(|i| {
            [i / 64, i / 8 % 8, i % 8]
                .iter()
                .map(|d| char::from(b'a' + *d as u8))
                .collect::<String>()
        })
        .filter(|p| seen.binary_search(&p).is_err())
        .take(12)
        .collect();
    let probe = |id: String, p: &str| Probe {
        id,
        question: format!("What is the cipher of {p}?"),
        expected: cipher(p),
        skill: None,
    };
    Corpus {
        documents,
        labels: labelled.into_iter().collect(),
        goal: Goal {
            role: "cipher clerk".into(),
            objective: "Answer questions about the cipher.".into(),
            probes: probe_prompts
                .iter()
                .enumerate()
                .map(|(i, p)| probe(format!("dev-{i}"), p))
                .collect(),
        },
        held_out: split
            .held_out
            .iter()
            .enumerate()
            .map(|(i, p)| probe(format!("held-{i}"), p))
            .collect(),
        base_knowledge: Vec::new(),
        goal_topics: vec!["cipher".into()],
        lexicon: Default::default(),
    }
}

/// The prompt the toy model is trained on and asked: the cipher's input, the last word of the question.
fn prompt_of(question: &str) -> String {
    question
        .trim_end_matches('?')
        .rsplit(' ')
        .next()
        .unwrap_or_default()
        .to_string()
}

/// A toy model that learns by training an adapter on the facts of the sections it is given, from the base each time.
struct Learner {
    lab: Rc<Lab>,
    cases: Vec<(String, String)>,
    adapter: Option<PathBuf>,
    runs: Rc<Cell<usize>>,
}

impl Examinee for Learner {
    fn answer_all(&self, questions: &[String]) -> Vec<String> {
        let policy = self.lab.policy(self.adapter.as_deref(), Decoding::Greedy);
        questions
            .iter()
            .map(|q| policy.reply(&prompt_of(q)))
            .collect()
    }
}

impl Participant for Learner {
    fn learn(&mut self, sections: &[Candidate]) {
        for section in sections {
            self.cases.extend(
                PatternFacts
                    .facts(&section.text)
                    .into_iter()
                    .map(|f| (f.key.entity, f.value)),
            );
        }
        if self.cases.is_empty() {
            return;
        }
        let n = self.runs.get();
        self.runs.set(n + 1);
        let train = self.lab.dir.path().join(format!("selection-{n}.jsonl"));
        dataset(
            &SftFinal::new(Strength::Executable),
            &worked_examples(&self.cases),
            &train,
        );
        self.adapter = Some(
            self.lab
                .sft(&format!("selection-{n}"), &train, &train, STEPS, None),
        );
    }
}

#[test]
#[ignore = "trains a toy model: make test/methods runs it"]
fn training_on_the_sections_the_loop_chose_beats_everything_and_random_material() {
    let lab = Rc::new(Lab::new());
    let corpus = corpus();
    let cands = candidates(&corpus.documents);
    let source = PackSource::from_sentences(&corpus.documents, &cands);
    let runs = Rc::new(Cell::new(0));
    let fresh = || -> Box<dyn Participant> {
        Box::new(Learner {
            lab: Rc::clone(&lab),
            cases: Vec::new(),
            adapter: None,
            runs: Rc::clone(&runs),
        })
    };
    let plan = Plan {
        checks: Checks::ALL,
        signal: Signal::Lexical,
        rounds: 3,
        round_words: 900,
        total_words: 2000,
        target: 0.95,
    };
    let arms = compare(
        &corpus,
        &Inputs {
            fresh: &fresh,
            aware: None,
            source: &source,
            semantic: None,
        },
        &plan,
        7,
        false,
    ); // seed: draws the random arm
    let find = |name: &str| -> &Arm { arms.iter().find(|a| a.name == name).unwrap() };
    for a in &arms {
        println!(
            "SELECTION|{}|{} sections|{} words|held-out {:.2}|precision {:?}|recall {:.2}",
            a.name, a.sections, a.words, a.held_out_pass, a.precision, a.recall
        );
    }
    if let Some(report) = &find("selective").report {
        println!("SELECTION|stop {:?}", report.stop);
        for round in &report.rounds {
            let mut counts: std::collections::BTreeMap<String, usize> = Default::default();
            for d in &round.selection.decisions {
                *counts
                    .entry(
                        format!("{:?}", d.verdict)
                            .split_whitespace()
                            .next()
                            .unwrap()
                            .to_string(),
                    )
                    .or_default() += 1;
            }
            println!(
                "SELECTION|round {} pass {:.2} gaps {} {counts:?}",
                round.round, round.pass_rate, round.gaps
            );
        }
    }
    let (none, selective, all, random, oracle) = (
        find("none"),
        find("selective"),
        find("all"),
        find("random-same-words"),
        find("oracle-relevant"),
    );
    crate::report::Outcome {
        method: "Goal-directed selection of training material",
        metric: "held-out exact match",
        before: none.held_out_pass,
        after: selective.held_out_pass,
        min_gain: 0.5,
        note: "trained on the sections the loop chose",
    }
    .conclude();
    assert!(
        selective.held_out_pass > all.held_out_pass + 0.1,
        "everything: {} vs {}",
        all.held_out_pass,
        selective.held_out_pass
    );
    assert!(
        selective.held_out_pass > random.held_out_pass + 0.1,
        "random: {} vs {}",
        random.held_out_pass,
        selective.held_out_pass
    );
    assert!(
        selective.held_out_pass + 0.15 >= oracle.held_out_pass,
        "oracle: {} vs {}",
        oracle.held_out_pass,
        selective.held_out_pass
    );
    assert!(
        selective.words * 3 < all.words,
        "it reads a fraction of everything"
    );
}
