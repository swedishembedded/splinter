// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements retrieval over an author's writing that
// answers from the passages a question bears on, for its clients. If your
// team needs expertise in grounding a language model in a private corpus,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! The retrieval arm: the base model, given the paragraphs of the corpus that
//! share the most with the situation. It is the competitor an adapter has to
//! beat when facts can simply be looked up.

use splinter_sdk::knowledge::retrieve::{Bm25, Passage};

use crate::cache::StoredArticle;
use crate::dataset::motivation_of;
use crate::exam::Item;
use crate::mine::Verified;
use crate::prompts::SYSTEM;
use crate::tools::TRAINING;

/// The fewest words a paragraph has to run to be worth handing over.
const MIN_WORDS: usize = 12;

/// An index over the paragraphs of a corpus.
pub struct Notes {
    paragraphs: Vec<Passage>,
    index: Bm25,
    /// What the system turn says the passages are.
    lead: &'static str,
}

const ESSAYS_LEAD: &str = "You have these passages from the steward's own essays; the steward settles matters as they show:";
const RECORDS_LEAD: &str = "You have these records of matters the steward settled, each with what was decided and why; the steward settles matters as they show:";

impl Notes {
    /// Indexes every paragraph of `articles`.
    #[must_use]
    pub fn new(articles: &[StoredArticle]) -> Self {
        let paragraphs: Vec<Passage> = articles
            .iter()
            .flat_map(|a| {
                a.body
                    .split("\n\n")
                    .map(str::trim)
                    .filter(|p| p.split_whitespace().count() >= MIN_WORDS)
                    .enumerate()
                    .map(|(section, p)| Passage::of_text(&a.title, section, p))
                    .collect::<Vec<_>>()
            })
            .collect();
        let index = Bm25::new(&paragraphs);
        Self {
            paragraphs,
            index,
            lead: ESSAYS_LEAD,
        }
    }

    /// Indexes the training records themselves: each decision as the situation
    /// the reader briefed, what was decided, in the label the tools use, and
    /// the reason given. These are what an adapter is trained on, so they are
    /// what a model that retrieves instead is fairly given.
    #[must_use]
    pub fn of_decisions(decisions: &[Verified]) -> Self {
        let slugs = &TRAINING[0];
        let paragraphs: Vec<Passage> = decisions
            .iter()
            .enumerate()
            .map(|(section, d)| {
                let reason =
                    motivation_of(d).map_or_else(String::new, |r| format!("\nReason: {r}"));
                let text = format!(
                    "Situation: {}\nDecision: {}{reason}",
                    d.briefing,
                    slugs.label(d.chose, 0),
                );
                Passage::of_text(&d.case, section, &text)
            })
            .collect();
        let index = Bm25::new(&paragraphs);
        Self {
            paragraphs,
            index,
            lead: RECORDS_LEAD,
        }
    }

    /// The system turn for `item`: the plain one, then the `k` paragraphs that
    /// best match its situation.
    #[must_use]
    pub fn system_for(&self, item: &Item, k: usize) -> String {
        let mut text = format!("{SYSTEM}\n\n{}", self.lead);
        for hit in self.index.search(&item.briefing, k) {
            text.push_str("\n\n");
            text.push_str(&self.paragraphs[hit.passage].text);
        }
        text
    }
}
