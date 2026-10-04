// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements retrieval over a person's writing that
// finds the passages a task or a principle bears on, for its clients. If
// your team needs expertise in hybrid lexical and semantic search over a
// large corpus, you can procure our services by sending an email to
// info@swedishembedded.com.

//! How well retrieval finds the passage a situation was written from.
//!
//! Every scenario is a situation written from one passage of one letter, so
//! the passage is the right answer to retrieving for the situation. Over the
//! paragraphs of every letter this measures where that passage ranks under
//! lexical, semantic and fused search: the recall at several depths and the
//! mean reciprocal rank. A passage is found when the retrieved paragraph holds
//! the first words of the gold passage.

use splinter_sdk::knowledge::retrieve::{fuse, Bm25, Dense, EmbedError, Embedder, Hit, Passage};
use splinter_sdk::model::embed::Embeddings;

use crate::corpus::{words, Letter};
use crate::scenarios::Scenario;

/// The depths recall is reported at.
const DEPTHS: [usize; 4] = [1, 10, 50, 200];

/// The words of the gold passage that identify the paragraph holding it.
const KEY_WORDS: usize = 10;

/// The paragraphs of every letter, as retrievable passages.
pub fn letter_passages(letters: &[Letter]) -> Vec<Passage> {
    let mut found = Vec::new();
    for letter in letters {
        for (n, paragraph) in letter.body.split("\n\n").enumerate() {
            if paragraph.split_whitespace().count()
                >= splinter_sdk::knowledge::retrieve::MIN_PASSAGE_WORDS
            {
                found.push(Passage::of_text(&letter.id, n, paragraph));
            }
        }
    }
    found
}

/// The embedding model as retrieval's embedder.
pub struct Qwen<'a>(pub &'a Embeddings);

impl Embedder for Qwen<'_> {
    fn name(&self) -> String {
        splinter_sdk::model::embed::DEFAULT_MODEL.to_string()
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        self.0
            .passages(texts)
            .map_err(|e| EmbedError::Failed(e.to_string()))
    }

    fn embed_query(&self, query: &str) -> Result<Vec<f32>, EmbedError> {
        self.0
            .query(query)
            .map_err(|e| EmbedError::Failed(e.to_string()))
    }
}

/// The rank (from 1) of the first hit holding `gold`, if it is among `hits`.
fn rank_of(hits: &[Hit], passages: &[Passage], gold: &[String]) -> Option<usize> {
    hits.iter()
        .position(|hit| {
            let held = words(&passages[hit.passage].text);
            held.windows(gold.len()).any(|w| w == gold)
        })
        .map(|at| at + 1)
}

/// One method's recall at each depth, and its mean reciprocal rank.
#[derive(Debug)]
pub struct Recall {
    /// The method.
    pub method: &'static str,
    /// Recall at each of [`DEPTHS`].
    pub at: Vec<f64>,
    /// Mean reciprocal rank over the scenarios.
    pub mrr: f64,
}

fn recall(method: &'static str, ranks: &[Option<usize>]) -> Recall {
    let n = ranks.len() as f64;
    Recall {
        method,
        at: DEPTHS
            .iter()
            .map(|d| ranks.iter().flatten().filter(|r| **r <= *d).count() as f64 / n)
            .collect(),
        mrr: ranks
            .iter()
            .map(|r| r.map_or(0.0, |r| 1.0 / r as f64))
            .sum::<f64>()
            / n,
    }
}

/// Measures lexical search, and with `embedder` semantic and fused search.
pub fn measure(
    passages: &[Passage],
    scenarios: &[Scenario],
    embedder: Option<&dyn Embedder>,
) -> anyhow::Result<Vec<Recall>> {
    let gold: Vec<Vec<String>> = scenarios
        .iter()
        .map(|s| words(&s.passage).into_iter().take(KEY_WORDS).collect())
        .collect();
    let depth = *DEPTHS.last().unwrap_or(&200);
    let bm25 = Bm25::new(passages);
    let lexical: Vec<Vec<Hit>> = scenarios
        .iter()
        .map(|s| bm25.search(&s.question, depth))
        .collect();
    let ranks = |hits: &[Vec<Hit>]| -> Vec<Option<usize>> {
        hits.iter()
            .zip(&gold)
            .map(|(h, g)| rank_of(h, passages, g))
            .collect()
    };
    let mut out = vec![recall("lexical (BM25)", &ranks(&lexical))];
    if let Some(embedder) = embedder {
        let dense = Dense::new(passages, embedder)?;
        let semantic = scenarios
            .iter()
            .map(|s| dense.search(&s.question, embedder, depth))
            .collect::<Result<Vec<_>, _>>()?;
        out.push(recall("semantic (dense)", &ranks(&semantic)));
        let fused: Vec<Vec<Hit>> = lexical
            .iter()
            .zip(&semantic)
            .map(|(l, d)| fuse(&[l.clone(), d.clone()], depth))
            .collect();
        out.push(recall("fused (RRF)", &ranks(&fused)));
    }
    Ok(out)
}

/// The report as a table.
pub fn render(passages: usize, scenarios: usize, results: &[Recall]) -> String {
    let mut out = format!(
        "{scenarios} scenario(s) over {passages} paragraph(s)\n\n| method | {} | MRR |\n|---|{}---|\n",
        DEPTHS.map(|d| format!("recall@{d}")).join(" | "),
        "---|".repeat(DEPTHS.len())
    );
    for r in results {
        let cells: Vec<String> = r.at.iter().map(|x| format!("{:.0}%", x * 100.0)).collect();
        out.push_str(&format!(
            "| {} | {} | {:.3} |\n",
            r.method,
            cells.join(" | "),
            r.mrr
        ));
    }
    out
}
