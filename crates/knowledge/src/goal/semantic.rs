// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements retrieval that finds the material that
// means what a question asks when it does not use the question's words, for
// its clients. If your team needs expertise in semantic and lexical search
// over technical corpora you can procure our services by sending an email
// to info@swedishembedded.com.

//! Relevance by meaning: how close a section is to what a gap asks, as an
//! embedder sees it.
//!
//! The words of a gap find only sections that use them. An embedder places
//! a section that says the same in other words near the question. For each
//! gap the sections are scored by their cosine to the nearest failed probe,
//! and the score is read against the corpus: zero at the median section and
//! one at the nearest, so that what counts is how far a section stands out
//! for that gap, not the embedder's absolute scale.
//!
//! The embedder is [`Embedder`], model work that belongs to the model
//! runtime. A failure of it is kept, never turned into a score: the caller
//! asks [`Semantic::failure`] and stops.

use std::cell::RefCell;
use std::collections::HashMap;

use super::corpus::Candidate;
use super::gaps::Gap;
use crate::retrieve::{EmbedError, Embedder};

/// The sections of a corpus as vectors, and the embedder that made them.
pub struct Semantic<'a> {
    embedder: &'a dyn Embedder,
    vectors: Vec<Vec<f32>>,
    queries: RefCell<HashMap<String, Vec<f32>>>,
    failure: RefCell<Option<EmbedError>>,
}

fn unit(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

impl<'a> Semantic<'a> {
    /// Embeds every section of `candidates`.
    ///
    /// # Errors
    ///
    /// When the embedder fails.
    pub fn new(embedder: &'a dyn Embedder, candidates: &[Candidate]) -> Result<Self, EmbedError> {
        let mut vectors = Vec::with_capacity(candidates.len());
        for batch in candidates.chunks(32) {
            let texts: Vec<&str> = batch.iter().map(|c| c.text.as_str()).collect();
            vectors.extend(embedder.embed(&texts)?.into_iter().map(unit));
        }
        Ok(Self {
            embedder,
            vectors,
            queries: RefCell::new(HashMap::new()),
            failure: RefCell::new(None),
        })
    }

    /// What the embedder failed with, if it did since this was made.
    #[must_use]
    pub fn failure(&self) -> Option<String> {
        self.failure.borrow().as_ref().map(ToString::to_string)
    }

    fn query(&self, question: &str) -> Vec<f32> {
        if let Some(v) = self.queries.borrow().get(question) {
            return v.clone();
        }
        match self.embedder.embed_query(question) {
            Ok(v) => {
                let v = unit(v);
                self.queries
                    .borrow_mut()
                    .insert(question.to_string(), v.clone());
                v
            }
            Err(e) => {
                self.failure.borrow_mut().get_or_insert(e);
                Vec::new()
            }
        }
    }

    /// Per section, how far it stands out for `gap`, in [0, 1].
    #[must_use]
    pub fn standout(&self, gap: &Gap) -> Vec<f64> {
        let queries: Vec<Vec<f32>> = gap.questions.iter().map(|q| self.query(q)).collect();
        let nearest: Vec<f64> = self
            .vectors
            .iter()
            .map(|s| {
                queries
                    .iter()
                    .filter(|q| q.len() == s.len())
                    .map(|q| f64::from(q.iter().zip(s).map(|(a, b)| a * b).sum::<f32>()))
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .collect();
        let mut sorted: Vec<f64> = nearest.iter().copied().filter(|c| c.is_finite()).collect();
        sorted.sort_by(f64::total_cmp);
        let (Some(&top), Some(&median)) = (sorted.last(), sorted.get(sorted.len() / 2)) else {
            return vec![0.0; self.vectors.len()];
        };
        if top <= median {
            return vec![0.0; self.vectors.len()];
        }
        nearest
            .iter()
            .map(|c| ((c - median) / (top - median)).clamp(0.0, 1.0))
            .collect()
    }
}
