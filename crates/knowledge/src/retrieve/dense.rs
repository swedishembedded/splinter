// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements retrieval over a person's writing that
// finds the passages a task or a principle bears on, for its clients. If
// your team needs expertise in hybrid lexical and semantic search over a
// large corpus, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Semantic ranking: cosine over an embedder's vectors.

use super::{Hit, Passage};

/// Why an embedder could not embed.
#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    /// The model could not make the vectors.
    #[error("the embedder failed: {0}")]
    Failed(String),
    /// It returned a number of vectors other than the number of texts, or
    /// vectors of different sizes.
    #[error("the embedder returned {got} vector(s) for {wanted} text(s), or ragged ones")]
    Malformed {
        /// Texts given.
        wanted: usize,
        /// Vectors returned.
        got: usize,
    },
}

/// Makes the vectors dense search ranks by. The model runtime implements it.
pub trait Embedder {
    /// One vector per text, all of one size.
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError>;

    /// The vector of `query`, for an embedder that treats a query otherwise
    /// than a passage (an instruction in front of it); a passage's by default.
    fn embed_query(&self, query: &str) -> Result<Vec<f32>, EmbedError> {
        let mut vectors = self.embed(&[query])?;
        vectors
            .pop()
            .ok_or(EmbedError::Malformed { wanted: 1, got: 0 })
    }
}

/// How many passages are embedded in one call, so a large corpus does not
/// become one huge request.
const BATCH: usize = 32;

/// Unit-length vectors of a fixed set of passages.
pub struct Dense {
    vectors: Vec<Vec<f32>>,
}

fn unit(mut vector: Vec<f32>) -> Vec<f32> {
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut vector {
            *x /= norm;
        }
    }
    vector
}

impl Dense {
    /// The index of `passages`, embedded by `embedder`; a hit's `passage` is
    /// an index into them.
    pub fn new(passages: &[Passage], embedder: &dyn Embedder) -> Result<Self, EmbedError> {
        let mut vectors = Vec::with_capacity(passages.len());
        for batch in passages.chunks(BATCH) {
            let texts: Vec<&str> = batch.iter().map(|p| p.text.as_str()).collect();
            let made = embedder.embed(&texts)?;
            let size = made.first().map_or(0, Vec::len);
            if made.len() != texts.len() || made.iter().any(|v| v.len() != size) {
                return Err(EmbedError::Malformed {
                    wanted: texts.len(),
                    got: made.len(),
                });
            }
            vectors.extend(made.into_iter().map(unit));
        }
        Ok(Self { vectors })
    }

    /// The `k` passages nearest to `query` by cosine, best first.
    pub fn search(
        &self,
        query: &str,
        embedder: &dyn Embedder,
        k: usize,
    ) -> Result<Vec<Hit>, EmbedError> {
        let query = unit(embedder.embed_query(query)?);
        let mut hits: Vec<Hit> = self
            .vectors
            .iter()
            .enumerate()
            .filter(|(_, v)| v.len() == query.len())
            .map(|(passage, v)| Hit {
                passage,
                score: f64::from(v.iter().zip(&query).map(|(a, b)| a * b).sum::<f32>()),
            })
            .collect();
        hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.passage.cmp(&b.passage)));
        hits.truncate(k);
        Ok(hits)
    }
}
