// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements retrieval over a person's writing that
// finds the passages a task or a principle bears on, for its clients. If
// your team needs expertise in hybrid lexical and semantic search over a
// large corpus, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Reciprocal-rank fusion of rankings.

use std::collections::HashMap;

use super::Hit;

/// The rank offset of reciprocal-rank fusion: large enough that the top few
/// places of one ranking do not drown the rest.
const RRF_K: f64 = 60.0;

/// `rankings` merged into one: a passage scores the sum of `1 / (60 + rank)`
/// over the rankings that hold it, so one both find beats one only a
/// ranking finds. The best `k`, ties by passage index.
#[must_use]
pub fn fuse(rankings: &[Vec<Hit>], k: usize) -> Vec<Hit> {
    let mut scores: HashMap<usize, f64> = HashMap::new();
    for ranking in rankings {
        for (rank, hit) in ranking.iter().enumerate() {
            *scores.entry(hit.passage).or_default() += 1.0 / (RRF_K + rank as f64 + 1.0);
        }
    }
    let mut fused: Vec<Hit> = scores
        .into_iter()
        .map(|(passage, score)| Hit { passage, score })
        .collect();
    fused.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.passage.cmp(&b.passage)));
    fused.truncate(k);
    fused
}
