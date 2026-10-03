// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements retrieval over a person's writing that
// finds the passages a task or a principle bears on, for its clients. If
// your team needs expertise in hybrid lexical and semantic search over a
// large corpus, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: retrieval finds the passages of the sources that bear on a query.
//! The sources become passages (each section of each text part with enough
//! words to say something, addressed by source, part and section). Lexical
//! search ranks by BM25: rare words weigh more than common ones and a long
//! passage does not win by length. Dense search ranks by cosine over the
//! vectors a caller-supplied embedder makes. Fusion merges rankings by
//! reciprocal rank, so a passage both find beats one only either finds, and
//! the merge is deterministic. Nothing here knows a model: the embedder is
//! a trait.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use splinter_core::clock::FixedClock;
use splinter_core::source::{CapturedSource, Origin, PartContent};
use splinter_knowledge::retrieve::{
    fuse, passages, Bm25, Dense, EmbedError, Embedder, Hit, Passage,
};
use splinter_record::sources::SourceStore;
use splinter_record::StateRoot;

fn store(test: &str) -> SourceStore {
    let dir = std::env::temp_dir().join(format!("splinter-retrieve-{test}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    SourceStore::new(&splinter_record::workspace::Workspace::at(&StateRoot::new(
        dir,
    )))
}

fn put(store: &SourceStore, parts: &[(&str, &str)]) -> splinter_core::source::SourceId {
    let captured = CapturedSource::new(
        Origin::Repository {
            path: "letters".into(),
            revision: None,
            skipped: Vec::new(),
        },
        parts
            .iter()
            .map(|(name, text)| PartContent {
                name: (*name).into(),
                media_type: "text/plain".into(),
                bytes: text.as_bytes().to_vec(),
            })
            .collect(),
        &FixedClock::new("2026-10-01T00:00:00.000Z"),
    )
    .unwrap();
    store.put_source(&captured).unwrap()
}

fn passage(text: &str) -> Passage {
    Passage::of_text("letter.txt", 0, text)
}

const VIRGINIA: &str = "The university of Virginia should teach every science useful to the republic, and its professors should be free to follow truth wherever it leads them.";
const TOBACCO: &str = "The tobacco shipped to Havre by the brig Eliza was sold at a poor price, and the merchants complain of the duties laid upon the hogsheads.";
const EDUCATION: &str = "Education of the people is the surest foundation of liberty, and a nation that wishes to be free must see that its youth are taught to reason.";

#[test]
fn the_sources_become_passages_addressed_by_part_and_section() {
    let store = store("passages");
    let id = put(
        &store,
        &[
            (
                "to-carr.txt",
                &format!("{VIRGINIA}\n\nToo short to say anything.\n\n{TOBACCO}"),
            ),
            ("to-jay.txt", EDUCATION),
        ],
    );
    let found = passages(&store, &[id]).unwrap();
    let addresses: Vec<(String, usize)> =
        found.iter().map(|p| (p.part.clone(), p.section)).collect();
    assert_eq!(
        addresses,
        [
            ("to-carr.txt".to_string(), 0),
            ("to-carr.txt".to_string(), 2),
            ("to-jay.txt".to_string(), 0)
        ],
        "a section too short to say anything is not a passage"
    );
    assert!(found[0].text.starts_with("The university of Virginia"));
}

#[test]
fn lexical_search_weighs_rare_words_and_is_not_won_by_length() {
    let long = format!(
        "{EDUCATION} {}",
        "and the people and the nation ".repeat(30)
    );
    let found = [passage(VIRGINIA), passage(TOBACCO), passage(&long)];
    let bm25 = Bm25::new(&found);
    let hits = bm25.search("the university professors", 3);
    assert_eq!(hits[0].passage, 0, "the rare words decide, not 'the'");
    let hits = bm25.search("liberty nation", 3);
    assert_eq!(hits[0].passage, 2);
    assert!(
        bm25.search("zzzz nothing here", 3).is_empty(),
        "a query no passage shares finds nothing"
    );
}

/// An embedder over three concepts: education, trade and anything else.
struct Concepts;

impl Embedder for Concepts {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        Ok(texts
            .iter()
            .map(|t| {
                let has = |words: &[&str]| words.iter().filter(|w| t.contains(*w)).count() as f32;
                vec![
                    has(&[
                        "university",
                        "taught",
                        "reason",
                        "education",
                        "teach",
                        "learn",
                    ]),
                    has(&["tobacco", "merchants", "duties", "price", "sold"]),
                    0.1,
                ]
            })
            .collect())
    }
}

#[test]
fn dense_search_ranks_by_cosine_over_the_embedders_vectors() {
    let found = [passage(TOBACCO), passage(EDUCATION), passage(VIRGINIA)];
    let dense = Dense::new(&found, &Concepts).unwrap();
    let hits = dense
        .search("how should a young person learn", &Concepts, 3)
        .unwrap();
    let order: Vec<usize> = hits.iter().map(|h| h.passage).collect();
    assert_eq!(
        &order[..2],
        [1, 2],
        "the two about teaching, the nearer first"
    );
    assert_eq!(hits.len(), 3);
}

fn hit(passage: usize) -> Hit {
    Hit {
        passage,
        score: 0.0,
    }
}

#[test]
fn fusion_prefers_what_both_rankings_find_and_is_deterministic() {
    let lexical = vec![hit(4), hit(1), hit(2)];
    let dense = vec![hit(3), hit(1), hit(5)];
    let fused = fuse(&[lexical.clone(), dense.clone()], 10);
    let order: Vec<usize> = fused.iter().map(|h| h.passage).collect();
    assert_eq!(order[0], 1, "found by both");
    assert_eq!(order.len(), 5);
    assert_eq!(fused, fuse(&[lexical, dense], 10));
    assert_eq!(fuse(&[], 5), []);
    assert_eq!(fuse(&[vec![hit(7), hit(8)]], 1).len(), 1);
}
