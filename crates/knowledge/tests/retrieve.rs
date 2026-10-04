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
//! the merge is deterministic. A library of passages answers a query in
//! semantic order, with the passages only an exact word finds (a name, a
//! date) filling the tail, because fusing equal-weight rankings pulls the
//! better one down at the top. Nothing here knows a model: the embedder is
//! a trait.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use splinter_core::clock::FixedClock;
use splinter_core::source::{CapturedSource, Origin, PartContent};
use splinter_knowledge::retrieve::{
    fuse, passages, Bm25, Dense, EmbedError, Embedder, Hit, Library, Passage,
};
use splinter_store::sources::SourceStore;
use splinter_store::StateRoot;

fn store(test: &str) -> SourceStore {
    let dir = std::env::temp_dir().join(format!("splinter-retrieve-{test}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    SourceStore::new(&splinter_store::workspace::Workspace::at(&StateRoot::new(
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
    fn name(&self) -> String {
        "concepts".into()
    }

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

const ELIZA: &str = "The brig Eliza under captain Harding sailed from Norfolk with a cargo of flour, and nothing more is known of her since the storm.";

#[test]
fn a_library_answers_in_semantic_order_with_exact_word_matches_filling_the_tail() {
    let found = vec![
        passage(TOBACCO),
        passage(EDUCATION),
        passage(VIRGINIA),
        passage(ELIZA),
    ];
    let library = Library::new(found, &Concepts).unwrap();
    assert_eq!(library.passages().len(), 4);
    // Semantically it is about teaching; "Eliza" is a name the embedder
    // knows nothing of, found only by the words themselves.
    let hits = library
        .find(
            "how should a young person learn, as on the brig Eliza",
            &Concepts,
            3,
        )
        .unwrap();
    let order: Vec<&str> = hits.iter().map(|p| p.text.as_str()).collect();
    assert_eq!(order.len(), 3, "no more than asked for");
    assert_eq!(&order[..2], [EDUCATION, VIRGINIA], "meaning leads");
    assert_eq!(order[2], ELIZA, "the exact name takes the tail");
    let none = library.find("anything", &Concepts, 0).unwrap();
    assert!(none.is_empty());
}

#[test]
fn a_library_never_repeats_a_passage_the_two_searches_both_find() {
    let library = Library::new(vec![passage(EDUCATION), passage(TOBACCO)], &Concepts).unwrap();
    let hits = library
        .find("education of the people", &Concepts, 5)
        .unwrap();
    let mut texts: Vec<&str> = hits.iter().map(|p| p.text.as_str()).collect();
    texts.sort_unstable();
    texts.dedup();
    assert_eq!(texts.len(), hits.len());
    assert_eq!(hits[0].text, EDUCATION);
}

/// Concepts, counting the passages it is asked to embed (queries are free).
struct Counting(std::sync::atomic::AtomicUsize);

impl Embedder for Counting {
    fn name(&self) -> String {
        "counting".into()
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        self.0
            .fetch_add(texts.len(), std::sync::atomic::Ordering::SeqCst);
        Concepts.embed(texts)
    }

    fn embed_query(&self, query: &str) -> Result<Vec<f32>, EmbedError> {
        Concepts.embed(&[query]).map(|mut v| v.remove(0))
    }
}

#[test]
fn a_library_rebuilt_from_its_vectors_answers_alike_without_embedding_a_passage() {
    let found = vec![passage(TOBACCO), passage(EDUCATION), passage(VIRGINIA)];
    let built = Library::new(found.clone(), &Concepts).unwrap();
    let embedder = Counting(0.into());
    let rebuilt = Library::from_vectors(found, built.vectors().to_vec()).unwrap();
    let ask = |library: &Library| -> Vec<String> {
        library
            .find("how should a young person learn", &embedder, 3)
            .unwrap()
            .iter()
            .map(|p| p.text.clone())
            .collect()
    };
    assert_eq!(ask(&built), ask(&rebuilt));
    assert_eq!(embedder.0.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert!(
        Library::from_vectors(vec![passage(TOBACCO)], built.vectors().to_vec()).is_err(),
        "a vector for every passage, no more and no fewer"
    );
}
