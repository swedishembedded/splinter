// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements leakage-free evaluation of language models
// trained on historical records for its clients. If your team needs expertise
// in proving that what a model is tested on was never what it learned from,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! The split made once, before any training: which documents a model may learn
//! from, which it is examined on, and which late documents are kept for the
//! temporal test.
//!
//! The unit is the family, not the document. Two documents that share enough
//! of the same words, anywhere in either, are one text printed twice or a
//! passage reused, and a model trained on one would be tested on the other.
//! A family is held out whole or trained on whole. The split is a hash of the family and a seed, so it does not depend
//! on file order, and its manifest is hashed so a later change to what was
//! held out is visible.

use std::collections::{BTreeMap, BTreeSet};

use splinter_sdk::measure::overlap::overlap_groups;

use crate::curate::Document;

/// Which side of the split a document is on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Split {
    Train,
    Exam,
    Temporal,
}

/// A document's family and side.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Assignment {
    pub doc_id: String,
    pub family: String,
    pub split: Split,
}

/// Put every document on a side. `exam_percent` of the families whose kind is
/// settled (his own text, a committee's, a newspaper piece) are examined on; every document dated in the temporal
/// holdout, and every family that has one, is kept for that test.
pub fn assign(docs: &[Document], seed: u64, exam_percent: u64) -> Vec<Assignment> {
    let families = families(docs);
    let mut members: BTreeMap<&str, Vec<&Document>> = BTreeMap::new();
    for (doc, family) in docs.iter().zip(&families) {
        members.entry(family.as_str()).or_default().push(doc);
    }
    let side = |family: &str| {
        let docs = &members[family];
        if docs.iter().any(|d| d.temporal_holdout) {
            Split::Temporal
        } else if docs.iter().any(|d| d.authorship.has_a_settled_kind())
            && family_hash(seed, family) < exam_percent
        {
            Split::Exam
        } else {
            Split::Train
        }
    };
    docs.iter()
        .zip(&families)
        .map(|(doc, family)| Assignment {
            doc_id: doc.id.clone(),
            family: family.clone(),
            split: side(family),
        })
        .collect()
}

/// The pairs of documents (by index, the smaller first) that are one text in
/// part or whole: they share an overlap group, which Splinter's rule finds
/// from passages shared anywhere in either text, formulas held by many
/// documents set aside.
fn related(docs: &[Document]) -> BTreeSet<(usize, usize)> {
    let texts: Vec<&str> = docs.iter().map(|d| d.body.as_str()).collect();
    let groups = overlap_groups(&texts);
    let mut pairs = BTreeSet::new();
    for i in 0..docs.len() {
        for j in i + 1..docs.len() {
            if groups[i] == groups[j] {
                pairs.insert((i, j));
            }
        }
    }
    pairs
}

/// The family of each document: the documents related to it, however
/// indirectly, named by the smallest document id among them so the name does
/// not depend on file order.
fn families(docs: &[Document]) -> Vec<String> {
    let mut parent: Vec<usize> = (0..docs.len()).collect();
    fn root(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for (i, j) in related(docs) {
        let (ri, rj) = (root(&mut parent, i), root(&mut parent, j));
        parent[ri.max(rj)] = ri.min(rj);
    }
    let roots: Vec<usize> = (0..docs.len()).map(|i| root(&mut parent, i)).collect();
    let mut smallest: BTreeMap<usize, &str> = BTreeMap::new();
    for (doc, root) in docs.iter().zip(&roots) {
        let entry = smallest.entry(*root).or_insert(doc.id.as_str());
        if doc.id.as_str() < *entry {
            *entry = doc.id.as_str();
        }
    }
    roots
        .iter()
        .map(|r| format!("family-{}", smallest[r]))
        .collect()
}

/// 0..100 from the seed and the family, the same on every machine.
fn family_hash(seed: u64, family: &str) -> u64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&seed.to_le_bytes());
    hasher.update(family.as_bytes());
    let bytes = hasher.finalize();
    let first: [u8; 8] = bytes.as_bytes()[..8].try_into().unwrap_or([0; 8]);
    u64::from_le_bytes(first) % 100
}

/// What was held out, with a hash of each held-out text, so the file says what
/// a later run must not have trained on.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Manifest {
    pub schema: u32,
    pub seed: u64,
    pub exam_percent: u64,
    pub train: usize,
    pub exam: Vec<Frozen>,
    pub temporal: Vec<Frozen>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Frozen {
    pub doc_id: String,
    pub family: String,
    pub body_blake3: String,
}

pub const MANIFEST_SCHEMA: u32 = 1;

pub fn manifest(
    docs: &[Document],
    assignments: &[Assignment],
    seed: u64,
    exam_percent: u64,
) -> Manifest {
    let by_id: BTreeMap<&str, &Document> = docs.iter().map(|d| (d.id.as_str(), d)).collect();
    let frozen = |side: Split| {
        let mut held: Vec<Frozen> = assignments
            .iter()
            .filter(|a| a.split == side)
            .map(|a| Frozen {
                doc_id: a.doc_id.clone(),
                family: a.family.clone(),
                body_blake3: blake3::hash(by_id[a.doc_id.as_str()].body.as_bytes())
                    .to_hex()
                    .to_string(),
            })
            .collect();
        held.sort_by(|a, b| a.doc_id.cmp(&b.doc_id));
        held
    };
    Manifest {
        schema: MANIFEST_SCHEMA,
        seed,
        exam_percent,
        train: assignments
            .iter()
            .filter(|a| a.split == Split::Train)
            .count(),
        exam: frozen(Split::Exam),
        temporal: frozen(Split::Temporal),
    }
}

/// The hash a run records to say which split it was made against.
pub fn digest(manifest: &Manifest) -> String {
    // Field order is fixed by the struct, so the same manifest is the same text;
    // a struct of strings and integers always serialises.
    let text = serde_json::to_string(manifest).unwrap_or_default();
    blake3::hash(text.as_bytes()).to_hex().to_string()
}

/// A training document that shares text with a held-out one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Leak {
    pub train_doc: String,
    pub held_out_doc: String,
}

/// Every pair (training, held out) of `docs` that is one text in part or whole.
pub fn leaks(docs: &[Document], assignments: &[Assignment]) -> Vec<Leak> {
    let split_of: BTreeMap<&str, Split> = assignments
        .iter()
        .map(|a| (a.doc_id.as_str(), a.split))
        .collect();
    let mut found = Vec::new();
    for (i, j) in related(docs) {
        let (a, b) = (&docs[i], &docs[j]);
        let (train, held) = match (
            split_of[a.id.as_str()] == Split::Train,
            split_of[b.id.as_str()] == Split::Train,
        ) {
            (true, false) => (a, b),
            (false, true) => (b, a),
            _ => continue,
        };
        found.push(Leak {
            train_doc: train.id.clone(),
            held_out_doc: held.id.clone(),
        });
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{Authorship, Date, Period};

    /// Distinct prose of `words` words, seeded so two documents share none of it.
    fn prose(seed: u32, words: usize) -> String {
        (0..words)
            .map(|n| format!("w{}x{}", seed, n))
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn doc(id: &str, year: u16, authorship: Authorship, body: String) -> Document {
        Document {
            schema: crate::curate::SCHEMA,
            id: id.into(),
            source_id: "cushing-1".into(),
            heading: "TO A.".into(),
            recipient: None,
            note: "[MS.]".into(),
            date: Date {
                year,
                month: None,
                day: None,
            },
            period: Period::of(year),
            authorship,
            authorship_confidence: 0.8,
            authorship_basis: "test".into(),
            temporal_holdout: crate::document::in_temporal_holdout(year),
            body,
        }
    }

    fn corpus(n: u32) -> Vec<Document> {
        (0..n)
            .map(|i| {
                doc(
                    &format!("d{i}"),
                    1770,
                    Authorship::DraftInHand,
                    prose(i, 120),
                )
            })
            .collect()
    }

    fn side<'a>(a: &'a [Assignment], id: &str) -> &'a Assignment {
        a.iter().find(|x| x.doc_id == id).unwrap()
    }

    #[test]
    fn about_the_asked_share_of_families_is_examined() {
        let docs = corpus(200);
        let exam = assign(&docs, 7, 20)
            .iter()
            .filter(|a| a.split == Split::Exam)
            .count();
        assert!((25..=55).contains(&exam), "{exam} of 200 at 20%");
    }

    #[test]
    fn the_split_is_the_same_for_the_same_seed_whatever_the_order() {
        let docs = corpus(60);
        let mut shuffled = docs.clone();
        shuffled.reverse();
        let a = assign(&docs, 7, 20);
        let b = assign(&shuffled, 7, 20);
        for d in &docs {
            assert_eq!(side(&a, &d.id).split, side(&b, &d.id).split, "{}", d.id);
        }
    }

    #[test]
    fn another_seed_holds_out_other_families() {
        let docs = corpus(80);
        let held = |seed| {
            assign(&docs, seed, 20)
                .into_iter()
                .filter(|a| a.split == Split::Exam)
                .map(|a| a.doc_id)
                .collect::<Vec<_>>()
        };
        assert_ne!(held(1), held(2));
    }

    #[test]
    fn two_printings_of_one_text_are_one_family_and_never_divided() {
        let shared = prose(900, 200);
        let mut docs = corpus(40);
        docs.push(doc(
            "print-a",
            1771,
            Authorship::DraftInHand,
            shared.clone(),
        ));
        docs.push(doc(
            "print-b",
            1771,
            Authorship::SignedScribal,
            format!("{shared} one more line"),
        ));
        for seed in 0..30 {
            let a = assign(&docs, seed, 50);
            assert_eq!(
                side(&a, "print-a").split,
                side(&a, "print-b").split,
                "seed {seed}"
            );
            assert_eq!(side(&a, "print-a").family, side(&a, "print-b").family);
        }
    }

    /// A document of its own prose with `shared` words of another text inside it.
    fn quoting(id: &str, seed: u32, shared: &str) -> Document {
        let own = prose(seed, 150);
        doc(
            id,
            1771,
            Authorship::DraftInHand,
            format!("{own} {shared} {}", prose(seed + 5000, 150)),
        )
    }

    #[test]
    fn a_passage_reused_inside_two_longer_texts_makes_them_one_family() {
        let passage = prose(910, 40);
        let mut docs = corpus(30);
        docs.push(quoting("resolve-a", 1, &passage));
        docs.push(quoting("resolve-b", 2, &passage));
        for seed in 0..30 {
            let a = assign(&docs, seed, 50);
            assert_eq!(
                side(&a, "resolve-a").family,
                side(&a, "resolve-b").family,
                "seed {seed}"
            );
            assert_eq!(
                side(&a, "resolve-a").split,
                side(&a, "resolve-b").split,
                "seed {seed}"
            );
        }
    }

    #[test]
    fn a_few_words_two_texts_happen_to_share_do_not_make_them_one_family() {
        let phrase = prose(911, 10);
        let mut docs = corpus(10);
        docs.push(quoting("a", 1, &phrase));
        docs.push(quoting("b", 2, &phrase));
        let a = assign(&docs, 3, 50);
        assert_ne!(side(&a, "a").family, side(&a, "b").family);
    }

    #[test]
    fn boilerplate_held_by_many_texts_joins_none_of_them() {
        let formula = prose(912, 40);
        let docs: Vec<Document> = (0..12)
            .map(|i| quoting(&format!("d{i}"), 20 + i, &formula))
            .collect();
        let a = assign(&docs, 3, 50);
        let families: std::collections::BTreeSet<&str> =
            a.iter().map(|x| x.family.as_str()).collect();
        assert_eq!(families.len(), 12);
    }

    #[test]
    fn a_leak_is_found_by_shared_passages_even_when_the_families_differ() {
        let passage = prose(913, 40);
        let docs = vec![
            quoting("held", 1, &passage),
            quoting("trained", 2, &passage),
        ];
        let assignments = vec![
            Assignment {
                doc_id: "held".into(),
                family: "f1".into(),
                split: Split::Exam,
            },
            Assignment {
                doc_id: "trained".into(),
                family: "f2".into(),
                split: Split::Train,
            },
        ];
        assert_eq!(
            leaks(&docs, &assignments),
            vec![Leak {
                train_doc: "trained".into(),
                held_out_doc: "held".into()
            }]
        );
    }

    #[test]
    fn the_late_years_and_the_family_of_a_late_document_are_kept_for_the_temporal_test() {
        let shared = prose(901, 200);
        let mut docs = corpus(20);
        docs.push(doc("late", 1795, Authorship::DraftInHand, shared.clone()));
        docs.push(doc("early-copy", 1771, Authorship::DraftInHand, shared));
        let a = assign(&docs, 3, 20);
        assert_eq!(side(&a, "late").split, Split::Temporal);
        assert_eq!(
            side(&a, "early-copy").split,
            Split::Temporal,
            "a copy of a late document is late"
        );
        assert!(a.iter().filter(|x| x.split == Split::Temporal).count() == 2);
    }

    #[test]
    fn a_family_whose_kind_is_unsettled_is_never_examined() {
        let docs: Vec<Document> = (0..100)
            .map(|i| {
                doc(
                    &format!("d{i}"),
                    1770,
                    Authorship::ContextOnly,
                    prose(i, 120),
                )
            })
            .collect();
        assert!(assign(&docs, 5, 50).iter().all(|a| a.split == Split::Train));
    }

    #[test]
    fn a_family_of_newspaper_pieces_can_be_examined_so_every_attribution_is_tested() {
        let docs: Vec<Document> = (0..100)
            .map(|i| {
                doc(
                    &format!("d{i}"),
                    1770,
                    Authorship::PseudonymousAttributed,
                    prose(i, 120),
                )
            })
            .collect();
        assert!(assign(&docs, 5, 50).iter().any(|a| a.split == Split::Exam));
    }

    #[test]
    fn the_manifest_names_what_was_held_out_by_a_hash_of_its_text() {
        let docs = corpus(60);
        let a = assign(&docs, 7, 20);
        let m = manifest(&docs, &a, 7, 20);
        assert_eq!(m.schema, MANIFEST_SCHEMA);
        assert_eq!(m.exam.len() + m.temporal.len() + m.train, 60);
        let first = &m.exam[0];
        let body = &docs.iter().find(|d| d.id == first.doc_id).unwrap().body;
        assert_eq!(
            first.body_blake3,
            blake3::hash(body.as_bytes()).to_hex().to_string()
        );
    }

    #[test]
    fn the_digest_changes_when_what_was_held_out_changes() {
        let docs = corpus(60);
        let a = assign(&docs, 7, 20);
        let m = manifest(&docs, &a, 7, 20);
        assert_eq!(digest(&m), digest(&m.clone()));
        let mut other = m.clone();
        other.exam[0].body_blake3.push('0');
        assert_ne!(digest(&m), digest(&other));
        assert!(!digest(&m).is_empty());
    }

    #[test]
    fn a_training_text_that_shares_words_with_a_held_out_one_is_a_leak() {
        let shared = prose(902, 200);
        let docs = vec![
            doc("kept-out", 1770, Authorship::DraftInHand, shared.clone()),
            doc(
                "sneaked-in",
                1770,
                Authorship::DraftInHand,
                format!("{shared} extra"),
            ),
            doc("unrelated", 1770, Authorship::DraftInHand, prose(5, 200)),
        ];
        let assignments = vec![
            Assignment {
                doc_id: "kept-out".into(),
                family: "f1".into(),
                split: Split::Exam,
            },
            Assignment {
                doc_id: "sneaked-in".into(),
                family: "f2".into(),
                split: Split::Train,
            },
            Assignment {
                doc_id: "unrelated".into(),
                family: "f3".into(),
                split: Split::Train,
            },
        ];
        assert_eq!(
            leaks(&docs, &assignments),
            vec![Leak {
                train_doc: "sneaked-in".into(),
                held_out_doc: "kept-out".into()
            }]
        );
    }

    #[test]
    fn a_split_made_by_assign_has_no_leaks() {
        let shared = prose(903, 200);
        let mut docs = corpus(50);
        docs.push(doc("x", 1771, Authorship::DraftInHand, shared.clone()));
        docs.push(doc("y", 1771, Authorship::DraftInHand, shared));
        assert!(leaks(&docs, &assign(&docs, 11, 40)).is_empty());
    }
}
