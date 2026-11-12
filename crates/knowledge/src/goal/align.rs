// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that find the statements of a
// large body of documents that say the same thing in different words and
// settle which of them stands, for its clients. If your team needs expertise
// in reconciling paraphrased statements across documents you can procure our
// services by sending an email to info@swedishembedded.com.

//! Which fact keys are one fact stated in other words.
//!
//! Admission keeps a key in the words of its quote, so `The maintenance
//! period of alder pump 3 is 250 hours` and `The service interval of alder
//! pump 3 is 400 hours` are two facts to it, and the older never meets the
//! newer. An embedder proposes which attribute phrases mean the same; code decides whether to believe it. A pair of phrases is
//! aligned only if
//!
//! * their cosine reaches [`MIN_COSINE`];
//! * it is the best open pairing for both, by [`MARGIN`] (a phrase pairs with
//!   one other, and not when another is nearly as close);
//! * no document uses both: a document that says two things in two phrasings
//!   means two things;
//! * entity names are never aligned, only attributes: `hesitant` and `reluctant`
//!   look alike to an embedder and so do `hesitant` and `anxious`, and a false
//!   merge turns two facts into a ruled conflict;
//! * only what is stated about identified things (an entity with a number in
//!   its name) is evidence for an attribute pair. A thing with an identifier
//!   has one set of attributes; a class, such as a type of client, has an
//!   open-ended set, and two of its situations stated in different documents
//!   are two situations. Where there is no such evidence nothing is aligned.
//!
//! The alignment decides only which statements are compared. Which one
//! stands is still the documents' headers and the stated values
//! ([`crate::objects::resolve_aligned`]), never the embedder's say.

use std::collections::{BTreeMap, BTreeSet};

use crate::objects::Pack;
use crate::retrieve::{EmbedError, Embedder};

/// The least cosine at which two phrases may be one.
pub const MIN_COSINE: f64 = 0.6;

/// How much closer than any other free phrase a pair must be, for both.
pub const MARGIN: f64 = 0.05;

/// The key as the pack states it, split into attribute and entity.
fn split(key: &str) -> Option<(String, String)> {
    let (a, e) = key.split_once('|')?;
    Some((a.trim().to_string(), e.trim().to_string()))
}

/// Phrases to align and the documents that use each.
type Uses = BTreeMap<String, BTreeSet<String>>;

/// Pairs of `phrases` that mean one thing, as the phrase each maps to.
fn pair_up(
    phrases: &Uses,
    embedder: &dyn Embedder,
) -> Result<BTreeMap<String, String>, EmbedError> {
    let names: Vec<&String> = phrases.keys().collect();
    if names.len() < 2 {
        return Ok(BTreeMap::new());
    }
    let texts: Vec<&str> = names.iter().map(|n| n.as_str()).collect();
    let mut vectors = Vec::new();
    for batch in texts.chunks(32) {
        vectors.extend(embedder.embed(batch)?);
    }
    let unit: Vec<Vec<f32>> = vectors
        .into_iter()
        .map(|v| {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                v.into_iter().map(|x| x / norm).collect()
            } else {
                v
            }
        })
        .collect();
    // Every pair that could be one, strongest first.
    let n = names.len();
    let mut cosine = vec![vec![f64::NEG_INFINITY; n]; n];
    let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let c = f64::from(
                unit[i]
                    .iter()
                    .zip(&unit[j])
                    .map(|(a, b)| a * b)
                    .sum::<f32>(),
            );
            let together = phrases[names[i]]
                .intersection(&phrases[names[j]])
                .next()
                .is_some();
            if c >= MIN_COSINE && !together {
                cosine[i][j] = c;
                cosine[j][i] = c;
                pairs.push((c, i, j));
            }
        }
    }
    pairs.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let mut taken: BTreeSet<usize> = BTreeSet::new();
    let mut aligned = BTreeMap::new();
    for (c, i, j) in pairs {
        if taken.contains(&i) || taken.contains(&j) {
            continue;
        }
        // It must stand out: no other phrase still free is nearly as close to either.
        let rival = |from: usize, not: usize| {
            (0..n)
                .filter(|k| *k != not && *k != from && !taken.contains(k))
                .map(|k| cosine[from][k])
                .fold(f64::NEG_INFINITY, f64::max)
        };
        if c - rival(i, j).max(rival(j, i)) < MARGIN {
            continue;
        }
        taken.extend([i, j]);
        // The pair is named by its smaller phrase, so the choice is repeatable.
        let shared = names[i].min(names[j]).clone();
        aligned.insert(names[i].clone(), shared.clone());
        aligned.insert(names[j].clone(), shared);
    }
    Ok(aligned)
}

/// A map from each fact key of `pack` that is another phrasing of a key to the
/// key they share; keys that stand alone are absent.
///
/// # Errors
///
/// When the embedder fails.
pub fn align_keys(
    pack: &Pack,
    embedder: &dyn Embedder,
) -> Result<BTreeMap<String, String>, EmbedError> {
    let mut attributes: Uses = BTreeMap::new();
    let mut keys: BTreeSet<String> = BTreeSet::new();
    for object in pack.objects.iter().filter(|o| o.kind == "fact") {
        let Some(key) = object.data["key"].as_str() else {
            continue;
        };
        let Some((attribute, entity)) = split(key) else {
            continue;
        };
        keys.insert(key.to_string());
        // Only what is stated about an identified thing is evidence: it has one
        // set of attributes, so a phrase that replaces another across documents
        // is a paraphrase. A class (a type of client) has an open-ended set of
        // them, and two situations stated in different documents are just two.
        if entity.chars().any(|c| c.is_ascii_digit()) {
            attributes
                .entry(attribute)
                .or_default()
                .insert(object.document.clone());
        }
    }
    let attribute_of = pair_up(&attributes, embedder)?;
    let mut aligned = BTreeMap::new();
    for key in keys {
        let Some((attribute, entity)) = split(&key) else {
            continue;
        };
        let shared = format!(
            "{} | {}",
            attribute_of.get(&attribute).unwrap_or(&attribute),
            entity
        );
        if shared != key {
            aligned.insert(key, shared);
        }
    }
    Ok(aligned)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objects::{admit, Document, ProposedObject};
    use serde_json::json;

    /// Phrases are points on a line; synonyms sit at the same place.
    struct Line(Vec<(&'static str, f32)>);

    impl Embedder for Line {
        fn name(&self) -> String {
            "line".into()
        }

        fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
            Ok(texts
                .iter()
                .map(|t| {
                    let at = self.0.iter().find(|(p, _)| p == t).map_or(1.5, |(_, a)| *a);
                    vec![at.cos(), at.sin()]
                })
                .collect())
        }
    }

    fn fact(id: &str, doc: &str, attribute: &str, entity: &str, value: &str) -> ProposedObject {
        ProposedObject {
            id: id.into(),
            kind: "fact".into(),
            document: doc.into(),
            quote: format!("The {attribute} of {entity} is {value}."),
            data: json!({"key": format!("{attribute} | {entity}"), "value": value}),
        }
    }

    fn documents() -> Vec<Document> {
        let doc = |id: &str, date: &str, body: &str| {
            Document::parse(
                id,
                &format!("---\nauthority: 3\ndate: {date}\n---\n# Pump manual\n\n{body}"),
            )
        };
        vec![
            doc("old", "2024-01-01", "The service interval of pump 3 is 250 hours.\nThe rated torque of pump 3 is 30 newton metres.\n"),
            doc("new", "2026-01-01", "The maintenance period of pump 3 is 400 hours.\nThe nominal force of pump 3 is 30 newton metres.\n"),
        ]
    }

    #[test]
    fn paraphrases_that_no_document_uses_together_are_one_key() {
        let docs = documents();
        let pack = admit(
            "t",
            &docs,
            &[
                fact("a", "old", "service interval", "pump 3", "250 hours"),
                fact("b", "old", "rated torque", "pump 3", "30 newton metres"),
                fact("c", "new", "maintenance period", "pump 3", "400 hours"),
                fact("d", "new", "nominal force", "pump 3", "30 newton metres"),
            ],
        );
        let embedder = Line(vec![
            ("service interval", 0.0),
            ("maintenance period", 0.1),
            ("rated torque", 0.9),
            ("nominal force", 0.95),
        ]);
        let aligned = align_keys(&pack, &embedder).unwrap();
        // Each pair is named by its smaller phrase.
        assert_eq!(
            aligned["service interval | pump 3"],
            "maintenance period | pump 3"
        );
        assert_eq!(aligned["rated torque | pump 3"], "nominal force | pump 3");
        // The ruling is the documents' and the values': the newer document of equal
        // authority supersedes the older on the interval, and nothing is ruled on the torque.
        let (resolutions, flags) = crate::objects::resolve_aligned(&pack, &aligned);
        assert!(flags.is_empty(), "{flags:?}");
        assert_eq!(resolutions.len(), 1, "{resolutions:?}");
        assert_eq!(
            (
                resolutions[0].winner.as_str(),
                resolutions[0].superseded.as_slice()
            ),
            ("c", ["a".to_string()].as_slice())
        );
    }

    #[test]
    fn phrases_one_document_uses_together_are_never_merged() {
        let docs = documents();
        let pack = admit(
            "t",
            &docs,
            &[
                fact("a", "old", "service interval", "pump 3", "250 hours"),
                fact("b", "old", "rated torque", "pump 3", "30 newton metres"),
            ],
        );
        // Identical places on the line, but one document states both.
        let embedder = Line(vec![("service interval", 0.0), ("rated torque", 0.0)]);
        assert!(align_keys(&pack, &embedder).unwrap().is_empty());
    }
}
