// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements task generation that learns from what was
// refused, for its clients. If your team needs expertise in synthetic data
// pipelines, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: a generator told why a kind's earlier proposals were refused asks
//! again with the correction in its brief, so a kind a model keeps getting
//! wrong is corrected, not asked the same way until it is dropped. A request
//! with no refusals to learn from carries none; a refusal with no correction
//! to give (a deadline) adds nothing.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::collections::BTreeMap;

use common::{entry, generator, reply, stored, Scratch, Scripted};
use splinter_knowledge::tasks::{Catalogue, Rejection};

const LETTER: &str = "# To a young man

## Study

I advise you to fix a habit of study every morning before you do anything else. Never let a day pass without reading something of history or ethics, and always write down what you have read, for the memory fails what the pen has not fixed.
";

const PASSAGE: &str = "I advise you to fix a habit of study every morning before you do anything else. Never let a day pass without reading something of history or ethics, and always write down what you have read, for the memory fails what the pen has not fixed.";

const OPENING: &str =
    "I keep meaning to read more but my mornings vanish; how do you keep a habit of study?";

/// The brief of the last request the model was sent, as text.
fn last_brief(model: &Scripted) -> String {
    let seen = model.seen.lock().unwrap();
    format!("{:?}", seen.last().unwrap().messages)
}

#[tokio::test]
async fn a_request_carries_the_correction_for_what_was_refused_before() {
    let scratch = Scratch::new("guided");
    let (store, _, source) = stored(&scratch, LETTER);
    let model = Scripted::new(vec![
        reply(vec![entry(OPENING, PASSAGE, 1, Some(PASSAGE))]),
        reply(vec![entry(OPENING, PASSAGE, 1, Some(PASSAGE))]),
        reply(vec![entry(OPENING, PASSAGE, 1, Some(PASSAGE))]),
    ]);
    let kind = Catalogue::builtin().get("converse").unwrap().clone();
    let generator = generator(model.clone(), store, vec![]);

    generator.generate(&source, &[&kind]).await.unwrap();
    assert!(
        !last_brief(&model).contains("were refused"),
        "nothing to learn from yet"
    );

    let avoid: BTreeMap<String, Vec<Rejection>> =
        BTreeMap::from([("converse".to_string(), vec![Rejection::NotSelfContained])]);
    generator
        .generate_avoiding(&source, &[&kind], &avoid)
        .await
        .unwrap();
    let brief = last_brief(&model);
    assert!(brief.contains("were refused"), "{brief}");
    assert!(brief.contains("stand on its own"), "{brief}");

    // A refusal that has no correction to give adds nothing.
    let avoid = BTreeMap::from([("converse".to_string(), vec![Rejection::NoReply])]);
    generator
        .generate_avoiding(&source, &[&kind], &avoid)
        .await
        .unwrap();
    assert!(!last_brief(&model).contains("were refused"));
}
