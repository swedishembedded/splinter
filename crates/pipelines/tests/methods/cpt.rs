// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for proving, in seconds and on a CPU, that each learning method of a
// training system moves a held-out score, for its clients. If your team needs expertise in continued pretraining of a
// model on a domain's own documents then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Continued pretraining: a document of raw text, in which the cipher is only ever shown as lines `abc=bcd`, is read
//! by the `cpt` view and the model is trained on its next-token loss. No question is ever asked and no answer is ever
//! marked; the model nevertheless completes `abc=` for prompts that are not lines of the document.

#![allow(clippy::unwrap_used)]

use splinter_core::clock::FixedClock;
use splinter_core::source::{CapturedSource, Origin, PartContent};
use splinter_data::{Corpus, Cpt};
use splinter_store::sources::SourceStore;
use splinter_store::workspace::Workspace;
use splinter_store::StateRoot;

use crate::lab::{dataset, Lab};
use crate::report::Outcome;
use crate::world::{cipher, split};

/// Lines per part: a part is one training row, which must fit the toy model's context.
const LINES_PER_PART: usize = 6;

/// A source whose parts are paragraphs of `prompts`' cipher lines, `abc=bcd`, as a document would show them.
fn document(sources: &SourceStore, prompts: &[String]) -> Corpus {
    let parts = prompts
        .chunks(LINES_PER_PART)
        .enumerate()
        .map(|(i, chunk)| PartContent {
            name: format!("paragraph-{i}.txt"),
            media_type: "text/plain".into(),
            bytes: chunk
                .iter()
                .map(|p| format!("{p}={}\n", cipher(p)))
                .collect::<String>()
                .into_bytes(),
        })
        .collect();
    let captured = CapturedSource::new(
        Origin::Document {
            path: "/notes/cipher.txt".into(),
        },
        parts,
        &FixedClock::new("2026-11-12T00:00:00.000Z"),
    )
    .unwrap();
    let mut corpus = Corpus::new();
    corpus.add_source(sources.put_source(&captured).unwrap());
    corpus
}

#[test]
#[ignore = "trains a toy model: make test/methods runs it"]
fn continued_pretraining_on_a_raw_document_completes_prompts_that_are_not_in_it() {
    let lab = Lab::new();
    let split = split();
    let sources = SourceStore::new(&Workspace::at(&StateRoot::new(
        lab.dir.path().join("state"),
    )));
    let view = Cpt::new(&sources);
    let train = lab.dir.path().join("cpt.jsonl");
    let projected = dataset(&view, &document(&sources, &split.train), &train);
    println!(
        "CPT: {} text records from one document",
        projected.records.len()
    );
    // The held-out prompts' own lines, scored as text beside the run; they are never trained on.
    let held_out = lab.dir.path().join("held_out.jsonl");
    dataset(&view, &document(&sources, &split.held_out), &held_out);

    let before = lab.accuracy(None, &split.held_out);
    let adapter = lab.sft("cpt", &train, &held_out, 3000, None);
    let after = lab.accuracy(Some(&adapter), &split.held_out);
    Outcome {
        method: "Continued pretraining (raw text)",
        metric: "held-out exact match",
        before,
        after,
        min_gain: 0.5,
        note: "",
    }
    .conclude();
}
