// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements retrieval over a person's writing that
// finds the passages a task or a principle bears on, for its clients. If
// your team needs expertise in hybrid lexical and semantic search over a
// large corpus, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: a model reads a question and a passage together and says whether
//! the passage bears on the question. Its reply's first word is the verdict:
//! `YES` is relevant; `NO`, a reply that is not a verdict and no reply at
//! all are not, so a reader that cannot say leaves a passage where search
//! put it. The model is shown the question and the passage and asked
//! nothing else.

// Helpers outside a test function unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::sync::Arc;

use common::{scratch_context, Scripted};
use splinter_agent::solve::Model;
use splinter_knowledge::retrieve::{Passage, Reranker};
use splinter_pipelines::rerank::ModelReranker;

const EDUCATION: &str = "Education of the people is the surest foundation of liberty, and a nation that wishes to be free must see that its youth are taught to reason.";
const TOBACCO: &str = "The tobacco shipped to Havre by the brig Eliza was sold at a poor price, and the merchants complain of the duties laid upon the hogsheads.";
const QUESTION: &str = "How should the young learn?";

#[test]
fn the_first_word_of_the_readers_reply_is_the_verdict() {
    let (_scratch, ctx) =
        scratch_context("rerank-verdicts", Scripted::new(|_| String::new()), false);
    let script = Scripted::new(|prompt| {
        if prompt.contains("surest foundation") {
            "YES\nit speaks of teaching".into()
        } else if prompt.contains("hogsheads") {
            "No, it is about trade.".into()
        } else {
            "I could not say".into()
        }
    });
    let name = "local:rerank/reader".parse().unwrap();
    ctx.add_model(
        "local:rerank/reader".parse().unwrap(),
        Model::new(Arc::new(script.clone()), "scripted/reader"),
    );
    let reranker = ModelReranker::new(&ctx, &name).unwrap();
    let passage = |text: &str| Passage::of_text("letters.txt", 0, text);
    assert!(reranker.relevant(QUESTION, &passage(EDUCATION)).unwrap());
    assert!(!reranker.relevant(QUESTION, &passage(TOBACCO)).unwrap());
    assert!(
        !reranker
            .relevant(
                QUESTION,
                &passage("Something else entirely, said at some length.")
            )
            .unwrap(),
        "a reply that is not a verdict does not promote a passage"
    );
    let prompts = script.prompts.lock().unwrap();
    assert!(prompts[0].contains(QUESTION) && prompts[0].contains(EDUCATION));
}
