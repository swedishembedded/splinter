// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gated self-improvement loops where a model
// update is adopted only on held-out evidence. If your team needs expertise
// in evaluation-gated fine-tuning, you can procure our services by sending
// an email to info@swedishembedded.com.

//! Spec: the variants of trained tasks (the same fact in other words) join
//! the held-out records in the exam the release gate decides on, and are
//! reported apart from them.
//!
//! * The promotion rule decides on the records and the variants together;
//!   generalisation (the held-out records) and recall under paraphrase (the
//!   variants) are each reported with their own comparison and sign test.
//! * The variants are counted: how many were measured, and how many were
//!   left out and why. A variant of a task that was held out, not trained
//!   on, is not a measure of what was learned, and a variant whose wording
//!   the candidate trained on is left out as leaked.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use crate::common;

use common::gate::variants::{variant_asking, variant_set, variant_set_of};
use common::gate::{
    anchor_file, candidate_on, dataset, decide, gate_context, question, Brain, ANCHOR,
};
use common::Scratch;
use splinter_eval::promotion::{Reason, MIN_GAIN_FAMILIES};
use splinter_orchestrator::Context;
use splinter_pipelines::release::anchor;

fn freeze_anchor(scratch: &Scratch, ctx: &Context) {
    anchor::freeze(ctx, &[anchor_file(&scratch.0, 4)]).unwrap();
}

#[test]
fn variants_of_the_trained_facts_join_the_exam_and_are_reported_apart() {
    let (scratch, ctx) = gate_context("release-variants", Brain::Honest);
    freeze_anchor(&scratch, &ctx);
    // Twelve facts: three held out (a quarter is the most), so the sign test
    // over the records alone can never reach significance (three wins is
    // p = 0.125). The family-resampled interval of three unanimous families
    // is [1, 1], with no width, so the promotion rule asks for the families
    // the exact sign test needs first: three are too few.
    let (alone_scratch, alone_ctx) = gate_context("release-variants-alone", Brain::Honest);
    freeze_anchor(&alone_scratch, &alone_ctx);
    let (records_only, _) = candidate_on(
        &alone_ctx,
        dataset(&alone_ctx, "alpha", 12),
        &[ANCHOR, "alpha"],
    );
    let alone = decide(&alone_ctx, &records_only);
    let measured = alone.gate.improvement.measured.as_ref().unwrap();
    assert_eq!(measured.suite.tasks, 3);
    assert_eq!(measured.variants, None, "no variant was written");
    assert!(measured.sign_test.p_value > measured.alpha, "{measured:#?}");
    let promotion = alone.promotion.as_ref().unwrap();
    let dev = promotion.dev.clone().unwrap();
    assert_eq!((dev.items, dev.interval), (3, Some((1.0, 1.0))));
    assert_eq!(
        promotion.reasons,
        [Reason::TooFewFamilies {
            families: Some(3),
            required: MIN_GAIN_FAMILIES
        }]
    );
    assert!(!alone.gate.improvement.passed, "{:#?}", alone.gate);
    assert_eq!(alone.release, None);

    let (candidate, _) = candidate_on(&ctx, dataset(&ctx, "alpha", 12), &[ANCHOR, "alpha"]);
    // Nine variants of three trained facts; a variant of a held-out fact
    // (not learned from); and a variant whose wording is a question the
    // candidate trained on.
    variant_set(&ctx, "alpha", &[0, 1, 2], 3);
    variant_set(&ctx, "alpha", &[11], 3);
    variant_set_of(
        &ctx,
        "alpha",
        vec![(3, variant_asking(&ctx, "alpha", 3, &question("alpha", 4)))],
    );

    let decided = decide(&ctx, &candidate);
    let gate = &decided.gate;
    assert!(gate.passed, "{gate:#?}");
    let improvement = gate.improvement.measured.as_ref().unwrap();
    assert_eq!(
        improvement.suite.tasks,
        3 + 9,
        "held-out records and variants"
    );
    assert_eq!(improvement.comparison.candidate_wins, 12);
    assert!(
        improvement.sign_test.p_value < improvement.alpha,
        "with the variants the reported sign test agrees"
    );
    // The two kinds of task are reported apart: generalisation to the
    // held-out records, recall of trained facts under paraphrase.
    let generalisation = improvement.generalisation.as_ref().unwrap();
    assert_eq!(generalisation.suite.tasks, 3);
    assert_eq!(generalisation.comparison.candidate_wins, 3);
    assert!(
        generalisation.sign_test.p_value > improvement.alpha,
        "three records alone are not significant: {generalisation:?}"
    );
    // What the tasks say beside the family-level decision: twelve wins, no
    // losses, over the held-out records and the variants together.
    let by_task = improvement.task_level.unwrap();
    assert!(by_task.p_value < 0.001, "{by_task:?}");
    assert!(by_task.low <= by_task.difference && by_task.difference <= by_task.high);
    assert!(improvement
        .generalisation
        .as_ref()
        .unwrap()
        .task_level
        .is_some());
    let recall = improvement.recall.as_ref().unwrap();
    assert_eq!(recall.suite.tasks, 9);
    assert_eq!(recall.comparison.candidate_wins, 9);
    let variants = improvement.variants.as_ref().unwrap();
    assert_eq!(variants.tasks, 9, "{variants:#?}");
    assert_eq!(
        variants.excluded.iter().collect::<Vec<_>>(),
        [(&"leaked".to_string(), &1)],
        "the question trained on measures nothing"
    );
    assert!(decided.release.is_some());
}

#[test]
fn variants_of_what_the_candidate_did_not_learn_do_not_count_as_evidence() {
    let (scratch, ctx) = gate_context("release-variants-unlearned", Brain::Honest);
    freeze_anchor(&scratch, &ctx);
    // It learned nothing of "alpha": its variants stay wrong, like its
    // held-out records, so the extra tasks add ties, not wins.
    let (candidate, _) = candidate_on(&ctx, dataset(&ctx, "alpha", 12), &[ANCHOR]);
    variant_set(&ctx, "alpha", &[0, 1, 2], 3);

    let decided = decide(&ctx, &candidate);
    assert_eq!(decided.release, None, "{:#?}", decided.gate);
    let gate = decided.gate;
    let improvement = gate.improvement.measured.as_ref().unwrap();
    assert_eq!(improvement.variants.as_ref().unwrap().tasks, 9);
    assert_eq!(improvement.sign_test.discordant, 0);
    assert_eq!(improvement.recall.as_ref().unwrap().comparison.paired, 9);
    assert_eq!(
        improvement
            .generalisation
            .as_ref()
            .unwrap()
            .comparison
            .paired,
        3
    );
    assert!(!gate.improvement.passed);
}
