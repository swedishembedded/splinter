// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning agents that finish within the time
// they are given, for its clients. If your team needs expertise in turning an
// open-ended learning job into a bounded one, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec: a budget is shared between the stages so that generating tasks, which
//! has no natural end, cannot spend the time training needs: each open-ended
//! stage stops at a share of the budget counted from the start, the shares
//! only grow, and the last of them leaves time for what follows. A run
//! without a budget has no deadlines.

use std::time::{Duration, Instant};

use splinter_pipelines::budget::StageDeadlines;

#[test]
fn each_open_ended_stage_stops_at_a_growing_share_that_leaves_time_for_training() {
    let start = Instant::now();
    let end = start + Duration::from_secs(10 * 3600);
    let d = StageDeadlines::of(start, Some(end));
    let (tasks, attempts, teach) = (d.tasks.unwrap(), d.attempts.unwrap(), d.teach.unwrap());
    assert!(start < tasks && tasks < attempts && attempts < teach && teach < end);
    assert!(
        end - teach >= Duration::from_secs(3600),
        "training, the exam and the gate keep a real share"
    );
}

#[test]
fn a_run_without_a_budget_has_no_stage_deadlines() {
    let d = StageDeadlines::of(Instant::now(), None);
    assert_eq!((d.tasks, d.attempts, d.teach), (None, None, None));
}

#[test]
fn a_budget_already_spent_gives_deadlines_no_later_than_now() {
    let start = Instant::now();
    let d = StageDeadlines::of(start, Some(start));
    assert_eq!(d.tasks, Some(start));
}

/// When the writer's own text carries a share of the training examples,
/// training has that much more to do and nothing model-made to wait for:
/// the stages that make dialogue data give up their share in proportion,
/// and the tail - training, the exam, the gate - grows by it. Half the
/// examples from the writer's text doubles the tail; none leaves the
/// deadlines as they were.
#[test]
fn the_tail_grows_with_the_share_of_examples_the_writers_text_carries() {
    let start = Instant::now();
    let end = start + Duration::from_secs(10 * 3600);
    let plain = StageDeadlines::sharing(start, Some(end), 0.0);
    assert_eq!(plain, StageDeadlines::of(start, Some(end)));
    let half = StageDeadlines::sharing(start, Some(end), 0.5);
    let tail = |d: &StageDeadlines| end - d.teach.unwrap();
    assert_eq!(tail(&half), tail(&plain) * 2);
    assert!(half.tasks.unwrap() < plain.tasks.unwrap());
    assert!(half.attempts.unwrap() < plain.attempts.unwrap());
    assert!(start < half.tasks.unwrap());
    // The open-ended stages never lose everything: the tail is capped.
    let most = StageDeadlines::sharing(start, Some(end), 0.95);
    assert!(most.tasks.unwrap() > start + Duration::from_secs(60));
    assert!(tail(&most) <= Duration::from_secs(8 * 3600));
}
