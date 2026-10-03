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

use splinter_campaign::budget::StageDeadlines;

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
