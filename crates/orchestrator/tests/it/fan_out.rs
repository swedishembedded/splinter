// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that acquire a capability
// from a document or a tool and prove it with evidence. If your team needs
// expertise in agent infrastructure or small-model training loops, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: work items run concurrently up to a limit, and what comes back is
//! the same whatever the limit.
//!
//! A local model owns one device and is asked one thing at a time; a model
//! reached over an API can be asked several. Either way the caller sees the
//! results in the order it gave the items, so what is derived from them -
//! an experience set, say - does not depend on how fast each answer came.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use splinter_orchestrator::concurrency::fan_out;

/// Runs `fan_out` on a runtime of its own.
fn run<T>(future: impl std::future::Future<Output = T>) -> T {
    tokio::runtime::Builder::new_multi_thread()
        .enable_time()
        .build()
        .unwrap_or_else(|e| panic!("a runtime: {e}"))
        .block_on(future)
}

#[test]
fn results_come_back_in_the_order_the_items_were_given() {
    // Later items finish first.
    let results = run(fan_out(0..6u64, 6, |n| async move {
        tokio::time::sleep(Duration::from_millis((6 - n) * 15)).await;
        n * 10
    }));
    assert_eq!(results, [0, 10, 20, 30, 40, 50]);
}

#[test]
fn no_more_than_the_limit_runs_at_once() {
    let running = AtomicUsize::new(0);
    let most = AtomicUsize::new(0);
    let results = run(fan_out(0..12usize, 3, |n| {
        let (running, most) = (&running, &most);
        async move {
            let now = running.fetch_add(1, Ordering::SeqCst) + 1;
            most.fetch_max(now, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(20)).await;
            running.fetch_sub(1, Ordering::SeqCst);
            n
        }
    }));
    assert_eq!(results.len(), 12);
    assert_eq!(
        most.load(Ordering::SeqCst),
        3,
        "the limit is used, not passed"
    );
}

#[test]
fn a_limit_of_one_is_sequential_and_returns_what_a_wider_one_does() {
    let work = |n: u32| async move {
        tokio::time::sleep(Duration::from_millis(u64::from(7 - n) * 3)).await;
        n * n
    };
    let narrow = run(fan_out(1..=6u32, 1, work));
    let wide = run(fan_out(1..=6u32, 8, work));
    assert_eq!(narrow, wide);
    // Zero is no limit at all, so it is one.
    assert_eq!(run(fan_out(1..=6u32, 0, work)), narrow);
}

#[test]
fn an_item_not_yet_started_sees_a_stop_raised_meanwhile() {
    // With a limit of two, item 1 raises the stop; items beyond the two that
    // were already running when it did see it before they start, and every
    // item still returns - those running are drained, not dropped.
    let stop = AtomicBool::new(false);
    let results = run(fan_out(0..8usize, 2, |n| {
        let stop = &stop;
        async move {
            let stopped_before_starting = stop.load(Ordering::SeqCst);
            if n == 1 {
                stop.store(true, Ordering::SeqCst);
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
            (n, stopped_before_starting)
        }
    }));
    assert_eq!(results.len(), 8, "every item returns");
    assert!(results[..2].iter().all(|(_, stopped)| !stopped));
    assert!(
        results[4..].iter().all(|(_, stopped)| *stopped),
        "{results:?}"
    );
}

#[test]
fn a_slow_first_item_does_not_hold_the_slots_of_the_items_behind_it() {
    // The first item waits until the other four are done, with a limit of
    // two. A finished item that is not yet at the head of the results must
    // not keep its slot, or the items behind it never start and one runaway
    // episode stalls the whole solve.
    let finished = AtomicUsize::new(0);
    let results = run(async {
        tokio::time::timeout(
            Duration::from_secs(5),
            fan_out(0..5usize, 2, |n| {
                let finished = &finished;
                async move {
                    if n == 0 {
                        while finished.load(Ordering::SeqCst) < 4 {
                            tokio::time::sleep(Duration::from_millis(5)).await;
                        }
                    } else {
                        tokio::time::sleep(Duration::from_millis(5)).await;
                        finished.fetch_add(1, Ordering::SeqCst);
                    }
                    n
                }
            }),
        )
        .await
    });
    assert_eq!(
        results.expect("the other items finish while the first is still running"),
        [0, 1, 2, 3, 4]
    );
}
