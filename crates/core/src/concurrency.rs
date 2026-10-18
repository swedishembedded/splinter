// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that keep a model server's
// decode slots full instead of asking it one prompt at a time, for its
// clients. If your team needs expertise in batched inference pipelines, you
// can procure our services by sending an email to info@swedishembedded.com.

//! Bounded concurrency for work a stage fans out.
//!
//! A model that decodes several sequences together (a batching server, an
//! API) is asked several things at once; one that answers a request at a
//! time is asked one. [`fan_out`] runs work items concurrently up to a limit
//! and returns their results in the order the items were given, so what a
//! stage derives from them does not depend on which answer came first. It
//! performs no I/O of its own: the work does.

use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};

use futures::stream::{self, StreamExt};

/// Runs `work` over `items`, at most `limit` at a time (zero is one), and
/// returns the results in the order of `items`.
///
/// Items start in order and one starts as soon as a slot is free, whichever
/// item freed it: a finished item never keeps its slot while an earlier,
/// slower one is still running, or one runaway episode would hold back every
/// item behind it. A work item that checks a stop - a cancel, a spent budget -
/// when it begins sees what the items that had finished by then raised. Items
/// already running when a stop is raised are drained, never dropped.
pub async fn fan_out<I, T, F, Fut>(items: I, limit: usize, work: F) -> Vec<T>
where
    I: IntoIterator,
    F: FnMut(I::Item) -> Fut,
    Fut: Future<Output = T>,
{
    fan_out_each(items, limit, work, |_| {}).await
}

/// [`fan_out`], calling `each` with every result as it arrives, in the order
/// the results finish: for a caller that makes what is done so far durable
/// while the rest runs.
pub async fn fan_out_each<I, T, F, Fut>(
    items: I,
    limit: usize,
    mut work: F,
    mut each: impl FnMut(&T),
) -> Vec<T>
where
    I: IntoIterator,
    F: FnMut(I::Item) -> Fut,
    Fut: Future<Output = T>,
{
    let mut finished = stream::iter(items.into_iter().enumerate())
        .map(|(at, item)| {
            let running = work(item);
            async move { (at, running.await) }
        })
        .buffer_unordered(limit.max(1));
    let mut done: Vec<(usize, T)> = Vec::new();
    while let Some((at, result)) = finished.next().await {
        each(&result);
        done.push((at, result));
    }
    done.sort_by_key(|(at, _)| *at);
    done.into_iter().map(|(_, result)| result).collect()
}

/// Runs the blocking `work` over `items` on at most `limit` threads (zero is
/// one) and returns the results in the order of `items`.
///
/// For work that blocks its thread until it is done - a verifier that waits
/// on a model, say - where [`fan_out`] would make no progress on the other
/// items. Items start in order, each on the first thread free. A panic in
/// `work` propagates once the threads are joined.
pub fn fan_out_blocking<I, T, F>(items: &[I], limit: usize, work: F) -> Vec<T>
where
    I: Sync,
    T: Send,
    F: Fn(&I) -> T + Sync,
{
    let next = AtomicUsize::new(0);
    let done: Mutex<Vec<Option<T>>> = Mutex::new((0..items.len()).map(|_| None).collect());
    std::thread::scope(|scope| {
        for _ in 0..limit.max(1).min(items.len()) {
            scope.spawn(|| loop {
                let at = next.fetch_add(1, Ordering::SeqCst);
                let Some(item) = items.get(at) else { break };
                let result = work(item);
                done.lock().unwrap_or_else(PoisonError::into_inner)[at] = Some(result);
            });
        }
    });
    done.into_inner()
        .unwrap_or_else(PoisonError::into_inner)
        .into_iter()
        .flatten()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn blocking_work_runs_up_to_the_limit_at_once_and_comes_back_in_item_order() {
        let (running, most) = (AtomicUsize::new(0), AtomicUsize::new(0));
        let items: Vec<u64> = (0..9).collect();
        let results = fan_out_blocking(&items, 3, |n| {
            let now = running.fetch_add(1, Ordering::SeqCst) + 1;
            most.fetch_max(now, Ordering::SeqCst);
            // Later items finish first.
            std::thread::sleep(Duration::from_millis((9 - n) * 5));
            running.fetch_sub(1, Ordering::SeqCst);
            n * 10
        });
        assert_eq!(results, [0, 10, 20, 30, 40, 50, 60, 70, 80]);
        assert_eq!(most.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn a_limit_of_zero_is_one_and_no_items_is_no_results() {
        let items = [1, 2, 3];
        assert_eq!(fan_out_blocking(&items, 0, |n| n + 1), [2, 3, 4]);
        assert!(fan_out_blocking::<u8, u8, _>(&[], 4, |n| *n).is_empty());
    }
}
