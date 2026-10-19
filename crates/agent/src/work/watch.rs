// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements coding agents that work inside a confined
// directory and leave a complete, replayable record of what they did, for
// its clients. If your team needs expertise in agent runtimes or in
// measuring what an agent really did, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The watcher of one send's events.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

use futures::FutureExt;
use sven_sdk::{CancelToken, EventStream, SessionEvent};
use tokio::sync::oneshot;

use super::{Observed, Observer, CANCEL_POLL};

/// The task that watches one send's events: it counts tool calls, stops the
/// run at the cap or when the caller cancels, and hands every event to the
/// observer. (The repetition stop is not here: see [`super::guard`].)
pub(super) struct Watch {
    pub(super) events: EventStream,
    pub(super) stop: CancelToken,
    pub(super) external: Option<CancelToken>,
    pub(super) cap: Option<u32>,
    pub(super) observer: Option<Observer>,
    pub(super) tool_calls: Arc<AtomicU32>,
    pub(super) capped: Arc<AtomicBool>,
}

/// A call as `name\u{1}arguments`, shortened for a message.
pub(super) fn describe(asked: &str) -> String {
    let (name, args) = asked.split_once('\u{1}').unwrap_or((asked, ""));
    let args: String = args.chars().take(160).collect();
    format!("{name} {args}")
}

impl Watch {
    /// Runs until the send is `finished`, then drains what is buffered;
    /// returns how many events were lost to a slow observer.
    pub(super) async fn run(mut self, mut finished: oneshot::Receiver<()>) -> u64 {
        let mut dropped = 0;
        let mut tick = tokio::time::interval(CANCEL_POLL);
        loop {
            tokio::select! {
                received = self.events.recv() => match received {
                    Some(event) => dropped += self.handle(&event.event),
                    None => return dropped,
                },
                _ = tick.tick() => {
                    if self.external.as_ref().is_some_and(CancelToken::is_cancelled) {
                        self.stop.cancel();
                    }
                }
                _ = &mut finished => break,
            }
        }
        // What the finished send left buffered: `now_or_never` reads an
        // event that is ready and stops at the first that is not.
        while let Some(Some(event)) = self.events.recv().now_or_never() {
            dropped += self.handle(&event.event);
        }
        dropped
    }

    /// Takes one event of the stream; returns how many events it says were
    /// lost (a gap marker stands in for them, it is not an event of the run).
    fn handle(&self, event: &SessionEvent) -> u64 {
        if let SessionEvent::EventsDropped { count } = event {
            return self.lost(*count);
        }
        self.see(event);
        0
    }

    fn see(&self, event: &SessionEvent) {
        if matches!(event, SessionEvent::ToolCallStarted(_)) {
            let calls = self.tool_calls.fetch_add(1, Ordering::SeqCst) + 1;
            if self.cap.is_some_and(|cap| calls > cap) {
                self.capped.store(true, Ordering::SeqCst);
                self.stop.cancel();
            }
        }
        if let Some(observer) = &self.observer {
            observer(Observed::Event(event));
        }
    }

    fn lost(&self, count: u64) -> u64 {
        if let Some(observer) = &self.observer {
            observer(Observed::Dropped(count));
        }
        count
    }
}
