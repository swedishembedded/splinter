// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements scheduling of several language models on one
// accelerator so that model swaps are amortised over many conversations, for
// its clients. If your team needs expertise in running multi-model agent
// simulations on limited GPU memory, you can procure our services by sending
// an email to info@swedishembedded.com.

//! A gate that lets many sessions share a card that holds one base at a time.
//!
//! Loading a base takes minutes; a session alternates between the user's model
//! and the policy's at every turn. Run one after the other, a session would
//! swap twice a turn. Run together behind a gate, every session waiting for
//! the user's model is served before the card is handed to the policy, and
//! back, so a round of turns costs two swaps however many sessions there are.
//!
//! Within a side the sessions take the card one at a time: a model serves one
//! generation at a time, and a session left waiting inside the model's own
//! queue would go silent for as long as the others take, which a stream
//! treats as a dead connection.
//!
//! The gate decides only whose turn it is: when nobody of the side holding the
//! card is running or waiting, and every session still alive is waiting for the
//! other side, the card passes to it. A session on its way from one side to the
//! other is waited for, so that the card does not change sides for the first
//! session to ask and back again for the laggards.

use std::pin::Pin;
use std::sync::{Arc, Mutex, PoisonError};
use std::task::{Context, Poll};

use futures::Stream;
use splinter_sdk::agent::sven::model::{CompletionRequest, ModelProvider, ResponseStream};
use tokio::sync::Notify;

/// Which model has the card.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    /// The simulated user's model.
    User,
    /// The policy.
    Policy,
}

impl Side {
    fn index(self) -> usize {
        match self {
            Side::User => 0,
            Side::Policy => 1,
        }
    }

    fn other(self) -> Self {
        match self {
            Side::User => Side::Policy,
            Side::Policy => Side::User,
        }
    }
}

struct State {
    holder: Side,
    /// Sessions that have not yet left.
    alive: usize,
    running: [usize; 2],
    waiting: [usize; 2],
    swaps: usize,
}

impl State {
    /// Passes the card on when nobody of the holding side is left to be served.
    fn pass_on(&mut self) -> bool {
        let here = self.holder.index();
        let there = self.holder.other().index();
        if self.running[here] == 0
            && self.waiting[here] == 0
            && self.waiting[there] > 0
            && self.waiting[there] >= self.alive
        {
            self.holder = self.holder.other();
            self.swaps += 1;
            true
        } else {
            false
        }
    }
}

/// See the module documentation.
pub struct Gate {
    state: Mutex<State>,
    changed: Notify,
}

/// Held while a side uses the card; dropping it hands the card on if nobody of
/// the side is left.
pub struct Permit {
    gate: Arc<Gate>,
    side: Side,
}

impl Drop for Permit {
    fn drop(&mut self) {
        {
            let mut state = self
                .gate
                .state
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            state.running[self.side.index()] -= 1;
            state.pass_on();
        }
        self.gate.changed.notify_waiters();
    }
}

impl Gate {
    /// A gate for `sessions` sessions, whose card is first held by `first`.
    #[must_use]
    pub fn new(first: Side, sessions: usize) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(State {
                holder: first,
                alive: sessions,
                running: [0; 2],
                waiting: [0; 2],
                swaps: 0,
            }),
            changed: Notify::new(),
        })
    }

    /// Waits for the card for `side`.
    pub async fn enter(self: &Arc<Self>, side: Side) -> Permit {
        let mut queued = false;
        loop {
            let notified = self.changed.notified();
            tokio::pin!(notified);
            // Registered before the state is read, so a hand-over between the
            // read and the wait is not missed.
            notified.as_mut().enable();
            {
                let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
                if queued {
                    state.waiting[side.index()] -= 1;
                }
                if state.holder == side && state.running[side.index()] == 0 {
                    state.running[side.index()] += 1;
                    return Permit {
                        gate: Arc::clone(self),
                        side,
                    };
                }
                state.waiting[side.index()] += 1;
                queued = true;
                if state.pass_on() {
                    drop(state);
                    self.changed.notify_waiters();
                }
            }
            notified.await;
        }
    }

    /// A session that will ask for the card no more; the card passes on if
    /// the others are waiting for the other side.
    pub fn leave(&self) {
        {
            let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            state.alive = state.alive.saturating_sub(1);
            state.pass_on();
        }
        self.changed.notify_waiters();
    }

    /// How many times the card changed sides.
    #[must_use]
    pub fn swaps(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .swaps
    }
}

/// A stream that keeps its permit until it ends or is dropped.
struct Held {
    inner: ResponseStream,
    _permit: Permit,
}

impl Stream for Held {
    type Item = <ResponseStream as Stream>::Item;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

/// A model provider that takes the card for `side` for each completion, and
/// keeps it until the reply has been streamed out.
pub struct Gated {
    inner: Arc<dyn ModelProvider>,
    gate: Arc<Gate>,
    side: Side,
}

impl Gated {
    /// `inner`, served only when `side` holds the card of `gate`.
    #[must_use]
    pub fn new(inner: Arc<dyn ModelProvider>, gate: Arc<Gate>, side: Side) -> Self {
        Self { inner, gate, side }
    }
}

#[async_trait::async_trait]
impl ModelProvider for Gated {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    async fn complete(&self, req: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let permit = self.gate.enter(self.side).await;
        let inner = self.inner.complete(req).await?;
        Ok(Box::pin(Held {
            inner,
            _permit: permit,
        }))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use super::*;

    /// Eight sessions of three rounds, each turn taking some time: the card
    /// changes sides once per half round, not once per turn.
    #[tokio::test]
    async fn many_sessions_share_the_card_one_at_a_time_with_two_swaps_a_round() {
        let gate = Gate::new(Side::User, 8);
        let inside = Arc::new([AtomicUsize::new(0), AtomicUsize::new(0)]);
        let mut sessions = Vec::new();
        for n in 0..8u64 {
            let (gate, inside) = (Arc::clone(&gate), Arc::clone(&inside));
            sessions.push(tokio::spawn(async move {
                for _ in 0..3 {
                    for side in [Side::User, Side::Policy] {
                        let permit = gate.enter(side).await;
                        let before = inside[side.index()].fetch_add(1, Ordering::SeqCst);
                        assert_eq!(before, 0, "a side serves one session at a time");
                        assert_eq!(
                            inside[side.other().index()].load(Ordering::SeqCst),
                            0,
                            "the two models never share the card"
                        );
                        tokio::time::sleep(Duration::from_millis(2 + n)).await;
                        inside[side.index()].fetch_sub(1, Ordering::SeqCst);
                        drop(permit);
                    }
                }
                gate.leave();
            }));
        }
        for s in sessions {
            s.await.expect("a session");
        }
        // User, policy, user, policy, user, policy: five hand-overs.
        assert!(gate.swaps() <= 5 + 2, "{} swaps", gate.swaps());
    }

    #[tokio::test]
    async fn a_lone_session_alternates_and_a_finished_one_frees_the_rest() {
        let gate = Gate::new(Side::User, 1);
        let _a = gate.enter(Side::User).await;
        let waiting = {
            let gate = Arc::clone(&gate);
            tokio::spawn(async move {
                let _p = gate.enter(Side::Policy).await;
            })
        };
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(
            !waiting.is_finished(),
            "the policy waits while the user holds the card"
        );
        drop(_a);
        tokio::time::timeout(Duration::from_secs(2), waiting)
            .await
            .expect("handed over")
            .expect("ran");
    }
}
