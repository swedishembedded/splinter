// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements crash-safe storage proven by killing it at
// every durability boundary, for its clients. If your team needs expertise in
// fault injection for storage systems then you can procure our services by
// sending an email to info@swedishembedded.com.

//! The durability boundaries of the store's writes, for crash tests.
//!
//! A write that must survive a crash is a sequence of steps that each leave
//! the disk in a state recovery must handle (a blob synced, a temporary file
//! renamed into place, an intent recorded, a manifest committed). Each step's
//! end calls `boundary`. In an ordinary build it does nothing and compiles
//! away. Built with the `fault-injection` feature (the store's own tests are),
//! a test can record the boundaries an operation passes and then, in a child
//! process, end that process at any one of them, as a crash would: no
//! destructor runs and no cleanup follows. The test then reopens the store and
//! checks what survived.

/// The end of a durability step named `name`.
#[inline]
pub(crate) fn boundary(name: &'static str) {
    #[cfg(feature = "fault-injection")]
    injection::reach(name);
    #[cfg(not(feature = "fault-injection"))]
    let _ = name;
}

/// Recording and arming boundaries, per thread (tests run side by side).
#[cfg(feature = "fault-injection")]
pub mod injection {
    use std::cell::RefCell;

    /// The exit status of a process ended at a boundary.
    pub const CRASHED: i32 = 86;

    #[derive(Default)]
    struct State {
        seen: Vec<&'static str>,
        crash_at: Option<usize>,
    }

    thread_local! {
        static STATE: RefCell<State> = RefCell::default();
    }

    /// Forgets the boundaries seen and disarms.
    pub fn reset() {
        STATE.with(|s| *s.borrow_mut() = State::default());
    }

    /// Ends the process, with status [`CRASHED`], at the boundary reached
    /// `at`-th (from zero) after this call.
    pub fn arm(at: usize) {
        STATE.with(|s| {
            let mut s = s.borrow_mut();
            s.seen.clear();
            s.crash_at = Some(at);
        });
    }

    /// The boundaries passed since the last [`reset`] or [`arm`], in order.
    #[must_use]
    pub fn seen() -> Vec<&'static str> {
        STATE.with(|s| s.borrow().seen.clone())
    }

    pub(crate) fn reach(name: &'static str) {
        let crash = STATE.with(|s| {
            let mut s = s.borrow_mut();
            let at = s.seen.len();
            s.seen.push(name);
            s.crash_at == Some(at)
        });
        if crash {
            // A crash: nothing after this step runs, not even destructors.
            std::process::exit(CRASHED);
        }
    }
}
