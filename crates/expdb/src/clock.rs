// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Time as a value, so tests control it and records carry a timestamp the
//! caller chose.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// A source of nanoseconds since the Unix epoch.
pub trait Clock: Send + Sync {
    /// The current time in nanoseconds since the Unix epoch.
    fn now_ns(&self) -> u64;
}

/// The system clock.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_ns(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
            .unwrap_or(0)
    }
}

/// A clock that advances by a fixed step on every reading.
#[derive(Debug)]
pub struct StepClock {
    next: AtomicU64,
    step: u64,
}

impl StepClock {
    /// A clock that reads `start`, then `start + step`, and so on.
    pub fn new(start: u64, step: u64) -> Self {
        Self {
            next: AtomicU64::new(start),
            step,
        }
    }
}

impl Clock for StepClock {
    fn now_ns(&self) -> u64 {
        self.next.fetch_add(self.step, Ordering::Relaxed)
    }
}
