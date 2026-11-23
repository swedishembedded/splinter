// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements restart-safe controllers for long-running
// automation, with fenced leases and an auditable event trail, for its
// clients. If your team needs expertise in durable workflow state, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The records and states of the run state.

use std::fmt;

use splinter_core::digest::Digest;

use super::RunStateError;

/// Declares a state enum with its stored text form.
macro_rules! text_enum {
    ($(#[$meta:meta])* $name:ident { $($(#[$vmeta:meta])* $variant:ident = $text:literal),+ $(,)? }) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum $name {
            $($(#[$vmeta])* $variant),+
        }

        impl $name {
            /// The text this state is stored and logged as.
            #[must_use]
            pub fn as_str(self) -> &'static str {
                match self { $(Self::$variant => $text),+ }
            }

            pub(super) fn parse(text: &str) -> Result<Self, RunStateError> {
                match text {
                    $($text => Ok(Self::$variant),)+
                    other => Err(RunStateError::Corrupt(format!(
                        "unknown {} {other:?}", stringify!($name)
                    ))),
                }
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.as_str())
            }
        }
    };
}

text_enum! {
    /// Where a stage is in its life.
    StageState {
        /// Known, not started.
        Pending = "pending",
        /// Being run.
        Running = "running",
        /// Finished with a product; terminal.
        Completed = "completed",
        /// Finished with an error; may be retried.
        Failed = "failed",
    }
}

text_enum! {
    /// Where a logical rollout slot is.
    SlotState {
        /// Waiting for an attempt.
        Pending = "pending",
        /// An attempt is in flight.
        Running = "running",
        /// Holds its committed trial result; terminal.
        Finished = "finished",
        /// Given up on; terminal.
        Abandoned = "abandoned",
    }
}

text_enum! {
    /// Where a physical attempt at a slot is.
    AttemptState {
        /// Handed to the backend, not yet confirmed running.
        Launched = "launched",
        /// Confirmed running.
        Running = "running",
        /// Ended with a result.
        Succeeded = "succeeded",
        /// Ended with an error.
        Failed = "failed",
        /// Its backend was lost without an outcome (a restart found nothing).
        Lost = "lost",
    }
}

impl AttemptState {
    /// True for states an attempt does not leave.
    #[must_use]
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Succeeded | Self::Failed | Self::Lost)
    }
}

/// What a stage transition records beside the new state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StageOutcome {
    /// Nothing: for `Pending` and `Running`.
    None,
    /// The committed product's digest: required for `Completed`.
    Product(Digest),
    /// Why the stage failed: required for `Failed`.
    Error(String),
}

/// A stage as stored.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageRecord {
    /// The stage's content address.
    pub stage_id: Digest,
    /// Its name, for people.
    pub name: String,
    /// Its kind.
    pub kind: String,
    /// Its state.
    pub state: StageState,
    /// Bumped by every transition; the optimistic-concurrency token.
    pub version: u64,
    /// The product it committed, once `Completed`.
    pub product: Option<Digest>,
    /// Why it failed, once `Failed`.
    pub error: Option<String>,
    /// When it last changed, in Unix seconds.
    pub updated_at: u64,
}

/// The identity of a logical rollout slot: the factory contract's own.
pub use splinter_core::factory::execution::SlotKey;

/// A logical slot as stored.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotRecord {
    /// The slot.
    pub key: SlotKey,
    /// Its state.
    pub state: SlotState,
    /// The committed trial result, once `Finished`.
    pub trial: Option<Digest>,
    /// When it last changed, in Unix seconds.
    pub updated_at: u64,
}

/// The identity of a physical attempt; assigned by the database, never reused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AttemptId(pub u64);

impl fmt::Display for AttemptId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A physical attempt as stored.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttemptRecord {
    /// The attempt.
    pub id: AttemptId,
    /// The slot it works on.
    pub slot: SlotKey,
    /// The backend's handle for it (a container id, a pid), as JSON text.
    pub backend: String,
    /// Its state.
    pub state: AttemptState,
    /// When it was claimed, in Unix seconds.
    pub started_at: u64,
    /// When it reached a terminal state.
    pub finished_at: Option<u64>,
    /// The trial result it produced, when it succeeded.
    pub result: Option<Digest>,
}
