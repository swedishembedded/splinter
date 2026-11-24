// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements online reinforcement-learning controllers
// that never train a group twice or lose one they produced, for its clients.
// If your team needs expertise in asynchronous policy optimisation, you can
// procure our services by sending an email to info@swedishembedded.com.

//! A group of rollouts of one task, as an online policy-gradient run
//! produces, scores and hands them to a learner.
//!
//! A group is `size` independent rollouts of one task; its rewards are
//! compared with each other, so it is the unit a learner admits, trains and
//! commits. Its life, recorded durably by the controller:
//!
//! ```text
//! pending -> running -> scored -> filtered            (no signal: not trained)
//!                              -> stale               (too old for the learner)
//!                              -> sent -> applied -> committed
//!                                      -> stale       (the learner dropped it as too old)
//!                                         applied -> sent (re-sent: the learner lost an update it had not committed)
//! ```
//!
//! `filtered`, `stale` and `committed` are terminal. A group is durable at
//! the learner only once `committed` (in a checkpoint); until then a resumed
//! controller sends it again, and the learner deduplicates by [`GroupKey`].
//!
//! Each rollout of a group is `pending`, `running`, `finished` (with a
//! [`RolloutOutcome`]) or `abandoned` (its infrastructure failed more often
//! than the run allows: no outcome, never a reward of zero).

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// The longest group key accepted.
pub const MAX_GROUP_KEY_LEN: usize = 256;

/// Why a value of this module was refused.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum GroupError {
    /// A group key that is empty, too long, or holds whitespace or a control
    /// character.
    #[error(
        "group key {0:?} is not 1-{MAX_GROUP_KEY_LEN} printable characters without whitespace"
    )]
    Key(String),
    /// A version span whose oldest version is newer than its newest.
    #[error("policy-version span [{oldest}, {newest}] is reversed")]
    Span {
        /// The oldest version given.
        oldest: u64,
        /// The newest version given.
        newest: u64,
    },
    /// A reward that is not a finite number.
    #[error("reward {0} is not finite")]
    Reward(f64),
    /// A stored state name this code does not know.
    #[error("unknown {kind} state {name:?}")]
    State {
        /// Which state enum.
        kind: &'static str,
        /// The text read.
        name: String,
    },
}

/// The identity of a group: what the learner deduplicates by.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct GroupKey(String);

impl GroupKey {
    /// A validated key.
    ///
    /// # Errors
    /// The key is empty, longer than [`MAX_GROUP_KEY_LEN`], or holds
    /// whitespace or a control character.
    pub fn new(key: impl Into<String>) -> Result<Self, GroupError> {
        let key = key.into();
        let printable = key.chars().all(|c| !c.is_whitespace() && !c.is_control());
        if key.is_empty() || key.len() > MAX_GROUP_KEY_LEN || !printable {
            return Err(GroupError::Key(key));
        }
        Ok(Self(key))
    }

    /// The key of the `seq`-th group of the run namespace `namespace`.
    ///
    /// # Errors
    /// The namespace makes an invalid key.
    pub fn numbered(namespace: &str, seq: u64) -> Result<Self, GroupError> {
        Self::new(format!("{namespace}:{seq}"))
    }

    /// The key's text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for GroupKey {
    type Error = GroupError;
    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<GroupKey> for String {
    fn from(value: GroupKey) -> Self {
        value.0
    }
}

impl fmt::Display for GroupKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Declares a state enum with its stored text form.
macro_rules! state_enum {
    ($(#[$meta:meta])* $name:ident { $($(#[$vmeta:meta])* $variant:ident = $text:literal),+ $(,)? }) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        #[serde(rename_all = "snake_case")]
        pub enum $name {
            $($(#[$vmeta])* $variant),+
        }

        impl $name {
            /// The text this state is stored and logged as.
            #[must_use]
            pub fn as_str(self) -> &'static str {
                match self { $(Self::$variant => $text),+ }
            }
        }

        impl FromStr for $name {
            type Err = GroupError;
            fn from_str(text: &str) -> Result<Self, Self::Err> {
                match text {
                    $($text => Ok(Self::$variant),)+
                    other => Err(GroupError::State { kind: stringify!($name), name: other.to_string() }),
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

state_enum! {
    /// Where a group is in its life (the module documentation draws it).
    GroupState {
        /// Created, no rollout started.
        Pending = "pending",
        /// Some rollout started, not every one ended.
        Running = "running",
        /// Every rollout ended; the group waits to be filtered, dropped or sent.
        Scored = "scored",
        /// Not training material; terminal.
        Filtered = "filtered",
        /// Too far behind the learner's policy; terminal.
        Stale = "stale",
        /// Handed to the learner, not yet applied.
        Sent = "sent",
        /// Applied by the learner, not yet in a durable checkpoint.
        Applied = "applied",
        /// In a durable checkpoint of the learner; terminal.
        Committed = "committed",
    }
}

impl GroupState {
    /// Whether a group leaves this state no more.
    #[must_use]
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Filtered | Self::Stale | Self::Committed)
    }

    /// Whether a group may go from this state to `to`.
    #[must_use]
    pub fn may_become(self, to: Self) -> bool {
        use GroupState::{Applied, Committed, Filtered, Pending, Running, Scored, Sent, Stale};
        matches!(
            (self, to),
            (Pending, Running)
                | (Running, Scored)
                | (Scored, Filtered | Stale | Sent)
                | (Sent, Applied | Stale)
                | (Applied, Committed | Sent)
        )
    }
}

state_enum! {
    /// Where one rollout of a group is.
    RolloutState {
        /// Waiting for a rollout slot.
        Pending = "pending",
        /// Started; a resumed controller starts it again.
        Running = "running",
        /// Ended with an outcome; terminal.
        Finished = "finished",
        /// Its infrastructure failed past the run's retry budget; terminal,
        /// with no outcome.
        Abandoned = "abandoned",
    }
}

impl RolloutState {
    /// Whether a rollout leaves this state no more.
    #[must_use]
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Finished | Self::Abandoned)
    }
}

/// The oldest and the newest policy version among a rollout's model
/// requests, stored as `[oldest, newest]` (the agent runtime's
/// `policy_version_span`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "[u64; 2]", into = "[u64; 2]")]
pub struct VersionSpan {
    oldest: u64,
    newest: u64,
}

impl VersionSpan {
    /// The span from `oldest` to `newest`.
    ///
    /// # Errors
    /// `oldest` is newer than `newest`.
    pub fn new(oldest: u64, newest: u64) -> Result<Self, GroupError> {
        if oldest > newest {
            return Err(GroupError::Span { oldest, newest });
        }
        Ok(Self { oldest, newest })
    }

    /// Every request at `version`.
    #[must_use]
    pub fn at(version: u64) -> Self {
        Self {
            oldest: version,
            newest: version,
        }
    }

    /// The oldest version.
    #[must_use]
    pub fn oldest(self) -> u64 {
        self.oldest
    }

    /// The newest version.
    #[must_use]
    pub fn newest(self) -> u64 {
        self.newest
    }

    /// How many versions `oldest` is behind `learner`: what the staleness
    /// bound is compared against (zero when the learner is not ahead).
    #[must_use]
    pub fn staleness(self, learner: u64) -> u64 {
        learner.saturating_sub(self.oldest)
    }
}

impl TryFrom<[u64; 2]> for VersionSpan {
    type Error = GroupError;
    fn try_from([oldest, newest]: [u64; 2]) -> Result<Self, Self::Error> {
        Self::new(oldest, newest)
    }
}

impl From<VersionSpan> for [u64; 2] {
    fn from(span: VersionSpan) -> Self {
        [span.oldest, span.newest]
    }
}

/// How one finished rollout ended. An infrastructure failure is not an
/// outcome: it is retried or abandoned, never scored.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "RawOutcome")]
pub struct RolloutOutcome {
    /// The verified reward; `None` when the rollout could not be scored
    /// (left out of its group's baseline, never read as zero).
    pub reward: Option<f64>,
    /// The policy tokens it sampled: what a learner trains on and what the
    /// token allowance counts.
    pub tokens: u64,
    /// The policy versions its requests were decoded by.
    pub span: VersionSpan,
    /// Whether every request carries the exact token ids and behaviour
    /// log-probabilities of a policy trace: only such a rollout is training
    /// material.
    pub exact: bool,
    /// Whether some request stopped at its length limit.
    pub truncated: bool,
}

/// The unchecked wire form of a [`RolloutOutcome`].
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawOutcome {
    reward: Option<f64>,
    tokens: u64,
    span: VersionSpan,
    exact: bool,
    truncated: bool,
}

impl TryFrom<RawOutcome> for RolloutOutcome {
    type Error = GroupError;
    fn try_from(raw: RawOutcome) -> Result<Self, Self::Error> {
        let outcome = Self {
            reward: raw.reward,
            tokens: raw.tokens,
            span: raw.span,
            exact: raw.exact,
            truncated: raw.truncated,
        };
        outcome.validate()?;
        Ok(outcome)
    }
}

impl RolloutOutcome {
    /// Checks the reward is finite.
    ///
    /// # Errors
    /// The reward is NaN or infinite.
    pub fn validate(&self) -> Result<(), GroupError> {
        match self.reward {
            Some(reward) if !reward.is_finite() => Err(GroupError::Reward(reward)),
            _ => Ok(()),
        }
    }

    /// Whether this rollout may be trained on: scored, exact, and not cut at
    /// a length limit (a learner trains none of a truncated rollout and
    /// leaves it out of its group's advantages, so it cannot make a group
    /// one).
    #[must_use]
    pub fn trainable(&self) -> bool {
        self.reward.is_some() && self.exact && !self.truncated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_group_moves_only_along_its_drawn_lifecycle() {
        use GroupState::*;
        let all = [
            Pending, Running, Scored, Filtered, Stale, Sent, Applied, Committed,
        ];
        let allowed = [
            (Pending, Running),
            (Running, Scored),
            (Scored, Filtered),
            (Scored, Stale),
            (Scored, Sent),
            (Sent, Applied),
            (Sent, Stale),
            (Applied, Committed),
            (Applied, Sent),
        ];
        for from in all {
            for to in all {
                assert_eq!(
                    from.may_become(to),
                    allowed.contains(&(from, to)),
                    "{from} -> {to}"
                );
            }
            if from.is_terminal() {
                assert!(
                    all.iter().all(|to| !from.may_become(*to)),
                    "{from} is terminal"
                );
            }
            assert_eq!(from.as_str().parse::<GroupState>().unwrap(), from);
        }
    }

    #[test]
    fn keys_spans_and_outcomes_are_validated_where_they_are_read() {
        assert!(GroupKey::new("").is_err());
        assert!(GroupKey::new("a b").is_err());
        assert!(GroupKey::new("a\u{7}").is_err());
        assert_eq!(GroupKey::numbered("run", 7).unwrap().as_str(), "run:7");
        assert!(VersionSpan::new(3, 2).is_err());
        assert_eq!(VersionSpan::new(2, 5).unwrap().staleness(9), 7);
        assert_eq!(VersionSpan::at(9).staleness(4), 0);

        let line = r#"{"reward":0.5,"tokens":10,"span":[3,4],"exact":true,"truncated":false}"#;
        let outcome: RolloutOutcome = serde_json::from_str(line).unwrap();
        assert_eq!(outcome.span, VersionSpan::new(3, 4).unwrap());
        assert_eq!(serde_json::to_string(&outcome).unwrap(), line);
        assert!(serde_json::from_str::<RolloutOutcome>(&line.replace("[3,4]", "[4,3]")).is_err());
        assert!(serde_json::from_str::<RolloutOutcome>(
            &line.replace("\"exact\"", "\"x\":1,\"exact\"")
        )
        .is_err());
        let unscored: RolloutOutcome = serde_json::from_str(&line.replace("0.5", "null")).unwrap();
        assert!(
            !unscored.trainable(),
            "an unscored rollout is not training material"
        );
    }
}
