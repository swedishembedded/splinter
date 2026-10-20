// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What the Lua explorer (`lua/explore.lua`) said about a module.

/// The explorer's verdict on one module, from its `key=value` output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Exploration {
    /// Every reachable state satisfies every invariant.
    Complete {
        /// Distinct reachable states.
        states: u64,
        /// Transitions executed.
        transitions: u64,
    },
    /// A reachable state violates an invariant.
    InvariantViolated {
        /// The violated invariant.
        name: String,
    },
    /// The module itself failed (error in a step, a mutated input, bad data).
    RuntimeError {
        /// The explorer's message.
        message: String,
    },
    /// More states than the exploration bound.
    BoundExceeded,
    /// The output is not the explorer's.
    Unreadable {
        /// What was wrong with the output.
        message: String,
    },
}

fn field<'a>(output: &'a str, key: &str) -> Option<&'a str> {
    output
        .lines()
        .find_map(|line| line.strip_prefix(key)?.strip_prefix('='))
}

fn number(output: &str, key: &str) -> Option<u64> {
    field(output, key)?.trim().parse().ok()
}

/// Reads the explorer's stdout and exit code (0 ok, 1 invariant violated,
/// 2 runtime error, 3 bound exceeded).
pub fn parse(output: &str, exit_code: Option<i32>) -> Exploration {
    let unreadable = |what: &str| Exploration::Unreadable {
        message: format!("{what} (exit code {exit_code:?})"),
    };
    match (field(output, "result"), exit_code) {
        (Some("ok"), Some(0)) => match (number(output, "states"), number(output, "transitions")) {
            (Some(states), Some(transitions)) => Exploration::Complete {
                states,
                transitions,
            },
            _ => unreadable("result=ok without state counts"),
        },
        (Some("invariant-violated"), Some(1)) => match field(output, "invariant") {
            Some(name) => Exploration::InvariantViolated {
                name: name.trim().to_string(),
            },
            None => unreadable("invariant-violated without a name"),
        },
        (Some("runtime-error"), Some(2)) => Exploration::RuntimeError {
            message: field(output, "error")
                .unwrap_or("no message")
                .trim()
                .to_string(),
        },
        (Some("bound-exceeded"), Some(3)) => Exploration::BoundExceeded,
        _ => unreadable("unexpected explorer output"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_each_result() {
        assert_eq!(
            parse("result=ok\nstates=4\ntransitions=3\ndepth=3\n", Some(0)),
            Exploration::Complete {
                states: 4,
                transitions: 3
            }
        );
        assert_eq!(
            parse(
                "result=invariant-violated\nstates=2\ninvariant=Bounded\ntrace.0=[]\n",
                Some(1)
            ),
            Exploration::InvariantViolated {
                name: "Bounded".into()
            }
        );
        assert_eq!(
            parse("result=runtime-error\nerror=step enqueue: boom\n", Some(2)),
            Exploration::RuntimeError {
                message: "step enqueue: boom".into()
            }
        );
        assert_eq!(
            parse("result=bound-exceeded\nstates=5\n", Some(3)),
            Exploration::BoundExceeded
        );
    }

    #[test]
    fn a_result_that_disagrees_with_the_exit_code_is_unreadable() {
        assert!(matches!(
            parse("result=ok\nstates=4\ntransitions=3\n", Some(1)),
            Exploration::Unreadable { .. }
        ));
        assert!(matches!(
            parse("garbage", Some(0)),
            Exploration::Unreadable { .. }
        ));
        assert!(matches!(
            parse("result=ok\n", Some(0)),
            Exploration::Unreadable { .. }
        ));
    }
}
