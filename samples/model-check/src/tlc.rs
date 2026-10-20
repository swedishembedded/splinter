// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What TLC said.

/// The outcome of one TLC run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TlcOutcome {
    /// No error; the counts TLC reported.
    Clean {
        /// States TLC generated.
        generated: u64,
        /// Distinct states it found.
        distinct: u64,
    },
    /// A named invariant failed.
    InvariantViolated {
        /// The invariant's name in the configuration.
        name: String,
    },
    /// A state with no successor was reached.
    Deadlock,
    /// Anything else: bad configuration, parse error, crash.
    Failed {
        /// Why, with TLC's output.
        message: String,
    },
}

fn count_before(output: &str, suffix: &str) -> Option<u64> {
    output.lines().find_map(|line| {
        line.split(',')
            .find_map(|part| part.trim().strip_suffix(suffix)?.trim().parse().ok())
    })
}

/// Reads TLC's console output and exit code (0 clean, 12 invariant violated,
/// 11 deadlock; anything else is a failure to run).
pub fn parse(output: &str, exit_code: Option<i32>) -> TlcOutcome {
    let failed = |what: &str| TlcOutcome::Failed {
        message: format!("{what} (exit code {exit_code:?}): {}", output.trim()),
    };
    let violated = output.lines().find_map(|line| {
        line.strip_prefix("Error: Invariant ")?
            .strip_suffix(" is violated.")
    });
    match (exit_code, violated) {
        (Some(12), Some(name)) => TlcOutcome::InvariantViolated {
            name: name.to_string(),
        },
        (Some(11), _) if output.contains("Error: Deadlock reached.") => TlcOutcome::Deadlock,
        (Some(0), _) if output.contains("Model checking completed. No error has been found.") => {
            match (
                count_before(output, "states generated"),
                count_before(output, "distinct states found"),
            ) {
                (Some(generated), Some(distinct)) => TlcOutcome::Clean {
                    generated,
                    distinct,
                },
                _ => failed("no state counts in the output of a clean run"),
            }
        }
        _ => failed("TLC did not finish a check"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CLEAN: &str = "Model checking completed. No error has been found.\n\
        53 states generated, 35 distinct states found, 0 states left on queue.\n";

    #[test]
    fn a_clean_run_reports_its_state_counts() {
        assert_eq!(
            parse(CLEAN, Some(0)),
            TlcOutcome::Clean {
                generated: 53,
                distinct: 35
            }
        );
    }

    #[test]
    fn a_violated_invariant_is_named() {
        let out =
            "Error: Invariant Bounded is violated.\nError: The behavior up to this point is:\n";
        assert_eq!(
            parse(out, Some(12)),
            TlcOutcome::InvariantViolated {
                name: "Bounded".into()
            }
        );
    }

    #[test]
    fn a_deadlock_is_reported() {
        let out = "Error: Deadlock reached.\nError: The behavior up to this point is:\n";
        assert_eq!(parse(out, Some(11)), TlcOutcome::Deadlock);
    }

    #[test]
    fn anything_else_is_a_failure_that_keeps_the_output() {
        for (out, code) in [
            ("File not found.\n", Some(255)),
            ("", None),
            (CLEAN, Some(1)),
            (
                "Model checking completed. No error has been found.\n",
                Some(0),
            ),
        ] {
            match parse(out, code) {
                TlcOutcome::Failed { message } => assert!(message.contains(out.trim())),
                other => panic!("{out:?} {code:?} parsed as {other:?}"),
            }
        }
    }
}
