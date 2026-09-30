// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The manual the `learn` specs learn from, and the scripted policy that
//! learns it: it writes two recall tasks grounded in the manual, answers
//! the baud rate right and the idle current wrong until it is given
//! feedback, and critiques a failed attempt.

use serde_json::json;

use super::Scripted;

/// A short manual with two facts worth asking about.
pub const MANUAL: &str = "# Frobnicator manual

The Frobnicator is a bench power controller.

## Console

The console UART of the Frobnicator runs at 115200 baud with eight data bits.

## Power

The Frobnicator draws 40 mA when idle and 900 mA at full load.
";

/// The quote the baud-rate task is grounded in.
pub const BAUD_QUOTE: &str = "runs at 115200 baud";

/// The baud-rate question the policy answers right.
pub const BAUD_QUESTION: &str = "At what baud rate does the Frobnicator console UART run?";

/// The policy; see the module documentation.
pub fn manual_policy() -> Scripted {
    Scripted::new(|prompt| {
        if prompt.contains("You write training tasks") {
            json!({ "tasks": [
                {
                    "instruction": BAUD_QUESTION,
                    "reference": "115200 baud",
                    "evidence": [{ "section": 1, "quote": BAUD_QUOTE }]
                },
                {
                    "instruction": "How much current does the Frobnicator draw when idle?",
                    "reference": "40 mA",
                    "evidence": [{ "section": 2, "quote": "40 mA when idle" }]
                }
            ]})
            .to_string()
        } else if prompt.contains("You are reviewing an attempt") {
            "The idle current is wrong: it is not what the power specification gives.".into()
        } else if prompt.contains("baud rate") {
            "115200 baud".into()
        } else if prompt.contains("Feedback on an earlier attempt") {
            "40 mA".into()
        } else {
            "12 mA".into()
        }
    })
}
