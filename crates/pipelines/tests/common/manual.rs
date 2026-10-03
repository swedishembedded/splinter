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

/// The idle-current question the policy answers wrong until it is given
/// feedback.
pub const IDLE_QUESTION: &str = "How much current does the Frobnicator draw when idle?";

/// Other words for [`BAUD_QUESTION`].
pub const BAUD_VARIANTS: [&str; 3] = [
    "Which baud rate is the Frobnicator's console UART set to?",
    "What speed, in baud, does the Frobnicator console serial line use?",
    "Tell me the console UART baud rate of the Frobnicator.",
];

/// Other words for [`IDLE_QUESTION`].
pub const IDLE_VARIANTS: [&str; 3] = [
    "What is the Frobnicator's idle current draw?",
    "How many milliamps does the idle Frobnicator consume?",
    "Tell me the Frobnicator's current consumption when idle.",
];

/// A generator's reply to a request for variants of the question it is
/// shown: the three other wordings, repeating the answer; `None` when the
/// prompt is no such request.
pub fn variants_reply(prompt: &str) -> Option<String> {
    if !prompt.contains("differently worded questions") {
        return None;
    }
    let (wordings, reference) = if prompt.contains(BAUD_QUESTION) {
        (BAUD_VARIANTS, "115200 baud")
    } else {
        (IDLE_VARIANTS, "40 mA")
    };
    let tasks: Vec<_> = wordings
        .iter()
        .map(
            |w| json!({ "instruction": w, "subject": "Frobnicator", "reference": reference, "evidence": [{ "section": 0 }] }),
        )
        .collect();
    Some(json!({ "tasks": tasks }).to_string())
}

/// The policy; see the module documentation. It writes the variants of the
/// tasks it wrote when asked.
pub fn manual_policy() -> Scripted {
    Scripted::new(manual_reply)
}

/// The policy's reply to `prompt`; see [`manual_policy`].
pub fn manual_reply(prompt: &str) -> String {
    {
        if let Some(variants) = variants_reply(prompt) {
            variants
        } else if prompt.contains("You write training tasks") {
            json!({ "tasks": [
                {
                    "instruction": BAUD_QUESTION,
                    "subject": "Frobnicator",
                    "reference": "115200 baud",
                    "evidence": [{ "section": 1, "quote": BAUD_QUOTE }]
                },
                {
                    "instruction": IDLE_QUESTION,
                    "subject": "Frobnicator",
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
    }
}
