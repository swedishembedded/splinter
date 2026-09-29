// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The answer contract: how a question is put to the model, the one reply
//! shape it must answer in (`{"answer": string}`), how that reply is parsed,
//! how a fact is written as training data in the same shape, and how an
//! answer is judged against its reference.
//!
//! One module owns all five because they only work together: training on
//! bare answers trains the wrapper away and every strict parse of the
//! model's own correct reply then fails; a judge that normalizes
//! differently from the reference writer scores formatting, not knowledge.

use anyhow::Context;

/// The exact prompt one question sees, so an evaluated reply is asked the
/// same way a one-shot question is.
#[must_use]
pub fn question_prompt(question: &str) -> String {
    let prompt = format!(
        "Answer the question below from your knowledge. Reply with EXACTLY one JSON object \
         and nothing else - no prose, no code fences:\n\
         {{\"answer\": string}}\n\n\
         QUESTION:\n{}",
        question
    );
    prompt
}

/// Strips optional markdown code fences around a reply, so a model that
/// answered perfectly inside ```json fences still parses. Fences are the
/// one tolerated decoration; prose around the object is not.
#[must_use]
pub fn strip_fences(reply: &str) -> &str {
    let trimmed = reply.trim();
    let without = trimmed
        .strip_prefix("```")
        .and_then(|r| {
            r.trim_start_matches(|c: char| c.is_ascii_alphanumeric())
                .strip_prefix('\n')
        })
        .unwrap_or(trimmed);
    without
        .strip_suffix("```")
        .map(|r| r.trim())
        .unwrap_or(without)
}

/// Strict parse of an answer reply: exactly one
/// `{"answer": string}` object (fences tolerated, prose is not).
pub fn parse_answer(reply: &str) -> anyhow::Result<String> {
    let value: serde_json::Value =
        serde_json::from_str(strip_fences(reply)).with_context(|| {
            // A parse failure is a scored event; the raw reply is the evidence
            // a repair decision needs, so it rides in the error chain.
            format!("reply is not exactly one JSON object: {reply:?}")
        })?;
    let object = value
        .as_object()
        .with_context(|| "reply is not a JSON object".to_string())?;
    let answer = object
        .get("answer")
        .and_then(|v| v.as_str())
        .with_context(|| "reply object has no string \"answer\"".to_string())?;
    Ok(answer.to_string())
}

/// One fact as a chat training record, in the schema of the experience
/// pool: the question as context (not supervised), the answer as the
/// supervised turn - wrapped the way [`parse_answer`] reads it.
#[must_use]
pub fn training_record(run_id: &str, question: &str, answer: &str) -> serde_json::Value {
    // The assistant side teaches the reply shape `ask` parses, not just the
    // fact: fine-tuning on bare answers trains the wrapper away, and a
    // model that answers "84 MHz" without the {"answer": ...} object then
    // fails every strict parse of its own correct reply.
    let reply = serde_json::json!({ "answer": answer }).to_string();
    serde_json::json!({
        "messages": [
            { "role": "user", "content": question, "train": false },
            { "role": "assistant", "content": reply, "train": true },
        ],
        "metadata": { "run_id": run_id, "verified_by": [] },
    })
}

/// One number found in text, with the unit token attached to it.
#[derive(Debug, PartialEq, Clone)]
struct Number {
    value: f64,
    /// Parsed from the digits, so "168" stays exact and "10.5" is
    /// known-fractional; the tolerance rule keys off this.
    integer_written: bool,
    /// Lowercased unit right after the number ("/" dropped); empty when
    /// the number stands bare.
    unit: String,
}

/// Extracts every number with its attached unit from `text`. A unit is
/// the alphanumeric run after the number, across one run of whitespace
/// ("42 Mbit/s" and "42Mbit/s" both read unit "mbits"); punctuation ends
/// the scan, so "16-bit" reads unit "bit" and "3." reads unit "".
fn extract_numbers(text: &str) -> Vec<Number> {
    let chars: Vec<char> = text.chars().collect();
    let mut numbers = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if !chars[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        let start = i;
        let mut integer_written = true;
        while i < chars.len() && chars[i].is_ascii_digit() {
            i += 1;
        }
        if i + 1 < chars.len() && chars[i] == '.' && chars[i + 1].is_ascii_digit() {
            integer_written = false;
            i += 1;
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
        }
        let digits: String = chars[start..i].iter().collect();
        let value: f64 = digits.parse().unwrap_or(f64::INFINITY);
        // The unit: whitespace, then alphanumerics (µ included) and '/'.
        let mut j = i;
        while j < chars.len() && chars[j].is_whitespace() {
            j += 1;
        }
        let mut unit = String::new();
        while j < chars.len() && (chars[j].is_alphanumeric() || chars[j] == '/') {
            unit.push(chars[j].to_ascii_lowercase());
            j += 1;
        }
        if unit.chars().next().is_some_and(|c| c.is_ascii_digit()) {
            unit.clear(); // digits after the number are more digits, not a unit
        }
        numbers.push(Number {
            value,
            integer_written,
            unit: unit.replace('/', ""),
        });
    }
    numbers
}

/// Normalized text for comparison 1: lowercase, punctuation stripped
/// (except the decimal point between digits), whitespace collapsed and
/// the space between a number and its unit removed - so "10.5 Mbit/s"
/// and "10.5Mbps" normalize to the same string.
pub fn normalize_answer(text: &str) -> String {
    let chars: Vec<char> = text.to_lowercase().chars().collect();
    let mut out = String::new();
    let mut pending_space = false;
    for (i, &c) in chars.iter().enumerate() {
        if c.is_whitespace() {
            pending_space = out.chars().next_back().is_some();
            continue;
        }
        let decimal = c == '.'
            && out.chars().next_back().is_some_and(|p| p.is_ascii_digit())
            && chars.get(i + 1).is_some_and(|n| n.is_ascii_digit());
        if !c.is_alphanumeric() && !decimal {
            pending_space = out.chars().next_back().is_some_and(|p| p.is_ascii_digit());
            continue;
        }
        // Erase unit-spacing variance: no space between digit and letter.
        let prev_digit = out.chars().next_back().is_some_and(|p| p.is_ascii_digit());
        if pending_space && !(prev_digit && c.is_alphabetic()) {
            out.push(' ');
        }
        pending_space = false;
        out.push(c);
    }
    // Unit synonyms: "Mbit/s" and "Mbps" are one unit written two ways.
    // Only the "/s"-family collapses; nothing else is silently rewritten.
    out.replace("mbits", "mbps")
        .replace("kbits", "kbps")
        .replace("gbits", "gbps")
        .replace("bits", "bps")
}

/// Comparison 1: the normalized texts are equal.
fn text_matches(expected: &str, got: &str) -> bool {
    normalize_answer(expected) == normalize_answer(got)
}

/// Comparison 2: every reference number appears in the answer with the
/// same value and unit. An integer-written reference demands an exact
/// match; a fractional one allows 1% relative tolerance. Matching is
/// one-to-one, so a reference saying "3" twice is not satisfied by one.
fn numeric_matches(expected: &str, got: &str) -> bool {
    let answer = extract_numbers(got);
    let mut used = vec![false; answer.len()];
    for want in extract_numbers(expected) {
        let found = answer.iter().zip(&used).position(|(have, &used)| {
            !used
                && have.unit == want.unit
                && if want.integer_written {
                    have.value == want.value
                } else {
                    (have.value - want.value).abs() <= 0.01 * want.value.abs()
                }
        });
        match found {
            Some(index) => used[index] = true,
            None => return false,
        }
    }
    true
}

/// The two-way judgment: correct when both ways agree, with the basis
/// naming which comparison decided. A numberless reference is text-only.
pub fn judge(expected: &str, got: &str) -> (bool, &'static str) {
    let has_numbers = !extract_numbers(expected).is_empty();
    let by_text = text_matches(expected, got);
    if !has_numbers {
        return (by_text, "text");
    }
    if !by_text {
        return (false, "text");
    }
    // Both ways must agree; the numeric way decided (numbers are the
    // decisive content when a reference carries any).
    (numeric_matches(expected, got), "numeric")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit-spacing variance must not fail the text way.
    #[test]
    fn unit_spacing_variance_normalizes_away() {
        assert_eq!(
            normalize_answer("10.5 Mbit/s"),
            normalize_answer("10.5Mbps")
        );
        assert_eq!(normalize_answer("3.3 V"), normalize_answer("3.3V"));
        assert!(text_matches("10.5 Mbit/s", "10.5Mbps"));
    }
    /// Case, punctuation and whitespace collapse.
    #[test]
    fn case_punctuation_and_whitespace_collapse() {
        assert!(text_matches(
            "The max clock is 168 MHz.",
            "the max clock is 168 mhz"
        ));
        assert!(text_matches(
            "Three SPI controllers",
            "three   spi controllers"
        ));
        // A decimal point inside a number survives; one between words does not.
        assert!(text_matches("10.5 Mbps", "10.5Mbps"));
        assert!(text_matches("1. stop", "1 stop"));
    }
    /// Number tolerance: exact for integers, 1% for fractional.
    #[test]
    fn numeric_tolerance_follows_the_reference_form() {
        // Integer reference: 168.001 is a different answer, not tolerance.
        assert!(numeric_matches("168 MHz", "168 MHz"));
        assert!(!numeric_matches("168 MHz", "168.001 MHz"));
        // Fractional reference: 1% relative tolerance applies.
        assert!(numeric_matches("10.5 Mbps", "10.51 Mbps"));
        assert!(!numeric_matches("10.5 Mbps", "10.7 Mbps"));
        // Units must match, not merely values.
        assert!(!numeric_matches("10.5 Mbps", "10.5 kBd"));
    }
    /// A number missing from the answer, or a wrong one, fails it.
    #[test]
    fn missing_and_wrong_numbers_fail() {
        assert!(!numeric_matches("4 timers and 168 MHz", "168 MHz"));
        assert!(!numeric_matches("3.3 V", "5.0 V"));
        // Two references need two answer numbers: one-to-one.
        assert!(!numeric_matches("2 buses, 2 clocks", "2 buses"));
        // Extra answer numbers do not rescue a mismatch.
        assert!(!numeric_matches("16-bit timer", "32-bit timer"));
    }
    /// A numberless reference is judged by text only - the numeric way
    /// would find nothing to check and must not be the gate.
    #[test]
    fn numberless_references_judge_by_text() {
        assert_eq!(
            judge("Yes, the HSI is trimmable", "yes the hsi is trimmable"),
            (true, "text")
        );
        assert_eq!(judge("Yes", "No"), (false, "text"));
    }
    /// The fence-stripping parser accepts fenced answers and refuses
    /// prose-wrapped JSON - the same strict gate `ask` applies.
    #[test]
    fn fenced_answers_parse_and_prose_wrapped_json_is_refused() {
        assert_eq!(
            parse_answer("```json\n{\"answer\": \"42 Mbit/s\"}\n```").unwrap(),
            "42 Mbit/s"
        );
        assert!(
            parse_answer("The answer is {\"answer\": \"42 Mbit/s\"}").is_err(),
            "prose around the object is a parse failure"
        );
    }

    /// An ask reply parses to its answer string; prose is a refusal.
    #[test]
    fn ask_replies_parse_strictly() {
        assert_eq!(
            parse_answer("{\"answer\": \"42 Mbit/s\"}").unwrap(),
            "42 Mbit/s"
        );
        assert_eq!(
            parse_answer("```\n{\"answer\": \"168 MHz\"}\n```").unwrap(),
            "168 MHz"
        );
        assert!(parse_answer("The answer is 168 MHz.").is_err());
        assert!(parse_answer("{\"result\": \"168 MHz\"}").is_err());
    }

    /// An explore-produced record is exactly what the trainer's dataset reader
    /// parses,
    /// and the assistant side teaches the shape `ask` parses: the answer
    /// wrapped as one {"answer": ...} object. Training on bare answers
    /// makes a fine-tuned model drop the wrapper and every strict parse
    /// then fails on the model's own (correct) reply.
    #[test]
    fn an_explore_record_is_valid_pool_input() {
        let record = training_record("explore-test", "What is the max?", "42 Mbit/s");
        assert_eq!(
            record["messages"][0],
            serde_json::json!({"role": "user", "content": "What is the max?", "train": false})
        );
        assert_eq!(
            record["messages"][1],
            serde_json::json!({"role": "assistant", "content": r#"{"answer":"42 Mbit/s"}"#, "train": true})
        );
        assert_eq!(record["metadata"]["run_id"], "explore-test");
        assert_eq!(record["metadata"]["verified_by"], serde_json::json!([]));
    }
}
