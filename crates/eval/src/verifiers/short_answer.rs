// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verifiers that grade a model's short answers
// without mistaking its verbosity for a wrong answer, for its clients. If
// your team needs expertise in evidence-based evaluation of language models,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Taking a final short answer out of a sentence.
//!
//! A model that answers "There are **3,000** millilitres in 3 litres" has
//! answered, and so has one that says "The output is:" above a fenced block
//! holding the output. The answer is taken out only when it is unambiguous:
//! the candidates are what the answer emphasises or fences, else (for a
//! numeric reference) the numbers it states, and exactly one distinct
//! candidate must be left. A
//! hedge that lists several candidates therefore never yields an answer, and
//! neither does a sentence around a reference that has no type to look for.

use super::normalise::Normalisation;

/// Whether `answer` has exactly one candidate of `reference`'s type and it is
/// `reference`, both under `normalisation`.
pub(crate) fn sole_candidate_is(
    answer: &str,
    reference: &str,
    normalisation: &Normalisation,
) -> bool {
    let reference = normalisation.apply(reference);
    let numeric = is_number(&reference);
    let mut spans = emphasised_spans(answer);
    spans.extend(fenced_blocks(answer));
    let mut candidates: Vec<String> = Vec::new();
    let mut offer = |text: &str| {
        let text = normalisation.apply(text);
        let found = if numeric {
            numbers_in(&text)
        } else {
            vec![text]
        };
        for candidate in found.into_iter().filter(|c| !c.is_empty()) {
            if !candidates.contains(&candidate) {
                candidates.push(candidate);
            }
        }
    };
    if !spans.is_empty() {
        spans.iter().for_each(|span| offer(span));
    } else if numeric {
        offer(answer);
    }
    candidates.len() == 1 && candidates[0] == reference
}

/// The texts between a pair of `**`, in order.
fn emphasised_spans(text: &str) -> Vec<&str> {
    let parts: Vec<&str> = text.split("**").collect();
    // Part i is inside a pair when i is odd and a closing `**` follows it.
    parts
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 2 == 1 && i + 1 < parts.len())
        .map(|(_, part)| *part)
        .collect()
}

/// The contents of the fenced blocks of `text` (between a line opening with
/// three backticks, whatever language it names, and the next three
/// backticks), in order. An unclosed fence holds nothing.
fn fenced_blocks(text: &str) -> Vec<&str> {
    let mut blocks = Vec::new();
    let mut rest = text;
    while let Some(open) = rest.find("```") {
        let after = &rest[open + 3..];
        let Some(body_start) = after.find('\n') else {
            break;
        };
        let body = &after[body_start + 1..];
        let Some(close) = body.find("```") else {
            break;
        };
        blocks.push(&body[..close]);
        rest = &body[close + 3..];
    }
    blocks
}

/// Whether `text` is one plain number: optional minus, digits, optional
/// decimals.
fn is_number(text: &str) -> bool {
    let digits = text.strip_prefix('-').unwrap_or(text);
    let (whole, fraction) = digits.split_once('.').unwrap_or((digits, "0"));
    !whole.is_empty()
        && whole.bytes().all(|b| b.is_ascii_digit())
        && !fraction.is_empty()
        && fraction.bytes().all(|b| b.is_ascii_digit())
}

/// The numbers `text` states, as written, skipping digits that belong to a
/// word (`H2O`, `x2`).
fn numbers_in(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut found = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if !chars[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        let start = i;
        while i < chars.len() && chars[i].is_ascii_digit() {
            i += 1;
        }
        if i + 1 < chars.len() && chars[i] == '.' && chars[i + 1].is_ascii_digit() {
            i += 1;
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
        }
        let before = start.checked_sub(1).map(|b| chars[b]);
        let after = chars.get(i).copied();
        if before.is_some_and(char::is_alphabetic) || after.is_some_and(char::is_alphabetic) {
            continue;
        }
        let negative = before == Some('-')
            && start
                .checked_sub(2)
                .is_none_or(|b| !chars[b].is_alphanumeric());
        let number: String = chars[start..i].iter().collect();
        found.push(if negative {
            format!("-{number}")
        } else {
            number
        });
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: Normalisation = Normalisation::LENIENT;

    #[test]
    fn numbers_are_found_whole_and_signed() {
        assert_eq!(numbers_in("-4 and 3.5 and x2 and 7."), ["-4", "3.5", "7"]);
        assert_eq!(numbers_in("well-known 12"), ["12"]);
    }

    #[test]
    fn emphasis_needs_a_closing_pair() {
        assert_eq!(emphasised_spans("a **b** c **d** e"), ["b", "d"]);
        assert!(emphasised_spans("a **b c").is_empty());
    }

    #[test]
    fn one_candidate_of_the_type_is_the_answer() {
        assert!(sole_candidate_is("It is 7.", "7", &N));
        assert!(!sole_candidate_is("7 or 8", "7", &N));
        assert!(sole_candidate_is("Note 3: **7**", "7", &N));
        assert!(!sole_candidate_is("The capital is Paris", "Paris", &N));
    }

    #[test]
    fn a_fenced_block_is_a_candidate_and_two_different_ones_are_a_hedge() {
        let output = "The output of the Lua code is:\n\n```\naaabbbccc\n```";
        assert!(sole_candidate_is(output, "aaabbbccc", &N));
        let lines = "The Lua program prints:\n\n```text\nuppercase\nlowercase\n```\n\nbecause ...";
        assert!(sole_candidate_is(lines, "uppercase\nlowercase", &N));
        let numeric = "It prints:\n```\n6\n```\nExplanation: t[6] = 6 adds a sixth element.";
        assert!(sole_candidate_is(numeric, "6", &N));
        let hedge = "Either\n```\n1.5\n```\nor\n```\n2.5\n```";
        assert!(!sole_candidate_is(hedge, "1.5", &N));
        assert!(!sole_candidate_is(output, "aaabbb", &N));
        assert_eq!(fenced_blocks("a ```lua\nx\n``` b ```\ny"), ["x\n"]);
    }
}
