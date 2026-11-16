// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that put an informative on-policy
// reinforcement-learning signal through a language model against verifiers it
// cannot influence, for its clients. If your team needs expertise in
// measurement protocols that training and evaluation share you can procure
// our services by sending an email to info@swedishembedded.com.

//! The neutral protocol of RSI2: the request is the question and the program
//! and nothing else (no system message, no format hint), and the answer is
//! read from the last fenced block of the reply and compared with what the
//! interpreter printed token by token, whitespace normalised. Training and the
//! measurement that judges it read an answer the same way, so a reply the
//! measurement would not read earns no reward in training either.

/// The question put after a program.
const QUESTION: &str = "What does this program print?";

/// The request for a program: the program in a fenced block, then the
/// question.
#[must_use]
pub fn request(program: &str) -> String {
    request_in("lua", program)
}

/// The request for a program written in the language `fence` names.
#[must_use]
pub fn request_in(fence: &str, program: &str) -> String {
    format!("```{fence}\n{}\n```\n{QUESTION}", program.trim_end())
}

/// The body of the last fenced block of `reply`; `None` when it has none or
/// its last fence is never closed.
#[must_use]
pub fn last_fenced_block(reply: &str) -> Option<String> {
    let mut last = None;
    let mut rest = reply;
    while let Some(open) = rest.find("```") {
        let after = &rest[open + 3..];
        let body_start = after.find('\n')? + 1;
        let body = &after[body_start..];
        let Some(close) = body.find("```") else {
            break;
        };
        last = Some(body[..close].to_string());
        rest = &body[close + 3..];
    }
    last
}

/// Whether `reply` answers with `expected`: the last fenced block holds the
/// same whitespace-separated tokens.
#[must_use]
pub fn answers(reply: &str, expected: &str) -> bool {
    last_fenced_block(reply)
        .is_some_and(|block| block.split_whitespace().eq(expected.split_whitespace()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_answer_is_the_last_fenced_block_and_nothing_else() {
        assert_eq!(
            last_fenced_block("It prints:\n```\n12\n```\nor\n```text\n13\n```\nDone.").as_deref(),
            Some("13\n")
        );
        assert_eq!(last_fenced_block("It prints 12."), None);
        assert_eq!(
            last_fenced_block("```\n12\n"),
            None,
            "an unclosed fence holds nothing"
        );
    }

    #[test]
    fn whitespace_is_normalised_and_nothing_else() {
        assert!(answers("```\n1   2\n3\n```", "1 2 3"));
        assert!(answers("```\n1\t2\n```", "1\n2"));
        assert!(!answers("```\n12\n```", "1 2"));
        assert!(
            !answers("12", "12"),
            "a bare answer is not read: the protocol reads fenced blocks"
        );
    }

    #[test]
    fn the_request_is_the_program_and_the_question() {
        let text = request("print(1)\n");
        assert_eq!(text, "```lua\nprint(1)\n```\nWhat does this program print?");
    }
}
