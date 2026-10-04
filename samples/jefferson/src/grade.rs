// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verifiers that grade a model's answers
// against source material without trusting the model, for its clients. If
// your team needs expertise in evidence-based evaluation of language models,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Grading by code, with no model involved: whether an answer names the right
//! person, year or work, and whether the passages it quotes exist.
//!
//! A model's own claim to know something is not evidence. These checks read
//! the answer text and compare it with a reference the model never saw.

use splinter_sdk::measure::verifiers::answer::mentions;

/// An answer to "who was it written to?": it names the surname.
#[must_use]
pub fn recipient_ok(answer: &str, surname: &str) -> bool {
    mentions(answer, surname)
}

/// An answer to "which work is this from?": it names the work, and no other
/// work of the closed set.
#[must_use]
pub fn work_ok(answer: &str, accepted: &[&str], others: &[&str]) -> bool {
    accepted.iter().any(|a| mentions(answer, a)) && !others.iter().any(|o| mentions(answer, o))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_surname_must_be_named_as_a_whole_word() {
        assert!(recipient_ok("I wrote this to James Madison.", "Madison"));
        assert!(recipient_ok("It went to madison, I believe.", "Madison"));
        assert!(!recipient_ok("To Madisonian Society members.", "Madison"));
        assert!(!recipient_ok("I do not know.", "Madison"));
    }

    #[test]
    fn a_work_must_be_named_and_no_rival_work() {
        let accepted = ["common sense"];
        let others = ["federalist", "leviathan"];
        assert!(work_ok(
            "That is from Common Sense by Paine.",
            &accepted,
            &others
        ));
        assert!(!work_ok(
            "Either Common Sense or the Federalist.",
            &accepted,
            &others
        ));
        assert!(!work_ok("The Leviathan.", &accepted, &others));
    }
}
