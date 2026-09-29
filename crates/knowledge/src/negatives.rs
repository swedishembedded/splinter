// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements delegated-task coding agents that grow
// their own training data. If your team needs expertise in data extraction
// from technical documents, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Scope negatives: out-of-scope variants of in-scope questions, trained
//! toward a fixed abstention so a model learns where its knowledge ends.

/// The fixed reply a question about an out-of-scope device trains toward:
/// the scope boundary. Without such examples an adapter answers
/// out-of-family questions with in-family numbers - the measured probe
/// returned "168 MHz" for an STM32F103 whose true maximum is 72 MHz - and
/// a confident wrong number on the wrong chip is worse than an honest
/// refusal.
pub const NOT_COVERED: &str = "That device is not covered by this fact sheet.";

/// The negative variant of one anchored question: the in-scope device
/// identifier is replaced with an out-of-scope one, keeping the question
/// otherwise verbatim. `None` when the question does not name the
/// identifier (the anchor gate makes that rare, not impossible).
pub fn negative_question(
    question: &str,
    identifier: &str,
    negative: &str,
    identifiers: &[String],
) -> Option<String> {
    let start = question.find(identifier)?;
    let mut end = start + identifier.len();
    // A question names the fact sheet's devices as a slash-separated run
    // ("STM32F405 / STM32F407"). Substituting one member leaves a covered
    // device in the question, which trains a compound-question refusal
    // instead of the scope boundary the probe asks for; the whole run goes.
    loop {
        let tail = &question[end..];
        let ws = tail.len() - tail.trim_start().len();
        let Some(after) = tail[ws..].strip_prefix('/') else {
            break;
        };
        let ws2 = after.len() - after.trim_start().len();
        let candidate = &question[end + ws + 1 + ws2..];
        let Some((_, id_len)) = identifiers.iter().find_map(|id| {
            candidate
                .get(..id.len())
                .filter(|head| head.eq_ignore_ascii_case(id))
                .map(|_| (id, id.len()))
        }) else {
            break;
        };
        end += ws + 1 + ws2 + id_len;
    }
    Some(format!(
        "{}{negative}{}",
        &question[..start],
        &question[end..]
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scope negatives teach the boundary an all-positive dataset cannot
    /// express: the same question with an out-of-scope device substituted
    /// must train the abstention, because the measured alternative - a
    /// facts-only adapter - answered "168 MHz" for an STM32F103 whose true
    /// maximum is 72 MHz. A confident wrong number on the wrong chip is
    /// worse than an honest refusal.
    #[test]
    fn a_scope_negative_swaps_the_device_and_trains_the_boundary() {
        let ids: Vec<String> = ["STM32F405", "STM32F407"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let question = "What is the maximum CPU clock frequency of the STM32F407?";
        let negative = negative_question(question, "STM32F407", "STM32F103", &ids).unwrap();
        assert_eq!(
            negative,
            "What is the maximum CPU clock frequency of the STM32F103?"
        );
        // No identifier in the question: no negative variant exists.
        assert_eq!(
            negative_question("What is 2+2?", "STM32F407", "STM32F103", &ids),
            None
        );
    }

    /// A question usually names the fact sheet's devices as a
    /// slash-separated run ("STM32F405 / STM32F407"). Substituting one
    /// member leaves the covered device in the question, which trains a
    /// compound-question refusal instead of the scope boundary - the probe
    /// asks about one out-of-scope device alone. The whole run must go.
    #[test]
    fn a_scope_negative_replaces_the_whole_device_run() {
        let ids: Vec<String> = ["STM32F405", "STM32F407"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let question = "What is the maximum CPU clock frequency of the STM32F405 / STM32F407?";
        let negative = negative_question(question, "STM32F405", "STM32F103", &ids).unwrap();
        assert_eq!(
            negative,
            "What is the maximum CPU clock frequency of the STM32F103?"
        );
        // A non-identifier token after the slash stops the run.
        let question = "How many streams does DMA1 on the STM32F405 / its sibling have?";
        let negative = negative_question(question, "STM32F405", "STM32F103", &ids).unwrap();
        assert_eq!(
            negative,
            "How many streams does DMA1 on the STM32F103 / its sibling have?"
        );
    }
}
