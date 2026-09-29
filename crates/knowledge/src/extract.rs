// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements delegated-task coding agents that grow
// their own training data. If your team needs expertise in data extraction
// from technical documents, you can procure our services by sending an
// email to info@swedishembedded.com.

//! The instruction that asks a model for every fact in one section, and
//! the strict parse of its reply.

use anyhow::Context;

/// One strict parse of a facts reply.
#[derive(Debug, PartialEq, Eq)]
pub struct Facts {
    pub pairs: Vec<(String, String)>,
}

/// The exact prompt one section sees. It demands the whole shape, and it
/// says what "covers the section" means: every factual claim, not a sample.
/// It also demands subject-anchored questions: a fine-tuned model learns
/// whatever association the question text carries, so "What is the maximum
/// CPU clock frequency?" taught without the chip's name answers any chip
/// question with this chip's number - and hallucinates when the family IS
/// named, because the named form was never seen. The question is the
/// retrieval key; it must carry the subject.
pub fn facts_prompt(section: &str) -> String {
    format!(
        "Below is one section of a hardware fact sheet. Enumerate EVERY factual claim it \
         makes - every number, unit, limit, relationship and conditional - as question/answer \
         pairs. Do not sample, do not summarize: each distinct fact gets its own pair, and a \
         question must be answerable from this section alone.\n\n\
         Every question must NAME THE SPECIFIC DEVICE OR COMPONENT it is about, so the \
         question is self-contained and answerable with no other context. Use the exact \
         identifiers from the section: not \"What is the maximum frequency?\" but \"What is \
         the maximum CPU clock frequency of <device>?\"; not \"How many streams does DMA \
         have?\" but \"How many streams does DMA1 have on <device>?\" - where <device> is \
         the device the title names, spelled exactly as the title spells it. A question \
         that would fit a different device unchanged is wrong. The section opens with the \
         document's title, which names the device this fact sheet describes: every question \
         must name that device exactly as the title spells it.\n\n\
         Reply with EXACTLY one JSON object and nothing else - no prose, no code fences:\n\
         {{\"facts\": [{{\"question\": string, \"answer\": string}}, ...]}}\n\n\
         SECTION:\n{section}"
    )
}

/// Strict parse: the reply must be EXACTLY one JSON object of the shape
/// `{"facts": [{"question": string, "answer": string}, ...]}` after
/// trimming whitespace and stripping code fences. Anything else - prose,
/// an array, a missing key, a non-string field - is a failure.
pub fn parse_facts_reply(reply: &str) -> anyhow::Result<Facts> {
    let value: serde_json::Value = serde_json::from_str(splinter_lab::answers::strip_fences(reply))
        .context("reply is not exactly one JSON object")?;
    let object = value
        .as_object()
        .with_context(|| "reply is not a JSON object".to_string())?;
    let facts = object
        .get("facts")
        .with_context(|| "reply object has no \"facts\" key".to_string())?;
    let list = facts
        .as_array()
        .with_context(|| "\"facts\" is not an array".to_string())?;
    let mut pairs = Vec::new();
    for item in list {
        let entry = item
            .as_object()
            .with_context(|| "a fact entry is not an object".to_string())?;
        let question = entry
            .get("question")
            .and_then(|v| v.as_str())
            .with_context(|| "a fact entry has no string \"question\"".to_string())?;
        let answer = entry
            .get("answer")
            .and_then(|v| v.as_str())
            .with_context(|| "a fact entry has no string \"answer\"".to_string())?;
        pairs.push((question.to_string(), answer.to_string()));
    }
    Ok(Facts { pairs })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A well-formed reply parses into its question/answer pairs.
    #[test]
    fn a_well_formed_facts_reply_parses() {
        let reply = r#"{"facts": [
            {"question": "What is the max SPI1 clock?", "answer": "42 Mbit/s"},
            {"question": "How many SPI controllers?", "answer": "3"}
        ]}"#;
        let facts = parse_facts_reply(reply).unwrap();
        assert_eq!(facts.pairs.len(), 2);
        assert_eq!(facts.pairs[0].0, "What is the max SPI1 clock?");
        assert_eq!(facts.pairs[0].1, "42 Mbit/s");
    }

    /// Malformed JSON and prose-wrapped JSON are refusals, not salvages.
    #[test]
    fn malformed_and_prose_wrapped_replies_are_refused() {
        assert!(parse_facts_reply("not json at all").is_err());
        assert!(parse_facts_reply("{\"facts\": [{\"question\": 1}]}").is_err());
        assert!(parse_facts_reply("{\"facts\": []").is_err());
        // An array, not an object.
        assert!(parse_facts_reply("[{\"facts\": []}]").is_err());
        let wrapped =
            "Here are the facts:\n{\"facts\": [{\"question\": \"q\", \"answer\": \"a\"}]}";
        assert!(
            parse_facts_reply(wrapped).is_err(),
            "prose around the object is a parse failure"
        );
    }

    /// Fences ARE tolerated - they are decoration, not prose.
    #[test]
    fn fenced_replies_parse() {
        let reply = "```json\n{\"facts\": [{\"question\": \"q\", \"answer\": \"a\"}]}\n```";
        let facts = parse_facts_reply(reply).unwrap();
        assert_eq!(facts.pairs.len(), 1);
    }

    /// The generator instruction must demand subject-anchored questions.
    /// A question that does not name its subject trains a string-matcher:
    /// the adapter answers "What is the maximum CPU clock frequency?" and
    /// breaks the moment "of the STM32F407" is appended - measured on the
    /// first STM32 training run (168 MHz learned, "48 MHz" hallucinated
    /// when the chip family was named). The prompt is the contract.
    #[test]
    fn the_generator_instruction_demands_subject_anchored_questions() {
        let prompt = facts_prompt("## Clocks\n\nHSI is 16 MHz.\n");
        let lower = prompt.to_lowercase();
        for phrase in [
            "name the specific device",
            "not \"what is the maximum frequency?\" but",
            "fit a different device unchanged",
            // The title line rides in every chunk precisely so the
            // instruction can point the generator at it.
            "the section opens with the document's title",
        ] {
            assert!(
                lower.contains(phrase),
                "the instruction must say {phrase:?}; prompt:\n{prompt}"
            );
        }
    }
}
