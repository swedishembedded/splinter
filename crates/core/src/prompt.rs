// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fine-tuning pipelines whose training data
// match what the model sees when it answers, for its clients. If your team
// needs expertise in training-data curation for agents, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The one system prompt the policy answers under and is trained under.
//!
//! What a model is trained on must be what it sees when it answers. Every
//! model run on a task - a solve, a teacher's solve, an `ask`, a probe of
//! the release gate, the judge's and the critic's runs - is sent
//! [`SYSTEM_PROMPT`] as its one system turn, and every chat record a view
//! projects starts with that same turn. It is short and names nothing about
//! where it runs (no working directory, no date, no tool list), so the
//! records and the runs match token for token in their system turn.

/// The system turn of every model run on a task and of every chat record.
pub const SYSTEM_PROMPT: &str = "You are a helpful assistant. Answer the user's request directly \
                                 and concisely, using a tool when one is offered and it helps.";

/// The identity of [`CODING_AGENT_PROMPT`]: changing its text means a new id,
/// so a record that names the id and the digest shows the change.
pub const CODING_AGENT_PROMPT_ID: &str = "coding-agent-1";

/// The system turn of a solver working on a repository through tools: the
/// order to work in (look, reproduce, change, verify), no repeated call and
/// when to stop. It is the same for every solver, base or adapter, and a
/// record trained under another prompt is rendered again under this one.
pub const CODING_AGENT_PROMPT: &str = "You are a software engineer working in a git repository \
through tools. You are given a report of behaviour that is wrong. Work in this order. First look: \
list and search the repository and read the code and the tests that concern the report before you \
change anything. Then reproduce the symptom from the report with a short command or test and read \
what it prints. Then change the code (edit the source, not the tests the report relies on). Then \
run the repository's tests and the reproduction again and read the results; if they fail, read the \
failure and fix the cause. Never repeat a call whose result you have already seen: if you are \
about to run the same call again, do something different (read other code, change the call, or \
change the code). Stop when the behaviour is fixed and the tests pass, and say in one or two \
sentences what you changed.";

/// The system turn of a policy trained to be `persona`: who it is, how it
/// speaks, and that it answers plainly. The persona is the name the plan
/// found in the goal; a model trained under this prompt is asked under it, so
/// the person is reachable from the prompt and does not wait on a word in the
/// question.
#[must_use]
pub fn persona_prompt(persona: &str) -> String {
    let name = persona.trim();
    format!(
        "You are {name}. Answer as {name} would: in the first person, in {name}'s own voice, \
         from {name}'s own experience and opinions. Answer the user's request directly and \
         concisely."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_coding_agent_prompt_is_the_registered_text_and_not_the_chat_prompt() {
        assert_ne!(CODING_AGENT_PROMPT, SYSTEM_PROMPT);
        for part in [
            "You are a software engineer working in a git repository through tools.",
            "First look:",
            "Then reproduce the symptom",
            "Never repeat a call whose result you have already seen",
            "say in one or two sentences what you changed.",
        ] {
            assert!(CODING_AGENT_PROMPT.contains(part), "{part}");
        }
        assert!(
            !CODING_AGENT_PROMPT.contains("  "),
            "one text, single-spaced"
        );
        assert!(CODING_AGENT_PROMPT.is_ascii());
    }

    #[test]
    fn a_persona_prompt_names_the_person_and_is_not_the_default() {
        let prompt = persona_prompt("  Thomas Jefferson ");
        assert!(prompt.starts_with("You are Thomas Jefferson. "), "{prompt}");
        assert!(prompt.contains("first person"));
        assert_ne!(prompt, SYSTEM_PROMPT);
        // The same name is the same prompt: a dataset and a release agree.
        assert_eq!(prompt, persona_prompt("Thomas Jefferson"));
    }
}
