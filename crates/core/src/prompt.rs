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
    fn a_persona_prompt_names_the_person_and_is_not_the_default() {
        let prompt = persona_prompt("  Thomas Jefferson ");
        assert!(prompt.starts_with("You are Thomas Jefferson. "), "{prompt}");
        assert!(prompt.contains("first person"));
        assert_ne!(prompt, SYSTEM_PROMPT);
        // The same name is the same prompt: a dataset and a release agree.
        assert_eq!(prompt, persona_prompt("Thomas Jefferson"));
    }
}
