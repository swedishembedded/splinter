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

/// The key of a trajectory's agent `extra` that holds the solve profile: what a
/// solver was held to besides the task, among it the digest of its system
/// prompt.
pub const SOLVE_PROFILE_KEY: &str = "solve_profile";

/// The key of a trajectory's agent `extra` that names the directory the
/// episode worked in.
pub const WORKSPACE_TREE_KEY: &str = "workspace_tree";

/// The system prompt a trajectory's solve profile says the run was held to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RecordedPrompt {
    /// The trajectory records no solve profile (it predates it): the run was
    /// under the chat prompt, [`SYSTEM_PROMPT`].
    Unrecorded,
    /// A prompt this build holds the text of.
    Known(&'static str),
    /// A prompt that only its digest is known by; the record cannot be shown
    /// the prompt the run was under.
    Unknown(String),
}

/// The prompt `agent_extra`, the `extra` of a trajectory's agent, records.
#[must_use]
pub fn recorded_prompt(agent_extra: Option<&serde_json::Value>) -> RecordedPrompt {
    let Some(recorded) = agent_extra
        .and_then(|extra| extra.get(SOLVE_PROFILE_KEY))
        .and_then(|profile| profile.get("system_prompt"))
    else {
        return RecordedPrompt::Unrecorded;
    };
    let digest = recorded
        .get("digest")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    [SYSTEM_PROMPT, CODING_AGENT_PROMPT]
        .into_iter()
        .find(|known| crate::digest::Digest::of(known.as_bytes()).as_str() == digest)
        .map_or_else(
            || RecordedPrompt::Unknown(digest.to_string()),
            RecordedPrompt::Known,
        )
}

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
    fn the_prompt_a_solve_profile_records_is_known_by_its_digest() {
        let profile = |prompt: &str| {
            serde_json::json!({ SOLVE_PROFILE_KEY: { "system_prompt": {
                "id": null, "digest": crate::digest::Digest::of(prompt.as_bytes()),
            } } })
        };
        assert_eq!(recorded_prompt(None), RecordedPrompt::Unrecorded);
        assert_eq!(
            recorded_prompt(Some(&serde_json::json!({}))),
            RecordedPrompt::Unrecorded
        );
        assert_eq!(
            recorded_prompt(Some(&profile(CODING_AGENT_PROMPT))),
            RecordedPrompt::Known(CODING_AGENT_PROMPT)
        );
        assert_eq!(
            recorded_prompt(Some(&profile(SYSTEM_PROMPT))),
            RecordedPrompt::Known(SYSTEM_PROMPT)
        );
        assert!(matches!(
            recorded_prompt(Some(&profile("You are a pirate."))),
            RecordedPrompt::Unknown(_)
        ));
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
