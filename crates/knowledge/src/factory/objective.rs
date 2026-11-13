// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that compile a learning objective
// into verifiable training environments, for its clients. If your team needs
// expertise in turning a skill to be learned into an environment with a
// trustworthy reward, you can procure our services by sending an email to
// info@swedishembedded.com.

//! What a person asks to be learned, and the verifiability of each skill in
//! it.
//!
//! The first decision the factory makes is how a skill's reward could be
//! established, because that decides how hard the reward can be gamed and
//! what a verdict may claim ([`VerifierClass`]). It is made by rules over
//! the skill's words, the sources and the constraints, never by a model, so
//! the same objective always classifies the same way and the reasons are
//! written down beside the answer.

use serde::{Deserialize, Serialize};
use splinter_core::verifiability::VerifierClass;

use super::plugins::{
    DB_WORKFLOW, DOCUMENT_QUESTIONS, FIRMWARE_EMULATOR, PROGRAM_OUTPUT_LUA, PROGRAM_OUTPUT_PYTHON,
    REPOSITORY_BUGS,
};

/// How much the factory may build.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Budget {
    /// The most environments (generator configurations) to build.
    pub environments: usize,
    /// How many train instances and how many dev instances each environment
    /// is drawn for.
    pub instances_per_split: usize,
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            environments: 4,
            instances_per_split: 8,
        }
    }
}

/// An objective: the text of what is to be learned, where the material is,
/// what must hold, and how much to spend.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObjectiveSpec {
    /// What is to be learned, in plain language. Each non-empty line (a
    /// leading `-`, `*` or number is dropped) is one skill.
    pub text: String,
    /// The sources the objective rests on, as the caller names them.
    #[serde(default)]
    pub sources: Vec<String>,
    /// What must hold of any environment built (a language, a budget in
    /// time, that real ratings exist for calibration).
    #[serde(default)]
    pub constraints: Vec<String>,
    /// How much may be built.
    #[serde(default)]
    pub budget: Budget,
}

impl ObjectiveSpec {
    /// The skills the text names, one per non-empty line.
    #[must_use]
    pub fn skills(&self) -> Vec<String> {
        self.text
            .lines()
            .map(|line| {
                line.trim()
                    .trim_start_matches(['-', '*', '+'])
                    .trim_start_matches(|c: char| c.is_ascii_digit())
                    .trim_start_matches(['.', ')'])
                    .trim()
                    .to_string()
            })
            .filter(|line| !line.is_empty())
            .collect()
    }
}

/// How one skill's reward can be established, and by what.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Classification {
    /// The skill, as written.
    pub skill: String,
    /// How its reward is established.
    pub class: VerifierClass,
    /// The plugin that builds environments for it; `None` when the factory
    /// has none for this kind of skill.
    pub plugin: Option<String>,
    /// The words and rules the decision rests on, so it can be argued with.
    pub reasons: Vec<String>,
}

impl Classification {
    /// What a verdict on this skill must say beside its number.
    #[must_use]
    pub fn label(&self) -> Option<&'static str> {
        self.class.label()
    }
}

fn mentions(text: &str, stems: &[&str]) -> Option<String> {
    stems
        .iter()
        .find(|stem| text.contains(*stem))
        .map(|stem| (*stem).to_string())
}

/// Classifies `skill` given the objective's sources and constraints.
///
/// Rules, in order; the first that matches decides:
///
/// 1. A skill about people (a conversation, a tone, a user's satisfaction)
///    has no program that can grade it: a person's own evidence when the
///    constraints say real ratings or users exist, a judge model when they
///    say a calibration set does, otherwise a proxy.
/// 2. Firmware and emulated hardware: the emulator is the oracle.
/// 3. A repository with tests: the tests are the oracle.
/// 4. A stateful workflow over a database or an API: the final state is.
/// 5. Questions about a document, when a source is given: the source's own
///    text is.
/// 6. A program whose output is to be predicted or computed: the
///    interpreter is.
/// 7. Anything else is a proxy: writing quality and the like can be rubric
///    graded, but nothing here can say the rubric measures the skill.
#[must_use]
pub fn classify(skill: &str, spec: &ObjectiveSpec) -> Classification {
    let skill_lower = skill.to_lowercase();
    let constraints = spec.constraints.join("\n").to_lowercase();
    let mut reasons = Vec::new();
    let mut decide = |class: VerifierClass, plugin: Option<&str>, why: String| {
        reasons.push(why);
        Classification {
            skill: skill.to_string(),
            class,
            plugin: plugin.map(ToString::to_string),
            reasons: reasons.clone(),
        }
    };

    if let Some(word) = mentions(
        &skill_lower,
        &[
            "conversation",
            "counsel",
            "empath",
            "tone",
            "satisfaction",
            "persuad",
            "negotiat",
            "interview",
            "tutor",
            "coach",
            "rapport",
            "politeness",
            "user simulat",
        ],
    ) {
        let (class, why) = if mentions(
            &constraints,
            &["real user", "human rating", "user study", "human evidence"],
        )
        .is_some()
        {
            (
                VerifierClass::HumanEvidence,
                "the constraints say real ratings or users exist",
            )
        } else if mentions(
            &constraints,
            &[
                "calibration set",
                "calibrated",
                "human-labelled",
                "human labelled",
            ],
        )
        .is_some()
        {
            (
                VerifierClass::CalibratedJudge,
                "the constraints give a calibration set, so a judge can be checked against people",
            )
        } else {
            (VerifierClass::Proxy, "no real ratings or calibration set are given, so any reward is a stand-in for a person")
        };
        return decide(
            class,
            None,
            format!(
                "interactive skill (\"{word}\"): {why}; no automatic environment is built for it"
            ),
        );
    }
    if let Some(word) = mentions(
        &skill_lower,
        &[
            "firmware",
            "microcontroller",
            "emulator",
            "peripheral",
            "interrupt",
            "uart",
            "zephyr",
            "register map",
        ],
    ) {
        return decide(
            VerifierClass::Objective,
            Some(FIRMWARE_EMULATOR),
            format!("embedded skill (\"{word}\"): the emulator's observable state is the oracle"),
        );
    }
    if let Some(word) = mentions(
        &skill_lower,
        &[
            "repository",
            "codebase",
            "failing test",
            "bug fix",
            "fix the bug",
            "refactor",
            "pull request",
            "regression",
        ],
    ) {
        return decide(
            VerifierClass::Objective,
            Some(REPOSITORY_BUGS),
            format!("repository skill (\"{word}\"): the project's own tests are the oracle"),
        );
    }
    if let Some(word) = mentions(
        &skill_lower,
        &[
            "database",
            "sql",
            "workflow",
            "tool use",
            "tool-use",
            "orders",
            "inventory",
            "ticket",
            "booking",
            "crm",
            "api call",
        ],
    ) {
        return decide(
            VerifierClass::Objective,
            Some(DB_WORKFLOW),
            format!("stateful tool use (\"{word}\"): the final state of the system is diffed against the expected state"),
        );
    }
    if !spec.sources.is_empty() {
        if let Some(word) = mentions(
            &skill_lower,
            &[
                "document",
                "manual",
                "datasheet",
                "specification",
                "policy",
                "report",
                "answer questions",
                "question",
            ],
        ) {
            return decide(
                VerifierClass::Objective,
                Some(DOCUMENT_QUESTIONS),
                format!("document skill (\"{word}\") with {} source(s): the source's own text grounds each reference", spec.sources.len()),
            );
        }
    }
    if let Some(word) = mentions(
        &skill_lower,
        &[
            "predict the output",
            "program output",
            "what does this code print",
            "what this program prints",
            "trace",
            "interpreter",
            "execute",
            "arithmetic",
            "evaluate",
            "compute",
            "lua",
            "python",
        ],
    ) {
        let plugin = if skill_lower.contains("lua") || constraints.contains("lua") {
            PROGRAM_OUTPUT_LUA
        } else {
            PROGRAM_OUTPUT_PYTHON
        };
        return decide(
            VerifierClass::Objective,
            Some(plugin),
            format!(
                "program skill (\"{word}\"): running the program in its interpreter is the oracle"
            ),
        );
    }
    decide(
        VerifierClass::Proxy,
        None,
        "no deterministic check matches this skill; a rubric would be a proxy until calibrated against people".to_string(),
    )
}

/// The classification of every skill of `spec`.
#[must_use]
pub fn classify_all(spec: &ObjectiveSpec) -> Vec<Classification> {
    spec.skills()
        .iter()
        .map(|skill| classify(skill, spec))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(text: &str, sources: &[&str], constraints: &[&str]) -> ObjectiveSpec {
        ObjectiveSpec {
            text: text.into(),
            sources: sources.iter().map(ToString::to_string).collect(),
            constraints: constraints.iter().map(ToString::to_string).collect(),
            budget: Budget::default(),
        }
    }

    #[test]
    fn each_line_is_a_skill_and_list_markers_are_dropped() {
        let objective = spec(
            "- predict Lua output\n\n2. fix failing tests in a repository\n* answer\n",
            &[],
            &[],
        );
        assert_eq!(
            objective.skills(),
            [
                "predict Lua output",
                "fix failing tests in a repository",
                "answer"
            ]
        );
    }

    #[test]
    fn a_program_skill_is_objective_and_the_language_picks_the_plugin() {
        let objective = spec("", &[], &[]);
        let lua = classify("Predict the output of short Lua programs", &objective);
        assert_eq!(lua.class, VerifierClass::Objective);
        assert_eq!(lua.plugin.as_deref(), Some(PROGRAM_OUTPUT_LUA));
        assert_eq!(lua.label(), None);
        let py = classify("Predict the output of Python snippets", &objective);
        assert_eq!(py.plugin.as_deref(), Some(PROGRAM_OUTPUT_PYTHON));
        assert!(!lua.reasons.is_empty());
    }

    #[test]
    fn stateful_repository_firmware_and_document_skills_find_their_plugins() {
        let objective = spec("", &["manual.md"], &[]);
        for (skill, plugin) in [
            (
                "Handle customer orders in the inventory database",
                DB_WORKFLOW,
            ),
            ("Fix a failing test in a repository", REPOSITORY_BUGS),
            ("Configure a UART peripheral in firmware", FIRMWARE_EMULATOR),
            ("Answer questions about the manual", DOCUMENT_QUESTIONS),
        ] {
            let found = classify(skill, &objective);
            assert_eq!(found.plugin.as_deref(), Some(plugin), "{skill}");
        }
    }

    #[test]
    fn a_document_skill_without_a_source_is_not_assumed_to_be_grounded() {
        let found = classify("Answer questions about the manual", &spec("", &[], &[]));
        assert_eq!(found.plugin, None);
        assert_eq!(found.class, VerifierClass::Proxy);
    }

    #[test]
    fn an_interactive_skill_is_a_proxy_unless_ratings_or_calibration_exist() {
        let skill = "Hold an empathetic counselling conversation";
        let bare = classify(skill, &spec("", &[], &[]));
        assert_eq!(bare.class, VerifierClass::Proxy);
        assert_eq!(bare.plugin, None);
        assert_eq!(bare.label(), Some("proxy only"));
        let judged = classify(
            skill,
            &spec("", &[], &["a calibration set of 200 rated chats"]),
        );
        assert_eq!(judged.class, VerifierClass::CalibratedJudge);
        let rated = classify(skill, &spec("", &[], &["real users rate each session"]));
        assert_eq!(rated.class, VerifierClass::HumanEvidence);
    }

    #[test]
    fn a_skill_nothing_matches_is_a_proxy_with_its_reason() {
        let found = classify("Write persuasive essays", &spec("", &[], &[]));
        assert_eq!(found.class, VerifierClass::Proxy);
        assert!(
            found.reasons[0].contains("no deterministic check"),
            "{:?}",
            found.reasons
        );
    }
}
