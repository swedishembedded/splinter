// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems whose every record is
// content-addressed and traceable to its source, for its clients. If your
// team needs expertise in training-data provenance or reproducible
// pipelines, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The roles models play in a learning run, and who plays each.
//!
//! A pipeline asks for a role - the model that teaches, the model that
//! writes tasks - and never for a particular model. Configuration and the
//! command decide which model fulfils it, by one rule:
//!
//! 1. a model the command names for the role;
//! 2. else the configured assistant, for the roles an assistant can play
//!    (teacher, generator, planner, judge);
//! 3. else, for the planner, the generator; for the router, the model
//!    configured for the front door;
//! 4. else the policy itself.
//!
//! The policy is the model being trained and is never reassigned.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::model_ref::ModelRef;

/// A part a model plays in a run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// The model being trained: it solves, critiques and retries.
    Policy,
    /// The model that teaches what the policy never solves.
    Teacher,
    /// The model that writes the tasks.
    Generator,
    /// The model that surveys the sources and plans the run.
    Planner,
    /// The model that judges the exam.
    Judge,
    /// The model that says what is wrong with a failed attempt.
    Critic,
    /// The model that reads a sentence and routes it to a command.
    Router,
}

impl Role {
    /// Every role, in the order a report lists them.
    pub const ALL: [Role; 7] = [
        Role::Policy,
        Role::Teacher,
        Role::Generator,
        Role::Planner,
        Role::Judge,
        Role::Critic,
        Role::Router,
    ];

    /// The role's name, as a report and a command line write it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Role::Policy => "policy",
            Role::Teacher => "teacher",
            Role::Generator => "generator",
            Role::Planner => "planner",
            Role::Judge => "judge",
            Role::Critic => "critic",
            Role::Router => "router",
        }
    }
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// The models a command names for roles.
pub type RoleOverrides = BTreeMap<Role, ModelRef>;

/// The configured models a role falls back on.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Fallbacks {
    /// The assistant model: plays the roles that help the policy learn.
    pub assistant: Option<ModelRef>,
    /// The model the configuration names for reading sentences.
    pub front_door: Option<ModelRef>,
}

/// Which model plays each role.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct ModelAssignments(BTreeMap<Role, ModelRef>);

impl ModelAssignments {
    /// Every role assigned, from what `named` names and what `fallbacks`
    /// configure.
    #[must_use]
    pub fn resolve(named: &RoleOverrides, fallbacks: &Fallbacks) -> Self {
        let policy = ModelRef::policy_default();
        let pick = |role: Role, fallback: Option<&ModelRef>, last: &ModelRef| {
            named
                .get(&role)
                .or(fallback)
                .cloned()
                .unwrap_or_else(|| last.clone())
        };
        let assistant = fallbacks.assistant.as_ref();
        let teacher = pick(Role::Teacher, assistant, &policy);
        let generator = pick(Role::Generator, assistant, &policy);
        let planner = pick(Role::Planner, assistant, &generator);
        let judge = pick(Role::Judge, assistant, &policy);
        let critic = pick(Role::Critic, None, &policy);
        let router = pick(Role::Router, fallbacks.front_door.as_ref(), &policy);
        Self(BTreeMap::from([
            (Role::Policy, policy),
            (Role::Teacher, teacher),
            (Role::Generator, generator),
            (Role::Planner, planner),
            (Role::Judge, judge),
            (Role::Critic, critic),
            (Role::Router, router),
        ]))
    }

    /// The model that plays `role`.
    #[must_use]
    pub fn get(&self, role: Role) -> &ModelRef {
        match self.0.get(&role) {
            Some(model) => model,
            None => unreachable!("every role is assigned"),
        }
    }

    /// Every role and the model that plays it, in the order of [`Role::ALL`].
    pub fn iter(&self) -> impl Iterator<Item = (Role, &ModelRef)> {
        self.0.iter().map(|(role, model)| (*role, model))
    }
}
