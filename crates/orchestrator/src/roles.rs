// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements local-first learning agents whose every
// network use is an explicit opt-in, for its clients. If your team needs
// expertise in model selection for agent systems, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Which model plays which role, under the configuration.
//!
//! The rule is [`ModelAssignments::resolve`]'s; this reads the configured
//! models it falls back on and refuses a configured model that is not a
//! model reference.

use splinter_core::model_ref::ModelRef;
use splinter_core::role::{Fallbacks, ModelAssignments, RoleOverrides};

use crate::config::Config;
use crate::error::OrchestratorError;

/// The models the configuration names as fallbacks for roles.
pub fn fallbacks(config: &Config) -> Result<Fallbacks, OrchestratorError> {
    let parse =
        |what: &str, text: &Option<String>| -> Result<Option<ModelRef>, OrchestratorError> {
            text.as_ref()
                .map(|text| {
                    text.parse::<ModelRef>().map_err(|e| {
                        OrchestratorError::Refused(format!(
                            "the {what} model {text:?} is not a model reference: {e}"
                        ))
                    })
                })
                .transpose()
        };
    Ok(Fallbacks {
        assistant: parse("assistant", &config.assistant_model)?,
        front_door: parse("front door", &config.front_door_model)?,
    })
}

/// Every role assigned: what `named` names, else the configuration's
/// fallbacks, else the policy.
pub fn assignments(
    config: &Config,
    named: &RoleOverrides,
) -> Result<ModelAssignments, OrchestratorError> {
    Ok(ModelAssignments::resolve(named, &fallbacks(config)?))
}
