// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements data harmonisation pipelines that pool
// heterogeneous research records under auditable rules, for its clients. If
// your team needs expertise in model-assisted data integration, you can
// procure our services by sending an email to info@swedishembedded.com.

//! A model asked, through sven, how a documented variable maps onto a shared
//! concept.
//!
//! The call is typed: the reply must be a [`MappingProposal`] naming one of
//! the concepts offered (or `none`), and one that is not is sent back for
//! correction. Whether the mapping is believed is not the model's to say:
//! [`splinter_knowledge::harmonize::admit`] decides that from the codebook
//! and the data.

use std::time::Duration;

use async_trait::async_trait;
use serde::Serialize;
use splinter_knowledge::codebook::VariableDoc;
use splinter_knowledge::harmonize::{ConceptSpec, MappingProposal, MappingProposer};

use crate::solve::Model;
use crate::typed::TypedCall;

const TASK: &str = "Map the documented variable onto one of the concepts. Give the \
factor that converts a recorded value into the concept's unit, the range of real \
measurements in that unit, every recorded code the codebook marks as refused, \
don't know, missing or not applicable, and a quotation copied exactly from the \
codebook entry that shows what the variable measures and in which unit. Answer \
`none` as the concept if the variable measures none of them.";

const ROLE: &str = "You are a careful data manager harmonising survey and cohort \
variables. You rely only on the codebook text you are given.";

/// Corrections allowed for a reply that is not a valid proposal.
const REPAIRS: u32 = 2;

#[derive(Serialize)]
struct Input<'a> {
    variable: &'a VariableDoc,
    concepts: &'a [ConceptSpec],
}

/// A [`MappingProposer`] that asks `model` through a typed sven call.
#[derive(Clone)]
pub struct SvenMapper {
    model: Model,
    deadline: Duration,
}

impl SvenMapper {
    /// A mapper on `model`, each proposal bounded by `deadline`.
    #[must_use]
    pub fn new(model: Model, deadline: Duration) -> Self {
        Self { model, deadline }
    }
}

#[async_trait]
impl MappingProposer for SvenMapper {
    fn identity(&self) -> &str {
        &self.model.identity
    }

    async fn propose(
        &self,
        variable: &VariableDoc,
        concepts: &[ConceptSpec],
    ) -> Result<MappingProposal, String> {
        let offered: Vec<String> = concepts.iter().map(|c| c.name.clone()).collect();
        let call = TypedCall::<MappingProposal>::new("map_variable", TASK, ROLE, self.deadline)
            .repairs(REPAIRS)
            .postcondition(move |p: &MappingProposal| {
                if p.concept == "none" || offered.contains(&p.concept) {
                    Ok(())
                } else {
                    Err(format!(
                        "concept must be one of {offered:?} or none, not {}",
                        p.concept
                    ))
                }
            });
        call.run(&self.model, &Input { variable, concepts })
            .await
            .map_err(|e| e.to_string())
    }
}
