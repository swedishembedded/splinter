// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gated self-improvement loops where a model
// update is adopted only on held-out evidence. If your team needs expertise
// in evaluation-gated fine-tuning, you can procure our services by sending
// an email to info@swedishembedded.com.

//! Spec: a backend trains a file only in the formats it reads, and refuses
//! an objective it has no trainer for by name.

use splinter_data::Objective;
use splinter_model::{PolicyError, TrainingCapabilities};

#[test]
fn only_the_objectives_brain_trains_from_a_file_are_trainable() {
    let brain = TrainingCapabilities::BRAIN;
    for objective in [
        Objective::Sft,
        Objective::Classification,
        Objective::Preference,
        Objective::Feedback,
        Objective::Cpt,
    ] {
        assert!(brain.require(objective).is_ok(), "{objective:?}");
    }
    for objective in [Objective::Contrastive, Objective::Reward] {
        assert!(
            matches!(
                brain.require(objective),
                Err(PolicyError::ObjectiveNotTrainable { objective: o }) if o == objective
            ),
            "{objective:?}"
        );
        assert!(!brain.supports(objective.line_format()), "{objective:?}");
    }
}
