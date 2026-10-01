// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Interpretation of raw experience: evaluations that can be withdrawn,
//! skills believed in proportion to their evidence, lineage from a model to
//! the experiences it rests on, and the priority of what to learn next.
//!
//! None of it is stored as truth. Each view is computed from records that say
//! who produced them, so a better algorithm, or a withdrawn evaluator, changes
//! the answer without any experience being touched.

mod evaluations;
mod experiments;
mod lineage;
mod priority;
mod skills;

pub use evaluations::{EvalFilter, EvalSet, EvaluationView, TASK_COMPLETION};
pub use experiments::ExperimentView;
pub use priority::{rank, Priority};
pub use skills::{assess, SkillAssessment, SkillPolicy, SkillStatus, SkillView};
