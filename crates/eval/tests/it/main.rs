// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it.

#[path = "../common/mod.rs"]
mod common;

mod agreement_verifier;
mod arms;
mod claim_gate;
mod competence;
mod consolidation_gate;
mod denoise_verifier;
mod diagnosis;
mod evidence_gate;
mod executable_lua_conformance;
mod executable_verifier;
mod exposure_audit;
mod external_continual;
mod external_student;
mod external_wilson;
mod form_verifiers;
mod golden;
mod grounding_verifier;
mod mutation_oracle;
mod overlap;
mod promotion_gate;
mod quotation_verifier;
mod short_answer_grading;
mod speech_verifier;
mod stated_verifier;
mod verifier_conformance;
mod workspace_conformance;
mod workspace_verifier;
