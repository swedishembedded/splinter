// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The names tasks and privileged items travel under, shared by whatever
//! generates them, whatever grades them and whatever projects them.

/// The task kind denoise tasks carry, shared by the generator, the view and
/// the verifier.
pub const DENOISE: &str = "denoise";

/// The privileged kind an authored executable check travels as.
pub const EXECUTABLE_CHECK: &str = "executable-check";

/// The privileged kind of a check that established a task's reference by
/// running code the instruction shows (what does this program print): the
/// code alone, run with no solution before it, must produce the reference.
/// It proves the reference, it does not grade an answer - an answer is
/// text, not code to run - so an executable verifier never reads it as a
/// check of the solver.
pub const OUTPUT_CHECK: &str = "output-check";

/// The privileged kind a generated test travels as.
pub const GENERATED_TEST: &str = "generated-test";
