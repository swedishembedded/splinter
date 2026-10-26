// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The names tasks and privileged items travel under, shared by whatever
//! generates them, whatever grades them and whatever projects them.

/// The task kind denoise tasks carry, shared by the generator, the view and
/// the verifier.
pub const DENOISE: &str = "denoise";

/// The task kind of a general request with no reference answer: what a
/// base model is asked so that its own answer can be kept and replayed as
/// it is, graded by nothing.
pub const GENERAL: &str = "general";

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

/// The task kind of a change to a source tree: the instruction asks for a
/// behaviour, the environment is the tree, and the answer is what the
/// solver changed in it.
pub const WORKSPACE_CHANGE: &str = "workspace-change";

/// The privileged kind of a check run in a solver's changed tree to decide
/// a [`WORKSPACE_CHANGE`] task: the commands that must pass after the
/// change, and the files the solver may not touch. It grades the tree, so
/// the solver never sees it.
pub const WORKSPACE_CHECK: &str = "workspace-check";

/// The privileged kind of the process that solves a task: a sven harness
/// document that the task's author adapted to the task. It is the task's own
/// way of being solved, not part of what the solver is shown, so a view that
/// strips privileged items drops it.
pub const SOLVE_PROCESS: &str = "solve-process";
