// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements executable evaluation of code changes
// against a project's own test suite, for its clients. If your team needs
// expertise in test-driven benchmarks for code-editing agents, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The proof a combined task is admitted on.
//!
//! Run the way a solver's answer will be graded, in this order, each step
//! refusing with a [`CombineReason`]:
//!
//! 1. every fail-to-pass command fails on the tree the solver is given, with
//!    the hidden files laid over it, one command at a time (the grader stops
//!    at the first failure, so a check run whole would only show the first);
//! 2. the reference change, the union of the members', makes the check pass,
//!    and does so again on a second run;
//! 3. no member's reference change alone makes the check pass: the faults
//!    are independent and every one of them has to be put right.

use splinter_core::clock::Clock;
use splinter_core::experience::Task;
use splinter_eval::verifiers::workspace::{grade, Phase, WorkspaceCheck};
use splinter_sandbox::{Patch, WorkspaceEnvironment};

use super::single::Single;
use super::{CombineReason, Stop};
use crate::tasks::workspace::WorkspaceGenerator;

impl WorkspaceGenerator {
    /// Shows that `task`, whose check is `check` and whose reference change
    /// is `reference`, is a task of `members`' faults and nothing less.
    pub(super) fn prove(
        &self,
        env: &WorkspaceEnvironment,
        task: &Task,
        check: &WorkspaceCheck,
        reference: &str,
        members: &[&Single],
        clock: &dyn Clock,
    ) -> Result<(), Stop> {
        let unchanged = Patch {
            diff: Vec::new(),
            files: Vec::new(),
        };
        for index in 0..check.fail_to_pass.len() {
            let graded = grade(
                env,
                &self.store,
                &self.scratch.join("grade"),
                &only_target(check, index),
                &unchanged,
            )
            .map_err(crate::tasks::workspace::WorkspaceError::from)?;
            if graded.passed() {
                return Err(Stop::Refuse(CombineReason::NotFailing));
            }
        }
        if !self.passes(env, task, reference, clock)? {
            return Err(Stop::Refuse(CombineReason::NotRestorable));
        }
        if !self.passes(env, task, reference, clock)? {
            return Err(Stop::Refuse(CombineReason::Flaky));
        }
        for member in members {
            if self.passes(env, task, &member.reference, clock)? {
                return Err(Stop::Refuse(CombineReason::SingleSuffices));
            }
        }
        Ok(())
    }
}

/// `check` with only the fail-to-pass command at `index`.
fn only_target(check: &WorkspaceCheck, index: usize) -> WorkspaceCheck {
    WorkspaceCheck {
        fail_to_pass: vec![check.fail_to_pass[index].clone()],
        pass_to_pass: Vec::new(),
        expectations: check
            .expectations
            .iter()
            .filter(|e| e.phase == Phase::FailToPass && e.index == index)
            .map(|e| splinter_eval::verifiers::test_report::TestExpectation {
                index: 0,
                ..e.clone()
            })
            .collect(),
        ..check.clone()
    }
}
