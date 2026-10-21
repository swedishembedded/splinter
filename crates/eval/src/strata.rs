// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements difficulty-stratified reporting of
// code-editing agent evaluations, for its clients. If your team needs
// expertise in showing where a fine-tuned agent gains or loses by task
// difficulty, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Difficulty strata: a report-only grouping of paired tasks by a structural
//! proxy read from the task record, shown next to the family sign test.
//!
//! The proxy has three parts, each measured on the task as generated:
//!
//! * **checkers**: the tests its fail-to-pass commands must be seen to run
//!   (the commands themselves when none are named);
//! * **footprint**: the distinct files its reference patch changes (none when
//!   the task carries no reference);
//! * **instruction length**: characters of the request the solver sees.
//!
//! A task's stratum is how many of the three exceed their cutoff, so it is
//! 0..=3. A value equal to its cutoff is not above it: ties go to the lower
//! stratum. The cutoffs are parameters, fixed by a registration before the
//! data is read; nothing here decides, gates or thresholds anything.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::verifiers::workspace::{Phase, WorkspaceCheck};
use crate::verifiers::VerifyError;
use splinter_core::experience::{PrivilegedKind, Task};

/// How big a task is on each part of the proxy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskSize {
    /// Tests the fail-to-pass commands must run.
    pub checkers: usize,
    /// Distinct files the reference patch changes.
    pub footprint: usize,
    /// Characters in the instruction.
    pub instruction_chars: usize,
}

impl TaskSize {
    /// The size of `task`; `None` when it carries no workspace check.
    pub fn of_task(task: &Task) -> Result<Option<Self>, VerifyError> {
        let Some((_, check)) = WorkspaceCheck::of_task(task)? else {
            return Ok(None);
        };
        let named: usize = check
            .expectations
            .iter()
            .filter(|e| e.phase == Phase::FailToPass)
            .map(|e| e.tests.len())
            .sum();
        let files: BTreeSet<&str> = task
            .privileged
            .iter()
            .filter(|p| p.kind == PrivilegedKind::Reference)
            .flat_map(|p| p.content.lines())
            .filter_map(|line| line.strip_prefix("diff --git a/")?.split(" b/").next())
            .collect();
        Ok(Some(Self {
            checkers: if named > 0 {
                named
            } else {
                check.fail_to_pass.len()
            },
            footprint: files.len(),
            instruction_chars: task.instruction.chars().count(),
        }))
    }
}

/// The largest value of each part that still counts as not above.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StratumCutoffs {
    /// Checkers above this count are above.
    pub checkers: usize,
    /// A footprint above this many files is above.
    pub footprint: usize,
    /// An instruction longer than this many characters is above.
    pub instruction_chars: usize,
}

impl StratumCutoffs {
    /// The stratum of `size`: how many parts are strictly above their cutoff.
    #[must_use]
    pub fn stratum(&self, size: &TaskSize) -> usize {
        usize::from(size.checkers > self.checkers)
            + usize::from(size.footprint > self.footprint)
            + usize::from(size.instruction_chars > self.instruction_chars)
    }
}

/// One stratum's paired tasks, counted per task.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StratumLine {
    /// The stratum: how many parts of the proxy are above their cutoff.
    pub stratum: usize,
    /// Paired tasks in it.
    pub paired: usize,
    /// Of those, the ones the candidate got right.
    pub candidate_right: usize,
    /// Of those, the ones the baseline got right.
    pub baseline_right: usize,
    /// Paired tasks only the candidate got right.
    pub wins: usize,
    /// Paired tasks only the baseline got right.
    pub losses: usize,
    /// Paired tasks both got right or both got wrong.
    pub ties: usize,
}

/// The strata of a comparison.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Strata {
    /// The cutoffs the strata were computed with.
    pub cutoffs: StratumCutoffs,
    /// Each stratum with a paired task, lowest first.
    pub lines: Vec<StratumLine>,
    /// Paired tasks whose record has no workspace check, so no stratum.
    pub unstratified: usize,
}

impl Strata {
    /// The strata of `tasks`: each one's size, when known, and its paired
    /// verdicts (candidate, baseline), when it has both.
    #[must_use]
    pub fn of(
        cutoffs: StratumCutoffs,
        tasks: impl IntoIterator<Item = (Option<TaskSize>, Option<(bool, bool)>)>,
    ) -> Self {
        let mut lines: BTreeMap<usize, StratumLine> = BTreeMap::new();
        let mut unstratified = 0;
        for (size, paired) in tasks {
            let Some((candidate, baseline)) = paired else {
                continue;
            };
            let Some(size) = size else {
                unstratified += 1;
                continue;
            };
            let stratum = cutoffs.stratum(&size);
            let line = lines.entry(stratum).or_insert(StratumLine {
                stratum,
                paired: 0,
                candidate_right: 0,
                baseline_right: 0,
                wins: 0,
                losses: 0,
                ties: 0,
            });
            line.paired += 1;
            line.candidate_right += usize::from(candidate);
            line.baseline_right += usize::from(baseline);
            match (candidate, baseline) {
                (true, false) => line.wins += 1,
                (false, true) => line.losses += 1,
                _ => line.ties += 1,
            }
        }
        Self {
            cutoffs,
            lines: lines.into_values().collect(),
            unstratified,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CUTOFFS: StratumCutoffs = StratumCutoffs {
        checkers: 2,
        footprint: 3,
        instruction_chars: 500,
    };

    fn size(checkers: usize, footprint: usize, instruction_chars: usize) -> TaskSize {
        TaskSize {
            checkers,
            footprint,
            instruction_chars,
        }
    }

    #[test]
    fn a_value_equal_to_its_cutoff_belongs_to_the_lower_stratum() {
        assert_eq!(CUTOFFS.stratum(&size(2, 3, 500)), 0);
        assert_eq!(CUTOFFS.stratum(&size(3, 3, 500)), 1);
        assert_eq!(CUTOFFS.stratum(&size(3, 4, 500)), 2);
        assert_eq!(CUTOFFS.stratum(&size(3, 4, 501)), 3);
    }

    #[test]
    fn tasks_are_counted_in_their_stratum_and_unpaired_ones_are_left_out() {
        let strata = Strata::of(
            CUTOFFS,
            [
                (Some(size(1, 1, 10)), Some((true, false))),
                (Some(size(1, 1, 10)), Some((true, true))),
                (Some(size(9, 9, 900)), Some((false, true))),
                (Some(size(9, 9, 900)), None),
                (None, Some((true, true))),
            ],
        );
        assert_eq!(strata.unstratified, 1);
        let lines: Vec<_> = strata
            .lines
            .iter()
            .map(|l| (l.stratum, l.paired, l.wins, l.losses, l.ties))
            .collect();
        assert_eq!(lines, [(0, 2, 1, 0, 1), (3, 1, 0, 1, 0)]);
        assert_eq!(strata.lines[0].candidate_right, 2);
        assert_eq!(strata.lines[0].baseline_right, 1);
    }
}
