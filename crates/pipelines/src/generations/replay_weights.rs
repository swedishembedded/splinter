// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements continual learning that spends replay on
// the capabilities a model is still gaining or losing, for its clients. If
// your team needs expertise in measuring retention across successive
// fine-tunes of an agent, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Replay weights from learning progress.
//!
//! Every generation measures its champion on the whole protected panel, so
//! each earlier generation's exam has a pass rate under every release that
//! has been the champion. That series is the competence history
//! [`crate::curriculum::progress`] reads: its change between releases,
//! smoothed, is how fast the policy is still gaining or losing that
//! generation's capability. [`column_histories`] builds the series from the
//! measured rows, and [`replay_weights`] turns them into the weight of every
//! family an earlier release trained on.
//!
//! The unit is the exam column, not the family the exam holds: an exam's
//! families are reserved apart from the families trained on, so the training
//! records of a generation inherit the weight of that generation's exam, the
//! way the failure-targeted replay gives them the champion's failure rate on
//! it. A panel measurement records no device time, so the cost divisor of the
//! signal is neutral here.

use std::collections::BTreeMap;

use splinter_core::digest::Digest;

use super::panel::PanelRow;
use crate::curriculum::progress::{weights, History, Params, Round};

/// Each panel column's pass rate under every release that was measured on
/// it, oldest first. `rows` are the champion's measurements in the order they
/// were made; `column_of` maps a protected task to its column in the current
/// panel, and a cell of a task it does not name is left out. Two measurements
/// of one release are one round, the newer.
#[must_use]
pub fn column_histories(
    rows: &[&PanelRow],
    column_of: &BTreeMap<Digest, usize>,
) -> BTreeMap<usize, History> {
    let mut histories: BTreeMap<usize, History> = BTreeMap::new();
    let mut previous: Option<&PanelRow> = None;
    for row in rows {
        // The same release measured again replaces its round.
        let remeasured =
            previous.is_some_and(|p| row.release.is_some() && p.release == row.release);
        let mut tallies: BTreeMap<usize, Round> = BTreeMap::new();
        for cell in &row.cells {
            let (Some(&column), Some(passed)) = (column_of.get(&cell.task), cell.verdict) else {
                continue;
            };
            let round = tallies.entry(column).or_default();
            round.attempts += 1;
            round.passes += u32::from(passed);
        }
        for (column, round) in tallies {
            let history = histories.entry(column).or_default();
            if remeasured {
                history.rounds.pop();
            }
            history.rounds.push(round);
        }
        previous = Some(row);
    }
    histories
}

/// The weight of each column: its share of the replay, from its history.
/// Columns sum to 1 and none is below the exploration floor.
#[must_use]
pub fn replay_weights(
    histories: &BTreeMap<usize, History>,
    params: &Params,
) -> BTreeMap<usize, f64> {
    let named: BTreeMap<String, History> = histories
        .iter()
        .map(|(column, history)| (column.to_string(), history.clone()))
        .collect();
    let scored = weights(&named, params);
    histories
        .keys()
        .map(|column| (*column, scored[&column.to_string()].weight))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generations::panel::Cell;
    use splinter_core::release::ReleaseId;

    fn release(n: u8) -> Option<ReleaseId> {
        Some(ReleaseId(Digest::of(&[n])))
    }

    /// A row measured on the tasks `t0..`, `passes[c]` of 10 tasks of each
    /// column `c` solved.
    fn row(release: Option<ReleaseId>, passes: &[usize]) -> PanelRow {
        let mut cells = Vec::new();
        for (column, &won) in passes.iter().enumerate() {
            for n in 0..10 {
                cells.push(Cell {
                    task: Digest::of(format!("c{column}t{n}").as_bytes()),
                    family: format!("f{column}"),
                    column,
                    verdict: Some(n < won),
                });
            }
        }
        PanelRow {
            generation: 0,
            model: "m".into(),
            release,
            cells,
        }
    }

    fn all_tasks(columns: usize) -> BTreeMap<Digest, usize> {
        (0..columns)
            .flat_map(|c| (0..10).map(move |n| (Digest::of(format!("c{c}t{n}").as_bytes()), c)))
            .collect()
    }

    #[test]
    fn a_column_has_one_round_per_release_and_remeasuring_a_release_replaces_it() {
        let rows = [
            row(release(1), &[2, 5]),
            row(release(2), &[6, 5]),
            row(release(2), &[7, 5]),
        ];
        let refs: Vec<&PanelRow> = rows.iter().collect();
        let histories = column_histories(&refs, &all_tasks(2));
        let passes = |c: usize| -> Vec<u32> {
            histories[&c]
                .rounds
                .iter()
                .map(|r: &Round| r.passes)
                .collect()
        };
        assert_eq!(passes(0), [2, 7]);
        assert_eq!(passes(1), [5, 5]);
        assert!(histories[&0].rounds.iter().all(|r| r.attempts == 10));
    }

    #[test]
    fn cells_of_tasks_outside_the_current_panel_and_without_a_verdict_are_not_counted() {
        let mut one = row(release(1), &[4]);
        one.cells[0].verdict = None;
        let mut known = all_tasks(1);
        known.remove(&Digest::of(b"c0t9"));
        let histories = column_histories(&[&one], &known);
        let round = histories[&0].rounds[0];
        assert_eq!((round.attempts, round.passes), (8, 3));
    }

    #[test]
    fn the_column_still_moving_gets_the_larger_share_and_none_starves() {
        // Column 0 rose from 0.2 to 0.8 over two releases; column 1 sits at 0.5; column 2 at 1.0.
        let rows = [row(release(1), &[2, 5, 10]), row(release(2), &[8, 5, 10])];
        let refs: Vec<&PanelRow> = rows.iter().collect();
        let params = Params::default();
        let shares = replay_weights(&column_histories(&refs, &all_tasks(3)), &params);
        assert!(shares[&0] > 2.0 * shares[&1], "{shares:?}");
        assert!(shares[&0] > 2.0 * shares[&2], "{shares:?}");
        let floor = params.epsilon / 3.0;
        assert!(shares.values().all(|w| *w >= floor - 1e-12), "{shares:?}");
        assert!((shares.values().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn a_column_measured_under_one_release_only_is_weighed_by_novelty_alone() {
        let rows = [row(release(1), &[3, 3])];
        let refs: Vec<&PanelRow> = rows.iter().collect();
        let shares = replay_weights(&column_histories(&refs, &all_tasks(2)), &Params::default());
        assert!((shares[&0] - shares[&1]).abs() < 1e-9);
    }
}
