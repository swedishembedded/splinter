// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Hand-computed fixtures for the continual-learning metrics.

use super::*;

fn task(id: &str, unit: &str, learned_at: usize, base: Option<f64>) -> TaskColumn {
    TaskColumn {
        id: id.into(),
        unit: unit.into(),
        learned_at,
        base,
    }
}

/// Three generations, three tasks learned one per generation. Rows are the
/// generations, columns a, b, c:
///
/// ```text
/// base  0.4 0.1 0.2
/// k=0   0.6 0.2  -
/// k=1   0.8 0.7 0.3
/// k=2   0.5 0.6 0.9
/// ```
fn fixture() -> ResultsMatrix {
    let mut m = ResultsMatrix::new(
        3,
        vec![
            task("a", "a", 0, Some(0.4)),
            task("b", "b", 1, Some(0.1)),
            task("c", "c", 2, Some(0.2)),
        ],
    )
    .unwrap();
    for (k, i, v) in [
        (0, 0, 0.6),
        (0, 1, 0.2),
        (1, 0, 0.8),
        (1, 1, 0.7),
        (1, 2, 0.3),
        (2, 0, 0.5),
        (2, 1, 0.6),
        (2, 2, 0.9),
    ] {
        m.set(k, i, v).unwrap();
    }
    m
}

fn value(measure: &Measure) -> f64 {
    match measure {
        Measure::Measured { value, .. } => *value,
        Measure::NotMeasured { reason } => panic!("not measured: {reason}"),
    }
}

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-12, "{a} != {b}");
}

#[test]
fn the_five_metrics_equal_the_hand_computed_values() {
    let metrics = ContinualMetrics::of(&fixture());
    close(value(&metrics.acc), (0.5 + 0.6 + 0.9) / 3.0);
    // a: 0.5 - 0.6, b: 0.6 - 0.7.
    close(value(&metrics.bwt), -0.1);
    // a peaked at 0.8 after it was learned (0.8 - 0.5), b: 0.7 - 0.6. BWT
    // measures against the score right after learning, so the two differ.
    close(value(&metrics.forgetting), (0.3 + 0.1) / 2.0);
    // b before learning 0.2 against base 0.1, c: 0.3 against 0.2; a has no
    // earlier generation.
    close(value(&metrics.forward_transfer), 0.1);
    // Gains at the learning step: 0.2, 0.6, 0.7.
    close(value(&metrics.learning_gain), 0.5);
}

#[test]
fn the_gain_curve_follows_everything_learned_so_far() {
    let curve = ContinualMetrics::of(&fixture()).gain_curve;
    assert_eq!(curve.len(), 3);
    close(value(&curve[0].gain), 0.2);
    close(value(&curve[1].gain), 0.5);
    close(value(&curve[2].gain), (0.1 + 0.5 + 0.7) / 3.0);
}

#[test]
fn an_unmeasured_cell_is_skipped_and_never_read_as_zero() {
    let mut m = fixture();
    m.clear(2, 1); // b has no final score
    m.clear(0, 1); // b has no score before it was learned
    let metrics = ContinualMetrics::of(&m);
    close(value(&metrics.acc), (0.5 + 0.9) / 2.0);
    close(value(&metrics.bwt), -0.1 + 0.0 - 0.0 + 0.0); // only a
    close(value(&metrics.forgetting), 0.3);
    close(value(&metrics.forward_transfer), 0.1); // only c
}

#[test]
fn nothing_measured_is_typed_not_measured() {
    let m = ResultsMatrix::new(2, vec![task("a", "a", 0, None)]).unwrap();
    let metrics = ContinualMetrics::of(&m);
    for measure in [
        &metrics.acc,
        &metrics.bwt,
        &metrics.forgetting,
        &metrics.forward_transfer,
        &metrics.learning_gain,
    ] {
        assert!(
            matches!(measure, Measure::NotMeasured { .. }),
            "{measure:?}"
        );
    }
    // A single generation has nothing to forget or transfer to.
    let one = ResultsMatrix::new(1, vec![task("a", "a", 0, Some(0.1))]).unwrap();
    assert!(matches!(
        ContinualMetrics::of(&one).forgetting,
        Measure::NotMeasured { .. }
    ));
}

#[test]
fn tasks_of_one_unit_count_once() {
    let mut m = ResultsMatrix::new(
        2,
        vec![
            task("a1", "fam", 0, None),
            task("a2", "fam", 0, None),
            task("b", "other", 0, None),
        ],
    )
    .unwrap();
    for (i, first, last) in [(0, 1.0, 0.0), (1, 1.0, 0.5), (2, 1.0, 1.0)] {
        m.set(0, i, first).unwrap();
        m.set(1, i, last).unwrap();
    }
    // fam: mean(-1, -0.5) = -0.75; other: 0. Unit mean, not task mean.
    let Measure::Measured {
        value,
        units,
        tasks,
        unit_values,
        ..
    } = ContinualMetrics::of(&m).bwt
    else {
        panic!("measured");
    };
    close(value, -0.375);
    assert_eq!((units, tasks), (2, 3));
    assert_eq!(unit_values.len(), 2);
}

#[test]
fn a_matrix_that_contradicts_itself_is_refused() {
    assert!(ResultsMatrix::new(2, vec![task("a", "a", 2, None)]).is_err());
    assert!(ResultsMatrix::new(2, vec![task("a", "a", 0, Some(1.5))]).is_err());
    let mut m = ResultsMatrix::new(2, vec![task("a", "a", 0, None)]).unwrap();
    assert!(m.set(2, 0, 0.5).is_err());
    assert!(m.set(0, 1, 0.5).is_err());
    assert!(m.set(0, 0, f64::NAN).is_err());
}

#[test]
fn an_interval_is_attached_only_where_units_allow() {
    let metrics = ContinualMetrics::of(&fixture()).with_intervals(|values| {
        (values.len() >= 2).then(|| Interval {
            level: 0.9,
            low: -1.0,
            high: 1.0,
        })
    });
    let Measure::Measured { interval, .. } = &metrics.bwt else {
        panic!("measured");
    };
    assert_eq!(interval.map(|i| i.level), Some(0.9));
    let Measure::Measured { interval, .. } = &metrics.forward_transfer else {
        panic!("measured");
    };
    assert!(interval.is_some());
    let ours = ResultsMatrix::new(2, vec![task("a", "a", 0, Some(0.5))]).unwrap();
    let mut ours = ours;
    ours.set(0, 0, 0.7).unwrap();
    ours.set(1, 0, 0.9).unwrap();
    let Measure::Measured { interval, .. } = ContinualMetrics::of(&ours)
        .with_intervals(|_| None)
        .learning_gain
    else {
        panic!("measured");
    };
    assert!(
        interval.is_none(),
        "one unit has no interval, not a zero-width one"
    );
}
