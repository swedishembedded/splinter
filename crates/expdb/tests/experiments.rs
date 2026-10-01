// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: an experience is evidence for a question, so the question is a
//! record too: hypothesis, arms, what was measured and what was generated.
#![allow(clippy::unwrap_used)]

mod common;

use std::collections::BTreeMap;

use common::{attempt, coding_task, Scratch};
use splinter_expdb::model::{Conclusion, Experiment, ExperimentResult, Outcome, Rel};
use splinter_expdb::WriterIdentity;

#[test]
fn an_experiment_links_its_hypothesis_its_results_and_what_it_generated() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = db
        .collector(&WriterIdentity::new("exp", "job", "node", 0))
        .unwrap();
    let (_, instance, _) = coding_task(1);
    let experiment = c
        .start_experiment(Experiment {
            name: "inspect-first".into(),
            hypothesis: "inspecting state before editing improves debugging success".into(),
            control: "normal policy".into(),
            treatment: "strategy-conditioned policy".into(),
            tasks: vec![instance.id().unwrap()],
        })
        .unwrap();
    let (_, control_attempt) = attempt(&mut c, 1, "normal", 2, Outcome::Fail);
    let (_, treated_attempt) = attempt(&mut c, 1, "conditioned", 2, Outcome::Pass);
    c.link(experiment, Rel::Generated, control_attempt).unwrap();
    c.link(experiment, Rel::Generated, treated_attempt).unwrap();
    c.record_result(ExperimentResult {
        experiment,
        metrics: BTreeMap::from([
            ("control_success".to_owned(), 0.44),
            ("treatment_success".to_owned(), 0.63),
        ]),
        conclusion: Conclusion::Supported,
    })
    .unwrap();
    c.flush().unwrap();

    let view = db
        .snapshot()
        .unwrap()
        .experiment(experiment)
        .unwrap()
        .unwrap();
    assert_eq!(view.experiment.control, "normal policy");
    assert_eq!(view.results.len(), 1);
    assert_eq!(view.results[0].conclusion, Conclusion::Supported);
    let mut generated = view.generated.clone();
    generated.sort();
    let mut expected = vec![control_attempt, treated_attempt];
    expected.sort();
    assert_eq!(generated, expected);
}

#[test]
fn experiments_can_be_listed_by_what_they_concluded() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = db
        .collector(&WriterIdentity::new("exp", "job", "node", 0))
        .unwrap();
    let make = |c: &mut splinter_expdb::ingest::Collector, name: &str, conclusion| {
        let id = c
            .start_experiment(Experiment {
                name: name.into(),
                hypothesis: "h".into(),
                control: "a".into(),
                treatment: "b".into(),
                tasks: vec![],
            })
            .unwrap();
        c.record_result(ExperimentResult {
            experiment: id,
            metrics: BTreeMap::new(),
            conclusion,
        })
        .unwrap();
    };
    make(&mut c, "one", Conclusion::Supported);
    make(&mut c, "two", Conclusion::Refuted);
    make(&mut c, "three", Conclusion::Supported);
    c.flush().unwrap();

    let supported = db
        .snapshot()
        .unwrap()
        .experiments_concluding(Conclusion::Supported)
        .unwrap();
    let mut names: Vec<_> = supported
        .iter()
        .map(|v| v.experiment.name.as_str())
        .collect();
    names.sort_unstable();
    assert_eq!(names, ["one", "three"]);
}
