// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that let a training campaign draw
// only on environments whose verifiers were attacked and whose difficulty was
// measured, for its clients. If your team needs expertise in admitting
// training environments to reinforcement-learning campaigns you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: a campaign takes its environments from the factory store, and only
//! admitted environments whose recorded base band is learnable enter it; the
//! tasks are graded by the environment's own verifier on instances staged
//! and removed per attempt.
#![allow(clippy::unwrap_used)]

use splinter_knowledge::factory::admission::{AdmissionPolicy, Status};
use splinter_knowledge::factory::band::BandMeasurement;
use splinter_knowledge::factory::compile::compile;
use splinter_knowledge::factory::host::Host;
use splinter_knowledge::factory::objective::{Budget, ObjectiveSpec};
use splinter_knowledge::factory::plugins::builtin;
use splinter_knowledge::factory::store::EnvironmentStore;
use splinter_pipelines::campaign::brain::TaskFamily;
use splinter_pipelines::campaign::factory::{select, FactoryFamily};
use splinter_pipelines::campaign::EnvironmentLabels;
use splinter_sandbox::host::DEFAULT_SEARCH_PATH;

fn compiled_store(dir: &std::path::Path) -> EnvironmentStore {
    let store = EnvironmentStore::new(dir.join("envs"));
    let spec = ObjectiveSpec {
        text: "Predict the output of short Lua programs".into(),
        sources: Vec::new(),
        constraints: vec!["lua".into()],
        budget: Budget {
            environments: 2,
            instances_per_split: 4,
        },
    };
    let policy = AdmissionPolicy {
        instances_per_split: 4,
        ..AdmissionPolicy::default()
    };
    let seed = splinter_model::seed::draw().value();
    println!("seed {seed}");
    compile(
        &spec,
        &builtin(),
        &Host::with_path(DEFAULT_SEARCH_PATH),
        &policy,
        seed,
        &store,
    )
    .unwrap();
    store
}

fn measure(store: &EnvironmentStore, id: &str, attempts: u32, passes: u32) {
    store
        .record_band(
            id,
            &BandMeasurement {
                model: "base".into(),
                attempts,
                passes,
            },
        )
        .unwrap();
}

#[test]
fn only_admitted_environments_with_a_learnable_recorded_band_enter_a_campaign() {
    let dir = tempfile::tempdir().unwrap();
    let store = compiled_store(dir.path());
    let ids: Vec<String> = store
        .list()
        .unwrap()
        .iter()
        .map(|r| r.environment.clone())
        .collect();
    assert!(ids.len() >= 2, "{ids:?}");

    // Nothing measured yet: nothing may enter.
    let none = select(&store, &[]).unwrap();
    assert!(none.chosen.is_empty());
    assert!(
        none.refused
            .iter()
            .all(|r| r.reason.contains("Admitted") || r.reason.contains("no base")),
        "{:?}",
        none.refused
    );

    measure(&store, &ids[0], 32, 12);
    measure(&store, &ids[1], 32, 31);
    let selection = select(&store, &[]).unwrap();
    assert_eq!(selection.chosen.len(), 1, "{selection:?}");
    assert_eq!(selection.chosen[0].environment, ids[0]);
    let refused = selection
        .refused
        .iter()
        .find(|r| r.environment == ids[1])
        .unwrap();
    assert!(
        refused.reason.contains("band") || refused.reason.contains("Rejected"),
        "{refused:?}"
    );

    assert!(select(&store, &["no-such-environment".into()]).is_err());
}

#[test]
fn tasks_are_graded_by_the_environments_own_verifier_and_staged_per_attempt() {
    let dir = tempfile::tempdir().unwrap();
    let store = compiled_store(dir.path());
    let id = store.list().unwrap()[0].environment.clone();
    measure(&store, &id, 32, 12);
    let selection = select(&store, std::slice::from_ref(&id)).unwrap();
    let family = FactoryFamily::open(
        store.clone(),
        &selection,
        Host::with_path(DEFAULT_SEARCH_PATH),
    )
    .unwrap();
    let task = family.pool()[0].clone();
    assert_eq!(family.environment_of(&task).as_deref(), Some(id.as_str()));
    assert_eq!(family.dev().len(), 4);

    let prompt = family.prompt(&task).unwrap();
    assert!(
        !prompt[1].1.contains("solve.sh"),
        "the solution is not shown"
    );
    // The reference answer: what the interpreter prints for the program.
    let instance = store
        .instances(&id, splinter_core::verifiability::Split::Train)
        .unwrap()
        .remove(0);
    let host = Host::with_path(DEFAULT_SEARCH_PATH);
    let session = host.open(&instance).unwrap();
    session.solve().unwrap();
    let answer = session
        .workspace()
        .unwrap()
        .text("answer.txt")
        .unwrap()
        .to_string();
    assert!(family.grade(&task, &answer).unwrap().passed);
    assert!(!family.grade(&task, "definitely not it").unwrap().passed);

    // Quarantining marks the stored report; the environment cannot be chosen again.
    family
        .quarantine(&id, "the reward stopped meaning anything")
        .unwrap();
    assert_eq!(store.report(&id).unwrap().status, Status::Quarantined);
    assert!(select(&store, &[id]).unwrap().chosen.is_empty());
}
