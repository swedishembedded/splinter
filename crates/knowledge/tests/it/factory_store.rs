// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements admission and red-team testing of training
// environments, for its clients. If your team needs expertise in keeping
// weak verifiers out of reinforcement-learning campaigns, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The database workflow environment and the compiler and store, end to end.

#![allow(clippy::unwrap_used, clippy::expect_used)] // a panic is the failure report

use std::collections::{BTreeMap, BTreeSet};

use splinter_core::verifiability::{Split, VerifierClass};
use splinter_knowledge::factory::admission::{admit, Status};
use splinter_knowledge::factory::artifact::Instance;
use splinter_knowledge::factory::band::BandMeasurement;
use splinter_knowledge::factory::compile::compile;
use splinter_knowledge::factory::host::{Host, Verdict};
use splinter_knowledge::factory::objective::{Budget, ObjectiveSpec};
use splinter_knowledge::factory::plugin::{InstanceRequest, Plugin};
use splinter_knowledge::factory::plugins::builtin;
use splinter_knowledge::factory::plugins::db_workflow::DbWorkflow;
use splinter_knowledge::factory::plugins::repository_bugs::RepositoryBugs;
use splinter_knowledge::factory::store::{EnvironmentStore, StoreError};

use super::factory_admission::{
    in_band, plain_instruction, run, small_policy, Scripted, SOLVE_FROM_SOLUTION,
};

fn db_request(seed: u64, goals: u32, traps: u32) -> InstanceRequest {
    InstanceRequest {
        split: Split::Train,
        seed,
        difficulty: BTreeMap::from([
            ("goals".to_string(), goals),
            ("traps".to_string(), traps),
            ("rows".to_string(), 6),
            ("distractor_tables".to_string(), 1),
        ]),
    }
}

#[test]
fn the_database_workflow_environment_passes_admission_and_every_database_attack_fails() {
    let report = run(&DbWorkflow::new(), Some(&in_band()));
    for g in &report.gates {
        assert!(g.passed, "{}: {}", g.name, g.detail);
    }
    assert_eq!(report.status, Status::Admitted, "{:?}", report.reasons);
    assert!(report.beaten.is_empty(), "{:?}", report.beaten);
    assert!(
        report.attacks_run >= 40,
        "the plugin's own attacks run beside the generic ones: {}",
        report.attacks_run
    );
}

#[test]
fn the_final_state_diff_rejects_acting_on_a_request_whose_answer_is_to_change_nothing() {
    // seed: scanning for an instance with a restock request that needs no restock
    let plugin = DbWorkflow::new();
    let host = Host::default();
    let found = (0..200)
        .map(|s| plugin.generate(&db_request(s, 4, 3)).unwrap())
        .find(|instance| {
            let ops = instance.folder().text("tests/ops.json").unwrap();
            instance.instruction().contains("must have at least") && !ops.contains("restock")
        });
    let instance = found.expect("some seed gives a restock request with nothing to restock");
    let session = host.open(&instance).unwrap();
    assert!(session.solve().unwrap().succeeded());
    assert!(
        session.verify().unwrap().solved(),
        "the reference alone solves it"
    );

    let sku = instance
        .instruction()
        .split("Product ")
        .nth(1)
        .unwrap()
        .split(' ')
        .next()
        .unwrap()
        .to_string();
    let blind = session
        .run_in_workspace(&["python3", "tools.py", "restock", &sku, "1"])
        .unwrap();
    assert!(blind.succeeded());
    let verdict = session.verify().unwrap();
    assert!(
        !verdict.solved(),
        "one unit too many is collateral damage: {verdict:?}"
    );
    let Verdict::Graded(vector) = verdict else {
        panic!("graded")
    };
    assert!(
        vector["state"] > 0.0 && vector["state"] < 1.0,
        "partial credit by table: {vector:?}"
    );
}

#[test]
fn the_shops_tool_refuses_what_its_rules_forbid_and_changes_nothing() {
    let instance = DbWorkflow::new().generate(&db_request(5, 3, 0)).unwrap(); // seed: any instance has a shipped or pending order 200
    let session = Host::default().open(&instance).unwrap();
    let before = session.workspace_digest().unwrap();
    let refused = session
        .run_in_workspace(&["python3", "tools.py", "refund", "99999"])
        .unwrap();
    assert_eq!(refused.exit_code, Some(2), "{refused:?}");
    assert!(refused.stdout.contains("refused"));
    assert_eq!(session.workspace_digest().unwrap(), before);
}

fn objective() -> ObjectiveSpec {
    ObjectiveSpec {
        text: "- Predict the output of short Lua programs\n- Handle orders in an inventory database through its tools\n- Hold an empathetic counselling conversation\n- Configure a UART peripheral in firmware".into(),
        sources: vec![],
        constraints: vec![],
        budget: Budget { environments: 4, instances_per_split: 2 },
    }
}

#[test]
fn an_objective_compiles_to_stored_environments_and_says_what_it_could_not_build_and_why() {
    let root = tempfile::tempdir().unwrap();
    let store = EnvironmentStore::new(root.path());
    // seed: any compile seed must give the same shape of outcome
    let report = compile(
        &objective(),
        &builtin(),
        &Host::default(),
        &small_policy(),
        31,
        &store,
    )
    .unwrap();

    assert_eq!(report.skills.len(), 4);
    let lua = &report.skills[0];
    assert_eq!(lua.classification.class, VerifierClass::Objective);
    assert_eq!(
        lua.environments.len(),
        2,
        "four environments over two buildable skills, a ladder of two each"
    );
    assert!(
        lua.environments
            .iter()
            .all(|e| e.status == Status::PendingBand),
        "{:?}",
        lua.environments
    );
    assert_ne!(
        lua.environments[0].difficulty, lua.environments[1].difficulty,
        "a ladder differs by difficulty"
    );

    let counselling = &report.skills[2];
    assert_eq!(counselling.label.as_deref(), Some("proxy only"));
    assert!(counselling.environments.is_empty());
    assert!(counselling
        .not_built
        .as_deref()
        .unwrap()
        .contains("no automatic environment"));
    let firmware = &report.skills[3];
    assert!(
        firmware
            .not_built
            .as_deref()
            .unwrap()
            .contains("not built yet"),
        "{:?}",
        firmware.not_built
    );

    let outcomes: Vec<_> = report.skills.iter().flat_map(|s| &s.environments).collect();
    assert_eq!(report.pending_band(), 4, "{outcomes:#?}");
    assert_eq!(
        report.admitted(),
        0,
        "nothing enters a campaign before its base pass rate is measured"
    );
    assert_eq!(store.list().unwrap().len(), 4);
    for environment in &lua.environments {
        let train = store
            .instances(&environment.environment, Split::Train)
            .unwrap();
        assert_eq!(train.len(), 2);
        let dev = store
            .instances(&environment.environment, Split::Dev)
            .unwrap();
        assert_eq!(dev.len(), 2);
        let instance = train[0].clone();
        assert_eq!(instance.meta().environment, environment.environment);
        assert_eq!(instance.meta().split, Split::Train);
    }
}

#[test]
fn a_measured_base_pass_rate_admits_a_stored_environment_and_the_decision_survives_a_reload() {
    let root = tempfile::tempdir().unwrap();
    let store = EnvironmentStore::new(root.path());
    let report = compile(
        &objective(),
        &builtin(),
        &Host::default(),
        &small_policy(),
        5,
        &store,
    )
    .unwrap(); // seed: any
    let id = report.skills[0].environments[0].environment.clone();
    assert_eq!(store.report(&id).unwrap().status, Status::PendingBand);

    let judged = store.record_band(&id, &in_band()).unwrap();
    assert_eq!(judged.status, Status::Admitted);
    let reloaded = EnvironmentStore::new(root.path()).report(&id).unwrap();
    assert!(reloaded.may_enter_campaign());
    assert_eq!(reloaded.band.unwrap().attempts, 32);

    let too_easy = store
        .record_band(
            &id,
            &BandMeasurement {
                model: "base".into(),
                attempts: 32,
                passes: 32,
            },
        )
        .unwrap();
    assert_eq!(
        too_easy.status,
        Status::Rejected,
        "a newer measurement is judged again"
    );
}

#[test]
fn a_second_compile_with_another_seed_does_not_repeat_what_was_trained_on_and_the_same_seed_is_refused_as_not_novel(
) {
    let root = tempfile::tempdir().unwrap();
    let store = EnvironmentStore::new(root.path());
    let spec = ObjectiveSpec {
        text: "Predict the output of Python snippets".into(),
        budget: Budget {
            environments: 1,
            instances_per_split: 2,
        },
        ..ObjectiveSpec::default()
    };
    let first = compile(
        &spec,
        &builtin(),
        &Host::default(),
        &small_policy(),
        100,
        &store,
    )
    .unwrap(); // seed: any
    assert_eq!(first.pending_band(), 1);
    let again = compile(
        &spec,
        &builtin(),
        &Host::default(),
        &small_policy(),
        100,
        &store,
    )
    .unwrap();
    assert_eq!(
        again.skills[0].environments[0].status,
        Status::Rejected,
        "{:?}",
        again.skills[0].environments
    );
    assert!(again.skills[0].environments[0]
        .reasons
        .iter()
        .any(|r| r.contains("already trained on")));
}

#[test]
fn a_quarantined_environment_keeps_its_report_and_stores_no_instance() {
    let root = tempfile::tempdir().unwrap();
    let store = EnvironmentStore::new(root.path());
    let weak = Scripted {
        name: "shape-only",
        verifier: "if grep -Eq '^[0-9]+$' \"$APP/answer.txt\" 2>/dev/null; then r=1; else r=0; fi\nprintf '{\"reward\": %s}' \"$r\" > \"$LOGS/reward.json\"\n",
        workspace: &[],
        solve: SOLVE_FROM_SOLUTION,
        instruction: plain_instruction,
    };
    let (report, instances) = admit(
        &weak,
        &BTreeMap::new(),
        &Host::default(),
        &small_policy(),
        3,
        &BTreeSet::new(),
        None,
    )
    .unwrap(); // seed: any
    assert_eq!(report.status, Status::Quarantined);
    store.save(&report, &instances).unwrap();
    assert!(store
        .instances(&report.environment, Split::Train)
        .unwrap()
        .is_empty());
    assert!(store
        .report(&report.environment)
        .unwrap()
        .beaten
        .iter()
        .any(|b| b.attack == "format-only"));
}

#[test]
fn an_environment_id_cannot_name_a_path_outside_the_store() {
    let store = EnvironmentStore::new(tempfile::tempdir().unwrap().path());
    for bad in [
        "../etc",
        "env-../../x",
        "env-zz",
        "",
        "env-0123456789abcdef/..",
    ] {
        assert!(
            matches!(store.report(bad), Err(StoreError::BadId(_))),
            "{bad}"
        );
    }
    assert!(matches!(
        store.report("env-0123456789abcdef"),
        Err(StoreError::NotFound(_))
    ));
}

#[test]
fn a_collection_is_one_archive_and_a_task_folder_exists_only_while_it_is_staged() {
    let root = tempfile::tempdir().unwrap();
    let store = EnvironmentStore::new(root.path().join("store"));
    let single = ObjectiveSpec {
        text: "Handle orders in an inventory database through its tools".into(),
        budget: Budget {
            environments: 1,
            instances_per_split: 2,
        },
        ..ObjectiveSpec::default()
    };
    let report = compile(
        &single,
        &builtin(),
        &Host::default(),
        &small_policy(),
        9,
        &store,
    )
    .unwrap(); // seed: any
    let id = report.skills[0].environments[0].environment.clone();

    // The environment is two files however many instances it holds.
    let files: Vec<_> = std::fs::read_dir(root.path().join("store").join(&id))
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    assert_eq!(files.len(), 2, "{files:?}");
    assert!(
        files.contains(&"instances.json".to_string())
            && files.contains(&"admission.json".to_string())
    );

    // Staged, it is a runnable Harbor folder; dropped, it is gone.
    let instance = store.instances(&id, Split::Dev).unwrap().remove(0);
    let staging = root.path().join("staging");
    let path = {
        let staged = EnvironmentStore::stage(&staging, &instance).unwrap();
        assert!(staged.path().join("task.toml").is_file());
        assert!(staged.path().join("tests/test.sh").is_file());
        staged.path().to_path_buf()
    };
    assert!(!path.exists());
    assert_eq!(
        std::fs::read_dir(&staging).unwrap().count(),
        0,
        "nothing is left behind"
    );

    // A reload from the archive is byte-identical to what was generated.
    let again = store.instances(&id, Split::Dev).unwrap().remove(0);
    assert_eq!(again.version(), instance.version());
}

#[test]
fn the_repository_environment_passes_admission_and_no_attack_on_the_test_runner_wins() {
    let report = run(&RepositoryBugs::new(), Some(&in_band()));
    for g in &report.gates {
        assert!(g.passed, "{}: {}", g.name, g.detail);
    }
    assert_eq!(report.status, Status::Admitted, "{:?}", report.reasons);
    assert!(report.beaten.is_empty(), "{:?}", report.beaten);
    assert!(
        report.attacks_run >= 40,
        "generic and runner attacks both ran: {}",
        report.attacks_run
    );
}

#[test]
fn fixing_only_some_of_the_faults_earns_a_share_of_the_tests_and_no_reward() {
    let plugin = RepositoryBugs::new();
    let request = InstanceRequest {
        split: Split::Dev,
        seed: 11, // seed: the share of passing tests asserted below depends on which faults this instance holds
        difficulty: BTreeMap::from([("bugs".to_string(), 4), ("detail".to_string(), 3)]),
    };
    let instance = plugin.generate(&request).unwrap();
    let session = Host::default().open(&instance).unwrap();
    assert!(!session.verify().unwrap().solved());
    // Put one faulty file back by hand: some tests pass, the reward is still zero.
    let fixed = instance
        .folder()
        .under("solution/textkit")
        .next()
        .map(|(path, entry)| (path["solution/".len()..].to_string(), entry.bytes.clone()))
        .unwrap();
    session.write(&fixed.0, &fixed.1).unwrap();
    let verdict = session.verify().unwrap();
    let Verdict::Graded(vector) = verdict else {
        panic!("graded")
    };
    assert_eq!(vector["reward"], 0.0, "{vector:?}");
    assert!(vector["tests"] > 0.8 && vector["tests"] < 1.0, "{vector:?}");
    // The full reference solves it.
    assert!(session.solve().unwrap().succeeded());
    assert!(session.verify().unwrap().solved());
}

#[test]
fn the_canary_is_what_stops_a_library_that_patches_the_test_framework() {
    let plugin = RepositoryBugs::new();
    let request = InstanceRequest {
        split: Split::Train,
        seed: 3, // seed: the patch must leave this instance unsolved, which depends on its faults
        difficulty: BTreeMap::from([("bugs".to_string(), 2), ("detail".to_string(), 3)]),
    };
    let instance = plugin.generate(&request).unwrap();
    let patching = "import unittest\n\n\ndef _run(self, result=None):\n    result.startTest(self)\n    result.addSuccess(self)\n    result.stopTest(self)\n    return result\n\n\nunittest.TestCase.run = _run\n";

    let session = Host::default().open(&instance).unwrap();
    session
        .write("textkit/__init__.py", patching.as_bytes())
        .unwrap();
    assert!(
        !session.verify().unwrap().solved(),
        "the canary reports the patch"
    );

    // The same verifier without its canary is beaten by the same patch.
    let mut folder = instance.folder().clone();
    let runner = folder.text("tests/runner.py").unwrap().replace(
        "canary_failed = 0 if canary.wasSuccessful() else 1",
        "canary_failed = 1",
    );
    folder.put_text("tests/runner.py", &runner).unwrap();
    let naive = Instance::from_folder(folder).unwrap();
    let session = Host::default().open(&naive).unwrap();
    session
        .write("textkit/__init__.py", patching.as_bytes())
        .unwrap();
    assert!(
        session.verify().unwrap().solved(),
        "without the canary the patch earns the reward"
    );
}
