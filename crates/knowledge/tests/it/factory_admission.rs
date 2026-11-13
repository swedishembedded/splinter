// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements admission and red-team testing of training
// environments, for its clients. If your team needs expertise in keeping
// weak verifiers out of reinforcement-learning campaigns, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Admission of environments end to end: real interpreters, real scripts,
//! and verifiers that are deliberately weak, each of which the red team must
//! catch.

#![allow(clippy::unwrap_used, clippy::expect_used)] // a panic is the failure report

use std::collections::{BTreeMap, BTreeSet};

use splinter_core::verifiability::{Split, VerifierClass};
use splinter_knowledge::factory::admission::{admit, AdmissionPolicy, Status};
use splinter_knowledge::factory::artifact::{
    environment_id, EnvMeta, Instance, Parts, SolverTools, SCHEMA_VERSION,
};
use splinter_knowledge::factory::band::BandMeasurement;
use splinter_knowledge::factory::folder::Folder;
use splinter_knowledge::factory::host::Host;
use splinter_knowledge::factory::plugin::{InstanceRequest, Knob, Plugin, PluginError};
use splinter_knowledge::factory::plugins::program_output::ProgramOutput;

const NO_KNOBS: &[Knob] = &[];

/// A plugin whose answer is a number drawn from the seed and whose verifier,
/// workspace and reference are whatever the test says.
pub(super) struct Scripted {
    pub(super) name: &'static str,
    pub(super) verifier: &'static str,
    pub(super) workspace: &'static [(&'static str, &'static str)],
    pub(super) solve: &'static str,
    pub(super) instruction: fn(u64) -> String,
}

fn number(seed: u64) -> u64 {
    100 + seed % 9000
}

const HEADER: &str = "APP=\"${APP_DIR:-/app}\"; TESTS=\"${TESTS_DIR:-/tests}\"; LOGS=\"${LOGS_DIR:-/logs/verifier}\"; mkdir -p \"$LOGS\"\n";

pub(super) const SOLVE_FROM_SOLUTION: &str =
    "#!/bin/sh\ncp \"${SOLUTION_DIR:-/solution}/answer.txt\" \"${APP_DIR:-/app}/answer.txt\"\n";

pub(super) fn plain_instruction(_: u64) -> String {
    "Write the secret number to answer.txt.".into()
}

impl Plugin for Scripted {
    fn id(&self) -> &'static str {
        self.name
    }
    fn version(&self) -> &'static str {
        "1"
    }
    fn verifier_class(&self) -> VerifierClass {
        VerifierClass::Objective
    }
    fn knobs(&self) -> &'static [Knob] {
        NO_KNOBS
    }
    fn novelty_text(&self, instance: &Instance) -> String {
        format!(
            "secret {} drawn {}",
            instance.meta().seed % 9000,
            instance.meta().seed
        )
    }
    fn generate(&self, request: &InstanceRequest) -> Result<Instance, PluginError> {
        let n = number(request.seed);
        let mut workspace = Folder::new();
        workspace.put_text("answer.txt", "")?;
        for (path, text) in self.workspace {
            workspace.put_text(path, &text.replace("{N}", &n.to_string()))?;
        }
        let mut tests = Folder::new();
        tests.put_script("test.sh", &format!("#!/bin/sh\n{HEADER}{}", self.verifier))?;
        tests.put_text("expected.txt", &format!("{n}\n"))?;
        let mut solution = Folder::new();
        solution.put_script("solve.sh", self.solve)?;
        solution.put_text("answer.txt", &format!("{n}\n"))?;
        let difficulty = BTreeMap::new();
        let meta = EnvMeta {
            schema_version: SCHEMA_VERSION,
            environment: environment_id(self.name, "1", &difficulty),
            plugin: self.name.into(),
            plugin_version: "1".into(),
            verifier_class: VerifierClass::Objective,
            verifier: format!("{}@1", self.name),
            difficulty,
            split: request.split,
            seed: request.seed,
            solver_tools: SolverTools::Files,
        };
        Ok(Instance::assemble(
            meta,
            Parts {
                instruction: (self.instruction)(n),
                workspace,
                packages: vec![],
                tests,
                solution,
                category: "test".into(),
            },
        )?)
    }
}

const EXACT: &str = "if [ \"$(cat \"$APP/answer.txt\" 2>/dev/null)\" = \"$(cat \"$TESTS/expected.txt\")\" ]; then r=1; else r=0; fi\nprintf '{\"reward\": %s}' \"$r\" > \"$LOGS/reward.json\"\n";

fn control() -> Scripted {
    Scripted {
        name: "control",
        verifier: EXACT,
        workspace: &[],
        solve: SOLVE_FROM_SOLUTION,
        instruction: plain_instruction,
    }
}

pub(super) fn run(
    plugin: &dyn Plugin,
    band: Option<&BandMeasurement>,
) -> splinter_knowledge::factory::admission::AdmissionReport {
    // seed: any admission seed must give the same verdicts on these environments
    let (report, _) = admit(
        plugin,
        &BTreeMap::new(),
        &Host::default(),
        &small_policy(),
        17,
        &BTreeSet::new(),
        band,
    )
    .expect("admission runs");
    report
}

/// Two train and two dev instances: every attack still runs on each.
pub(super) fn small_policy() -> AdmissionPolicy {
    AdmissionPolicy {
        instances_per_split: 2,
        ..AdmissionPolicy::default()
    }
}

pub(super) fn in_band() -> BandMeasurement {
    BandMeasurement {
        model: "base".into(),
        attempts: 32,
        passes: 16,
    }
}

pub(super) fn gate<'a>(
    report: &'a splinter_knowledge::factory::admission::AdmissionReport,
    name: &str,
) -> &'a splinter_knowledge::factory::admission::GateResult {
    report
        .gates
        .iter()
        .find(|g| g.name == name)
        .unwrap_or_else(|| panic!("no gate {name}"))
}

#[test]
fn an_honest_environment_passes_every_gate_and_still_waits_for_its_base_pass_rate() {
    let report = run(&control(), None);
    for g in &report.gates {
        assert!(g.passed, "{}: {}", g.name, g.detail);
    }
    assert_eq!(report.status, Status::PendingBand);
    assert!(!report.may_enter_campaign());
    assert!(report.attacks_run >= 40, "{}", report.attacks_run);
    assert!(report.beaten.is_empty(), "{:?}", report.beaten);
}

#[test]
fn a_base_pass_rate_inside_the_band_admits_and_outside_it_rejects_with_the_direction_to_turn() {
    let admitted = run(&control(), Some(&in_band()));
    assert_eq!(admitted.status, Status::Admitted);
    assert!(admitted.may_enter_campaign());

    let easy = run(
        &control(),
        Some(&BandMeasurement {
            model: "base".into(),
            attempts: 32,
            passes: 32,
        }),
    );
    assert_eq!(easy.status, Status::Rejected);
    assert!(gate(&easy, "base-band").detail.contains("too easy"));

    let hard = run(
        &control(),
        Some(&BandMeasurement {
            model: "base".into(),
            attempts: 32,
            passes: 0,
        }),
    );
    assert!(gate(&hard, "base-band").detail.contains("too hard"));

    let thin = run(
        &control(),
        Some(&BandMeasurement {
            model: "base".into(),
            attempts: 4,
            passes: 2,
        }),
    );
    assert_eq!(
        thin.status,
        Status::PendingBand,
        "four attempts do not make a band"
    );
}

#[test]
fn a_verifier_that_checks_only_the_shape_of_the_answer_is_quarantined() {
    let weak = Scripted {
        name: "shape-only",
        verifier: "if grep -Eq '^[0-9]+$' \"$APP/answer.txt\" 2>/dev/null; then r=1; else r=0; fi\nprintf '{\"reward\": %s}' \"$r\" > \"$LOGS/reward.json\"\n",
        workspace: &[],
        solve: SOLVE_FROM_SOLUTION,
        instruction: plain_instruction,
    };
    let report = run(&weak, Some(&in_band()));
    assert_eq!(report.status, Status::Quarantined, "{:?}", report.reasons);
    assert!(
        report.beaten.iter().any(|b| b.attack == "format-only"),
        "{:?}",
        report.beaten
    );
    assert!(!report.may_enter_campaign());
}

#[test]
fn a_verifier_whose_answer_is_the_same_for_every_instance_is_quarantined_by_the_cross_instance_attack(
) {
    let constant = Scripted {
        name: "constant",
        verifier: "if [ \"$(cat \"$APP/answer.txt\" 2>/dev/null)\" = 7 ]; then r=1; else r=0; fi\nprintf '{\"reward\": %s}' \"$r\" > \"$LOGS/reward.json\"\n",
        workspace: &[],
        // The reference writes the constant: it passes, and so does everyone who guesses it.
        solve: "#!/bin/sh\necho 7 > \"${APP_DIR:-/app}/answer.txt\"\n",
        instruction: plain_instruction,
    };
    let report = run(&constant, None);
    assert_eq!(report.status, Status::Quarantined);
    assert!(
        report.beaten.iter().any(|b| b.attack.starts_with("harvest")
            || b.attack.starts_with("cross-instance")
            || b.attack.starts_with("constant")),
        "{:?}",
        report.beaten
    );
}

#[test]
fn an_answer_that_can_be_read_from_the_workspace_is_quarantined() {
    let leaky = Scripted {
        name: "leaky",
        verifier: EXACT,
        workspace: &[("notes.txt", "reminder to self\n{N}\nend of notes\n")],
        solve: SOLVE_FROM_SOLUTION,
        instruction: plain_instruction,
    };
    let report = run(&leaky, Some(&in_band()));
    assert_eq!(report.status, Status::Quarantined);
    assert!(
        report
            .beaten
            .iter()
            .any(|b| b.attack.starts_with("leak:static:environment/notes.txt")),
        "{:?}",
        report.beaten
    );
    assert!(
        report
            .beaten
            .iter()
            .any(|b| b.attack.starts_with("harvest")),
        "harvesting the visible lines earns the reward: {:?}",
        report.beaten
    );
}

#[test]
fn a_verifier_that_runs_a_script_the_solver_can_edit_is_quarantined_as_tampering() {
    let trusting = Scripted {
        name: "trusting",
        verifier: "if (cd \"$APP\" && sh check.sh); then r=1; else r=0; fi\nprintf '{\"reward\": %s}' \"$r\" > \"$LOGS/reward.json\"\n",
        workspace: &[("check.sh", "[ \"$(cat answer.txt)\" = \"{N}\" ]\n")],
        solve: SOLVE_FROM_SOLUTION,
        instruction: |_| "Write the secret number to answer.txt; check.sh tells you if it is right.".into(),
    };
    let report = run(&trusting, None);
    assert_eq!(report.status, Status::Quarantined, "{:?}", report.reasons);
    assert!(
        report.beaten.iter().any(
            |b| b.attack == "tamper:exit-zero-scripts" || b.attack == "tamper:blank-everything"
        ),
        "{:?}",
        report.beaten
    );
}

#[test]
fn a_verifier_that_imports_from_the_working_directory_is_quarantined() {
    let importing = Scripted {
        name: "importing",
        verifier: "(cd \"$APP\" && python3 -c \"\nimport sqlite3, json, os\nok = open('answer.txt').read().strip() == open(os.environ['TESTS_DIR'] + '/expected.txt').read().strip()\nopen(os.environ['LOGS_DIR'] + '/reward.json', 'w').write(json.dumps({'reward': 1.0 if ok else 0.0}))\n\" 2>/dev/null)\n",
        workspace: &[],
        solve: SOLVE_FROM_SOLUTION,
        instruction: plain_instruction,
    };
    let report = run(&importing, None);
    assert_eq!(report.status, Status::Quarantined, "{:?}", report.reasons);
    assert!(
        report
            .beaten
            .iter()
            .any(|b| b.attack == "tamper:hijack-imports"),
        "{:?}",
        report.beaten
    );
}

#[test]
fn a_verifier_that_passes_an_empty_attempt_or_crashes_on_one_is_rejected_before_any_attack() {
    let passes_nothing = Scripted {
        name: "passes-nothing",
        verifier: "printf '{\"reward\": 1}' > \"$LOGS/reward.json\"\n",
        workspace: &[],
        solve: SOLVE_FROM_SOLUTION,
        instruction: plain_instruction,
    };
    let report = run(&passes_nothing, Some(&in_band()));
    assert_ne!(report.status, Status::Admitted);
    assert!(!gate(&report, "nop-fails").passed);

    let crashes = Scripted {
        name: "crashes",
        verifier: "[ -s \"$APP/answer.txt\" ] || exit 1\nprintf '{\"reward\": 1}' > \"$LOGS/reward.json\"\n",
        workspace: &[],
        solve: SOLVE_FROM_SOLUTION,
        instruction: plain_instruction,
    };
    let report = run(&crashes, Some(&in_band()));
    assert!(!gate(&report, "nop-fails").passed, "a crash is not a grade");
    assert!(!gate(&report, "robust").passed || !gate(&report, "nop-fails").passed);
}

#[test]
fn a_reference_that_does_not_solve_its_own_task_is_rejected() {
    let broken = Scripted {
        name: "broken-reference",
        verifier: EXACT,
        workspace: &[],
        solve: "#!/bin/sh\necho wrong > \"${APP_DIR:-/app}/answer.txt\"\n",
        instruction: plain_instruction,
    };
    let report = run(&broken, Some(&in_band()));
    assert_eq!(report.status, Status::Rejected);
    assert!(!gate(&report, "reference-passes").passed);
}

#[test]
fn a_repeated_instance_is_not_novel() {
    // A plugin that ignores the seed gives the same folder for every seed.
    struct Repeating(Scripted);
    impl Plugin for Repeating {
        fn id(&self) -> &'static str {
            "repeating"
        }
        fn version(&self) -> &'static str {
            "1"
        }
        fn verifier_class(&self) -> VerifierClass {
            VerifierClass::Objective
        }
        fn knobs(&self) -> &'static [Knob] {
            NO_KNOBS
        }
        fn novelty_text(&self, _: &Instance) -> String {
            "the same every time".into()
        }
        fn generate(&self, request: &InstanceRequest) -> Result<Instance, PluginError> {
            let mut fixed = request.clone();
            fixed.seed = 1; // seed: the point of this plugin is to ignore the seed
            let inner = self.0.generate(&fixed)?;
            let mut meta = inner.meta().clone();
            meta.seed = request.seed;
            meta.split = request.split;
            let text = inner
                .folder()
                .text("task.toml")
                .unwrap_or_default()
                .to_string();
            let mut folder = inner.folder().clone();
            folder.put_text(
                "task.toml",
                &text
                    .replace(
                        &format!("seed = \"{}\"", fixed.seed),
                        &format!("seed = \"{}\"", request.seed),
                    )
                    .replace(
                        "split = \"train\"",
                        &format!("split = \"{}\"", request.split),
                    ),
            )?;
            Ok(Instance::from_folder(folder).expect("reassembled"))
        }
    }
    let report = run(&Repeating(control()), Some(&in_band()));
    assert!(
        !gate(&report, "novel").passed,
        "{}",
        gate(&report, "novel").detail
    );
    assert_ne!(report.status, Status::Admitted);
    assert!(!report.may_enter_campaign());
    assert_eq!(Split::Dev.as_str(), "dev");
}

#[test]
fn lua_and_python_program_output_environments_pass_admission_and_reward_only_the_interpreters_output(
) {
    for plugin in [ProgramOutput::lua(), ProgramOutput::python()] {
        let report = run(&plugin, Some(&in_band()));
        for g in &report.gates {
            assert!(g.passed, "{} {}: {}", plugin.id(), g.name, g.detail);
        }
        assert_eq!(report.status, Status::Admitted, "{:?}", report.reasons);
        assert_eq!(report.verifier_class, VerifierClass::Objective);
        assert_eq!(report.label, None);
        assert!(gate(&report, "reproducible").passed);
    }
}

#[test]
fn a_difficulty_knob_gives_a_different_environment_that_is_admitted_on_its_own() {
    let hard = BTreeMap::from([
        ("blocks".to_string(), 5),
        ("features".to_string(), 4),
        ("magnitude".to_string(), 3),
    ]);
    let (report, instances) = admit(
        &ProgramOutput::lua(),
        &hard,
        &Host::default(),
        &small_policy(),
        23, // seed: any admission seed
        &BTreeSet::new(),
        Some(&in_band()),
    )
    .expect("admission runs");
    assert_eq!(report.status, Status::Admitted, "{:?}", report.reasons);
    let default = run(&ProgramOutput::lua(), None);
    assert_ne!(report.environment, default.environment);
    assert_eq!(instances.len(), 4);
    assert_eq!(
        instances
            .iter()
            .filter(|i| i.meta().split == Split::Dev)
            .count(),
        2
    );
}
