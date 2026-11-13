// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements emulator-verified firmware training
// environments, for its clients. If your team needs expertise in testing
// firmware against a reference model in simulation, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Firmware tasks verified in an emulator: faults injected into a small
//! bare-metal library for a Cortex-M33 board.
//!
//! The firmware (`fwkit`: a CRC, a ring buffer, fixed-point arithmetic, bit
//! operations and two codecs) runs on the emulated VEMU Cortex-M33 board and
//! prints its results on the console UART for a set of inputs in
//! `src/vectors.h`. The emulator's binary is the oracle's machine, read only
//! and run as a command: the plugin never modifies it. The grader holds a
//! reference model of the library (`model.py`) and hidden input seeds: for
//! each, it builds a private copy of the workspace with that seed's inputs
//! written over `src/vectors.h`, with its own compiler flags and linker
//! script, runs it in the emulator and compares the console output line by
//! line with the model. The workspace carries sample inputs only, so output
//! printed from a table cannot pass.
//!
//! Faults are mutations of the library's C sources, kept only when they
//! build, change the console output on the sample inputs and on at least one
//! hidden seed. The train split injects faults into the CRC, ring buffer and
//! fixed-point modules, the dev split into the bit-operation and codec
//! modules: a dev instance is a module the policy never practised on.
//!
//! Knobs: `faults` (how many, in different modules of the split's two or
//! three) and `detail` (3 the first differing console lines of the sample
//! run, 2 which kinds of line differ, 1 only that the output differs).

use std::collections::BTreeMap;

use splinter_core::rng::SplitMix64;
use splinter_core::verifiability::{Split, VerifierClass};
use splinter_eval::verifiers::mutation::mutants;

use crate::factory::artifact::{
    environment_id, EnvMeta, Instance, Parts, SolverTools, SCHEMA_VERSION,
};
use crate::factory::attack::{Attack, AttackKind, Edit};
use crate::factory::folder::Folder;
use crate::factory::host::Host;
use crate::factory::plugin::{
    resolve_difficulty, InstanceRequest, Knob, Plugin, PluginError, Reference,
};

use super::FIRMWARE_EMULATOR;

/// The plugin's version. Any change to the firmware, the model, the mutation
/// choice, the verifier or the instruction text is a new version.
pub const VERSION: &str = "1";

/// The modules train faults are injected into.
pub const TRAIN_MODULES: [&str; 3] = ["crc", "ring", "fixed"];
/// The modules dev faults are injected into.
pub const DEV_MODULES: [&str; 2] = ["bits", "codec"];

const KNOBS: &[Knob] = &[
    Knob { name: "faults", min: 1, max: 3, default: 1, doc: "how many faults are injected, each in another module" },
    Knob { name: "detail", min: 1, max: 3, default: 3, doc: "how much the report says: 3 the first differing console lines, 2 the kinds of line, 1 only that the output differs" },
];

/// Candidate mutants asked of the operators per module.
const MUTANTS_PER_FILE: usize = 30;
/// Candidates one generation may build and run while looking for valid faults.
const MAX_CANDIDATES: usize = 30;
/// How many hidden seeds the grader runs.
const HIDDEN_RUNS: usize = 3;

const MODEL: &str = include_str!("firmware_emulator/model.py");
const VERIFY: &str = include_str!("firmware_emulator/verify.py");
const LINKER: &str = include_str!("firmware_emulator/fwkit/link-m33.ld");

const FIXTURE: [(&str, &str); 16] = [
    ("Makefile", include_str!("firmware_emulator/fwkit/Makefile")),
    ("link-m33.ld", LINKER),
    (
        "src/bits.c",
        include_str!("firmware_emulator/fwkit/src/bits.c"),
    ),
    (
        "src/bits.h",
        include_str!("firmware_emulator/fwkit/src/bits.h"),
    ),
    (
        "src/codec.c",
        include_str!("firmware_emulator/fwkit/src/codec.c"),
    ),
    (
        "src/codec.h",
        include_str!("firmware_emulator/fwkit/src/codec.h"),
    ),
    (
        "src/console.c",
        include_str!("firmware_emulator/fwkit/src/console.c"),
    ),
    (
        "src/console.h",
        include_str!("firmware_emulator/fwkit/src/console.h"),
    ),
    (
        "src/crc.c",
        include_str!("firmware_emulator/fwkit/src/crc.c"),
    ),
    (
        "src/crc.h",
        include_str!("firmware_emulator/fwkit/src/crc.h"),
    ),
    (
        "src/fixed.c",
        include_str!("firmware_emulator/fwkit/src/fixed.c"),
    ),
    (
        "src/fixed.h",
        include_str!("firmware_emulator/fwkit/src/fixed.h"),
    ),
    (
        "src/main.c",
        include_str!("firmware_emulator/fwkit/src/main.c"),
    ),
    (
        "src/mcu.h",
        include_str!("firmware_emulator/fwkit/src/mcu.h"),
    ),
    (
        "src/ring.c",
        include_str!("firmware_emulator/fwkit/src/ring.c"),
    ),
    (
        "src/startup.c",
        include_str!("firmware_emulator/fwkit/src/startup.c"),
    ),
];
const RING_H: &str = include_str!("firmware_emulator/fwkit/src/ring.h");

/// What running the grader's check on a tree said.
#[derive(Debug, PartialEq, Eq)]
enum Check {
    Ok,
    /// The build failed, or the check itself did not run: what it said.
    BuildFailed(String),
    /// The first differing lines `(line, expected, got)` and how many differ.
    Differs(Vec<(usize, String, String)>, usize),
}

fn parse_check(stdout: &str) -> Check {
    let first = stdout.lines().next().unwrap_or("");
    if first.starts_with("ok") {
        return Check::Ok;
    }
    if first.starts_with("differs") {
        let shown: Vec<(usize, String, String)> = stdout
            .lines()
            .filter_map(|l| l.strip_prefix("differs "))
            .filter_map(|rest| {
                let mut parts = rest.splitn(3, " | ");
                let line = parts.next()?.trim().parse().ok()?;
                Some((line, parts.next()?.to_string(), parts.next()?.to_string()))
            })
            .collect();
        let count = stdout
            .lines()
            .find_map(|l| l.strip_prefix("count "))
            .and_then(|n| n.trim().parse().ok())
            .unwrap_or(shown.len());
        return Check::Differs(shown, count);
    }
    Check::BuildFailed(stdout.trim().to_string())
}

/// The firmware bug-injection plugin.
pub struct FirmwareEmulator {
    host: Host,
}

impl FirmwareEmulator {
    /// A plugin whose scripts find the emulator through `host`'s `VEMU`
    /// variable ([`Host::with_variable`]).
    #[must_use]
    pub fn new(host: Host) -> Self {
        Self { host }
    }

    fn io<E: std::fmt::Display>(e: E) -> PluginError {
        PluginError::Generation(e.to_string())
    }

    fn check(
        &self,
        grader: &std::path::Path,
        tree: &Folder,
        seed: u64,
    ) -> Result<Check, PluginError> {
        let scratch = tempfile::tempdir().map_err(Self::io)?;
        let dir = scratch.path().join("w");
        tree.write_to(&dir)?;
        let ran = self
            .host
            .exec(
                scratch.path(),
                vec![
                    "python3".into(),
                    "-I".into(),
                    grader.join("model.py").display().to_string(),
                    "check".into(),
                    dir.display().to_string(),
                    seed.to_string(),
                ],
                BTreeMap::new(),
            )
            .map_err(Self::io)?;
        let mut said = ran.stdout;
        if said.trim().is_empty() {
            said = format!("the check printed nothing: {}", ran.stderr.trim());
        }
        Ok(parse_check(&said))
    }
}

impl Plugin for FirmwareEmulator {
    fn id(&self) -> &'static str {
        FIRMWARE_EMULATOR
    }

    fn version(&self) -> &'static str {
        VERSION
    }

    fn verifier_class(&self) -> VerifierClass {
        VerifierClass::Objective
    }

    fn knobs(&self) -> &'static [Knob] {
        KNOBS
    }

    fn shares_answers(&self) -> bool {
        true
    }

    fn generate(&self, request: &InstanceRequest) -> Result<Instance, PluginError> {
        let difficulty = resolve_difficulty(self.id(), KNOBS, &request.difficulty)?;
        let detail = difficulty["detail"];
        let modules: &[&str] = match request.split {
            Split::Train => &TRAIN_MODULES,
            Split::Dev => &DEV_MODULES,
        };
        let faults = (difficulty["faults"] as usize).min(modules.len());
        let mut rng = SplitMix64::new(request.seed);
        let sample = rng.next_u64();
        let hidden: Vec<u64> = (0..HIDDEN_RUNS).map(|_| rng.next_u64()).collect();

        let grader = tempfile::tempdir().map_err(Self::io)?;
        std::fs::write(grader.path().join("model.py"), MODEL).map_err(Self::io)?;
        std::fs::write(grader.path().join("link-m33.ld"), LINKER).map_err(Self::io)?;

        let mut original = Folder::new();
        for (path, text) in FIXTURE {
            original.put_text(path, text)?;
        }
        original.put_text("src/ring.h", RING_H)?;
        // The sample inputs the solver sees.
        let header = self.header(grader.path(), sample)?;
        original.put_text("src/vectors.h", &header)?;
        let baseline = self.check(grader.path(), &original, sample)?;
        if baseline != Check::Ok {
            return Err(PluginError::Generation(format!(
                "the unmodified firmware does not match its model on the sample inputs: {baseline:?}"
            )));
        }

        // Valid faults: build, differ on the sample and on a hidden seed.
        let detected = |tree: &Folder| -> Result<Option<Check>, PluginError> {
            let on_sample = self.check(grader.path(), tree, sample)?;
            if !matches!(on_sample, Check::Differs(..)) {
                return Ok(None);
            }
            for &seed in &hidden {
                if matches!(self.check(grader.path(), tree, seed)?, Check::Differs(..)) {
                    return Ok(Some(on_sample));
                }
            }
            Ok(None)
        };
        let mut chosen: Vec<(String, String)> = Vec::new();
        for _ in 0..MAX_CANDIDATES {
            if chosen.len() == faults {
                break;
            }
            let module = modules[rng.below(modules.len())];
            let file = format!("src/{module}.c");
            if chosen.iter().any(|(path, _)| *path == file) {
                continue;
            }
            let Some(source) = original.text(&file) else {
                continue;
            };
            let candidates = mutants(source, MUTANTS_PER_FILE);
            let Some(mutant) = rng.pick(&candidates) else {
                continue;
            };
            let mut tree = original.clone();
            tree.put_text(&file, &mutant.text)?;
            if detected(&tree)?.is_some() {
                chosen.push((file, mutant.text.clone()));
            }
        }
        if chosen.len() < faults {
            return Err(PluginError::Generation(format!(
                "only {} valid faults found in {MAX_CANDIDATES} candidates, {faults} asked",
                chosen.len()
            )));
        }
        let mut mutated = original.clone();
        for (file, text) in &chosen {
            mutated.put_text(file, text)?;
        }
        let Some(Check::Differs(shown, count)) = detected(&mutated)? else {
            return Err(PluginError::Generation(
                "the combined faults are not detected".into(),
            ));
        };

        let mut tests = Folder::new();
        tests.put_script(
            "test.sh",
            "#!/bin/sh\n# Builds the workspace against hidden inputs, runs it in the emulator and compares the console with the model; see verify.py.\npython3 -I \"${TESTS_DIR:-/tests}/verify.py\"\n",
        )?;
        tests.put_text("verify.py", VERIFY)?;
        tests.put_text("model.py", MODEL)?;
        tests.put_text("link-m33.ld", LINKER)?;
        let seeds_json = serde_json::to_string(&hidden).map_err(Self::io)?;
        tests.put_text("hidden_inputs.json", &format!("{seeds_json}\n"))?;
        tests.put_text("sample_inputs.json", &format!("{sample}\n"))?;

        let mut solution = Folder::new();
        let mut solve = String::from(
            "#!/bin/sh\n# The reference: the original source of each faulty module.\nset -e\n",
        );
        for (n, (file, _)) in chosen.iter().enumerate() {
            solution.put_text(
                &format!("files/{n}"),
                original.text(file).unwrap_or_default(),
            )?;
            solve.push_str(&format!(
                "cp \"${{SOLUTION_DIR:-/solution}}/files/{n}\" \"${{APP_DIR:-/app}}/{file}\"\n"
            ));
        }
        solution.put_script("solve.sh", &solve)?;

        let symptoms = match detail {
            3 => {
                shown
                    .iter()
                    .take(3)
                    .map(|(line, want, have)| {
                        format!("- console line {line}: expected `{want}`, printed `{have}`\n")
                    })
                    .collect::<String>()
                    + &format!("- {count} console lines differ in all\n")
            }
            2 => {
                let mut kinds: Vec<&str> = shown
                    .iter()
                    .filter_map(|(_, want, _)| want.split(' ').next())
                    .collect();
                kinds.dedup();
                format!("- lines of kind {} differ\n", kinds.join(", "))
            }
            _ => "- the console output differs from the reference model\n".to_string(),
        };
        let instruction = format!(
            "# Fix the firmware\n\nThe firmware in `/app` (a CRC, a ring buffer, fixed-point arithmetic, bit operations and two codecs) runs on the emulated VEMU Cortex-M33 board: `make run` builds it and prints its results on the console for the sample inputs in `src/vectors.h`. Its output no longer matches the reference model.\n\nSymptoms on the sample inputs:\n{symptoms}\nFix the library sources so that the console output is correct for every input. The grader builds your sources with its own flags against inputs you do not have; it needs `vemu` and `clang`.\n"
        );

        let meta = EnvMeta {
            schema_version: SCHEMA_VERSION,
            environment: environment_id(self.id(), VERSION, &difficulty),
            plugin: self.id().into(),
            plugin_version: VERSION.into(),
            verifier_class: VerifierClass::Objective,
            verifier: format!("emulator-output-vs-model@{VERSION}"),
            difficulty,
            split: request.split,
            seed: request.seed,
            solver_tools: SolverTools::Shell,
        };
        Ok(Instance::assemble(
            meta,
            Parts {
                instruction,
                workspace: mutated,
                packages: vec![
                    "python3".into(),
                    "clang".into(),
                    "lld".into(),
                    "make".into(),
                ],
                tests,
                solution,
                category: "firmware".into(),
            },
        )?)
    }

    fn novelty_text(&self, instance: &Instance) -> String {
        let text = instance.instruction();
        text.split_once("Symptoms on the sample inputs:\n")
            .map_or(text, |(_, rest)| rest)
            .to_string()
    }

    fn attacks(&self, instance: &Instance, _reference: &Reference) -> Vec<Attack> {
        let write = |name: &str, body: &str| Attack {
            name: format!("fw:{name}"),
            kind: AttackKind::Tamper,
            edits: vec![Edit::Write {
                path: "src/main.c".into(),
                bytes: body.as_bytes().to_vec(),
            }],
        };
        let mut attacks = vec![
            write("silent", "int main(void) { return 0; }\n"),
            write(
                "prints-only-end",
                "#include \"console.h\"\nint main(void) { console_puts(\"END\\r\\n\"); return 0; }\n",
            ),
        ];
        // Printing the correct output of the sample inputs from a table: the
        // hidden inputs differ, so it cannot pass.
        let sample: Option<u64> = instance
            .folder()
            .text("tests/sample_inputs.json")
            .and_then(|t| t.trim().parse().ok());
        if let Some(seed) = sample {
            attacks.extend(self.table_attack(seed));
        }
        attacks
    }
}

impl FirmwareEmulator {
    /// The inputs header the model draws for `seed`.
    fn header(&self, grader: &std::path::Path, seed: u64) -> Result<String, PluginError> {
        let out = grader.join(format!("vectors-{seed}.h"));
        let ran = self
            .host
            .exec(
                grader,
                vec![
                    "python3".into(),
                    "-I".into(),
                    grader.join("model.py").display().to_string(),
                    "vectors".into(),
                    seed.to_string(),
                    out.display().to_string(),
                ],
                BTreeMap::new(),
            )
            .map_err(Self::io)?;
        if !ran.succeeded() {
            return Err(PluginError::Generation(format!(
                "the model failed: {}",
                ran.stderr.trim()
            )));
        }
        std::fs::read_to_string(&out).map_err(Self::io)
    }

    /// A main that prints the sample inputs' correct output from a table.
    fn table_attack(&self, seed: u64) -> Option<Attack> {
        let grader = tempfile::tempdir().ok()?;
        std::fs::write(grader.path().join("model.py"), MODEL).ok()?;
        let ran = self
            .host
            .exec(
                grader.path(),
                vec![
                    "python3".into(),
                    "-I".into(),
                    "-c".into(),
                    "import sys; sys.path.insert(0, '.'); import model; print('\\n'.join(model.vectors(int(sys.argv[1]))[1]))".into(),
                    seed.to_string(),
                ],
                BTreeMap::new(),
            )
            .ok()?;
        let mut body = String::from("#include \"console.h\"\nint main(void) {\n");
        for line in ran.stdout.lines() {
            body.push_str(&format!("\tconsole_puts(\"{line}\\r\\n\");\n"));
        }
        body.push_str("\treturn 0;\n}\n");
        Some(Attack {
            name: "fw:prints-the-sample-from-a-table".into(),
            kind: AttackKind::Tamper,
            edits: vec![Edit::Write {
                path: "src/main.c".into(),
                bytes: body.into_bytes(),
            }],
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // a panic is the failure report
mod tests {
    use super::*;

    #[test]
    fn train_and_dev_modules_are_disjoint() {
        for dev in DEV_MODULES {
            assert!(!TRAIN_MODULES.contains(&dev));
        }
    }

    #[test]
    fn a_check_report_is_read_as_ok_differing_lines_or_a_failed_build() {
        assert_eq!(parse_check("ok \n"), Check::Ok);
        assert!(matches!(
            parse_check(
                "build-failed clang: error
"
            ),
            Check::BuildFailed(_)
        ));
        let Check::Differs(shown, count) = parse_check(
            "differs 3 | crc 1 ab12 | crc 1 ab13\ndiffers 9 | END | <missing>\ncount 5\n",
        ) else {
            panic!("differs")
        };
        assert_eq!(
            shown,
            vec![
                (3, "crc 1 ab12".to_string(), "crc 1 ab13".to_string()),
                (9, "END".to_string(), "<missing>".to_string())
            ]
        );
        assert_eq!(count, 5);
    }
}
