// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements data factories that run a team's own tools
// as reproducible, sandboxed pipeline steps, for its clients. If your team
// needs expertise in turning scripts into reliable pipeline operations, you
// can procure our services by sending an email to info@swedishembedded.com.

//! External operators: a task repository's own program, declared in
//! `operators/`, is an operation of its recipes. It runs in the sandbox on
//! the request it is given, its parameters are checked against its schema
//! before anything runs, and its code is pinned: a change to it is a change
//! of the run's inputs.

use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use splinter_core::factory::items::{ItemIndex, ITEMS_FILE};
use splinter_factory::resolve::resolve;
use splinter_factory::run::{Factory, Services};

use crate::fakes::{FakeBackend, Outcome};
use crate::repo::{connections, Repo};

const OPERATOR: &str = "schema: splinter.operator/v1
name: text.upper
revision: 1
summary: a text file upper-cased
inputs: [{ name: text, kind: files, many: false, optional: false }]
output: { name: upper, kind: files, many: false, optional: false }
parameters:
  type: object
  properties: { suffix: { type: string } }
  additionalProperties: false
reproducibility: deterministic
command: [./bin/upper.sh]
code: [bin/upper.sh]
";

const PROGRAM: &str = r#"#!/bin/sh
exec python3 - "$2" <<'PY'
import json, os, sys
request = json.load(open(sys.argv[1]))
source = request["inputs"]["text"][0]
text = open(source).read()
if "REJECT" in text:
    print("asks to be rejected")
    sys.exit(3)
name = os.path.join(request["out"], os.path.basename(source))
open(name, "w").write(text.upper() + request["params"].get("suffix", ""))
PY
"#;

const RECIPE: &str = "schema: splinter.recipe/v2
name: shout
inputs:
  notes: { kind: files, path: notes }
stages:
  - id: upper
    uses: text.upper
    foreach: $inputs.notes
    inputs: { text: $item }
    with: { suffix: \"!\" }
";

fn repo() -> Repo {
    let repo = Repo::new();
    repo.write("notes/a.txt", "alpha");
    repo.write("notes/c.txt", "REJECT me");
    repo.write("operators/upper.yaml", OPERATOR);
    program(&repo, PROGRAM);
    repo
}

fn program(repo: &Repo, text: &str) {
    repo.write("bin/upper.sh", text);
    let path = repo.root.join("bin/upper.sh");
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
}

fn run(repo: &Repo, dir: &Path) {
    let resolved = resolve(&repo.root, RECIPE, connections()).unwrap();
    let backend = FakeBackend::with(Outcome::Pass);
    let services = Services {
        backend: &backend,
        trainer: None,
        models: None,
    };
    Factory::open(&resolved, dir, services, "test")
        .unwrap()
        .run()
        .unwrap();
}

#[test]
fn a_repositorys_operator_runs_in_the_sandbox_on_its_request() {
    if splinter_sandbox::landlock::abi_version().is_none() {
        return;
    }
    let repo = repo();
    let dir = tempfile::tempdir().unwrap();
    run(&repo, dir.path());
    assert_eq!(
        std::fs::read_to_string(dir.path().join("upper/a.txt/a.txt")).unwrap(),
        "ALPHA!"
    );
    let index: ItemIndex = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("upper").join(ITEMS_FILE)).unwrap(),
    )
    .unwrap();
    assert_eq!(index.rejected.len(), 1);
    assert_eq!(index.rejected[0].reason, "asks to be rejected");
}

#[test]
fn an_operators_parameters_and_code_are_checked_and_pinned() {
    let repo = repo();
    let refused = resolve(
        &repo.root,
        &RECIPE.replace("suffix: \"!\"", "shout: true"),
        connections(),
    )
    .unwrap_err()
    .0
    .join("\n");
    assert!(refused.contains("shout"), "{refused}");

    let before = resolve(&repo.root, RECIPE, connections()).unwrap().lock;
    program(&repo, &PROGRAM.replace("upper()", "upper().strip()"));
    let after = resolve(&repo.root, RECIPE, connections()).unwrap().lock;
    assert!(
        before
            .changes(&after)
            .iter()
            .any(|c| c.contains("operator:text.upper")),
        "{:?}",
        before.changes(&after)
    );

    repo.write(
        "operators/broken.yaml",
        "schema: splinter.operator/v1\nname: x\n",
    );
    let broken = resolve(&repo.root, RECIPE, connections())
        .unwrap_err()
        .0
        .join("\n");
    assert!(broken.contains("operators/broken.yaml"), "{broken}");
}
