// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements sandboxed execution of a repository's own
// programs, for its clients. If your team needs expertise in running
// untrusted build and data tools safely, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Security: a task repository's operator is a program the run did not
//! write, and it reaches only what its machine grants. Without grants it
//! cannot write into the repository, read a file outside its inputs and
//! code, or open a network connection; each attempt is refused and the
//! operator still produces its output.

use std::net::TcpListener;
use std::os::unix::fs::PermissionsExt;

use serde_json::Value;
use splinter_factory::resolve::resolve;
use splinter_factory::run::{Factory, Services};

use crate::fakes::{FakeBackend, Outcome};
use crate::repo::{connections, Repo};

const OPERATOR: &str = "schema: splinter.operator/v1
name: probe.escape
revision: 1
summary: tries what an operator must not do, and says what happened
inputs: [{ name: text, kind: files, many: false, optional: false }]
output: { name: probed, kind: files, many: false, optional: false }
parameters:
  type: object
  properties: { secret: { type: string }, port: { type: integer }, repository: { type: string } }
  additionalProperties: false
reproducibility: deterministic
command: [./bin/probe.sh]
code: [bin/probe.sh]
";

const PROGRAM: &str = r#"#!/bin/sh
exec python3 - "$2" <<'PY'
import json, os, socket, sys
request = json.load(open(sys.argv[1]))
params = request["params"]
found = {}
def attempt(name, action):
    try:
        action()
        found[name] = "allowed"
    except Exception as e:
        found[name] = "refused: " + type(e).__name__
attempt("write_repository", lambda: open(os.path.join(params["repository"], "escaped.txt"), "w").write("x"))
attempt("read_secret", lambda: open(params["secret"]).read())
attempt("network", lambda: socket.create_connection(("127.0.0.1", params["port"]), timeout=3).close())
json.dump(found, open(os.path.join(request["out"], "probe.json"), "w"))
PY
"#;

#[test]
fn an_operator_reaches_nothing_its_machine_does_not_grant() {
    assert!(
        splinter_sandbox::landlock::abi_version().is_some(),
        "an operator's confinement needs Landlock, which this kernel does not offer"
    );
    let repo = Repo::new();
    repo.write("notes/a.txt", "alpha");
    repo.write("operators/probe.yaml", OPERATOR);
    repo.write("bin/probe.sh", PROGRAM);
    std::fs::set_permissions(
        repo.root.join("bin/probe.sh"),
        std::fs::Permissions::from_mode(0o755),
    )
    .unwrap();
    let elsewhere = tempfile::tempdir().unwrap();
    let secret = elsewhere.path().join("secret.txt");
    std::fs::write(&secret, "a key").unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let recipe = format!(
        "schema: splinter.recipe/v2
name: probed
inputs:
  notes: {{ kind: files, path: notes/a.txt }}
stages:
  - id: probe
    uses: probe.escape
    inputs: {{ text: $inputs.notes }}
    with: {{ secret: \"{}\", port: {port}, repository: \"{}\" }}
",
        secret.display(),
        repo.root.display()
    );
    let resolved = resolve(&repo.root, &recipe, connections()).unwrap();
    let backend = FakeBackend::with(Outcome::Pass);
    let services = Services {
        backend: &backend,
        trainer: None,
        models: None,
        variables: None,
        online: None,
    };
    let dir = tempfile::tempdir().unwrap();
    Factory::open(&resolved, dir.path(), services, "test")
        .unwrap()
        .run()
        .unwrap();
    let found: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("probe/probe.json")).unwrap(),
    )
    .unwrap();
    for probe in ["write_repository", "read_secret", "network"] {
        assert!(
            found[probe].as_str().unwrap().starts_with("refused"),
            "{probe}: {found:#}"
        );
    }
    assert!(!repo.root.join("escaped.txt").exists());
}
