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
        variables: None,
        online: None,
        build: splinter_core::build::BuildInfo::this_executable(),
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

/// `dir` as a path relative to the process's working directory, as a
/// user types `--out runs/x`.
fn relative_to_cwd(dir: &Path) -> std::path::PathBuf {
    let cwd = std::env::current_dir().unwrap().canonicalize().unwrap();
    let dir = dir.canonicalize().unwrap();
    let common = cwd.ancestors().find(|a| dir.starts_with(a)).unwrap();
    let ups = cwd.strip_prefix(common).unwrap().components().count();
    let mut relative: std::path::PathBuf = std::iter::repeat_n("..", ups).collect();
    relative.push(dir.strip_prefix(common).unwrap());
    relative
}

#[test]
fn an_operator_is_given_absolute_paths_when_the_run_is_named_relatively() {
    if splinter_sandbox::landlock::abi_version().is_none() {
        return;
    }
    let repo = repo();
    let dir = tempfile::tempdir_in(env!("CARGO_TARGET_TMPDIR")).unwrap();
    let relative = relative_to_cwd(dir.path());
    assert!(relative.is_relative());
    run(&repo, &relative);
    assert_eq!(
        std::fs::read_to_string(dir.path().join("upper/a.txt/a.txt")).unwrap(),
        "ALPHA!"
    );
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

const COMPILER: &str = "schema: splinter.operator/v1
name: course.compile
revision: 1
summary: a course compiled into Harbor tasks
inputs: [{ name: course, kind: files, many: false, optional: false }]
output: { name: tasks, kind: files, many: false, optional: false }
parameters: { type: object, additionalProperties: false }
reproducibility: deterministic
command: [./bin/compile.sh]
code: [bin/compile.sh]
";

/// Every line of the course is a task of the train split, and one more of
/// the eval split.
const COMPILE: &str = r#"#!/bin/sh
exec python3 - "$2" <<'PY'
import json, os, sys
request = json.load(open(sys.argv[1]))
course = request["inputs"]["course"][0]
lines = [l.strip() for l in open(os.path.join(course, "files", "course.txt")) if l.strip()]
tasks = [("train/%03d-%s" % (i + 1, l), l) for i, l in enumerate(lines)] + [("eval/001-held", "held")]
for id, word in tasks:
    d = os.path.join(request["out"], id)
    os.makedirs(os.path.join(d, "tests"))
    open(os.path.join(d, "task.toml"), "w").write('[task]\nname = "org/%s"\n' % id)
    open(os.path.join(d, "instruction.md"), "w").write("Say %s.\n" % word)
    open(os.path.join(d, "tests", "test.sh"), "w").write("exit 0\n")
PY
"#;

#[test]
fn a_collection_a_stage_compiles_is_pinned_when_it_is_made() {
    if splinter_sandbox::landlock::abi_version().is_none() {
        return;
    }
    let repo = Repo::new();
    repo.write("courses/words/course.txt", "alpha\nbeta\n");
    repo.write("operators/compile.yaml", COMPILER);
    repo.write("bin/compile.sh", COMPILE);
    std::fs::set_permissions(
        repo.root.join("bin/compile.sh"),
        std::fs::Permissions::from_mode(0o755),
    )
    .unwrap();
    // seed: a recipe must state every seed; the fake backend's outcomes are fixed, so none draws.
    let recipe = "schema: splinter.recipe/v2
name: compiled
inputs:
  course: { kind: files, path: courses/words }
models:
  student: \"brain:Qwen/Qwen3-4B\"
stages:
  - { id: compile, uses: course.compile, inputs: { course: $inputs.course } }
  - { id: tasks, uses: tasks.pin, inputs: { collection: $stages.compile.tasks }, with: { split: train } }
  - id: local
    uses: attempt.collect
    inputs: { tasks: $stages.tasks.tasks }
    with: { agent: sven, model: student, environment: environments/vim.yaml, attempts: 1, seed: 3 }
";
    let resolved = resolve(&repo.root, recipe, connections()).unwrap();
    let backend = FakeBackend::with(Outcome::Pass);
    let services = Services {
        backend: &backend,
        trainer: None,
        models: None,
        variables: None,
        online: None,
        build: splinter_core::build::BuildInfo::this_executable(),
    };
    let dir = tempfile::tempdir().unwrap();
    Factory::open(&resolved, dir.path(), services, "test")
        .unwrap()
        .run()
        .unwrap();
    let attempted: Vec<String> = backend
        .executed()
        .iter()
        .map(|s| s.task.id.clone())
        .collect();
    assert_eq!(attempted, ["train/001-alpha", "train/002-beta"]);
}

const SECRETIVE: &str = "schema: splinter.operator/v1
name: text.sign
revision: 1
summary: a text file signed with the machine's key
inputs: [{ name: text, kind: files, many: false, optional: false }]
output: { name: signed, kind: files, many: false, optional: false }
parameters: { type: object, additionalProperties: false }
reproducibility: deterministic
command: [./bin/sign.sh]
code: [bin/sign.sh]
env: [SIGNING_KEY]
";

const SIGN: &str = "#!/bin/sh\nprintf '%s' \"$SIGNING_KEY\" > signed.txt\n";

const SIGNED: &str = "schema: splinter.recipe/v2
name: signed
inputs:
  notes: { kind: files, path: notes }
stages:
  - { id: signed, uses: text.sign, inputs: { text: $inputs.notes } }
";

#[test]
fn an_operator_gets_only_the_variables_its_machine_grants() {
    use splinter_factory::grants::Grants;
    use splinter_factory::resolve::resolve_in;
    if splinter_sandbox::landlock::abi_version().is_none() {
        return;
    }
    let repo = Repo::new();
    repo.write("notes/a.txt", "alpha");
    repo.write("operators/sign.yaml", SECRETIVE);
    repo.write("bin/sign.sh", SIGN);
    std::fs::set_permissions(
        repo.root.join("bin/sign.sh"),
        std::fs::Permissions::from_mode(0o755),
    )
    .unwrap();
    let refused = resolve_in(&repo.root, SIGNED, connections(), &Grants::default())
        .unwrap_err()
        .0
        .join("\n");
    assert!(
        refused.contains("variable SIGNING_KEY") && refused.contains("does not grant"),
        "{refused}"
    );

    // An operator the machine does not grant is no obstacle to a recipe that does not use it.
    let unrelated = "schema: splinter.recipe/v2\nname: unrelated\ninputs:\n  notes: { kind: files, path: notes }\nstages:\n  - { id: pin, uses: tasks.pin, inputs: { collection: $inputs.notes } }\n";
    let refused = resolve_in(&repo.root, unrelated, connections(), &Grants::default())
        .err()
        .map(|e| e.0.join("\n"))
        .unwrap_or_default();
    assert!(!refused.contains("does not grant"), "{refused}");

    let grants = Grants::parse(
        "schema: splinter.operator-grants/v1\ngrants:\n  text.sign: { env: [SIGNING_KEY] }\n",
    )
    .unwrap();
    let resolved = resolve_in(&repo.root, SIGNED, connections(), &grants).unwrap();
    let backend = FakeBackend::with(Outcome::Pass);
    let host = |name: &str| (name == "SIGNING_KEY").then(|| "machine-key".to_owned());
    let services = Services {
        backend: &backend,
        trainer: None,
        models: None,
        variables: Some(&host),
        online: None,
        build: splinter_core::build::BuildInfo::this_executable(),
    };
    let dir = tempfile::tempdir().unwrap();
    Factory::open(&resolved, dir.path(), services, "test")
        .unwrap()
        .run()
        .unwrap();
    assert_eq!(
        std::fs::read_to_string(dir.path().join("signed/signed.txt")).unwrap(),
        "machine-key"
    );
}

const EXAMINER: &str = "schema: splinter.operator/v1
name: exam.sealed
revision: 1
summary: the examiner, on sealed tasks it alone holds
inputs:
  - { name: candidate, kind: files, many: false, optional: false }
  - { name: model, kind: model, many: false, optional: false }
output: { name: verdict, kind: evaluation, many: false, optional: false }
parameters: { type: object, additionalProperties: false }
reproducibility: nondeterministic
command: [./eval/exam.sh]
code: [eval/exam.sh, eval/verdict.json]
";

/// The stand-in examiner: reads the candidate it is handed, then gives the
/// verdict the test prepared (part of its pinned code: what an operator
/// reads of the repository is what its identity pins).
const EXAM: &str = r#"#!/bin/sh
exec python3 - "$2" <<'PY'
import json, os, shutil, sys
request = json.load(open(sys.argv[1]))
candidate = json.load(open(os.path.join(request["inputs"]["candidate"][0], "candidate.json")))
assert candidate["adapter_digest"].startswith("sha256:")
shutil.copy(os.path.join(request["repository"], "eval", "verdict.json"), os.path.join(request["out"], "verdict.json"))
PY
"#;

#[test]
fn an_examiner_operator_examines_a_candidate_and_its_verdict_is_verified() {
    use crate::factory_run::{resolved, run, signed_verdict};
    use crate::fakes::FakeTrainer;
    use crate::repo::{evaluation, EVALUATION, RECIPE};
    use splinter_core::factory::product::Decision;
    if splinter_sandbox::landlock::abi_version().is_none() {
        return;
    }
    let repo = Repo::new();
    let backend = FakeBackend {
        candidate: Some(Outcome::Pass),
        ..FakeBackend::with(Outcome::Pass)
    };
    let trainer = FakeTrainer::default();
    let training_only = RECIPE.split(EVALUATION).next().unwrap();
    let dir = tempfile::tempdir().unwrap();
    run(
        &resolved(&repo, training_only),
        dir.path(),
        &backend,
        &trainer,
    )
    .unwrap();
    let candidate =
        splinter_factory::run::training::candidate_of(&dir.path().join("lora-r8")).unwrap();
    let digest = candidate
        .artifacts
        .iter()
        .find(|a| a.path.ends_with(".safetensors"))
        .unwrap()
        .digest
        .to_string();
    // seed: the examiner's test key; the gate checks a signature under it.
    let signer = ed25519_dalek::SigningKey::from_bytes(&[3; 32]);
    let public: String = signer
        .verifying_key()
        .to_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    repo.write("eval/examiner.pub", &public);
    repo.write(
        "eval/verdict.json",
        &signed_verdict(&signer, &digest, None, 0.05, "Promote"),
    );
    repo.write("operators/exam.yaml", EXAMINER);
    repo.write("eval/exam.sh", EXAM);
    std::fs::set_permissions(
        repo.root.join("eval/exam.sh"),
        std::fs::Permissions::from_mode(0o755),
    )
    .unwrap();
    // seed: the plan states its seed; the examiner draws nothing from it.
    repo.write(
        "eval/sealed.yaml",
        "schema: splinter.evalplan/v1\nattempts: 1\nseed: 5\ndecision: { alpha: 0.05, comparisons: 1 }\n\
         authority: { operator: exam.sealed, public_key: eval/examiner.pub }\n",
    );
    let recipe = training_only.to_owned() + &evaluation("lora-r8", "eval/sealed.yaml", "student");
    let examined = tempfile::tempdir().unwrap();
    splinter_factory::run::fork(dir.path(), examined.path()).unwrap();
    run(
        &resolved(&repo, &recipe),
        examined.path(),
        &backend,
        &trainer,
    )
    .unwrap();
    let result =
        splinter_factory::run::eval::result_of(&examined.path().join("eval-lora-r8")).unwrap();
    assert!(result.protected && result.signature.is_some());
    assert_eq!(result.decision, Decision::Improved);
}

#[test]
fn an_objective_compiled_into_environments_is_pinned_as_tasks() {
    use splinter_factory::run::tasks::tasks_of;
    let repo = Repo::new();
    repo.write(
        "objectives/python.md",
        "Predict the output of Python snippets\n",
    );
    // seed: the recipe states the generators' seed; the spec reads only what is admitted.
    let recipe = "schema: splinter.recipe/v2
name: built
stages:
  - id: built
    uses: environment.build
    with: { objective: objectives/python.md, environments: 1, instances: 2, seed: 7 }
  - { id: tasks, uses: tasks.pin, inputs: { collection: $stages.built.environments }, with: { split: train } }
";
    let resolved = resolve(&repo.root, recipe, connections()).unwrap();
    let backend = FakeBackend::with(Outcome::Pass);
    let services = Services {
        backend: &backend,
        trainer: None,
        models: None,
        variables: None,
        online: None,
        build: splinter_core::build::BuildInfo::this_executable(),
    };
    let dir = tempfile::tempdir().unwrap();
    Factory::open(&resolved, dir.path(), services, "test")
        .unwrap()
        .run()
        .unwrap();
    assert!(dir.path().join("built/compile.json").is_file());
    let pinned = tasks_of(&dir.path().join("tasks")).unwrap();
    assert_eq!(pinned.tasks.len(), 2, "{:?}", pinned.tasks);
    assert!(pinned.tasks.iter().all(|t| t.id.starts_with("train/")));
    for task in &pinned.tasks {
        assert!(dir
            .path()
            .join("tasks")
            .join(&task.id)
            .join("task.toml")
            .is_file());
    }
}

/// What an operator can read of the repository is what its stage's identity
/// pins: its own file and its `code`. A repository file it does not name is
/// out of its reach, so a cached product never depends on a file whose
/// change would not have invalidated it.
#[test]
fn an_operator_reads_only_what_its_identity_pins() {
    if splinter_sandbox::landlock::abi_version().is_none() {
        return;
    }
    let repo = repo();
    repo.write("private/notes.txt", "not pinned by the operator");
    program(
        &repo,
        "#!/bin/sh\nexec python3 - \"$2\" <<'PY'\nimport json, os, sys\nrequest = json.load(open(sys.argv[1]))\n\
         peek = open(os.path.join(request['repository'], 'private/notes.txt')).read()\n\
         open(os.path.join(request['out'], 'peek.txt'), 'w').write(peek)\nPY\n",
    );
    let resolved = resolve(&repo.root, RECIPE, connections()).unwrap();
    let backend = FakeBackend::with(Outcome::Pass);
    let services = Services {
        backend: &backend,
        trainer: None,
        models: None,
        variables: None,
        online: None,
        build: splinter_core::build::BuildInfo::this_executable(),
    };
    let dir = tempfile::tempdir().unwrap();
    let ran = Factory::open(&resolved, dir.path(), services, "test")
        .unwrap()
        .run();
    let refused = ran
        .expect_err("the unpinned file was out of reach")
        .to_string();
    assert!(refused.contains("text.upper"), "{refused}");
    assert!(!dir.path().join("upper/a.txt/peek.txt").exists());
}

/// Writes a file of 32 MiB beside its output: more than a run may write by default.
const LARGE: &str = r#"#!/bin/sh
exec python3 - "$2" <<'PY'
import json, os, sys
request = json.load(open(sys.argv[1]))
source = request["inputs"]["text"][0]
with open(os.path.join(request["out"], os.path.basename(source)), "wb") as out:
    out.write(b"x" * (32 * 1024 * 1024))
PY
"#;

fn try_run(repo: &Repo, dir: &Path) -> Result<(), String> {
    let resolved = resolve(&repo.root, RECIPE, connections()).map_err(|e| e.0.join("\n"))?;
    let backend = FakeBackend::with(Outcome::Pass);
    let services = Services {
        backend: &backend,
        trainer: None,
        models: None,
        variables: None,
        online: None,
        build: splinter_core::build::BuildInfo::this_executable(),
    };
    Factory::open(&resolved, dir, services, "test")
        .and_then(|mut f| f.run())
        .map(|_| ())
        .map_err(|e| e.to_string())
}

#[test]
fn an_operator_declares_the_resources_one_run_needs() {
    if splinter_sandbox::landlock::abi_version().is_none() {
        return;
    }
    let repo = repo();
    repo.write("notes/c.txt", "gamma");
    program(&repo, LARGE);
    let dir = tempfile::tempdir().unwrap();
    assert!(
        try_run(&repo, dir.path()).is_err(),
        "32 MiB is past the default file size"
    );

    let declared = OPERATOR.replace(
        "code: [bin/upper.sh]",
        "code: [bin/upper.sh]\nresources: { memory_mib: 2048, file_size_mib: 64, processes: 64 }",
    );
    repo.write("operators/upper.yaml", &declared);
    let dir = tempfile::tempdir().unwrap();
    try_run(&repo, dir.path()).unwrap();
    let written = std::fs::metadata(dir.path().join("upper/a.txt/a.txt")).unwrap();
    assert_eq!(written.len(), 32 * 1024 * 1024);

    repo.write(
        "operators/upper.yaml",
        &OPERATOR.replace(
            "code: [bin/upper.sh]",
            "code: [bin/upper.sh]\nresources: { memory_mib: 0 }",
        ),
    );
    let refused = resolve(&repo.root, RECIPE, connections())
        .unwrap_err()
        .0
        .join("\n");
    assert!(refused.contains("memory_mib"), "{refused}");
}

/// Lists its program's directory and tries to read the file beside it that its code does not pin.
const LISTER: &str = r#"#!/bin/sh
exec python3 - "$2" "$0" <<'PY'
import json, os, sys
request = json.load(open(sys.argv[1]))
here = os.path.dirname(sys.argv[2])
source = request["inputs"]["text"][0]
try:
    open(os.path.join(here, "secret.txt")).read()
    secret = "read"
except OSError:
    secret = "refused"
with open(os.path.join(request["out"], os.path.basename(source)), "w") as out:
    out.write(",".join(sorted(os.listdir(here))) + " " + secret)
PY
"#;

#[test]
fn an_operator_lists_the_directories_of_its_code_and_reads_only_what_it_pins() {
    if splinter_sandbox::landlock::abi_version().is_none() {
        return;
    }
    let repo = repo();
    repo.write("notes/c.txt", "gamma");
    repo.write("bin/secret.txt", "not pinned");
    program(&repo, LISTER);
    let dir = tempfile::tempdir().unwrap();
    run(&repo, dir.path());
    assert_eq!(
        std::fs::read_to_string(dir.path().join("upper/a.txt/a.txt")).unwrap(),
        "secret.txt,upper.sh refused"
    );
}
