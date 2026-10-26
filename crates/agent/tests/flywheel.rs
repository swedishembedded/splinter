// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agent-driven training-task generation that
// is judged by what the tasks do when run, for its clients. If your team
// needs expertise in synthetic task generation then you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: the invent harness is run as `sven harness run` on a workspace the
//! host lays out (`data/`, `brief.md`, `templates/solve/`, and `tasks/`, a git
//! repository that persists from run to run). What a run leaves is read back
//! from git, never from the files lying about: the task directories of the
//! commits the run added, and the outcome of every idea it logged. A fake
//! `sven` stands in for the binary, so the subprocess contract is tested
//! without a model.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::json;
use splinter_agent::flywheel::{
    DataFile, Invent, InventRequest, InventWorkspace, SvenLaunch, Yield,
};

/// What a run of invent does in the fake: it checks the workspace it was
/// given, records how it was called, and (as the real harness does) commits
/// one task directory per admitted idea and logs every idea's outcome. The
/// slug and the outcomes of the run are the files `slug` and `outcomes` beside
/// the fake. A directory it leaves uncommitted is not a task.
const FAKE_SVEN: &str = r#"#!/bin/sh
here=$(dirname "$0")
while [ $# -gt 0 ]; do
  case "$1" in
    --workspace) ws=$2 ;;
    --input-file) inputs=$2 ;;
    --trajectory) trajectory=$2 ;;
    --conversations) conversations=$2 ;;
    --model) model=$2 ;;
    --api-key-file) key=$2 ;;
    --timeout) timeout=$2 ;;
  esac
  shift
done
[ -f "$ws/data/notes.md" ] && [ -f "$ws/brief.md" ] && [ -f "$ws/templates/solve/harness.yaml" ] || { echo "workspace not laid out" >&2; exit 2; }
[ "$(cat "$here/refuse" 2>/dev/null)" = yes ] && { echo "refused" >&2; exit 2; }
cp "$inputs" "$here/inputs.json"
cp "$ws/brief.md" "$here/brief.md"
printf '%s' "$PATH" > "$here/path"
printf '%s %s %s' "$model" "$key" "$timeout" > "$here/args"
cd "$ws/tasks" || exit 2
if [ ! -d .git ]; then
  git init -q && git config user.name fake && git config user.email fake@localhost
fi
touch log.jsonl
slug=$(cat "$here/slug")
mkdir -p "$slug/environment" "$slug/tests" "$slug/solution"
echo "do the $slug task" > "$slug/instruction.md"
echo "echo 1" > "$slug/environment/calc.sh"
printf '#!/bin/sh\nexit 1\n' > "$slug/tests/test.sh"
printf '#!/bin/sh\nexit 0\n' > "$slug/solution/solve.sh"
cat "$here/outcomes" >> log.jsonl
git add -A "$slug" log.jsonl && git commit -q -m "$slug"
mkdir -p stray && echo "not committed" > stray/instruction.md
mkdir -p "$conversations"
echo '{}' > "$trajectory"
echo '{"status":"completed","conclusion":"success"}'
"#;

struct Rig {
    root: tempfile::TempDir,
}

impl Rig {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let sven = root.path().join("bin/sven");
        std::fs::create_dir_all(sven.parent().unwrap()).unwrap();
        std::fs::write(&sven, FAKE_SVEN).unwrap();
        std::fs::set_permissions(&sven, std::fs::Permissions::from_mode(0o755)).unwrap();
        // The fake reads its settings from beside itself.
        let solve = root.path().join("solve");
        std::fs::create_dir_all(&solve).unwrap();
        std::fs::write(solve.join("harness.yaml"), "api_version: \"2\"\n").unwrap();
        Self { root }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.root.path().join(name)
    }

    fn fake(&self, name: &str) -> PathBuf {
        self.path("bin").join(name)
    }

    fn invent(&self) -> Invent {
        let key = self.path("keys.json");
        std::fs::write(&key, r#"{"brain": "secret"}"#).unwrap();
        Invent::new(
            SvenLaunch {
                sven: self.path("bin/sven"),
                model: "brain/served-model".into(),
                endpoint: "http://127.0.0.1:1/v1".into(),
                key_file: Some(key),
                first_byte_timeout: None,
                runs: self.path("runs"),
            },
            PathBuf::from("invent-harness"),
            self.path("solve"),
        )
        .with_tool_dirs(vec![self.path("runtime/bin")])
    }

    /// One run that logs `outcomes` and commits the task `slug`.
    async fn run(
        &self,
        invent: &Invent,
        workspace: &InventWorkspace,
        slug: &str,
        outcomes: &[&str],
    ) -> Result<splinter_agent::flywheel::Invented, splinter_agent::flywheel::FlywheelError> {
        std::fs::write(self.fake("slug"), slug).unwrap();
        let lines: String = outcomes
            .iter()
            .map(|o| format!("{{\"outcome\": \"{o}\"}}\n"))
            .collect();
        std::fs::write(self.fake("outcomes"), lines).unwrap();
        invent
            .run(
                workspace,
                &InventRequest {
                    data: &[DataFile {
                        name: "notes.md".into(),
                        text: "the material".into(),
                    }],
                    brief: "Tasks for a maintainer.",
                    count: 3,
                    deadline: Duration::from_secs(1800),
                },
            )
            .await
    }
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap()
}

#[tokio::test]
async fn a_run_yields_the_committed_task_directories_and_the_outcomes_it_logged() {
    let rig = Rig::new();
    let invent = rig.invent();
    let workspace = InventWorkspace::create(&rig.path("ws")).unwrap();

    let invented = rig
        .run(
            &invent,
            &workspace,
            "calc-sum",
            &["admitted", "rejected", "duplicate", "failed", "failed"],
        )
        .await
        .unwrap();

    // The committed directory is a task; the uncommitted one is not.
    assert_eq!(invented.tasks.len(), 1);
    let task = &invented.tasks[0];
    assert!(task.ends_with("calc-sum"), "{task:?}");
    assert_eq!(read(&task.join("instruction.md")), "do the calc-sum task\n");
    assert!(task.join("tests/test.sh").is_file());
    assert_eq!(
        invented.yielded,
        Yield {
            admitted: 1,
            rejected: 1,
            duplicate: 1,
            failed: 2,
            unreadable: 0
        }
    );
    // The run was given what the contract says.
    let inputs: serde_json::Value = serde_json::from_str(&read(&rig.fake("inputs.json"))).unwrap();
    assert_eq!(inputs, json!({"count": 3}));
    assert_eq!(read(&rig.fake("brief.md")), "Tasks for a maintainer.");
    assert_eq!(
        read(&rig.fake("args")),
        format!(
            "brain/served-model {} 1800",
            rig.path("keys.json").display()
        )
    );
    // The runtime's directory and sven's own are on the child's PATH, so the
    // harness's shell finds the task runtime and `sven harness validate`.
    let path = read(&rig.fake("path"));
    assert!(
        path.contains(&rig.path("runtime/bin").display().to_string()),
        "{path}"
    );
    assert!(
        path.contains(&rig.path("bin").display().to_string()),
        "{path}"
    );
    // The trajectories are kept and recorded with how the run ended.
    assert!(invented.run.trajectory.is_file());
    assert_eq!(invented.run.conclusion.as_deref(), Some("success"));
}

#[tokio::test]
async fn a_later_run_yields_only_its_own_commits_and_outcomes() {
    let rig = Rig::new();
    let invent = rig.invent();
    let workspace = InventWorkspace::create(&rig.path("ws")).unwrap();
    rig.run(&invent, &workspace, "first-task", &["admitted", "failed"])
        .await
        .unwrap();

    let second = rig
        .run(
            &invent,
            &workspace,
            "second-task",
            &["admitted", "duplicate"],
        )
        .await
        .unwrap();

    assert_eq!(second.tasks.len(), 1);
    assert!(second.tasks[0].ends_with("second-task"));
    assert_eq!(second.yielded.admitted, 1);
    assert_eq!(second.yielded.duplicate, 1);
    assert_eq!(second.yielded.failed, 0);
}

#[tokio::test]
async fn a_document_sven_refuses_is_an_error_not_an_empty_run() {
    let rig = Rig::new();
    std::fs::write(rig.fake("refuse"), "yes").unwrap();
    let workspace = InventWorkspace::create(&rig.path("ws")).unwrap();

    let error = rig
        .run(&rig.invent(), &workspace, "calc-sum", &["admitted"])
        .await
        .unwrap_err();

    assert!(error.to_string().contains("refused"), "{error}");
}
