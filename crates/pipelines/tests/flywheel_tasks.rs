// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agent-driven training-task generation that
// is judged by what the tasks do when run, for its clients. If your team
// needs expertise in synthetic task generation then you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: with the flywheel harnesses named, the tasks stage runs invent once
//! per window of the sources and admits what it commits.
//!
//! * a task directory the run committed becomes a workspace task of the set
//!   (admission proves its tests fail on the untouched environment and pass
//!   after its reference solution);
//! * a committed directory that does not hold what the contract names is
//!   refused with the reason and counted, never dropped silently;
//! * the report carries the yield invent logged - admitted, rejected,
//!   duplicate, failed - beside what the stage admitted and refused.
//!
//! * with a validate harness, a passing attempt is kept only when every
//!   focus decides keep; a discard is reported with its reasons, and a
//!   validator run that gave no decision is reported apart, never counted as
//!   a discard, and its attempt is not kept either.
//!
//! A fake `sven` stands in for the binary, so no model is needed.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::time::{Duration, Instant};

use common::{scratch_context, Scratch, Scripted};
use splinter_agent::CancelToken;
use splinter_core::experience::{Experience, PrivilegedKind, Provenance};
use splinter_core::kinds::WORKSPACE_CHANGE;
use splinter_core::model_ref::ModelRef;
use splinter_core::source::SourceId;
use splinter_orchestrator::Context;
use splinter_pipelines::sources::{self, SourceTarget};
use splinter_pipelines::tasks::{generate, Generation, TaskHarness, TasksGenerated};
use splinter_pipelines::validate::{validate_attempts, ValidateRequest, Validated};
use splinter_pipelines::verify::{verify_set, Grading};
use splinter_store::experiences::{ExperienceSet, SetId};

/// A harness run in the fake: it commits one task directory that meets the
/// contract (calc.sh must print 3) and one that lacks its tests, and logs
/// the outcome of five ideas.
const FAKE_SVEN: &str = r#"#!/bin/sh
while [ $# -gt 0 ]; do
  case "$1" in
    --workspace) ws=$2 ;;
    --input-file) inputs=$2 ;;
    --trajectory) trajectory=$2 ;;
    --conversations) conversations=$2 ;;
  esac
  shift
done
if [ -f "$ws/task.md" ]; then
  # A validator run: it must be shown the work, the tests and the conversation.
  [ -f "$ws/tests.md" ] && [ -f "$ws/trajectory.md" ] && [ "$(cat "$ws/workspace/calc.sh")" = "echo 3" ] || { echo "attempt not laid out" >&2; exit 2; }
  focus=$(sed -n 's/.*"focus":"\([a-z]*\)".*/\1/p' "$inputs")
  decision=$(sed -n "s/^$focus //p" "$(dirname "$0")/decisions")
  case "$decision" in
    keep|discard) echo "{\"status\":\"completed\",\"conclusion\":\"success\",\"output\":{\"decision\":{\"decision\":\"$decision\",\"evidence\":\"e\",\"rejection_reasons\":[\"not $focus\"]}}}" ;;
    *) echo '{"status":"failed","conclusion":"error"}' ;;
  esac
  exit 0
fi
cd "$ws/tasks" || exit 2
[ -d .git ] || { git init -q && git config user.name fake && git config user.email fake@localhost; }
mkdir -p calc-sum/environment calc-sum/tests calc-sum/solution no-tests/environment no-tests/solution
echo "Make calc.sh print the sum of one and two." > calc-sum/instruction.md
echo "echo 1" > calc-sum/environment/calc.sh
printf '#!/bin/sh\n[ "$(sh calc.sh)" = 3 ]\n' > calc-sum/tests/test.sh
printf '#!/bin/sh\necho "echo 3" > calc.sh\n' > calc-sum/solution/solve.sh
echo "Make calc.sh print the product of two and three." > no-tests/instruction.md
echo "echo 1" > no-tests/environment/calc.sh
printf '#!/bin/sh\ntrue\n' > no-tests/solution/solve.sh
for outcome in admitted admitted rejected duplicate failed; do echo "{\"outcome\": \"$outcome\"}" >> log.jsonl; done
git add -A . && git commit -q -m tasks
mkdir -p "$conversations"
echo '{}' > "$trajectory"
echo '{"status":"completed","conclusion":"success"}'
"#;

fn write(path: &Path, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

/// A state root with a source, the fake `sven`, and the harnesses naming it.
struct World {
    /// Removes the state root when the test ends.
    _scratch: Scratch,
    ctx: Context,
    harness: TaskHarness,
    source: SourceId,
}

fn world(test: &str, validate: bool) -> World {
    let (scratch, ctx) = scratch_context(test, Scripted::new(|_| "{}".to_string()), false);
    let dir = scratch.0.join("harness");
    let sven = dir.join("bin/sven");
    write(&sven, FAKE_SVEN);
    std::fs::set_permissions(&sven, std::fs::Permissions::from_mode(0o755)).unwrap();
    write(&dir.join("solve/harness.yaml"), "api_version: \"2\"\n");
    let docs = scratch.0.join("docs");
    write(
        &docs.join("manual.md"),
        "# Manual\n\n## Sum\n\nCalc adds numbers.\n",
    );
    let source = sources::add(
        &ctx,
        &SourceTarget::from_learn_arg(&docs.display().to_string()).unwrap(),
    )
    .unwrap()
    .source
    .id;
    let harness = TaskHarness {
        invent: dir.join("invent"),
        solve: dir.join("solve"),
        validate: validate.then(|| dir.join("validate")),
        sven,
        endpoint: "http://127.0.0.1:1/v1".into(),
        model: "brain/served".into(),
        key_file: None,
        angles: 2,
        run_timeout_secs: 600,
    };
    World {
        _scratch: scratch,
        ctx,
        harness,
        source,
    }
}

fn invent(world: &World) -> TasksGenerated {
    generate(
        &world.ctx,
        &Generation {
            sources: std::slice::from_ref(&world.source),
            sections: &[],
            kinds: &["recall".to_string()],
            generator: &ModelRef::policy_default(),
            goal: None,
            author: None,
            written: &[],
            denoise_seed: None,
            deadline: Some(Instant::now() + Duration::from_secs(300)),
            cancel: CancelToken::new(),
            harness: Some(&world.harness),
        },
    )
    .unwrap()
}

#[test]
fn the_tasks_a_run_commits_are_admitted_the_malformed_one_is_refused_and_the_yield_is_reported() {
    let generated = invent(&world("flywheel-tasks", false));

    assert_eq!(generated.tasks, 1, "{generated:#?}");
    assert_eq!(generated.per_kind[WORKSPACE_CHANGE].admitted, 1);
    let invent = generated.invent.expect("the report carries invent's yield");
    assert_eq!(invent.yielded.admitted, 2);
    assert_eq!(invent.yielded.rejected, 1);
    assert_eq!(invent.yielded.duplicate, 1);
    assert_eq!(invent.yielded.failed, 1);
    // Invent logged two admissions; the stage admitted one and says why not
    // the other.
    assert_eq!(invent.refused.len(), 1, "{invent:#?}");
    assert!(invent.refused[0].directory.ends_with("no-tests"));
    assert!(
        invent.refused[0].reason.contains("tests/test.sh"),
        "{invent:#?}"
    );
    assert_eq!(generated.rejected.len(), 1, "{:?}", generated.rejected);
}

/// The attempt that applies the reference solution's change to the one
/// admitted task, verified: it passes.
fn passing_attempt(world: &World, generated: &TasksGenerated) -> SetId {
    let tasks = world.ctx.tasks();
    let entry = tasks
        .get_set(&generated.task_set)
        .unwrap()
        .members
        .remove(0);
    let task = tasks.get(&entry.task).unwrap();
    let reference = task
        .privileged
        .iter()
        .find(|p| p.kind == PrivilegedKind::Reference)
        .unwrap()
        .content
        .clone();
    let provenance = Provenance::new("test/solver", world.ctx.clock());
    let experience = Experience::answered_without_a_run(task, &reference, provenance).unwrap();
    let store = world.ctx.experiences();
    let id = store.put(&experience).unwrap();
    let set = store
        .put_set(&ExperienceSet {
            name: "attempts".into(),
            members: vec![id],
        })
        .unwrap();
    let verified = verify_set(&world.ctx, &set, Grading::CodeOnly, &CancelToken::new()).unwrap();
    assert_eq!(verified.passed, 1, "{verified:#?}");
    set
}

fn validate(world: &World, attempts: &SetId, decisions: &str) -> Validated {
    write(&world.harness.sven.with_file_name("decisions"), decisions);
    validate_attempts(
        &world.ctx,
        &ValidateRequest {
            attempts,
            harness: &world.harness,
            deadline: None,
            cancel: CancelToken::new(),
        },
    )
    .unwrap()
}

#[test]
fn an_attempt_is_kept_only_when_every_focus_decides_keep() {
    let world = world("flywheel-validate-keep", true);
    let attempts = passing_attempt(&world, &invent(&world));

    let all_keep = validate(
        &world,
        &attempts,
        "outcome keep\nprocess keep\nteaching keep\n",
    );
    assert_eq!((all_keep.passing, all_keep.kept), (1, 1), "{all_keep:#?}");
    let members = world
        .ctx
        .experiences()
        .get_set(&all_keep.experience_set)
        .unwrap();
    assert_eq!(members.members.len(), 1);

    let one_discard = validate(
        &world,
        &attempts,
        "outcome keep\nprocess keep\nteaching discard\n",
    );
    assert_eq!(one_discard.kept, 0);
    assert_eq!(one_discard.discarded.len(), 1, "{one_discard:#?}");
    assert_eq!(one_discard.discarded[0].focus, "teaching");
    assert_eq!(one_discard.discarded[0].reasons, ["not teaching"]);
    assert!(one_discard.undecided.is_empty());
    assert!(world
        .ctx
        .experiences()
        .get_set(&one_discard.experience_set)
        .unwrap()
        .members
        .is_empty());
}

#[test]
fn a_validator_run_without_a_decision_is_reported_apart_and_is_not_a_discard() {
    let world = world("flywheel-validate-undecided", true);
    let attempts = passing_attempt(&world, &invent(&world));

    let validated = validate(
        &world,
        &attempts,
        "outcome keep\nprocess keep\nteaching broken\n",
    );

    assert_eq!(validated.kept, 0, "nothing vouches for the attempt");
    assert!(validated.discarded.is_empty(), "{validated:#?}");
    assert_eq!(validated.undecided.len(), 1, "{validated:#?}");
    assert_eq!(validated.undecided[0].focus, "teaching");
}
