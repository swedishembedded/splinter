// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A fake `sven` for the flywheel harness specs: it stands in for the
//! binary so no model is needed.

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

/// A harness run in the fake: it commits one task directory that meets the
/// contract (calc.sh must print 3) and one that lacks its tests, and logs
/// the outcome of five ideas.
pub const FAKE_SVEN: &str = r#"#!/bin/sh
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

/// [`FAKE_SVEN`] written as `dir/sven`, executable.
pub fn install(dir: &Path) -> PathBuf {
    std::fs::create_dir_all(dir).unwrap();
    let sven = dir.join("sven");
    std::fs::write(&sven, FAKE_SVEN).unwrap();
    std::fs::set_permissions(&sven, std::fs::Permissions::from_mode(0o755)).unwrap();
    sven
}
