#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for local installs of packaged
# command-line tools for its clients. If your team needs expertise in release
# engineering then you can procure our services by sending an email to
# info@swedishembedded.com.
#
# The spec of install-worker.sh (part of `make install`), against a scratch
# prefix and a fake python3 that records what it is asked to do:
#   - the worker goes into its own virtual environment under
#     <prefix>/lib/splinter/harbor-worker, installed from this checkout, and
#     <prefix>/bin/splinter-harbor-worker links to its program;
#   - installing again reinstalls into the same environment and leaves one link;
#   - a worker that does not answer --version fails the install.

setup() {
  SCRIPT="$(cd "$(dirname "$BATS_TEST_FILENAME")" && pwd)/install-worker.sh"
  WORK="$BATS_TEST_TMPDIR"
  PREFIX="$WORK/prefix"
  LOG="$WORK/python.log"
  : >"$LOG"
  # A fake python3: `-m venv DIR` makes DIR/bin/pip and DIR/bin/splinter-harbor-worker, which record their calls.
  cat >"$WORK/python3" <<FAKE
#!/bin/sh
echo "python3 \$*" >>"$LOG"
if [ "\$1" = "-m" ] && [ "\$2" = "venv" ]; then
  mkdir -p "\$3/bin"
  printf '#!/bin/sh\necho "pip \$*" >>"$LOG"\n' >"\$3/bin/pip"
  printf '#!/bin/sh\n[ -n "\$BROKEN" ] && exit 1\necho "splinter-harbor-worker 0.1.0 (harbor 0.20.0)"\n' >"\$3/bin/splinter-harbor-worker"
  chmod +x "\$3/bin/pip" "\$3/bin/splinter-harbor-worker"
fi
FAKE
  chmod +x "$WORK/python3"
}

@test "the worker is installed into its own environment and linked onto the prefix's PATH" {
  run bash "$SCRIPT" --prefix "$PREFIX" --python "$WORK/python3" --source workers/harbor
  [ "$status" -eq 0 ]
  [ -L "$PREFIX/bin/splinter-harbor-worker" ]
  [ "$(readlink "$PREFIX/bin/splinter-harbor-worker")" = "$PREFIX/lib/splinter/harbor-worker/bin/splinter-harbor-worker" ]
  grep -q "python3 -m venv $PREFIX/lib/splinter/harbor-worker" "$LOG"
  grep -q "pip install --upgrade workers/harbor" "$LOG"
  [[ "$output" == *"splinter-harbor-worker 0.1.0 (harbor 0.20.0)"* ]]
}

@test "installing again reuses the environment and leaves one link" {
  bash "$SCRIPT" --prefix "$PREFIX" --python "$WORK/python3" --source workers/harbor
  run bash "$SCRIPT" --prefix "$PREFIX" --python "$WORK/python3" --source workers/harbor
  [ "$status" -eq 0 ]
  [ "$(grep -c "pip install" "$LOG")" -eq 2 ]
  [ "$(ls "$PREFIX/bin" | wc -l)" -eq 1 ]
}

@test "a worker that does not answer --version fails the install" {
  BROKEN=1 run bash "$SCRIPT" --prefix "$PREFIX" --python "$WORK/python3" --source workers/harbor
  [ "$status" -ne 0 ]
}
