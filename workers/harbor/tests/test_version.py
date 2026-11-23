# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
"""The worker names its own version and Harbor's: the controller records it as part of every rollout's identity."""

import re
import subprocess
import sys
from pathlib import Path


def test_the_version_names_the_worker_and_its_pinned_harbor() -> None:
    out = subprocess.run([str(Path(sys.executable).with_name("splinter-harbor-worker")), "--version"], capture_output=True, text=True, check=True)
    assert re.fullmatch(r"splinter-harbor-worker \S+ \(harbor 0\.20\.0\)\n", out.stdout), out.stdout
