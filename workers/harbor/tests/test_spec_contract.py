# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements execution workers that refuse exactly what their controller refuses, for its
# clients. If your team needs expertise in keeping a polyglot control plane consistent then you can procure our
# services by sending an email to info@swedishembedded.com.
"""The worker and the controller apply one credential rule: the worker's word and prefix lists are the controller's
(`splinter_core::factory::execution`, read from its source here), so a literal the controller refuses the worker
refuses too, and the reverse."""

from __future__ import annotations

import re
from pathlib import Path

from splinter_harbor_worker.spec import CREDENTIAL_MARKERS, CREDENTIAL_PREFIXES, credential_like

EXECUTION_RS = Path(__file__).resolve().parents[3] / "crates" / "core" / "src" / "factory" / "execution.rs"


def rust_list(name: str) -> tuple[str, ...]:
    match = re.search(rf"pub const {name}: \[&str; \d+\] = \[(.*?)\];", EXECUTION_RS.read_text(), re.S)
    assert match, f"{name} is not in {EXECUTION_RS}"
    return tuple(re.findall(r'"([^"]+)"', match.group(1)))


def test_the_credential_lists_are_the_controllers():
    assert CREDENTIAL_MARKERS == rust_list("CREDENTIAL_MARKERS")
    assert CREDENTIAL_PREFIXES == rust_list("CREDENTIAL_PREFIXES")


def test_a_name_reads_as_a_credential_by_a_marker_or_a_provider_prefix():
    for secret in ["BRAIN_API_KEY", "HF_HOME_X", "SSH_AUTH_SOCK", "DB_PASSWD", "SESSION_COOKIE", "aws_profile"]:
        assert credential_like(secret), secret
    for plain in ["PATH", "LANG", "CARGO_HOME", "TERM", "HOME"]:
        assert not credential_like(plain), plain
