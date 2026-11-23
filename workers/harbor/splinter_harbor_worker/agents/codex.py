# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for measuring different coding
# agents on the same tasks under the same conditions, for its clients. If your
# team needs expertise in fair, independent agent comparison then you can
# procure our services by sending an email to info@swedishembedded.com.
"""Harbor adapter for OpenAI's Codex CLI, a third agent beside Sven and Claude Code.

The ``codex`` executable is staged by the worker at ``/splinter/bin/codex``. It runs non-interactively (``codex exec``)
with its own sandbox and approvals off (``--dangerously-bypass-approvals-and-sandbox``: the trial's sandbox is the
boundary) and the spec's arguments passed through as an argv. Its configuration lives in ``CODEX_HOME`` below the
trial's isolated HOME, never the user's: one model provider, the worker's gateway, spoken over the Responses API, whose
key is the trial's gateway token in the variable the provider's ``env_key`` names. An appended system prompt is the
global ``AGENTS.md`` of that home. It leaves no ATIF trace, so the worker reports no turns for it.
"""

from __future__ import annotations

import json

from splinter_harbor_worker.agents.base import HOME, SplinterAgent, sandbox_args

CODEX_HOME = f"{HOME}/.codex"
KEY_VARIABLE = "SPLINTER_GATEWAY_TOKEN"


def _toml_string(text: str) -> str:
    """``text`` as a TOML basic string (JSON's escapes are TOML's for these characters)."""
    return json.dumps(text)


class CodexAgent(SplinterAgent):
    BINARY = "codex"

    @staticmethod
    def name() -> str:
        return "codex"

    def parse_version(self, stdout: str) -> str:
        return stdout.strip().split()[-1]

    def agent_env(self, token: str | None) -> dict[str, str]:
        env = {"CODEX_HOME": CODEX_HOME}
        if token is not None:
            env[KEY_VARIABLE] = token
        return env

    def home_files(self) -> dict[str, str]:
        settings = self._settings
        # Written even with no gateway, so the directory CODEX_HOME names exists (Codex refuses one that does not).
        lines = []
        if settings.get("gateway_url"):
            lines += [
                f"model = {_toml_string(settings['model'])}",
                'model_provider = "splinter"',
                "",
                "[model_providers.splinter]",
                'name = "splinter gateway"',
                f"base_url = {_toml_string(settings['gateway_url'].rstrip('/') + '/v1')}",
                f'env_key = "{KEY_VARIABLE}"',
                'wire_api = "responses"',
            ]
        files = {".codex/config.toml": "\n".join(lines) + "\n"}
        if settings.get("system_prompt"):
            files[".codex/AGENTS.md"] = settings["system_prompt"]
        return files

    def argv(self, instruction: str, run_timeout_sec: int) -> list[str]:
        settings = self._settings
        return [*self._command.split(), "exec", "--dangerously-bypass-approvals-and-sandbox", "--skip-git-repo-check",
                *sandbox_args(settings.get("args", []), settings.get("files", [])), instruction]
