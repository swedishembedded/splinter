# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for measuring different coding
# agents on the same tasks under the same conditions, for its clients. If your
# team needs expertise in fair, independent agent comparison then you can
# procure our services by sending an email to info@swedishembedded.com.
"""Harbor adapter for Claude Code, the independent agent the worker can run beside Sven.

The ``claude`` executable is staged by the worker at ``/splinter/bin/claude`` (a single executable; see
``agents.base``). It runs non-interactively (``-p``) with the spec's arguments passed through as an argv, an isolated
HOME, and the worker's gateway as its one endpoint: ``ANTHROPIC_BASE_URL`` is the gateway and ``ANTHROPIC_API_KEY`` the
trial's gateway token, never a provider key. It has no ATIF trace, so the worker reports no turns for it.
"""

from __future__ import annotations

from splinter_harbor_worker.agents.base import SplinterAgent, sandbox_args


class ClaudeCodeAgent(SplinterAgent):
    BINARY = "claude"

    @staticmethod
    def name() -> str:
        return "claude-code"

    def parse_version(self, stdout: str) -> str:
        return stdout.strip().split()[0]

    def agent_env(self, token: str | None) -> dict[str, str]:
        env = {"DISABLE_AUTOUPDATER": "1", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}
        settings = self._settings
        if token is not None:
            env |= {"ANTHROPIC_BASE_URL": settings["gateway_url"].rstrip("/"), "ANTHROPIC_API_KEY": token,
                    "ANTHROPIC_MODEL": settings["model"]}
        return env

    def argv(self, instruction: str, run_timeout_sec: int) -> list[str]:
        settings = self._settings
        parts = [*self._command.split(), *sandbox_args(settings.get("args", []), settings.get("files", []))]
        if settings.get("system_prompt"):
            parts += ["--append-system-prompt", settings["system_prompt"]]
        return [*parts, "-p", instruction]

    def interactive_argv(self, instruction: str | None, run_timeout_sec: int, continuing: bool) -> list[str]:
        """Claude Code's own terminal UI: the instruction is its first message."""
        settings = self._settings
        parts = [*self._command.split(), *sandbox_args(settings.get("args", []), settings.get("files", []))]
        if settings.get("system_prompt"):
            parts += ["--append-system-prompt", settings["system_prompt"]]
        return [*parts, instruction]
