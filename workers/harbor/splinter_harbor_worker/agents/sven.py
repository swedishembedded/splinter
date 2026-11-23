# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for running a coding agent over
# many tasks with every model request accounted for, for its clients. If your
# team needs expertise in agent evaluation and training-data collection then
# you can procure our services by sending an email to info@swedishembedded.com.
"""Harbor adapter for Sven: the build under test runs headless inside the task sandbox.

The Sven binary is staged by the worker at ``/splinter/bin/sven`` (see ``agents.base``). Sven runs from the task
workspace with an isolated HOME holding the configuration that points it at the worker's gateway, the trial's gateway
token as its API key file, and the spec's own arguments. Its trace (ATIF ``trajectory.json``) and its own request log
(``requests.jsonl``) land in ``/logs/agent``, where the worker's evidence reads them.
"""

from __future__ import annotations

import json

import yaml

from splinter_harbor_worker.agents.base import FILES_DIR, TOKEN_FILE, SplinterAgent, sandbox_args

TRACE = "/logs/agent/trajectory.json"
# Every model request as sent, so a fault can be replayed (`sven replay`).
REQUEST_LOG = "/logs/agent/requests.jsonl"


class SvenAgent(SplinterAgent):
    SUPPORTS_ATIF: bool = True
    BINARY = "sven"
    TRACE_PATH = TRACE

    @staticmethod
    def name() -> str:
        return "sven"

    def parse_version(self, stdout: str) -> str:
        return stdout.strip().removeprefix("sven ").strip()

    def home_files(self) -> dict[str, str]:
        settings = self._settings
        if not settings.get("gateway_url"):
            return {}  # the agent's own provider configuration, from its arguments
        # `agent.stream_idle_timeout_secs` stays inside the run's time budget: a local model can stay silent through
        # the prefill of a long prompt for longer than Sven's default, so only a stuck run should trip it.
        config = {
            "providers": {"gateway": {"name": settings["protocol"], "base_url": f"{settings['gateway_url'].rstrip('/')}/v1",
                                      "api_key_file": TOKEN_FILE}},
            "model": {"provider": "gateway", "name": settings["model"]},
            "agent": {"stream_idle_timeout_secs": max(60, int(settings["timeout_sec"]) - 60)},
        }
        return {".config/sven/config.yaml": yaml.safe_dump(config, sort_keys=False)}

    def argv(self, instruction: str, run_timeout_sec: int) -> list[str]:
        settings = self._settings
        if settings.get("harness"):
            return self.harness_argv(instruction, run_timeout_sec)
        parts = [*self._command.split(), "--headless", "--output-format", "compact",
                 "--run-timeout", str(run_timeout_sec), "--output-trace", TRACE, "--record-requests", REQUEST_LOG]
        if settings.get("system_prompt"):
            parts += ["--append-system-prompt", settings["system_prompt"]]
        parts += sandbox_args(settings.get("args", []), settings.get("files", []))
        parts.append(instruction)
        return parts

    def harness_argv(self, instruction: str, run_timeout_sec: int) -> list[str]:
        """``sven harness run``: the spec's harness in place of the conversation, given the instruction as its ``task``
        input, on the trial's workspace, its model reached through the gateway with the trial's token file (the
        harness reads no sven configuration), suspended by its own timeout before the task's. Its ATIF trajectory
        lands where the conversation's would; it records no model requests of its own."""
        settings = self._settings
        parts = [*self._command.split(), "harness", "run", f"{FILES_DIR}/{settings['harness']}",
                 "--input", json.dumps({"task": instruction}), "--workspace", ".", "--approvals", "auto",
                 "--timeout", str(run_timeout_sec), "--trajectory", TRACE]
        if settings.get("gateway_url"):
            parts += ["--model", f"{settings['protocol']}/{settings['model']}",
                      "--model-endpoint", f"{settings['gateway_url'].rstrip('/')}/v1", "--api-key-file", TOKEN_FILE]
        parts += sandbox_args(settings.get("args", []), settings.get("files", []))
        parts.append("--json")
        return parts

    def interactive_argv(self, instruction: str | None, run_timeout_sec: int, continuing: bool) -> list[str]:
        """Sven's own terminal UI (``--interactive``): the instruction is the first message and the person types the
        rest; the turns they type are marked ``extra.sven.origin: human`` in the trace. Continuing a trial gives Sven
        its earlier trajectory as the trace to load and extend (``--trace``, the one persistence flag of the TUI), and
        sends no instruction: the conversation so far already holds it."""
        settings = self._settings
        parts = [*self._command.split(), "--interactive", "--run-timeout", str(run_timeout_sec),
                 *(["--trace", TRACE] if continuing else ["--output-trace", TRACE]), "--record-requests", REQUEST_LOG]
        if settings.get("system_prompt"):
            parts += ["--append-system-prompt", settings["system_prompt"]]
        parts += sandbox_args(settings.get("args", []), settings.get("files", []))
        if instruction is not None:
            parts.append(instruction)
        return parts
