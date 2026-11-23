# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for running third-party coding
# agents the same way under the same limits so their results can be compared,
# for its clients. If your team needs expertise in fair agent comparison and
# confinement then you can procure our services by sending an email to
# info@swedishembedded.com.
"""The shared shape of the worker's agent adapters.

An adapter is a Harbor agent whose behaviour is given entirely by the ``splinter`` keyword argument the worker builds from
the ``ExecutionSpec`` (``AgentConfig.kwargs``, written to the trial's ``config.json``, so never a secret):

``args``           the agent's own arguments, passed as an argv, never through a shell;
``env``            the variables the agent gets, each a literal or the name of a variable of the worker's own
                   environment, resolved when the trial starts so the value never reaches a file;
``files``          configuration files the arguments refer to, uploaded below ``/splinter/files``; an argument
                   equal to a file's path (or ``--option=path``) names the uploaded copy;
``files_root``     where the worker finds them;
``system_prompt``  text appended to the agent's own system prompt;
``gateway_url``, ``protocol``, ``model``  the model, when the worker routes it through its gateway;
``timeout_sec``    the agent's wall-clock limit.

Per trial an adapter gives the agent an isolated HOME (no user configuration leaks in), the trial's gateway token as its
only model credential (never a provider key), and the task workspace as its working directory. The agent's standard
output, standard error and exit code always land in ``/logs/agent``; a non-zero exit never fails the trial by itself
(whether it matters is the verifier's call), and the worker's evidence reads the exit code and the agent's own markers.
"""

from __future__ import annotations

import json
import os
import shlex
import tempfile
from abc import abstractmethod
from pathlib import Path

from harbor.agents.installed.base import BaseInstalledAgent

from splinter_harbor_worker import runtime
from splinter_harbor_worker.guidance import GUIDANCE_FILE, guidance_record, restore_human_marks

TOOLS_DIR = "/splinter/bin"
FILES_DIR = "/splinter/files"
# The host's dynamic loader and libraries, which sandboxes that bring their own libc get beside the tools.
HOST_LOADER = "/splinter/host/ld.so --library-path /splinter/host/lib"
HOST_CERTIFICATES = "/splinter/host/certs/ca-certificates.crt"
IMAGE_CERTIFICATES = ("/etc/ssl/certs/ca-certificates.crt", "/etc/pki/tls/certs/ca-bundle.crt", "/etc/ssl/cert.pem")
HOME = "/tmp/splinter-agent-home"
TOKEN_FILE = f"{HOME}/gateway-token"
# The agent stops itself this long before the task's agent timeout, so a run that runs out of time still writes its
# record instead of being killed by Harbor with nothing; on a short timeout the margin is half of it.
TIMEOUT_MARGIN_SEC = 120


def agent_run_timeout(agent_timeout_sec: float) -> int:
    """The agent's own time limit for a task whose agent timeout is ``agent_timeout_sec``."""
    return max(1, int(agent_timeout_sec - min(TIMEOUT_MARGIN_SEC, agent_timeout_sec / 2)))


def resolve_env(declared: dict[str, dict], environ: dict[str, str] | None = None) -> dict[str, str]:
    """The agent's variables: literals as given, ``from_env`` values read from the worker's environment now."""
    environ = os.environ if environ is None else environ
    resolved = {}
    for name, value in declared.items():
        if value.get("from_env") is not None:
            if value["from_env"] not in environ:
                raise RuntimeError(f"environment variable {value['from_env']} (for {name}) is not set in the worker")
            resolved[name] = environ[value["from_env"]]
        else:
            resolved[name] = value["literal"]
    return resolved


def sandbox_args(args: list[str], files: list[str]) -> list[str]:
    """``args`` with every reference to a configuration file (the argument, or the value of ``--option=``) replaced
    by the path of its uploaded copy."""
    uploaded = {path: f"{FILES_DIR}/{path}" for path in files}
    result = []
    for arg in args:
        option, equals, value = arg.partition("=")
        if arg in uploaded:
            result.append(uploaded[arg])
        elif equals and value in uploaded:
            result.append(f"{option}={uploaded[value]}")
        else:
            result.append(arg)
    return result


class SplinterAgent(BaseInstalledAgent):
    """A Harbor agent configured by a ``splinter`` keyword argument (see the module docstring)."""

    BINARY: str  # the executable's name below TOOLS_DIR
    TRACE_PATH: str | None = None  # where the agent keeps its trajectory; an agent without one cannot continue a trial
    _command: str  # how the sandbox starts it: directly, or through the host's loader
    _certificates: dict[str, str]

    def __init__(self, *args, splinter: dict, human_guided: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._settings = splinter
        # Set by ``splinter-harbor-worker play``: a person types into the agent. Harbor writes it to the trial's
        # config (``config.agent.kwargs.human_guided``), which is how a reader tells a guided trial from an automatic one.
        self._human_guided = bool(human_guided)
        self._command = f"{TOOLS_DIR}/{self.BINARY}"
        self._certificates = {}

    def get_version_command(self) -> str | None:
        return f"{self._command} --version"

    # -- what a concrete agent says ------------------------------------------------------------------------------

    @abstractmethod
    def argv(self, instruction: str, run_timeout_sec: int) -> list[str]:
        """The agent's command line (without a shell); starts with ``self._command`` split into words."""

    def interactive_argv(self, instruction: str | None, run_timeout_sec: int, continuing: bool) -> list[str]:
        """The agent's command line when a person drives it on a terminal. ``instruction`` is the task, sent as the first
        message (``None``: none, the conversation goes on); ``continuing``: its trajectory is already at ``TRACE_PATH``."""
        raise NotImplementedError(f"{self.BINARY} cannot be played interactively")

    def agent_env(self, token: str | None) -> dict[str, str]:
        """Variables the agent needs of the gateway; ``token`` is this trial's, ``None`` without a gateway."""
        return {}

    def home_files(self) -> dict[str, str]:
        """Files (path below the agent's HOME -> text) the agent reads its model configuration from."""
        return {}

    # -- Harbor's lifecycle ---------------------------------------------------------------------------------------

    async def install(self, environment) -> None:
        """The binary is staged by the worker (a mount at ``/splinter/bin``), not installed: the first of its
        two ways of starting that works, directly or through the host's loader."""
        own = " || ".join(f"test -e {path}" for path in IMAGE_CERTIFICATES)
        result = await environment.exec(command=f"{{ {own}; }} || ! test -e {HOST_CERTIFICATES}")
        if getattr(result, "return_code", 0) != 0:
            self._certificates = {"SSL_CERT_FILE": HOST_CERTIFICATES}
        path = f"{TOOLS_DIR}/{self.BINARY}"
        for command in (path, f"{HOST_LOADER} {path}"):
            result = await environment.exec(command=f"{command} --version")
            if getattr(result, "return_code", 1) == 0:
                self._command = command
                return
        raise RuntimeError(f"the sandbox has no runnable {self.BINARY}; the worker stages it at {path}")

    async def run(self, instruction: str, environment, context) -> None:
        settings = self._settings
        trial = Path(self.logs_dir).parent.name
        timeout = int(settings["timeout_sec"])
        run_timeout = agent_run_timeout(timeout)
        gateway_url = settings.get("gateway_url")
        token = runtime.gateway().token_for(trial) if gateway_url else None

        env = {
            **self._certificates,
            "HOME": HOME,
            "XDG_CONFIG_HOME": f"{HOME}/.config",
            "XDG_DATA_HOME": f"{HOME}/.local/share",
            "XDG_STATE_HOME": f"{HOME}/.local/state",
            "XDG_CACHE_HOME": f"{HOME}/.cache",
            # The attempt's seed is the agent's to use as its own flags say (a model that cannot honour a sampling seed
            # makes Sven refuse the run); the worker only tells it.
            "SPLINTER_SEED": str(settings["seed"]),
            **resolve_env(settings.get("env", {})),
            **self.agent_env(token),
        }
        await environment.exec(command=f"mkdir -p {HOME} {FILES_DIR} /logs/agent")
        await self._upload_files(environment)
        with tempfile.TemporaryDirectory() as scratch:
            for relative, text in self.home_files().items():
                local = Path(scratch) / "file"
                local.write_text(text)
                await environment.exec(command=f"mkdir -p {shlex.quote(str(Path(HOME, relative).parent))}")
                await environment.upload_file(local, f"{HOME}/{relative}")
        if token is not None:
            # The token goes into a private file the agent's configuration names; passed as the environment of this
            # one command, it is on no command line.
            await environment.exec(command=f'umask 077 && printf "%s" "$SPLINTER_GATEWAY_TOKEN" > {TOKEN_FILE}',
                                   env={"SPLINTER_GATEWAY_TOKEN": token})
        if self._human_guided:
            await self._run_guided(instruction, environment, env, run_timeout)
            return
        command = (
            'cd "${SPLINTER_WORKSPACE:-/workspace}" && '
            f"{shlex.join(self.argv(instruction, run_timeout))} "
            "</dev/null >/logs/agent/stdout.txt 2>/logs/agent/stderr.txt; "
            'printf "%s" "$?" > /logs/agent/exit-code.txt'
        )
        # The agent's command is a shell line only to redirect its output; its words are quoted.
        await self.exec_as_agent(environment, command=command, env=env)

    async def _run_guided(self, instruction: str, environment, env: dict[str, str], run_timeout: int) -> None:
        """The agent on a terminal of the sandbox's own that the person's seat watches and types into; ends when the
        person quits the agent. The agent's screen is the person's, so nothing of it is redirected to a log; its exit
        code and ``guidance.json`` (``guidance.py``) are written as for an automatic run."""
        settings = self._settings
        if not hasattr(environment, "exec_terminal"):
            raise RuntimeError("a human-guided trial needs a terminal in its sandbox: use the userns or proot provider")
        continue_trace = settings.get("continue_trace")
        if continue_trace:
            # The earlier conversation becomes the start of this trial's own trajectory, which the agent extends.
            if self.TRACE_PATH is None:
                raise RuntimeError(f"{self.BINARY} leaves no trajectory, so a conversation cannot be continued")
            await environment.upload_file(Path(continue_trace), self.TRACE_PATH)
        parts = self.interactive_argv(None if continue_trace else instruction, run_timeout, bool(continue_trace))
        command = ('cd "${SPLINTER_WORKSPACE:-/workspace}" && '
                   f"{shlex.join(parts)}; " 'printf "%s" "$?" > /logs/agent/exit-code.txt')
        seat = runtime.seat()
        try:
            async with seat.session() as keys:
                await environment.exec_terminal(command, output=seat.show, keys=keys, size=seat.size(),
                                                env={**env, "TERM": seat.term})
        finally:  # also when the agent's time limit cut the session short: the person's turns so far are recorded
            self._write_guidance(seat.kind)

    def _write_guidance(self, seat_kind: str) -> None:
        trace = Path(self.logs_dir) / "trajectory.json"
        try:
            document = json.loads(trace.read_text()) if trace.is_file() else None
        except json.JSONDecodeError:  # cut off while the agent wrote it
            document = None
        settings = self._settings
        restored = 0
        if document is not None and settings.get("continue_trace"):
            restored = restore_human_marks(document, json.loads(Path(settings["continue_trace"]).read_text()))
            if restored:
                trace.write_text(json.dumps(document, indent=2) + "\n")
        record = guidance_record(document, seat_kind, settings.get("continued_from"), restored)
        (Path(self.logs_dir) / GUIDANCE_FILE).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")

    async def _upload_files(self, environment) -> None:
        root = Path(self._settings.get("files_root") or ".").resolve()
        for relative in self._settings.get("files", []):
            source = (root / relative).resolve()
            if not source.is_relative_to(root) or not source.is_file():
                raise RuntimeError(f"agent file {relative} is not a file below {root}")
            target = f"{FILES_DIR}/{relative}"
            await environment.exec(command=f"mkdir -p {shlex.quote(str(Path(target).parent))}")
            await environment.upload_file(source, target)
