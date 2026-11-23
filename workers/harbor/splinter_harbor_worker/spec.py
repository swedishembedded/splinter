# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements execution backends that refuse an unsafe or
# unsupported request before they spend any compute on it, for its clients. If
# your team needs expertise in safe agent execution infrastructure then you
# can procure our services by sending an email to info@swedishembedded.com.
"""The ``splinter.execution/1`` specification, as the worker reads it.

The controller's contract is the JSON Schema ``schemas/factory/splinter.execution.v1.schema.json``; this module checks
the same invariants and what only this worker can know: the agents, sandbox providers and protocols it implements, and
that every variable a specification names is set in the worker's environment. A specification the worker cannot honour
is refused with the line it came from, before Harbor starts, never weakened.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from splinter_harbor_worker.errors import UsageError

EXECUTION_SCHEMA = "splinter.execution/1"
AGENTS = ("sven", "claude-code", "codex", "oracle", "nop")
# `auto`: the strongest the machine has that honours the specification (sandbox/choice.py).
PROVIDERS = ("auto", "docker", "userns", "confined", "proot")
PROTOCOLS = ("openai", "anthropic")
# A literal under a name like these is a secret written into a specification that is stored and shown.
_CREDENTIAL_WORDS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
_DIGEST = re.compile(r"^(blake3|sha256):[0-9a-f]{64}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def credential_like(name: str) -> bool:
    """Whether a variable name reads as a credential (the controller's own rule)."""
    upper = name.upper()
    return any(word in upper for word in _CREDENTIAL_WORDS)


def _plain_relative(path: str) -> bool:
    parts = path.split("/")
    return bool(path) and not path.startswith("/") and all(p not in ("", ".", "..") for p in parts)


class _Strict(BaseModel):
    # An unknown field is a contract mismatch with the controller; failing here is cheaper than guessing.
    model_config = ConfigDict(extra="forbid", frozen=True)


class SlotKey(_Strict):
    pass_: str = Field(alias="pass")
    task: str
    slot: int = Field(ge=0)

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    def label(self) -> str:
        return f"{self.pass_}/{self.task}/{self.slot}"

    def as_json(self) -> dict:
        return {"pass": self.pass_, "task": self.task, "slot": self.slot}


class TaskRef(_Strict):
    id: str
    name: str
    digest: str
    family: str | None = None
    split: str | None = None


class FileRef(_Strict):
    path: str
    digest: str

    @field_validator("path")
    @classmethod
    def _path(cls, value: str) -> str:
        if not _plain_relative(value):
            raise ValueError(f"path {value!r} is not a plain relative path")
        return value

    @field_validator("digest")
    @classmethod
    def _digest(cls, value: str) -> str:
        if not _DIGEST.match(value):
            raise ValueError(f"{value!r} is not a blake3:<hex> or sha256:<hex> digest")
        return value


class EnvValue(_Strict):
    literal: str | None = None
    from_env: str | None = None

    @field_validator("from_env")
    @classmethod
    def _name(cls, value: str | None) -> str | None:
        if value is not None and not _ENV_NAME.match(value):
            raise ValueError(f"{value!r} is not an environment variable name")
        return value

    def model_post_init(self, _context) -> None:
        if (self.literal is None) == (self.from_env is None):
            raise ValueError("an environment value is exactly one of literal and from_env")


class AgentInvocation(_Strict):
    name: str
    version: str | None = None
    args: list[str] = []
    files: list[FileRef] = []
    env: dict[str, EnvValue] = {}
    system_prompt: FileRef | None = None
    harness: FileRef | None = None  # a sven harness run in place of the agent's conversation


# The agent's own headers the gateway passes on to an Anthropic upstream; a connection may drop any of them.
FORWARDED_HEADERS = ("anthropic-version", "anthropic-beta")


class ModelBinding(_Strict):
    connection: str
    model: str
    protocol: Literal["openai", "anthropic"]
    url: str | None = None
    token_env: str | None = None
    drop_headers: list[str] = []

    @field_validator("drop_headers")
    @classmethod
    def _droppable(cls, value: list[str]) -> list[str]:
        unknown = [name for name in value if name not in FORWARDED_HEADERS]
        if unknown:
            raise ValueError(f"the gateway forwards only {', '.join(FORWARDED_HEADERS)}; it cannot drop {unknown}")
        return value


class Mount(_Strict):
    source: str
    digest: str
    target: str
    read_only: bool

    @field_validator("source")
    @classmethod
    def _source(cls, value: str) -> str:
        if not _plain_relative(value):
            raise ValueError(f"path {value!r} is not a plain relative path")
        return value

    @field_validator("target")
    @classmethod
    def _target(cls, value: str) -> str:
        if not value.startswith("/") or ".." in value.split("/"):
            raise ValueError(f"mount target {value!r} is not an absolute path")
        return value


class SandboxSpec(_Strict):
    provider: str
    image: str | None = None
    cpus: int | None = None
    memory_mib: int | None = None
    timeout_seconds: int | None = None
    network: Literal["model-gateway-only", "none", "open"] = "model-gateway-only"
    mounts: list[Mount] = []
    # The least a provider must give: strong (process and filesystem isolation), confined (filesystem writes and
    # network confined by the kernel) or any.
    isolation: Literal["strong", "confined", "any"] = "any"


class RunBudget(_Strict):
    completion_tokens: int
    turns: int


class ExecutionSpec(_Strict):
    schema_: str = Field(alias="schema")
    slot: SlotKey
    attempt: int = Field(ge=0)
    task: TaskRef
    task_dir: str
    agent: AgentInvocation
    model: ModelBinding | None = None
    sandbox: SandboxSpec
    seed: int
    budget: RunBudget | None = None
    timeout_policy: Literal["graded", "excluded"] = "graded"

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    @field_validator("task_dir")
    @classmethod
    def _task_dir(cls, value: str) -> str:
        if not _plain_relative(value):
            raise ValueError(f"path {value!r} is not a plain relative path")
        return value

    @property
    def key(self) -> tuple[str, str, int, int]:
        """The attempt's identity: slot and physical attempt."""
        return (self.slot.pass_, self.slot.task, self.slot.slot, self.attempt)

    def refusal(self, environ: Mapping[str, str], check_environment: bool = True) -> str | None:
        """Why the worker cannot run this specification, or ``None``. ``environ`` is the worker's environment."""
        if self.schema_ != EXECUTION_SCHEMA:
            return f"schema {self.schema_!r} is not {EXECUTION_SCHEMA}"
        if self.agent.name not in AGENTS:
            return f"unknown agent {self.agent.name!r}; this worker runs: {', '.join(AGENTS)}"
        if self.sandbox.provider not in PROVIDERS:
            return f"unknown sandbox provider {self.sandbox.provider!r}; this worker runs: {', '.join(PROVIDERS)}"
        if self.task.id != self.slot.task:
            return f"the slot names task {self.slot.task} and the specification carries task {self.task.id}"
        for name, value in self.agent.env.items():
            if value.literal is not None and credential_like(name):
                return (f"environment variable {name} looks like a credential; "
                        "name the variable it is read from (from_env) instead of a literal")
            if check_environment and value.from_env is not None and value.from_env not in environ:
                return f"environment variable {value.from_env} (for {name}) is not set in the worker's environment"
        if self.model is not None:
            if self.model.url is None:
                return f"model connection {self.model.connection!r} has no url; the gateway needs an endpoint"
            if check_environment and self.model.token_env is not None and not environ.get(self.model.token_env):
                return f"model token variable {self.model.token_env} is not set in the worker's environment"
        if self.agent.harness is not None and self.agent.name != "sven":
            return f"agent {self.agent.name} cannot run harness {self.agent.harness.path}: only sven runs a harness"
        if self.agent.harness is not None and self.model is None:
            return "a harness is told its model, and the specification names none"
        if self.agent.name in ("oracle", "nop") and (self.agent.args or self.agent.files):
            return f"agent {self.agent.name} takes no arguments or files"
        return None


def parse_specs(text: str, environ: Mapping[str, str] | None = None, source: str = "specs",
                check_environment: bool = True) -> list[ExecutionSpec]:
    """Every specification of a JSON-lines text, each validated and checked against what the worker can run;
    duplicates of one (slot, attempt) are refused. The error names the line."""
    environ = os.environ if environ is None else environ
    specs: list[ExecutionSpec] = []
    seen: dict[tuple, int] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{source}:{number}"
        try:
            spec = ExecutionSpec.model_validate(json.loads(line))
        except json.JSONDecodeError as exc:
            raise UsageError(f"{where}: not JSON: {exc}") from None
        except ValidationError as exc:
            raise UsageError(f"{where}: not a splinter.execution/1 specification: {_summary(exc)}") from None
        if reason := spec.refusal(environ, check_environment):
            raise UsageError(f"{where}: {reason}")
        if spec.key in seen:
            raise UsageError(f"{where}: attempt {spec.attempt} of slot {spec.slot.label()} repeats line {seen[spec.key]}")
        seen[spec.key] = number
        specs.append(spec)
    if not specs:
        raise UsageError(f"{source}: holds no specification")
    return specs


def load_specs(path: Path, environ: Mapping[str, str] | None = None,
               check_environment: bool = True) -> list[ExecutionSpec]:
    try:
        text = path.read_text()
    except OSError as exc:
        raise UsageError(f"cannot read {path}: {exc}") from None
    return parse_specs(text, environ, str(path), check_environment)


def _summary(error: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in error.errors()[:5])
