# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions that turn a list of agent runs into
# sandboxed, network-confined, reproducible trials, for its clients. If your
# team needs expertise in confining agent execution then you can procure our
# services by sending an email to info@swedishembedded.com.
"""From ``ExecutionSpec``s to one Harbor job: refuse what cannot be honoured, stage the tasks, build the trials.

Harbor decides a trial's network from the task's own ``task.toml`` and the job's allowlists, so the worker never runs a
pinned task directory as it is: ``stage_task`` copies it with ``[environment].network_mode`` set to what the
specification says (and its image, when it names one). The model gateway is then opened to the agent phase only
(Harbor's per-phase allowlist), and only on the providers that can enforce an allowlist.

Network modes by provider (a combination a provider cannot enforce is refused before anything runs, never weakened):

=================  =====================================  ===========================================
mode               docker                                 userns                      proot
=================  =====================================  ===========================================
open               public                                 public                      public
model-gateway-only no-network + gateway in the agent      no-network + gateway in     refused
                   phase (Harbor's egress control)        the agent phase (per port)
none               no-network (egress control)            no-network                  refused
=================  =====================================  ===========================================
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from harbor.models.job.config import JobConfig, RetryConfig
from harbor.models.task.config import NetworkMode, TaskConfig as TaskToml
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig

from splinter_harbor_worker.agents.base import TOOLS_DIR
from splinter_harbor_worker.agents.claude_code import ClaudeCodeAgent
from splinter_harbor_worker.agents.codex import CodexAgent
from splinter_harbor_worker.agents.sven import SvenAgent
from splinter_harbor_worker.errors import UsageError
from splinter_harbor_worker.fscopy import copy_tree
from splinter_harbor_worker.guidance import HUMAN_GUIDED_KWARG
from splinter_harbor_worker.sandbox.choice import STRENGTH, choose
from splinter_harbor_worker.spec import ExecutionSpec

JOB_NAME = "job"
SVEN_BIN_ENV = "SPLINTER_SVEN_BIN"
CLAUDE_BIN_ENV = "SPLINTER_CLAUDE_BIN"
CODEX_BIN_ENV = "SPLINTER_CODEX_BIN"
GATEWAY_HOST_ENV = "SPLINTER_GATEWAY_HOST"
DEFAULT_GATEWAY_HOST = "127.0.0.1"
USERNS_ENVIRONMENT = "splinter_harbor_worker.sandbox.userns:UserNamespaceEnvironment"
PROOT_ENVIRONMENT = "splinter_harbor_worker.sandbox.proot:ProotEnvironment"
CONFINED_ENVIRONMENT = "splinter_harbor_worker.sandbox.confined:ConfinedEnvironment"
AGENT_CLASSES = {
    "sven": "splinter_harbor_worker.agents.sven:SvenAgent",
    "claude-code": "splinter_harbor_worker.agents.claude_code:ClaudeCodeAgent",
    "codex": "splinter_harbor_worker.agents.codex:CodexAgent",
}
# The executable each staged agent runs, as its adapter names it, and the variable that may point at it.
AGENT_BINARIES = {"sven": (SvenAgent.BINARY, SVEN_BIN_ENV), "claude-code": (ClaudeCodeAgent.BINARY, CLAUDE_BIN_ENV),
                  "codex": (CodexAgent.BINARY, CODEX_BIN_ENV)}
_IPV4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def trial_name(spec: ExecutionSpec) -> str:
    """The Harbor trial directory of an attempt: readable task name, and a digest of the attempt's identity, so the
    same specification is always the same trial (resume, reconcile) and two attempts never share one."""
    identity = f"{spec.slot.label()}#{spec.attempt}"
    task = re.sub(r"[^A-Za-z0-9._-]+", "-", spec.slot.task.rsplit("/", 1)[-1])[:32].strip("-_.") or "task"
    return f"{task}__{hashlib.sha256(identity.encode()).hexdigest()[:10]}"


def report_path(out: Path, spec: ExecutionSpec) -> Path:
    """``attempts/<pass>/<task id with / as __>/<slot>-<attempt>.json``."""
    return (out / "attempts" / spec.slot.pass_ / spec.slot.task.replace("/", "__")
            / f"{spec.slot.slot}-{spec.attempt}.json")


def effective_network(spec: ExecutionSpec) -> str:
    """The task baseline the specification asks for: ``public`` or ``no-network``."""
    return "public" if spec.sandbox.network == "open" else "no-network"


def staged_task_dir(out: Path, spec: ExecutionSpec) -> Path:
    name = spec.slot.task.replace("/", "__")
    image = hashlib.sha256((spec.sandbox.image or "").encode()).hexdigest()[:8] if spec.sandbox.image else "task"
    return out / "staged" / "tasks" / f"{name}.{effective_network(spec)}.{image}"


def tool_dir(out: Path, agent: str) -> Path:
    return out / "staged" / "bin" / agent


# -- refusals -----------------------------------------------------------------------------------------------------


def gateway_host() -> str:
    return os.environ.get(GATEWAY_HOST_ENV, DEFAULT_GATEWAY_HOST)


def network_refusal(spec: ExecutionSpec, host: str) -> str | None:
    """Why this provider cannot enforce the network the specification asks for, or ``None``."""
    provider, network, routed = spec.sandbox.provider, spec.sandbox.network, spec.model is not None
    if network == "none" and routed:
        return "network 'none' leaves the agent no way to reach its model; the specification binds a model"
    if network == "model-gateway-only" and not routed and spec.agent.name in AGENT_CLASSES:
        return (f"network 'model-gateway-only' needs a model binding to route; agent {spec.agent.name!r} "
                "has none, so it could reach nothing")
    if routed and network == "model-gateway-only":
        if provider in ("userns", "confined") and not _IPV4.match(host):
            return f"the {provider} provider admits IPv4 addresses only; the gateway host {host!r} is not one"
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(root: Path, relative: str, what: str, where: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise UsageError(f"{where}: {what} {relative!r} leaves {root}")
    return path


def _check_file(root: Path, ref, what: str, where: str) -> Path:
    path = _inside(root, ref.path, what, where)
    if not path.is_file():
        raise UsageError(f"{where}: {what} {ref.path!r} is not a file below {root}")
    if ref.digest.startswith("sha256:") and ref.digest.removeprefix("sha256:") != _sha256_file(path):
        raise UsageError(f"{where}: {what} {ref.path!r} does not match its digest {ref.digest}")
    return path


def agent_binary(agent: str) -> Path:
    name, variable = AGENT_BINARIES[agent]
    found = os.environ.get(variable) or shutil.which(name)
    if not found or not Path(found).is_file():
        raise UsageError(f"agent {agent!r} needs the {name} executable: put it on PATH or name it in {variable}")
    return Path(found).resolve()


def _host_version(binary: Path) -> str:
    result = subprocess.run([str(binary), "--version"], capture_output=True, text=True, check=False, timeout=60)
    if result.returncode != 0:
        raise UsageError(f"{binary} --version failed: {result.stderr.strip()[-300:]}")
    words = result.stdout.split()
    return words[-1] if words else ""


@dataclass(frozen=True)
class Preflight:
    """What refusing-or-accepting a batch learned: the agent binaries to stage."""

    binaries: dict[str, Path]
    gateway_host: str


def resolve_sandboxes(specs: list[ExecutionSpec], allowed: tuple[str, ...] = STRENGTH) -> list[ExecutionSpec]:
    """``specs`` with each sandbox's provider the one it runs on here (``choice.choose``): ``auto`` becomes the
    strongest available provider of ``allowed`` that honours it; a named one is checked. Refuses what no provider here
    can run."""
    resolved = []
    for spec in specs:
        try:
            provider = choose(spec.sandbox, allowed=allowed)
        except UsageError as exc:
            raise UsageError(f"{spec.slot.label()}#{spec.attempt}: {exc}") from None
        resolved.append(spec.model_copy(update={"sandbox": spec.sandbox.model_copy(update={"provider": provider})}))
    return resolved


def preflight(specs: list[ExecutionSpec], tasks_root: Path, files_root: Path) -> Preflight:
    """Refuse every specification the worker cannot honour, before Harbor starts and before anything is staged."""
    host = gateway_host()
    binaries: dict[str, Path] = {}
    for spec in specs:
        where = f"{spec.slot.label()}#{spec.attempt}"
        task = _inside(tasks_root, spec.task_dir, "task directory", where)
        if not (task / "task.toml").is_file():
            raise UsageError(f"{where}: task directory {spec.task_dir!r} holds no task.toml below {tasks_root}")
        if reason := network_refusal(spec, host):
            raise UsageError(f"{where}: {reason}")
        for mount in spec.sandbox.mounts:
            if not _inside(files_root, mount.source, "mount source", where).exists():
                raise UsageError(f"{where}: mount source {mount.source!r} does not exist below {files_root}")
        for ref in spec.agent.files:
            _check_file(files_root, ref, "agent file", where)
        if spec.agent.system_prompt is not None:
            _check_file(files_root, spec.agent.system_prompt, "system prompt", where)
        if spec.agent.harness is not None:
            _check_file(files_root, spec.agent.harness, "harness", where)
        if spec.agent.name in AGENT_BINARIES and spec.agent.name not in binaries:
            binaries[spec.agent.name] = agent_binary(spec.agent.name)
        if spec.agent.name == "sven" and spec.agent.version is not None:
            found = _host_version(binaries["sven"])
            if found != spec.agent.version:
                raise UsageError(f"{where}: the specification pins sven {spec.agent.version}; {binaries['sven']} "
                                 f"is {found}")
    return Preflight(binaries, host)


# -- staging ------------------------------------------------------------------------------------------------------


def stage_task(out: Path, tasks_root: Path, spec: ExecutionSpec) -> Path:
    """The task directory the trial runs: the pinned one with the specification's network and image written into its
    ``task.toml``. Rebuilt on every run, so the staged copy is always what this specification asks for."""
    source = (tasks_root / spec.task_dir).resolve()
    target = staged_task_dir(out, spec)
    if target.exists():
        shutil.rmtree(target)
    copy_tree(source, target)
    config = TaskToml.model_validate_toml((target / "task.toml").read_text())
    config.environment.network_mode = NetworkMode(effective_network(spec))
    config.environment.allowed_hosts = None
    if spec.sandbox.image:
        config.environment.docker_image = spec.sandbox.image
    # The pinned tasks are read-only (hard links of the run's pin), and the copy keeps their mode.
    (target / "task.toml").chmod(0o644)
    (target / "task.toml").write_text(config.model_dump_toml())
    return target


def stage_tools(out: Path, binaries: dict[str, Path]) -> None:
    """One directory per agent holding its executable, mounted read-only at ``/splinter/bin``: a hard link where
    the filesystem allows, else a copy, so a binary rebuilt on the host while a trial runs changes nothing."""
    for agent, binary in binaries.items():
        directory = tool_dir(out, agent)
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True)
        target = directory / AGENT_BINARIES[agent][0]
        try:
            os.link(binary, target)
        except OSError:
            shutil.copy2(binary, target)


def agent_timeout_sec(staged: Path, spec: ExecutionSpec) -> float:
    if spec.sandbox.timeout_seconds is not None:
        return float(spec.sandbox.timeout_seconds)
    return float(tomllib.loads((staged / "task.toml").read_text()).get("agent", {}).get("timeout_sec") or 600.0)


# -- trial configs ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GatewayAddress:
    host: str
    port: int

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True)
class Guided:
    """What makes a trial one a person plays (``splinter-harbor-worker play``): the agent runs interactively, the trial's
    configuration says ``human_guided``, and, when it continues an earlier conversation, where that trajectory is
    (``continue_trace``) and which trial it came from (``continued_from``, recorded in ``guidance.json``)."""

    continue_trace: Path | None = None
    continued_from: dict | None = None


def _agent_config(spec: ExecutionSpec, files_root: Path, gateway: GatewayAddress | None, timeout: float,
                  guided: Guided | None) -> AgentConfig:
    model_name = f"{spec.model.connection}/{spec.model.model}" if spec.model else None
    override = float(spec.sandbox.timeout_seconds) if spec.sandbox.timeout_seconds is not None else None
    hosts = [gateway.host] if gateway and spec.sandbox.network == "model-gateway-only" else []
    if spec.agent.name in ("oracle", "nop"):
        return AgentConfig(name=spec.agent.name, model_name=model_name, override_timeout_sec=override,
                           extra_allowed_hosts=hosts)
    prompt = None
    if spec.agent.system_prompt is not None:
        prompt = (files_root / spec.agent.system_prompt.path).read_text()
    settings = {
        "args": list(spec.agent.args),
        "env": {name: value.model_dump(exclude_none=True) for name, value in spec.agent.env.items()},
        "files": [ref.path for ref in spec.agent.files] + ([spec.agent.harness.path] if spec.agent.harness else []),
        "harness": spec.agent.harness.path if spec.agent.harness else None,
        "files_root": str(files_root),
        "system_prompt": prompt,
        "seed": spec.seed,
        "timeout_sec": timeout,
        "gateway_url": gateway.url if gateway and spec.model else None,
        "protocol": spec.model.protocol if spec.model else None,
        "model": spec.model.model if spec.model else None,
    }
    kwargs: dict = {"splinter": settings}
    if guided is not None:
        settings["continue_trace"] = str(guided.continue_trace) if guided.continue_trace else None
        settings["continued_from"] = guided.continued_from
        kwargs[HUMAN_GUIDED_KWARG] = True
    return AgentConfig(import_path=AGENT_CLASSES[spec.agent.name], model_name=model_name, override_timeout_sec=override,
                       extra_allowed_hosts=hosts, kwargs=kwargs)


def _mounts(spec: ExecutionSpec, files_root: Path, out: Path) -> list[dict]:
    mounts = [{"type": "bind", "source": str(_inside(files_root, m.source, "mount source", spec.slot.label())),
               "target": m.target, **({"read_only": True} if m.read_only else {})} for m in spec.sandbox.mounts]
    if spec.agent.name in AGENT_BINARIES:
        mounts.append({"type": "bind", "source": str(tool_dir(out, spec.agent.name)), "target": TOOLS_DIR,
                       "read_only": True})
    return mounts


def _environment_config(spec: ExecutionSpec, files_root: Path, out: Path, image_cache: Path,
                        gateway: GatewayAddress | None) -> EnvironmentConfig:
    provider = spec.sandbox.provider
    kwargs: dict = {}
    if provider != "docker":
        kwargs["images"] = str(image_cache)
        if gateway is not None:
            kwargs["allow_ports"] = gateway.port
    config: dict = {"mounts": _mounts(spec, files_root, out) or None, "kwargs": kwargs}
    if provider == "userns":
        config["import_path"] = USERNS_ENVIRONMENT
    elif provider == "proot":
        config["import_path"] = PROOT_ENVIRONMENT
    elif provider == "confined":
        config["import_path"] = CONFINED_ENVIRONMENT
    else:
        # Harbor's own Docker environment enforces the resource figures; the namespace and proot providers do not.
        config["override_cpus"] = spec.sandbox.cpus
        config["override_memory_mb"] = spec.sandbox.memory_mib
    return EnvironmentConfig(**config)


def unenforced_limits(spec: ExecutionSpec) -> list[str]:
    """The resource limits the specification names and the provider cannot enforce (reported, not silently dropped)."""
    if spec.sandbox.provider == "docker":
        return []
    return [name for name, value in (("cpus", spec.sandbox.cpus), ("memory_mib", spec.sandbox.memory_mib))
            if value is not None]


def trial_config(spec: ExecutionSpec, *, out: Path, staged: Path, files_root: Path, image_cache: Path,
                 gateway: GatewayAddress | None, job_id: UUID | None = None,
                 guided: Guided | None = None) -> TrialConfig:
    timeout = agent_timeout_sec(staged, spec)
    return TrialConfig(
        task=TaskConfig(path=staged),
        trial_name=trial_name(spec),
        trials_dir=out / "harbor" / JOB_NAME,
        agent=_agent_config(spec, files_root, gateway, timeout, guided),
        environment=_environment_config(spec, files_root, out, image_cache, gateway),
        job_id=job_id,
    )


def job_config(out: Path, staged: list[Path], concurrency: int) -> JobConfig:
    """The one native Harbor job: ``harbor view OUT/harbor`` opens it. Its trials are given directly (see ``job``), so
    the task list is only the distinct staged directories Harbor validates and caches."""
    return JobConfig(job_name=JOB_NAME, jobs_dir=out / "harbor", n_concurrent_trials=concurrency, quiet=True,
                     retry=RetryConfig(max_retries=0), tasks=[TaskConfig(path=p) for p in sorted(set(staged))])
