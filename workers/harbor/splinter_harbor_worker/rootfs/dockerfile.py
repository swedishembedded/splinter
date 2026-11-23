# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for reproducible container-image based
# build and test environments on hosts without a container daemon for its clients.
# If your team needs expertise in hermetic benchmark environments then you can
# procure our services by sending an email to info@swedishembedded.com.
"""Parse the Dockerfile subset that builds a single-stage image.

Supported: ``FROM`` (once), ``ARG``, ``ENV``, ``WORKDIR``, ``RUN`` (shell and exec form),
``COPY`` and ``ADD`` of local files, ``USER root``, and the metadata instructions that
do not change the filesystem (``LABEL``, ``EXPOSE``, ``CMD``, ``ENTRYPOINT``, ``VOLUME``,
``STOPSIGNAL``, ``HEALTHCHECK``). Anything else (multi-stage builds, ``COPY --from``,
``ADD`` of URLs, ``RUN --mount``, ``ONBUILD``, ``SHELL``, a non-root ``USER``) raises
``DockerfileError`` naming the line: an image that cannot be reproduced faithfully is
refused, never approximated.

Variables (``$NAME``, ``${NAME}``, ``${NAME:-default}``) are expanded from the ``ARG`` and
``ENV`` values in effect, in every instruction but ``RUN``, whose commands the shell
expands; ``RUN`` sees the build arguments and ``ENV`` values in its environment.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field

from splinter_harbor_worker.errors import UsageError


class DockerfileError(UsageError):
    pass


@dataclass(frozen=True)
class Step:
    kind: str  # RUN | COPY | WORKDIR
    line: int
    argv: tuple[str, ...] = ()  # RUN: the command to execute
    sources: tuple[str, ...] = ()  # COPY: paths in the build context
    destination: str = ""  # COPY / WORKDIR (as written, after expansion)
    env: tuple[tuple[str, str], ...] = ()  # RUN: ARG and ENV values in effect
    workdir: str = "/"  # RUN: the working directory


@dataclass(frozen=True)
class Plan:
    base: str
    steps: tuple[Step, ...]
    env: dict[str, str] = field(default_factory=dict)  # ENV instructions, in order
    workdir: str = "/"


_METADATA = {"LABEL", "EXPOSE", "CMD", "ENTRYPOINT", "VOLUME", "STOPSIGNAL", "HEALTHCHECK", "MAINTAINER"}
_VARIABLE = re.compile(r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}|(?P<bare>[A-Za-z_][A-Za-z0-9_]*))")


def _logical_lines(text: str) -> list[tuple[int, str]]:
    lines, pending, start = [], "", 0
    for number, raw in enumerate(text.splitlines(), 1):
        stripped = raw.strip()
        if not pending and (not stripped or stripped.startswith("#")):
            continue
        if pending and stripped.startswith("#"):
            continue  # a comment line inside a continued instruction
        if not pending:
            start = number
        if raw.rstrip().endswith("\\"):
            pending += raw.rstrip()[:-1] + " "
            continue
        lines.append((start, (pending + raw).strip()))
        pending = ""
    if pending:
        lines.append((start, pending.strip()))
    return lines


def _expand(text: str, variables: dict[str, str]) -> str:
    def replace(match: re.Match) -> str:
        name = match["braced"] or match["bare"]
        if name in variables:
            return variables[name]
        return match["default"] or ""

    return _VARIABLE.sub(replace, text)


def _assignments(arguments: str, line: int) -> list[tuple[str, str]]:
    """``a=b c="d e"`` or the legacy ``a b c`` form of ENV."""
    words = shlex.split(arguments)
    if words and "=" not in words[0]:
        if len(words) < 2:
            raise DockerfileError(f"line {line}: ENV needs a name and a value")
        return [(words[0], " ".join(words[1:]))]
    pairs = []
    for word in words:
        name, separator, value = word.partition("=")
        if not separator or not name:
            raise DockerfileError(f"line {line}: {word!r} is not NAME=value")
        pairs.append((name, value))
    return pairs


def parse(text: str, build_args: dict[str, str] | None = None) -> Plan:
    base = None
    steps: list[Step] = []
    args = dict(build_args or {})  # overrides for ARG defaults
    declared: dict[str, str] = {}
    env: dict[str, str] = {}
    workdir = "/"
    for line, logical in _logical_lines(text):
        keyword, _, arguments = logical.partition(" ")
        keyword = keyword.upper()
        arguments = arguments.strip()
        variables = {**declared, **env}
        if keyword == "FROM":
            if base is not None:
                raise DockerfileError(f"line {line}: multi-stage builds are not supported")
            words = _expand(arguments, variables).split()
            if len(words) != 1:
                raise DockerfileError(f"line {line}: FROM supports a single image reference")
            base = words[0]
        elif base is None:
            if keyword != "ARG":
                raise DockerfileError(f"line {line}: {keyword} before FROM")
            name, _, default = arguments.partition("=")
            declared[name] = args.get(name, default)
        elif keyword == "ARG":
            name, _, default = arguments.partition("=")
            declared[name] = args.get(name, _expand(default, variables))
        elif keyword == "ENV":
            for name, value in _assignments(_expand(arguments, variables), line):
                env[name] = value
        elif keyword == "WORKDIR":
            path = _expand(arguments, variables)
            workdir = path if path.startswith("/") else f"{workdir.rstrip('/')}/{path}"
            steps.append(Step("WORKDIR", line, destination=workdir))
        elif keyword == "RUN":
            if arguments.startswith("--"):
                raise DockerfileError(f"line {line}: RUN options ({arguments.split()[0]}) are not supported")
            if arguments.startswith("["):
                try:
                    argv = tuple(json.loads(arguments))
                except json.JSONDecodeError:
                    raise DockerfileError(f"line {line}: malformed exec form") from None
            else:
                argv = ("/bin/sh", "-c", arguments)
            steps.append(Step("RUN", line, argv=argv, env=tuple({**declared, **env}.items()), workdir=workdir))
        elif keyword in ("COPY", "ADD"):
            expanded = _expand(arguments, variables)
            words = json.loads(expanded) if expanded.startswith("[") else shlex.split(expanded)
            options = [w for w in words if w.startswith("--")]
            words = [w for w in words if not w.startswith("--")]
            if any(not o.startswith(("--chown=", "--chmod=")) for o in options):
                raise DockerfileError(f"line {line}: {keyword} option {options[0]} is not supported")
            if len(words) < 2:
                raise DockerfileError(f"line {line}: {keyword} needs a source and a destination")
            if keyword == "ADD" and any(re.match(r"\w+://", w) or w.endswith((".tar", ".tgz", ".tar.gz")) for w in words[:-1]):
                raise DockerfileError(f"line {line}: ADD of URLs and archives is not supported; use COPY")
            destination = words[-1]
            if not destination.startswith("/"):
                destination = f"{workdir.rstrip('/')}/{destination}"
            steps.append(Step("COPY", line, sources=tuple(words[:-1]), destination=destination))
        elif keyword == "USER":
            if _expand(arguments, variables) not in ("root", "0", "root:root", "0:0"):
                raise DockerfileError(f"line {line}: USER {arguments}: only root is supported")
        elif keyword in _METADATA:
            continue
        else:
            raise DockerfileError(f"line {line}: {keyword} is not supported")
    if base is None:
        raise DockerfileError("no FROM instruction")
    return Plan(base=base, steps=tuple(steps), env=env, workdir=workdir)
