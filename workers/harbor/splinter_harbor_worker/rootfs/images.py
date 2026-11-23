# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for running published agent benchmarks
# on hosts without a container daemon for its clients. If your team needs expertise
# in hermetic benchmark environments then you can procure our services by sending
# an email to info@swedishembedded.com.
"""A Harbor task's image as archives a sandbox unpacks, without a container daemon.

``ensure_task_image(docker_image, environment_dir, cache, ...)`` returns what a sandbox's root filesystem is made of:
the archives to unpack, in order, into an empty tree, its environment and its working directory.

* The task's ``[environment].docker_image`` for this host's platform: its layers, pulled through ``registry.py``
  (every blob verified against its digest; a tag is resolved once and the manifest digest recorded).
* The task's ``environment/Dockerfile`` when the task names no image: built in memory into one archive
  (``archive.build_archive``).
* That Dockerfile in place of a pinned image that has no build for this platform, only when the run asks for it
  (``platform_fallback="dockerfile"``): the result is the same recipe on another architecture, not the pinned image,
  and its provenance says so and why.

The cache holds only archives (``blobs/``, ``images/``): an unpacked image is tens of thousands of files, which a disk
with an inode quota cannot hold per image. The tree exists only in the memory of the sandbox that unpacks it.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from splinter_harbor_worker.rootfs.archive import build_archive
from splinter_harbor_worker.rootfs.build import DEFAULT_PATH, base_env
from splinter_harbor_worker.rootfs.registry import PlatformUnavailable, RegistryClient, host_platform, parse_reference

FALLBACKS = ("dockerfile",)


@dataclass(frozen=True)
class TaskImage:
    layers: tuple[Path, ...]  # archives unpacked in order (OCI layer semantics) into an empty root
    env: dict[str, str]  # always holds PATH
    workdir: str  # absolute
    provenance: dict = field(default_factory=dict)


def ensure_task_image(
    docker_image: str | None,
    environment_dir: Path,
    cache: Path,
    *,
    name: str,
    client: RegistryClient | None = None,
    platform_fallback: str | None = None,
    log: Callable[[str], None] = lambda message: None,
    builder: str = "userns",
) -> TaskImage:
    """The root filesystem of a task whose image is ``docker_image`` and build context ``environment_dir``; a
    Dockerfile is built by ``builder`` (``archive.BUILDERS``)."""
    if platform_fallback not in (None, *FALLBACKS):
        raise ValueError(f"platform_fallback {platform_fallback!r}: use one of {', '.join(FALLBACKS)}")
    client = client or RegistryClient(cache)
    dockerfile = environment_dir / "Dockerfile"
    if docker_image:
        try:
            return _pulled(docker_image, cache, client, log)
        except PlatformUnavailable as exc:
            if platform_fallback != "dockerfile" or not dockerfile.is_file():
                hint = ("" if not dockerfile.is_file() else
                        "; the task's environment/Dockerfile can be built for this platform instead with "
                        "--ek platform_fallback=dockerfile")
                raise PlatformUnavailable(f"{exc}{hint}") from None
            log(f"{name}: {exc}; building its environment/Dockerfile for this platform instead")
            return _built(dockerfile, environment_dir, cache, name, client, log,
                          {"image": docker_image, "reason": str(exc)}, builder)
    if dockerfile.is_file():
        return _built(dockerfile, environment_dir, cache, name, client, log, {}, builder)
    raise ValueError(f"{environment_dir.parent}: the task has neither [environment].docker_image nor environment/Dockerfile")


def _pulled(reference: str, cache: Path, client: RegistryClient, log: Callable[[str], None]) -> TaskImage:
    platform = host_platform()
    image = client.pull(reference, platform)
    registry = client.host(parse_reference(reference))
    settings = image.config.get("config") or {}
    workdir = settings.get("WorkingDir") or "/"
    log(f"{reference}: {image.manifest_digest} for {'/'.join(platform)}, {len(image.layers)} layers")
    return TaskImage(image.layers, {"PATH": DEFAULT_PATH, **base_env(image.config)},
                     workdir if workdir.startswith("/") else f"/{workdir}", {
                         "source": "pull", "reference": reference, "manifest_digest": image.manifest_digest,
                         "registry": registry,
                         "platform": "/".join(platform),
                         "layers": (image.config.get("rootfs") or {}).get("diff_ids", []),
                     })


def _built(dockerfile: Path, context: Path, cache: Path, name: str, client: RegistryClient,
           log: Callable[[str], None], why: dict, builder: str = "userns") -> TaskImage:
    digest = hashlib.sha256(dockerfile.read_bytes()).hexdigest()
    context_digest = _context_digest(context)
    archive, record = build_archive(dockerfile, context, cache=cache, name=f"{_slug(name)}-{context_digest[:12]}",
                                    client=client, log=log, builder=builder)
    return TaskImage((archive,), record["env"], record["workdir"], {
        "source": "dockerfile", "dockerfile_sha256": digest, "context_sha256": context_digest, "base": record["base"],
        "platform": "/".join(host_platform()), "archive": archive.name,
        "builder": record.get("builder", "userns"), **why,
    })


def _context_digest(context: Path) -> str:
    """The digest of the whole build context (the Dockerfile and every file a step may COPY): the archive's name carries
    it, so a context republished with any changed file is built again instead of reusing the old image."""
    h = hashlib.sha256()
    for path in sorted(p for p in context.rglob("*") if p.is_file() or p.is_symlink()):
        rel = path.relative_to(context).as_posix().encode()
        body = os.readlink(path).encode() if path.is_symlink() else path.read_bytes()
        h.update(len(rel).to_bytes(8, "big") + rel + (b"L" if path.is_symlink() else b"F")
                 + len(body).to_bytes(8, "big") + body)
    return h.hexdigest()


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", name.lower()).strip("-.")[:60] or "task"
