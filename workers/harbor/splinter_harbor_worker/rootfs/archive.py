# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for running published agent benchmarks
# on hosts without a container daemon for its clients. If your team needs expertise
# in hermetic benchmark environments then you can procure our services by sending
# an email to info@swedishembedded.com.
"""Build a Dockerfile into one archive, in memory: the cache gains one file per image, never a tree.

``build_archive(dockerfile, context, cache=..., name=...)`` returns ``<cache>/images/<name>.tar`` and its
``<name>.json`` (environment, working directory, base image and its manifest digest). The build runs in a private
user and mount namespace (``unshare --user --map-root-user --mount``) whose working tree is a tmpfs: the base's
layers are unpacked into it, the steps run there exactly as ``build.build_root`` runs them (``COPY``, ``WORKDIR``,
``RUN`` in ``chroot_launch.py`` with the chown shim, apt without its privilege drop), and the finished tree is
written as one uncompressed tar. Nothing of the tree touches the shared disk, whose inode quota a few unpacked images
exhaust (an image is tens of thousands of files; a whole dataset's images are hundreds of thousands).

Where user namespaces are not available (``builder="proot"``) the same steps run through ``proot`` on a tree below the
cache's ``work/``, which is removed once archived.

A sandbox unpacks the archive into its own tmpfs (``sandbox/userns_keeper.py``). The name carries the
build context's digest (``images._context_digest``: the Dockerfile and every file beside it), so an unchanged context
is built once and a context with any changed file is built again; a concurrent build of the same name waits for the
first under a lock.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from collections.abc import Callable
from pathlib import Path

from splinter_harbor_worker.errors import ExternalToolError
from splinter_harbor_worker.rootfs import dockerfile
from splinter_harbor_worker.rootfs.build import DEFAULT_PATH, base_env, ensure_shim, finish_tree, prepare_tree, run_step
from splinter_harbor_worker.rootfs.layers import make_deletable, unpack_layers
from splinter_harbor_worker.rootfs.locks import locked
from splinter_harbor_worker.rootfs.proot import proot_binary
from splinter_harbor_worker.rootfs.registry import RegistryClient

SOURCE_ROOT = Path(__file__).resolve().parents[2]
# The build's tmpfs: the largest tree a Dockerfile may produce here.
BUILD_SIZE_MB = 32768
BUILD_TIMEOUT_SEC = 4 * 3600
# How a Dockerfile is built: in a user and mount namespace, or with proot where namespaces are not available.
BUILDERS = ("userns", "proot")


def build_archive(
    dockerfile_path: Path,
    context: Path,
    *,
    cache: Path,
    name: str,
    client: RegistryClient | None = None,
    log: Callable[[str], None] = lambda message: None,
    builder: str = "userns",
) -> tuple[Path, dict]:
    """(the archive of the built tree, its image record); an archive of the same name is reused. ``builder``
    ``proot`` builds where user namespaces are not available: the tree is a directory below the cache's ``work/``,
    removed once archived, and the ``RUN`` steps run through ``proot``."""
    if builder not in BUILDERS:
        raise ValueError(f"builder {builder!r}: use one of {', '.join(BUILDERS)}")
    archive = cache / "images" / f"{name}.tar"
    record_path = archive.with_suffix(".json")
    with locked(cache / "locks" / f"build-{name}.lock"):
        if archive.is_file() and record_path.is_file():
            return archive, json.loads(record_path.read_text())
        text = dockerfile_path.read_text()
        plan = dockerfile.parse(text)
        client = client or RegistryClient(cache)
        base = client.pull(plan.base)
        shim = ensure_shim(cache) if builder == "userns" and any(step.kind == "RUN" for step in plan.steps) else None
        env = {"PATH": DEFAULT_PATH, **base_env(base.config), **plan.env}
        mountpoint = cache / "work" / name
        mountpoint.mkdir(parents=True, exist_ok=True)
        archive.parent.mkdir(parents=True, exist_ok=True)
        partial = archive.with_name(f"{archive.name}.partial-{os.getpid()}")
        log_path = cache / "logs" / f"{name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        spec = {"dockerfile": str(dockerfile_path), "context": str(context), "layers": [str(p) for p in base.layers],
                "env": env, "mountpoint": str(mountpoint), "archive": str(partial), "shim": str(shim) if shim else None,
                "log": str(log_path), "proot": proot_binary() if builder == "proot" else None}
        log(f"{name}: building {dockerfile_path} on {plan.base} " + ("with proot" if spec["proot"] else "in memory"))
        prefix = [] if spec["proot"] else ["unshare", "--user", "--map-root-user", "--mount", "--"]
        try:
            result = subprocess.run(
                [*prefix, sys.executable, "-m", "splinter_harbor_worker.rootfs.archive"],
                input=json.dumps(spec), capture_output=True, text=True, check=False, timeout=BUILD_TIMEOUT_SEC,
                env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(SOURCE_ROOT), "LANG": "C.UTF-8"})
        except subprocess.TimeoutExpired:
            partial.unlink(missing_ok=True)
            raise ExternalToolError(f"{name}: the build did not finish in {BUILD_TIMEOUT_SEC} s (log {log_path})") from None
        finally:
            if spec["proot"] and (mountpoint / "root").exists():  # a failed build's tree is on disk
                make_deletable(mountpoint / "root")
                shutil.rmtree(mountpoint / "root")
            mountpoint.rmdir()
        if result.returncode != 0:
            partial.unlink(missing_ok=True)
            raise ExternalToolError(f"{name}: building {dockerfile_path} failed: {result.stderr.strip()[-2500:]}")
        record = {
            "env": env, "workdir": plan.workdir, "base": {"reference": plan.base, "manifest_digest": base.manifest_digest},
            "dockerfile_sha256": hashlib.sha256(text.encode()).hexdigest(), "builder": builder,
        }
        partial.replace(archive)
        record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        return archive, record


def _build_in_namespace(spec: dict) -> None:
    """Inside the build's namespace: a tmpfs working tree, the base, the steps, one tar."""
    from splinter_harbor_worker.sandbox.userns_keeper import mount_tmpfs

    mountpoint = Path(spec["mountpoint"])
    if not spec.get("proot"):
        mount_tmpfs(str(mountpoint), BUILD_SIZE_MB)
    tree = mountpoint / "root"
    unpack_layers([Path(p) for p in spec["layers"]], tree)
    plan = dockerfile.parse(Path(spec["dockerfile"]).read_text())
    prepare_tree(tree, Path(spec["shim"]) if spec["shim"] else None)
    for step in plan.steps:
        run_step(step, tree, Path(spec["context"]), spec["env"], Path(spec["log"]), spec.get("proot"))
    finish_tree(tree)
    make_deletable(tree)
    with tarfile.open(spec["archive"], "w", format=tarfile.PAX_FORMAT) as out:
        for entry in sorted(tree.iterdir()):
            out.add(entry, arcname=entry.name)
    if spec.get("proot"):
        shutil.rmtree(tree)  # on disk, not in a tmpfs that goes with the namespace


if __name__ == "__main__":
    try:
        _build_in_namespace(json.loads(sys.stdin.read()))
    except Exception as exc:  # the parent reports the end of stderr
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
