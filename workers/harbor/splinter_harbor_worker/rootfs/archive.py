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

A Dockerfile whose steps after its first ``COPY`` are only ``COPY`` and ``WORKDIR`` (a task image: install the tools, then
put the task's files in) is built in two layers (``build_layers``): the steps before the first ``COPY`` once, as a shared
archive named by the digest of the base and those steps, and the rest as a small archive of only the files they write. A
dataset of thousands of tasks then holds one copy of the installed tools and a few kilobytes per task, and builds the
installation once, not once per task.

A sandbox's keeper unpacks the archives (``sandbox/userns_keeper.py``): the shared one once per node, the task's
layer into the sandbox's own overlay over it. The name carries the
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
import posixpath
import tarfile
import tempfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath

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
    until: int | None = None,
) -> tuple[Path, dict]:
    """(the archive of the built tree, its image record); an archive of the same name is reused. ``until`` builds only the
    first that many steps and records the directories and symbolic links of the tree (``dirs``, ``symlinks``), which a
    layer built on it needs to know. ``builder``
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
        listing = cache / "work" / f"{name}.listing.json" if until is not None else None
        spec = {"dockerfile": str(dockerfile_path), "context": str(context), "layers": [str(p) for p in base.layers],
                "until": until, "listing": str(listing) if listing else None,
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
        if listing is not None:
            record.update(json.loads(listing.read_text()))
            listing.unlink()
        partial.replace(archive)
        record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        return archive, record


LAYERED = ("COPY", "WORKDIR")


def shared_prefix(plan: dockerfile.Plan) -> int | None:
    """How many leading steps can be built once for every image that shares them: those before the first ``COPY``, when
    there are any and everything after is ``COPY`` or ``WORKDIR`` (a step that runs commands on the task's files would
    need the whole tree). ``None`` when the Dockerfile does not have that shape."""
    first = next((i for i, step in enumerate(plan.steps) if step.kind == "COPY"), None)
    if not first or any(step.kind not in LAYERED for step in plan.steps[first:]):
        return None
    return first


def prefix_name(plan: dockerfile.Plan, base_digest: str, until: int) -> str:
    """The name of the shared archive: the base image, the environment and the first ``until`` steps decide its content."""
    key = json.dumps([base_digest, sorted(plan.env.items()), [repr(step) for step in plan.steps[:until]]], sort_keys=True)
    return "prefix-" + hashlib.sha256(key.encode()).hexdigest()[:20]


def build_layers(
    dockerfile_path: Path,
    context: Path,
    *,
    cache: Path,
    name: str,
    client: RegistryClient | None = None,
    log: Callable[[str], None] = lambda message: None,
    builder: str = "userns",
) -> tuple[tuple[Path, ...], dict]:
    """(the archives to unpack in order, the image record): the shared prefix and the task's own layer where the
    Dockerfile has the shape ``shared_prefix`` names, else the one archive ``build_archive`` makes."""
    plan = dockerfile.parse(dockerfile_path.read_text())
    until = shared_prefix(plan)
    client = client or RegistryClient(cache)
    if until is not None:
        base = client.pull(plan.base)
        shared, shared_record = build_archive(dockerfile_path, context, cache=cache, client=client, log=log, builder=builder,
                                              name=prefix_name(plan, base.manifest_digest, until), until=until)
        layer = _layer(plan.steps[until:], context, shared_record, cache, name, log)
        if layer is not None:
            record = {k: v for k, v in shared_record.items() if k not in ("dirs", "symlinks")}
            record["workdir"] = plan.workdir
            record["archives"] = [shared.name, layer.name]
            return (shared, layer), record
    archive, record = build_archive(dockerfile_path, context, cache=cache, name=name, client=client, log=log, builder=builder)
    return (archive,), record


def _layer(steps: tuple, context: Path, shared: dict, cache: Path, name: str, log: Callable[[str], None]) -> Path | None:
    """The archive of exactly what ``steps`` (``COPY`` and ``WORKDIR``) write on top of the shared tree, or ``None`` when
    a destination passes through a symbolic link of that tree, which the layer cannot resolve without unpacking it."""
    archive = cache / "images" / f"{name}.layer.tar"
    if archive.is_file():
        return archive
    dirs, symlinks = set(shared["dirs"]), set(shared["symlinks"])
    destinations = [posixpath.normpath(step.destination) for step in steps]
    for destination in destinations:
        path = PurePosixPath(destination)
        if any(str(parent) in symlinks for parent in (path, *path.parents)):
            return None
    # Trials are threads of one process as well as other processes: the lock is held across both, and whoever comes
    # second finds the archive the first built.
    with locked(cache / "locks" / f"layer-{name}.lock"):
        if archive.is_file():
            return archive
        _write_layer(steps, context, dirs, destinations, cache, name, archive)
    log(f"{name}: layer {archive.name}")
    return archive


def _write_layer(steps: tuple, context: Path, dirs: set, destinations: list, cache: Path, name: str,
                 archive: Path) -> None:
    """Run ``steps`` on an empty tree and write it as ``archive``, under a name of its own until it is whole."""
    work = Path(tempfile.mkdtemp(prefix=f"{name}-", dir=_workdir(cache)))
    partial = None
    try:
        tree = work / "root"
        tree.mkdir()
        existing = [d for d in destinations if d in dirs]
        for d in existing:  # a destination that is a directory of the shared tree is one here: `COPY x /dir` goes into it
            (tree / d.lstrip("/")).mkdir(parents=True, exist_ok=True)
        for step in steps:
            run_step(step, tree, context, {}, work / "layer.log", None)
        archive.parent.mkdir(parents=True, exist_ok=True)
        handle, name_aside = tempfile.mkstemp(prefix=f".{archive.name}.", suffix=".partial", dir=archive.parent)
        os.close(handle)
        partial = Path(name_aside)
        partial.chmod(0o644)  # as readable as a file written in place; mkstemp makes it the owner's alone

        def as_root(info: tarfile.TarInfo) -> tarfile.TarInfo:
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            return info

        with tarfile.open(partial, "w", format=tarfile.PAX_FORMAT) as out:
            for path in sorted(tree.rglob("*")):
                inside = "/" + path.relative_to(tree).as_posix()
                if path.is_dir() and not path.is_symlink() and inside in dirs:
                    continue  # the directory is already there, with its own mode
                out.add(path, arcname=path.relative_to(tree).as_posix(), recursive=False, filter=as_root)
        partial.replace(archive)
        partial = None
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)
        shutil.rmtree(work, ignore_errors=True)


def _workdir(cache: Path) -> Path:
    path = cache / "work"
    path.mkdir(parents=True, exist_ok=True)
    return path


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
    steps = plan.steps if spec.get("until") is None else plan.steps[: spec["until"]]
    for step in steps:
        run_step(step, tree, Path(spec["context"]), spec["env"], Path(spec["log"]), spec.get("proot"))
    finish_tree(tree)
    make_deletable(tree)
    if spec.get("listing"):
        dirs, symlinks = [], []
        for root, names, files in os.walk(tree):
            relative = "/" + Path(root).relative_to(tree).as_posix().removeprefix(".")
            relative = relative.rstrip("/") or "/"
            for name in (*names, *files):
                path = Path(root) / name
                if path.is_symlink():
                    symlinks.append(posixpath.join(relative, name))
                elif path.is_dir():
                    dirs.append(posixpath.join(relative, name))
        Path(spec["listing"]).write_text(json.dumps({"dirs": sorted(dirs), "symlinks": sorted(symlinks)}))
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
