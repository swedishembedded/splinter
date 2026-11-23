# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for reproducible container-image based
# build and test environments on hosts without a container daemon for its clients.
# If your team needs expertise in hermetic benchmark environments then you can
# procure our services by sending an email to info@swedishembedded.com.
"""Build a root filesystem from a Dockerfile on a host without a container daemon.

    <data>/rootfs/<name>/tree          the finished tree (hard links into the content store)
    <data>/rootfs/<name>/image.json    environment, working directory and provenance
    <cache>/blobs/sha256/<hex>         base image layers, by digest, shared by every build
    <cache>/bases/<key>/tree           a base image unpacked once, shared by every build
    <cache>/store/                     the files of all roots and bases, by content

The steps, in order:

1. The ``FROM`` image is pulled for the host's platform (``registry.py``) and unpacked once
   per set of layers (``layers.py``).
2. A *copy* of the base is the working tree (never the shared files: steps write in place).
3. ``COPY`` copies from the build context; ``WORKDIR`` creates the directory; ``RUN`` runs
   the command as root in a user namespace pivoted into the tree (``chroot_launch.py``)
   with the network of the host, so packages download.
4. Build-time scaffolding is removed and the tree is recreated from the content store
   (``store.py``), where identical files across images are one inode.

Running package managers in a namespace that maps a single identity needs two things.
apt drops privileges to its ``_apt`` user for downloads, which that namespace cannot
express, so ``APT::Sandbox::User "root"`` keeps it as root. And packages chown files to
owners the namespace cannot represent (the kernel answers EINVAL), which ``nochown.c``
reports as done through ``LD_PRELOAD``; every file belongs to the one identity anyway.
dpkg also gets ``--force-unsafe-io``: its fsync per file is very slow on a network
filesystem, and a failed build is simply rerun.

The result records the base image's environment plus the Dockerfile's ``ENV`` lines and
its last ``WORKDIR`` in ``image.json``; the sandbox takes its ``PATH``, environment and
working directory from there (``sandbox/userns.py``).
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import posixpath
import shutil
import stat
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from splinter_harbor_worker.errors import ExternalToolError, UsageError
from splinter_harbor_worker.fscopy import copy_file
from splinter_harbor_worker.rootfs import dockerfile
from splinter_harbor_worker.rootfs.locks import locked
from splinter_harbor_worker.rootfs.proot import PROOT_ENV, proot_argv
from splinter_harbor_worker.rootfs.layers import make_deletable, remove_tree, resolve_in_root, unpack_layers
from splinter_harbor_worker.rootfs.registry import RegistryClient
from splinter_harbor_worker.rootfs.store import link_tree

LAUNCHER = Path(__file__).with_name("chroot_launch.py")
SHIM_SOURCE = Path(__file__).with_name("nochown.c")
SCAFFOLD = ".bench-build"
APT_CONFIG = "etc/apt/apt.conf.d/99bench-build"
DPKG_CONFIG = "etc/dpkg/dpkg.cfg.d/99bench-build"
DEFAULT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
STEP_TIMEOUT_SEC = 3600


def build_root(
    dockerfile_path: Path,
    context: Path,
    *,
    name: str,
    data: Path,
    cache: Path,
    client: RegistryClient | None = None,
    pins: dict[str, str] | None = None,
    log: Callable[[str], None] = lambda message: None,
) -> Path:
    """Build ``<data>/rootfs/<name>`` from the Dockerfile; return its directory.

    ``pins`` maps image references to the digest they must resolve to (the registry is
    asked only for references without a pin; the caller records new ones). An existing
    build of the same Dockerfile and base is reused. A step that fails leaves the tree
    in ``<cache>/failed/<name>`` for inspection and raises with the end of its output."""
    text = dockerfile_path.read_text()
    plan = dockerfile.parse(text)
    target = data / "rootfs" / name
    digest = hashlib.sha256(text.encode()).hexdigest()
    existing = _read_image(target)
    if existing and existing.get("dockerfile_sha256") == digest and existing["base"]["reference"] == _pinned(
        plan.base, pins
    ):
        log(f"{name}: up to date")
        return target
    client = client or RegistryClient(cache)
    reference = _pinned(plan.base, pins)
    store = cache / "store"
    base_root, base_config = ensure_base(client, reference, cache, store, log)
    work = cache / "work" / name
    remove_tree(work)
    work.mkdir(parents=True)
    tree = work / "root"
    try:
        _copy_exact(base_root, tree)
        env = base_env(base_config) | plan.env
        shim = ensure_shim(cache) if any(step.kind == "RUN" for step in plan.steps) else None
        prepare_tree(tree, shim)
        for step in plan.steps:
            log(f"{name}: {step.kind} (line {step.line})")
            run_step(step, tree, context, env, cache / "logs" / f"{name}.log")
        finish_tree(tree)
        make_deletable(tree)
        partial = data / "rootfs" / f"{name}.partial"
        remove_tree(partial)
        stats = link_tree(tree, partial / "root", store)
        (partial / "image.json").write_text(
            json.dumps(
                {
                    "env": {"PATH": DEFAULT_PATH, **env},
                    "workdir": plan.workdir,
                    "base": {"reference": reference, "layers": base_config["rootfs"]["diff_ids"]},
                    "dockerfile_sha256": digest,
                    "files": stats.files,
                    "new_files": stats.stored,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        remove_tree(target)
        partial.replace(target)
    except BaseException:
        failed = cache / "failed" / name
        remove_tree(failed)
        failed.parent.mkdir(parents=True, exist_ok=True)
        if tree.exists():
            tree.replace(failed)
        raise
    finally:
        remove_tree(work)
    return target


def _pinned(reference: str, pins: dict[str, str] | None) -> str:
    if "@" in reference or not pins or reference not in pins:
        return reference
    return f"{reference.partition(':')[0]}@{pins[reference]}"


def _read_image(target: Path) -> dict | None:
    try:
        return json.loads((target / "image.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None


def ensure_base(
    client: RegistryClient, reference: str, cache: Path, store: Path, log: Callable[[str], None]
) -> tuple[Path, dict]:
    image = client.pull(reference)
    key = hashlib.sha256("\n".join(p.name for p in image.layers).encode()).hexdigest()[:24]
    base = cache / "bases" / key
    # Trials and builds that start together share a base: one process unpacks it, the others wait for it.
    with locked(cache / "locks" / f"base-{key}.lock"):
        if not (base / "root").is_dir():
            log(f"unpacking {reference} ({len(image.layers)} layers)")
            scratch = cache / "work" / f"base-{key}"
            remove_tree(scratch)
            try:
                unpack_layers(list(image.layers), scratch)
                make_deletable(scratch)
                remove_tree(base)
                link_tree(scratch, base / "root", store)
            finally:
                remove_tree(scratch)
    return base / "root", image.config


def base_env(config: dict) -> dict[str, str]:
    env = {}
    for item in config.get("config", {}).get("Env") or []:
        key, _, value = item.partition("=")
        env[key] = value
    return env


def _copy_exact(source: Path, target: Path) -> None:
    """A tree of independent files (no shared inodes) with the source's modes and times."""
    target.mkdir(parents=True)
    for entry in sorted(os.scandir(source), key=lambda e: e.name):
        here, there = Path(entry.path), target / entry.name
        info = entry.stat(follow_symlinks=False)
        if stat.S_ISLNK(info.st_mode):
            os.symlink(os.readlink(here), there)
            os.utime(there, ns=(info.st_atime_ns, info.st_mtime_ns), follow_symlinks=False)
        elif stat.S_ISDIR(info.st_mode):
            _copy_exact(here, there)
            os.chmod(there, stat.S_IMODE(info.st_mode))
            os.utime(there, ns=(info.st_atime_ns, info.st_mtime_ns))
        elif stat.S_ISREG(info.st_mode):
            shutil.copyfile(here, there)
            os.chmod(there, stat.S_IMODE(info.st_mode))
            os.utime(there, ns=(info.st_atime_ns, info.st_mtime_ns))
        elif stat.S_ISFIFO(info.st_mode):
            os.mkfifo(there, stat.S_IMODE(info.st_mode))


def ensure_shim(cache: Path) -> Path:
    """The compiled ``nochown.c`` (host C compiler), built once per cache."""
    shim = cache / "nochown.so"
    if shim.is_file() and shim.stat().st_mtime >= SHIM_SOURCE.stat().st_mtime:
        return shim
    cache.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which("cc") or shutil.which("gcc")
    if compiler is None:
        raise ExternalToolError("building a root filesystem needs a C compiler on the host (cc or gcc)")
    result = subprocess.run(
        [compiler, "-shared", "-fPIC", "-O2", "-o", str(shim), str(SHIM_SOURCE)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise ExternalToolError(f"compiling {SHIM_SOURCE.name}: {result.stderr.strip()}")
    return shim


def prepare_tree(tree: Path, shim: Path | None) -> None:
    """Build-time scaffolding: the chown shim, apt without its privilege drop, dpkg without fsync."""
    scaffold = tree / SCAFFOLD
    scaffold.mkdir()
    if shim is not None:
        copy_file(shim, scaffold / "nochown.so")
    for relative, text in (
        (APT_CONFIG, 'APT::Sandbox::User "root";\n'),
        (DPKG_CONFIG, "force-unsafe-io\n"),
    ):
        path = tree / relative
        if path.parent.is_dir():
            path.write_text(text)
    resolv = tree / "etc" / "resolv.conf"
    if resolv.is_symlink():
        resolv.unlink()


def finish_tree(tree: Path) -> None:
    for relative in (SCAFFOLD, APT_CONFIG, DPKG_CONFIG, "etc/resolv.conf"):
        path = tree / relative
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)


def run_step(step: dockerfile.Step, tree: Path, context: Path, env: dict[str, str], log_path: Path,
             proot: str | None = None) -> None:
    """One step of a Dockerfile on ``tree``; a ``RUN`` runs in its own namespaces, or through ``proot`` (the binary)
    where namespaces are not available."""
    if step.kind == "WORKDIR":
        resolve_in_root(tree, PurePosixPath(step.destination), follow_last=True).mkdir(parents=True, exist_ok=True)
    elif step.kind == "COPY":
        _copy_into(step, tree, context)
    else:
        _run_command(step, tree, env, log_path, proot)


def _copy_into(step: dockerfile.Step, tree: Path, context: Path) -> None:
    sources: list[Path] = []
    for pattern in step.sources:
        matches = sorted(glob.glob(str(context / pattern)))
        if not matches:
            raise UsageError(f"line {step.line}: COPY source {pattern!r} matches nothing in {context}")
        for match in matches:
            path = Path(os.path.normpath(match))
            if context.resolve() not in (path.resolve(), *path.resolve().parents):
                raise UsageError(f"line {step.line}: COPY source {pattern!r} is outside the build context")
            sources.append(path)
    destination = posixpath.normpath(step.destination)
    into_directory = (
        step.destination.endswith(("/", "/."))
        or len(sources) > 1
        or resolve_in_root(tree, PurePosixPath(destination), follow_last=True).is_dir()
    )
    base = resolve_in_root(tree, PurePosixPath(destination), follow_last=True)
    for source in sources:
        if source.is_dir():
            _copy_directory(source, base)
        else:
            target = base / source.name if into_directory else base
            target.parent.mkdir(parents=True, exist_ok=True)
            copy_file(source, target)


def _copy_directory(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for entry in sorted(os.scandir(source), key=lambda e: e.name):
        if entry.is_dir(follow_symlinks=False):
            _copy_directory(Path(entry.path), target / entry.name)
        else:
            copy_file(entry.path, target / entry.name)


def _run_command(step: dockerfile.Step, tree: Path, env: dict[str, str], log_path: Path, proot: str | None) -> None:
    spec = {
        "root": str(tree),
        "cwd": step.workdir,
        "argv": list(step.argv),
        "env": {
            "PATH": DEFAULT_PATH,
            "HOME": "/root",
            **env,
            **dict(step.env),
            "LD_PRELOAD": f"/{SCAFFOLD}/nochown.so",
        },
    }
    argv = [
        "unshare", "--user", "--map-root-user", "--mount", "--pid", "--fork", "--kill-child", "--ipc", "--uts",
        "--", sys.executable, "-I", str(LAUNCHER), json.dumps(spec),
    ]
    environment = None
    if proot is not None:
        # proot's root (-0) fakes ownership changes itself, so no chown shim; the host's name resolution is bound.
        resolver = [("/etc/resolv.conf", "/etc/resolv.conf")] if Path("/etc/resolv.conf").is_file() else []
        argv = proot_argv(proot, tree, resolver, spec["cwd"], spec["argv"])
        environment = {**{key: value for key, value in spec["env"].items() if key != "LD_PRELOAD"}, **PROOT_ENV}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        log.write(f"\n=== line {step.line}: {' '.join(step.argv)[:300]}\n".encode())
        log.flush()
        try:
            result = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                    check=False, timeout=STEP_TIMEOUT_SEC, env=environment)
        except subprocess.TimeoutExpired:
            raise ExternalToolError(f"line {step.line}: RUN timed out after {STEP_TIMEOUT_SEC} s (log {log_path})") from None
    if result.returncode != 0:
        tail = log_path.read_text(errors="replace")[-1500:]
        raise ExternalToolError(f"line {step.line}: RUN exited {result.returncode}; end of {log_path}:\n{tail}")
