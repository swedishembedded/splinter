# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for running agent benchmarks inside locked-down development containers
# for its clients. If your team needs expertise in agent evaluation infrastructure then you can procure our services
# by sending an email to info@swedishembedded.com.
"""Harbor environment that runs a task in its own image with ``proot``, where Linux namespaces are not available.

For hosts that are themselves containers which may not create user namespaces (a hardened development container on
rootless Docker: no ``CAP_SYS_ADMIN``, Docker's default seccomp filter), so ``splinter_harbor_worker.sandbox.userns``
cannot start. The worker selects it for ``sandbox.provider = "proot"``. Only a task that brings its own image (``environment/Dockerfile`` or
``[environment].docker_image``, as every compiled task does) runs here, and the run must give an image cache
(``images=DIR``): the image is pulled, or its Dockerfile built with ``proot`` running the ``RUN`` steps
(``splinter_harbor_worker.rootfs.archive``, builder ``proot``), into that cache once.

Each sandbox unpacks the image into a directory of its own (``<trial>/sandbox-<session>/image``), writable as a
container's tree and private to the sandbox, and every command runs in it through ``proot -0``: the image is ``/``, the
command runs as the image's root, and the sandbox's working directory, environment, tools (``/splinter/bin``),
``/tests``, ``/solution`` and ``/logs`` are those ``UserNamespaceEnvironment`` gives a task's own image. ``proot``
translates paths by tracing system calls (``ptrace``): it needs no privilege and no namespace.

What it does not give, so a run on it is not a sandbox in the security sense:

* Containment. The command runs as the invoking user, with the host's ``/proc`` and ``/dev``; a program that means to
  leave the tree can (``/proc/<pid>/root`` of a host process is the host's root). The container this runs in is the
  boundary. Host files are never bound writable but the trial's own logs and state: the host runtime the tools need
  and the host's name-resolution files are copied into the tree.
* Network policy. The sandbox shares the host's network: a task whose network mode is not ``public`` is refused
  (Harbor checks the capabilities below), and the model gateway on the host's loopback is reachable as is.
* A storage limit, and read-only binds (a read-only Harbor mount is refused).
* A background process outliving its command: ``proot`` returns when every process it traces has ended, so a command
  that leaves a server running holds its exec until the command's time limit.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import os
import pty
import shutil
import subprocess
import termios
from collections.abc import AsyncIterator
from pathlib import Path, PurePosixPath

from harbor.environments.capabilities import EnvironmentCapabilities

from splinter_harbor_worker.fscopy import copy_file, copy_tree
from splinter_harbor_worker.sandbox.userns import (
    _HOST_CERTIFICATES, _HOST_RUNTIME, _OUTPUT_GRACE_SEC, OutputSink, UserNamespaceEnvironment,
    _exit_status, _feed, _pump,
)
from splinter_harbor_worker.sandbox.terminal import FRAME, RESIZE, SIZE, TYPED, set_size
from splinter_harbor_worker.sandbox.userns_keeper import prepare
from splinter_harbor_worker.procgroup import signal_group
from splinter_harbor_worker.rootfs.images import ensure_task_image
from splinter_harbor_worker.rootfs.layers import make_deletable, resolve_in_root, unpack_layers
from splinter_harbor_worker.rootfs.proot import HOST_SYSTEM, PROOT_ENV, proot_argv, proot_binary
from splinter_harbor_worker.rootfs.registry import RegistryClient

# The host's name-resolution files, copied into the tree at start (a container runtime provides them the same way).
_HOST_NAME_FILES = ("etc/resolv.conf", "etc/hosts")
# A file only the sandbox's tree holds, while start() checks that commands see the tree as their root.
_MARKER = ".splinter-proot-root"
# Where the host's certificate store is copied in the tree, for programs from the host in an image without one.
_CERTIFICATES = f"{_HOST_RUNTIME}/certs/ca-certificates.crt"


def _shared_libraries(program: Path) -> list[Path]:
    """The libraries ``program`` loads on this host and its program interpreter, as ``ldd`` resolves them (none for a
    static program or a script)."""
    result = subprocess.run(["ldd", str(program)], capture_output=True, text=True, check=False)
    found = []
    for line in result.stdout.splitlines():
        path = next((word for word in line.split() if word.startswith("/")), None)
        if path is not None and Path(path).is_file():
            found.append(Path(path))
    return found


class ProotEnvironment(UserNamespaceEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._proot = proot_binary()

    @staticmethod
    def type() -> str:
        return "proot"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        # No network policy can be enforced: the sandbox shares the host's network, so Harbor refuses a task whose
        # network mode is not public.
        return EnvironmentCapabilities(mounted=True)

    def _validate_definition(self) -> None:
        proot_binary()

    def _extra_mounts(self) -> list[tuple[Path, PurePosixPath, bool]]:
        mounts = super()._extra_mounts()
        if read_only := [str(target) for _, target, ro in mounts if ro]:
            raise ValueError(f"proot environment: read-only mounts ({', '.join(read_only)}) cannot be made; proot binds "
                             "are writable")
        return mounts

    # -- lifecycle -------------------------------------------------------------

    async def _start_own_image(self) -> None:
        """Resolve the task's image (pulled, or built with proot, once into the cache) and unpack this sandbox's own
        copy of it."""
        self._pulled = await asyncio.to_thread(
            ensure_task_image, self.task_env_config.docker_image, self.environment_dir, self._images,
            name=self.environment_name, platform_fallback=self._platform_fallback,
            client=RegistryClient(self._images, mirror=self._registry_mirror),
            log=lambda message: self.logger.info(message), builder="proot")
        record = self.trial_paths.trial_dir / f"image-{self.session_id}.json"
        record.write_text(json.dumps({**self._pulled.provenance, "workdir": self._pulled.workdir, "sandbox": "proot"},
                                     indent=2, sort_keys=True) + "\n")
        await asyncio.to_thread(self._unpack)
        await self._check_translation()

    async def _check_translation(self) -> None:
        """A command must see the tree as its root, never the host's: proot that fails to translate a path runs the
        command on the host instead, silently. Refused before anything else runs."""
        marker = self._image_root / _MARKER
        marker.write_text(self.session_id)
        try:
            seen = await self.exec(f"cat /{_MARKER}", cwd="/", timeout_sec=60)
        finally:
            marker.unlink(missing_ok=True)
        if seen.return_code != 0 or seen.stdout.strip() != self.session_id:
            raise RuntimeError(f"proot environment: {self._proot} does not run commands in the image here "
                               f"({(seen.stderr or seen.stdout).strip()[-300:]}); refusing to run on the host")

    def _unpack(self) -> None:
        root = self._image_root
        if root.exists():
            make_deletable(root)
            shutil.rmtree(root)
        root.mkdir(parents=True)
        unpack_layers(list(self._pulled.layers), root)
        prepare(str(root), None)
        for relative in _HOST_NAME_FILES:
            host = Path("/") / relative
            if host.is_file():
                copy_file(host, root / relative)
        self._install_host_runtime(root)

    def _install_host_runtime(self, root: Path) -> None:
        """The host's loader and the libraries the tools load, at /splinter/host (``agents.sven``'s ``HOST_LOADER``),
        and its certificate store: copies, so a program of the sandbox can never change the host's."""
        runtime = root / str(_HOST_RUNTIME).lstrip("/")
        libraries = runtime / "lib"
        libraries.mkdir(parents=True, exist_ok=True)
        loader = None
        for tool in sorted(self._tools_dir.iterdir()):
            for library in _shared_libraries(tool):
                copy_file(library.resolve(), libraries / library.name)
                if library.name.startswith("ld-linux"):
                    loader = library.name
        if loader is not None and not (runtime / "ld.so").exists():
            (runtime / "ld.so").symlink_to(f"lib/{loader}")
        certificates = Path(_HOST_CERTIFICATES) / "ca-certificates.crt"
        if certificates.is_file():
            target = root / _CERTIFICATES.lstrip("/")
            target.parent.mkdir(parents=True, exist_ok=True)
            copy_file(certificates, target)

    async def _stop_keeper(self) -> None:
        """No process holds the copy: it is a directory, removed with the sandbox."""

    async def stop(self, delete: bool) -> None:
        if delete and self._image_root.exists():
            make_deletable(self._image_root)
        await super().stop(delete)

    # -- commands --------------------------------------------------------------

    def _argv(self, spec: dict) -> tuple[list[str], dict[str, str]]:
        """The proot command line and environment of a launch specification (``_launch_spec``)."""
        binds = [(host, inside) for host, inside, _ in spec["binds"]
                 if not inside.startswith(f"{_HOST_RUNTIME}/") and inside not in ("/etc/resolv.conf", "/etc/hosts")
                 and inside not in HOST_SYSTEM]
        argv = proot_argv(self._proot, self._image_root, binds, spec["cwd"], ["/bin/sh", "-c", spec["command"]])
        return argv, {**spec["env"], **PROOT_ENV}

    async def _stream(self, spec: dict, command: str, timeout_sec: int | None, sink: OutputSink,
                      keys: AsyncIterator[bytes] | None = None) -> int:
        argv, env = self._argv(spec)
        if "terminal" in spec:
            return await self._terminal(argv, env, spec["terminal"], command, timeout_sec, sink, keys)
        proc = await asyncio.create_subprocess_exec(
            *argv, stdin=asyncio.subprocess.PIPE if keys is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True, env=env)
        readers = [asyncio.ensure_future(_pump(proc.stdout, 1, sink)), asyncio.ensure_future(_pump(proc.stderr, 2, sink))]
        feeder = asyncio.ensure_future(_feed(proc.stdin, keys)) if keys is not None else None
        try:
            return await self._wait(proc, command, timeout_sec, [*readers, *([feeder] if feeder else [])])
        finally:
            if feeder is not None:
                feeder.cancel()
                await asyncio.gather(feeder, return_exceptions=True)
            _, pending = await asyncio.wait(readers, timeout=_OUTPUT_GRACE_SEC)
            for reader in pending:
                reader.cancel()

    async def _wait(self, proc: asyncio.subprocess.Process, command: str, timeout_sec: int | None,
                    helpers: list[asyncio.Future]) -> int:
        """The command's return code; past ``timeout_sec``, or when the caller is cancelled, its whole process group
        is killed first."""
        try:
            return await asyncio.wait_for(_exit_status(proc), timeout=timeout_sec)
        except (TimeoutError, asyncio.CancelledError) as stopped:
            signal_group(proc)
            for helper in helpers:
                helper.cancel()
            if isinstance(stopped, asyncio.CancelledError):
                with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                    await asyncio.wait_for(_exit_status(proc), timeout=_OUTPUT_GRACE_SEC)
                raise
            await proc.wait()
            raise TimeoutError(f"command timed out after {timeout_sec} s: {command[:200]}") from None
    async def _terminal(self, argv: list[str], env: dict[str, str], size: dict, command: str, timeout_sec: int | None,
                        sink: OutputSink, keys: AsyncIterator[bytes] | None) -> int:
        """``argv`` on a pseudo-terminal of its own (``exec_terminal``): what it writes goes to ``sink``, ``keys`` are
        the person's frames (``sandbox.terminal``); their end hangs the terminal up."""
        master, slave = pty.openpty()
        set_size(master, size["rows"], size["cols"])
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, stdin=slave, stdout=slave, stderr=slave, start_new_session=True, env=env,
                preexec_fn=lambda: fcntl.ioctl(0, termios.TIOCSCTTY, 0))
        finally:
            os.close(slave)
        loop = asyncio.get_running_loop()
        output: asyncio.Queue[bytes | None] = asyncio.Queue()

        def readable() -> None:
            try:
                chunk = os.read(master, 1 << 16)
            except OSError:  # EIO: every process holding the terminal has gone
                chunk = b""
            if not chunk:
                loop.remove_reader(master)
            output.put_nowait(chunk or None)

        async def relay() -> None:
            while (chunk := await output.get()) is not None:
                await sink(1, chunk)

        async def typed() -> None:
            buffer = b""
            async for frame in keys:
                buffer += frame
                while len(buffer) >= FRAME.size:
                    kind, length = FRAME.unpack_from(buffer)
                    if len(buffer) < FRAME.size + length:
                        break
                    body, buffer = buffer[FRAME.size:FRAME.size + length], buffer[FRAME.size + length:]
                    if kind == TYPED:
                        os.write(master, body)
                    elif kind == RESIZE and length == SIZE.size:
                        set_size(master, *SIZE.unpack(body))
            os.write(master, b"\x04")  # the person has gone: end of input, as a hung-up terminal gives

        loop.add_reader(master, readable)
        relayed = asyncio.ensure_future(relay())
        feeder = asyncio.ensure_future(typed()) if keys is not None else None
        try:
            return await self._wait(proc, command, timeout_sec, [relayed, *([feeder] if feeder else [])])
        finally:
            if feeder is not None:
                feeder.cancel()
                await asyncio.gather(feeder, return_exceptions=True)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(relayed, timeout=_OUTPUT_GRACE_SEC)
            loop.remove_reader(master)
            os.close(master)

    # -- file transfer: the tree is a host directory -----------------------------

    def _inside(self, path: str, *, follow_last: bool) -> Path:
        """The host path of sandbox path ``path`` in the tree, every link on the way resolved inside it: a link the
        agent planted points into the image, never at the host."""
        absolute = PurePosixPath(os.path.normpath(path if path.startswith("/") else f"{self._workdir}/{path}"))
        return resolve_in_root(self._image_root, absolute, follow_last=follow_last)

    async def _transfer_in(self, source: Path, target: str, *, directory: bool) -> None:
        destination = self._inside(target, follow_last=directory)
        if directory:
            copy_tree(source, destination, dirs_exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_symlink():
                destination.unlink()
            copy_file(source, destination)

    async def _transfer_out(self, source: str, target: Path, *, directory: bool) -> None:
        origin = self._inside(source, follow_last=True)
        if origin.is_symlink() or not (origin.is_dir() if directory else origin.is_file()):
            raise FileNotFoundError(f"proot environment: {source} is not a {'directory' if directory else 'file'} "
                                    "of the sandbox")
        if directory:
            copy_tree(origin, target, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            copy_file(origin, target)
