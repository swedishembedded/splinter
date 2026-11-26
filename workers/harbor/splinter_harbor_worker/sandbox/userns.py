# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for running agent benchmarks in
# isolated environments on shared compute for its clients. If your team needs
# expertise in agent evaluation infrastructure then you can procure our services
# by sending an email to info@swedishembedded.com.
"""Harbor environment that isolates a trial in Linux user namespaces.

For hosts without a container daemon that allow unprivileged user namespaces. The worker selects it for
``sandbox.provider = "userns"``. Only a task that brings its own image (``environment/Dockerfile`` or
``[environment].docker_image``) runs here, and the run must give an image cache (``images=DIR``): the image is
pulled, or its Dockerfile built once, into that cache as archives.

Each sandbox has a keeper (``userns_keeper.py``) holding a private, writable copy of the image in its own user,
mount and PID namespaces (``copy_placement``): on the local disk its trial directory is on, an overlay over the
image's shared layers unpacked once per node into ``<images>/unpacked`` (one tree per distinct set of shared layers,
kept for reuse and never evicted while the worker runs: a tree in use cannot be told from one that is not, so clear
it between runs if the disk needs the room), with only the task's own layer and the sandbox's writes in the sandbox's
directory, removed when the sandbox stops; with every layer unpacked into that directory when the overlay cannot be
had (the reason is logged and recorded in the trial's image record); or a tmpfs when the trial directory is on a
network filesystem. Every exec enters those namespaces (``nsenter``) and builds its own mount namespace with
``userns_launch.py``, which binds the trial's writable state, pivots into the tree and drops every capability before the
command runs. /logs/{agent,verifier,artifacts} are the trial's own directories, bind-mounted, so nothing needs
downloading. /tests and /solution are filled by Harbor's uploads.

Network: such hosts usually cannot create network namespaces, so the sandbox shares the host's network and the policy is
enforced on system calls (``userns_net.py``): a seccomp filter refuses IPv6, packet and vsock sockets and io_uring, and
every connect, bind and listen is decided by a supervisor process that checks a copy of the address. ``no-network``
allows no IP destination at all. An allowlist of IPv4 addresses, set per phase through Harbor's ``extra_allowed_hosts``,
admits connections to those addresses on the ports given by ``allow_ports=PORT,...`` and nothing else; that is how
the model gateway is opened to the agent phase only. Unix sockets keep working.

Tools: ``tools=name:path,...`` provides each host binary read-only as /splinter/bin/<name>, first on PATH. start()
snapshots each one into the trial (a hard link where the filesystem allows, else a copy), so a binary rebuilt on the
host while the trial runs changes nothing inside it. Harbor's mounts are honoured: a bind mount appears at its target,
read-only when it says so.

Host variables: Harbor fills ``${VAR}`` in a task's environment tables from the runner's own environment; a task may
import only the variables ``host_env=NAME,...`` names (none by default), so a downloaded task cannot ask for the run's
secrets. The launcher itself gets a fixed environment and its specification on standard input, and /dev is the
sandbox's own (no host shared memory, GPUs or disks).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shutil
import stat
import tomllib
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path, PurePosixPath

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.task.config import NetworkMode
from harbor.models.trial.paths import EnvironmentPaths

from splinter_harbor_worker.fscopy import copy_file, copy_tree
from splinter_harbor_worker.procgroup import signal_group
from splinter_harbor_worker.rootfs.build import ensure_shim
from splinter_harbor_worker.rootfs.images import FALLBACKS, TaskImage, ensure_task_image
from splinter_harbor_worker.rootfs.registry import RegistryClient

_LAUNCHER = Path(__file__).with_name("userns_launch.py")
_KEEPER = Path(__file__).with_name("userns_keeper.py")
_PYTHON = "/usr/bin/python3"
_TOOLS_DIR = PurePosixPath("/splinter/bin")
_HOST_RUNTIME = PurePosixPath("/splinter/host")
# Where an image sandbox's transfers bind their host side: below the per-command /dev, so nothing stays behind.
_TRANSFER = PurePosixPath("/dev/.splinter-transfer")
_CPU_TOPOLOGY = "/sys/devices/system/cpu"
_HOST_CERTIFICATES = "/etc/ssl/certs"
# How long a command's output may stay open after the command exits (a background process holding it).
_OUTPUT_GRACE_SEC = 10
# The storage of an image sandbox when the task states none (Harbor's own default task size).
_DEFAULT_STORAGE_MB = 10240
# A sandbox's image copy (mounted there in its keeper's namespace) and its overlay's upper and work directories, in
# the sandbox's directory; the image's shared layers unpacked once per node, in the image cache.
_IMAGE, _OVERLAY, _UNPACKED = "image", "overlay", "unpacked"
# Filesystems an unpacked image is not kept on: a lower layer there is refused by overlayfs, every one of an image's
# files costs a metadata round trip, and their inode quotas cannot hold unpacked trees. A copy there is held in memory.
_NETWORK_FILESYSTEMS = frozenset({"lustre", "nfs", "nfs4", "cifs", "smb3", "gpfs", "beegfs", "ceph", "glusterfs",
                                  "9p", "virtiofs"})


def on_network_filesystem(path: Path, mountinfo: str | None = None) -> bool:
    """Whether ``path`` is on a network filesystem: the mount whose point is its longest leading run of whole
    components, from ``/proc/self/mountinfo`` unless given."""
    if mountinfo is None:
        mountinfo = Path("/proc/self/mountinfo").read_text()
    parts = Path(os.path.abspath(path)).parts
    best, fstype = -1, ""
    for line in mountinfo.splitlines():
        fields = line.split()
        if " - " not in line or len(fields) < 5:
            continue
        point = Path(fields[4].replace("\\040", " ")).parts
        kind = line.split(" - ", 1)[1].split()[0]
        if parts[:len(point)] == point and len(point) > best:
            best, fstype = len(point), kind
    return fstype in _NETWORK_FILESYSTEMS or fstype.startswith("fuse.")


def copy_placement(image_root: Path, images: Path, mountinfo: str | None = None) -> dict:
    """How the keeper builds a sandbox's copy at ``image_root`` (``userns_keeper.py``): in memory only when
    ``image_root`` is on a network filesystem (hundreds of copies of a few hundred megabytes each in memory exhausted a
    node's memory limit while its local disk had room for all of them); else on that disk, as an overlay over the
    image's shared layers unpacked once per node under ``<images>/unpacked`` when the image cache is on a local disk
    too, or with every layer unpacked, and why, when it is not."""
    if on_network_filesystem(image_root, mountinfo):
        return {"in_memory": True, "lower_cache": None, "disk_reason": None}
    cache = images / _UNPACKED
    if on_network_filesystem(cache, mountinfo):
        return {"in_memory": False, "lower_cache": None,
                "disk_reason": f"the image cache {images} is on a network filesystem, which cannot hold a shared tree"}
    return {"in_memory": False, "lower_cache": str(cache), "disk_reason": None}


def remove_image_copy(sandbox: Path) -> None:
    """Remove what a sandbox's image copy left on disk (its tree, or its overlay's upper and work directories); the
    shared tree under the image cache stays for the next sandbox."""
    for name in (_IMAGE, _OVERLAY):
        remove_tree(sandbox / name)


def remove_tree(path: Path) -> None:
    """Remove a sandbox's image copy on disk, including directories the image or its commands left without write
    permission (the namespace's root owns them as this user on the host, so permissions are only restored, never
    widened for anyone else)."""

    def writable(function, failed, _exc) -> None:
        parent = os.path.dirname(failed)
        for target in (parent, failed):
            with contextlib.suppress(OSError):
                os.chmod(target, os.lstat(target).st_mode | stat.S_IRWXU, follow_symlinks=False)
        if function is os.open:
            remove_tree(Path(failed))  # a directory it could not enter (an overlay's work/work is mode 000): now it can
        else:
            function(failed)

    if os.path.lexists(path):
        shutil.rmtree(path, onexc=writable)
# Harbor's template for a host variable in a task's environment tables (harbor.utils.env).
_TEMPLATE = re.compile(r"\$\{([^}:]+)(?::-(.*))?\}")
# The whole environment of the launcher process: the runner's own may hold the run's secrets.
_LAUNCHER_ENV = {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C.UTF-8"}
# Writable sandbox paths kept in trial_dir/sandbox/<name> (the image's own tree is writable; only what Harbor fills
# from the host is bound).
_STATE_DIRS = {
    str(EnvironmentPaths.tests_dir): "tests",
    str(EnvironmentPaths.solution_dir): "solution",
}


def _host_runtime() -> tuple[Path, str] | None:
    """(the host's C library directory, its dynamic loader's file name), where the host
    keeps both together, which is how its own programs find them."""
    for line in Path("/proc/self/maps").read_text().splitlines():
        if line.rstrip().endswith("/libc.so.6"):
            libdir = Path(line.split()[-1]).resolve().parent
            loaders = sorted(libdir.glob("ld-linux-*.so*"))
            return (libdir, loaders[0].name) if loaders else None
    return None


# Where a command's output goes as it arrives: ``sink(stream, chunk)``, stream 1 for standard output and 2 for standard error.
OutputSink = Callable[[int, bytes], Awaitable[None]]


async def _pump(stream: asyncio.StreamReader, number: int, sink: OutputSink) -> None:
    """Hand ``stream`` to ``sink`` chunk by chunk until it ends."""
    while chunk := await stream.read(1 << 16):
        await sink(number, chunk)


async def _feed(stdin: asyncio.StreamWriter, keys: AsyncIterator[bytes]) -> None:
    """Write a command's input (a terminal's frames, or a plain command's bytes) as it comes; closed when it ends."""
    try:
        async for frame in keys:
            stdin.write(frame)
            await stdin.drain()
    except (ConnectionError, BrokenPipeError):
        pass  # the command has ended
    finally:
        if not stdin.is_closing():
            stdin.close()


async def _exit_status(proc: asyncio.subprocess.Process) -> int:
    """``proc``'s return code once it exits. ``Process.wait()`` returns only when the child's output pipes close too,
    which a process the command left running (a server in the background) may hold open for as long as it runs; a
    pidfd says when the child itself is gone, and asyncio's own watcher, which reaps it, sets the code."""
    loop = asyncio.get_running_loop()
    fd = os.pidfd_open(proc.pid)
    try:
        exited = loop.create_future()
        loop.add_reader(fd, lambda: exited.done() or exited.set_result(None))
        try:
            await exited
        finally:
            loop.remove_reader(fd)
    finally:
        os.close(fd)
    while proc.returncode is None:  # the watcher's callback is already scheduled on this loop
        await asyncio.sleep(0.01)
    return proc.returncode


def _parse_tools(spec: str | dict[str, str] | None) -> dict[str, Path]:
    if not spec:
        return {}
    items = spec.items() if isinstance(spec, dict) else (part.split(":", 1) for part in spec.split(",") if part)
    tools = {}
    for name, path in items:
        resolved = Path(path).expanduser().resolve()
        if not name or "/" in name or not resolved.is_file():
            raise ValueError(f"userns tools: {name!r} -> {path!r} is not a named existing file")
        tools[name] = resolved
    return tools


def _parse_ports(spec: str | int | None) -> list[int]:
    """`--ek allow_ports=` arrives as an int when it is a single port."""
    try:
        ports = [int(part) for part in str(spec or "").split(",") if part]
    except ValueError:
        raise ValueError(f"userns allow_ports: {spec!r} must be PORT[,PORT...]") from None
    if not all(0 < port < 65536 for port in ports):
        raise ValueError(f"userns allow_ports: {spec!r} holds a port outside 1-65535")
    return ports


class UserNamespaceEnvironment(BaseEnvironment):
    def __init__(self, *args, tools: str | dict[str, str] | None = None, allow_ports: str | int | None = None,
                 host_env: str | None = None, images: str | None = None, platform_fallback: str | None = None,
                 registry_mirror: str | None = None, **kwargs):
        self._host_env = {name for name in str(host_env or "").split(",") if name}
        self._images = Path(images).expanduser().resolve() if images else None
        if platform_fallback not in (None, *FALLBACKS):
            raise ValueError(f"userns platform_fallback: {platform_fallback!r} is not one of {', '.join(FALLBACKS)}")
        self._platform_fallback = platform_fallback
        self._registry_mirror = registry_mirror
        self._pulled: TaskImage | None = None  # the task's own image, resolved by start()
        self._keeper: asyncio.subprocess.Process | None = None
        self._keeper_pid: int | None = None  # the keeper itself: init of the sandbox's PID namespace
        self._copy_in_memory = True  # decided by start() from the filesystem the copy is on
        self._tools = _parse_tools(tools)
        self._allow_ports = _parse_ports(allow_ports)
        super().__init__(*args, **kwargs)
        self._refuse_host_variables()
        # One per environment instance: a trial's separate verifier must not see
        # the agent's workspace, tools or scratch space.
        self._sandbox = self.trial_paths.trial_dir / f"sandbox-{self.session_id}"
        if self._images is None or not (
                self.task_env_config.docker_image or (self.environment_dir / "Dockerfile").is_file()):
            raise ValueError(f"{self.type()} environment: runs only a task that brings its own image "
                             "(environment/Dockerfile or [environment].docker_image), with an image cache")

    def _refuse_host_variables(self) -> None:
        """Harbor fills ``${VAR}`` in a task's environment tables from the environment of the process running
        it, and that process may hold the run's secrets (the gateway's, a model server's key). A task may
        import a host variable into its sandboxes only when the run names it in ``host_env``; a downloaded
        task that asks for anything else is refused before anything starts."""
        tables = {"environment.env": dict(self.task_env_config.env)}
        task_toml = self.environment_dir.parent / "task.toml"
        if task_toml.is_file():
            document = tomllib.loads(task_toml.read_text())
            for section in ("verifier", "solution", "agent"):
                tables[f"{section}.env"] = document.get(section, {}).get("env", {}) or {}
        wanted = sorted({(table, match.group(1)) for table, env in tables.items() for value in env.values()
                         for match in _TEMPLATE.finditer(str(value)) if match.group(1) not in self._host_env})
        if wanted:
            names = ", ".join(f"{name} (in [{table}])" for table, name in wanted)
            raise ValueError(f"userns environment: the task imports host environment variable(s) {names}; a run "
                             "that means to give them to the task names them with --ek host_env=NAME,...")



    @staticmethod
    def type() -> str:
        return "userns"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(
            mounted=True, disable_internet=True, network_allowlist=True,
            network_allowlist_ipv4_addresses=True, dynamic_network_policy=True)

    async def _apply_network_policy(self, network_policy) -> None:
        """Every exec is a fresh sandbox built from the current policy; there is
        nothing to switch in place."""

    def _network_spec(self) -> dict:
        """The launcher's network policy: the (address, port) pairs IPv4 sockets may connect to (none for
        no-network); a public task also reaches everything but the host's loopback, where it reaches the gateway
        (``allow_ports`` on 127.0.0.1) and what it serves itself (recorded in ``bound``)."""
        policy = self._network_policy
        if policy.network_mode == NetworkMode.PUBLIC:
            return {"public": True, "allow": [["127.0.0.1", port] for port in self._allow_ports],
                    "bound": str(self._sandbox / "bound-ports")}
        if policy.network_mode == NetworkMode.NO_NETWORK:
            return {"allow": []}
        if not self._allow_ports:
            raise ValueError(
                "userns environment: an allowlist names addresses only; pass --ek "
                "allow_ports=PORT[,PORT...] to say which ports on them are reachable")
        return {"allow": [[host, port] for host in policy.allowed_hosts for port in self._allow_ports]}

    def _validate_definition(self) -> None:
        if shutil.which("unshare") is None:
            raise FileNotFoundError("userns environment needs util-linux unshare on PATH")

    @property
    def _workdir(self) -> str:
        """task.toml's workdir, else the image's, else the Dockerfile's last WORKDIR, else /workspace."""
        if self.task_env_config.workdir:
            return self.task_env_config.workdir
        if self._pulled:
            return self._pulled.workdir
        dockerfile = self.environment_dir / "Dockerfile"
        workdirs = [line.split(None, 1)[1].strip() for line in
                    (dockerfile.read_text().splitlines() if dockerfile.is_file() else [])
                    if line.strip().upper().startswith("WORKDIR ")]
        return workdirs[-1] if workdirs else "/workspace"

    # -- mount table -----------------------------------------------------------

    def _writable_mounts(self) -> list[tuple[PurePosixPath, Path]]:
        """(sandbox path, host path) of every writable mount, longest first."""
        state = {inside: name for inside, name in _STATE_DIRS.items()
                 if inside in (str(EnvironmentPaths.tests_dir), str(EnvironmentPaths.solution_dir))}
        mounts = [(PurePosixPath(inside), self._sandbox / name) for inside, name in state.items()]
        mounts += list(self._log_mounts().items())
        return sorted(mounts, key=lambda m: len(m[0].parts), reverse=True)

    def _log_mounts(self) -> dict[PurePosixPath, Path]:
        """Host directory behind each of /logs/{agent,verifier,artifacts}.

        Harbor says which of them a sandbox has (the agent's published
        artifacts are mirrored under artifacts/logs/artifacts and read from
        there; a separate verifier gets only its own logs). One it does not
        mention is private to this sandbox, so a verifier never sees the
        agent's logs and emptying its own artifacts directory cannot erase
        the agent's. Used without any mounts, the trial's directories apply."""
        defaults = {
            EnvironmentPaths.agent_dir: self.trial_paths.agent_dir,
            EnvironmentPaths.verifier_dir: self.trial_paths.verifier_dir,
            EnvironmentPaths.artifacts_dir: self.trial_paths.artifacts_dir,
        }
        if not self._mounts:
            return defaults
        given = {PurePosixPath(m["target"]): Path(m["source"]) for m in self._mounts}
        return {inside: given.get(inside, self._sandbox / "logs" / inside.name) for inside in defaults}

    def _host_path(self, inside: str) -> Path:
        """The host path behind sandbox path `inside`; only writable mounts map."""
        target = PurePosixPath(os.path.normpath(inside if inside.startswith("/") else f"{self._workdir}/{inside}"))
        for mount, host in self._writable_mounts():
            if target == mount or mount in target.parents:
                return host.joinpath(*target.relative_to(mount).parts)
        raise ValueError(f"userns environment: {inside} is not under a writable sandbox mount")




    def _launch_spec(self, command: str, cwd: str | None, env: dict[str, str] | None,
                     transfers: list[tuple[str, str, bool]] = ()) -> dict:
        """The launcher's specification for one command; ``transfers`` are (host, sandbox, read-only) binds
        below the command's own /dev that a file transfer needs for its length."""
        # Parents before children, so a nested mount lands on its parent's mount.
        binds = [(str(host), str(inside), False) for inside, host in reversed(self._writable_mounts())]
        binds.append((str(self._tools_dir), str(_TOOLS_DIR), True))
        binds += [(str(host), str(inside), read_only) for host, inside, read_only in self._extra_mounts()]
        runtime = _host_runtime()
        return self._own_image_spec(command, cwd, env, binds, runtime, list(transfers))

    def _own_image_spec(self, command: str, cwd: str | None, env: dict[str, str] | None,
                        binds: list[tuple[str, str, bool]], runtime: tuple[Path, str] | None,
                        transfers: list[tuple[str, str, bool]]) -> dict:
        """A command in the task's own image: the keeper's private copy is the root, writable, and the command
        runs as the image's root (the invoking user outside), as a container would run it."""
        binds = [*binds, *[(f"/etc/{name}", f"/etc/{name}", True) for name in ("resolv.conf", "hosts")
                           if Path("/etc", name).is_file()]]
        # Of the host's /sys only the CPU topology, which numeric libraries read to size their thread pools (a
        # container sees all of /sys; sysfs cannot be mounted without a network namespace of one's own).
        if Path(_CPU_TOPOLOGY).is_dir():
            binds.append((_CPU_TOPOLOGY, _CPU_TOPOLOGY, True))
        symlinks = []
        if runtime is not None:
            binds.append((str(runtime[0]), f"{_HOST_RUNTIME}/lib", True))
            symlinks.append((f"lib/{runtime[1]}", f"{_HOST_RUNTIME}/ld.so"))
        if Path(_HOST_CERTIFICATES).is_dir():
            # For programs mounted from the host (the Sven build) in an image without a certificate store.
            binds.append((_HOST_CERTIFICATES, f"{_HOST_RUNTIME}/certs", True))
        base_env = {**self._pulled.env, "HOME": "/root", "SPLINTER_WORKSPACE": self._pulled.workdir}
        base_env["PATH"] = f"{_TOOLS_DIR}:{self._pulled.env['PATH']}"
        return {
            "root": str(self._image_root), "image": True, "binds": binds, "symlinks": symlinks,
            "late_binds": transfers, "cwd": cwd or self._workdir, "env": {**base_env, **(env or {})},
            "command": command, "uid": 0, "gid": 0, "network": self._network_spec(),
        }

    @property
    def _image_root(self) -> Path:
        """Where the keeper mounts the image's private copy (empty on the host: it exists in the keeper's namespace)."""
        return self._sandbox / _IMAGE

    @property
    def _tools_dir(self) -> Path:
        return self._sandbox / "tools"

    def _extra_mounts(self) -> list[tuple[Path, PurePosixPath, bool]]:
        """Harbor's bind mounts other than the trial's own log directories, as
        (host path, sandbox path, read-only). The directory mounted at the tools
        directory is merged into it by start()."""
        own = {mount for mount, _ in self._writable_mounts()}
        mounts = []
        for mount in self._mounts:
            target = PurePosixPath(mount["target"])
            if target in own or target == _TOOLS_DIR:
                continue
            if mount["type"] != "bind":
                raise ValueError(f"userns environment: only bind mounts are supported, not {mount['type']}")
            source = Path(mount["source"]).resolve()
            if not source.exists() or not target.is_absolute():
                raise ValueError(f"userns mount {mount['source']} -> {mount['target']}: needs an existing "
                                 "host path and an absolute sandbox path")
            mounts.append((source, target, bool(mount.get("read_only"))))
        return mounts

    def _snapshot_tools(self) -> None:
        """Rebuild the tools directory: hard links (else copies) of the named
        tools, and the files of any directory mounted at /splinter/bin."""
        shutil.rmtree(self._tools_dir, ignore_errors=True)
        self._tools_dir.mkdir(parents=True)
        for mount in self._mounts:
            if PurePosixPath(mount["target"]) == _TOOLS_DIR:
                copy_tree(Path(mount["source"]), self._tools_dir, dirs_exist_ok=True)
        for name, path in self._tools.items():
            snapshot = self._tools_dir / name
            snapshot.unlink(missing_ok=True)
            try:
                os.link(path, snapshot)
            except OSError:
                copy_file(path, snapshot)

    # -- lifecycle -------------------------------------------------------------

    def _restore_setgid(self) -> None:
        """Harbor makes the trial dirs mode 0777, which clears a setgid bit
        inherited from the job directory. On a shared disk with group quotas the
        setgid group is the one that may write; without it every file of the
        trial takes the user's primary group and can fail with EDQUOT."""
        if not self.trial_paths.trial_dir.parent.stat().st_mode & stat.S_ISGID:
            return
        for d in (self.trial_paths.trial_dir, self.trial_paths.agent_dir,
                  self.trial_paths.verifier_dir, self.trial_paths.artifacts_dir):
            if d.is_dir():
                d.chmod(d.stat().st_mode | stat.S_ISGID)

    async def start(self, force_build: bool) -> None:
        del force_build
        self._restore_setgid()
        for host in {host for _, host in self._writable_mounts()} | {self._sandbox / "root"}:
            host.mkdir(parents=True, exist_ok=True)
        self._extra_mounts()  # refuses an unsupported mount before anything runs
        self._snapshot_tools()
        await self._start_own_image()
        # A separate verifier's environment directory is the task's tests/, which
        # Harbor expects to find in its workdir (/tests).
        await self._upload_environment_dir_after_start()

    async def _start_own_image(self) -> None:
        """Resolve the task's image (pulled or built once into the cache) and start the keeper of this sandbox's
        private, writable copy of it."""
        self._pulled = await asyncio.to_thread(
            ensure_task_image, self.task_env_config.docker_image, self.environment_dir, self._images,
            name=self.environment_name, platform_fallback=self._platform_fallback,
            client=RegistryClient(self._images, mirror=self._registry_mirror),
            log=lambda message: self.logger.info(message))
        self._write_image_record(None)
        self._image_root.mkdir(parents=True, exist_ok=True)
        placement = copy_placement(self._image_root, self._images)
        self._copy_in_memory = placement["in_memory"]
        size_mb = self.task_env_config.storage_mb or _DEFAULT_STORAGE_MB
        self._keeper = await asyncio.create_subprocess_exec(
            "unshare", "--user", "--map-root-user", "--mount", "--pid", "--fork", "--kill-child", "--",
            _PYTHON, "-I", str(_KEEPER),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            start_new_session=True, env=_LAUNCHER_ENV)
        shim = await asyncio.to_thread(ensure_shim, self._images)
        spec = {"mountpoint": str(self._image_root), "layers": [str(layer) for layer in self._pulled.layers],
                "size_mb": size_mb, "preload": str(shim), "overlay": str(self._sandbox / _OVERLAY), **placement}
        self._keeper.stdin.write(json.dumps(spec).encode() + b"\n")
        await self._keeper.stdin.drain()
        word, _, report = (await self._keeper.stdout.readline()).decode(errors="replace").strip().partition(" ")
        if word != "ready":
            error = (await self._keeper.stderr.read()).decode(errors="replace").strip()
            await self._stop_keeper()
            raise RuntimeError(f"userns environment: the image copy did not start: {error[-2000:]}")
        copy = json.loads(report)
        if copy["mode"] == "disk":
            self.logger.warning(f"userns: every layer of the image unpacked at {self._image_root}: {copy['reason']}")
        else:
            self.logger.info(f"userns: image copy at {self._image_root}: {copy}")
        self._write_image_record(copy)
        # `unshare --fork` stays outside the PID namespace; its one child is the keeper.
        children = Path(f"/proc/{self._keeper.pid}/task/{self._keeper.pid}/children").read_text().split()
        if len(children) != 1:
            await self._stop_keeper()
            raise RuntimeError(f"userns environment: cannot find the image keeper under {self._keeper.pid}")
        self._keeper_pid = int(children[0])

    def _write_image_record(self, copy: dict | None) -> None:
        """The trial's record of its image: provenance and working directory, and once the keeper is ready how the
        sandbox's copy was made (``userns_keeper.py``: overlay, disk with the reason, or memory)."""
        record = {**self._pulled.provenance, "workdir": self._pulled.workdir, **({"copy": copy} if copy else {})}
        path = self.trial_paths.trial_dir / f"image-{self.session_id}.json"
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    async def _stop_keeper(self) -> None:
        keeper, self._keeper, self._keeper_pid = self._keeper, None, None
        if keeper is None:
            return
        if keeper.stdin and not keeper.stdin.is_closing():
            keeper.stdin.close()
        try:
            await asyncio.wait_for(keeper.wait(), timeout=30)
        except TimeoutError:
            signal_group(keeper)
            await keeper.wait()



    async def stop(self, delete: bool) -> None:
        await self._stop_keeper()  # a copy in memory, and the overlay mount, end with the keeper's namespace
        if not self._copy_in_memory:
            await asyncio.to_thread(remove_image_copy, self._sandbox)  # what it left on disk goes with the sandbox
        if delete:
            shutil.rmtree(self._sandbox, ignore_errors=True)

    async def exec(self, command: str, cwd: str | None = None, env: dict[str, str] | None = None,
                   timeout_sec: int | None = None, user: str | int | None = None) -> ExecResult:
        del user  # one unprivileged identity inside the sandbox
        return await self._run(self._launch_spec(command, cwd, self._merge_env(env)), command, timeout_sec)

    async def exec_streaming(self, command: str, sink: OutputSink, cwd: str | None = None,
                             env: dict[str, str] | None = None, timeout_sec: int | None = None,
                             stdin: AsyncIterator[bytes] | None = None) -> int:
        """``exec`` that hands the command's output to ``sink`` as it arrives instead of keeping it; the return code.
        ``stdin`` is the command's standard input as it comes (ended when it ends; none: an empty input). Cancelling the
        call kills the command with its whole process group."""
        spec = self._launch_spec(command, cwd, self._merge_env(env))
        if stdin is not None:
            spec["stdin"] = True
        return await self._stream(spec, command, timeout_sec, sink, stdin)

    async def exec_terminal(self, command: str, *, output: Callable[[bytes], Awaitable[None]],
                            keys: AsyncIterator[bytes], size: tuple[int, int], cwd: str | None = None,
                            env: dict[str, str] | None = None, timeout_sec: int | None = None) -> int:
        """Run ``command`` on a pseudo-terminal of the sandbox's own, ``size`` (rows, columns) to begin with, for a
        person who watches and types into it: everything it writes goes to ``output`` as written, and ``keys`` yields
        the frames (``sandbox.terminal``) to put into it; ``keys`` ending hangs the terminal up. The return code;
        cancelling the call kills the command with its whole process group."""
        spec = self._launch_spec(command, cwd, self._merge_env(env))
        spec["terminal"] = {"rows": size[0], "cols": size[1]}

        async def sink(number: int, chunk: bytes) -> None:
            await output(chunk)

        return await self._stream(spec, command, timeout_sec, sink, keys)

    @property
    def workdir(self) -> str:
        """The sandbox directory commands run in unless told otherwise: the task's working directory."""
        return self._workdir



    async def _run(self, spec: dict, command: str, timeout_sec: int | None) -> ExecResult:
        output = {1: bytearray(), 2: bytearray()}

        async def keep(number: int, chunk: bytes) -> None:
            output[number] += chunk

        returncode = await self._stream(spec, command, timeout_sec, keep)
        return ExecResult(stdout=output[1].decode(errors="replace"), stderr=output[2].decode(errors="replace"),
                          return_code=returncode)

    async def _stream(self, spec: dict, command: str, timeout_sec: int | None, sink: OutputSink,
                      keys: AsyncIterator[bytes] | None = None) -> int:
        if self._keeper_pid is None:
            raise RuntimeError("userns environment: the sandbox is not started")
        # Enter the keeper's user, mount and PID namespaces, where the image copy is and whose init the keeper
        # is, and build the command's own mount namespace from there. A process the command leaves running (a
        # server started in the background) keeps running in the sandbox, as in a container, until it stops.
        argv = ["nsenter", f"--target={self._keeper_pid}", "--user", "--mount", "--pid", "--preserve-credentials",
                "--", "unshare", "--mount", "--ipc", "--uts", "--", _PYTHON, "-I", str(_LAUNCHER)]
        # The launcher gets no environment of the runner (which may hold the run's secrets) and its
        # specification on standard input, not on a command line the sandbox can read.
        proc = await asyncio.create_subprocess_exec(
            *argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, start_new_session=True, env=_LAUNCHER_ENV)
        proc.stdin.write(json.dumps(spec).encode() + b"\n")
        if keys is None:
            proc.stdin.close()
        readers = [asyncio.ensure_future(_pump(proc.stdout, 1, sink)), asyncio.ensure_future(_pump(proc.stderr, 2, sink))]
        feeder = asyncio.ensure_future(_feed(proc.stdin, keys)) if keys is not None else None
        try:
            returncode = await asyncio.wait_for(_exit_status(proc), timeout=timeout_sec)
        except (TimeoutError, asyncio.CancelledError) as stopped:
            # A command outlives neither its own time limit nor the caller's (Harbor cancels an agent at its timeout).
            signal_group(proc)
            if feeder is not None:
                feeder.cancel()
            for reader in readers:
                reader.cancel()
            if isinstance(stopped, asyncio.CancelledError):
                with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                    await asyncio.wait_for(_exit_status(proc), timeout=_OUTPUT_GRACE_SEC)  # gone when the caller goes on
                raise
            await proc.wait()
            raise TimeoutError(f"command timed out after {timeout_sec} s: {command[:200]}")
        if feeder is not None:
            feeder.cancel()  # nothing reads the seat's keys any more
            await asyncio.gather(feeder, return_exceptions=True)
        # A process the command left running may hold its output open; what it wrote by then is the output.
        _, pending = await asyncio.wait(readers, timeout=_OUTPUT_GRACE_SEC)
        for reader in pending:
            reader.cancel()
        return returncode

    # -- file transfer (host copies through the mount table) -------------------

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        try:
            target = self._host_path(target_path)
        except ValueError:
            await self._transfer_in(Path(source_path), target_path, directory=False)
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        copy_file(source_path, target)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        try:
            target = self._host_path(target_dir)
        except ValueError:
            await self._transfer_in(Path(source_dir), target_dir, directory=True)
            return
        copy_tree(source_dir, target, dirs_exist_ok=True)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            source = self._host_path(source_path)
        except ValueError:
            await self._transfer_out(source_path, Path(target_path), directory=False)
            return
        copy_file(source, target_path)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        try:
            source = self._host_path(source_dir)
        except ValueError:
            await self._transfer_out(source_dir, Path(target_dir), directory=True)
            return
        copy_tree(source, target_dir, dirs_exist_ok=True)

    # An image sandbox's own tree exists only in its keeper's namespace, so files move with a command run inside
    # the sandbox: the host side is bound for that command alone, and every path of the sandbox side is resolved
    # inside it (a link the agent planted points into the image, never at the host).

    async def _transfer_in(self, source: Path, target: str, *, directory: bool) -> None:
        inside = f"{_TRANSFER}/in"
        await self._transfer({"direction": "in", "source": inside, "target": target, "directory": directory},
                             [(str(source.resolve()), inside, True)], f"upload {source} -> {target}")

    async def _transfer_out(self, source: str, target: Path, *, directory: bool) -> None:
        holder = target if directory else target.parent
        holder.mkdir(parents=True, exist_ok=True)
        inside = f"{_TRANSFER}/out"
        await self._transfer({"direction": "out", "source": source, "directory": directory,
                              "target": inside if directory else f"{inside}/{target.name}"},
                             [(str(holder.resolve()), inside, False)], f"download {source} -> {target}")

    async def _transfer(self, job: dict, binds: list[tuple[str, str, bool]], what: str) -> None:
        spec = {**self._launch_spec("", "/", None, transfers=binds), "transfer": job}
        result = await self._run(spec, what, timeout_sec=3600)
        if result.return_code != 0:
            raise RuntimeError(f"userns environment: {what} failed: {(result.stderr or result.stdout).strip()[-500:]}")
