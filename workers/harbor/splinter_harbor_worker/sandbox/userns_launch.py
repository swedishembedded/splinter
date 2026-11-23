# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
"""Build a trial sandbox and run one command in it.

Run as `unshare --user --map-root-user --mount --pid --fork --ipc --uts --
python3 -I userns_launch.py` with the specification (JSON) on standard input, never
in the arguments: every process in the sandbox can read the launcher's command line
(the network supervisor keeps it), and the specification names host paths. Standard
input is /dev/null again before the command starts. In the fresh mount namespace it mounts
a tmpfs as the new root, recursively binds the host's /usr and /etc read-only
plus the writable state directories, pivots into the new root and detaches the
old one, so nothing else of the host is reachable. It then enters a nested user
namespace that maps the invoking user's uid and gid onto that root, so the
command runs as the invoking uid (peer-credential authentication on a bound
socket, D-Bus EXTERNAL for one, sees a consistent identity). Finally it empties
the capability bounding set, sets no_new_privs, installs the seccomp filter (if
any) and execs the command; exec as a non-root uid clears the remaining
capabilities and no_new_privs keeps them from coming back, so the command can
change nothing it was not given.

/dev is the sandbox's own: a tmpfs holding the character devices programs need
(null, zero, full, random, urandom, tty, bound from the host), a fresh devpts instance
for pseudo-terminals and a private /dev/shm. The host's /dev would give every sandbox
the host's shared memory (a channel between trials and to whatever the host keeps
there), its GPUs and its disks.

Recursive binds are deliberate: a root filesystem marked unbindable (as an
Apptainer overlay is) refuses a plain bind of any directory under it.
The network policy (``spec["network"]``: ``{"allow": [[ip, port], ...]}``, with
``"public": true`` and ``"bound": PATH`` for a task on the public network) is
enforced by userns_net.py: a seccomp filter plus a supervisor process that
decides every connect, bind and listen.

`--no-network UID GID COMMAND...` applies the same confinement (identity, capabilities, network
policy "none") to a command in the host's own file system; `benchlib.netless` builds it.

A command may run on a terminal (``spec["terminal"]``: ``{"rows": R, "cols": C}``), for a person who watches and types
into it (``bench task play``). The terminal is a pseudo-terminal of the sandbox's own devpts, opened after the pivot:
nothing of the runner's terminal enters the sandbox. The command is its session leader with the terminal as its standard
streams; this launcher stays outside the confinement as the relay, writing what the command writes to its standard output
as it was written, and reading frames from its standard input, which then stays connected (the specification is its first
line): a one-byte kind and a four-byte big-endian length, then the payload, ``d`` bytes typed into the terminal and ``w`` a
new size (two big-endian 16-bit numbers, rows and columns). Standard input ending hangs the terminal up. A command given
``spec["stdin"]`` (no terminal) reads the rest of the launcher's standard input as its own.

Standard library only.
"""

from __future__ import annotations

import ctypes
import fcntl
import importlib.util
import json
import os
import platform
import select
import shutil
import signal
import socket
import struct
import sys
import termios

MS_RDONLY, MS_NOSUID, MS_NODEV, MS_REMOUNT, MS_BIND, MS_REC, MS_PRIVATE = 1, 2, 4, 32, 4096, 16384, 1 << 18
MNT_DETACH = 2
PR_SET_NO_NEW_PRIVS, PR_CAPBSET_DROP = 38, 24
CAP_LAST = 63
CLONE_NEWUSER = 0x10000000
PIVOT_ROOT = {"aarch64": 41, "x86_64": 155}

libc = ctypes.CDLL(None, use_errno=True)


def _check(ret: int, what: str) -> None:
    if ret != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"{what}: {os.strerror(err)}")


def mount(src: str | None, dst: str, fstype: str | None, flags: int, data: str | None = None) -> None:
    _check(libc.mount(src and src.encode(), dst.encode(), fstype and fstype.encode(), flags,
                      data and data.encode()), f"mount {src} -> {dst}")


def bind(src: str, dst: str, read_only: bool) -> None:
    if os.path.isdir(src):
        os.makedirs(dst, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.lexists(dst):  # an existing file may sit on a read-only mount
            open(dst, "a").close()
    mount(src, dst, None, MS_BIND | MS_REC)
    if read_only:
        mount(None, dst, None, MS_REMOUNT | MS_BIND | MS_RDONLY | MS_NOSUID | MS_NODEV)


DEVICES = ("null", "zero", "full", "random", "urandom", "tty")


def make_dev(dev: str) -> None:
    """A minimal /dev at ``dev`` (as bubblewrap's --dev builds it)."""
    os.makedirs(dev, exist_ok=True)
    mount("tmpfs", dev, "tmpfs", MS_NOSUID, "mode=0755")
    for name in DEVICES:
        bind(f"/dev/{name}", f"{dev}/{name}", read_only=False)
    os.makedirs(f"{dev}/pts")
    mount("devpts", f"{dev}/pts", "devpts", MS_NOSUID, "newinstance,ptmxmode=0666,mode=0620")
    os.symlink("pts/ptmx", f"{dev}/ptmx")
    os.makedirs(f"{dev}/shm")
    mount("tmpfs", f"{dev}/shm", "tmpfs", MS_NOSUID | MS_NODEV, "mode=1777")
    for name, target in (("fd", "/proc/self/fd"), ("stdin", "/proc/self/fd/0"), ("stdout", "/proc/self/fd/1"),
                         ("stderr", "/proc/self/fd/2")):
        os.symlink(target, f"{dev}/{name}")


def become(uid: int, gid: int) -> None:
    """Enter a user namespace where `uid`/`gid` map onto the current root."""
    _check(libc.unshare(CLONE_NEWUSER), "unshare user namespace")
    for name, text in (("setgroups", "deny"), ("uid_map", f"{uid} 0 1"), ("gid_map", f"{gid} 0 1")):
        with open(f"/proc/self/{name}", "w") as f:
            f.write(text)


def load_net():
    """userns_net.py, next to this file (the script runs with -I, so no sibling imports)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "userns_net.py")
    module_spec = importlib.util.spec_from_file_location("userns_net", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


FRAME = struct.Struct(">cI")
TYPED, RESIZE = b"d", b"w"
SIZE = struct.Struct(">HH")


def read_spec() -> dict:
    """The specification: the first line of standard input, read a byte at a time so that nothing after it (a terminal's
    frames) is taken with it."""
    data = bytearray()
    while (byte := os.read(0, 1)) and byte != b"\n":
        data += byte
    return json.loads(data)


def main() -> int:
    spec = read_spec()
    if "terminal" not in spec and not spec.get("stdin"):
        devnull = os.open("/dev/null", os.O_RDONLY)
        os.dup2(devnull, 0)
        os.close(devnull)
    root = spec["root"]
    image = bool(spec.get("image"))
    network = dict(spec["network"])
    if network.get("bound"):
        # The record of the ports this sandbox serves on, outside its reach (the supervisor keeps the descriptor).
        network["bound_fd"] = os.open(network["bound"], os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_CLOEXEC, 0o600)
    net = load_net()  # before pivot_root hides the host's files
    mount(None, "/", None, MS_REC | MS_PRIVATE)
    if image:
        # The task's own image, a private writable copy (userns_keeper.py): mounted on itself so it can be the root.
        mount(root, root, None, MS_BIND | MS_REC)
    else:
        mount("tmpfs", root, "tmpfs", MS_NOSUID | MS_NODEV, "mode=0755")
    for src, dst, read_only in spec["binds"]:
        bind(src, root + dst, read_only)
    for target, link in spec["symlinks"]:
        if image and os.path.lexists(root + link):
            os.unlink(root + link)  # left by an earlier command in the persistent image
        os.makedirs(os.path.dirname(root + link), exist_ok=True)
        os.symlink(target, root + link)
    os.makedirs(root + "/proc", exist_ok=True)
    mount("proc", root + "/proc", "proc", MS_NOSUID | MS_NODEV)
    make_dev(root + "/dev")
    for src, dst, read_only in spec.get("late_binds", ()):  # a file transfer's host side, gone with this command
        bind(src, root + dst, read_only)
    os.makedirs(root + "/.oldroot", exist_ok=image)
    _check(libc.syscall(PIVOT_ROOT[platform.machine()], root.encode(), (root + "/.oldroot").encode()), "pivot_root")
    os.chdir("/")
    _check(libc.umount2(b"/.oldroot", MNT_DETACH), "umount old root")
    os.rmdir("/.oldroot")
    if not image:
        mount(None, "/", None, MS_REMOUNT | MS_BIND | MS_RDONLY | MS_NOSUID | MS_NODEV)
    if "transfer" in spec:
        return transfer(spec["transfer"])
    os.chdir(spec["cwd"])
    argv = ["/bin/bash", "-c", spec["command"]]
    if "terminal" in spec:
        return run_on_terminal(net, network, spec, argv)
    exec_confined(net, network, spec["uid"], spec["gid"], argv, spec["env"])
    return 127


def set_size(fd: int, rows: int, cols: int) -> None:
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))


def run_on_terminal(net, network, spec: dict, argv: list[str]) -> int:
    """Run ``argv`` confined, on a new pseudo-terminal, relaying it to and from this process's standard streams."""
    master, slave = os.openpty()  # the sandbox's /dev/ptmx: its own devpts instance
    set_size(master, spec["terminal"]["rows"], spec["terminal"]["cols"])
    pid = os.fork()
    if pid == 0:
        try:
            os.close(master)
            os.setsid()
            fcntl.ioctl(slave, termios.TIOCSCTTY, 0)
            for fd in (0, 1, 2):
                os.dup2(slave, fd)
            os.close(slave)
            exec_confined(net, network, spec["uid"], spec["gid"], argv, spec["env"])
        finally:
            os._exit(127)
    os.close(slave)
    return relay(master, pid)


def relay(master: int, pid: int) -> int:
    """Copy the terminal's output to standard output and standard input's frames into the terminal until the command
    ends; its exit status as a shell gives it."""
    pending = bytearray()
    reading, open_master = True, True
    while True:
        watched = [fd for fd, on in ((master, open_master), (0, reading)) if on]
        ready = select.select(watched, [], [], 0.2)[0] if watched else []
        if master in ready:
            open_master = copy_out(master)
        if 0 in ready:
            chunk = os.read(0, 1 << 16)
            if not chunk:
                reading = False  # the seat is gone: hang the terminal up
                with_group(pid, signal.SIGHUP)
            pending += chunk
            while len(pending) >= FRAME.size and len(pending) >= FRAME.size + FRAME.unpack_from(pending)[1]:
                kind, size = FRAME.unpack_from(pending)
                payload = bytes(pending[FRAME.size:FRAME.size + size])
                del pending[:FRAME.size + size]
                if kind == TYPED:
                    write_all(master, payload)
                elif kind == RESIZE and size == SIZE.size:
                    set_size(master, *SIZE.unpack(payload))
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            while open_master and select.select([master], [], [], 0.2)[0]:
                open_master = copy_out(master)
            return os.waitstatus_to_exitcode(status) if status & 0x7F == 0 else 128 + os.WTERMSIG(status)


def copy_out(master: int) -> bool:
    """One read of the terminal onto standard output; False once the terminal is closed on every side."""
    try:
        data = os.read(master, 1 << 16)
    except OSError:
        return False
    if data:
        write_all(1, data)
    return bool(data)


def write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


def with_group(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


def transfer(job: dict) -> int:
    """Copy a file or directory between a transfer bind (the host side) and the sandbox, after the pivot: every
    sandbox path resolves inside the sandbox, and links are followed on the way out, so the host receives files and
    never a link into the host. Runs with this launcher's code alone (the image needs no tools for it)."""
    source, target, outward = job["source"], job["target"], job["direction"] == "out"
    try:
        if job["directory"]:
            shutil.copytree(source, target, symlinks=not outward, dirs_exist_ok=True)
        else:
            os.makedirs(os.path.dirname(target) or "/", exist_ok=True)
            shutil.copy2(source, target)
    except (OSError, shutil.Error) as exc:
        print(f"transfer {source} -> {target}: {exc}", file=sys.stderr)
        return 1
    return 0


def exec_confined(net, network, uid: int, gid: int, argv: list[str], env: dict[str, str] | None) -> None:
    """Replace this process by ``argv`` under the sandbox's confinement: run as ``uid``/``gid``
    in a nested user namespace, with no capabilities, no_new_privs and the network policy."""
    public = bool(network.get("public"))
    policy = net.Policy([tuple(entry) for entry in network["allow"]], public=public, bound=network.get("bound_fd"))
    supervisor = net.start_supervisor(policy)
    become(uid, gid)

    for cap in range(CAP_LAST + 1):
        if libc.prctl(PR_CAPBSET_DROP, cap, 0, 0, 0) != 0 and ctypes.get_errno() != 22:  # EINVAL: no such cap
            _check(-1, f"drop capability {cap} from the bounding set")
    _check(libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0), "no_new_privs")
    listener = net.install_filter(net.filter_program(bool(network["allow"]), public=public), supervised=True)
    _, channel = supervisor
    with socket.socket(fileno=channel) as link:
        socket.send_fds(link, [b"x"], [listener])
    os.close(listener)  # the sandboxed command must not hold the listener
    if env is None:
        os.execvp(argv[0], argv)
    os.execve(argv[0], argv, env)


def main_no_network(argv: list[str]) -> int:
    """``--no-network UID GID COMMAND...``: run COMMAND in the host's file system with the
    sandbox's no-network policy (the verifier's), for measurements made outside a sandbox.
    Run under ``unshare --user --map-root-user``; UID and GID are the invoking user's."""
    uid, gid, *command = argv
    exec_confined(load_net(), {"allow": []}, int(uid), int(gid), command, None)
    return 127


if __name__ == "__main__":
    sys.exit(main_no_network(sys.argv[2:]) if sys.argv[1:2] == ["--no-network"] else main())
