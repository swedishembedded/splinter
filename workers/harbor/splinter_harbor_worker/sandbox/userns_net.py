# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for running agent benchmarks in
# isolated environments on shared compute for its clients. If your team needs
# expertise in agent evaluation infrastructure then you can procure our services
# by sending an email to info@swedishembedded.com.
"""Network policy of a user-namespace sandbox: seccomp plus a connect supervisor.

The sandbox shares the host's network namespace, because hosts that need this
backend usually cannot create one (here `unshare --net` fails with ENOSPC). The
policy is enforced on system calls instead:

* ``socket`` is filtered by a seccomp program: IPv6, packet and vsock sockets
  and io_uring are always refused, IPv4 sockets are refused unless the policy
  admits destinations, and then only stream sockets exist.
* ``connect``, ``bind`` and ``listen`` are not decided by the kernel filter
  (it cannot read the address) but by a supervisor process, through seccomp
  user notification. The supervisor copies the address out of the caller's
  memory once, checks the copy against the policy and performs the call itself
  on a duplicate of the caller's socket, so the caller cannot change the
  address between check and use. IPv4 sockets may connect only to the
  ``(address, port)`` pairs the policy allows and never bind or listen; Unix
  sockets must have a filesystem path (abstract names are shared with the whole
  host, so they are refused); other families (netlink) pass.
* ``sendto``/``sendmsg``/``sendmmsg`` with MSG_FASTOPEN would connect without a
  ``connect`` call, so the filter refuses that flag.

The policy "no network" is an empty allow list. The supervisor fails closed: if
it dies, ``connect`` and friends fail with ENOSYS.

A task whose network is public shares the host's network, which on a shared node
also holds other people's loopback services and the node's own interfaces. Its
policy is supervised too: IPv4 sockets of any type may connect anywhere but the
host's loopback (127.0.0.0/8 and 0.0.0.0), where only the allowed pairs (the
gateway) and ports this sandbox itself bound are reachable; a bind to any address
is made on 127.0.0.1 instead, so a server the sandbox starts is never exposed
beyond the host's loopback. The host's loopback is one namespace for every
sandbox, so ports are the sandbox's own: a bind to port P is made on a free port
the kernel picks, recorded as P's in a file outside the sandbox (``bound``), and
a connect of the same sandbox to 127.0.0.1:P reaches it, in this command or a
later one; two sandboxes that both serve on 8080 never meet (``getsockname``
shows the real port). IPv6 is refused (programs fall back to IPv4). Datagrams sent
with ``sendto`` to a loopback port are not checked.

Standard library only: the launcher loads this file by path inside the sandbox.
"""

from __future__ import annotations

import ctypes
import os
import platform
import select
import socket
import struct
import threading

# audit arch, and system call numbers, per host machine.
_ABI = {
    "aarch64": {"arch": 0xC00000B7, "socket": 198, "connect": 203, "bind": 200, "listen": 201,
                "sendto": 206, "sendmsg": 211, "sendmmsg": 269, "io_uring_setup": 425},
    "x86_64": {"arch": 0xC000003E, "socket": 41, "connect": 42, "bind": 49, "listen": 50,
               "sendto": 44, "sendmsg": 46, "sendmmsg": 307, "io_uring_setup": 425},
}
_SYS_SECCOMP = {"aarch64": 277, "x86_64": 317}
_SYS_PIDFD_OPEN, _SYS_PIDFD_GETFD = 434, 438
_SYS_CAPSET = {"aarch64": 91, "x86_64": 126}

AF_UNIX, AF_INET, AF_INET6, AF_NETLINK, AF_PACKET, AF_VSOCK = 1, 2, 10, 16, 17, 40
SOCK_STREAM, SOCK_TYPE_MASK = 1, 0xF
MSG_FASTOPEN = 0x20000000
_DENIED_DOMAINS = (AF_INET6, AF_PACKET, AF_VSOCK)
_SUPERVISED = ("connect", "bind", "listen")
EACCES, ENOSYS, EINVAL, ENAMETOOLONG, EFAULT, EBADF, EADDRINUSE = 13, 38, 22, 36, 14, 9, 98

# BPF opcodes
_LD_ABS, _AND_K, _JEQ_K, _JSET_K, _RET_K = 0x20, 0x54, 0x15, 0x45, 0x06
_ALLOW, _KILL, _NOTIFY = 0x7FFF0000, 0x80000000, 0x7FC00000
_DENY = 0x00050000 | EACCES
_OFF_NR, _OFF_ARCH = 0, 4


def _arg(i: int) -> int:
    return 16 + 8 * i  # low word of args[i] on little-endian hosts


class _Program:
    """A classic BPF program written with labels, resolved to jump offsets."""

    def __init__(self) -> None:
        self._ops: list = []
        self._labels: dict[str, int] = {}

    def label(self, name: str) -> None:
        self._labels[name] = len(self._ops)

    def load(self, offset: int) -> None:
        self._ops.append((_LD_ABS, offset, None, None))

    def mask(self, k: int) -> None:
        self._ops.append((_AND_K, k, None, None))

    def if_eq(self, k: int, then: str, otherwise: str | None = None) -> None:
        self._ops.append((_JEQ_K, k, then, otherwise))

    def if_set(self, k: int, then: str, otherwise: str | None = None) -> None:
        self._ops.append((_JSET_K, k, then, otherwise))

    def ret(self, value: int) -> None:
        self._ops.append((_RET_K, value, None, None))

    def assemble(self) -> bytes:
        out = b""
        for pc, (code, k, then, otherwise) in enumerate(self._ops):
            jt = jf = 0
            if code in (_JEQ_K, _JSET_K):
                jt = self._labels[then] - pc - 1
                jf = 0 if otherwise is None else self._labels[otherwise] - pc - 1
            out += struct.pack("<HBBI", code, jt, jf, k)
        return out


def filter_program(allow_inet: bool, machine: str | None = None, public: bool = False) -> bytes:
    """The seccomp program of a sandbox.

    ``allow_inet``: IPv4 stream sockets may exist (their connects are decided
    by the supervisor); otherwise any IPv4 socket is refused. ``public``: IPv4
    sockets of every type may exist (datagrams for name resolution)."""
    abi = _ABI[machine or platform.machine()]
    p = _Program()
    p.load(_OFF_ARCH)
    p.if_eq(abi["arch"], "arch_ok")
    p.ret(_KILL)  # a syscall of any other ABI
    p.label("arch_ok")
    p.load(_OFF_NR)
    p.if_eq(abi["io_uring_setup"], "deny")
    p.if_eq(abi["socket"], "socket")
    for name in _SUPERVISED:
        p.if_eq(abi[name], "notify")
    if allow_inet or public:
        p.if_eq(abi["sendto"], "flags3")
        p.if_eq(abi["sendmmsg"], "flags3")
        p.if_eq(abi["sendmsg"], "flags2")
    p.ret(_ALLOW)
    p.label("socket")
    p.load(_arg(0))
    for domain in _DENIED_DOMAINS:
        p.if_eq(domain, "deny")
    p.if_eq(AF_INET, "inet")
    p.ret(_ALLOW)
    p.label("inet")
    if public:
        p.ret(_ALLOW)
    elif allow_inet:
        p.load(_arg(1))
        p.mask(SOCK_TYPE_MASK)
        p.if_eq(SOCK_STREAM, "allow")
    p.label("deny")
    p.ret(_DENY)
    # Jumps only go forward: the flag checks branch to a refusal of their own.
    p.label("flags3")
    p.load(_arg(3))
    p.if_set(MSG_FASTOPEN, "refuse")
    p.ret(_ALLOW)
    p.label("flags2")
    p.load(_arg(2))
    p.if_set(MSG_FASTOPEN, "refuse")
    p.label("allow")
    p.ret(_ALLOW)
    p.label("notify")
    p.ret(_NOTIFY)
    p.label("refuse")
    p.ret(_DENY)
    return p.assemble()


# -- supervisor ------------------------------------------------------------------

_SECCOMP_SET_MODE_FILTER, _FLAG_NEW_LISTENER = 1, 8
_IOCTL_NOTIF_RECV, _IOCTL_NOTIF_SEND, _IOCTL_NOTIF_ID_VALID = 0xC0502100, 0xC0182101, 0x40082102
_libc = ctypes.CDLL(None, use_errno=True)
_libc.ioctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p]
_libc.connect.argtypes = _libc.bind.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
_libc.listen.argtypes = [ctypes.c_int, ctypes.c_int]
_libc.syscall.restype = ctypes.c_long


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]


class _Data(ctypes.Structure):
    _fields_ = [("nr", ctypes.c_int), ("arch", ctypes.c_uint32), ("ip", ctypes.c_uint64),
                ("args", ctypes.c_uint64 * 6)]


class _Notif(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint64), ("pid", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("data", _Data)]


class _Resp(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint64), ("val", ctypes.c_int64), ("error", ctypes.c_int32),
                ("flags", ctypes.c_uint32)]


def install_filter(program: bytes, supervised: bool) -> int:
    """Install the program in this thread group; returns the notification
    listener fd when ``supervised`` (else -1). no_new_privs must already be set."""
    buf = ctypes.create_string_buffer(program, len(program))
    prog = _SockFprog(len(program) // 8, ctypes.addressof(buf))
    flags = _FLAG_NEW_LISTENER if supervised else 0
    fd = _libc.syscall(_SYS_SECCOMP[platform.machine()], _SECCOMP_SET_MODE_FILTER, flags,
                       ctypes.byref(prog))
    if fd < 0:
        err = ctypes.get_errno()
        raise OSError(err, f"seccomp: {os.strerror(err)}")
    return fd if supervised else -1


_LOOPBACK = socket.inet_aton("127.0.0.1")


class Policy:
    """Where sandboxed IPv4 sockets may connect, and whether they may serve (public only, on loopback)."""

    def __init__(self, allow: list[tuple[str, int]], public: bool = False, bound: int | None = None) -> None:
        self.allow = {(socket.inet_aton(ip), port) for ip, port in allow}
        self.public = public
        self._bound = bound  # descriptor of the sandbox's record of the ports it bound (public only)

    def destination(self, sockaddr: bytes) -> bytes | None:
        """The address a connect to ``sockaddr`` is made to, or None when the policy refuses it: an allowed pair as
        it is, in a public sandbox anything off the host's loopback as it is and one of its own ports on loopback
        mapped to where that port really is."""
        if len(sockaddr) < 8:
            return None
        port, ip = struct.unpack("!H4s", sockaddr[2:8])
        if (ip, port) in self.allow:
            return sockaddr
        if not self.public:
            return None
        if ip[0] != 127 and ip != b"\0\0\0\0":
            return sockaddr  # anywhere but the host's loopback
        actual = self.ports().get(port)
        if actual is None:
            return None
        return sockaddr[:2] + struct.pack("!H", actual) + _LOOPBACK + sockaddr[8:]

    def ports(self) -> dict[int, int]:
        """The sandbox's own ports: the port each bind asked for, and the one it got (the latest bind wins)."""
        if self._bound is None:
            return {}
        ports = {}
        for line in os.pread(self._bound, 1 << 20, 0).decode().splitlines():
            asked, _, actual = line.partition(" ")
            if asked.isdigit() and actual.isdigit():
                ports[int(asked)] = int(actual)
        return ports

    def serving(self, asked: int) -> bool:
        """Whether the sandbox still listens on its port ``asked``: the port standing for it is held by a listening
        socket (a probe bind with SO_REUSEADDR fails only then, not for connections left in TIME_WAIT)."""
        actual = self.ports().get(asked) if asked else None
        if actual is None:
            return False
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", actual))
            except OSError as exc:
                return exc.errno == EADDRINUSE
        return False

    def record_bound(self, asked: int, actual: int) -> None:
        if self._bound is not None and actual:
            os.write(self._bound, f"{asked or actual} {actual}\n".encode())  # O_APPEND: one write is one line


def _errno_call(function, *args) -> int:
    """0 on success, else the errno of the failed C call."""
    return 0 if function(*args) == 0 else ctypes.get_errno()


def _thread_group(tid: int) -> int:
    """The process a thread belongs to. Notifications name the calling thread,
    and pidfd_open refuses anything but a thread-group leader before Linux 6.9."""
    with open(f"/proc/{tid}/status") as status:
        for line in status:
            if line.startswith("Tgid:"):
                return int(line.split()[1])
    raise OSError(EBADF, "no thread group")


def _getfd(pid: int, fd: int) -> int:
    pidfd = _libc.syscall(_SYS_PIDFD_OPEN, _thread_group(pid), 0)
    if pidfd < 0:
        raise OSError(ctypes.get_errno(), "pidfd_open")
    try:
        got = _libc.syscall(_SYS_PIDFD_GETFD, pidfd, fd, 0)
        if got < 0:
            raise OSError(ctypes.get_errno(), "pidfd_getfd")
        return got
    finally:
        os.close(pidfd)


def _read_memory(pid: int, address: int, length: int) -> bytes:
    with open(f"/proc/{pid}/mem", "rb", buffering=0) as mem:
        mem.seek(address)
        data = mem.read(length)
    if len(data) != length:
        raise OSError(EFAULT, "short read of the caller's memory")
    return data


class Supervisor:
    """Answers connect/bind/listen notifications for one sandbox."""

    def __init__(self, listener: int, policy: Policy, machine: str | None = None) -> None:
        self._listener = listener
        self._policy = policy
        self._nr = {_ABI[machine or platform.machine()][name]: name for name in _SUPERVISED}

    def serve(self) -> None:
        listener = select.poll()
        listener.register(self._listener, select.POLLIN)
        while True:
            # Every process under the filter gone (the command and whatever it left running): the listener hangs up.
            events = dict(listener.poll())
            if events.get(self._listener, 0) & (select.POLLHUP | select.POLLERR) and not \
                    events.get(self._listener, 0) & select.POLLIN:
                return
            notif = _Notif()
            if _libc.ioctl(self._listener, _IOCTL_NOTIF_RECV, ctypes.byref(notif)) != 0:
                if ctypes.get_errno() in (4, 2):  # EINTR; ENOENT: the caller was killed
                    continue
                return  # the last filtered process is gone
            threading.Thread(target=self._answer, args=(notif,), daemon=True).start()

    def _answer(self, notif: _Notif) -> None:
        try:
            error = self._decide(notif)
        except OSError as exc:
            error = exc.errno or EACCES
        except Exception:  # fail closed on any bug of ours
            error = EACCES
        valid = ctypes.c_uint64(notif.id)
        if _libc.ioctl(self._listener, _IOCTL_NOTIF_ID_VALID, ctypes.byref(valid)) != 0:
            return  # the caller is gone
        resp = _Resp(id=notif.id, val=0, error=-error, flags=0)
        _libc.ioctl(self._listener, _IOCTL_NOTIF_SEND, ctypes.byref(resp))

    def _decide(self, notif: _Notif) -> int:
        """errno for the caller's call: 0 after performing it successfully."""
        name = self._nr[notif.data.nr]
        args = notif.data.args
        sock = _getfd(notif.pid, args[0] & 0xFFFFFFFF)
        try:
            domain = socket.socket(fileno=os.dup(sock)).family  # also proves it is a socket
            if name == "listen":
                return self._listen(sock, domain, int(ctypes.c_int32(args[1] & 0xFFFFFFFF).value))
            length = args[2] & 0xFFFFFFFF
            if not 2 <= length <= 128:
                return EINVAL
            raw = _read_memory(notif.pid, args[1], length)
            address = self._checked_address(notif.pid, name, domain, raw)
            if name == "bind" and domain == AF_INET and self._policy.serving(struct.unpack("!H", raw[2:4])[0]):
                return EADDRINUSE  # as the kernel answers a second bind of a port in use (nginx -t relies on it)
            call = _libc.connect if name == "connect" else _libc.bind
            error = _errno_call(call, sock, address, len(address))
            if not error and name == "bind" and domain == AF_INET:
                with socket.socket(fileno=os.dup(sock)) as bound:
                    self._policy.record_bound(struct.unpack("!H", raw[2:4])[0], bound.getsockname()[1])
            return error
        finally:
            os.close(sock)

    def _listen(self, sock: int, domain: int, backlog: int) -> int:
        if domain == AF_INET:
            # Only a public sandbox serves, and its binds are on loopback.
            return _errno_call(_libc.listen, sock, backlog) if self._policy.public else EACCES
        if domain not in (AF_UNIX, AF_NETLINK):
            return EACCES
        return _errno_call(_libc.listen, sock, backlog)

    def _checked_address(self, pid: int, call: str, domain: int, raw: bytes):
        family = struct.unpack("=H", raw[:2])[0]
        if family != domain:
            raise OSError(EINVAL, "address family does not match the socket")
        if domain == AF_INET:
            if call == "bind" and self._policy.public and len(raw) >= 8:
                # Served on the host's loopback only, whatever address the program asked for, on a port of the
                # kernel's choosing that stands for the one asked (Policy.destination).
                local = raw[:2] + b"\0\0" + _LOOPBACK + raw[8:]
                return ctypes.create_string_buffer(local, len(local))
            destination = self._policy.destination(raw) if call == "connect" else None
            if destination is None:
                raise OSError(EACCES, "destination not allowed")
            return ctypes.create_string_buffer(destination, len(destination))
        if domain == AF_UNIX:
            return self._unix_address(pid, raw)
        if domain == AF_NETLINK:
            return ctypes.create_string_buffer(raw, len(raw))
        raise OSError(EACCES, "address family not allowed")

    @staticmethod
    def _unix_address(pid: int, raw: bytes):
        path = raw[2:].split(b"\0", 1)[0]
        if not path:  # unnamed or abstract: abstract names are shared with the whole host
            raise OSError(EACCES, "abstract or unnamed unix address")
        if not path.startswith(b"/"):
            path = os.fsencode(os.readlink(f"/proc/{pid}/cwd")) + b"/" + path
        if len(path) >= 108:
            raise OSError(ENAMETOOLONG, "unix socket path too long")
        packed = struct.pack("=H", AF_UNIX) + path + b"\0"
        return ctypes.create_string_buffer(packed, len(packed))


_CAP_SYS_PTRACE, _CAP_V3 = 19, 0x20080522


def _keep_only_ptrace() -> None:
    """Reduce the supervisor to CAP_SYS_PTRACE in the namespace that owns the
    sandbox. The sandboxed command runs in a namespace below it, and its
    process is not dumpable (its binary is owned by an id the sandbox does not
    map), so only an ancestor with this capability may read its memory and
    take its descriptors."""

    class Header(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]

    class Data(ctypes.Structure):
        _fields_ = [("effective", ctypes.c_uint32), ("permitted", ctypes.c_uint32),
                    ("inheritable", ctypes.c_uint32)]

    data = (Data * 2)()
    data[0].effective = data[0].permitted = 1 << _CAP_SYS_PTRACE
    if _libc.syscall(_SYS_CAPSET[platform.machine()], ctypes.byref(Header(_CAP_V3, 0)), data) != 0:
        raise OSError(ctypes.get_errno(), "capset")
    _libc.prctl(38, 1, 0, 0, 0)  # PR_SET_NO_NEW_PRIVS


def _blank_command_line() -> None:
    """Overwrite this process's argument strings in place. The supervisor lives in the sandbox's PID
    namespace, where every process can read its command line, and it inherited the launcher's, which
    names the launcher's host path (the checkout, the account's home)."""
    with open("/proc/self/stat", "rb") as stat:
        fields = stat.read().rsplit(b")", 1)[1].split()
    start, end = int(fields[45]), int(fields[46])  # arg_start, arg_end: fields 48 and 49 of stat(5)
    if 0 < start < end:
        with open("/proc/self/mem", "r+b", buffering=0) as memory:
            memory.seek(start)
            memory.write(b"\0" * (end - start))


def start_supervisor(policy: Policy) -> tuple[int, int]:
    """Fork the supervisor; returns (its pid, the socket to send it the listener on).

    Call after the sandbox root is in place and before the caller enters its
    own, unprivileged user namespace: the supervisor stays in the namespace
    that owns the sandbox, reduced to the one capability it needs, and refuses
    ptrace and /proc access by others."""
    mine, theirs = socket.socketpair()
    pid = os.fork()
    if pid:
        theirs.close()
        return pid, mine.detach()
    code = 1
    try:
        mine.close()
        _blank_command_line()
        _libc.prctl(4, 0, 0, 0, 0)  # PR_SET_DUMPABLE: nobody may attach to the supervisor
        _keep_only_ptrace()
        # It lives as long as a process under the filter does (a server the command leaves running keeps its policy),
        # and holds none of the command's output open meanwhile.
        null = os.open("/dev/null", os.O_RDWR)
        for fd in (0, 1, 2):
            os.dup2(null, fd)
        os.close(null)
        _, fds, _, _ = socket.recv_fds(theirs, 16, 1)
        theirs.close()
        Supervisor(fds[0], policy).serve()
        code = 0
    finally:
        os._exit(code)
