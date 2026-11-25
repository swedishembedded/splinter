# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements graders that run agent-written code apart
# from the checks that judge it, for its clients. If your team needs
# expertise in verifying untrusted code, you can procure our services by
# sending an email to info@swedishembedded.com.

"""A check of a Python answer, run apart from the answer (the python3
runtime's separate guard, crates/sandbox/src/runtime.rs).

This file is the check's process. It starts the answer in a second
interpreter (this file again, with --serve), which runs the answer's code
and reports the names it defines. Each function the answer defines is a
proxy here: calling it sends the arguments to the answer's process and
returns what the function returned. Arguments and results cross as plain
data only (None, bool, int, float, str, bytes, and lists, tuples, sets and
dicts of them), by exact type, so no object of the answer's (one equal to
everything, an int whose == is rewritten) ever reaches the check, and a
tuple stays a tuple. What the answer prints is written here, in order,
before the call returns. A name the answer defines that is also a builtin
(print, len) stays the check's own. An exception the answer raises is
raised here as the builtin exception of that name, else as RuntimeError;
an answer that ends its process fails the check. ANSWER and CHECK are the
two sources, as string literals.
"""
import json
import os
import subprocess
import sys

ANSWER = {answer}
CHECK = {check}


def _encode(value):
    """Plain data, by exact type, tagged so a tuple stays a tuple."""
    kind = type(value)
    if value is None or kind in (bool, int, str):
        return value
    if kind is float:
        if value != value or value in (float("inf"), float("-inf")):
            return {"float": repr(value)}
        return value
    if kind is list:
        return {"list": [_encode(v) for v in value]}
    if kind is tuple:
        return {"tuple": [_encode(v) for v in value]}
    if kind in (set, frozenset):
        return {kind.__name__: [_encode(v) for v in value]}
    if kind is dict:
        return {"dict": [[_encode(k), _encode(v)] for k, v in value.items()]}
    if kind is bytes:
        return {"bytes": value.hex()}
    raise TypeError(f"{kind.__name__} is not plain data")


def _decode(value):
    if not isinstance(value, dict):
        return value
    (kind, inner), = value.items()
    if kind == "float":
        return float(inner)
    if kind == "list":
        return [_decode(v) for v in inner]
    if kind == "tuple":
        return tuple(_decode(v) for v in inner)
    if kind == "set":
        return {_decode(v) for v in inner}
    if kind == "frozenset":
        return frozenset(_decode(v) for v in inner)
    if kind == "dict":
        return {_decode(k): _decode(v) for k, v in inner}
    if kind == "bytes":
        return bytes.fromhex(inner)
    raise ValueError(kind)


def _serve(requests, replies):
    import io as _io
    incoming = os.fdopen(requests, "r")
    outgoing = os.fdopen(replies, "w")
    namespace = {"__name__": "__main__"}

    def send(message, out):
        message["out"] = out
        outgoing.write(json.dumps(message) + "\n")
        outgoing.flush()

    captured = _io.StringIO()
    saved = sys.stdout
    sys.stdout = captured
    try:
        exec(compile(ANSWER, "<answer>", "exec"), namespace)
        loaded = {"ok": True}
    except BaseException as error:
        loaded = {"ok": False, "error": type(error).__name__, "message": str(error)}
    finally:
        sys.stdout = saved
    if loaded["ok"]:
        functions, values = [], {}
        for name, value in list(namespace.items()):
            if name.startswith("__") or type(value).__name__ == "module":
                continue
            if callable(value):
                functions.append(name)
                continue
            try:
                values[name] = _encode(value)
            except TypeError:
                pass
        loaded["functions"], loaded["values"] = functions, values
    send(loaded, captured.getvalue())
    for line in incoming:
        request = json.loads(line)
        captured = _io.StringIO()
        sys.stdout = captured
        try:
            target = namespace[request["name"]]
            args = [_decode(a) for a in request["args"]]
            kwargs = {k: _decode(v) for k, v in request["kwargs"]}
            message = {"ok": True, "value": _encode(target(*args, **kwargs))}
        except BaseException as error:
            message = {"ok": False, "error": type(error).__name__, "message": str(error)}
        finally:
            sys.stdout = saved
        send(message, captured.getvalue())


def _check():
    import builtins
    requests_r, requests_w = os.pipe()
    replies_r, replies_w = os.pipe()
    child = subprocess.Popen(
        [sys.executable, "-I", os.path.abspath(sys.argv[0]), "--serve", str(requests_r), str(replies_w)],
        pass_fds=(requests_r, replies_w),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.close(requests_r)
    os.close(replies_w)
    to_answer = os.fdopen(requests_w, "w")
    from_answer = os.fdopen(replies_r, "r")

    def receive():
        line = from_answer.readline()
        if not line:
            raise RuntimeError("the answer's process ended")
        message = json.loads(line)
        sys.stdout.write(message["out"])
        if not message["ok"]:
            kind = getattr(builtins, message["error"], None)
            if isinstance(kind, type) and issubclass(kind, Exception):
                raise kind(message["message"])
            raise RuntimeError(f"{message['error']}: {message['message']}")
        return message

    loaded = receive()

    def proxy(name):
        def call(*args, **kwargs):
            request = {"name": name, "args": [_encode(a) for a in args],
                       "kwargs": [[k, _encode(v)] for k, v in kwargs.items()]}
            to_answer.write(json.dumps(request) + "\n")
            to_answer.flush()
            return _decode(receive()["value"])
        call.__name__ = name
        return call

    # A name the answer gives that is also a builtin (print, len, sorted) stays
    # the check's own: only the answer's functions use the answer's.
    own = {name for name in loaded["functions"] + list(loaded["values"]) if not hasattr(builtins, name)}
    namespace = {"__name__": "__main__"}
    namespace.update({n: _decode(v) for n, v in loaded["values"].items() if n in own})
    namespace.update({n: proxy(n) for n in loaded["functions"] if n in own})
    try:
        exec(compile(CHECK, "<check>", "exec"), namespace)
    finally:
        to_answer.close()
        child.wait()


if len(sys.argv) == 4 and sys.argv[1] == "--serve":
    _serve(int(sys.argv[2]), int(sys.argv[3]))
else:
    _check()
