# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for giving sandboxed coding agents a
# model without ever giving them the provider's credential, for its clients.
# If your team needs expertise in confining agent runtimes and accounting for
# every model request they make then you can procure our services by sending
# an email to info@swedishembedded.com.
"""The model gateway the worker runs for one job.

A sandbox never holds the provider's credential. It sends its requests to this gateway with a per-trial token; the
gateway checks the token, forwards the request to the upstream the trial's specification names (rewriting the model and
injecting the credential it read from the worker's own environment), relays the answer unchanged and appends one record
per request to the trial's request log (``requests.jsonl``): status, error, prompt, completion and cached tokens.

Tokens are stateless: ``<trial>.<HMAC-SHA256(secret, trial)>`` with a secret drawn when the gateway starts. A token
arrives as ``Authorization: Bearer`` or ``x-api-key``; a request carrying two different credentials is refused. The
upstream's credential, the run secret and the tokens are never written to a log; the only place a token is written is
the sandbox's own private file the agent adapter creates for it.

A record's ``status`` says how the upstream did: an error event inside a 200 stream is recorded as 500 (with its
``error``), an upstream the gateway cannot reach, or one that drops a stream mid-answer, as 502, a stream the agent
closes before the answer ends as 499, and a request the agent gave up on before the upstream answered at all as 504.
The Rust classifier reads exactly these.

Requests go upstream with ``accept-encoding: identity``: replies are relayed as the upstream's raw bytes, so a
compressed answer would reach the agent with a content encoding it was not told about.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import hmac
import json
import logging
import os
import secrets
import select
import socket
import subprocess
import sys
import threading
import time
import zlib
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from splinter_harbor_worker.spec import FORWARDED_HEADERS

# Generation can legitimately stay silent for minutes on a slow local model; connecting cannot.
_UPSTREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=900.0, write=60.0, pool=None)
_UPSTREAM_LIMITS = httpx.Limits(max_connections=None, max_keepalive_connections=256, keepalive_expiry=30.0)
# The trials' requests are spread over this many connection pools, each trial always on the same one. A pool's
# bookkeeping scans every connection it holds, for each idle one, whenever a request starts or ends (httpcore 1.0), so
# one pool for a whole job grows quadratically with the trials running at once.
_UPSTREAM_POOLS = 64
_MODELS_TIMEOUT = httpx.Timeout(10.0)
# How often a request waiting on its upstream checks whether the agent left.
_DISCONNECT_POLL_SECONDS = 1.0
# An error text kept in a record is bounded: it comes from a remote server.
MAX_ERROR_CHARS = 500
REQUEST_LOG = "requests.jsonl"

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Upstream:
    """Where one trial's requests go. ``base_url`` ends where the protocol's own paths begin (``/chat/completions`` for
    ``openai``, ``/messages`` for ``anthropic``); ``token`` is the credential read from the worker's environment, or
    ``None`` for an endpoint that takes none."""

    base_url: str
    model: str
    protocol: str
    token: str | None = None
    # Headers of the agent's this upstream does not take (an endpoint that refuses ``anthropic-beta``); dropped.
    drop_headers: tuple[str, ...] = ()

    def __repr__(self) -> str:  # the credential never reaches a log through a traceback or a debug print
        return f"Upstream({self.base_url!r}, {self.model!r}, {self.protocol!r})"


def printable(text: object) -> str:
    """``text`` with every control character replaced by a space and bounded to ``MAX_ERROR_CHARS``: an error comes
    from a remote server and ends up in logs a terminal prints."""
    flat = "".join(ch if ch.isprintable() else " " for ch in str(text))
    return flat[:MAX_ERROR_CHARS]


def _signature(secret: str, trial: str) -> str:
    return hmac.new(secret.encode(), trial.encode(), hashlib.sha256).hexdigest()


class _Abandoned(Exception):
    """The agent disconnected before the upstream answered."""


class ModelGateway:
    """The gateway's application. Trials are registered before they run; their records go to the log path given."""

    def __init__(self, secret: str | None = None) -> None:
        self._secret = secret or secrets.token_hex(32)
        self._trials: dict[str, tuple[Upstream, Path]] = {}
        tls = httpx.create_ssl_context()  # what each client would load for itself, loaded once
        self._clients = [httpx.AsyncClient(timeout=_UPSTREAM_TIMEOUT, limits=_UPSTREAM_LIMITS, verify=tls)
                         for _ in range(_UPSTREAM_POOLS)]
        self._model_info: dict[tuple[str, str], dict] = {}
        self._log_lock = threading.Lock()
        self.app = self._build_app()

    def register(self, trial: str, upstream: Upstream, request_log: Path) -> None:
        """Serve ``trial`` from ``upstream``, recording its requests in ``request_log``."""
        self._trials[trial] = (upstream, request_log)

    def handover(self) -> dict:
        """What a gateway process needs to serve exactly these trials with these tokens (``from_handover``): the secret
        and the upstream credentials, so it travels only over a pipe."""
        return {"secret": self._secret,
                "trials": {trial: {**dataclasses.asdict(upstream), "log": str(log)}
                           for trial, (upstream, log) in self._trials.items()}}

    @classmethod
    def from_handover(cls, handover: dict) -> ModelGateway:
        gateway = cls(secret=handover["secret"])
        for trial, entry in handover["trials"].items():
            log = Path(entry.pop("log"))
            gateway.register(trial, Upstream(**{**entry, "drop_headers": tuple(entry["drop_headers"])}), log)
        return gateway

    def _client_for(self, trial: str) -> httpx.AsyncClient:
        """The connection pool ``trial``'s requests go through: always the same one, so its conversation keeps its
        connection."""
        return self._clients[zlib.crc32(trial.encode()) % len(self._clients)]

    def token_for(self, trial: str) -> str:
        """The token a trial's agent sends."""
        return f"{trial}.{_signature(self._secret, trial)}"

    def _trial(self, request: Request) -> str | None:
        bearer = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        api_key = request.headers.get("x-api-key", "").strip()
        if bearer and api_key and bearer != api_key:
            return None
        trial, _, signature = (bearer or api_key).rpartition(".")
        if trial in self._trials and hmac.compare_digest(signature, _signature(self._secret, trial)):
            return trial
        return None

    def _record(self, trial: str, status: int, usage: dict | None, error: str | None = None) -> None:
        usage = usage or {}
        record = {
            "ts": time.time(),
            "status": status,
            # Chat Completions' usage fields, else those of the Responses API and Anthropic's.
            "prompt_tokens": _first(usage, "prompt_tokens", "input_tokens"),
            "completion_tokens": _first(usage, "completion_tokens", "output_tokens"),
            # The part of the prompt the server's prefix cache served.
            "cached_tokens": (usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}).get(
                "cached_tokens", usage.get("cache_read_input_tokens")),
            "error": printable(error) if error else None,
        }
        log = self._trials[trial][1]
        with self._log_lock:
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a") as handle:
                handle.write(json.dumps(record) + "\n")

    def _headers(self, upstream: Upstream, request: Request) -> dict[str, str]:
        headers = {"content-type": "application/json", "accept-encoding": "identity"}
        if upstream.protocol == "anthropic":
            for name in FORWARDED_HEADERS:
                if name in upstream.drop_headers:
                    continue
                if value := request.headers.get(name):
                    headers[name] = value
            if upstream.token:
                headers["x-api-key"] = upstream.token
        elif upstream.token:
            headers["authorization"] = f"Bearer {upstream.token}"
        return headers

    async def _models(self, trial: str, upstream: Upstream) -> dict:
        """The upstream's own listing of its model (context length, capabilities), so an agent sizes its requests by
        the served model's window; a bare entry while the upstream lists nothing."""
        key = (upstream.base_url, upstream.model)
        if key in self._model_info:
            return self._model_info[key]
        headers = {"authorization": f"Bearer {upstream.token}", "x-api-key": upstream.token} if upstream.token else {}
        try:
            reply = await self._client_for(trial).get(f"{upstream.base_url.rstrip('/')}/models", headers=headers,
                                           timeout=_MODELS_TIMEOUT)
            reply.raise_for_status()
            listed = [m for m in reply.json().get("data", []) if isinstance(m, dict)]
        except (httpx.HTTPError, ValueError, AttributeError) as exc:
            _log.warning("%s lists no model information (%s)", upstream.base_url, printable(exc))
            return {"id": upstream.model, "object": "model"}
        info = next((m for m in listed if m.get("id") == upstream.model), {"id": upstream.model})
        self._model_info[key] = {**info, "object": "model"}
        return self._model_info[key]

    def _build_app(self) -> FastAPI:
        app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)

        @app.get("/v1/models")
        async def models(request: Request) -> Response:
            trial = self._trial(request)
            if trial is None:
                return _error(401, "missing or invalid gateway token")
            upstream = self._trials[trial][0]
            return JSONResponse({"object": "list", "data": [await self._models(trial, upstream)]})

        @app.post("/v1/chat/completions")
        async def chat(request: Request) -> Response:
            return await self._forward(request, "openai", "/chat/completions", record=True)

        @app.post("/v1/responses")
        async def responses(request: Request) -> Response:
            return await self._forward(request, "openai", "/responses", record=True)

        @app.post("/v1/messages")
        async def messages(request: Request) -> Response:
            return await self._forward(request, "anthropic", "/messages", record=True)

        @app.post("/v1/messages/count_tokens")
        async def count_tokens(request: Request) -> Response:
            return await self._forward(request, "anthropic", "/messages/count_tokens", record=False)

        return app

    async def _forward(self, request: Request, protocol: str, path: str, *, record: bool) -> Response:
        """One request of ``protocol``; ``?beta=true`` and other query strings are the agent's own and are not
        forwarded."""
        fail = _anthropic_error if protocol == "anthropic" else _error
        trial = self._trial(request)
        if trial is None:
            return fail(401, "missing, invalid or ambiguous gateway token")
        upstream = self._trials[trial][0]
        if upstream.protocol != protocol:
            return fail(404, f"this trial's model speaks {upstream.protocol}, not {protocol}")
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return fail(400, "request body is not JSON")
        if not isinstance(body, dict):
            return fail(400, "request body is not a JSON object")
        body["model"] = upstream.model
        url = f"{upstream.base_url.rstrip('/')}{path}"
        try:
            client = self._client_for(trial)
            reply = await self._send_unless_abandoned(
                request, client, client.build_request("POST", url, json=body, headers=self._headers(upstream, request)))
        except httpx.HTTPError as exc:
            if record:
                self._record(trial, 502, None, f"unreachable: {type(exc).__name__}: {exc}")
            return fail(502, f"the upstream is unreachable: {printable(exc)}")
        except _Abandoned:
            if record:
                self._record(trial, 504, None, "the upstream sent no response before the agent gave up")
            return fail(504, "the upstream sent no response")
        if record and body.get("stream") and reply.status_code == 200:
            return StreamingResponse(
                self._relay(reply, trial),
                status_code=reply.status_code,
                media_type=reply.headers.get("content-type", "text/event-stream"),
                background=BackgroundTask(reply.aclose),
            )
        content = await reply.aread()
        await reply.aclose()
        if record:
            usage, error = None, None
            try:
                answer = json.loads(content)
                usage = answer.get("usage")
                if reply.status_code >= 400 and isinstance(answer.get("error"), dict):
                    error = answer["error"].get("message")
            except (json.JSONDecodeError, AttributeError):
                pass
            if reply.status_code >= 400 and not error:
                error = f"the upstream answered {reply.status_code}"
            self._record(trial, reply.status_code, usage, error)
        return Response(content, status_code=reply.status_code, media_type=reply.headers.get("content-type"))

    @staticmethod
    async def _send_unless_abandoned(request: Request, client: httpx.AsyncClient,
                                     outgoing: httpx.Request) -> httpx.Response:
        """Send ``outgoing``; raise ``_Abandoned`` when the agent disconnects before the upstream has answered,
        cancelling the upstream request."""
        send = asyncio.ensure_future(client.send(outgoing, stream=True))
        while True:
            done, _ = await asyncio.wait({send}, timeout=_DISCONNECT_POLL_SECONDS)
            if done:
                return send.result()
            if await request.is_disconnected():
                send.cancel()
                raise _Abandoned

    async def _relay(self, reply: httpx.Response, trial: str) -> AsyncIterator[bytes]:
        """Pass the event stream through untouched, noting the last usage object it carries and any error event: a
        stream that began with 200 and then reports an error failed on the server's side, and is recorded as a 500
        with its message."""
        usage, error = None, None
        pending = b""
        finished = False
        try:
            chunks = reply.aiter_raw()
            while True:
                try:
                    chunk = await anext(chunks)
                except StopAsyncIteration:
                    break
                except httpx.HTTPError as exc:
                    # The server dropped the connection mid-answer: the server's failure, never the agent's.
                    message = f"the upstream ended the stream mid-answer: {type(exc).__name__}: {exc}"
                    self._record(trial, 502, None, message)
                    event = {"error": {"message": printable(message), "type": "server_error", "code": "upstream_cut"}}
                    yield f"\n\ndata: {json.dumps(event)}\n\n".encode()
                    finished = True
                    return
                yield chunk
                pending += chunk
                *lines, pending = pending.split(b"\n")
                for line in lines:
                    if not line.startswith(b"data: ") or not (b'"usage"' in line or b'"error"' in line):
                        continue
                    try:
                        event = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    # The Responses API nests the usage and a failure in the event's response.
                    response = event.get("response") if isinstance(event.get("response"), dict) else {}
                    usage = event.get("usage") or response.get("usage") or usage
                    failure = event.get("error") or response.get("error")
                    if isinstance(failure, dict):
                        error = failure.get("message") or json.dumps(failure)
            finished = True
            self._record(trial, 500 if error else reply.status_code, usage, error)
        finally:
            if not finished:
                # The agent closed the stream before it ended: neither a success nor the server's failure, but
                # never unrecorded.
                self._record(trial, 499, usage, "the agent closed the stream before the answer ended")

    async def aclose(self) -> None:
        for client in self._clients:
            await client.aclose()


class GatewayServer:
    """A ``ModelGateway`` served by uvicorn in ``processes`` child processes of its own, all accepting on one socket of
    ``host`` this process binds to a free port. The gateway relays every token every agent receives, and the
    interpreter that runs the Harbor job is busy for much of a run: on a thread of that interpreter each relayed chunk
    waited for it. A child gets the gateway's secret and registered trials (upstream credentials included) on its
    standard input, never on a command line, in its environment or in a file, and ends when that input closes, which
    is also what the parent's own end does."""

    def __init__(self, gateway: ModelGateway, host: str = "127.0.0.1", processes: int = 1) -> None:
        if processes < 1:
            raise ValueError("a gateway runs in at least one process")
        self.gateway = gateway
        self.host = host
        self._processes = processes
        self._socket = socket.create_server((host, 0), backlog=_LISTEN_BACKLOG)
        self._children: list[subprocess.Popen] = []

    def start(self) -> None:
        handover = (json.dumps(self.gateway.handover()) + "\n").encode()
        fd = self._socket.fileno()
        try:
            for _ in range(self._processes):
                child = subprocess.Popen(
                    [sys.executable, "-I", "-c", _CHILD, str(Path(__file__).resolve().parents[1]), str(fd)],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, pass_fds=(fd,))
                self._children.append(child)
                child.stdin.write(handover)
                child.stdin.flush()
            for child in self._children:
                if not _ready(child, _START_SECONDS):
                    raise RuntimeError(f"the model gateway did not start (its process ended with {child.poll()})")
        except BaseException:
            self.stop()
            raise

    @property
    def port(self) -> int:
        return self._socket.getsockname()[1]

    def stop(self) -> None:
        children, self._children = self._children, []
        for child in children:
            with contextlib.suppress(OSError):
                child.stdin.close()
        for child in children:
            try:
                child.wait(timeout=_STOP_SECONDS)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            if child.returncode != 0:
                _log.warning("a model gateway process ended with exit status %s", child.returncode)
        self._socket.close()


def _ready(child: subprocess.Popen, timeout: float) -> bool:
    """Whether ``child`` said it serves within ``timeout`` seconds."""
    if not select.select([child.stdout], [], [], timeout)[0]:
        return False
    return child.stdout.readline() == b"ready\n"


def serve_child(fd: int) -> None:
    """A gateway process (``GatewayServer``): the handover from standard input, the app on the inherited socket ``fd``
    until standard input closes."""
    handover = json.loads(sys.stdin.buffer.readline())
    gateway = ModelGateway.from_handover(handover)
    server = uvicorn.Server(uvicorn.Config(gateway.app, log_level="warning", access_log=False))

    def watch() -> None:
        while not server.started and not server.should_exit:
            time.sleep(0.02)
        sys.stdout.write("ready\n")
        sys.stdout.flush()
        sys.stdin.buffer.read()  # until the parent closes it, or ends
        server.should_exit = True

    threading.Thread(target=watch, name="gateway-parent", daemon=True).start()
    server.run(sockets=[socket.socket(fileno=fd)])


# Concurrent trials one gateway process relays before it becomes their wait (a load test of streaming agents measured
# where one process falls behind; the commit that set this value has the numbers).
TRIALS_PER_PROCESS = 128


def processes_for(concurrency: int) -> int:
    """The gateway processes for a job that runs ``concurrency`` trials at once: one per ``TRIALS_PER_PROCESS``, at
    most one per CPU."""
    return max(1, min(os.cpu_count() or 1, -(-concurrency // TRIALS_PER_PROCESS)))


# A child's program, run isolated (`-I`) so nothing of the environment changes what it imports; its first argument puts
# this very package first on the path.
_CHILD = ("import sys; sys.path.insert(0, sys.argv[1]); "
          "from splinter_harbor_worker.gateway import serve_child; serve_child(int(sys.argv[2]))")
_LISTEN_BACKLOG = 4096
_START_SECONDS = 20
_STOP_SECONDS = 20


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": {"message": message}}, status_code=status)


def _anthropic_error(status: int, message: str) -> JSONResponse:
    kind = "authentication_error" if status == 401 else "not_found_error" if status == 404 else "api_error"
    return JSONResponse({"type": "error", "error": {"type": kind, "message": message}}, status_code=status)


def _first(usage: dict, *names: str):
    return next((usage[name] for name in names if usage.get(name) is not None), None)
