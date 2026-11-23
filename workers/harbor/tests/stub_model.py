# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for testing agent execution
# backends against stub models and tiny tasks with no network, for its
# clients. If your team needs expertise in hermetic agent test harnesses then
# you can procure our services by sending an email to info@swedishembedded.com.
"""A local model server for the tests: OpenAI chat completions and Anthropic messages, plain or streamed.

It answers every request with one short text, or with the HTTP status ``fail_with`` (and an error body with a control
character in its message), and records every request it receives (path, headers, body), so a test can see exactly what
the gateway forwarded.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "stub-model"
USAGE = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12, "prompt_tokens_details": {"cached_tokens": 4}}


@dataclass
class Received:
    path: str
    headers: dict[str, str]
    body: dict | None


@dataclass
class StubModel:
    fail_with: int | None = None
    requests: list[Received] = field(default_factory=list)
    _server: ThreadingHTTPServer | None = None

    def __enter__(self) -> StubModel:
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args) -> None:
                pass

            def _send(self, status: int, body: dict, content_type: str = "application/json") -> None:
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("content-type", content_type)
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                stub.requests.append(Received(self.path, {k.lower(): v for k, v in self.headers.items()}, None))
                if self.path.endswith("/models"):
                    self._send(200, {"object": "list", "data": [{"id": MODEL, "object": "model", "context_length": 32768}]})
                else:
                    self._send(404, {"error": {"message": "no route"}})

            def do_POST(self) -> None:
                raw = self.rfile.read(int(self.headers.get("content-length", "0")))
                body = json.loads(raw or b"{}")
                stub.requests.append(Received(self.path, {k.lower(): v for k, v in self.headers.items()}, body))
                if stub.fail_with is not None:
                    self._send(stub.fail_with, {"error": {"message": "overloaded\x1b[31m\nretry later", "type": "server_error"}})
                elif self.path.endswith("/messages"):
                    self._send(200, {"id": "m", "type": "message", "role": "assistant", "model": MODEL,
                                     "content": [{"type": "text", "text": "Done."}], "stop_reason": "end_turn",
                                     "usage": {"input_tokens": 10, "output_tokens": 2, "cache_read_input_tokens": 4}})
                elif not body.get("stream"):
                    self._send(200, {"id": "c", "object": "chat.completion", "model": MODEL, "usage": USAGE,
                                     "choices": [{"index": 0, "finish_reason": "stop",
                                                  "message": {"role": "assistant", "content": "Done."}}]})
                else:
                    chunks = ({"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Done."}}]},
                              {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                              {"choices": [], "usage": USAGE})
                    data = b"".join(f"data: {json.dumps({'id': 'c', 'object': 'chat.completion.chunk', 'model': MODEL, **c})}\n\n".encode()
                                    for c in chunks) + b"data: [DONE]\n\n"
                    self._send(200, data, "text/event-stream")

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc) -> None:
        assert self._server is not None
        self._server.shutdown()
        self._server.server_close()

    @property
    def base_url(self) -> str:
        assert self._server is not None
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def posts(self) -> list[Received]:
        return [r for r in self.requests if r.body is not None]
