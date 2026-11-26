# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for testing agent execution
# backends against stub models and tiny tasks with no network, for its
# clients. If your team needs expertise in hermetic agent test harnesses then
# you can procure our services by sending an email to info@swedishembedded.com.
"""The model gateway on its own: per-trial tokens, what is forwarded upstream, and what is logged per request."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from splinter_harbor_worker.gateway import GatewayServer, ModelGateway, Upstream, printable

UPSTREAM_KEY = "sk-upstream-only"


@pytest.fixture
def served(tmp_path, stub_model) -> Iterator[tuple[GatewayServer, ModelGateway, Path]]:
    gateway = ModelGateway(secret="test-secret")
    log = tmp_path / "trial" / "gateway" / "requests.jsonl"
    gateway.register("trial-a", Upstream(stub_model.base_url, "real-model", "openai", UPSTREAM_KEY), log)
    server = GatewayServer(gateway)
    server.start()
    try:
        yield server, gateway, log
    finally:
        server.stop()


def records(log: Path) -> list[dict]:
    return [json.loads(line) for line in log.read_text().splitlines()]


def chat(server: GatewayServer, token: str | None, **body) -> httpx.Response:
    headers = {"authorization": f"Bearer {token}"} if token else {}
    return httpx.post(f"http://127.0.0.1:{server.port}/v1/chat/completions", headers=headers,
                      json={"model": "agent-name", "messages": [], **body}, timeout=30)


def test_a_request_is_forwarded_with_the_upstream_credential_and_the_real_model(served, stub_model) -> None:
    server, gateway, log = served
    reply = chat(server, gateway.token_for("trial-a"))
    assert reply.status_code == 200 and reply.json()["choices"][0]["message"]["content"] == "Done."
    (post,) = stub_model.posts()
    assert post.body["model"] == "real-model"
    assert post.headers["authorization"] == f"Bearer {UPSTREAM_KEY}"
    assert post.headers["accept-encoding"] == "identity"
    assert records(log)[0] | {"ts": 0} == {"ts": 0, "status": 200, "prompt_tokens": 10, "completion_tokens": 2,
                                           "cached_tokens": 4, "error": None}


def test_a_streamed_answer_is_relayed_and_its_usage_recorded(served) -> None:
    server, gateway, log = served
    reply = chat(server, gateway.token_for("trial-a"), stream=True)
    assert reply.status_code == 200 and "Done." in reply.text and reply.text.rstrip().endswith("data: [DONE]")
    (record,) = records(log)
    assert (record["status"], record["prompt_tokens"], record["completion_tokens"]) == (200, 10, 2)


def test_a_request_without_a_valid_token_is_refused_and_unlogged(served, stub_model) -> None:
    server, gateway, log = served
    assert chat(server, None).status_code == 401
    assert chat(server, "trial-a.deadbeef").status_code == 401
    assert chat(server, gateway.token_for("another-trial")).status_code == 401
    forged = ModelGateway(secret="other-secret").token_for("trial-a")
    assert chat(server, forged).status_code == 401
    assert stub_model.posts() == [] and not log.exists()


def test_two_different_credentials_in_one_request_are_refused(served) -> None:
    server, gateway, _ = served
    token = gateway.token_for("trial-a")
    reply = httpx.post(f"http://127.0.0.1:{server.port}/v1/chat/completions",
                       headers={"authorization": f"Bearer {token}", "x-api-key": "something-else"},
                       json={"model": "m", "messages": []})
    assert reply.status_code == 401


def test_an_upstream_error_is_recorded_with_its_status_and_a_printable_message(served, stub_model) -> None:
    server, gateway, log = served
    stub_model.fail_with = 500
    assert chat(server, gateway.token_for("trial-a")).status_code == 500
    (record,) = records(log)
    assert record["status"] == 500 and record["error"] == "overloaded [31m retry later"
    assert record["prompt_tokens"] is None  # unmeasured is absent, never 0


def test_an_unreachable_upstream_is_a_502_record(tmp_path) -> None:
    gateway = ModelGateway()
    log = tmp_path / "requests.jsonl"
    gateway.register("t", Upstream("http://127.0.0.1:9/v1", "m", "openai"), log)
    server = GatewayServer(gateway)
    server.start()
    try:
        assert chat(server, gateway.token_for("t")).status_code == 502
    finally:
        server.stop()
    (record,) = records(log)
    assert record["status"] == 502 and record["error"].startswith("unreachable")


def test_a_model_list_names_the_trials_model_and_a_route_of_the_other_protocol_is_refused(served) -> None:
    server, gateway, _ = served
    headers = {"authorization": f"Bearer {gateway.token_for('trial-a')}"}
    listing = httpx.get(f"http://127.0.0.1:{server.port}/v1/models", headers=headers, timeout=30).json()
    assert [m["id"] for m in listing["data"]] == ["real-model"]  # the stub lists another id: a bare entry is served
    wrong = httpx.post(f"http://127.0.0.1:{server.port}/v1/messages", headers=headers, json={"model": "m"})
    assert wrong.status_code == 404


def test_the_upstream_credential_is_never_in_a_representation_or_a_record(served) -> None:
    _, gateway, log = served
    assert UPSTREAM_KEY not in repr(Upstream("http://x", "m", "openai", UPSTREAM_KEY))
    assert UPSTREAM_KEY not in repr(gateway._trials)


def test_printable_strips_control_characters_and_bounds_the_length() -> None:
    assert printable("a\x00b\x1b[0m\r\nc") == "a b [0m  c"
    assert len(printable("x" * 10_000)) == 500


def test_a_connections_dropped_headers_never_reach_its_upstream(tmp_path, stub_model) -> None:
    gateway = ModelGateway(secret="test-secret")
    log = tmp_path / "gateway" / "requests.jsonl"
    gateway.register("kept", Upstream(stub_model.base_url, "m", "anthropic", UPSTREAM_KEY), log)
    gateway.register("dropped", Upstream(stub_model.base_url, "m", "anthropic", UPSTREAM_KEY,
                                         drop_headers=("anthropic-beta",)), log)
    server = GatewayServer(gateway)
    server.start()
    try:
        for trial in ("kept", "dropped"):
            reply = httpx.post(f"http://127.0.0.1:{server.port}/v1/messages",
                               headers={"x-api-key": gateway.token_for(trial), "anthropic-version": "2023-06-01",
                                        "anthropic-beta": "some-beta"},
                               json={"model": "x", "max_tokens": 8, "messages": []}, timeout=30)
            assert reply.status_code == 200
    finally:
        server.stop()
    kept, dropped = stub_model.posts()
    assert kept.headers["anthropic-beta"] == "some-beta"
    assert "anthropic-beta" not in dropped.headers and dropped.headers["anthropic-version"] == "2023-06-01"


def test_a_jobs_trials_are_spread_over_several_upstream_pools_each_trial_on_one(tmp_path) -> None:
    """One connection pool for every trial of a job was the gateway's bottleneck: the pool's bookkeeping scans every
    connection it holds for each request it starts and ends, so hundreds of concurrent streams through one pool ran at a
    small fraction of the rate of the same streams sent to the upstream directly. A trial keeps one pool, so its conversation reuses
    its keep-alive connection."""
    gateway = ModelGateway(secret="test-secret")
    trials = [f"trial-{n}" for n in range(512)]
    pools = {trial: gateway._client_for(trial) for trial in trials}
    assert all(gateway._client_for(trial) is pools[trial] for trial in trials)
    sizes = sorted(sum(1 for pool in pools.values() if pool is shared) for shared in set(pools.values()))
    assert len(sizes) > 1 and sizes[-1] <= 4 * len(trials) // len(sizes), sizes


def test_the_gateway_answers_while_the_workers_interpreter_is_held(served) -> None:
    """The gateway shares nothing with the interpreter that runs the Harbor job: on a live run the job's own loop
    held the interpreter much of the time, and a gateway thread waiting for it made every agent wait for its model.
    A request sent while this process holds its interpreter (a C loop never lets another thread run) is answered at
    once, not when the loop ends. It is one the gateway answers itself (no token), since the stub upstream is this
    process."""
    import subprocess
    import sys
    import time

    server, _, _ = served
    client = subprocess.Popen([sys.executable, "-I", "-c", (
        "import sys, time, httpx; time.sleep(0.3); t = time.monotonic(); "
        "r = httpx.post(sys.argv[1], json={'messages': []}, timeout=30); print(r.status_code, time.monotonic() - t)"),
        f"http://127.0.0.1:{server.port}/v1/chat/completions"],
        stdout=subprocess.PIPE, text=True)
    held_from = time.monotonic()
    sum(range(3 * 10**8))  # holds this interpreter for a few seconds
    held = time.monotonic() - held_from
    status, took = client.communicate(timeout=60)[0].split()
    assert status == "401" and float(took) < held / 2, (took, held)


def test_a_job_gets_a_gateway_process_per_share_of_its_concurrent_trials() -> None:
    from splinter_harbor_worker.gateway import TRIALS_PER_PROCESS, processes_for

    assert processes_for(1) == 1 and processes_for(TRIALS_PER_PROCESS) == 1
    assert processes_for(TRIALS_PER_PROCESS + 1) == 2 and processes_for(4 * TRIALS_PER_PROCESS) == 4
