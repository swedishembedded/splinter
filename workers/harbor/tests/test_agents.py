# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for testing agent execution
# backends against stub models and tiny tasks with no network, for its
# clients. If your team needs expertise in hermetic agent test harnesses then
# you can procure our services by sending an email to info@swedishembedded.com.
"""The model path end to end: a real agent (Sven) and a scripted one (a fake ``claude``) in the proot sandbox talk to a
stub model through the worker's gateway, and what the gateway logged reaches ``evidence.requests``. No secret of the
worker's reaches a file."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from conftest import make_spec, make_task, write_specs

from splinter_harbor_worker.cli import main

TASK = "train/002-ask-model"
SECRET = "sk-test-upstream-credential"

# A stand-in for the claude executable: posts one Anthropic message to ANTHROPIC_BASE_URL with ANTHROPIC_API_KEY over a
# bash TCP socket (the image has no curl), writes what it was started with, and exits 0.
FAKE_CLAUDE = r"""#!/bin/bash
printf '%s\n' "$@" > /logs/agent/argv.txt
printf '%s' "$HOME" > /logs/agent/home.txt
env > /logs/agent/env.txt
for f in /splinter/files/*/*; do cat "$f" > /logs/agent/file.txt; done 2>/dev/null
host=${ANTHROPIC_BASE_URL#http://}; port=${host##*:}; host=${host%%:*}
body='{"model":"x","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}'
exec 3<>/dev/tcp/$host/$port
printf 'POST /v1/messages HTTP/1.1\r\nHost: %s\r\nx-api-key: %s\r\ncontent-type: application/json\r\nconnection: close\r\ncontent-length: %s\r\n\r\n%s' \
  "$host" "$ANTHROPIC_API_KEY" "${#body}" "$body" >&3
cat <&3 > /logs/agent/reply.txt
echo expected > /app/answer.txt
"""


def run_one(tmp_path: Path, spec: dict, monkeypatch, local_images, env: dict[str, str] | None = None,
            files: dict[str, str] | None = None) -> tuple[int, Path]:
    make_task(tmp_path, TASK)
    for relative, text in (files or {}).items():
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_text(text)
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    specs = write_specs(tmp_path / "specs.jsonl", [spec])
    code = main(["run", "--specs", str(specs), "--tasks", str(tmp_path), "--out", str(tmp_path / "out"),
                 "--image-cache", str(tmp_path / "cache")])
    return code, tmp_path / "out"


def load_report(out: Path) -> dict:
    (path,) = (out / "attempts").rglob("*.json")
    return json.loads(path.read_text())


def fake_claude(tmp_path: Path) -> Path:
    path = tmp_path / "bin" / "claude"
    path.parent.mkdir()
    path.write_text(FAKE_CLAUDE)
    path.chmod(0o755)
    return path


def model(stub, **fields) -> dict:
    return {"connection": "stub", "model": "stub-model", "protocol": "anthropic", "url": stub.base_url,
            "token_env": "STUB_TOKEN", **fields}


def test_claude_requests_reach_evidence_through_the_gateway(tmp_path, monkeypatch, local_images, stub_model, validator):
    binary = fake_claude(tmp_path)
    spec = make_spec(TASK, agent="claude-code", args=["--verbose"], model=model(stub_model))
    code, out = run_one(tmp_path, spec, monkeypatch, local_images,
                        env={"SPLINTER_CLAUDE_BIN": str(binary), "STUB_TOKEN": SECRET})
    assert code == 0
    report = load_report(out)
    validator.validate(report)
    assert report["evidence"]["requests"] == [
        {"status": 200, "error": None, "prompt_tokens": 10, "completion_tokens": 2, "cached_tokens": 4}]
    assert report["usage"] == {"prompt_tokens": 10, "completion_tokens": 2, "cached_tokens": 4, "cost_usd": None}
    assert report["evidence"]["result"]["verifier_result"]["rewards"] == {"reward": 1.0}
    (post,) = stub_model.posts()
    # the upstream saw the upstream credential and the real model; the agent never held either
    assert post.headers["x-api-key"] == SECRET and post.body["model"] == "stub-model"
    assert post.headers["accept-encoding"] == "identity"
    trial = out / report["native_dir"].removeprefix("")
    assert SECRET not in "".join(p.read_text(errors="ignore") for p in out.rglob("*") if p.is_file())
    assert (trial / "agent" / "argv.txt").read_text().splitlines()[:3] == [
        "--verbose", "-p", "Write the word expected into /app/answer.txt."]
    assert (trial / "agent" / "home.txt").read_text() == "/tmp/splinter-agent-home"


def test_an_upstream_500_is_recorded_with_a_printable_error(tmp_path, monkeypatch, local_images, stub_model):
    stub_model.fail_with = 500
    binary = fake_claude(tmp_path)
    spec = make_spec(TASK, agent="claude-code", model=model(stub_model))
    code, out = run_one(tmp_path, spec, monkeypatch, local_images,
                        env={"SPLINTER_CLAUDE_BIN": str(binary), "STUB_TOKEN": SECRET})
    assert code == 0  # a model error is still a report
    (request,) = load_report(out)["evidence"]["requests"]
    assert request["status"] == 500
    assert request["error"] == "overloaded [31m retry later"
    assert request["error"].isprintable()


def test_sven_runs_headless_against_a_stub_model_and_its_trace_is_read(tmp_path, monkeypatch, local_images, stub_model,
                                                                        validator):
    assert shutil.which("sven"), "this test runs the real sven: put it on PATH (or set SPLINTER_SVEN_BIN)"
    spec = make_spec(TASK, agent="sven", model=model(stub_model, protocol="openai"))
    code, out = run_one(tmp_path, spec, monkeypatch, local_images, env={"STUB_TOKEN": SECRET})
    assert code == 0
    report = load_report(out)
    validator.validate(report)
    statuses = [r["status"] for r in report["evidence"]["requests"]]
    assert statuses and set(statuses) == {200}, report["evidence"]["requests"]
    assert report["agent_version"]
    trial = out / report["native_dir"]
    assert (trial / "agent" / "trajectory.json").is_file(), (trial / "agent" / "stderr.txt").read_text()
    assert report["trace"] == report["native_dir"] + "/agent/trajectory.json"
    assert report["evidence"]["turns"] >= 1
    assert all(p.headers.get("authorization") == f"Bearer {SECRET}" for p in stub_model.posts())


def test_arguments_files_prompt_and_environment_reach_the_agent_as_the_spec_says(tmp_path, monkeypatch, local_images,
                                                                              stub_model):
    binary = fake_claude(tmp_path)
    config = "model: tiny\n"
    digest = "sha256:" + hashlib.sha256(config.encode()).hexdigest()
    prompt = {"path": "prompts/p.md", "digest": "sha256:" + hashlib.sha256(b"Be brief.\n").hexdigest()}
    spec = make_spec(
        TASK, agent="claude-code", model=model(stub_model),
        args=["--settings", "agents/a.yaml", "--flag=with space", "$(touch /app/pwned)"],
        files=[{"path": "agents/a.yaml", "digest": digest}], system_prompt=prompt,
        env={"PLAIN": {"literal": "literal-value"}, "FROM_WORKER": {"from_env": "WORKER_ONLY_VALUE"}})
    code, out = run_one(tmp_path, spec, monkeypatch, local_images,
                        env={"SPLINTER_CLAUDE_BIN": str(binary), "STUB_TOKEN": SECRET,
                             "WORKER_ONLY_VALUE": "value-from-the-worker-environment"},
                        files={"agents/a.yaml": config, "prompts/p.md": "Be brief.\n"})
    assert code == 0
    report = load_report(out)
    trial = out / report["native_dir"]
    argv = (trial / "agent" / "argv.txt").read_text().splitlines()
    assert argv[:4] == ["--settings", "/splinter/files/agents/a.yaml", "--flag=with space", "$(touch /app/pwned)"]
    assert argv[4:7] == ["--append-system-prompt", "Be brief.", ""]  # the prompt text, newline kept, as one argument
    assert (trial / "agent" / "file.txt").read_text() == config
    environment = dict(line.split("=", 1) for line in (trial / "agent" / "env.txt").read_text().splitlines() if "=" in line)
    assert environment["PLAIN"] == "literal-value" and environment["FROM_WORKER"] == "value-from-the-worker-environment"
    assert "WORKER_ONLY_VALUE" not in environment and "STUB_TOKEN" not in environment  # nothing else of the worker's
    assert not (tmp_path / "pwned").exists()
    # a from_env value is resolved when the trial starts and never written to a file the worker keeps
    written = [p for p in (out / "harbor").rglob("*.json") if p.is_file()]
    assert all("value-from-the-worker-environment" not in p.read_text() for p in written)


# A stand-in for the codex executable: reads the provider its configuration names (CODEX_HOME/config.toml) and posts
# one Responses request to it with the key the provider's env_key names, then exits 0.
FAKE_CODEX = r"""#!/bin/bash
[ "$1" = --version ] && { echo "codex-cli 0.0.1"; exit 0; }
printf '%s\n' "$@" > /logs/agent/argv.txt
cp "$CODEX_HOME/config.toml" /logs/agent/config.toml
cp "$CODEX_HOME/AGENTS.md" /logs/agent/agents.md 2>/dev/null
while IFS= read -r line; do  # the image has no sed
  case $line in base_url\ =\ *) base=${line#*\"}; base=${base%\"};; env_key\ =\ *) key_var=${line#*\"}; key_var=${key_var%\"};; esac
done < "$CODEX_HOME/config.toml"
host=${base#http://}; host=${host%%/*}; port=${host##*:}; host=${host%%:*}
body='{"model":"x","input":"hi"}'
exec 3<>/dev/tcp/$host/$port
printf 'POST /v1/responses HTTP/1.1\r\nHost: %s\r\nauthorization: Bearer %s\r\ncontent-type: application/json\r\nconnection: close\r\ncontent-length: %s\r\n\r\n%s' \
  "$host" "${!key_var}" "${#body}" "$body" >&3
cat <&3 > /logs/agent/reply.txt
echo expected > /app/answer.txt
"""


def test_codex_speaks_the_responses_protocol_through_the_gateway(tmp_path, monkeypatch, local_images, stub_model,
                                                                 validator):
    path = tmp_path / "bin" / "codex"
    path.parent.mkdir()
    path.write_text(FAKE_CODEX)
    path.chmod(0o755)
    prompt = {"path": "prompts/p.md", "digest": "sha256:" + hashlib.sha256(b"Be brief.\n").hexdigest()}
    spec = make_spec(TASK, agent="codex", args=["--ephemeral"], system_prompt=prompt,
                     model=model(stub_model, protocol="openai"))
    code, out = run_one(tmp_path, spec, monkeypatch, local_images, env={"SPLINTER_CODEX_BIN": str(path),
                        "STUB_TOKEN": SECRET}, files={"prompts/p.md": "Be brief.\n"})
    assert code == 0
    report = load_report(out)
    validator.validate(report)
    assert report["agent_version"] == "0.0.1"
    assert report["evidence"]["requests"] == [
        {"status": 200, "error": None, "prompt_tokens": 10, "completion_tokens": 2, "cached_tokens": 4}]
    assert report["evidence"]["result"]["verifier_result"]["rewards"] == {"reward": 1.0}
    (post,) = stub_model.posts()
    assert post.path.endswith("/responses")
    assert post.headers["authorization"] == f"Bearer {SECRET}" and post.body["model"] == "stub-model"
    assert SECRET not in "".join(p.read_text(errors="ignore") for p in out.rglob("*") if p.is_file())
    agent = out / report["native_dir"] / "agent"
    argv = (agent / "argv.txt").read_text().rstrip("\n").splitlines()  # the instruction ends in a newline
    assert argv[0] == "exec" and argv[-1] == "Write the word expected into /app/answer.txt."
    assert {"--dangerously-bypass-approvals-and-sandbox", "--skip-git-repo-check", "--ephemeral"} <= set(argv)
    config = (agent / "config.toml").read_text()
    assert 'wire_api = "responses"' in config and 'model = "stub-model"' in config
    assert (agent / "agents.md").read_text() == "Be brief.\n"
