# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for testing agent execution
# backends against stub models and tiny tasks with no network, for its
# clients. If your team needs expertise in hermetic agent test harnesses then
# you can procure our services by sending an email to info@swedishembedded.com.
"""A specification the worker cannot honour is refused before Harbor starts: nothing is staged, no job directory exists,
the exit code is 2 and the progress log says why."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import make_spec, make_task, write_specs

from splinter_harbor_worker.cli import main

TASK = "train/003-refused"
MODEL = {"connection": "stub", "model": "m", "protocol": "openai", "url": "http://127.0.0.1:9/v1", "token_env": "T"}


def refuse(tmp_path: Path, spec: dict, capsys, monkeypatch) -> str:
    make_task(tmp_path, TASK)
    monkeypatch.setenv("T", "x")
    specs = write_specs(tmp_path / "specs.jsonl", [spec])
    code = main(["run", "--specs", str(specs), "--tasks", str(tmp_path), "--out", str(tmp_path / "out"),
                 "--image-cache", str(tmp_path / "cache")])
    assert code == 2
    assert not (tmp_path / "out" / "harbor").exists() and not (tmp_path / "out" / "staged").exists()
    events = [json.loads(line) for line in (tmp_path / "out" / "events.jsonl").read_text().splitlines()]
    assert [e["kind"] for e in events] == ["error"]
    message = capsys.readouterr().err
    assert events[0]["detail"] in message
    return events[0]["detail"]


def test_a_literal_secret_in_an_agent_environment_is_refused(tmp_path, capsys, monkeypatch) -> None:
    spec = make_spec(TASK, agent="sven", model=MODEL, env={"SVEN_KEY": {"literal": "sk-1"}})
    assert "SVEN_KEY looks like a credential" in refuse(tmp_path, spec, capsys, monkeypatch)


def test_an_unknown_agent_is_refused_with_the_agents_there_are(tmp_path, capsys, monkeypatch) -> None:
    message = refuse(tmp_path, make_spec(TASK, agent="aider"), capsys, monkeypatch)
    assert "unknown agent 'aider'" in message and "sven, claude-code, codex, oracle, nop" in message


def test_an_unknown_sandbox_provider_is_refused(tmp_path, capsys, monkeypatch) -> None:
    message = refuse(tmp_path, make_spec(TASK, provider="firecracker"), capsys, monkeypatch)
    assert "unknown sandbox provider 'firecracker'" in message


def test_a_variable_the_worker_does_not_have_is_refused(tmp_path, capsys, monkeypatch) -> None:
    spec = make_spec(TASK, agent="sven", model=MODEL, env={"X": {"from_env": "NOT_SET_ANYWHERE"}})
    assert "NOT_SET_ANYWHERE" in refuse(tmp_path, spec, capsys, monkeypatch)


def test_a_wrong_schema_version_is_refused(tmp_path, capsys, monkeypatch) -> None:
    message = refuse(tmp_path, {**make_spec(TASK), "schema": "splinter.execution/2"}, capsys, monkeypatch)
    assert "splinter.execution/2" in message


def test_an_unknown_field_is_a_contract_mismatch_and_is_refused(tmp_path, capsys, monkeypatch) -> None:
    assert "surprise" in refuse(tmp_path, {**make_spec(TASK), "surprise": 1}, capsys, monkeypatch)


def test_a_network_the_provider_cannot_enforce_is_refused_not_weakened(tmp_path, capsys, monkeypatch) -> None:
    spec = make_spec(TASK, agent="nop", network="none")  # proot shares the host's network
    message = refuse(tmp_path, spec, capsys, monkeypatch)
    assert "cannot be enforced by the proot provider" in message


def test_the_default_gateway_only_network_on_proot_is_refused(tmp_path, capsys, monkeypatch) -> None:
    spec = make_spec(TASK, agent="sven", network="model-gateway-only", model=MODEL)
    assert "proot" in refuse(tmp_path, spec, capsys, monkeypatch)


def test_a_read_only_mount_on_proot_is_refused(tmp_path, capsys, monkeypatch) -> None:
    (tmp_path / "knowledge").mkdir()
    mount = {"source": "knowledge", "digest": "blake3:" + "f" * 64, "target": "/knowledge", "read_only": True}
    spec = make_spec(TASK, agent="nop", sandbox={"mounts": [mount]})
    assert "read-only mounts" in refuse(tmp_path, spec, capsys, monkeypatch)


def test_a_model_over_the_network_none_is_refused(tmp_path, capsys, monkeypatch) -> None:
    spec = make_spec(TASK, agent="sven", provider="userns", network="none", model=MODEL)
    assert "no way to reach its model" in refuse(tmp_path, spec, capsys, monkeypatch)


def test_an_agent_file_that_does_not_match_its_digest_is_refused(tmp_path, capsys, monkeypatch) -> None:
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "a.yaml").write_text("x: 1\n")
    spec = make_spec(TASK, agent="sven", model=MODEL, network="open",
                     files=[{"path": "agents/a.yaml", "digest": "sha256:" + "0" * 64}])
    monkeypatch.setenv("SPLINTER_SVEN_BIN", sys.executable)
    assert "does not match its digest" in refuse(tmp_path, spec, capsys, monkeypatch)


def test_a_task_directory_that_leaves_the_task_root_is_refused(tmp_path, capsys, monkeypatch) -> None:
    spec = make_spec(TASK, agent="nop")
    spec["task_dir"] = "tasks/../../escape"
    assert "not a plain relative path" in refuse(tmp_path, spec, capsys, monkeypatch)


def test_the_command_line_refuses_before_harbor_with_exit_code_2(tmp_path) -> None:
    make_task(tmp_path, TASK)
    specs = write_specs(tmp_path / "specs.jsonl", [make_spec(TASK, agent="aider")])
    done = subprocess.run([sys.executable, "-m", "splinter_harbor_worker.cli", "run", "--specs", str(specs), "--tasks",
                           str(tmp_path), "--out", str(tmp_path / "out")], capture_output=True, text=True, check=False)
    assert done.returncode == 2 and "unknown agent 'aider'" in done.stderr
    assert not (tmp_path / "out" / "harbor").exists()


def test_the_same_slot_and_attempt_twice_is_refused(tmp_path, capsys, monkeypatch) -> None:
    make_task(tmp_path, TASK)
    spec = make_spec(TASK, agent="nop")
    specs = write_specs(tmp_path / "specs.jsonl", [spec, spec])
    assert main(["run", "--specs", str(specs), "--tasks", str(tmp_path), "--out", str(tmp_path / "out")]) == 2
    assert "repeats line 1" in capsys.readouterr().err


@pytest.mark.parametrize("bad", ["not json", "[]", ""])
def test_unreadable_specs_are_refused(tmp_path, bad, capsys) -> None:
    specs = tmp_path / "specs.jsonl"
    specs.write_text(bad + "\n")
    assert main(["run", "--specs", str(specs), "--tasks", str(tmp_path), "--out", str(tmp_path / "out")]) == 2


def test_a_header_the_gateway_does_not_forward_cannot_be_dropped(tmp_path, capsys, monkeypatch) -> None:
    model = {"connection": "c", "model": "m", "protocol": "anthropic", "url": "http://127.0.0.1:9/v1",
             "drop_headers": ["authorization"]}
    message = refuse(tmp_path, make_spec(TASK, agent="claude-code", model=model), capsys, monkeypatch)
    assert "authorization" in message and "anthropic-beta" in message
