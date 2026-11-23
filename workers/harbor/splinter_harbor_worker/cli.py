# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for running coding agents over many
# tasks in reproducible sandboxes for its clients. If your team needs expertise
# in agent execution infrastructure then you can procure our services by
# sending an email to info@swedishembedded.com.
"""``splinter-harbor-worker run|reconcile|play``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from splinter_harbor_worker.errors import WorkerError
from splinter_harbor_worker.play import play
from splinter_harbor_worker.runner import DEFAULT_CONCURRENCY, reconcile, run


def _version() -> str:
    from importlib.metadata import version

    return f"splinter-harbor-worker {version('splinter-harbor-worker')} (harbor {version('harbor')})"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="splinter-harbor-worker", description=__doc__)
    # The controller names its rollouts' implementation by this line: the worker's and Harbor's versions.
    parser.add_argument("--version", action="version", version=_version())
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser(
        "run", help="run ExecutionSpecs as one Harbor job and write an AttemptReport for each finished attempt")
    run_parser.add_argument("--specs", type=Path, required=True, help="ExecutionSpecs, one JSON object per line")
    run_parser.add_argument("--tasks", type=Path, required=True, metavar="TASK_ROOT",
                            help="the directory every spec's task_dir is relative to")
    run_parser.add_argument("--out", type=Path, required=True, metavar="OUT_DIR",
                            help="results: harbor/ (harbor view OUT_DIR/harbor), attempts/, events.jsonl")
    run_parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="trials at once")
    run_parser.add_argument("--files", type=Path, metavar="FILES_ROOT",
                            help="where agent files, system prompts and mount sources are found (default TASK_ROOT)")
    run_parser.add_argument("--image-cache", type=Path, metavar="DIR",
                            help="the userns and proot providers' image cache (default ~/.cache/splinter-harbor-worker)")

    play_parser = commands.add_parser(
        "play", help="run one ExecutionSpec with a person driving the agent on this terminal, then grade it")
    play_parser.add_argument("--spec", type=Path, required=True, metavar="SPEC.json",
                             help="one ExecutionSpec (a single JSON object); the agent must be sven or claude-code (codex cannot be played)")
    play_parser.add_argument("--tasks", type=Path, required=True, metavar="TASK_ROOT",
                             help="the directory the spec's task_dir is relative to")
    play_parser.add_argument("--out", type=Path, required=True, metavar="OUT_DIR",
                             help="a new directory: harbor/, attempts/, events.jsonl as for run")
    play_parser.add_argument("--files", type=Path, metavar="FILES_ROOT",
                             help="where agent files, system prompts and mount sources are found (default TASK_ROOT)")
    play_parser.add_argument("--image-cache", type=Path, metavar="DIR",
                             help="the userns and proot providers' image cache (default ~/.cache/splinter-harbor-worker)")
    play_parser.add_argument("--continue", dest="continue_from", type=Path, metavar="TRIAL_DIR",
                             help="go on with the conversation of this earlier played sven trial "
                                  "(OUT_DIR/harbor/job/<trial>) as a new attempt; the earlier trial is not modified")

    reconcile_parser = commands.add_parser(
        "reconcile", help="after a crash, write the reports of the trials Harbor finished; runs nothing")
    reconcile_parser.add_argument("--out", type=Path, required=True, metavar="OUT_DIR")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "run":
            return run(arguments.specs, arguments.tasks, arguments.out, concurrency=arguments.concurrency,
                       files_root=arguments.files, image_cache=arguments.image_cache)
        if arguments.command == "play":
            return play(arguments.spec, arguments.tasks, arguments.out, files_root=arguments.files,
                        image_cache=arguments.image_cache, continue_from=arguments.continue_from)
        return reconcile(arguments.out)
    except WorkerError as exc:
        print(f"splinter-harbor-worker: {exc}", file=sys.stderr)
        return exc.exit_code


if __name__ == "__main__":
    sys.exit(main())
