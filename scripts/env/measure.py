#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements measured training environments for
# reinforcement learning, for its clients. If your team needs expertise in
# calibrating environment difficulty to a model before a campaign spends
# compute on it, you can procure our services by sending an email to
# info@swedishembedded.com.

"""Measure a base policy's pass rate on environments' dev instances and record
it with `splinter env band`.

For each environment, the dev instances are exported as task folders and
attempted `--attempts` times each:

* an environment whose solver has files only (program output, document
  questions) is attempted closed-book: the instruction is sent to the model's
  OpenAI-compatible endpoint, the reply (minus a thinking block, and the
  contents of one fenced block when the reply has one) is written to
  `answer.txt`, and the task's own verifier grades it;
* an environment whose solver has a shell (database workflows, repositories,
  firmware) is attempted by `sven rollout run` on the exported folders, and its
  recorded rewards are read back.

A reward of exactly 1 is a pass. The exported folders are removed at the end.
Nothing here starts a model server or takes a GPU: serve the model first, under
the lock of the card it runs on, and hold the lock only while this runs.

Usage: measure.py --splinter BIN --store DIR --endpoint URL --model NAME \
           --work DIR ENV-ID...
"""
import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import urllib.request

ANSWER_SUFFIX = (
    "\n\nReply with only the contents of the answer file, line by line, and nothing else: "
    "no explanation, no code fence, no markdown."
)


def splinter(args, *more):
    out = subprocess.run([args.splinter, *more, "--json"], capture_output=True, text=True, check=False)
    if out.returncode != 0:
        sys.exit("splinter %s failed: %s" % (" ".join(more), out.stderr.strip()))
    return json.loads(out.stdout)


def solver_tools(folder):
    meta = tomllib.load(open(os.path.join(folder, "task.toml"), "rb"))
    return meta["metadata"]["splinter_env"]["solver_tools"]


def reply_to_answer(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    fenced = re.findall(r"```[a-zA-Z]*\n(.*?)```", text, flags=re.S)
    return (fenced[0] if len(fenced) == 1 else text).strip() + "\n"


def complete(args, prompt):
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt + ANSWER_SUFFIX}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    request = urllib.request.Request(
        args.endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + args.api_key},
    )
    with urllib.request.urlopen(request, timeout=args.timeout) as reply:
        return json.load(reply)["choices"][0]["message"]["content"]


def grade(folder, answer):
    """Runs the task's verifier on a copy of its workspace holding `answer`."""
    scratch = tempfile.mkdtemp(prefix="grade-", dir=os.environ.get("TMPDIR"))
    try:
        app = os.path.join(scratch, "app")
        shutil.copytree(os.path.join(folder, "environment"), app)
        with open(os.path.join(app, "answer.txt"), "w") as out:
            out.write(answer)
        logs = os.path.join(scratch, "logs")
        os.makedirs(logs)
        env = dict(os.environ, APP_DIR=app, TESTS_DIR=os.path.join(folder, "tests"), LOGS_DIR=logs)
        subprocess.run(["sh", os.path.join(folder, "tests", "test.sh")], env=env, cwd=app, capture_output=True, timeout=120)
        try:
            return json.load(open(os.path.join(logs, "reward.json"))).get("reward", 0.0) >= 1.0
        except (OSError, ValueError):
            return False
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def closed_book(args, folders):
    jobs = [(folder, n) for folder in folders for n in range(args.attempts)]

    def one(job):
        folder, _ = job
        prompt = open(os.path.join(folder, "instruction.md")).read()
        try:
            return grade(folder, reply_to_answer(complete(args, prompt)))
        except Exception as error:  # an endpoint failure is not a wrong answer
            print("attempt failed: %s" % error, file=sys.stderr)
            return None

    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(one, jobs))
    graded = [r for r in results if r is not None]
    return len(graded), sum(1 for r in graded if r)


def rollouts(args, folders, out):
    tasks = os.path.dirname(folders[0])
    subprocess.run(
        [args.sven, *(["-c", args.sven_config] if args.sven_config else []), "rollout", "run", tasks, "--out", out, "--attempts", str(args.attempts), "--jobs", str(args.jobs), "--no-confine"],
        check=False,
    )
    attempts = passes = 0
    for root, _, files in os.walk(out):
        for name in files:
            if name == "attempt.jsonl":
                for line in open(os.path.join(root, name)):
                    record = json.loads(line)
                    reward = record.get("reward")
                    if isinstance(reward, dict):
                        reward = reward.get("reward")
                    if reward is not None:
                        attempts += 1
                        passes += 1 if reward >= 1.0 else 0
    return attempts, passes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splinter", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--endpoint", required=True, help="the model's OpenAI-compatible base URL")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="none")
    parser.add_argument("--sven", help="the sven binary, for the environments whose solver has a shell")
    parser.add_argument("--sven-config", help="the sven configuration naming the served model (its base_url and name)")
    parser.add_argument("--work", required=True, help="a new directory the task folders and rollouts are written under")
    parser.add_argument("--attempts", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("environments", nargs="+", metavar="ENV-ID")
    args = parser.parse_args()
    os.makedirs(args.work)
    summary = {}
    try:
        for env in args.environments:
            out = os.path.join(args.work, env)
            folders = splinter(args, "env", "export", env, "--split", "dev", "--out", out, "--store", args.store)["directories"]
            if solver_tools(folders[0]) == "files":
                attempts, passes = closed_book(args, folders)
            elif args.sven:
                attempts, passes = rollouts(args, folders, os.path.join(args.work, "rollouts-" + env))
            else:
                print("%s needs a shell solver: pass --sven" % env, file=sys.stderr)
                continue
            summary[env] = {"attempts": attempts, "passes": passes}
            if attempts:
                splinter(args, "env", "band", env, "--model", args.model, "--attempts", str(attempts), "--passes", str(passes), "--store", args.store)
            print("%s: %d of %d attempts passed" % (env, passes, attempts), flush=True)
    finally:
        shutil.rmtree(args.work, ignore_errors=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
