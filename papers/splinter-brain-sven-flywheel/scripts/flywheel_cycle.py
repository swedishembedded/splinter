#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements controlled, reproducible validation of
# self-improving model pipelines for its clients. If your team needs expertise
# in proving that a learning loop turns verified experience into a better model
# without forgetting, you can procure our services by sending an email to
# info@swedishembedded.com.

"""One full turn of the flywheel, twice, on a controlled domain (experiments E2, E3).

The domain is invented facts (a bridge opened in a year, a river of a length)
whose subjects are invented names, so no model can know the values and nothing
can have leaked. The SOURCE of a generation is a directory with one sheet file
per fact: Splinter groups records by the source text they are grounded in, so a
fact is an independent unit of the held-out measurement only when it is a file
of its own. Splinter writes its own tasks from the sheets. The measurement is a
set of probes that word each fact differently from any sheet and that no
Splinter stage reads: a fact is answered when at least 3 of its 4 probes are
answered.

    flywheel_cycle.py build   --out DIR [--seed N] [--facts N]
    flywheel_cycle.py run     --out DIR --splinter BIN --brain BIN --policy-weights DIR
                              --helper DIR [--learn-arg ARG ...]
    flywheel_cycle.py summary --out DIR

Generation 0 and 1 are taught in turn (each by `splinter learn`, which
continues the champion and replays earlier releases); generation 2 is never
taught and measures what a trained policy answers about invented facts it was
not given. After each release the policy the alias serves is also asked every
probe through `brain serve`, from the serving manifest alone. After the second
release the alias is rolled back to the first and the probes are asked again.
Every command's raw output is kept in DIR/raw. The small models run in the
Splinter process; a run that needs the card for long holds the shared lock
itself (this script does not take it).
"""

import argparse
import importlib.util
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ANSWERED = 3
STEPS = ["base", "candidate-g0", "after-g0", "candidate-g1", "after-g1", "after-rollback"]


def load_generations(path):
    spec = importlib.util.spec_from_file_location("generations", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(args):
    out = Path(args.out)
    gens = load_generations(args.generations_py)
    record = out / "build.json"
    if record.exists():
        seed = json.loads(record.read_text())["seed"]
        if args.seed is not None and args.seed != seed:
            sys.exit(f"{record}: built with seed {seed}, not {args.seed}")
    else:
        seed = args.seed if args.seed is not None else secrets.randbits(64)
        print(f"seed {seed}", file=sys.stderr)
    out.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({"seed": seed, "facts_per_generation": args.facts}) + "\n")
    facts = gens.make_facts(seed, generations=3, per_generation=args.facts)
    (out / "facts.json").write_text(json.dumps([f.__dict__ for f in facts], indent=1))
    for g in range(3):
        mine = [f for f in facts if f.generation == g]
        sheets = out / f"sheets{g}"
        sheets.mkdir(exist_ok=True)
        for f in mine:
            (sheets / f"{f.key}.md").write_text(f"# {f.subject}\n\n{f.statement}\n")
        (out / f"probes{g}.jsonl").write_text(
            "".join(json.dumps(p) + "\n" for f in mine for p in gens.probes(f)))
    (out / "anchor.jsonl").write_text(
        "".join(json.dumps(a) + "\n" for a in gens.anchor_suite()))


class Runner:
    def __init__(self, args):
        self.out = Path(args.out)
        self.raw = self.out / "raw"
        self.raw.mkdir(exist_ok=True)
        self.state = self.out / "state"
        self.args = args
        self.env = dict(os.environ)
        self.env["BRAIN_QWEN_WEIGHTS"] = args.policy_weights
        self.env["SPLINTER_BRAIN_BIN"] = args.brain
        self.helper = f"local:{args.helper}@{args.helper_context}"
        self.log = self.raw / "commands.log"

    def splinter(self, name, *argv, ok=(0,)):
        command = [self.args.splinter, "--state", str(self.state), "--json", "-v",
                   "--policy-context-tokens", str(self.args.policy_context), *map(str, argv)]
        started = time.time()
        with open(self.log, "a") as log, open(self.raw / f"{name}.stderr", "w") as err:
            log.write("$ " + " ".join(command) + "\n")
            done = subprocess.run(command, stdout=subprocess.PIPE, stderr=err,
                                  env=self.env, text=True)
        elapsed = round(time.time() - started, 1)
        (self.raw / f"{name}.stdout").write_text(done.stdout)
        if done.returncode not in ok:
            sys.exit(f"{name}: exit {done.returncode}; see {self.raw / (name + '.stderr')}")
        report = json_report(done.stdout)
        (self.raw / f"{name}.json").write_text(json.dumps(
            {"command": command, "exit": done.returncode, "elapsed_s": elapsed,
             "report": report}, indent=1))
        return done.returncode, report

    def suite(self):
        suite = []
        for g in range(3):
            suite += [json.loads(l) for l in (self.out / f"probes{g}.jsonl").read_text().splitlines()]
        suite += [json.loads(l) for l in (self.out / "anchor.jsonl").read_text().splitlines()]
        return suite

    def measure(self, name, model="policy:default"):
        """Put `model` (the live policy, or a candidate by id) every probe and the anchor."""
        path = self.out / "suite.jsonl"
        path.write_text("".join(json.dumps(s) + "\n" for s in self.suite()))
        _, report = self.splinter(f"measure-{name}", "eval", model, "--suite", path)
        missed = [m["instruction"] for s in report.get("scores", []) for m in s.get("missed", [])]
        (self.out / f"missed-{name}.json").write_text(json.dumps(sorted(missed)))

    def serving_manifest(self):
        path = self.state / "serving" / "default.json"
        return json.loads(path.read_text()) if path.exists() else None

    def served_missed(self, name):
        """The probes the release the alias serves misses, asked over `brain serve`.

        Brain is started from the serving manifest alone, as a deployment would;
        greedy decoding. A probe is answered when the reference value appears as
        a whole number in the reply.
        """
        if self.serving_manifest() is None:
            return None
        port = 18000 + secrets.randbelow(1000)
        keys, ready = self.raw / f"serve-{name}.keys", self.raw / f"serve-{name}.ready"
        for stale in (keys, ready):
            stale.unlink(missing_ok=True)
        cmd = [self.args.brain, "serve", "--openai", f"127.0.0.1:{port}", "--adapter-targets", "all",
               "--api-keys-out", str(keys),
               "--ready-file", str(ready), "--adapter-manifest",
               str(self.state / "serving" / "default.json")]
        with open(self.raw / f"serve-{name}.log", "w") as log:
            server = subprocess.Popen(cmd, env=self.env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.time() + 600
            while not ready.exists():
                if server.poll() is not None or time.time() > deadline:
                    return {"error": "brain serve did not become ready", "missed": None}
                time.sleep(2)
            key = re.search(r"\S{16,}", keys.read_text()).group(0)
            auth = {"Authorization": f"Bearer {key}"}
            base = f"http://127.0.0.1:{port}/v1"
            listing = json.load(urllib.request.urlopen(
                urllib.request.Request(f"{base}/models", headers=auth), timeout=60))["data"]
            adapters = [m["id"] for m in listing if m.get("adapter")]
            if not adapters:
                return {"error": "no adapter is served", "models": listing, "missed": None}
            missed = []
            for item in self.suite():
                body = json.dumps({"model": adapters[0], "temperature": 0, "max_tokens": 128,
                                   "messages": [{"role": "user", "content": item["instruction"]}]})
                request = urllib.request.Request(
                    f"{base}/chat/completions", body.encode(),
                    {**auth, "Content-Type": "application/json"})
                reply = json.load(urllib.request.urlopen(request, timeout=300))
                text = reply["choices"][0]["message"]["content"]
                if not re.search(rf"(?<![\d.]){re.escape(item['reference'])}(?!\d)", text):
                    missed.append(item["instruction"])
            return {"adapter": adapters[0], "manifest": self.serving_manifest(),
                    "missed": sorted(missed)}
        finally:
            server.send_signal(signal.SIGTERM)
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                server.kill()


def json_report(stdout):
    """The JSON object a command printed: training prints its progress lines before it."""
    lines = stdout.splitlines()
    for n, line in enumerate(lines):
        if line.startswith("{"):
            try:
                return json.loads("\n".join(lines[n:]))
            except json.JSONDecodeError:
                continue
    return {"stdout": stdout}


def reason_of(stderr_path):
    """The last line a failed command printed on its standard error."""
    lines = [l for l in Path(stderr_path).read_text().splitlines() if l.strip()]
    return lines[-1] if lines else ""


def run(args):
    out = Path(args.out)
    r = Runner(args)
    shutil.rmtree(r.state, ignore_errors=True)
    r.state.mkdir(parents=True)
    exits = {}
    r.splinter("anchor-freeze", "eval", "--suite", "anchor", "--freeze", out / "anchor.jsonl")
    r.measure("base")
    for g in (0, 1):
        code, report = r.splinter(f"learn-g{g}", "learn", out / f"sheets{g}", "--kinds", "recall",
                             "--generator", r.helper, "--teacher", r.helper,
                             "--goal", "the invented places of the gazetteer",
                             *args.learn_arg, ok=(0, 1))
        exits[f"learn_g{g}"] = {"exit": code, "reason": reason_of(r.raw / f"learn-g{g}.stderr")}
        candidate = (report.get("candidate") or {}).get("candidate")
        if candidate:  # measured whether or not the gate released it
            r.measure(f"candidate-g{g}", candidate)
        r.measure(f"after-g{g}")
        r.splinter(f"releases-after-g{g}", "release", "list")
        (r.raw / f"serving-after-g{g}.json").write_text(json.dumps(r.serving_manifest(), indent=1))
        served = r.served_missed(f"after-g{g}")
        if served is not None:
            (out / f"served-after-g{g}.json").write_text(json.dumps(served, indent=1))
    code, _ = r.splinter("rollback", "rollback", "default", ok=(0, 1, 2))
    exits["rollback"] = {"exit": code, "reason": reason_of(r.raw / "rollback.stderr")}
    r.measure("after-rollback")
    (r.raw / "serving-after-rollback.json").write_text(json.dumps(r.serving_manifest(), indent=1))
    served = r.served_missed("after-rollback")
    if served is not None:
        (out / "served-after-rollback.json").write_text(json.dumps(served, indent=1))
    r.splinter("releases-final", "release", "list")
    (out / "exits.json").write_text(json.dumps(exits, indent=1) + "\n")


def row_of(name, missed, facts, anchor, gens):
    row = {"step": name, "anchor_correct": sum(a not in missed for a in anchor),
           "anchor_n": len(anchor)}
    for g in range(3):
        per_fact = {}
        mine = [f for f in facts if f["generation"] == g]
        for f in mine:
            ok = sum(p["instruction"] not in missed for p in gens.probes(gens.Fact(**f)))
            per_fact[f["subject"]] = ok >= ANSWERED
        row[f"g{g}_answered"] = sum(per_fact.values())
        row[f"g{g}_n"] = len(mine)
        row[f"g{g}_facts"] = per_fact
    return row


def summary(args):
    out = Path(args.out)
    facts = json.loads((out / "facts.json").read_text())
    anchor = [json.loads(l)["instruction"] for l in (out / "anchor.jsonl").read_text().splitlines()]
    gens = load_generations(args.generations_py)
    rows = []
    for name in STEPS:
        path = out / f"missed-{name}.json"
        if path.exists():
            rows.append(row_of(name, set(json.loads(path.read_text())), facts, anchor, gens))
    for name in STEPS[1:]:
        path = out / f"served-{name}.json"
        if path.exists() and json.loads(path.read_text()).get("missed") is not None:
            rows.append(row_of(f"served-{name}", set(json.loads(path.read_text())["missed"]),
                               facts, anchor, gens))
    text = json.dumps(rows, indent=1)
    (out / "summary.json").write_text(text + "\n")
    print(text)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("build", "run", "summary"):
        q = sub.add_parser(name)
        q.add_argument("--out", required=True)
        q.add_argument("--generations-py", default=str(
            Path(__file__).resolve().parents[3] / "samples/generations/generations.py"))
        if name == "build":
            q.add_argument("--seed", type=int)
            q.add_argument("--facts", type=int, default=30)
        if name == "run":
            q.add_argument("--splinter", required=True)
            q.add_argument("--brain", required=True)
            q.add_argument("--policy-weights", required=True)
            q.add_argument("--helper", required=True)
            q.add_argument("--helper-context", type=int, default=8192)
            q.add_argument("--policy-context", type=int, default=8192)
            q.add_argument("--learn-arg", action="append", default=[],
                           help="an extra argument of `splinter learn`, repeated (a flag and its value are two)")
    args = p.parse_args()
    {"build": build, "run": run, "summary": summary}[args.cmd](args)


if __name__ == "__main__":
    main()
