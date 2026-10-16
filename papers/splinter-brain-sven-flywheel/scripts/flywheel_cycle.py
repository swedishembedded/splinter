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
can have leaked. A fact sheet ("handbook") of each generation is the SOURCE;
Splinter writes its own tasks from it. The measurement is a set of probes that
code words differently from any sheet and that no Splinter stage reads: a fact
is answered when at least 3 of its 4 probes are answered.

    flywheel_cycle.py build   --out DIR [--seed N] [--facts N]
    flywheel_cycle.py run     --out DIR --splinter BIN --policy-weights DIR --helper DIR
    flywheel_cycle.py summary --out DIR

Generation 0 and 1 are taught in turn (each by `splinter learn`, which
continues the champion and replays earlier releases); generation 2 is never
taught and measures what a trained policy answers about invented facts it was
not given. After the second release the alias is rolled back to the first and
the probes are asked again. Every command's raw JSON output is kept in
DIR/raw. Each splinter command runs under the shared GPU lock (`gpu`).
"""

import argparse
import importlib.util
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path

PER_PROBE = 4
ANSWERED = 3


def load_generations(path):
    spec = importlib.util.spec_from_file_location("generations", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def handbook(facts, generation):
    """One section per fact: a heading naming the subject and the statement."""
    lines = [f"# Gazetteer of invented places, volume {generation}", ""]
    for f in facts:
        lines += [f"## {f.subject}", "", f.statement, ""]
    return "\n".join(lines)


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
        (out / f"handbook{g}.md").write_text(handbook(mine, g))
        (out / f"probes{g}.jsonl").write_text(
            "".join(json.dumps(p) + "\n" for f in mine for p in gens.probes(f)))
    (out / "anchor.jsonl").write_text(
        "".join(json.dumps(a) + "\n" for a in gens.anchor_suite()))
    return facts, gens


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
        self.env["BRAIN_BACKEND"] = "cuda"
        self.helper = f"local:{args.helper}@{args.helper_context}"
        self.log = self.raw / "commands.log"

    def splinter(self, name, *argv, ok=(0,)):
        command = ["gpu", self.args.splinter, "--state", str(self.state), "--json",
                   "--policy-context-tokens", str(self.args.policy_context), *map(str, argv)]
        started = time.time()
        with open(self.log, "a") as log:
            log.write("$ " + " ".join(command) + "\n")
            log.flush()
            done = subprocess.run(command, stdout=subprocess.PIPE, stderr=log,
                                  env=self.env, text=True)
        elapsed = round(time.time() - started, 1)
        (self.raw / f"{name}.stdout").write_text(done.stdout)
        if done.returncode not in ok:
            sys.exit(f"{name}: exit {done.returncode}; see {self.log}")
        try:
            report = json.loads(done.stdout)
        except json.JSONDecodeError:
            report = {"stdout": done.stdout}
        (self.raw / f"{name}.json").write_text(json.dumps(
            {"command": command, "exit": done.returncode, "elapsed_s": elapsed,
             "report": report}, indent=1))
        return done.returncode, report

    def measure(self, name):
        """Put the live policy every probe and the anchor; one eval, one load."""
        suite = []
        for g in range(3):
            suite += [json.loads(l) for l in (self.out / f"probes{g}.jsonl").read_text().splitlines()]
        suite += [json.loads(l) for l in (self.out / "anchor.jsonl").read_text().splitlines()]
        path = self.out / "suite.jsonl"
        path.write_text("".join(json.dumps(s) + "\n" for s in suite))
        _, report = self.splinter(f"measure-{name}", "eval", "policy:default", "--suite", path)
        missed = [m["instruction"] for s in report.get("scores", []) for m in s.get("missed", [])]
        (self.out / f"missed-{name}.json").write_text(json.dumps(sorted(missed)))
        return missed

    def serving_digest(self):
        path = self.state / "serving" / "default.json"
        return json.loads(path.read_text()) if path.exists() else None


def run(args):
    out = Path(args.out)
    r = Runner(args)
    shutil.rmtree(r.state, ignore_errors=True)
    r.splinter("anchor-freeze", "eval", "--suite", "anchor", "--freeze", out / "anchor.jsonl")
    r.measure("base")
    for g in (0, 1):
        code, _ = r.splinter(f"learn-g{g}", "learn", out / f"handbook{g}.md",
                             "--kinds", "recall", "--generator", r.helper, "--teacher", r.helper,
                             "--goal", "the invented places of the gazetteer", ok=(0, 1))
        (out / f"learn-g{g}.exit").write_text(f"{code}\n")
        r.measure(f"after-g{g}")
        _, listing = r.splinter(f"releases-after-g{g}", "release", "list")
        manifest = r.serving_digest()
        (r.raw / f"serving-after-g{g}.json").write_text(json.dumps(manifest, indent=1))
    code, _ = r.splinter("rollback", "rollback", "default", ok=(0, 1, 2))
    (out / "rollback.exit").write_text(f"{code}\n")
    r.measure("after-rollback")
    (r.raw / "serving-after-rollback.json").write_text(json.dumps(r.serving_digest(), indent=1))
    r.splinter("releases-final", "release", "list")


def summary(args):
    out = Path(args.out)
    facts = json.loads((out / "facts.json").read_text())
    anchor = [json.loads(l)["instruction"] for l in (out / "anchor.jsonl").read_text().splitlines()]
    gens = load_generations(args.generations_py)
    rows = []
    for name in ["base", "after-g0", "after-g1", "after-rollback"]:
        path = out / f"missed-{name}.json"
        if not path.exists():
            continue
        missed = set(json.loads(path.read_text()))
        row = {"step": name, "anchor_correct": sum(a not in missed for a in anchor), "anchor_n": len(anchor)}
        for g in range(3):
            answered = 0
            per_fact = {}
            mine = [f for f in facts if f["generation"] == g]
            for f in mine:
                fact = gens.Fact(**f)
                ok = sum(p["instruction"] not in missed for p in gens.probes(fact))
                per_fact[f["subject"]] = ok >= ANSWERED
                answered += ok >= ANSWERED
            row[f"g{g}_answered"] = answered
            row[f"g{g}_n"] = len(mine)
            row[f"g{g}_facts"] = per_fact
        rows.append(row)
    text = json.dumps(rows, indent=1)
    (out / "summary.json").write_text(text + "\n")
    print(text)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("build", "run", "summary"):
        q = sub.add_parser(name)
        q.add_argument("--out", required=True)
        q.add_argument("--generations-py", default=str(Path.home() / "code/splinter/samples/generations/generations.py"))
        if name == "build":
            q.add_argument("--seed", type=int)
            q.add_argument("--facts", type=int, default=24)
        if name == "run":
            q.add_argument("--splinter", required=True)
            q.add_argument("--brain", required=True)
            q.add_argument("--policy-weights", required=True)
            q.add_argument("--helper", required=True)
            q.add_argument("--helper-context", type=int, default=8192)
            q.add_argument("--policy-context", type=int, default=8192)
    args = p.parse_args()
    {"build": build, "run": run, "summary": summary}[args.cmd](args)


if __name__ == "__main__":
    main()
