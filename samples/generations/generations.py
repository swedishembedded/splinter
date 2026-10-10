#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements measured validation of continual learning
# that consolidates into the weights, for its clients. If your team needs
# expertise in proving that a model keeps accumulating capabilities
# generation after generation, you can procure our services by sending an
# email to info@swedishembedded.com.

"""The generations sample: invented neutral facts taught in generations,
each generation several `splinter absorb` updates and then a consolidation
(`full` or `merge`), measured after every step.

    generations.py build  --out DIR [--seed N]
    generations.py run    --out DIR --mode full|merge --splinter BIN ...
    generations.py report --out DIR

The facts, their sessions, their held-out probes and the anchor questions are
written by `build` from a seed and are the same for every run; `run` never
looks at a probe except to grade the model on it.
"""

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

GENERATIONS = 3
FACTS_PER_GENERATION = 12
UPDATES_PER_GENERATION = 3
PROBES_PER_FACT = 4

# The criteria the run is judged by, declared before any run.
#
# * recall: at the last update of every generation each fact taught in it is
#   answered (at least three of its four held-out probes right).
# * retention: a fact answered stays answered; a measurement may be short of
#   all the earlier generations' facts by this many per earlier generation.
# * anchor: the general questions answered may fall short of the base's count
#   by this many of forty.
ANSWERED_PROBES = 3
RETENTION_TOLERANCE = 1
ANCHOR_TOLERANCE = 2


@dataclass(frozen=True)
class Fact:
    key: str
    generation: int
    index: int
    kind: str
    subject: str
    value: str
    wrong_value: str
    statement: str
    question: str
    wrong: str


SYLLABLES = ["mar", "row", "by", "quil", "fox", "tes", "se", "ra", "dun", "vel", "ith", "or",
             "len", "bra", "kov", "ul", "nim", "pa", "zor", "tal", "wen", "gar", "olm", "fre",
             "sky", "dri", "mon", "hap", "ber", "lix", "yar", "cun"]

# Each kind: how the subject is named from invented names, the range of its
# value, the question a session opens with, the statement a person corrects
# with, and four questions that are never in a session.
KINDS = {
    "legs": {
        "subject": "{A} of {P}", "range": (11, 97),
        "question": "How many legs does the {S} have?",
        "statement": "The {S} has {V} legs.",
        "probes": ["Tell me how many legs the {S} has.",
                   "The {S}: how many legs does it have?",
                   "If I counted the legs of a {S}, what number would I reach?",
                   "What is the leg count of the {S}?"],
    },
    "bridge": {
        "subject": "{A} Bridge", "range": (1702, 1989),
        "question": "In which year was the {S} opened?",
        "statement": "The {S} was opened in {V}.",
        "probes": ["Tell me the year the {S} was opened.",
                   "When, in what year, did the {S} open?",
                   "The {S} opened in which year?",
                   "What year did the opening of the {S} take place?"],
    },
    "distance": {
        "subject": "{P}", "range": (113, 987),
        "question": "How far is {S} from {Q}, in kilometres?",
        "statement": "{S} is {V} kilometres from {Q}.",
        "probes": ["What distance in kilometres separates {S} from {Q}?",
                   "Tell me how many kilometres it is from {S} to {Q}.",
                   "{S} lies how far from {Q}, in kilometres?",
                   "I travel from {Q} to {S}: how many kilometres is that?"],
    },
    "festival": {
        "subject": "{A} festival", "range": (11, 97),
        "question": "How often is the {S} held, in years?",
        "statement": "The {S} is held every {V} years.",
        "probes": ["Every how many years does the {S} take place?",
                   "Tell me the interval in years between two {S}s.",
                   "The {S} comes round after how many years?",
                   "What is the number of years between each {S}?"],
    },
    "tower": {
        "subject": "{A} Tower", "range": (131, 989),
        "question": "How tall is the {S}, in metres?",
        "statement": "The {S} is {V} metres tall.",
        "probes": ["Tell me the height of the {S} in metres.",
                   "The {S} stands how many metres high?",
                   "What is the height in metres of the {S}?",
                   "How many metres tall is the {S}?"],
    },
    "population": {
        "subject": "{P}", "range": (2113, 9987),
        "question": "What is the population of {S}?",
        "statement": "{S} has a population of {V}.",
        "probes": ["Tell me how many people live in {S}.",
                   "How many inhabitants does {S} have?",
                   "{S} is home to how many people?",
                   "What is the number of residents of {S}?"],
    },
    "river": {
        "subject": "{A} river", "range": (113, 987),
        "question": "How long is the {S}, in kilometres?",
        "statement": "The {S} is {V} kilometres long.",
        "probes": ["Tell me the length of the {S} in kilometres.",
                   "The {S} runs for how many kilometres?",
                   "What is the length in kilometres of the {S}?",
                   "How many kilometres long is the {S}?"],
    },
}
KIND_ORDER = list(KINDS)


def _name(rng, used):
    """An invented capitalised name of two or three syllables, new to `used`."""
    while True:
        word = "".join(rng.choice(SYLLABLES) for _ in range(rng.choice((2, 3)))).capitalize()
        if word not in used and len(word) >= 5:
            used.add(word)
            return word


def make_facts(seed, generations=GENERATIONS, per_generation=FACTS_PER_GENERATION):
    """The facts of every generation, in order, from `seed` alone."""
    rng = random.Random(seed)
    used = set()
    facts = []
    for n in range(generations * per_generation):
        kind = KIND_ORDER[n % len(KIND_ORDER)]
        spec = KINDS[kind]
        names = {"A": _name(rng, used), "P": _name(rng, used), "Q": _name(rng, used)}
        subject = spec["subject"].format(**names)
        low, high = spec["range"]
        value = rng.randint(low, high)
        wrong = rng.randint(low, high)
        while wrong == value or str(wrong) in str(value) or str(value) in str(wrong):
            wrong = rng.randint(low, high)
        subject_parts = {"S": subject, "Q": names["Q"]}
        statement = spec["statement"].format(V=value, **subject_parts)
        question = spec["question"].format(**subject_parts)
        facts.append(Fact(
            key=f"g{n // per_generation}f{n % per_generation}",
            generation=n // per_generation,
            index=n % per_generation,
            kind=kind,
            subject=subject,
            value=str(value),
            wrong_value=str(wrong),
            statement=statement,
            question=question,
            wrong=spec["statement"].format(V=wrong, **subject_parts),
        ))
    return facts


def session(fact, version="1"):
    """The session in which `fact` is taught: the person asks, the agent is
    wrong, the person corrects it, the agent acknowledges."""
    steps = [
        ("user", fact.question),
        ("agent", fact.wrong),
        ("user", f"No, that is wrong. {fact.statement}"),
        ("agent", "Thank you for the correction."),
    ]
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": f"session-{fact.key}",
        "agent": {"name": "scripted", "version": version},
        "steps": [{"step_id": n + 1, "source": who, "message": text}
                  for n, (who, text) in enumerate(steps)],
    }


def probes(fact):
    """Four held-out questions about `fact`, worded as no session words it."""
    spec = KINDS[fact.kind]
    q = _other_name(fact)
    return [{"instruction": text.format(S=fact.subject, Q=q), "reference": fact.value,
             "kind": "recall"}
            for text in spec["probes"]]


def sealed_probes(fact):
    """The same questions as `splinter absorb --sealed-probes` reads them: a
    training record containing one is refused."""
    return [{"question": p["instruction"], "reference": p["reference"],
             "name": f"{fact.key}.{n}"} for n, p in enumerate(probes(fact))]


def _other_name(fact):
    """The second place a distance fact names, read back out of its statement."""
    found = re.search(r"kilometres from (\w+)\.", fact.statement)
    return found.group(1) if found else ""


ANCHOR = [
    ("What is the capital of France?", "Paris"), ("What is the capital of Japan?", "Tokyo"),
    ("What is the capital of Italy?", "Rome"), ("What is the capital of Egypt?", "Cairo"),
    ("What is the capital of Canada?", "Ottawa"), ("What is the capital of Spain?", "Madrid"),
    ("What is the capital of Germany?", "Berlin"), ("What is the capital of Norway?", "Oslo"),
    ("What is the chemical symbol for gold?", "Au"), ("What is the chemical symbol for sodium?", "Na"),
    ("What is the chemical symbol for iron?", "Fe"), ("What is the chemical symbol for silver?", "Ag"),
    ("How many days are in a week?", "7"), ("How many months are in a year?", "12"),
    ("How many sides does a triangle have?", "3"), ("How many legs does a spider have?", "8"),
    ("What is 7 plus 8?", "15"), ("What is 9 times 6?", "54"), ("What is 100 minus 37?", "63"),
    ("What is 12 times 12?", "144"), ("What is 81 divided by 9?", "9"), ("What is 25 plus 17?", "42"),
    ("What is the largest planet in the solar system?", "Jupiter"),
    ("What planet is known as the red planet?", "Mars"),
    ("What is the closest star to the Earth?", "Sun"),
    ("What gas do plants absorb from the air?", "carbon dioxide"),
    ("What is the freezing point of water in degrees Celsius?", "0"),
    ("What is the boiling point of water in degrees Celsius?", "100"),
    ("Who wrote the play Romeo and Juliet?", "Shakespeare"),
    ("What language is spoken in Brazil?", "Portuguese"),
    ("What is the largest ocean on Earth?", "Pacific"),
    ("What is the longest river in Egypt?", "Nile"),
    ("What colour do you get by mixing blue and yellow?", "green"),
    ("What is the square root of 64?", "8"), ("How many continents are there?", "7"),
    ("What is the first letter of the alphabet?", "A"),
    ("What is the opposite of hot?", "cold"), ("What do bees make?", "honey"),
    ("What metal is liquid at room temperature?", "mercury"),
    ("How many hours are in a day?", "24"),
]


def anchor_suite():
    return [{"instruction": q, "reference": a, "kind": "recall"} for q, a in ANCHOR]


def answered_facts(facts, missed):
    """The keys of the facts with at least ANSWERED_PROBES of their probes not
    among the `missed` instructions."""
    missed = set(missed)
    return {f.key for f in facts
            if sum(p["instruction"] not in missed for p in probes(f)) >= ANSWERED_PROBES}


def judge(rows, anchor_baseline):
    """Whether the run met the criteria declared at the top of this file."""
    by_generation = {}
    for r in rows:
        if r["kind"] == "update":
            by_generation[r["generation"]] = r
    recall = {g: {"answered": r["current"][0], "taught": r["current"][1]}
              for g, r in by_generation.items()}
    recall_held = all(v["taught"] > 0 and v["answered"] == v["taught"] for v in recall.values())
    short = []
    for r in rows:
        answered, held = r["retention"]
        allowed = RETENTION_TOLERANCE * r["generation"]
        if held and answered < held - allowed:
            short.append({"step": r["step"], "answered": answered, "held": held})
    low = [{"step": r["step"], "anchor": r["anchor"]} for r in rows
           if r["anchor"] < anchor_baseline - ANCHOR_TOLERANCE]
    verdict = {
        "recall": {"held": recall_held, "by_generation": recall},
        "retention": {"held": not short, "short": short},
        "anchor": {"held": not low, "baseline": anchor_baseline, "low": low},
    }
    verdict["success"] = all(v["held"] for v in verdict.values())
    return verdict


# --- the build ---------------------------------------------------------------

def build(out, seed, generations=GENERATIONS):
    out = Path(out)
    facts = make_facts(seed, generations=generations)
    out.mkdir(parents=True, exist_ok=True)
    (out / "facts.json").write_text(json.dumps([f.__dict__ for f in facts], indent=1))
    per_update = FACTS_PER_GENERATION // UPDATES_PER_GENERATION
    for f in facts:
        update = f.index // per_update
        folder = out / "sessions" / f"g{f.generation}u{update}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{f.key}.atif.json").write_text(json.dumps(session(f), indent=1))
    (out / "sealed.jsonl").write_text(
        "".join(json.dumps(p) + "\n" for f in facts for p in sealed_probes(f)))
    (out / "anchor.jsonl").write_text(
        "".join(json.dumps(a) + "\n" for a in anchor_suite()))
    return facts


def load_facts(out):
    return [Fact(**d) for d in json.loads((Path(out) / "facts.json").read_text())]


# --- the run -----------------------------------------------------------------

class Splinter:
    """The `splinter` binary on one state root."""

    def __init__(self, binary, state, log, env, context):
        self.binary, self.state, self.log, self.env = binary, state, log, env
        self.context = context

    def run(self, *args, ok=(0,)):
        command = [self.binary, "--state", str(self.state), "--json",
                   "--policy-context-tokens", str(self.context), *map(str, args)]
        with open(self.log, "a") as log:
            log.write("$ " + " ".join(command) + "\n")
            log.flush()
            done = subprocess.run(command, stdout=subprocess.PIPE, stderr=log,
                                  env=self.env, text=True)
        if done.returncode not in ok:
            raise SystemExit(f"{' '.join(command)} exited {done.returncode}; see {self.log}")
        try:
            return json.loads(done.stdout)
        except json.JSONDecodeError:
            return {"stdout": done.stdout}


def measure(splinter, work, facts_taught, earlier, all_facts, anchor):
    """Puts the policy the questions of the facts taught so far and the anchor;
    returns the answered counts. One eval, so the model loads once."""
    suite = [p for f in all_facts if f.key in {t.key for t in facts_taught + earlier}
             for p in probes(f)] + anchor
    path = work / "suite.jsonl"
    path.write_text("".join(json.dumps(s) + "\n" for s in suite))
    report = splinter.run("eval", "policy:default", "--suite", path)
    missed = []
    for score in report.get("scores", []):
        missed += [m["instruction"] for m in score.get("missed", [])]
    anchor_missed = {a["instruction"] for a in anchor} & set(missed)
    now = answered_facts(facts_taught, missed)
    before = answered_facts(earlier, missed)
    return {
        "current": [len(now), len(facts_taught)],
        "retention": [len(before), len(earlier)],
        "anchor": len(anchor) - len(anchor_missed),
        "answered": sorted(now | before),
    }


class Cycle:
    """The updates and consolidations of one policy on one state root."""

    def __init__(self, args, facts, state, tag):
        self.args, self.facts, self.state = args, facts, Path(state)
        self.out = Path(args.out)
        self.anchor = anchor_suite()
        self.work = self.out / f"work-{tag}"
        self.work.mkdir(parents=True, exist_ok=True)
        self.results = self.out / f"results-{tag}.jsonl"
        self.log = self.out / f"splinter-{tag}.log"
        self.rows = []
        env = dict(os.environ)
        env["BRAIN_QWEN_WEIGHTS"] = args.policy_weights
        if args.brain:
            env["SPLINTER_BRAIN_BIN"] = args.brain
        self.splinter = Splinter(args.splinter, self.state, self.log, env, args.policy_context)
        role = f"local:{args.helper}@{args.helper_context}"
        self.roles = ["--generator", role, "--teacher", role, "--judge", role]

    def record(self, step, generation, kind, started, ended, extra):
        row = {"step": step, "generation": generation, "kind": kind,
               "t_start": started, "t_end": ended, "wall_s": round(ended - started, 1), **extra}
        self.rows.append(row)
        with open(self.results, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    def measure(self, taught, earlier):
        return measure(self.splinter, self.work, taught, earlier, self.facts, self.anchor)

    def stage_seconds(self, report):
        """Seconds each stage of a recorded run took, from its run record."""
        run = report.get("run")
        if not run:
            return {}
        shown = self.splinter.run("runs", "show", run)
        return {st["stage"]: round(st["duration_ms"] / 1000, 1)
                for st in shown.get("stages", []) if st.get("duration_ms") is not None}

    def start(self):
        self.splinter.run("eval", "--suite", "anchor", "--freeze", self.out / "anchor.jsonl")
        base = self.measure([], [])
        self.record("base", -1, "base", time.time(), time.time(), base)

    def update(self, generation, update):
        mine = [f for f in self.facts if f.generation == generation]
        earlier = [f for f in self.facts if f.generation < generation]
        started = time.time()
        report = self.splinter.run("absorb", self.out / "sessions" / f"g{generation}u{update}",
                                   *self.roles, "--no-consolidate",
                                   "--sealed-probes", self.out / "sealed.jsonl", ok=(0, 1))
        ended = time.time()
        per = FACTS_PER_GENERATION // UPDATES_PER_GENERATION
        measured = self.measure(mine[:(update + 1) * per], earlier)
        gate = report.get("gate") or {}
        self.record(f"g{generation}.u{update}", generation, "update", started, ended, {
            **measured, "release": report.get("release"), "gate_passed": gate.get("passed"),
            "gate_answered": [gate.get("answered"), len(gate.get("claims", []))],
            "stopped": report.get("stopped"),
            "records": (report.get("dataset") or {}).get("trained"),
            "replayed": report.get("replayed_records"),
            "stage_s": self.stage_seconds(report),
        })

    def consolidate(self, generation, mode, lr=None, epochs=None, tag=""):
        mine = [f for f in self.facts if f.generation == generation]
        earlier = [f for f in self.facts if f.generation < generation]
        tuning = ([] if lr is None else ["--lr", lr]) + ([] if epochs is None else ["--epochs", epochs])
        started = time.time()
        report = self.splinter.run("consolidate", "--mode", mode, *tuning, ok=(0, 1))
        ended = time.time()
        done = report.get("consolidated") or {}
        measured = self.measure(mine, earlier)
        self.record(f"g{generation}.consolidate{tag}", generation, "consolidate", started, ended, {
            **measured, "release": done.get("release"), "mode": mode, "lr": lr, "epochs": epochs,
            "gate_passed": (done.get("gate") or {}).get("passed"),
            "stopped": done.get("stopped") or report.get("stopped"),
            "steps": done.get("steps_completed"), "final_loss": done.get("final_loss"),
            "records": (done.get("dataset") or {}).get("trained"),
            "replayed": done.get("replayed_records"),
            "stage_s": self.stage_seconds(report),
        })


def run(args):
    out = Path(args.out)
    facts = load_facts(out)
    cycle = Cycle(args, facts, out / f"state-{args.mode}", args.mode)
    cycle.start()
    for generation in range(1 + max(f.generation for f in facts)):
        for update in range(UPDATES_PER_GENERATION):
            cycle.update(generation, update)
        cycle.consolidate(generation, args.mode, args.lr, args.epochs)
    verdict = judge(cycle.rows, anchor_baseline=cycle.rows[0]["anchor"])
    (out / f"verdict-{args.mode}.json").write_text(json.dumps(verdict, indent=1))
    print(json.dumps(verdict), flush=True)


def tune(args):
    """One generation's updates once, then its full consolidation at each
    learning rate on a copy of that state: how fast the weights may move and
    still take the facts, found on facts of a seed of its own."""
    out = Path(args.out)
    facts = load_facts(out)
    base = Cycle(args, facts, out / "state-tune", "tune")
    base.start()
    for update in range(UPDATES_PER_GENERATION):
        base.update(0, update)
    for lr in args.lrs:
        copy = out / f"state-tune-{lr}"
        shutil.copytree(base.state, copy, symlinks=True)
        fork = Cycle(args, facts, copy, f"tune-{lr}")
        fork.consolidate(0, "full", lr=lr, epochs=args.epochs, tag=f"@{lr}")
        shutil.rmtree(copy)


# --- the report --------------------------------------------------------------

def peaks(samples, started, ended):
    """The highest GPU and host memory a watcher sampled between two instants."""
    inside = [s for s in samples if started <= s["t"] <= ended]
    if not inside:
        return None, None
    gpu = max(max(s["gpu_used_mib"]) for s in inside)
    return gpu, max(s["rss_mib"] for s in inside)


def report(args):
    out = Path(args.out)
    samples = []
    watch = out / "memwatch.jsonl"
    if watch.exists():
        samples = [json.loads(line) for line in watch.read_text().splitlines() if line]
    lines = []
    for mode in ("full", "merge"):
        path = out / f"results-{mode}.jsonl"
        if not path.exists():
            continue
        lines.append(f"\n### mode `{mode}`\n")
        lines.append("| step | wall s | current | earlier | anchor /40 | release gate | peak GPU MiB "
                     "| peak host MiB |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for line in path.read_text().splitlines():
            r = json.loads(line)
            gpu, rss = peaks(samples, r["t_start"], r["t_end"])
            cur = f"{r['current'][0]}/{r['current'][1]}"
            ret = f"{r['retention'][0]}/{r['retention'][1]}"
            lines.append(f"| {r['step']} | {r['wall_s']} | {cur} | {ret} | {r['anchor']} | "
                         f"{r.get('gate_passed')} | {gpu if gpu is not None else '-'} | "
                         f"{rss if rss is not None else '-'} |")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--seed", type=int, default=20261010)
    b.add_argument("--generations", type=int, default=GENERATIONS)
    r = sub.add_parser("run")
    r.add_argument("--mode", choices=("full", "merge"), required=True)
    t = sub.add_parser("tune")
    t.add_argument("--lrs", nargs="+", required=True, help="learning rates to consolidate at")
    t.add_argument("--epochs")
    for q in (r, t):
        q.add_argument("--out", required=True)
        q.add_argument("--splinter", required=True)
        q.add_argument("--brain")
        q.add_argument("--policy-weights", required=True, help="the policy's base checkpoint")
        q.add_argument("--helper", required=True,
                       help="the checkpoint that plays generator, teacher and judge")
        q.add_argument("--helper-context", type=int, default=8192)
        q.add_argument("--policy-context", type=int, default=8192,
                       help="the policy's context in tokens (its KV cache is built for exactly this)")
    r.add_argument("--lr", help="the full consolidation's peak learning rate (default: the command's)")
    r.add_argument("--epochs", help="the full consolidation's passes over its records (default: the command's)")
    s = sub.add_parser("report")
    s.add_argument("--out", required=True)
    args = p.parse_args()
    if args.command == "build":
        build(args.out, args.seed, args.generations)
    elif args.command == "run":
        run(args)
    elif args.command == "tune":
        tune(args)
    else:
        report(args)


if __name__ == "__main__":
    sys.exit(main())
