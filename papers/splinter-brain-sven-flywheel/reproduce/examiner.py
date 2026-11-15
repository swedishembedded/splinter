#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements independent examination of self-trained models,
# graded on held-out exams the learner never reads, for its clients. If your team
# needs expertise in measuring whether a model really improved, you can procure
# our services by sending an email to info@swedishembedded.com.

"""The reference examiner of the paper: an implementation of the examination protocol the paper
specifies (Method, "The examiner"), written from that specification and nothing else.

    examiner.py grade   --exam E --evidence ROOT --series S --brain BIN --checkpoint DIR --out OUT
    examiner.py analyse --exam E --evidence ROOT --series S --plan TOML --out OUT
    examiner.py compare --aggregate OUT/aggregate.json --measurements data/measurements.json --out FILE

`grade` serves the base and every candidate of the series on one `brain serve` per lineage (a
lineage's adapters share card ids with the other lineages', so a server offers one lineage at a
time), checks that the served base is the declared checkpoint and every adapter has its declared
digest, asks every exam item once and writes one grade file per model. `analyse` reads the grade
files and decides each step and the plan; it needs no GPU. `compare` sets the result beside the
published verdicts' numbers.

The protocol, as the paper states it:
  * request: the item's instruction followed by the environment's one-sentence answer instruction
    (REQUEST_SUFFIX) as the only user message, no system prompt, thinking off, one greedy completion
    of at most MAX_TOKENS tokens;
  * answer: the content of the reply's last complete fenced block, else the whole reply; a reply the
    token limit cut is graded as written (no retry, no repair);
  * comparison: token by token per line, whitespace runs collapsed, blank lines ignored;
  * exclusion: an exam item whose program equals, exactly or after removing comments and whitespace,
    a program on either compared model's hand-off list is left out of that step. The list holds every
    program the candidate's chain trained on and, beside them, its held-out dev and retention
    programs; all of them are matched, so the exclusion is a superset of the training overlap (the one
    item the published S4 verdicts excluded equals a dev program of lineage r2, not a training one);
  * a step: wins (candidate right, predecessor wrong) and losses (the reverse) over the compared
    items, the exact one-sided sign test, and the paired gain with a 95% interval clustered over
    the exam's templates (Student-t with G-1 degrees of freedom); the step holds when wins exceed
    losses at p < alpha;
  * the plan: a lineage succeeds when all its steps hold; the plan is met when at least the
    registered number of lineages succeed.
"""

import argparse
import concurrent.futures
import hashlib
import json
import math
import re
import socket
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

# The environment's answer instruction, appended to every item's instruction (registration of S1,
# arm H: "What does the following Lua code print? ... Answer with exactly what it prints, in one
# fenced code block").
REQUEST_SUFFIX = "\n\nAnswer with exactly what it prints, in one fenced code block."
MAX_TOKENS = 2048
KV_CACHE = "f32"
MAX_BATCH = 64

# ------------------------------------------------------------------ grading


_FENCE = re.compile(r"```[^\n`]*\n(.*?)```", re.S)
_THINK = re.compile(r"\A\s*<think>.*?</think>", re.S)


def extract_answer(reply):
    """The content of the reply's last complete fenced block, else the whole reply. An empty
    thinking block that a chat template emits with thinking off is not part of the answer."""
    reply = _THINK.sub("", reply, count=1)
    blocks = _FENCE.findall(reply)
    return blocks[-1] if blocks else reply


def tokens_per_line(text):
    """Each non-blank line as its whitespace-separated tokens."""
    return [line.split() for line in text.splitlines() if line.strip()]


def answers_match(answer, reference):
    """Token-by-token comparison per line, whitespace runs collapsed, blank lines ignored."""
    return tokens_per_line(answer) == tokens_per_line(reference)


def grade_reply(reply, reference):
    """Whether a reply answers the item, by the protocol's extraction and comparison."""
    return answers_match(extract_answer(reply), reference)


# ------------------------------------------------------------------ programs and exclusion

_LONG_BRACKET = re.compile(r"\[(=*)\[")
_ITEM_PROGRAM = re.compile(r"```lua\n(.*?)\n?```", re.S)


def item_program(instruction):
    """The Lua program of an exam item: the last ```lua block of its instruction."""
    found = _ITEM_PROGRAM.findall(instruction)
    if not found:
        raise ValueError("an exam instruction holds no ```lua block")
    return found[-1]


def strip_comments(src):
    """The program with its Lua comments (`--` to the end of the line, `--[[ ... ]]` and
    `--[==[ ... ]==]` long comments) removed; string literals, short and long, are kept as they are,
    so a `--` inside a string is not a comment."""
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if src.startswith("--", i):
            m = _LONG_BRACKET.match(src, i + 2)
            if m:
                close = "]" + m.group(1) + "]"
                end = src.find(close, m.end())
                i = n if end < 0 else end + len(close)
            else:
                end = src.find("\n", i)
                i = n if end < 0 else end
            continue
        if c in "\"'":
            j = i + 1
            while j < n and src[j] != c and src[j] != "\n":
                j += 2 if src[j] == "\\" else 1
            out.append(src[i:j + 1])
            i = j + 1
            continue
        m = _LONG_BRACKET.match(src, i)
        if m:
            close = "]" + m.group(1) + "]"
            end = src.find(close, m.end())
            end = n if end < 0 else end + len(close)
            out.append(src[i:end])
            i = end
            continue
        out.append(c)
        i += 1
    return "".join(out)


def normalised_program(src):
    """The program with comments and every whitespace character removed."""
    return re.sub(r"\s+", "", strip_comments(src))


def excluded_items(programs, listed):
    """Indices of the exam programs equal to a listed program, exactly or once both have their
    comments and whitespace removed."""
    exact = set(listed)
    normal = {normalised_program(p) for p in listed}
    return {i for i, p in enumerate(programs) if p in exact or normalised_program(p) in normal}


# ------------------------------------------------------------------ statistics


def sign_test_p(wins, losses):
    """The exact one-sided sign test: P(X >= wins) for X ~ Binomial(wins + losses, 1/2)."""
    n = wins + losses
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n


def _betacf(a, b, x):
    """Continued fraction of the regularised incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            return h
    raise ArithmeticError("incomplete beta continued fraction did not converge")


def _betainc(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t, df):
    """The Student-t distribution function with `df` degrees of freedom."""
    tail = 0.5 * _betainc(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t >= 0 else tail


def t_quantile(q, df):
    """The `q` quantile of the Student-t distribution (0.5 < q < 1), by bisection."""
    if not 0.5 < q < 1.0:
        raise ValueError("t_quantile takes 0.5 < q < 1")
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < q:
        hi *= 2.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def clustered_gain(diffs, clusters):
    """The mean paired difference d_i in {-1, 0, 1} with its 95% interval clustered over templates:
    SE = sqrt(G/(G-1) * sum_c (sum_{i in c} (d_i - mean))^2) / n, two-sided Student-t with G-1
    degrees of freedom, and the one-sided 95% lower bound. Undefined (None) with fewer than two
    clusters or no items."""
    n = len(diffs)
    if n == 0:
        return {"mean": None, "interval": None, "lower_one_sided": None, "se": None, "families": 0, "items": 0}
    mean = sum(diffs) / n
    sums = {}
    for d, c in zip(diffs, clusters):
        sums[c] = sums.get(c, 0.0) + (d - mean)
    g = len(sums)
    if g < 2:
        return {"mean": mean, "interval": None, "lower_one_sided": None, "se": None, "families": g, "items": n}
    se = math.sqrt(g / (g - 1) * sum(s * s for s in sums.values())) / n
    t2 = t_quantile(0.975, g - 1)
    t1 = t_quantile(0.95, g - 1)
    return {"mean": mean, "interval": [mean - t2 * se, mean + t2 * se], "lower_one_sided": mean - t1 * se,
            "se": se, "families": g, "items": n}


# ------------------------------------------------------------------ steps and the plan


def decide_step(candidate, predecessor, families, excluded, alpha):
    """One step: `candidate` and `predecessor` map item index -> right (bool) for every item;
    items in `excluded` are left out. Returns the step's aggregates."""
    compared = [i for i in sorted(candidate) if i not in excluded]
    wins = sum(1 for i in compared if candidate[i] and not predecessor[i])
    losses = sum(1 for i in compared if predecessor[i] and not candidate[i])
    p = sign_test_p(wins, losses)
    diffs = [int(candidate[i]) - int(predecessor[i]) for i in compared]
    return {
        "items": len(candidate),
        "excluded": len(candidate) - len(compared),
        "compared": len(compared),
        "items_right": sum(1 for i in compared if candidate[i]),
        "predecessor_items_right": sum(1 for i in compared if predecessor[i]),
        "unexcluded_items_right": sum(1 for i in candidate if candidate[i]),
        "wins": wins,
        "losses": losses,
        "p_one_sided": p,
        "step_holds": wins > losses and p < alpha,
        "paired_gain": clustered_gain(diffs, [families[i] for i in compared]),
    }


def decide_plan(steps, plan):
    """The per-lineage rule over the step results `steps[(series, step)]`: a lineage succeeds when
    every registered step holds; the plan is met when at least `series_required` lineages succeed.
    A missing or invalid step is not a step that holds."""
    series = sorted({s for s, _ in steps})
    succeeded = [s for s in series
                 if all(steps.get((s, k), {}).get("step_holds") and steps.get((s, k), {}).get("valid", True)
                        for k in plan["steps"])]
    pooled = {}
    for k in plan["steps"]:
        w = sum(steps[(s, k)]["wins"] for s in series if (s, k) in steps)
        lo = sum(steps[(s, k)]["losses"] for s in series if (s, k) in steps)
        pooled[str(k)] = {"wins": w, "losses": lo, "series": sum(1 for s in series if (s, k) in steps),
                          "p_one_sided_descriptive": sign_test_p(w, lo)}
    return {"series": series, "succeeded": succeeded, "failed": [s for s in series if s not in succeeded],
            "series_required": plan["series_required"], "series_expected": plan.get("series_expected"),
            "plan_met": len(succeeded) >= plan["series_required"], "pooled": pooled}


def load_plan(path):
    rule = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("alpha", "steps", "series_required"):
        if key not in rule:
            raise ValueError(f"{path}: the plan names no {key}")
    return rule


# ------------------------------------------------------------------ inputs


def load_exam(path):
    """The exam's items: [{"instruction", "reference", "family"}], with each item's program."""
    items = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            for key in ("instruction", "reference", "family"):
                if not isinstance(item.get(key), str):
                    raise ValueError(f"{path}:{n}: item has no string {key}")
            item["program"] = item_program(item["instruction"])
            items.append(item)
    if not items:
        raise ValueError(f"{path}: no items")
    return items


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def load_candidates(evidence, series_dir):
    """The hand-off records under EVIDENCE/SERIES/handoff/candidates/<lineage>/<step>/: each
    candidate's record and every program its hand-off list names (`listed`: the training programs
    and the held-out dev and retention programs), and the training programs alone (`trained`)."""
    root = Path(evidence) / series_dir / "handoff" / "candidates"
    if not root.is_dir():
        raise FileNotFoundError(f"{root}: no hand-off records")
    out = {}
    for record in sorted(root.glob("*/*/candidate.json")):
        cand = json.loads(record.read_text(encoding="utf-8"))
        listed, trained = [], []
        with open(record.parent / "training-tasks.jsonl", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    task = json.loads(line)
                    listed.append(task["program"])
                    if task.get("role") == "train":
                        trained.append(task["program"])
        adapter = Path(evidence) / cand["adapter"]
        out[(cand["series"], int(cand["step"]))] = {"record": cand, "adapter": adapter, "listed": listed,
                                                    "trained": trained}
    if not out:
        raise FileNotFoundError(f"{root}: no candidate.json")
    return out


# ------------------------------------------------------------------ checkpoint identity


def checkpoint_digests(directory):
    """The content digests a checkpoint directory is named by: brain's rule (config.json, weights and
    index, `name=sha256:<hex>` lines in name order) and the learner's rule (the `*.safetensors` files,
    `name sha256:<hex>` lines in name order). A declared base is the served checkpoint when it equals
    either."""
    directory = Path(directory)
    files = sorted(p for p in directory.iterdir() if p.is_file())
    digest = {p.name: "sha256:" + file_sha256(p) for p in files
              if p.name == "config.json" or p.name.endswith((".safetensors", ".bin", ".index.json"))}
    brain = "".join(f"{n}={digest[n]}\n" for n in sorted(digest))
    learner = "".join(f"{n} {digest[n]}\n" for n in sorted(digest) if n.endswith(".safetensors"))
    return {"brain": "sha256:" + hashlib.sha256(brain.encode()).hexdigest(),
            "learner": "sha256:" + hashlib.sha256(learner.encode()).hexdigest()}


# ------------------------------------------------------------------ serving


class Server:
    """One `brain serve` on loopback with the paper's flags, offering the adapters of one directory
    by digest. Stopped on exit from the `with` block, whatever happened inside it."""

    def __init__(self, brain, checkpoint, adapters, work, device=None, ready_timeout_s=3600):
        self.brain, self.checkpoint, self.adapters = Path(brain), Path(checkpoint), Path(adapters)
        self.work, self.device, self.ready_timeout_s = Path(work), device, ready_timeout_s
        self.proc = None

    def command(self, port):
        cmd = [str(self.brain)]
        if self.device:
            cmd += ["--device", self.device]
        return cmd + ["serve", "--openai", f"127.0.0.1:{port}", "--max-batch", str(MAX_BATCH),
                      "--kv-cache", KV_CACHE, "--serve-adapters", str(self.adapters), "--adapter-targets", "all",
                      "--checkpoint", str(self.checkpoint), "--models-dir", str(self.work / "models"),
                      "--runtime-dir", str(self.work), "--api-keys-out", str(self.work / "keys.json"),
                      "--ready-file", str(self.work / "ready")]

    def __enter__(self):
        self.work.mkdir(parents=True, exist_ok=True)
        (self.work / "models").mkdir(exist_ok=True)
        for stale in ("ready", "keys.json"):
            (self.work / stale).unlink(missing_ok=True)
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}"
        self.log = open(self.work / "serve.log", "ab")
        self.proc = subprocess.Popen(self.command(port), stdin=subprocess.DEVNULL, stdout=self.log,
                                     stderr=subprocess.STDOUT)
        deadline = time.monotonic() + self.ready_timeout_s
        # brain creates the ready file once every listener accepts; it never appears when a surface
        # fails, so the wait is bounded and also ends when the process does.
        while not (self.work / "ready").exists():
            if self.proc.poll() is not None:
                raise RuntimeError(f"brain serve exited with {self.proc.returncode}; see {self.work / 'serve.log'}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"brain serve not ready after {self.ready_timeout_s} s")
            time.sleep(1)
        self.key = json.loads((self.work / "keys.json").read_text())["openai"]
        return self

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.log.close()
        return False

    def _call(self, method, path, body=None, timeout=900):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base_url + path, data=data, method=method,
                                     headers={"Authorization": f"Bearer {self.key}",
                                              "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    def models(self):
        return self._call("GET", "/v1/models")["data"]

    def complete(self, model, content):
        """One greedy completion, thinking off, at most MAX_TOKENS tokens. Connection failures and
        HTTP 429/5xx are retried three times with backoff (a greedy request is idempotent)."""
        body = {"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0.0,
                "max_tokens": MAX_TOKENS, "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
        for attempt in range(4):
            try:
                out = self._call("POST", "/v1/chat/completions", body)
                choice = out["choices"][0]
                return choice["message"].get("content") or "", choice.get("finish_reason")
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503, 504) or attempt == 3:
                    raise
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                if attempt == 3:
                    raise
            time.sleep(5 * 2 ** attempt)
        raise AssertionError("unreachable")


def served_ids(cards):
    """The served base's id (the one card that is not an adapter) and the adapters' ids by digest."""
    bases = [c["id"] for c in cards if not c.get("adapter")]
    if len(bases) != 1:
        raise RuntimeError(f"expected one served base, the server lists {bases}")
    return bases[0], {c["adapter"]["digest"]: c["id"] for c in cards if c.get("adapter")}


def grade_model(server, model, exam, out_path):
    """Asks every item once and writes one line per item: index, right, finish reason, reply."""
    def ask(i):
        try:
            reply, finish = server.complete(model, exam[i]["instruction"] + REQUEST_SUFFIX)
            return {"index": i, "right": grade_reply(reply, exam[i]["reference"]), "finish_reason": finish,
                    "reply": reply}
        except Exception as e:  # recorded as a failed request, never as a wrong answer
            return {"index": i, "right": None, "error": f"{type(e).__name__}: {e}"}

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_BATCH) as pool:
        rows = list(pool.map(ask, range(len(exam))))
    tmp = out_path.with_suffix(".part")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(out_path)
    return rows


def grade(args):
    exam = load_exam(args.exam)
    candidates = load_candidates(args.evidence, args.series)
    ck = checkpoint_digests(args.checkpoint)
    out = Path(args.out)
    (out / "grades").mkdir(parents=True, exist_ok=True)
    integrity = {}
    for lineage in sorted({s for s, _ in candidates}):
        steps = sorted(k for s, k in candidates if s == lineage)
        adapter_dir = {candidates[(lineage, k)]["adapter"].parent.parent for k in steps}
        if len(adapter_dir) != 1:
            raise ValueError(f"{lineage}: its adapters are not in one directory: {adapter_dir}")
        flags = []
        for k in steps:
            c = candidates[(lineage, k)]
            declared = c["record"]["adapter_digest"]
            if "sha256:" + file_sha256(c["adapter"]) != declared:
                flags.append(f"step {k}: adapter file does not have the declared digest {declared}")
            if c["record"]["base_digest"] not in ck.values():
                flags.append(f"step {k}: declared base {c['record']['base_digest']} is not the served checkpoint")
        work = out / "serve" / lineage
        with Server(args.brain, args.checkpoint, adapter_dir.pop(), work, args.device) as server:
            base_id, by_digest = served_ids(server.models())
            models = {"base": base_id}
            for k in steps:
                digest = candidates[(lineage, k)]["record"]["adapter_digest"]
                if digest not in by_digest:
                    flags.append(f"step {k}: the server offers no adapter with digest {digest}")
                else:
                    models[str(k)] = by_digest[digest]
            for name, model in models.items():
                path = out / "grades" / f"{lineage}-{name}.jsonl"
                rows = grade_model(server, model, exam, path)
                failed = sum(1 for r in rows if r["right"] is None)
                print(f"{lineage} {name}: {sum(1 for r in rows if r['right'])} of {len(rows)} right, "
                      f"{failed} failed requests", flush=True)
        integrity[lineage] = flags
    (out / "integrity.json").write_text(json.dumps({"checkpoint": ck, "flags": integrity}, indent=1) + "\n")


def read_grades(path):
    grades, failures, truncated = {}, 0, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["right"] is None:
                failures += 1
                grades[row["index"]] = False
            else:
                grades[row["index"]] = bool(row["right"])
                truncated += row.get("finish_reason") == "length"
    return grades, failures, truncated


def analyse(args):
    exam = load_exam(args.exam)
    candidates = load_candidates(args.evidence, args.series)
    plan = load_plan(args.plan)
    out = Path(args.out)
    grades_dir = out / "grades"
    integrity = json.loads((out / "integrity.json").read_text()) if (out / "integrity.json").exists() else {"flags": {}}
    programs = [it["program"] for it in exam]
    families = [it["family"] for it in exam]
    steps = {}
    for (lineage, k), cand in sorted(candidates.items()):
        pred_name = "base" if k == plan["steps"][0] else str(k - 1)
        cg, cf, ct = read_grades(grades_dir / f"{lineage}-{k}.jsonl")
        pg, pf, pt = read_grades(grades_dir / f"{lineage}-{pred_name}.jsonl")
        listed = list(cand["listed"])
        if pred_name != "base":
            listed += candidates[(lineage, k - 1)]["listed"]
        excluded = excluded_items(programs, listed)
        overlap_trained = len(excluded_items(programs, cand["trained"]))
        step = decide_step(cg, pg, families, excluded, plan["alpha"])
        flags = [f for f in integrity["flags"].get(lineage, []) if f.startswith(f"step {k}:")]
        step.update({"series": lineage, "step": k, "predecessor": pred_name, "failed_requests": cf + pf,
                     "truncated_replies": ct, "integrity_flags": flags,
                     "excluded_training_overlap": overlap_trained,
                     "valid": not flags and cf + pf == 0})
        steps[(lineage, k)] = step
    result = {"protocol": {"request_suffix": REQUEST_SUFFIX, "max_tokens": MAX_TOKENS, "kv_cache": KV_CACHE,
                           "max_batch": MAX_BATCH, "temperature": 0.0, "thinking": "off",
                           "answer": "last complete fenced block, else the whole reply",
                           "truncated": "graded as written"},
              "exam_sha256": file_sha256(args.exam), "items": len(exam),
              "base_agreement": base_agreement(grades_dir, sorted({s for s, _ in candidates})),
              "steps": [steps[k] for k in sorted(steps)], "plan": decide_plan(steps, plan)}
    (out / "aggregate.json").write_text(json.dumps(result, indent=1) + "\n")
    (out / "table.md").write_text(markdown_table(result))
    print((out / "table.md").read_text(), end="")


def base_agreement(grades_dir, lineages):
    """How far the base's grades agree between the lineages' servers: the protocol is deterministic
    when every server grades the base alike."""
    sets = {s: read_grades(grades_dir / f"{s}-base.jsonl")[0] for s in lineages
            if (grades_dir / f"{s}-base.jsonl").exists()}
    if not sets:
        return None
    right = {s: sorted(i for i, r in g.items() if r) for s, g in sets.items()}
    union = set().union(*map(set, right.values()))
    common = set.intersection(*map(set, right.values()))
    return {"right": {s: len(v) for s, v in right.items()}, "right_on_every_server": len(common),
            "right_on_some_server": len(union)}


def _p(x):
    return f"{x:.2g}"


def markdown_table(result):
    lines = ["| Lineage | Step | Right | Predecessor | Excluded | Wins | Losses | p | Holds | Gain [95% CI] | Valid |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in result["steps"]:
        g = s["paired_gain"]
        ci = f"{g['mean']:.4f} [{g['interval'][0]:.4f}, {g['interval'][1]:.4f}]" if g["interval"] else "-"
        lines.append(f"| {s['series']} | {s['step']} | {s['items_right']} | {s['predecessor_items_right']} | "
                     f"{s['excluded']} | {s['wins']} | {s['losses']} | {_p(s['p_one_sided'])} | "
                     f"{'yes' if s['step_holds'] else 'no'} | {ci} | {'yes' if s['valid'] else 'no'} |")
    plan = result["plan"]
    lines.append("")
    lines.append(f"Plan: {len(plan['succeeded'])} of {len(plan['series'])} lineages succeed "
                 f"(required {plan['series_required']}): {'met' if plan['plan_met'] else 'not met'}.")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ comparison with the paper


def compare(args):
    """The reference examiner's steps beside the published verdicts' numbers (data/measurements.json)."""
    result = json.loads(Path(args.aggregate).read_text())
    runs = json.loads(Path(args.measurements).read_text())["runs"]
    metrics = {}
    for run in runs.values():
        for k, v in run["metrics"].items():
            metrics[f"{run['metric_prefix']}.{k}"] = v
    rows = ["| Lineage | Step | Right (paper / reference) | Wins (paper / reference) | Losses (paper / reference) "
            "| Excluded (paper / reference) | Holds (paper / reference) |", "|---|---|---|---|---|---|---|"]
    agree = 0
    for s in result["steps"]:
        pre = f"{args.prefix}.{s['series']}.g{s['step']}"
        paper = {k: metrics.get(f"{pre}.{k}") for k in ("right", "wins", "losses", "excluded", "holds")}
        holds = "yes" if s["step_holds"] else "no"
        agree += paper["holds"] == holds
        rows.append(f"| {s['series']} | {s['step']} | {paper['right']} / {s['items_right']} | {paper['wins']} / "
                    f"{s['wins']} | {paper['losses']} / {s['losses']} | {paper['excluded']} / {s['excluded']} | "
                    f"{paper['holds']} / {holds} |")
    rows.append("")
    rows.append(f"Steps whose decision agrees: {agree} of {len(result['steps'])}. Base items right per lineage "
                f"server: {result['base_agreement']}.")
    text = "\n".join(rows) + "\n"
    Path(args.out).write_text(text)
    print(text, end="")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("grade", "analyse"):
        s = sub.add_parser(name)
        s.add_argument("--exam", required=True, type=Path)
        s.add_argument("--evidence", required=True, type=Path, help="the extracted evidence dataset")
        s.add_argument("--series", required=True, help="the series directory in the dataset, e.g. rsi4")
        s.add_argument("--out", required=True, type=Path)
        if name == "grade":
            s.add_argument("--brain", required=True, type=Path, help="the brain binary")
            s.add_argument("--checkpoint", required=True, type=Path, help="the base checkpoint directory")
            s.add_argument("--device", help="brain's --device, e.g. gpu0")
        else:
            s.add_argument("--plan", required=True, type=Path, help="the registered rule (plan.toml)")
    s = sub.add_parser("compare")
    s.add_argument("--aggregate", required=True, type=Path)
    s.add_argument("--measurements", required=True, type=Path)
    s.add_argument("--prefix", default="rsi4.v", help="the metric prefix of the published step verdicts")
    s.add_argument("--out", required=True, type=Path)
    args = p.parse_args(argv)
    {"grade": grade, "analyse": analyse, "compare": compare}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
