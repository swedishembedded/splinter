# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for self-improving engineering
# agents for its clients. If your team needs expertise in closing the loop from
# verified agent work to better local models then you can procure our services
# by sending an email to info@swedishembedded.com.
"""The improvement loop: explore, learn, measure, promote, repeat - one model over
several workloads, on every card the campaign names.

    python3 scripts/campaigns/flywheel/loop.py --config CAMPAIGN.toml [--generations N] [--status]

``campaign.example.toml`` beside this file is a complete campaign file to copy.

A workload is a task collection with a train/eval split (``[[workload]]``): the
tasks the champion practises on and the held-out tasks it is examined on. One
generation, starting from the champion (the base model, or the last promoted
adapter):

1. **serve** the champion with brain on every card (one server per card; an
   agent's trial stays on one server, so its conversation prefix stays cached);
2. **explore** every workload's training tasks and sit every workload's exam,
   side by side; every graded trial is exported. An exam the champion already
   sat under the same stack (agent build, serving build, benchmark commit) is
   not repeated;
3. **publish** the exports and pass rates to the dataset repository (optional);
4. **learn**: every workload's training exports are admitted (held-out tasks
   are refused by Splinter's seal), one dataset is built over all of them, and
   each ``[[train.variant]]`` trains a candidate adapter on its own card, at
   once (``sft-step`` imitates the passing attempts of every generation so far
   from the base; ``policy-gradient`` reinforces this generation's attempts by
   their advantage and continues the champion's adapter);
5. **candidates**: every candidate sits every workload's exam, side by side, on
   its share of the cards (the champion too, when the stack changed since its
   exam);
6. **decide**: each candidate is paired task by task with the champion per
   workload (the benchmark's paired report); the candidate with the largest
   gain over all paired tasks is promoted when that gain passes
   ``promote.min_mean_delta`` and no workload falls by more than
   ``promote.max_workload_drop``. Every number is recorded.

Every step writes its result to ``<work>/generations/<n>/state.json`` before the
next starts, so the loop resumes where it stopped: a finished step is never
repeated, an unfinished one is redone from its start. Nothing here chooses a
model, a split or a recipe on its own: they are the campaign file's.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import tomllib
import urllib.request
from dataclasses import dataclass
from pathlib import Path


def log(message: str) -> None:
    print(f"[loop {time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def path(value: str) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(value))).resolve()


@dataclass(frozen=True)
class Campaign:
    doc: dict

    def __getitem__(self, key: str):
        return self.doc[key]

    @property
    def work(self) -> Path:
        return path(self["paths"]["work"])

    @property
    def workloads(self) -> list[dict]:
        names = [w["name"] for w in self.doc.get("workload", [])]
        if not names or len(set(names)) != len(names):
            raise SystemExit("loop: the campaign needs [[workload]] entries with distinct names")
        return self.doc["workload"]

    @property
    def cards(self) -> list[int]:
        cards = self["serve"]["cards"]
        if not cards or len(set(cards)) != len(cards):
            raise SystemExit("loop: serve.cards must list distinct cards")
        return list(cards)


# --------------------------------------------------------------------------- state


class Generation:
    """One generation's directory and its resumable state."""

    _lock = threading.Lock()

    def __init__(self, campaign: Campaign, number: int):
        self.number = number
        self.dir = campaign.work / "generations" / f"{number:03d}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file = self.dir / "state.json"
        self.state = json.loads(self.file.read_text()) if self.file.is_file() else {"generation": number}

    def done(self, step: str) -> bool:
        return step in self.state.get("done", [])

    def record(self, step: str, **values) -> None:
        with self._lock:
            for key, value in values.items():
                if isinstance(value, dict) and isinstance(self.state.get(key), dict):
                    self.state[key] = {**self.state[key], **value}
                else:
                    self.state[key] = value
            self.state.setdefault("done", []).append(step)
            tmp = self.file.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state, indent=2, sort_keys=True) + "\n")
            tmp.replace(self.file)
        log(f"generation {self.number}: {step} done")


def champion(campaign: Campaign) -> dict:
    """The model the next generation starts from, with the exam jobs (per workload)
    it was last measured by: the last promoted adapter, else the base."""
    best = {"name": "base", "adapter": None, "exams": {}, "exam_stack": None}
    for gen_dir in sorted((campaign.work / "generations").glob("[0-9][0-9][0-9]")):
        state = json.loads((gen_dir / "state.json").read_text()) if (gen_dir / "state.json").is_file() else {}
        if "champion_in" in state:
            best = {**state["champion_in"], "exams": state.get("exams") or state["champion_in"].get("exams", {}),
                    "exam_stack": state.get("exam_stack") or state["champion_in"].get("exam_stack")}
        decision = state.get("decision", {})
        if decision.get("promoted"):
            best = state["candidates"][decision["chosen"]]
    return best


# --------------------------------------------------------------------------- processes


def run(argv: list, log_file: Path, env: dict | None = None, cwd: Path | None = None) -> str:
    """Run a command to its end, its stderr appended to `log_file`; its stdout."""
    with open(log_file, "a") as err:
        done = subprocess.run([str(a) for a in argv], stdout=subprocess.PIPE, stderr=err, text=True,
                              env={**os.environ, **(env or {})}, cwd=cwd)
    if done.returncode != 0:
        raise SystemExit(f"loop: {argv[0]} {' '.join(str(a) for a in argv[1:3])} failed ({done.returncode}); "
                         f"see {log_file}")
    return done.stdout


def with_env_script(campaign: Campaign, argv: list) -> list[str]:
    """`argv` run under the campaign's environment script (CUDA paths), when it names one."""
    script = campaign["serve"].get("env_script")
    if not script:
        return [str(a) for a in argv]
    quoted = " ".join("'" + str(a).replace("'", "'\\''") + "'" for a in argv)
    return ["bash", "-c", f"source '{path(script)}' && exec {quoted}"]


def concurrently(steps) -> None:
    """Run `steps` side by side and wait for all; the first failure is raised once
    every step has finished, so none is cut off."""
    failures: list[BaseException] = []

    def guarded(step):
        try:
            step()
        except BaseException as error:  # noqa: BLE001 - re-raised below, after all finish
            failures.append(error)

    threads = [threading.Thread(target=guarded, args=(step,)) for step in steps]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise failures[0]


# --------------------------------------------------------------------------- serving


def port(campaign: Campaign, card: int) -> int:
    return campaign["serve"]["base_port"] + card


def stop_serving(campaign: Campaign, card: int) -> None:
    """Stop any brain binary serving `card`, whichever path started it."""
    pattern = f"/brain[^ /]* --device gpu{card} "
    for pid in subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split():
        os.kill(int(pid), signal.SIGTERM)
    deadline = time.time() + 120
    while subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0:
        if time.time() > deadline:
            raise SystemExit(f"loop: the brain on gpu{card} did not stop")
        time.sleep(1)
    (campaign.work / f"ready-gpu{card}").unlink(missing_ok=True)


def served_ids(campaign: Campaign, card: int) -> list[str]:
    key = path(campaign["serve"]["key_file"]).read_text().strip()
    request = urllib.request.Request(f"http://127.0.0.1:{port(campaign, card)}/v1/models",
                                     headers={"Authorization": f"Bearer {key}"})
    return [m["id"] for m in json.load(urllib.request.urlopen(request, timeout=30))["data"]]


def serve(campaign: Campaign, model: dict, cards: list[int]) -> str:
    """Serve `model` (base, or base + adapter) on each of `cards`; the model id to request."""
    s = campaign["serve"]
    for card in cards:
        stop_serving(campaign, card)
        ready = campaign.work / f"ready-gpu{card}"
        argv = [path(s["brain"]), "--device", f"gpu{card}", "--backend", "cuda", "serve", "--max-batch",
                s["max_batch"], "--speculative", "off", "--openai", f"127.0.0.1:{port(campaign, card)}",
                "--api-key-file", path(s["key_file"]), "--checkpoint", path(s["checkpoint"]), "--ready-file", ready]
        if model.get("adapter"):
            argv += ["--adapter", model["adapter"], "--adapter-targets", "all"]
        with open(campaign.work / f"brain-gpu{card}.log", "a") as out:
            subprocess.Popen(with_env_script(campaign, argv), stdout=out, stderr=out, stdin=subprocess.DEVNULL,
                             start_new_session=True, env={**os.environ, **s.get("env", {})})
    ids = set()
    for card in cards:
        ready, deadline = campaign.work / f"ready-gpu{card}", time.time() + 900
        while not ready.exists():
            if time.time() > deadline:
                raise SystemExit(f"loop: brain on gpu{card} not ready after 900 s")
            time.sleep(2)
        listed = served_ids(campaign, card)
        if model.get("adapter"):
            found = [i for i in listed if ":splinter:candidate:" in i]
            if len(found) != 1:
                raise SystemExit(f"loop: expected one adapter served on gpu{card}, found {found}")
            ids.add(found[0])
        elif s["base_model"] in listed:
            ids.add(s["base_model"])
        else:
            raise SystemExit(f"loop: {s['base_model']} is not served on gpu{card} (served: {listed})")
    if len(ids) != 1:
        raise SystemExit(f"loop: the cards serve different model ids: {sorted(ids)}")
    return ids.pop()


# --------------------------------------------------------------------------- benchmark


def profile_text(campaign: Campaign, workload: dict, model_id: str, attempts: int, cards: list[int]) -> str:
    """A run profile: every harness role of the workload on `model_id`, served on `cards`."""
    b = campaign["bench"]
    roles = "\n".join(f'{role} = "brain"' for role in ["main", *workload.get("roles", [])])
    urls = ", ".join(f'"http://127.0.0.1:{port(campaign, c)}/v1"' for c in cards)
    tools = "".join(f'{name} = "{path(tool)}"\n' for name, tool in workload.get("tools", {}).items())
    return (f'[sven]\nbuild = "{path(b["sven"])}"\n\n[run]\nenvironment = "userns"\nattempts = {attempts}\n'
            f'concurrency = {workload["concurrency"]}\n\n[roles]\n{roles}\n\n[upstreams.brain]\n'
            f'base_url = [{urls}]\nmodel = "{model_id}"\napi_key_env = "BRAIN_KEY"\n'
            + (f"\n[tools]\n{tools}" if tools else ""))


def bench(campaign: Campaign, *args, log_file: Path) -> str:
    return run([path(campaign["bench"]["dir"]) / "bench", *args], log_file, cwd=path(campaign["bench"]["dir"]))


def bench_run(campaign: Campaign, gen: Generation, workload: dict, model_id: str, attempts: int, split: str,
              name: str, cards: list[int]) -> str:
    """Run the workload's `split` tasks; the job directory."""
    prof = gen.dir / f"{name}.toml"
    prof.write_text(profile_text(campaign, workload, model_id, attempts, cards))
    jobs = campaign.work / "jobs" / f"{gen.number:03d}-{name}"
    jobs.mkdir(parents=True, exist_ok=True)
    before = set(jobs.iterdir())
    tmp = campaign.work / "tmp"
    tmp.mkdir(exist_ok=True)
    env = {"BRAIN_KEY": path(campaign["serve"]["key_file"]).read_text().strip(), "TMPDIR": str(tmp)}
    run([path(campaign["bench"]["dir"]) / "bench", "run", workload["collection"], "--profile", prof, "--set", split,
         "--jobs-dir", jobs], gen.dir / f"{name}.log", env=env, cwd=path(campaign["bench"]["dir"]))
    new = sorted(set(jobs.iterdir()) - before)
    if len(new) != 1:
        raise SystemExit(f"loop: expected one new job directory, found {new}")
    return str(new[0])


def bench_export(campaign: Campaign, gen: Generation, job: str, name: str) -> str:
    out = campaign.work / "exports" / f"{gen.number:03d}-{name}"
    bench(campaign, "export", job, "--out", out, log_file=gen.dir / f"{name}-export.log")
    return str(out)


def exam_problems(summary: dict) -> list[str]:
    """Why an exam says nothing about the model that sat it: trials the model
    server, the environment or the harness ended are not the model's failures."""
    found = []
    for key, what in (("model_server_errors", "ended by the model server"),
                      ("environment_errors", "ended by the environment"),
                      ("errors", "that raised an error")):
        if summary.get(key):
            found.append(f"{summary[key]} trial(s) {what}")
    return found


def stack(campaign: Campaign) -> dict:
    """What a measurement depends on besides the model; two exams compare only under one stack."""

    def digest(file: Path) -> str:
        return hashlib.sha256(file.read_bytes()).hexdigest()

    bench_commit = subprocess.run(["git", "-C", str(path(campaign["bench"]["dir"])), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, check=True).stdout.strip()
    return {"sven": digest(path(campaign["bench"]["sven"])), "brain": digest(path(campaign["serve"]["brain"])),
            "bench": bench_commit}


def adopted(job: str) -> str:
    """A job run before the loop, by its full path; refused until every trial of it
    records `finished_at`, so no generation learns from half a run."""
    full = path(job)
    results = [json.loads(r.read_text()) for r in [*full.glob("*/result.json"), *full.glob("*/*/result.json")]]
    if not results or not all(r.get("finished_at") for r in results):
        raise SystemExit(f"loop: the adopted job {full} has not finished")
    return str(full)


# --------------------------------------------------------------------------- learning

OBJECTIVES = ("sft-step", "policy-gradient")


def objective(train: dict) -> str:
    name = train.get("objective", "sft-step")
    if name not in OBJECTIVES:
        raise SystemExit(f"loop: train.objective {name!r} is not one of {', '.join(OBJECTIVES)}")
    return name


def variants(campaign: Campaign) -> list[dict]:
    """The training recipes of a generation: `[[train.variant]]` entries over the
    `[train]` defaults, each a candidate of its own; one when none is named."""
    base = {k: v for k, v in campaign["train"].items() if k != "variant"}
    named = campaign["train"].get("variant") or [{"name": "a"}]
    out = [{**base, **v} for v in named]
    if len({v["name"] for v in out}) != len(out):
        raise SystemExit("loop: [[train.variant]] names must be distinct")
    return out


def learning_exports(train: dict, states: list[dict], workload: str) -> list[str]:
    """A workload's exploration exports a candidate learns from, oldest first:
    imitation takes every generation's, the policy gradient only the newest."""
    exports = [s["explore_exports"][workload] for s in states if s.get("explore_exports", {}).get(workload)]
    return exports[-1:] if objective(train) == "policy-gradient" else exports


def dataset_args(train: dict) -> list[str]:
    view = objective(train)
    args = ["--view", view]
    if view == "policy-gradient":
        reward = train["reward"]
        for name, weight in reward.get("metrics", {}).items():
            args += ["--reward-metric", f"{name}={weight}"]
        args += ["--fail-ceiling", str(reward["fail_ceiling"]), "--advantage", reward["advantage"]]
    if train.get("images"):
        args.append("--images")
    if train.get("roles"):
        args += ["--roles", ",".join(train["roles"])]
    if train.get("only_actions"):
        args += ["--only-actions", ",".join(train["only_actions"])]
    block = train.get("dataset_max_block", train["max_block"])
    return [*args, "--from", train["base"], "--max-block", str(block), "--split-over-block"]


def train_args(train: dict, start: dict) -> list[str]:
    common = ["--base-precision", train["base_precision"], "--max-block", str(train["max_block"]), "--rank",
              str(train["rank"]), "--alpha", str(train["alpha"]), "--lr", str(train["lr"]), "--steps",
              str(train["steps"]), "--records-per-step", str(train["records_per_step"]), "--seed", str(train["seed"])]
    if objective(train) == "sft-step":
        return ["--from", train["base"], *common, "--eval-every", str(train["eval_every"]), "--patience",
                str(train["patience"]), "--monitor-share", str(train["monitor_share"])]
    pg = train["policy_gradient"]
    origin = f"{train['base']}+{start['adapter']}" if start.get("adapter") else train["base"]
    return ["--from", origin, *common, "--clip-eps", str(pg["clip_eps"]), "--kl-beta", str(pg["kl_beta"]),
            "--reference", pg["reference"], "--loss-agg", pg["loss_agg"], "--temperature", str(pg["temperature"])]


def splinter(campaign: Campaign, gen: Generation, card: int, *args, log_name: str) -> dict:
    sp = campaign["splinter"]
    out = run(with_env_script(campaign, [path(sp["binary"]), "--state", path(sp["state"]), "--models-dir",
                                         path(campaign["serve"]["models_dir"]), "--device", f"gpu{card}",
                                         "--backend", "cuda", "--json", *args]),
              gen.dir / f"{log_name}.log", env=campaign["serve"].get("env", {}))
    return json.loads(out)


def learn(campaign: Campaign, gen: Generation, states: list[dict], start: dict) -> dict:
    """Admit every workload's training exports, build each recipe's dataset and train
    every recipe's candidate at once, one per card."""
    cards = campaign.cards
    recipes = variants(campaign)
    if len(recipes) > len(cards):
        raise SystemExit(f"loop: {len(recipes)} training variants but {len(cards)} cards")
    for card in cards:
        stop_serving(campaign, card)  # a trainer takes its whole card
    ingested: dict[str, str] = gen.state.get("ingested", {})
    for workload in campaign.workloads:
        for export in learning_exports(campaign["train"], states, workload["name"]):
            if export not in ingested:
                report = splinter(campaign, gen, cards[0], "experiences", "ingest-bench", export, log_name="ingest")
                ingested[export] = report.get("train_set") or ""
                gen.record("ingest", ingested=ingested)
    results: dict[str, dict] = {}

    def one(recipe: dict, card: int) -> None:
        sets = sorted({ingested[e] for w in campaign.workloads for e in learning_exports(recipe, states, w["name"])
                       if ingested.get(e)})
        if not sets:
            raise SystemExit("loop: no training experience set was admitted")
        built = splinter(campaign, gen, card, "dataset", "build", *dataset_args(recipe), *sets,
                         log_name=f"dataset-{recipe['name']}")
        trained = splinter(campaign, gen, card, "train", built["dataset"], *train_args(recipe, start),
                           log_name=f"train-{recipe['name']}")
        keep = campaign.work / "adapters" / f"gen{gen.number + 1:03d}-{recipe['name']}.safetensors"
        keep.parent.mkdir(exist_ok=True)
        shutil.copy(Path(trained["adapter"]), keep)
        results[recipe["name"]] = {"name": f"gen{gen.number + 1:03d}-{recipe['name']}", "adapter": str(keep),
                                   "recipe": recipe, "dataset": built["dataset"], "sets": sets,
                                   "train_report": trained}

    concurrently([lambda r=r, c=c: one(r, c) for r, c in zip(recipes, cards)])
    return results


# --------------------------------------------------------------------------- deciding


def paired(report: dict) -> dict:
    """A paired report's summary with its task counts: paired tasks, and the tasks
    the candidate gained (wins) and lost (losses) against the champion."""
    deltas = [r["delta"] for r in report.get("records", [])]
    return {**report["summary"], "paired_tasks": len(deltas), "wins": sum(d > 0 for d in deltas),
            "losses": sum(d < 0 for d in deltas)}


def combine(per_workload: dict[str, dict]) -> dict:
    """The candidate's gain over all paired tasks of every workload, each workload's
    mean weighed by its paired tasks."""
    paired = {w: s.get("paired_tasks") or 0 for w, s in per_workload.items() if s.get("mean_delta") is not None}
    total = sum(paired.values())
    if not total:
        return {"mean_delta": None, "paired_tasks": 0}
    mean = sum(per_workload[w]["mean_delta"] * n for w, n in paired.items()) / total
    return {"mean_delta": mean, "paired_tasks": total}


def decide(per_candidate: dict[str, dict[str, dict]], rule: dict) -> dict:
    """Promote the candidate with the largest combined gain when that gain passes the
    rule and no workload falls by more than the rule allows."""
    scored = {name: combine(per) for name, per in per_candidate.items()}
    ranked = sorted((s["mean_delta"], name) for name, s in scored.items() if s["mean_delta"] is not None)
    if not ranked:
        return {"promoted": False, "chosen": None, "reason": "no paired tasks", "scores": scored}
    best, chosen = ranked[-1]
    floor = -rule.get("max_workload_drop", 1.0)
    drops = {w: s["mean_delta"] for w, s in per_candidate[chosen].items()
             if s.get("mean_delta") is not None and s["mean_delta"] < floor}
    ok = best > rule.get("min_mean_delta", 0.0) and not drops
    reason = (f"{chosen}: combined mean delta {best:+.3f} over {scored[chosen]['paired_tasks']} paired tasks"
              + (f"; workloads below {floor:+.2f}: {drops}" if drops else ""))
    return {"promoted": ok, "chosen": chosen, "reason": reason, "scores": scored, "per_workload": per_candidate}


def share(cards: list[int], parts: int) -> list[list[int]]:
    """`cards` divided into `parts` groups as evenly as possible (a group per model served)."""
    if parts < 1 or parts > len(cards):
        raise SystemExit(f"loop: {parts} models cannot share {len(cards)} cards")
    return [cards[i::parts] for i in range(parts)]


# --------------------------------------------------------------------------- the loop


def generation(campaign: Campaign, number: int, stop_before: str | None = None) -> bool:
    gen = Generation(campaign, number)
    start = gen.state.get("champion_in") or champion(campaign)
    if "champion_in" not in gen.state:
        gen.state["champion_in"] = start
    cards, current, workloads = campaign.cards, stack(campaign), campaign.workloads

    if not gen.done("explore"):
        adopting = campaign.doc.get("adopt", {}).get(str(number), {})
        everything_adopted = all(adopting.get(k, {}).get(w["name"]) for k in ("explore", "exam") for w in workloads)
        model_id = None if everything_adopted else serve(campaign, start, cards)
        explore_jobs, exams = {}, {}
        valid = start.get("exam_stack") == current

        adopt = campaign.doc.get("adopt", {}).get(str(number), {})

        def explore(w: dict) -> None:
            if adopt.get("explore", {}).get(w["name"]):
                explore_jobs[w["name"]] = adopted(adopt["explore"][w["name"]])
                return
            explore_jobs[w["name"]] = bench_run(campaign, gen, w, model_id, w["explore_attempts"], "train",
                                                f"{w['name']}-explore", cards)

        def exam(w: dict) -> None:
            if adopt.get("exam", {}).get(w["name"]):
                exams[w["name"]] = adopted(adopt["exam"][w["name"]])
                return
            exams[w["name"]] = (start["exams"][w["name"]] if valid and start.get("exams", {}).get(w["name"]) else
                                bench_run(campaign, gen, w, model_id, w["exam_attempts"], "eval",
                                          f"{w['name']}-exam", cards))

        concurrently([lambda w=w: explore(w) for w in workloads] + [lambda w=w: exam(w) for w in workloads])
        # An adopted exam ran on a stack this loop did not record: it is never taken as this stack's.
        adopted_exams = any(adopt.get("exam", {}).get(w["name"]) for w in workloads)
        gen.record("explore", explore_jobs=explore_jobs, exams=exams,
                   exam_stack=None if adopted_exams else current,
                   explore_exports={w: bench_export(campaign, gen, j, f"{w}-explore") for w, j in explore_jobs.items()})
    if stop_before == "learn" and not gen.done("learn"):
        log(f"generation {number}: halted before learn (--stop-before learn)")
        return False
    if not gen.done("learn"):
        states = [json.loads((d / "state.json").read_text())
                  for d in sorted((campaign.work / "generations").glob("[0-9][0-9][0-9]"))]
        gen.record("learn", candidates=learn(campaign, gen, states, start))
    if not gen.done("candidates"):
        models = dict(gen.state["candidates"])
        if gen.state.get("exam_stack") != current:
            # The champion's exam ran on another stack: it sits the exam again beside the
            # candidates, so the exams compared always ran on one stack.
            models["champion"] = start
        groups = share(cards, len(models))
        exams: dict[str, dict] = {}

        def sit(name: str, model: dict, group: list[int]) -> None:
            model_id = serve(campaign, model, group)
            jobs: dict[str, str] = {}
            concurrently([lambda w=w: jobs.__setitem__(w["name"], bench_run(
                campaign, gen, w, model_id, w["exam_attempts"], "eval", f"{name}-{w['name']}-exam", group))
                for w in workloads])
            exams[name] = jobs

        concurrently([lambda n=n, m=m, g=g: sit(n, m, g) for (n, m), g in zip(models.items(), groups)])
        again = exams.pop("champion", None)
        gen.record("candidates", candidates={n: {**gen.state["candidates"][n], "exams": e, "exam_stack": current}
                                             for n, e in exams.items()},
                   **({"exams": again, "exam_stack": current} if again else {}))
    if not gen.done("decide"):
        per_candidate: dict[str, dict[str, dict]] = {}
        for name, candidate in gen.state["candidates"].items():
            per_candidate[name] = {}
            for w in workloads:
                theirs, ours = candidate["exams"][w["name"]], gen.state["exams"][w["name"]]
                for who, job in ((name, theirs), ("champion", ours)):
                    summary = json.loads(bench(campaign, "--output", "json", "report", job,
                                               log_file=campaign.work / "report.log"))["summary"]
                    problems = exam_problems(summary)
                    if problems:
                        raise SystemExit(f"loop: {who}'s exam {job} decides nothing: " + "; ".join(problems))
                per_candidate[name][w["name"]] = paired(json.loads(bench(
                    campaign, "--output", "json", "report", theirs, "--against", ours,
                    log_file=campaign.work / "report.log")))
        gen.record("decide", decision=decide(per_candidate, campaign["promote"]))
    decision = gen.state["decision"]
    log(f"generation {number}: {decision['reason']} -> {'promoted' if decision['promoted'] else 'champion kept'}")
    return True


def status(campaign: Campaign) -> None:
    for gen_dir in sorted((campaign.work / "generations").glob("[0-9][0-9][0-9]")):
        state = json.loads((gen_dir / "state.json").read_text())
        decision = state.get("decision", {})
        print(f"{gen_dir.name}: done={state.get('done', [])} "
              f"decision={decision.get('reason', '-')} promoted={decision.get('promoted', '-')}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--generations", type=int, default=1, help="how many generations to run from where it stands")
    ap.add_argument("--status", action="store_true", help="print each generation's state and stop")
    ap.add_argument("--stop-before", choices=["learn"], help="halt a generation before this step")
    args = ap.parse_args()
    campaign = Campaign(tomllib.loads(args.config.read_text()))
    campaign.work.mkdir(parents=True, exist_ok=True)
    if args.status:
        status(campaign)
        return 0
    first = 0
    for gen_dir in sorted((campaign.work / "generations").glob("[0-9][0-9][0-9]")):
        state = json.loads((gen_dir / "state.json").read_text())
        first = int(gen_dir.name) + (1 if "decide" in state.get("done", []) else 0)
    for number in range(first, first + args.generations):
        if not generation(campaign, number, args.stop_before):
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
