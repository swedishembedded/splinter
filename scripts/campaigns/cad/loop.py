# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for self-improving engineering
# agents for its clients. If your team needs expertise in closing the loop from
# verified agent work to better local models then you can procure our services
# by sending an email to info@swedishembedded.com.
"""The CAD improvement loop: explore, learn, measure, promote, repeat.

    python3 scripts/campaigns/cad/loop.py --config CAMPAIGN.toml [--generations N] [--status]

``campaign.example.toml`` beside this file is a complete campaign file to copy.

One generation, starting from the champion (the base model, or the last
promoted adapter):

1. **serve** the champion with brain on the campaign's card;
2. **explore**: the benchmark runs every training-family task with the champion
   (the harness's every role on the served model); every graded trial is
   exported;
3. **measure** the champion on the held-out families (the exam), unless the
   generation it was promoted in already did;
4. **publish** both exports and the pass rates per generation to the dataset
   repository (and its remote, when configured);
5. **learn** (``train.objective``): the training exports are admitted (one
   experience per harness conversation; held-out tasks are refused by
   Splinter's seal), then either
   * ``sft-step`` (imitation): the passing conversations of the trained roles
     of every generation so far train a fresh LoRA adapter on the base; or
   * ``policy-gradient`` (reinforcement learning): every graded attempt of
     this generation (``explore_attempts`` >= 2 per task form a group) earns a
     reward from its verdict (a pass 1, a failure a share of ``fail_ceiling``
     by the weighted ``reward.metrics``), every action of it is weighed by the
     attempt's advantage over its group, and brain's clipped policy gradient
     trains the champion's adapter on, with a KL anchor;
6. **candidate**: the adapter is served and measured on the same exam.
   If the stack (agent build, serving build, benchmark commit) changed since
   the champion's exam, the champion sits it again (beside the candidate, on
   the other card, when there are two): the two exams that are compared
   always ran on one stack;
7. **decide**: the candidate replaces the champion only when its paired
   per-task gain on the exam is positive (the benchmark's paired report: mean
   delta and its 95% interval); the decision and its numbers are recorded.

Every step writes its result to ``<work>/generations/<n>/state.json`` before the
next starts, so the loop resumes where it stopped (a lease that ended, a
crash): a finished step is never repeated, an unfinished one is redone from its
start. Nothing here chooses a model, a split or a recipe on its own: they are
the campaign file's.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import threading
import tomllib
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


# --------------------------------------------------------------------------- state


class Generation:
    """One generation's directory and its resumable state."""

    def __init__(self, campaign: Campaign, number: int):
        self.number = number
        self.dir = campaign.work / "generations" / f"{number:03d}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file = self.dir / "state.json"
        self.state = json.loads(self.file.read_text()) if self.file.is_file() else {"generation": number}

    def done(self, step: str) -> bool:
        return step in self.state.get("done", [])

    _lock = threading.Lock()

    def record(self, step: str, **values) -> None:
        with self._lock:
            self._record(step, **values)

    def _record(self, step: str, **values) -> None:
        self.state.update(values)
        self.state.setdefault("done", []).append(step)
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, sort_keys=True) + "\n")
        tmp.replace(self.file)
        log(f"generation {self.number}: {step} done")


def champion(campaign: Campaign) -> dict:
    """The model the next generation starts from, with the exam job it was last
    measured by: the last promoted adapter, else the base."""
    best = {"name": "base", "adapter": None, "exam_job": None}
    for gen_dir in sorted((campaign.work / "generations").glob("[0-9][0-9][0-9]")):
        state = json.loads((gen_dir / "state.json").read_text()) if (gen_dir / "state.json").is_file() else {}
        if "champion_in" in state:
            best = {**state["champion_in"], "exam_job": state.get("exam_job") or state["champion_in"].get("exam_job"),
                    "exam_stack": state.get("exam_stack") or state["champion_in"].get("exam_stack")}
        if state.get("decision", {}).get("promoted"):
            best = state["candidate"]
    return best


# --------------------------------------------------------------------------- steps


def run(argv: list[str], log_file: Path, env: dict | None = None, timeout: float | None = None,
        cwd: Path | None = None) -> str:
    """Run `argv`, its stderr appended to `log_file`; its stdout. A failure stops the loop."""
    log(" ".join(str(a) for a in argv))
    with open(log_file, "a") as err:
        done = subprocess.run([str(a) for a in argv], stdout=subprocess.PIPE, stderr=err, text=True,
                              env={**os.environ, **(env or {})}, timeout=timeout, check=False, cwd=cwd)
    if done.returncode != 0:
        raise SystemExit(f"loop: {argv[0]} exited {done.returncode}; see {log_file}")
    return done.stdout


def with_env_script(campaign: Campaign, argv: list) -> list[str]:
    """`argv` run after sourcing the campaign's `serve.env_script` (the machine's
    CUDA libraries and the like), when it names one."""
    script = campaign["serve"].get("env_script")
    argv = [str(a) for a in argv]
    if not script:
        return argv
    return ["bash", "-c", 'source "$0" && exec "$@"', str(path(script)), *argv]


def cards(campaign: Campaign) -> tuple[tuple[int, int], tuple[int, int] | None]:
    """The (card, port) everything runs on, and the (card, port) the exams run on
    beside it when the campaign names a second card."""
    s = campaign["serve"]
    exam = (s["exam_card"], s["exam_port"]) if "exam_card" in s else None
    return (s["card"], s["port"]), exam


def stop_serving(campaign: Campaign, card: int) -> None:
    """Stop any brain binary serving `card`, whichever path started it
    (a symlinked and a resolved directory are the same binaries)."""
    pattern = f"/brain[^ /]* --device gpu{card} "
    for pid in subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split():
        os.kill(int(pid), signal.SIGTERM)
    deadline = time.time() + 120
    while subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0:
        if time.time() > deadline:
            raise SystemExit(f"loop: the brain on gpu{card} did not stop")
        time.sleep(1)
    (campaign.work / f"ready-gpu{card}").unlink(missing_ok=True)


def serve(campaign: Campaign, model: dict, gen: Generation, card: int, port: int) -> str:
    """Serve `model` (base, or base + adapter) on `card`; the model id to request."""
    s = campaign["serve"]
    binary = path(s["brain"])
    ready = campaign.work / f"ready-gpu{card}"
    stop_serving(campaign, card)
    argv = [binary, "--device", f"gpu{card}", "--backend", "cuda", "serve", "--max-batch", str(s["max_batch"]),
            "--speculative", "off", "--openai", f"127.0.0.1:{port}", "--api-key-file", path(s["key_file"]),
            "--checkpoint", path(s["checkpoint"]), "--ready-file", ready]
    if model.get("adapter"):
        argv += ["--adapter", model["adapter"], "--adapter-targets", "all"]
    server_log = campaign.work / f"brain-gpu{card}.log"
    env = {**os.environ, **campaign["serve"].get("env", {})}
    with open(server_log, "a") as out:
        subprocess.Popen(with_env_script(campaign, argv), stdout=out, stderr=out, stdin=subprocess.DEVNULL,
                         start_new_session=True, env=env)
    deadline = time.time() + 900
    while not ready.exists():
        if time.time() > deadline:
            raise SystemExit(f"loop: brain on gpu{card} not ready after 900 s; see {server_log}")
        time.sleep(2)
    key = path(s["key_file"]).read_text().strip()
    import urllib.request

    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models", headers={"Authorization": f"Bearer {key}"})
    listed = [m["id"] for m in json.load(urllib.request.urlopen(request, timeout=30))["data"]]
    if model.get("adapter"):
        ids = [i for i in listed if ":splinter:candidate:" in i]
        if len(ids) != 1:
            raise SystemExit(f"loop: expected one adapter model served, found {ids}")
        return ids[0]
    if s["base_model"] not in listed:
        raise SystemExit(f"loop: {s['base_model']} is not served (served: {listed})")
    return s["base_model"]


def profile(campaign: Campaign, gen: Generation, model_id: str, attempts: int, name: str, port: int) -> Path:
    """A run profile with every harness role on `model_id` at `port`."""
    b = campaign["bench"]
    roles = "\n".join(f'{role} = "brain"' for role in ["main", *b["roles"]])
    text = (f'[sven]\nbuild = "{path(b["sven"])}"\n\n[run]\nenvironment = "userns"\nattempts = {attempts}\n'
            f'concurrency = {b["concurrency"]}\n\n[roles]\n{roles}\n\n[upstreams.brain]\n'
            f'base_url = "http://127.0.0.1:{port}/v1"\nmodel = "{model_id}"\napi_key_env = "BRAIN_KEY"\n')
    out = gen.dir / f"{name}.toml"
    out.write_text(text)
    return out


def bench_run(campaign: Campaign, gen: Generation, prof: Path, split: str, name: str) -> str:
    """Run the collection's `split` tasks; the job directory."""
    b = campaign["bench"]
    # Each run gets a directory of its own, so two runs at once never take each other's job.
    jobs = campaign.work / "jobs" / f"{gen.number:03d}-{name}"
    jobs.mkdir(parents=True, exist_ok=True)
    before = set(jobs.iterdir())
    env = {"BRAIN_KEY": path(campaign["serve"]["key_file"]).read_text().strip(),
           "TMPDIR": str(campaign.work / "tmp")}
    (campaign.work / "tmp").mkdir(exist_ok=True)
    run([path(b["dir"]) / "bench", "run", b["collection"], "--profile", prof, "--set", split,
         "--jobs-dir", jobs], gen.dir / f"{name}.log", env=env, cwd=path(b["dir"]))
    new = sorted(set(jobs.iterdir()) - before)
    if len(new) != 1:
        raise SystemExit(f"loop: expected one new job directory, found {new}")
    return str(new[0])


def bench_export(campaign: Campaign, gen: Generation, job: str, name: str) -> str:
    out = campaign.work / "exports" / f"{gen.number:03d}-{name}"
    run([path(campaign["bench"]["dir"]) / "bench", "export", job, "--out", out], gen.dir / f"{name}-export.log",
        cwd=path(campaign["bench"]["dir"]))
    return str(out)


def bench_compare(campaign: Campaign, job: str, against: str) -> dict:
    out = run([path(campaign["bench"]["dir"]) / "bench", "--output", "json", "report", job, "--against", against],
              campaign.work / "report.log", cwd=path(campaign["bench"]["dir"]))
    return json.loads(out)["summary"]


def splinter(campaign: Campaign, gen: Generation, *args, log_name: str = "splinter") -> dict:
    sp = campaign["splinter"]
    out = run(with_env_script(campaign, [path(sp["binary"]), "--state", path(sp["state"]), "--models-dir",
                                         path(campaign["serve"]["models_dir"]), "--device",
                                         f"gpu{campaign['serve']['card']}", "--backend", "cuda", "--json", *args]),
              gen.dir / f"{log_name}.log", env=campaign["serve"].get("env", {}))
    return json.loads(out)


OBJECTIVES = ("sft-step", "policy-gradient")


def objective(campaign: Campaign) -> str:
    """How a generation learns: `sft-step` imitates the trained roles' passing
    actions; `policy-gradient` reinforces every graded attempt by its advantage
    over the other attempts at its task."""
    name = campaign["train"].get("objective", "sft-step")
    if name not in OBJECTIVES:
        raise SystemExit(f"loop: train.objective {name!r} is not one of {', '.join(OBJECTIVES)}")
    return name


def learning_exports(campaign: Campaign, states: list[dict]) -> list[str]:
    """The exploration exports a generation learns from, oldest first. Imitation
    takes every generation's; the policy gradient only the newest, the rollouts
    of the very policy it trains (on-policy: older rollouts came from another one)."""
    exports = [s["explore_export"] for s in states if s.get("explore_export")]
    return exports[-1:] if objective(campaign) == "policy-gradient" else exports


def dataset_args(campaign: Campaign) -> list[str]:
    """`splinter dataset build` arguments: the view and, for the policy
    gradient, how a trial's reward is read from its verdict."""
    t = campaign["train"]
    view = objective(campaign)
    args = ["--view", view]
    if view == "policy-gradient":
        reward = t["reward"]
        for name, weight in reward.get("metrics", {}).items():
            args += ["--reward-metric", f"{name}={weight}"]
        args += ["--fail-ceiling", str(reward["fail_ceiling"]), "--advantage", reward["advantage"]]
    if t.get("images"):
        args.append("--images")
    # Records are measured by the base the training starts from (its tokenizer,
    # chat template and image handling), as the trainer will measure them. The
    # dataset's block can still be set smaller than the training's.
    block = t.get("dataset_max_block", t["max_block"])
    return [*args, "--roles", ",".join(t["roles"]), "--from", t["base"], "--max-block", str(block),
            "--split-over-block"]


def train_args(campaign: Campaign, start: dict) -> list[str]:
    """`splinter train` arguments. Imitation trains a fresh adapter on the base
    with a held-out monitor; the policy gradient continues the champion's
    adapter (the policy its rollouts came from) under brain's clipped objective,
    with no held-out monitor (brain refuses one beside it; the exam judges)."""
    t = campaign["train"]
    common = ["--base-precision", t["base_precision"], "--max-block", str(t["max_block"]), "--rank", str(t["rank"]),
              "--alpha", str(t["alpha"]), "--lr", str(t["lr"]), "--steps", str(t["steps"]), "--records-per-step",
              str(t["records_per_step"]), "--seed", str(t["seed"])]
    if objective(campaign) == "sft-step":
        return ["--from", t["base"], *common, "--eval-every", str(t["eval_every"]), "--patience", str(t["patience"]),
                "--monitor-share", str(t["monitor_share"])]
    pg = t["policy_gradient"]
    origin = f"{t['base']}+{start['adapter']}" if start.get("adapter") else t["base"]
    return ["--from", origin, *common, "--clip-eps", str(pg["clip_eps"]), "--kl-beta", str(pg["kl_beta"]),
            "--reference", pg["reference"], "--loss-agg", pg["loss_agg"], "--temperature", str(pg["temperature"])]


def learn(campaign: Campaign, gen: Generation, train_exports: list[str], start: dict) -> dict:
    """Admit the generation's training exports, build the dataset its objective
    learns from and train the next adapter."""
    set_ids = []
    for export in train_exports:
        report = splinter(campaign, gen, "experiences", "ingest-bench", export, log_name="ingest")
        if report.get("train_set"):
            set_ids.append(report["train_set"])
    if not set_ids:
        raise SystemExit("loop: no training experience set was admitted")
    built = splinter(campaign, gen, "dataset", "build", *dataset_args(campaign), *sorted(set(set_ids)),
                     log_name="dataset")
    dataset = built["dataset"]
    stop_serving(campaign, campaign["serve"]["card"])  # the trainer takes the whole card
    trained = splinter(campaign, gen, "train", dataset, *train_args(campaign, start), log_name="train")
    adapter = Path(trained["adapter"])
    keep = campaign.work / "adapters" / f"gen{gen.number + 1:03d}.safetensors"
    keep.parent.mkdir(exist_ok=True)
    shutil.copy(adapter, keep)
    return {"objective": objective(campaign), "set_ids": sorted(set(set_ids)), "dataset": dataset,
            "dataset_report": built, "adapter": str(keep), "train_report": trained}


def publish(campaign: Campaign) -> None:
    """Copy every export into the dataset repository, refresh the progress table and chart."""
    repo = path(campaign["paths"]["dataset_repo"])
    generations = []
    for gen_dir in sorted((campaign.work / "generations").glob("[0-9][0-9][0-9]")):
        state = json.loads((gen_dir / "state.json").read_text())
        n = gen_dir.name
        for key in ("explore_export", "exam_export", "candidate_exam_export"):
            if state.get(key):
                target = repo / "generations" / n / key.removesuffix("_export")
                if not target.exists():
                    shutil.copytree(state[key], target)
        jobs = [state[k] for k in ("explore_job", "exam_job") if state.get(k)]
        if jobs:
            generations.append(f"g{state['generation']}={','.join(jobs)}")
    if generations:
        # The dataset's own tools need their own packages (the chart: matplotlib):
        # the campaign names the interpreter that has them.
        # Not resolved: a virtual environment's interpreter is a symlink whose
        # target is the bare base interpreter, without the environment's packages.
        named = campaign["paths"].get("python")
        python = os.path.expanduser(os.path.expandvars(named)) if named else sys.executable
        subprocess.run([python, repo / "scripts" / "progress.py",
                        *[a for g in generations for a in ("--generation", g)]], check=True)
    subprocess.run(["git", "-C", repo, "add", "-A"], check=True)
    if subprocess.run(["git", "-C", repo, "diff", "--cached", "--quiet"]).returncode != 0:
        subprocess.run(["git", "-C", repo, "commit", "-q", "-m", "loop: generations and progress"], check=True)
    remote = campaign["paths"].get("dataset_remote")
    if remote:
        subprocess.run([path(campaign["paths"]["hf"]), "upload", remote, repo, "--repo-type", "dataset",
                        "--exclude", ".git/*", "--commit-message", "loop: generations and progress"], check=True)


def decide(summary: dict, rule: dict) -> dict:
    """Promote the candidate only on a positive paired gain on the exam."""
    mean = summary.get("mean_delta")
    low = (summary.get("interval95") or [None, None])[0]
    if mean is None:
        return {"promoted": False, "reason": "no paired tasks", "summary": summary}
    if rule.get("require_interval_above_zero", False):
        ok = low is not None and low > 0
        reason = f"mean delta {mean:+.3f}, lower bound {low}: {'above' if ok else 'not above'} zero"
    else:
        ok = mean > rule.get("min_mean_delta", 0.0)
        reason = f"mean delta {mean:+.3f} {'>' if ok else '<='} {rule.get('min_mean_delta', 0.0)}"
    return {"promoted": ok, "reason": reason, "summary": summary}


def stack(campaign: Campaign) -> dict:
    """What a measurement depends on besides the model: the agent build, the
    serving build and the benchmark's commit. Two exams compare only under one stack."""
    import hashlib

    def digest(file: Path) -> str:
        return hashlib.sha256(file.read_bytes()).hexdigest()

    bench_commit = subprocess.run(["git", "-C", str(path(campaign["bench"]["dir"])), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, check=True).stdout.strip()
    return {"sven": digest(path(campaign["bench"]["sven"])), "brain": digest(path(campaign["serve"]["brain"])),
            "bench": bench_commit}


def adopted(job: str) -> str:
    """A job run before the loop, by its full path; refused until its run records
    `finished_at`, so no generation learns from half a run."""
    full = path(job)
    results = [json.loads(r.read_text()) for r in full.glob("*/result.json")]
    if not results or not all(r.get("finished_at") for r in results):
        raise SystemExit(f"loop: the adopted job {full} has not finished")
    return str(full)


# --------------------------------------------------------------------------- the loop


def concurrently(steps) -> None:
    """Run `steps` side by side (one per card) and wait for all; the first
    failure is raised once every step has finished, so none is cut off."""
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


def generation(campaign: Campaign, number: int, stop_before: str | None = None) -> bool:
    """Run generation `number` from where it stands; False when it halted
    before `stop_before` (the step is left for a later run)."""
    gen = Generation(campaign, number)
    start = gen.state.get("champion_in") or champion(campaign)
    gen.state["champion_in"] = start
    b = campaign["bench"]
    adopt = campaign.doc.get("adopt", {}).get(str(number), {})
    (card, port), exam_card = cards(campaign)
    current = stack(campaign)

    def explore() -> None:
        if gen.done("explore"):
            return
        if adopt.get("explore_job"):
            job = adopted(adopt["explore_job"])
        else:
            model_id = serve(campaign, start, gen, card, port)
            job = bench_run(campaign, gen, profile(campaign, gen, model_id, b["explore_attempts"], "explore", port),
                            "train", "explore")
        gen.record("explore", explore_job=job, explore_export=bench_export(campaign, gen, job, "explore"))

    def exam() -> None:
        if gen.done("exam"):
            return
        job = adopted(adopt["exam_job"]) if adopt.get("exam_job") else None
        if not job and start.get("exam_job") and start.get("exam_stack") == current:
            job = start["exam_job"]  # measured before under this very stack
        if not job:
            c, p = exam_card or (card, port)
            model_id = serve(campaign, start, gen, c, p)
            job = bench_run(campaign, gen, profile(campaign, gen, model_id, b["exam_attempts"], "exam", p), "eval",
                            "exam")
        gen.record("exam", exam_job=job, exam_stack=current, exam_export=bench_export(campaign, gen, job, "exam"))

    if exam_card:
        # Two cards: the champion explores on one while it sits the exam on the other.
        concurrently([explore, exam])
    else:
        explore()
        exam()
    if not gen.done("publish"):
        publish(campaign)
        gen.record("publish")
    if stop_before == "learn" and not gen.done("learn"):
        log(f"generation {number}: halted before learn (--stop-before learn)")
        return False
    if not gen.done("learn"):
        states = [json.loads((d / "state.json").read_text())
                  for d in sorted((campaign.work / "generations").glob("[0-9][0-9][0-9]"))]
        if exam_card:
            stop_serving(campaign, exam_card[0])
        gen.record("learn", learned=learn(campaign, gen, learning_exports(campaign, states), start))
    def exam_again() -> None:
        # The stack changed since the champion sat the exam (an upgraded agent or
        # server build): measure it again, so champion and candidate meet on one stack.
        if gen.done("candidate") or gen.done("exam-again") or gen.state.get("exam_stack") == current:
            return
        c, p = (card, port)  # beside the candidate's exam when there are two cards
        model_id = serve(campaign, start, gen, c, p)
        job = bench_run(campaign, gen, profile(campaign, gen, model_id, b["exam_attempts"], "exam-again", p),
                        "eval", "exam-again")
        gen.record("exam-again", exam_job=job, exam_stack=current,
                   exam_export=bench_export(campaign, gen, job, "exam-again"))

    def candidate_exam() -> None:
        if gen.done("candidate"):
            return
        candidate = {"name": f"gen{number + 1:03d}", "adapter": gen.state["learned"]["adapter"]}
        c, p = exam_card or (card, port)
        model_id = serve(campaign, candidate, gen, c, p)
        job = bench_run(campaign, gen, profile(campaign, gen, model_id, b["exam_attempts"], "candidate", p), "eval",
                        "candidate")
        candidate["exam_job"] = job
        candidate["exam_stack"] = current
        gen.record("candidate", candidate=candidate, candidate_exam_export=bench_export(campaign, gen, job,
                                                                                         "candidate-exam"))

    if exam_card:
        # Two cards: the champion sits the exam again on one while the candidate sits it on the other.
        concurrently([exam_again, candidate_exam])
    else:
        exam_again()
        candidate_exam()
    if not gen.done("decide"):
        summary = bench_compare(campaign, gen.state["candidate"]["exam_job"], gen.state["exam_job"])
        gen.record("decide", decision=decide(summary, campaign["promote"]))
        publish(campaign)
    log(f"generation {number}: {gen.state['decision']['reason']} -> "
        f"{'promoted' if gen.state['decision']['promoted'] else 'champion kept'}")
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
    ap.add_argument("--stop-before", choices=["learn"],
                    help="halt a generation before this step (its exploration and exam data are kept; rerun to go on)")
    args = ap.parse_args()
    campaign = Campaign(tomllib.loads(args.config.read_text()))
    campaign.work.mkdir(parents=True, exist_ok=True)
    if args.status:
        status(campaign)
        return 0
    existing = sorted((campaign.work / "generations").glob("[0-9][0-9][0-9]"))
    first = 0
    for gen_dir in existing:
        state = json.loads((gen_dir / "state.json").read_text())
        first = int(gen_dir.name) + (1 if "decide" in state.get("done", []) else 0)
    for number in range(first, first + args.generations):
        if not generation(campaign, number, args.stop_before):
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
