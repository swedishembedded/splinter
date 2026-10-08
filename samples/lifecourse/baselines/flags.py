# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements long-horizon risk prediction from cohort and
# survey data for its clients. If your team needs expertise in building,
# validating and proving time-to-event models on real health records, you
# can procure our services by sending an email to info@swedishembedded.com.
"""Death with a condition listed on the certificate (T2 of the secondary estimands).

The linked mortality file flags deaths on whose certificate diabetes or
hypertension appears anywhere (`causes.jsonl`, written by `lifecourse build`).
The outcome of a flag is death WITH it by year t, competing with death
WITHOUT it. A death with no multiple-cause data is neither: it is censored at
its time, and counted, because calling it flag-free would invent an outcome.

Three predictors of the flag's cumulative incidence are written, each as
`<out>/<name>/r<repeat>-k<fold>.jsonl` with one line per test subject,
`{"subject_id": ..., "flag_cif": {"diabetes": [15 values], "hypertension": [15 values]}}`:

* `flag-cox-<inputs>`: one regularised Cox model for death with the flag and
  one for death without it, combined by the Aalen-Johansen formula;
* `flag-share-<inputs>`: the all-cause incidence of `cs-cox-<inputs>` (read
  from its prediction file of the same fold) times a penalised logistic model
  of P(flag | died, x) fitted on the training decedents. It assumes the share
  of deaths carrying the flag does not depend on when death comes.

The all-cause incidence of a model used as a ranker is the control that says
whether either predicts anything beyond "who dies"; it needs no file here.

    python -I flags.py --data <lifecourse dir> --out <baselines dir> [--model NAME ...] [--jobs N]
"""
import argparse
import json
import os
import sys
import warnings

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features  # noqa: E402
import models  # noqa: E402
import run  # noqa: E402

FLAGS = ("diabetes", "hypertension")


def load_flags(data_dir, ids):
    """Per subject in `ids`: for each flag 1 (listed), 0 (not listed) or -1 (no
    multiple-cause data, or not a decedent)."""
    out = {f: np.full(len(ids), -1, dtype=np.int8) for f in FLAGS}
    row = {str(s): i for i, s in enumerate(ids)}
    with open(os.path.join(data_dir, "causes.jsonl")) as f:
        for line in f:
            c = json.loads(line)
            i = row.get(c["subject_id"])
            if i is None or not c["mcod_available"]:
                continue
            out["diabetes"][i] = int(c["diabetes_mcod"])
            out["hypertension"][i] = int(c["hypertension_mcod"])
    return out


def flag_outcome(time, cause, flag):
    """(time, cause) with cause 0 = death with the flag, 1 = death without it,
    -1 = censored (and a death without multiple-cause data)."""
    out = np.full(len(time), -1, dtype=np.int8)
    died = cause >= 0
    out[died & (flag == 1)] = 0
    out[died & (flag == 0)] = 1
    return time, out


class FlagCox:
    """Death with the flag against death without it, as two cause-specific Cox models."""

    def __init__(self, inputs):
        self.inputs, self.name = inputs, f"flag-cox-{inputs}"

    def fit_predict(self, ctx):
        d = ctx.data
        time, cause = d["time"], d["cause"]
        curves = {}
        for flag in FLAGS:
            t, c = flag_outcome(time, cause, d["flag_" + flag])
            fit, val = ctx.inner()
            prep_inner = models.Preprocessor(d, self.inputs).fit(d, fit)
            x_fit, x_val = prep_inner.transform(d, fit), prep_inner.transform(d, val)
            prep = models.Preprocessor(d, self.inputs).fit(d, ctx.train)
            x_tr, x_te = prep.transform(d, ctx.train), prep.transform(d, ctx.test)
            dh = np.zeros((len(ctx.test), len(models.GRID) - 1, 2))
            for k in (0, 1):
                l1, alphas = models.select_coxnet(
                    x_fit, t[fit], c[fit] == k, x_val, t[val], c[val] == k)
                ctx.chosen[f"{flag}:{k}"] = dict(l1_ratio=l1, alpha=alphas[-1])
                beta = models.fit_coxnet(x_tr, t[ctx.train], c[ctx.train] == k, l1, alphas)
                h0 = np.diff(models.breslow(x_tr @ beta, t[ctx.train], c[ctx.train] == k, models.GRID))
                dh[:, :, k] = np.exp(x_te @ beta)[:, None] * h0[None, :]
            _, per_cause = models.cumulative_incidence(dh, names=("flag", "other"))
            curves[flag] = per_cause["flag"]
        return curves


class FlagShare:
    """All-cause incidence times the probability that a death carries the flag."""

    def __init__(self, inputs, baselines_dir):
        self.inputs, self.name = inputs, f"flag-share-{inputs}"
        self.baselines_dir = baselines_dir

    def all_cause(self, ctx, repeat, fold):
        path = os.path.join(self.baselines_dir, f"cs-cox-{self.inputs}", f"r{repeat}-k{fold}.jsonl")
        by_id = {}
        with open(path) as f:
            for line in f:
                p = json.loads(line)
                by_id[p["subject_id"]] = p["cif"]
        return np.array([by_id[str(s)] for s in ctx.data["ids"][ctx.test]])

    def fit_predict(self, ctx, repeat, fold):
        from sklearn.linear_model import LogisticRegressionCV
        d = ctx.data
        base = self.all_cause(ctx, repeat, fold)
        prep = models.Preprocessor(d, self.inputs).fit(d, ctx.train)
        x_tr, x_te = prep.transform(d, ctx.train), prep.transform(d, ctx.test)
        curves = {}
        for flag in FLAGS:
            y = d["flag_" + flag][ctx.train]
            have = y >= 0
            clf = LogisticRegressionCV(Cs=6, cv=3, max_iter=2000, scoring="neg_log_loss")
            with warnings.catch_warnings():  # scikit-learn announces a default it will change
                warnings.simplefilter("ignore", FutureWarning)
                clf.fit(x_tr[have], y[have])
            share = clf.predict_proba(x_te)[:, 1]
            curves[flag] = base * share[:, None]
        return curves


def write(directory, repeat, fold, ids, curves, meta):
    os.makedirs(directory, exist_ok=True)
    stem = os.path.join(directory, f"r{repeat}-k{fold}")
    for f in FLAGS:
        if not np.all(np.isfinite(curves[f])):
            raise ValueError(f"{stem}: non-finite predictions for {f}")
    tmp = stem + ".jsonl.tmp"
    with open(tmp, "w") as out:
        for i, sid in enumerate(ids):
            line = {"subject_id": str(sid),
                    "flag_cif": {f: [round(float(v), 9) for v in curves[f][i]] for f in FLAGS}}
            out.write(json.dumps(line) + "\n")
    os.replace(tmp, stem + ".jsonl")
    with open(stem + ".meta.json", "w") as out:
        json.dump(meta, out, indent=1, sort_keys=True)


def make(name, baselines_dir):
    kind, inputs = name.rsplit("-", 1)
    if kind == "flag-cox":
        return FlagCox(inputs)
    if kind == "flag-share":
        return FlagShare(inputs, baselines_dir)
    raise ValueError(f"unknown model {name!r}")


def run_one(args):
    data_dir, out_dir, name, repeat, fold = args
    target = os.path.join(out_dir, name, f"r{repeat}-k{fold}.jsonl")
    if os.path.exists(target):
        return f"{name} r{repeat} k{fold}: done already"
    timelines, partition, build = run.paths(data_dir)
    data = features.load(timelines, partition, os.path.join(out_dir, "_features.npz"))
    data.update({"flag_" + f: v for f, v in load_flags(data_dir, data["ids"]).items()})
    repeats, _ = run.load_partition(partition)
    with open(build) as f:
        horizons = json.load(f)["horizons"]
    train, test = run.split(data, repeats, repeat, fold)
    ctx = models.Context(data, train, test, run.fold_seed(repeat, fold), horizons)
    model = make(name, out_dir)
    curves = (model.fit_predict(ctx, repeat, fold) if isinstance(model, FlagShare)
              else model.fit_predict(ctx))
    meta = dict(baseline=name, repeat=repeat, fold=fold, seed=ctx.seed, n_train=len(train),
                n_test=len(test), chosen=ctx.chosen)
    write(os.path.join(out_dir, name), repeat, fold, data["ids"][test], curves, meta)
    return f"{name} r{repeat} k{fold}: done"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--repeat", action="append", type=int)
    ap.add_argument("--fold", action="append", type=int)
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    if not 1 <= a.jobs <= run.MAX_JOBS:
        ap.error(f"--jobs must be between 1 and {run.MAX_JOBS}")
    timelines, partition, _ = run.paths(a.data)
    features.load(timelines, partition, os.path.join(a.out, "_features.npz"))
    repeats, spec = run.load_partition(partition)
    tasks = [(a.data, a.out, n, r, k) for n in a.model
             for r in (a.repeat if a.repeat is not None else range(len(repeats)))
             for k in (a.fold if a.fold is not None else range(spec["folds"]))]
    if a.jobs == 1:
        for line in map(run_one, tasks):
            print(line, flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=a.jobs) as pool:
            for line in pool.map(run_one, tasks):
                print(line, flush=True)


if __name__ == "__main__":
    main()
