# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements long-horizon risk prediction from cohort and
# survey data for its clients. If your team needs expertise in turning
# wearable sensor streams into validated risk inputs, you can procure our
# services by sending an email to info@swedishembedded.com.
"""Do accelerometer summaries add to the same model without them? (paper section 4.6)

On the subjects of the 2003-2006 cycles with a valid recording (`accel.jsonl`,
from `accel.py`), the folds of the partition are restricted to them: each
fold's training and test subjects are the fold's own, kept only if they have
a recording. Two arms are fitted and scored on those folds with the same
model, the spline Cox model on every input:

* `accel-base`: the inputs as they are;
* `accel-plus`: the same inputs with the eight summaries appended;
* `accel-seq`: the base with a convolutional residual network on the ten-minute
  grid of the recording (`accel_seq.npz`, from `accel.py --sequences-out`);
* `accel-plus-seq`: the same network on top of the base with the summaries
  (paper section 4.8).

Predictions are written in the baseline format for the subjects kept, to be
scored by `lifecourse external --subset` and compared by `lifecourse compare`.

    python -I accel_experiment.py --data <lifecourse dir> --out <baselines dir> [--jobs N]
"""
import argparse
import json
import os
import sys
import time

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import accel  # noqa: E402
import features  # noqa: E402
import models  # noqa: E402
import run  # noqa: E402

ARMS = ("base", "plus", "seq", "plus-seq")
MODEL = "spline-cox-net-all"


def load_block(path, ids):
    """(matrix [n, len(FEATURES)] with NaN where there is no recording, has-recording mask)."""
    row = {str(s): i for i, s in enumerate(ids)}
    block = np.full((len(ids), len(accel.FEATURES)), np.nan)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            i = row.get(r["subject_id"])
            if i is not None:
                block[i] = [r[k] for k in accel.FEATURES]
    return block, ~np.isnan(block).any(axis=1)


def with_block(data, block):
    """A copy of `data` with the summaries appended as numeric inputs."""
    out = dict(data)
    out["num"] = np.hstack([data["num"], block.astype(data["num"].dtype)])
    out["num_names"] = np.array([str(x) for x in data["num_names"]] + ["accel:" + k for k in accel.FEATURES])
    return out


def load_sequences(path, ids):
    """(array [n, channels, bins], zeros where there is no recording)."""
    z = np.load(path, allow_pickle=False)
    row = {str(s): i for i, s in enumerate(ids)}
    out = np.zeros((len(ids),) + z["x"].shape[1:], dtype=np.float32)
    for sid, x in zip(z["ids"], z["x"]):
        i = row.get(str(sid))
        if i is not None:
            out[i] = x
    return out


def restrict(train, test, has):
    """The fold's own subjects that have a recording."""
    return train[has[train]], test[has[test]]


def run_one(args):
    data_dir, out_dir, arm, repeat, fold = args
    target = os.path.join(out_dir, "accel-" + arm, f"r{repeat}-k{fold}.jsonl")
    if os.path.exists(target):
        return f"accel-{arm} r{repeat} k{fold}: done already"
    timelines, partition, build = run.paths(data_dir)
    data = features.load(timelines, partition, os.path.join(out_dir, "_features.npz"))
    block, has = load_block(os.path.join(data_dir, "accel.jsonl"), data["ids"])
    repeats, _ = run.load_partition(partition)
    with open(build) as f:
        horizons = json.load(f)["horizons"]
    train, test = restrict(*run.split(data, repeats, repeat, fold), has)
    d = with_block(data, block) if arm.startswith("plus") else data
    ctx = models.Context(d, train, test, run.fold_seed(repeat, fold), horizons)
    t0 = time.time()
    if arm.endswith("seq"):
        import residual
        d["seq"] = load_sequences(os.path.join(data_dir, "accel_seq.npz"), data["ids"])
        cif, cause_cif = residual.ResidualSeq().fit_predict(ctx)
    else:
        cif, cause_cif = models.REGISTRY[MODEL]().fit_predict(ctx)
    meta = dict(baseline="accel-" + arm, repeat=repeat, fold=fold, seed=ctx.seed, n_train=len(train),
                n_test=len(test), chosen=ctx.chosen, seconds=round(time.time() - t0, 1))
    run.write_predictions(os.path.join(out_dir, "accel-" + arm), repeat, fold, d["ids"][test], cif, cause_cif, meta)
    return f"accel-{arm} r{repeat} k{fold}: {meta['seconds']}s, {len(train)} train, {len(test)} test"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", action="append", choices=ARMS, help="only these arms (default: all)")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    if not 1 <= a.jobs <= run.MAX_JOBS:
        ap.error(f"--jobs must be between 1 and {run.MAX_JOBS}")
    timelines, partition, _ = run.paths(a.data)
    features.load(timelines, partition, os.path.join(a.out, "_features.npz"))
    repeats, spec = run.load_partition(partition)
    tasks = [(a.data, a.out, arm, r, k) for arm in (a.arm or ARMS) for r in range(len(repeats)) for k in range(spec["folds"])]
    if a.jobs == 1:
        for line in map(run_one, tasks):
            print(line, flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=a.jobs) as pool:
            for line in pool.map(run_one, tasks):
                print(line, flush=True)
    run.manifest(a.data, a.out, ["accel-" + arm for arm in (a.arm or ARMS)])


if __name__ == "__main__":
    main()
