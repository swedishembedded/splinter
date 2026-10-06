# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements long-horizon risk prediction from cohort and
# survey data for its clients. If your team needs expertise in building,
# validating and proving time-to-event models on real health records, you
# can procure our services by sending an email to info@swedishembedded.com.
"""Conventional baselines on the lifecourse cross-validation folds.

For every repeat and fold of `partition.json` (training: the other folds of
that repeat; test: the fold; the locked test excluded everywhere) each
baseline is fitted on the training subjects and writes its out-of-fold
predictions for every test subject:

    <out>/<baseline>/r<repeat>-k<fold>.jsonl       one line per subject:
        {"subject_id": ..., "cif": [15 values], "cause_cif": {code: [15 values]}}
    <out>/<baseline>/r<repeat>-k<fold>.meta.json   hyperparameters chosen on
        validation subjects held out of training, sizes, seconds
    <out>/<baseline>/manifest.json                 digests, versions, seed

`cif[i]` is the predicted all-cause cumulative incidence by year i+1.
`lifecourse external` scores these files with the metrics the arms are
scored with. Usage:

    python -I run.py --data <lifecourse dir> --out <baselines dir> \\
        [--baseline NAME ...] [--repeat R ...] [--fold K ...] [--jobs N]
    python -I run.py --data <dir> --out <dir> --manifest
"""
import argparse
import json
import os
import sys
import time

# One thread per worker: parallelism is across folds, bounded by --jobs.
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features  # noqa: E402
import models  # noqa: E402

SEED = 20261006
MAX_JOBS = 16


def fold_seed(repeat, fold):
    return SEED + 100 * repeat + fold


def paths(data_dir):
    return (os.path.join(data_dir, "timelines.jsonl"), os.path.join(data_dir, "partition.json"),
            os.path.join(data_dir, "build.json"))


def load_partition(path):
    """Folds only: repeats[r][k] is the list of subject ids of fold k."""
    with open(path) as f:
        p = json.load(f)
    return p["repeats"], p["spec"]


def split(data, repeats, repeat, fold):
    """(training rows, test rows) of a fold; both outside the locked test by
    construction of the cached data, and checked to be disjoint and complete."""
    row = {str(s): i for i, s in enumerate(data["ids"])}
    test = np.array(sorted(row[s] for s in repeats[repeat][fold]))
    train = np.array(sorted(row[s] for k, ids in enumerate(repeats[repeat]) if k != fold for s in ids))
    assert not set(train) & set(test), "training and test folds overlap"
    return train, test


def write_predictions(directory, repeat, fold, ids, cif, cause_cif, meta):
    os.makedirs(directory, exist_ok=True)
    stem = os.path.join(directory, f"r{repeat}-k{fold}")
    if not np.all(np.isfinite(cif)):
        raise ValueError(f"{stem}: non-finite predictions")
    tmp = stem + ".jsonl.tmp"
    with open(tmp, "w") as f:
        for i, sid in enumerate(ids):
            line = {"subject_id": str(sid), "cif": [round(float(v), 9) for v in cif[i]]}
            if cause_cif is not None:
                line["cause_cif"] = {c: [round(float(v), 9) for v in m[i]] for c, m in cause_cif.items()}
            f.write(json.dumps(line) + "\n")
    os.replace(tmp, stem + ".jsonl")
    with open(stem + ".meta.json", "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)


def run_one(args):
    """One baseline on one fold; skipped when its predictions exist."""
    data_dir, out_dir, name, repeat, fold = args
    target = os.path.join(out_dir, name, f"r{repeat}-k{fold}.jsonl")
    if os.path.exists(target):
        return f"{name} r{repeat} k{fold}: done already"
    timelines, partition, build = paths(data_dir)
    data = features.load(timelines, partition, os.path.join(out_dir, "_features.npz"))
    repeats, _ = load_partition(partition)
    with open(build) as f:
        horizons = json.load(f)["horizons"]
    train, test = split(data, repeats, repeat, fold)
    ctx = models.Context(data, train, test, fold_seed(repeat, fold), horizons)
    t0 = time.time()
    cif, cause_cif = models.REGISTRY[name]().fit_predict(ctx)
    meta = dict(baseline=name, repeat=repeat, fold=fold, seed=ctx.seed, n_train=len(train),
                n_test=len(test), chosen=ctx.chosen, seconds=round(time.time() - t0, 1))
    write_predictions(os.path.join(out_dir, name), repeat, fold, data["ids"][test], cif, cause_cif, meta)
    return f"{name} r{repeat} k{fold}: {meta['seconds']}s {json.dumps(ctx.chosen, sort_keys=True)}"


def manifest(data_dir, out_dir, names):
    import sklearn
    import sksurv
    timelines, partition, _ = paths(data_dir)
    base = dict(
        timelines_sha256=features.sha256_file(timelines),
        partition_sha256=features.sha256_file(partition),
        seed=SEED, seed_rule="seed + 100 * repeat + fold",
        software=dict(python=sys.version.split()[0], numpy=np.__version__, sklearn=sklearn.__version__,
                      sksurv=sksurv.__version__),
        locked_test="excluded from every training and test set; never scored by this harness",
    )
    for name in names:
        directory = os.path.join(out_dir, name)
        if not os.path.isdir(directory):
            continue
        folds = {}
        for fn in sorted(os.listdir(directory)):
            if fn.endswith(".jsonl"):
                with open(os.path.join(directory, fn[:-6] + ".meta.json")) as f:
                    meta = json.load(f)
                folds[fn[:-6]] = dict(sha256=features.sha256_file(os.path.join(directory, fn)),
                                      chosen=meta["chosen"], n_train=meta["n_train"], n_test=meta["n_test"])
        with open(os.path.join(directory, "manifest.json"), "w") as f:
            json.dump(dict(base, baseline=name, folds=folds), f, indent=1, sort_keys=True)
        print(f"{name}: manifest for {len(folds)} folds")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", action="append", choices=sorted(models.REGISTRY))
    ap.add_argument("--repeat", action="append", type=int)
    ap.add_argument("--fold", action="append", type=int)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--manifest", action="store_true", help="only (re)write the manifests")
    a = ap.parse_args()
    if not 1 <= a.jobs <= MAX_JOBS:
        ap.error(f"--jobs must be between 1 and {MAX_JOBS}")
    names = a.baseline or sorted(models.REGISTRY)
    os.makedirs(a.out, exist_ok=True)
    if a.manifest:
        manifest(a.data, a.out, names)
        return
    timelines, partition, _ = paths(a.data)
    features.load(timelines, partition, os.path.join(a.out, "_features.npz"))  # cache once, before workers
    repeats, spec = load_partition(partition)
    tasks = [(a.data, a.out, n, r, k) for n in names
             for r in (a.repeat if a.repeat is not None else range(len(repeats)))
             for k in (a.fold if a.fold is not None else range(spec["folds"]))]
    if a.jobs == 1:
        results = map(run_one, tasks)
        for line in results:
            print(line, flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=a.jobs) as pool:
            for line in pool.map(run_one, tasks):
                print(line, flush=True)
    manifest(a.data, a.out, names)


if __name__ == "__main__":
    main()
