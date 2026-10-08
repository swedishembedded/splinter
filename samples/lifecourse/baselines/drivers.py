# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements long-horizon risk prediction from cohort and
# survey data for its clients. If your team needs expertise in building,
# validating and proving time-to-event models on real health records, you
# can procure our services by sending an email to info@swedishembedded.com.
"""Which inputs the survival model relies on, and in which direction (T7).

Rules, fixed before any result (paper section 4.9):

* Model: the spline Cox model on every input (`spline-cox-net-all`), fitted
  per fold on the training subjects; repeat 0, five folds.
* Importance of an input: the fall in the held-out Breslow partial
  log-likelihood per event (nats) when the input's values are permuted among
  the fold's test subjects (mean of `--permutations` shuffles). An input
  carries independent information iff the mean fall over folds is at least
  0.001 and the fall is positive in every fold.
* Direction: the hazard ratio between the 90th and the 10th percentile of a
  numeric input (the geometric mean over test subjects of the model's ratio
  with only that input set to each value), or between each level and the most
  common level of a categorical input (levels with at least 2% of the test
  subjects). No ratio is given for a numeric input missing for more than 20%
  of the subjects: setting it for everyone would invent a condition.
* Sensitivity to early death: the same analysis without the subjects who die
  in the first two years (`--exclude-early 2`); reported as the rank
  correlation of the importances and the overlap of the ten most important.

Importance and direction are associations of the fitted model. They do not
say what an intervention would do.

    python -I drivers.py --data <lifecourse dir> --out <dir> [--exclude-early 2] [--jobs N]
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features  # noqa: E402
import models  # noqa: E402
import run  # noqa: E402

MIN_DROP = 0.001  # nats per event
MAX_MISSING = 0.20
MIN_LEVEL_SHARE = 0.02
PERCENTILES = (10, 90)


def late_rows(time, cause, years):
    """True for rows that did not die before `years`."""
    return ~((cause >= 0) & (time < years))


def _mean_log_ratio(lp_a, lp_b):
    return float(np.exp(np.mean(lp_a - lp_b)))


def contrast_numeric(predict, data, rows, col):
    """{'p90 vs p10': hazard ratio} for numeric column `col`, or {} when it is mostly missing."""
    values = data["num"][rows, col].astype(np.float64)
    seen = values[~np.isnan(values)]
    if len(seen) < (1.0 - MAX_MISSING) * len(values):
        return {}
    lo, hi = np.percentile(seen, PERCENTILES)
    num = data["num"].copy()
    out = []
    for v in (hi, lo):
        num[rows, col] = v
        out.append(predict(dict(data, num=num)))
    return {"p90 vs p10": _mean_log_ratio(*out)}


def contrast_categorical(predict, data, rows, col, levels):
    """{'level vs modal': hazard ratio} for each level with enough subjects."""
    codes = data["cat"][rows, col]
    present = {v: int(np.sum(codes == v)) for v in np.unique(codes) if v >= 0}
    modal = max(present, key=present.get)
    cat = data["cat"].copy()

    def lp_with(code):
        cat[rows, col] = code
        return predict(dict(data, cat=cat))

    base = lp_with(modal)
    return {f"{levels[v]} vs {levels[modal]}": _mean_log_ratio(lp_with(v), base)
            for v, n in sorted(present.items()) if v != modal and n >= MIN_LEVEL_SHARE * len(rows)}


def analyse(data, ctx, n_permutations, seed):
    """Importance and contrasts of every input on one fold: {'importance': {name: nats}, 'contrast': {name: {label: HR}}}."""
    model = models.SplineCoxNet("all")
    fitted = model.fit(ctx)
    test = ctx.test
    t_te, c_te = ctx.outcome(test)
    event = c_te >= 0

    def predict(d):
        return model.linear_predictor(fitted, d, test)

    base = models.partial_loglik(predict(data), t_te, event)
    rng = np.random.RandomState(seed)
    num_names = [str(x) for x in data["num_names"]]
    cat_names = [str(x) for x in data["cat_names"]]
    levels = json.loads(str(data["cat_levels"]))
    importance, contrast = {}, {}
    num, cat = data["num"].copy(), data["cat"].copy()
    for table, names, key in ((num, num_names, "num"), (cat, cat_names, "cat")):
        for col, name in enumerate(names):
            kept = table[test, col].copy()
            falls = []
            for _ in range(n_permutations):
                table[test, col] = kept[rng.permutation(len(kept))]
                falls.append(base - models.partial_loglik(predict(dict(data, **{key: table})), t_te, event))
            table[test, col] = kept
            importance[name] = float(np.mean(falls))
            contrast[name] = (contrast_numeric(predict, data, test, col) if key == "num"
                              else contrast_categorical(predict, data, test, col, levels[name]))
    return {"importance": importance, "contrast": contrast}


def run_fold(args):
    data_dir, out_dir, repeat, fold, permutations, early = args
    timelines, partition, _ = run.paths(data_dir)
    data = features.load(timelines, partition, os.path.join(out_dir, "_features.npz"))
    repeats, _ = run.load_partition(partition)
    train, test = run.split(data, repeats, repeat, fold)
    if early:
        keep = late_rows(data["time"], data["cause"], early)
        train, test = train[keep[train]], test[keep[test]]
    ctx = models.Context(data, train, test, run.fold_seed(repeat, fold), {})
    result = analyse(data, ctx, permutations, run.fold_seed(repeat, fold))
    result.update(fold=fold, n_train=len(train), n_test=len(test), events_test=int((data["cause"][test] >= 0).sum()))
    return result


def combine(folds):
    """Across folds: mean importance, standard error, folds with a positive fall, and the geometric mean ratio with its range."""
    names = list(folds[0]["importance"])
    rows = []
    for name in names:
        drops = np.array([f["importance"][name] for f in folds])
        labels = sorted({k for f in folds for k in f["contrast"][name]})
        ratios = {k: [f["contrast"][name][k] for f in folds if k in f["contrast"][name]] for k in labels}
        rows.append(dict(
            name=name, drop=float(drops.mean()), se=float(drops.std(ddof=1) / np.sqrt(len(drops))),
            positive=int((drops > 0).sum()), folds=len(drops),
            informative=bool(drops.mean() >= MIN_DROP and (drops > 0).all()),
            ratios={k: dict(hr=float(np.exp(np.mean(np.log(v)))), low=float(min(v)), high=float(max(v)))
                    for k, v in ratios.items()}))
    return sorted(rows, key=lambda r: -r["drop"])


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def compare(main, early):
    """Rank correlation and top-ten overlap between the full analysis and the one without early deaths."""
    order = [r["name"] for r in main]
    other = {r["name"]: r["drop"] for r in early}
    top = lambda rows: {r["name"] for r in sorted(rows, key=lambda r: -r["drop"])[:10]}  # noqa: E731
    return dict(spearman=spearman([r["drop"] for r in main], [other[n] for n in order]),
                top10_overlap=len(top(main) & top(early)))


def markdown(rows, header):
    lines = [header, "", "| rank | input | fall in partial log-likelihood [SE] | folds positive | informative | hazard ratio (range over folds) |",
             "|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        hr = "; ".join(f"{k}: {v['hr']:.2f} ({v['low']:.2f}-{v['high']:.2f})" for k, v in r["ratios"].items()) or "none"
        lines.append(f"| {i} | {r['name']} | {r['drop']:.4f} [{r['se']:.4f}] | {r['positive']}/{r['folds']} | "
                     f"{'yes' if r['informative'] else 'no'} | {hr} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--permutations", type=int, default=10)
    ap.add_argument("--exclude-early", type=float, default=0.0, help="drop subjects who die before this many years")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    k = len(run.load_partition(run.paths(a.data)[1])[0][a.repeat])
    jobs = [(a.data, a.out, a.repeat, f, a.permutations, a.exclude_early) for f in range(k)]
    with ProcessPoolExecutor(max_workers=min(a.jobs, run.MAX_JOBS, k)) as pool:
        folds = list(pool.map(run_fold, jobs))
    rows = combine(folds)
    tag = f"early{a.exclude_early:g}" if a.exclude_early else "all"
    header = (f"spline-cox-net-all, repeat {a.repeat}, {k} folds, {a.permutations} permutations per input"
              + (f", subjects dying before {a.exclude_early:g} y excluded" if a.exclude_early else "")
              + f"; test subjects per fold about {int(np.mean([f['n_test'] for f in folds]))}, "
                f"deaths per fold about {int(np.mean([f['events_test'] for f in folds]))}")
    with open(os.path.join(a.out, f"drivers-{tag}.json"), "w") as f:
        json.dump(dict(header=header, rows=rows), f, indent=1)
    with open(os.path.join(a.out, f"drivers-{tag}.md"), "w") as f:
        f.write(markdown(rows, header) + "\n")
    print(markdown(rows, header))


if __name__ == "__main__":
    main()
