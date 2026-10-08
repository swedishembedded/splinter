# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements long-horizon risk prediction from cohort and
# survey data for its clients. If your team needs expertise in building,
# validating and proving time-to-event models on real health records, you
# can procure our services by sending an email to info@swedishembedded.com.
"""Learning curves of the integrated Brier score and a power-law fit (paper section 5.3).

For each model, the score of every cross-validation fold when a share (10, 25,
50, 75 and 100 per cent) of the fold's training subjects is used, the test
folds identical. The curve is the mean over the folds of the repeats asked
for; `E(N) = e_inf + a * N**(-b)` is fitted by least squares to the means
(N the number of training subjects) and its parameters get percentile
intervals from resampling whole (repeat, fold) pairs. A fit to five points is
a description of the curve, not an estimate of what more subjects would give.

Scores are read from the per-fold JSON files the scorer writes:
`baselines/<name>/scores/` for the full share and `recipe/<name>-p<share>/`
for subsamples (deep recipes `-s1` members), under `--data`.

    python -I curves.py --data <lifecourse dir> [--repeat R ...] [--bootstrap N] [--out table.md]
"""
import argparse
import json
import os

import numpy as np

SHARES = (10, 25, 50, 75, 100)
N_FULL = 38250  # about four fifths of the 47,816 non-locked subjects
# model -> (directory of the full-share scores, pattern of the subsample directories)
MODELS = {
    "elastic-net Cox": ("baselines/cox-net-all/scores", "recipe/cox-net-all-p{}"),
    "spline Cox": ("baselines/spline-cox-net-all/scores", "recipe/spline-cox-net-all-p{}"),
    "inverse-weighted logistic": ("baselines/logit-ipcw-all/scores", "recipe/logit-ipcw-all-p{}"),
    "additive PE model": ("recipe/additive-s1", "recipe/additive-p{}-s1"),
    "deep set encoder": ("recipe/horizon-default-s1", "recipe/horizon-p{}-s1"),
}


def fold_scores(directory, repeats):
    """{(repeat, fold): IBS} for the folds of `repeats` present in `directory`."""
    out = {}
    for r in repeats:
        for k in range(5):
            path = os.path.join(directory, f"r{r}-k{k}.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                m = json.load(f)["metrics"]["ibs_0_15"]
            value = m["value"] if isinstance(m, dict) else m
            if value is not None:
                out[(r, k)] = float(value)
    return out


def fit_power_law(shares, means, n_full=N_FULL):
    """(e_inf, a, b) of E(N) = e_inf + a N^-b through the points (share, mean error)."""
    from scipy.optimize import least_squares
    n = np.array(shares, dtype=float) * n_full
    y = np.array(means, dtype=float)

    def residual(p):
        return p[0] + p[1] * n ** (-p[2]) - y

    best = None
    for b0 in (0.3, 0.6, 1.0):
        r = least_squares(residual, [y.min() * 0.9, (y.max() - y.min()) * n.min() ** b0, b0],
                          bounds=([0, 0, 0.05], [1, 1e6, 4]))
        if best is None or r.cost < best.cost:
            best = r
    return tuple(float(v) for v in best.x)


def analyse(data, repeats, bootstrap, seed=1):
    """Per model: mean score by share over the folds every share has, and the fit."""
    rng = np.random.RandomState(seed)
    rows = {}
    for name, (full, pattern) in MODELS.items():
        by_share = {}
        for share in SHARES:
            d = full if share == 100 else pattern.format(share)
            scores = fold_scores(os.path.join(data, d), repeats)
            if scores:
                by_share[share] = scores
        if len(by_share) < 2:
            continue
        common = sorted(set.intersection(*[set(v) for v in by_share.values()]))
        if not common:
            continue
        shares = sorted(by_share)
        mat = np.array([[by_share[s][f] for f in common] for s in shares])  # shares x folds
        row = dict(folds=len(common), shares=shares,
                   mean_ibs=[float(m) for m in mat.mean(axis=1)])
        if len(shares) >= 4:
            e, a, b = fit_power_law([s / 100 for s in shares], mat.mean(axis=1))
            boots = []
            for _ in range(bootstrap):
                idx = rng.randint(0, len(common), size=len(common))
                try:
                    boots.append(fit_power_law([s / 100 for s in shares], mat[:, idx].mean(axis=1)))
                except Exception:  # a resample whose fit does not converge is dropped
                    pass
            boots = np.array(boots)
            row.update(e_inf=e, a=a, b=b)
            if len(boots) >= bootstrap // 2:
                row.update(e_inf_ci=[float(np.percentile(boots[:, 0], 2.5)), float(np.percentile(boots[:, 0], 97.5))],
                           b_ci=[float(np.percentile(boots[:, 2], 2.5)), float(np.percentile(boots[:, 2], 97.5))])
        rows[name] = row
    return rows


def table(rows):
    shares = sorted({s for r in rows.values() for s in r["shares"]})
    lines = ["| share of training subjects | " + " | ".join(rows) + " |",
             "|---|" + "---|" * len(rows)]
    for s in shares:
        cells = []
        for r in rows.values():
            cells.append(f"{r['mean_ibs'][r['shares'].index(s)]:.5f}" if s in r["shares"] else "")
        lines.append(f"| {s}% | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("| model | folds | asymptote e_inf [95% CI] | exponent b [95% CI] |")
    lines.append("|---|---|---|---|")
    for name, r in rows.items():
        if "e_inf" in r:
            e = r.get("e_inf_ci")
            b = r.get("b_ci")
            lines.append(f"| {name} | {r['folds']} | {r['e_inf']:.4f}" + (f" [{e[0]:.4f}, {e[1]:.4f}]" if e else "")
                         + f" | {r['b']:.2f}" + (f" [{b[0]:.2f}, {b[1]:.2f}]" if b else "") + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--repeat", action="append", type=int, help="repeats to use (default: 0 to 4)")
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = analyse(a.data, a.repeat if a.repeat is not None else range(5), a.bootstrap)
    text = f"{json.dumps(rows, indent=1)}\n\n{table(rows)}\n"
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
