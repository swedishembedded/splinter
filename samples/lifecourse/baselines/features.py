# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements long-horizon risk prediction from cohort and
# survey data for its clients. If your team needs expertise in building,
# validating and proving time-to-event models on real health records, you
# can procure our services by sending an email to info@swedishembedded.com.
"""Timelines as a feature matrix, and the preprocessing every baseline shares.

`load` streams `timelines.jsonl` once and keeps only the subjects outside the
locked test (the partition's locked list is read for that exclusion and for
nothing else). `Preprocessor` is fitted on training subjects only: it imputes,
transforms, scales, one-hot encodes and adds missingness indicators from
statistics of the rows it is given, then applies them to any other rows.
"""
import hashlib
import json
import os

import numpy as np

CODES = ("death:cvd", "death:cancer", "death:other")
# The conventional risk factors of the Rust `standard` arm (src/concepts.rs,
# STANDARD_RISK_FACTORS): measured at the examination, no history.
STANDARD_NUMERIC = ("age", "sbp", "dbp", "total_chol_mgdl", "hdl_mgdl", "hba1c_pct", "bmi")
STANDARD_CATEGORICAL = (
    "sex", "smoking", "told_diabetes", "told_heart_attack", "told_chd",
    "told_stroke", "told_heart_failure",
)
# Weights recalled at fixed offsets, in the order the builder writes them.
RECALL_NAMES = ("hist:weight_1y_ago", "hist:weight_10y_ago", "hist:weight_age25", "hist:weight_heaviest")
SLACK = 1e-9


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def locked_ids(partition_path):
    """The locked-test subject ids, to exclude them; nothing else is read from them."""
    with open(partition_path) as f:
        return set(json.load(f)["locked"])


def _outcome(subject):
    """(time since entry, cause index or -1), as the Rust `observed` reads it."""
    entry = subject["entry"]
    windows = subject["at_risk"]
    if len(windows) != 1 or windows[0]["code"] != "*":
        raise ValueError(f"{subject['subject_id']}: expected one '*' at-risk window")
    end = windows[0]["to"]
    first = None
    for e in subject["events"]:
        if e["t"] > entry and e["code"] in CODES:
            if e["t"] <= end + SLACK * max(abs(end), 1.0) and (first is None or e["t"] < first[0]):
                first = (e["t"], CODES.index(e["code"]))
    if first is None:
        return end - entry, -1
    return first[0] - entry, first[1]


def _recalled(subject):
    """The four recalled weights: 1 year ago, 10 years ago, age 25, heaviest."""
    entry = subject["entry"]
    out = [np.nan] * 4
    for o in subject["observations"]:
        if o["var"] != "weight_kg" or o["t"] >= entry:
            continue
        if abs(o["t"] - (entry - 1.0)) < 1e-9 and np.isnan(out[0]):
            out[0] = o["value"]
        elif abs(o["t"] - (entry - 10.0)) < 1e-9 and np.isnan(out[1]):
            out[1] = o["value"]
        elif o["t"] == 25.0 and np.isnan(out[2]):
            out[2] = o["value"]
        else:
            out[3] = o["value"]
    return out


def load(timelines_path, partition_path, cache_path):
    """Feature arrays for the subjects outside the locked test.

    Keyed in the cache by the digests of both files; returns a dict of numpy
    arrays: ids, weight, source, entry, time, cause, num, num_names, cat,
    cat_names, cat_levels (JSON text).
    """
    key = sha256_file(timelines_path) + sha256_file(partition_path)
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=False)
        if str(z["key"]) == key:
            return {k: z[k] for k in z.files}
    locked = locked_ids(partition_path)
    rows = []
    num_vars, cat_vars, hist_vars = set(), set(), set()
    with open(timelines_path) as f:
        for line in f:
            s = json.loads(line)
            if s["subject_id"] in locked:
                continue
            exam = {}
            for o in s["observations"]:
                if o["t"] == s["entry"]:
                    exam[o["var"]] = o["value"]
            for var, v in exam.items():
                (cat_vars if isinstance(v, str) else num_vars).add(var)
            hist = {e["code"]: s["entry"] - e["t"] for e in s["events"] if e["t"] < s["entry"]}
            hist_vars.update(hist)
            time, cause = _outcome(s)
            rows.append((s["subject_id"], s["weight"], s["source"], s["entry"], time, cause,
                         exam, hist, _recalled(s)))
    num_names = sorted(num_vars) + sorted("hist:years_since_" + h for h in hist_vars) + list(RECALL_NAMES)
    cat_names = sorted(cat_vars)
    levels = {c: sorted({r[6][c] for r in rows if c in r[6]}) for c in cat_names}
    n = len(rows)
    num = np.full((n, len(num_names)), np.nan, dtype=np.float32)
    cat = np.full((n, len(cat_names)), -1, dtype=np.int8)
    sorted_hist = sorted(hist_vars)
    n_exam = len(num_vars)
    exam_names = sorted(num_vars)
    for i, (_, _, _, _, _, _, exam, hist, rec) in enumerate(rows):
        for j, name in enumerate(exam_names):
            if name in exam:
                num[i, j] = exam[name]
        for j, h in enumerate(sorted_hist):
            if h in hist:
                num[i, n_exam + j] = hist[h]
        num[i, n_exam + len(sorted_hist):] = rec
        for j, c in enumerate(cat_names):
            if c in exam:
                cat[i, j] = levels[c].index(exam[c])
    data = dict(
        ids=np.array([r[0] for r in rows]), weight=np.array([r[1] for r in rows]),
        source=np.array([r[2] for r in rows]), entry=np.array([r[3] for r in rows]),
        time=np.array([r[4] for r in rows]), cause=np.array([r[5] for r in rows], dtype=np.int8),
        num=num, num_names=np.array(num_names), cat=cat, cat_names=np.array(cat_names),
        cat_levels=np.array(json.dumps(levels)), key=np.array(key),
    )
    tmp = cache_path + ".tmp.npz"
    np.savez(tmp, **data)
    os.replace(tmp, cache_path)
    return data


def columns(data, inputs):
    """Indices of the numeric and categorical columns of `inputs`: 'standard' or 'all'."""
    num_names = [str(x) for x in data["num_names"]]
    cat_names = [str(x) for x in data["cat_names"]]
    if inputs == "all":
        return list(range(len(num_names))), list(range(len(cat_names)))
    if inputs == "standard":
        return ([num_names.index(n) for n in STANDARD_NUMERIC],
                [cat_names.index(c) for c in STANDARD_CATEGORICAL])
    raise ValueError(f"unknown inputs {inputs!r}")


class Preprocessor:
    """Imputation, transform and scaling fitted on `rows` of the data only."""

    CLIP = 6.0

    def __init__(self, data, inputs):
        self.num_cols, self.cat_cols = columns(data, inputs)
        self.levels = json.loads(str(data["cat_levels"]))
        self.cat_names = [str(x) for x in data["cat_names"]]

    def fit(self, data, rows):
        x = data["num"][rows][:, self.num_cols].astype(np.float64)
        miss = np.isnan(x)
        self.indicator = miss.any(axis=0)
        self.median = np.where(miss.all(axis=0), 0.0, np.nanmedian(np.where(miss.all(axis=0)[None, :], 0.0, x), axis=0))
        filled = np.where(miss, self.median, x)
        # Right-skewed non-negative measurements (laboratory values, counts)
        # are log-transformed; decided from the training rows alone.
        mean = filled.mean(axis=0)
        sd = filled.std(axis=0)
        skew = np.where(sd > 0, ((filled - mean) ** 3).mean(axis=0) / np.where(sd > 0, sd, 1) ** 3, 0.0)
        self.log = (filled.min(axis=0) >= 0) & (skew > 2.0)
        t = np.where(self.log, np.log1p(np.maximum(filled, 0)), filled)
        self.mean = t.mean(axis=0)
        self.scale = np.where(t.std(axis=0) > 0, t.std(axis=0), 1.0)
        c = data["cat"][rows][:, self.cat_cols]
        self.cat_levels = []
        for j, col in enumerate(self.cat_cols):
            present = np.unique(c[:, j])
            self.cat_levels.append([int(v) for v in present])  # -1 is the missing level
        return self

    def transform(self, data, rows):
        x = data["num"][rows][:, self.num_cols].astype(np.float64)
        miss = np.isnan(x)
        filled = np.where(miss, self.median, x)
        t = np.where(self.log, np.log1p(np.maximum(filled, 0)), filled)
        z = np.clip((t - self.mean) / self.scale, -self.CLIP, self.CLIP)
        parts = [z, miss[:, self.indicator].astype(np.float64)]
        c = data["cat"][rows][:, self.cat_cols]
        for j, present in enumerate(self.cat_levels):
            for v in present:
                parts.append((c[:, j] == v).astype(np.float64)[:, None])
        return np.hstack(parts)

    def names(self, data):
        nn = [str(data["num_names"][j]) for j in self.num_cols]
        out = list(nn) + [f"missing:{n}" for n, m in zip(nn, self.indicator) if m]
        for j, present in zip(self.cat_cols, self.cat_levels):
            col = self.cat_names[self.cat_cols.index(j)]
            out += [f"{col}={self.levels[col][v] if v >= 0 else 'missing'}" for v in present]
        return out
