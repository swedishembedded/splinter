# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements long-horizon risk prediction from cohort and
# survey data for its clients. If your team needs expertise in building,
# validating and proving time-to-event models on real health records, you
# can procure our services by sending an email to info@swedishembedded.com.
"""Association of mortality-equivalent age acceleration with each outcome (T3).

`lifecourse lifeexp --aa-dir DIR` writes, for the subjects of one repeat's
pooled out-of-fold predictions, the age a Gompertz table gives for the
predicted restricted mean survival time and the difference to the real age
(the acceleration). Here a Cox model per outcome estimates the hazard ratio
per 5 years of acceleration, adjusted for age and sex, with survey weights and
variance clustered on the sampling PSUs. The acceleration is a function of a
model fitted to predict death, so its association with death is partly by
construction; the informative contrasts are across causes and certificate flags.

    python -I assoc.py --data <lifecourse dir> --aa <aa-*.jsonl> [--out table.md]
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features  # noqa: E402

OUTCOMES = ("all-cause", "cardiovascular", "cancer", "other", "diabetes mention", "hypertension mention")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def frame(data_dir, aa_path, out_dir):
    """One row per subject in the acceleration file with time, cause, flags, weight, cluster."""
    timelines = os.path.join(data_dir, "timelines.jsonl")
    partition = os.path.join(data_dir, "partition.json")
    data = features.load(timelines, partition, os.path.join(out_dir, "_features.npz"))
    row = {str(s): i for i, s in enumerate(data["ids"])}
    causes = {c["subject_id"]: c for c in read_jsonl(os.path.join(data_dir, "causes.jsonl"))}
    design = {d["subject_id"]: d for d in read_jsonl(os.path.join(data_dir, "design.jsonl"))}
    rows = []
    for r in read_jsonl(aa_path):
        sid = r["subject_id"]
        i = row[sid]
        d = design[sid]
        rows.append(dict(
            subject_id=sid, age=r["age"], female=float(r["sex"] == "female"),
            accel5=r["acceleration"] / 5.0, time=float(data["time"][i]), cause=int(data["cause"][i]),
            weight=float(data["weight"][i]), cluster=d["cycle"] * 1_000_000 + d["stratum"] * 100 + d["psu"],
            diabetes=causes[sid]["diabetes_mcod"], hypertension=causes[sid]["hypertension_mcod"],
            available=causes[sid]["mcod_available"]))
    return pd.DataFrame(rows)


def event_of(df, outcome):
    """0/1 event indicator per row for an outcome; the others censor."""
    died = df["cause"] >= 0
    if outcome == "all-cause":
        return died
    code = {"cardiovascular": 0, "cancer": 1, "other": 2}.get(outcome)
    if code is not None:
        return df["cause"] == code
    flag = "diabetes" if outcome == "diabetes mention" else "hypertension"
    # A death without multiple-cause data is censored, not called flag-free.
    return died & df["available"] & (df[flag] == True)  # noqa: E712


def hazard_ratio(df, outcome):
    """(HR per 5 years of acceleration, low, high, events)."""
    from lifelines import CoxPHFitter
    frame_ = df[["time", "age", "female", "accel5", "weight", "cluster"]].copy()
    frame_["event"] = event_of(df, outcome).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = CoxPHFitter().fit(frame_, duration_col="time", event_col="event",
                                weights_col="weight", cluster_col="cluster", robust=True)
    s = fit.summary.loc["accel5"]
    return (float(s["exp(coef)"]), float(s["exp(coef) lower 95%"]), float(s["exp(coef) upper 95%"]),
            int(frame_["event"].sum()))


def table(df):
    lines = ["| outcome | events | hazard ratio per 5 years of acceleration [95% CI] |", "|---|---|---|"]
    for o in OUTCOMES:
        hr, lo, hi, n = hazard_ratio(df, o)
        lines.append(f"| {o} | {n} | {hr:.2f} [{lo:.2f}, {hi:.2f}] |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--aa", required=True)
    ap.add_argument("--features-dir", help="where the feature cache is (default: <data>/baselines)")
    ap.add_argument("--out")
    a = ap.parse_args()
    df = frame(a.data, a.aa, a.features_dir or os.path.join(a.data, "baselines"))
    text = (f"{len(df)} subjects; Cox models adjusted for age and sex, survey weights, "
            f"variance clustered on PSUs\n\n{table(df)}\n")
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
