# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements long-horizon risk prediction from cohort and
# survey data for its clients. If your team needs expertise in turning
# wearable sensor streams into validated risk inputs, you can procure our
# services by sending an email to info@swedishembedded.com.
"""Daily-summary features from the NHANES hip accelerometer (2003-2006).

The recipe is fixed in the paper (section 4.6) before the files were opened.
Per minute counts (PAXINTEN) of one subject, in order:

* non-wear is a run of at least `NONWEAR_MINUTES` minutes of zero counts that
  may be interrupted by at most `SPIKE_MINUTES` minutes of counts up to
  `SPIKE_COUNTS` (Troiano et al., 2008);
* a day is valid with at least `VALID_DAY_MINUTES` minutes of wear, a subject
  with at least `VALID_DAYS` valid days;
* over valid days, in wear minutes only: mean counts per minute, minutes per
  day at or above `MVPA_COUNTS` (moderate to vigorous), below `SEDENTARY_COUNTS`
  (sedentary) and in between (light), the highest 30-minute mean count, and the
  probability that a sedentary minute is followed by an active one.

    python -I accel.py --zip <PAXRAW_C.zip> --zip <PAXRAW_D.zip> --out <data>/accel.jsonl

reads the transport files out of the archives into a scratch directory it
creates and removes. Subject ids are `nhanes-<cycle start>-<SEQN>`.
"""
import argparse
import json
import os
import re
import sys
import tempfile
import zipfile

import numpy as np

NONWEAR_MINUTES = 60
SPIKE_MINUTES = 2
SPIKE_COUNTS = 100
VALID_DAY_MINUTES = 600
VALID_DAYS = 4
MVPA_COUNTS = 2020
SEDENTARY_COUNTS = 100
DAY = 1440
FEATURES = ("valid_days", "wear_min_day", "mean_cpm", "mvpa_min_day", "sedentary_min_day",
            "light_min_day", "peak30_cpm", "sedentary_to_active")


def clean_counts(raw):
    """Counts as integers. The transport file stores a zero count as the
    smallest positive number a SAS transport value can hold (5.4e-79), which
    would never be a zero minute."""
    raw = np.asarray(raw, dtype=np.float64)
    return np.where(np.abs(raw) < 1e-6, 0.0, np.round(raw))


def zero_runs(c):
    """[start, end) of every run of zero counts."""
    z = np.concatenate([[0], (c == 0).astype(np.int8), [0]])
    d = np.diff(z)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def nonwear(c):
    """Boolean per minute: inside a non-wear period."""
    out = np.zeros(len(c), dtype=bool)
    runs = zero_runs(c)
    merged = []
    for s, e in runs:
        if merged:
            ps, pe = merged[-1]
            gap = c[pe:s]
            if len(gap) <= SPIKE_MINUTES and np.all(gap <= SPIKE_COUNTS):
                merged[-1] = (ps, e)
                continue
        merged.append((s, e))
    for s, e in merged:
        if e - s >= NONWEAR_MINUTES:
            out[s:e] = True
    return out


def summarise(c, minute_of_recording):
    """The features of one subject, or None without enough valid days.

    `c` are the counts, `minute_of_recording` the 1-based minute of each (PAXN).
    """
    order = np.argsort(minute_of_recording, kind="stable")
    c = np.asarray(c, dtype=np.float64)[order]
    day = (np.asarray(minute_of_recording)[order].astype(np.int64) - 1) // DAY
    worn = ~nonwear(c)
    valid_days = [d for d in np.unique(day) if worn[day == d].sum() >= VALID_DAY_MINUTES]
    if len(valid_days) < VALID_DAYS:
        return None
    keep = np.isin(day, valid_days) & worn
    x = c[keep]
    n_days = len(valid_days)
    active = x >= SEDENTARY_COUNTS
    sedentary = ~active
    # Transitions only between minutes that are consecutive in the recording and both worn.
    consecutive = keep[:-1] & keep[1:] & (day[:-1] == day[1:])
    from_sedentary = consecutive & (c[:-1] < SEDENTARY_COUNTS)
    to_active = from_sedentary & (c[1:] >= SEDENTARY_COUNTS)
    csum = np.cumsum(np.concatenate([[0.0], np.where(keep, c, 0.0)]))
    peak30 = float(np.max(csum[30:] - csum[:-30]) / 30.0) if len(c) >= 30 else float(x.mean())
    return dict(
        valid_days=float(n_days),
        wear_min_day=float(len(x) / n_days),
        mean_cpm=float(x.mean()),
        mvpa_min_day=float((x >= MVPA_COUNTS).sum() / n_days),
        sedentary_min_day=float(sedentary.sum() / n_days),
        light_min_day=float(((x >= SEDENTARY_COUNTS) & (x < MVPA_COUNTS)).sum() / n_days),
        peak30_cpm=peak30,
        sedentary_to_active=float(to_active.sum() / max(from_sedentary.sum(), 1)),
    )


def read_cycle(xpt_path, start):
    """{subject id: features} for one cycle's file."""
    import pandas as pd
    df = pd.read_sas(xpt_path, format="xport")
    out = {}
    seqn = df["SEQN"].to_numpy().astype(np.int64)
    counts = clean_counts(df["PAXINTEN"].to_numpy())
    minute = df["PAXN"].to_numpy()
    bounds = np.flatnonzero(np.diff(seqn)) + 1
    for lo, hi in zip(np.concatenate([[0], bounds]), np.concatenate([bounds, [len(seqn)]])):
        f = summarise(counts[lo:hi], minute[lo:hi])
        if f is not None:
            out[f"nhanes-{start}-{seqn[lo]}"] = f
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", action="append", required=True, help="PAXRAW_<cycle>.zip; the cycle is read from the name")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cycles = {"C": 2003, "D": 2005}
    rows = {}
    for z in a.zip:
        m = re.search(r"PAXRAW_([A-Z])\.zip$", z, re.I)
        if not m or m.group(1).upper() not in cycles:
            sys.exit(f"{z}: not a PAXRAW_C or PAXRAW_D archive")
        with tempfile.TemporaryDirectory() as scratch, zipfile.ZipFile(z) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".xpt") and "/" not in n]
            if len(names) != 1:
                sys.exit(f"{z}: expected one transport file, found {zf.namelist()}")
            zf.extract(names[0], scratch)
            got = read_cycle(os.path.join(scratch, names[0]), cycles[m.group(1).upper()])
        print(f"{z}: {len(got)} subjects with at least {VALID_DAYS} valid days", flush=True)
        rows.update(got)
    with open(a.out, "w") as f:
        for sid in sorted(rows):
            f.write(json.dumps(dict(subject_id=sid, **rows[sid])) + "\n")


if __name__ == "__main__":
    main()
