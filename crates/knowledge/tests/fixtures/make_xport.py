#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
"""Write the SAS transport fixture for crates/knowledge/tests/tabular.rs.

pyreadstat writes `mixed.xpt` (an independent implementation of the format)
and reads it back into `mixed.json`, the values the Rust reader must
reproduce: numbers of every magnitude and sign, fractions IBM floats cannot
hold exactly, missing values, text with trailing blanks, a long label, and
enough rows that the data spans many 80-byte records. The last row ends in
a blank text field, which a reader that strips trailing blanks before
counting rows (pandas 3.0 does) mistakes for padding and drops; pyreadstat,
the reference here, keeps it.

Run: python3 crates/knowledge/tests/fixtures/make_xport.py  (needs pandas, pyreadstat)
"""
import json
import os

import numpy as np
import pandas as pd
import pyreadstat

here = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(7)
n = 37
df = pd.DataFrame(
    {
        "SEQN": np.arange(1, n + 1, dtype=float),
        "RIDAGEYR": rng.integers(0, 85, n).astype(float),
        "LBXGH": np.round(rng.normal(5.6, 0.8, n), 1),
        "TINY": rng.normal(0, 1e-9, n),
        "HUGE": rng.normal(0, 1e12, n),
        "NEG": -rng.exponential(3.0, n),
        "NAME": [("abc"[: (i % 4)] + " ") * (i % 3) for i in range(n)],
    }
)
df.loc[[2, 5, 30], "LBXGH"] = np.nan
df.loc[[0, 36], "NEG"] = 0.0
labels = ["Respondent sequence number", "Age in years at screening", "Glycohemoglobin (%)", "", "Large value with a label that is forty", "", "Text"]
xpt = os.path.join(here, "mixed.xpt")
pyreadstat.write_xport(df, xpt, file_label="fixture", column_labels=labels, table_name="MIXED", file_format_version=5)
back, _ = pyreadstat.read_xport(xpt)
assert back.shape[0] == n, f"reference read {back.shape[0]} rows of {n}"
out = {
    "dataset": "MIXED",
    "columns": [
        {"name": c, "values": [None if (isinstance(v, float) and np.isnan(v)) else v for v in back[c].tolist()]}
        for c in back.columns
    ],
}
with open(os.path.join(here, "mixed.json"), "w") as f:
    json.dump(out, f)
    f.write("\n")
print("wrote mixed.xpt and mixed.json:", back.shape)
