# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for retrieval whose quality is measured against known answers, for its
# clients. If your team needs expertise in lexical, semantic and hybrid search then you can procure our services by
# sending an email to info@swedishembedded.com.
"""Oracle `retrieval/benchmark`: a retrieval benchmark whose relevant passages are known (plan 3C).

A small corpus of device notes, and queries of the kinds retrieval must handle: an exact lexical match, a match by
meaning only, several relevant passages, a query among distractors, a query nothing answers, a passage repeated in
several places, two conflicting versions of one fact, and the corpus reordered. Each query is labelled with the
passages that answer it.

The rankings Splinter's three retrievers must produce are computed here independently: BM25 by the plain
implementation of `bm25.py` (which is itself held to rank_bm25), dense by cosine over a fixed table of word vectors
(a text's vector is the sum of its words' vectors; the Rust spec embeds with the same table), and their fusion by
reciprocal rank (`1 / (60 + rank)`, ties by passage index). For each ranking and each cut-off the benchmark records
recall@k, precision@k, the reciprocal rank of the first relevant passage and nDCG@k (binary gains, a log2 discount);
it records the same for rank_bm25's own BM25Okapi ranking, the independent lexical baseline, whose idf differs.

Usage: python -I benchmark.py --out FILE
"""

import argparse
import importlib.util
import json
import math
from pathlib import Path

from rank_bm25 import BM25Okapi

NAME = "retrieval/benchmark"
REFERENCES = ["rank-bm25"]
RRF_K = 60.0
CUTOFFS = [1, 3, 5]

_spec = importlib.util.spec_from_file_location("bm25", Path(__file__).with_name("bm25.py"))
bm25 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bm25)

# Concept axes: serial, power, storage, network, display. Integer components, so every cosine is exact enough that
# the two implementations order passages alike.
VECTORS = {
    "uart": [3, 0, 0, 0, 0], "baud": [2, 0, 0, 0, 0], "serial": [3, 0, 0, 0, 0], "console": [2, 0, 0, 0, 0],
    "battery": [0, 3, 0, 0, 0], "voltage": [0, 2, 0, 0, 0], "charger": [0, 2, 0, 0, 0],
    "eeprom": [0, 0, 3, 0, 0], "flash": [0, 0, 3, 0, 0], "memory": [0, 0, 2, 0, 0], "stored": [0, 0, 1, 0, 0],
    "saved": [0, 0, 2, 0, 0],
    "ethernet": [0, 0, 0, 3, 0], "wifi": [0, 0, 0, 3, 0], "router": [0, 0, 0, 2, 0], "network": [0, 0, 0, 2, 0],
    "lcd": [0, 0, 0, 0, 3], "screen": [0, 0, 0, 0, 2],
}

PASSAGES = [
    "The UART runs at 115200 baud on the debug header.",
    "Battery voltage is read every second by the charger.",
    "Configuration is kept in EEPROM and survives a reset.",
    "The Ethernet port negotiates its speed with the router.",
    "The LCD screen shows the battery level.",
    "Flash memory holds the firmware image.",
    "The UART runs at 115200 baud on the debug header.",
    "Version 1 of the board: the UART runs at 9600 baud.",
    "Version 2 of the board: the UART runs at 115200 baud.",
    "The debug header sits next to the reset button on the board.",
    "Wifi credentials are stored in flash memory.",
    "The charger stops when the battery is full.",
]

QUERIES = [
    {"kind": "exact", "query": "115200 baud", "relevant": [0, 6, 8]},
    {"kind": "meaning", "query": "what speed does the serial console use", "relevant": [0, 6, 7, 8]},
    {"kind": "several", "query": "where are settings saved", "relevant": [2, 5, 10]},
    {"kind": "distractors", "query": "what is on the debug header of the board", "relevant": [0, 6]},
    {"kind": "no answer", "query": "bluetooth pairing code", "relevant": []},
    {"kind": "repeated", "query": "UART debug header", "relevant": [0, 6]},
    {"kind": "versions", "query": "which version of the board runs the UART at 9600 baud", "relevant": [7]},
]


def vector(text):
    total = [0.0] * 5
    for word in bm25.tokens(text):
        for axis, value in enumerate(VECTORS.get(word, [0] * 5)):
            total[axis] += value
    return total


def unit(v):
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v] if norm > 0 else v


def dense(passages, query, k):
    q = unit(vector(query))
    scored = [(i, sum(a * b for a, b in zip(unit(vector(p)), q))) for i, p in enumerate(passages)]
    scored.sort(key=lambda h: (-h[1], h[0]))
    return [i for i, _ in scored[:k]]


def fused(rankings, k):
    scores = {}
    for ranking in rankings:
        for rank, passage in enumerate(ranking):
            scores[passage] = scores.get(passage, 0.0) + 1.0 / (RRF_K + rank + 1.0)
    order = sorted(scores, key=lambda p: (-scores[p], p))
    return order[:k]


def okapi(passages, query, k):
    """rank_bm25's own ranking: the independent lexical baseline (its idf is not Splinter's)."""
    model = BM25Okapi([bm25.tokens(p) for p in passages])
    scores = model.get_scores(bm25.tokens(query))
    order = [i for i in sorted(range(len(passages)), key=lambda i: (-scores[i], i)) if scores[i] > 0]
    return order[:k]


def measures(ranking, relevant):
    """recall@k, precision@k and nDCG@k for each cut-off, and the reciprocal rank: absent where undefined."""
    relevant = set(relevant)
    out = {}
    for k in CUTOFFS:
        top = ranking[:k]
        found = sum(1 for p in top if p in relevant)
        ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
        gain = sum(1.0 / math.log2(i + 2) for i, p in enumerate(top) if p in relevant)
        out[f"recall@{k}"] = found / len(relevant) if relevant else None
        out[f"precision@{k}"] = found / k
        out[f"ndcg@{k}"] = gain / ideal if relevant else None
    first = next((i for i, p in enumerate(ranking) if p in relevant), None)
    out["mrr"] = 1.0 / (first + 1) if first is not None else (None if not relevant else 0.0)
    return out


def case(passages, query, order):
    depth = len(passages)
    lexical = [i for i, _ in bm25.ranking(passages, query["query"], depth)]
    semantic = dense(passages, query["query"], depth)
    hybrid = fused([lexical, semantic], depth)
    baseline = okapi(passages, query["query"], depth)
    relevant = [order.index(p) for p in query["relevant"]]
    return {
        **query,
        "order": order,
        "relevant": sorted(relevant),
        "rankings": {"bm25": lexical, "dense": semantic, "fused": hybrid},
        "measures": {name: measures(r, relevant) for name, r in
                     [("bm25", lexical), ("dense", semantic), ("fused", hybrid), ("rank_bm25", baseline)]},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    straight = list(range(len(PASSAGES)))
    # The same corpus reordered: every ranking must name the same passages, wherever they now sit.
    reordered = list(reversed(straight))
    cases = []
    for order in (straight, reordered):
        passages = [PASSAGES[i] for i in order]
        cases += [case(passages, q, order) for q in QUERIES]
    fixture = {"fixture": NAME, "references": REFERENCES, "passages": PASSAGES, "vectors": VECTORS,
               "cases": cases}
    with open(args.out, "w") as f:
        json.dump(fixture, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
