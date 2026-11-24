# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for retrieval whose rankings are provably what they claim, for its
# clients. If your team needs expertise in lexical and hybrid search then you can procure our services by sending an
# email to info@swedishembedded.com.
"""Oracle `retrieval/bm25`: BM25 rankings by an independent, deliberately plain implementation of the formula
splinter-knowledge's `retrieve::Bm25` states, cross-checked against rank_bm25 where the two conventions agree.

Splinter's convention (stated in its lexical.rs): words are maximal runs of alphanumeric characters, lower-cased;
k1 = 1.2, b = 0.75; the idf is Lucene's, ln(1 + (N - n + 0.5) / (n + 0.5)); a query word counts once however often
it is repeated; a passage sharing no word with the query is no hit; ties go to the earlier passage. rank_bm25's
BM25Okapi has another idf (ln((N - n + 0.5) / (n + 0.5)) with an epsilon floor), so it is checked here only for what
both conventions imply: per term the same tf saturation and length normalisation (the score divided by the idf),
for words in fewer than half the passages, where rank_bm25's idf is not floored.

Usage: python -I bm25.py --out FILE [--seed N --cases M]
"""

import argparse
import json
import math
import random
import re

from rank_bm25 import BM25Okapi

NAME = "retrieval/bm25"
K1, B = 1.2, 0.75
WORDS = ["vim", "buffer", "macro", "register", "yank", "motion", "regex", "split", "window", "fold", "mark", "jump"]


def tokens(text):
    return [w.lower() for w in re.split(r"[^0-9A-Za-z]+", text) if w]


def ranking(passages, query, k):
    docs = [tokens(p) for p in passages]
    n_docs = len(docs)
    average = sum(len(d) for d in docs) / n_docs
    holders = {}
    for doc in docs:
        for word in set(doc):
            holders[word] = holders.get(word, 0) + 1
    terms = sorted(set(tokens(query)))
    hits = []
    for index, doc in enumerate(docs):
        score = 0.0
        for term in terms:
            count = doc.count(term)
            if count == 0:
                continue
            idf = math.log((n_docs - holders[term] + 0.5) / (holders[term] + 0.5) + 1.0)
            norm = 1.0 - B + B * len(doc) / average
            score += idf * count * (K1 + 1.0) / (count + K1 * norm)
        if score > 0.0:
            hits.append((index, score))
    hits.sort(key=lambda h: (-h[1], h[0]))
    return hits[:k]


def saturation_agrees(passages, term):
    """For one query word, rank_bm25's per-passage score divided by its idf is this oracle's divided by its own."""
    docs = [tokens(p) for p in passages]
    okapi = BM25Okapi(docs, k1=K1, b=B)
    theirs = okapi.get_scores([term])
    idf = okapi.idf.get(term, 0.0)
    mine = {i: s for i, s in ranking(passages, term, len(passages))}
    n_docs, holders = len(docs), sum(term in d for d in docs)
    my_idf = math.log((n_docs - holders + 0.5) / (holders + 0.5) + 1.0)
    for i in range(len(docs)):
        a = theirs[i] / idf if idf else 0.0
        b = mine.get(i, 0.0) / my_idf
        assert abs(a - b) <= 1e-9 * max(1.0, abs(b)), (term, i, a, b)


def draw(rng):
    passages = []
    for _ in range(rng.randint(2, 12)):
        words = [rng.choice(WORDS) for _ in range(rng.randint(1, 20))]
        passages.append(" ".join(words) + rng.choice(["", ".", "!"]))
    query = " ".join(rng.choice(WORDS + ["absent"]) for _ in range(rng.randint(1, 5)))
    return passages, query, rng.randint(1, len(passages))


def case(passages, query, k):
    docs = [tokens(p) for p in passages]
    for term in set(tokens(query)):
        holders = sum(term in d for d in docs)
        # rank_bm25's idf is its plain formula only below half the passages (above, it is floored): compare there.
        if 0 < holders < len(docs) / 2:
            saturation_agrees(passages, term)
    return {"passages": passages, "query": query, "k": k,
            "hits": [{"passage": i, "score": s} for i, s in ranking(passages, query, k)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cases", type=int, default=150)
    args = parser.parse_args()
    rng = random.Random(20261124 if args.seed is None else args.seed)
    fixed = [
        (["Yank the line.", "yank YANK yank", "a split window"], "yank yank", 3),   # a repeated query word counts once
        (["macro", "macro", "register"], "macro", 3),                              # ties to the earlier passage
        (["fold", "mark"], "absent words", 2),                                     # no shared word, no hit
    ]
    cases = [case(*f) for f in fixed] + [case(*draw(rng)) for _ in range(args.cases)]
    fixture = {"fixture": NAME, "references": ["rank-bm25"], "cases": cases}
    if args.seed is not None:
        fixture |= {"differential": True, "seed": args.seed}
    with open(args.out, "w") as out:
        out.write(json.dumps(fixture, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
